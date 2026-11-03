import time
import uuid
import sys
import ssl
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import redis
import redis.exceptions

from scripts.diagnose_upstash_connectivity import (
    run_diagnostics,
    get_masked_url_info,
    verify_fail_closed_policy,
    sanitize_and_validate_redis_url,
    classify_connection_error,
)


class MockUpstashRedisClient:
    """Simulates an Upstash Redis server with TLS, ZSET, Lua eval, and key deletion."""
    def __init__(self):
        self.connection_pool = MagicMock()
        self.connection_pool.connection_kwargs = {"ssl": True, "socket_timeout": 3.0}
        self.connection_pool.connection_class = redis.connection.SSLConnection
        self._store = {}
        self._ttls = {}

    def ping(self):
        return True

    def zadd(self, key, mapping):
        if key not in self._store:
            self._store[key] = {}
        for member, score in mapping.items():
            self._store[key][member] = float(score)
        return len(mapping)

    def zcard(self, key):
        return len(self._store.get(key, {}))

    def zremrangebyscore(self, key, min_s, max_s):
        if key not in self._store:
            return 0
        min_val = float("-inf") if min_s == "-inf" else float(min_s)
        max_val = float("inf") if max_s == "+inf" else float(max_s)
        to_del = [m for m, s in self._store[key].items() if min_val <= s <= max_val]
        for m in to_del:
            del self._store[key][m]
        return len(to_del)

    def zrange(self, key, start, stop, withscores=False):
        if key not in self._store:
            return []
        items = sorted(self._store[key].items(), key=lambda x: x[1])
        if stop == -1:
            slice_items = items[start:]
        else:
            slice_items = items[start:stop + 1]
        if withscores:
            return [(m.encode("utf-8") if isinstance(m, str) else m, s) for m, s in slice_items]
        return [m.encode("utf-8") if isinstance(m, str) else m for m, _ in slice_items]

    def eval(self, script, numkeys, *args):
        # Sliding window script simulation
        if "ZREMRANGEBYSCORE" in script:
            key, now, window, max_reqs, event_id = args[0], float(args[1]), float(args[2]), int(args[3]), args[4]
            clear_before = now - window
            self.zremrangebyscore(key, "-inf", clear_before)
            curr = self.zcard(key)
            if curr < max_reqs:
                self.zadd(key, {event_id: now})
                self._ttls[key] = int(window) + 1
                return [1, 0, curr + 1]
            else:
                oldest = self.zrange(key, 0, 0, withscores=True)
                retry_after = 1
                if oldest:
                    retry_after = max(1, int(oldest[0][1] + window - now))
                return [0, retry_after, curr]

        # Distributed lock compare-and-delete script
        if "GET" in script and "DEL" in script:
            key, token = args[0], args[1]
            current = self._store.get(key)
            if current == token:
                del self._store[key]
                return 1
            return 0

        raise NotImplementedError("Script not mocked")

    def ttl(self, key):
        return self._ttls.get(key, -1)

    def set(self, key, val, nx=False, px=None):
        if nx and key in self._store:
            return None
        self._store[key] = val
        return True

    def get(self, key):
        return self._store.get(key)

    def delete(self, *keys):
        deleted = 0
        for k in keys:
            if k in self._store:
                del self._store[k]
                deleted += 1
            if k in self._ttls:
                del self._ttls[k]
        return deleted


def test_masked_url_info_does_not_leak_secrets():
    """Confirms URL masking conceals passwords, tokens, and precise hosts."""
    raw_url = "rediss://default:AbCdEf123456SecretToken@sensible-slug-12345.upstash.io:6379/0"
    masked = get_masked_url_info(raw_url)
    assert "AbCdEf123456SecretToken" not in masked
    assert "default" not in masked
    assert "sensible-slug-12345" not in masked
    assert "Scheme=rediss://" in masked
    assert "TLS=Yes" in masked
    assert "Port=6379" in masked


def test_sanitize_and_validate_redis_url_quoted_and_whitespace():
    """Confirms accidental surrounding quotes and whitespace are safely stripped."""
    quoted_double = '"rediss://default:secret@myhost.upstash.io:6379/0"'
    cleaned, meta = sanitize_and_validate_redis_url(quoted_double)
    assert cleaned == "rediss://default:secret@myhost.upstash.io:6379/0"
    assert meta["scheme"] == "rediss"
    assert meta["is_tls_scheme"] is True
    assert meta["port"] == 6379
    assert meta["has_password"] is True
    assert "secret" not in meta["masked_host"]

    quoted_single = "'  rediss://default:secret@myhost.upstash.io:6379  '"
    cleaned_s, meta_s = sanitize_and_validate_redis_url(quoted_single)
    assert cleaned_s == "rediss://default:secret@myhost.upstash.io:6379"
    assert meta_s["scheme"] == "rediss"
    assert meta_s["is_tls_scheme"] is True


def test_sanitize_and_validate_redis_url_plaintext_detection():
    """Confirms plaintext redis:// is detected and flagged."""
    raw_url = "redis://default:secret@myhost.upstash.io:6379"
    cleaned, meta = sanitize_and_validate_redis_url(raw_url)
    assert cleaned == raw_url
    assert meta["scheme"] == "redis"
    assert meta["is_tls_scheme"] is False


def test_sanitize_and_validate_redis_url_invalid_scheme():
    """Confirms invalid schemes raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported Redis URL scheme"):
        sanitize_and_validate_redis_url("https://myhost.upstash.io:6379")


def test_tls_connection_pool_class_detection():
    """Confirms issubclass checks accurately distinguish SSLConnection from plain Connection."""
    client_ssl = redis.from_url("rediss://default:mock@myhost.upstash.io:6379")
    assert issubclass(client_ssl.connection_pool.connection_class, redis.connection.SSLConnection)

    client_plain = redis.from_url("redis://default:mock@myhost.upstash.io:6379")
    assert not issubclass(client_plain.connection_pool.connection_class, redis.connection.SSLConnection)


def test_classify_connection_error_plaintext_to_tls_mismatch():
    """Confirms plaintext connection error against TLS port is accurately diagnosed."""
    meta = {"scheme": "redis", "is_tls_scheme": False}
    exc = redis.exceptions.ConnectionError("Connection closed by server.")
    diag = classify_connection_error(exc, meta)
    assert "PLAINTEXT-TO-TLS MISMATCH" in diag
    assert "Change URI scheme from 'redis://' to 'rediss://'" in diag


def test_classify_connection_error_authentication():
    """Confirms authentication failures are accurately diagnosed."""
    meta = {"scheme": "rediss", "is_tls_scheme": True}
    exc = redis.exceptions.AuthenticationError("WRONGPASS invalid username-password pair")
    diag = classify_connection_error(exc, meta)
    assert "AUTHENTICATION FAILURE" in diag
    assert "rediss://default:<PASSWORD_OR_TOKEN>@<HOST>:<PORT>" in diag


def test_classify_connection_error_timeout():
    """Confirms timeout errors are classified with network remediation guidance."""
    meta = {"scheme": "rediss", "is_tls_scheme": True}
    exc = redis.exceptions.TimeoutError("Timeout connecting to server")
    diag = classify_connection_error(exc, meta)
    assert "NETWORK TIMEOUT" in diag


def test_classify_connection_error_tls_handshake():
    """Confirms SSL certificate/handshake failures are classified."""
    meta = {"scheme": "rediss", "is_tls_scheme": True}
    exc = Exception("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    diag = classify_connection_error(exc, meta)
    assert "TLS/SSL HANDSHAKE ERROR" in diag


def test_classify_connection_error_tls_closed_by_server():
    """Confirms connection closed while TLS is active indicates auth, rate limit or instance status."""
    meta = {"scheme": "rediss", "is_tls_scheme": True}
    exc = redis.exceptions.ConnectionError("Connection closed by server.")
    diag = classify_connection_error(exc, meta)
    assert "CONNECTION CLOSED BY SERVER (TLS Active)" in diag


def test_run_diagnostics_flow_simulation():
    """Executes the full diagnostic workflow against the simulated Upstash client."""
    mock_client = MockUpstashRedisClient()
    mock_redis = MagicMock()
    mock_redis.from_url.return_value = mock_client
    mock_redis.connection = redis.connection

    with patch("scripts.diagnose_upstash_connectivity.redis", mock_redis):
        result = run_diagnostics("rediss://default:mock@mock.upstash.io:6379/0")
        assert result is True

    # Confirm all diagnostic keys were cleaned up
    assert len(mock_client._store) == 0, f"Uncleaned keys detected: {mock_client._store.keys()}"


def test_verify_fail_closed_policy_execution():
    """Confirms fail-closed behavior on outage."""
    assert verify_fail_closed_policy() is True
