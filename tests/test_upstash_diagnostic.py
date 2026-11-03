import time
import uuid
import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts.diagnose_upstash_connectivity import run_diagnostics, get_masked_url_info, verify_fail_closed_policy


class MockUpstashRedisClient:
    """Simulates an Upstash Redis server with TLS, ZSET, Lua eval, and key deletion."""
    def __init__(self):
        self.connection_pool = MagicMock()
        self.connection_pool.connection_kwargs = {"ssl": True, "socket_timeout": 3.0}
        self.connection_pool.connection_class = "SSLConnection"
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


def test_run_diagnostics_flow_simulation():
    """Executes the full diagnostic workflow against the simulated Upstash client."""
    mock_client = MockUpstashRedisClient()
    mock_redis = MagicMock()
    mock_redis.from_url.return_value = mock_client

    with patch("scripts.diagnose_upstash_connectivity.redis", mock_redis):
        result = run_diagnostics("rediss://default:mock@mock.upstash.io:6379/0")
        assert result is True

    # Confirm all diagnostic keys were cleaned up
    assert len(mock_client._store) == 0, f"Uncleaned keys detected: {mock_client._store.keys()}"


def test_verify_fail_closed_policy_execution():
    """Confirms fail-closed behavior on outage."""
    assert verify_fail_closed_policy() is True
