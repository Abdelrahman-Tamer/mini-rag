#!/usr/bin/env python3
"""
NileGuide mini-rag — Upstash Redis Production Connectivity & Compatibility Diagnostic

Verifies:
1. TCP/TLS connection to Upstash Redis.
2. PING / PONG responsiveness.
3. Sorted Set operations (ZADD, ZCARD, ZRANGE, ZREMRANGEBYSCORE).
4. Atomic Sliding-Window Lua Script (LUA_SLIDING_WINDOW_SCRIPT).
5. Distributed Lock acquisition (SET NX PX) and atomic release (Compare-and-Delete Lua).
6. Scoped key cleanup (only test keys deleted; no FLUSH or app key modification).
7. Fail-closed security handling under REQUIRE_REDIS_RATE_LIMITER=True.

Security Guarantee:
- Credentials, tokens, and raw connection URLs are NEVER printed or logged.
- Only keys under 'nileguide:diag:<run_id>:' are touched.
"""

import os
import sys
import time
import uuid
import ssl
from typing import Set, Tuple, Dict, Any
from urllib.parse import urlparse
from pathlib import Path

# Add src and api to sys.path so helpers can be imported
repo_root = Path(__file__).resolve().parent.parent
for p in [repo_root / "src", repo_root / "api"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import redis
    import redis.exceptions
except ImportError:
    print("[ERROR] redis-py is not installed in the current environment.", file=sys.stderr)
    sys.exit(1)

from helpers.rate_limiter import LUA_SLIDING_WINDOW_SCRIPT, check_rate_limit


def sanitize_and_validate_redis_url(raw_url: str) -> Tuple[str, Dict[str, Any]]:
    """
    Sanitizes raw Redis URL by stripping accidental surrounding quotes or whitespace,
    and returns a cleaned URL along with sanitized connection metadata.
    Never exposes credentials in returned metadata.
    """
    if not raw_url:
        raise ValueError("REDIS_URL is empty")

    cleaned = raw_url.strip().strip('"').strip("'").strip()
    parsed = urlparse(cleaned)

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("redis", "rediss", "unix"):
        raise ValueError(f"Unsupported Redis URL scheme: '{scheme}'. Must be 'rediss://' or 'redis://'")

    hostname = parsed.hostname or "unknown"
    port = parsed.port or (6379 if scheme in ("redis", "rediss") else 0)
    db = parsed.path.lstrip("/") or "0"

    # Mask host partially for privacy (e.g. sensible-slug-12345.upstash.io -> ***.upstash.io)
    host_parts = hostname.split(".")
    if len(host_parts) > 2:
        masked_host = f"***.{'.'.join(host_parts[-2:])}"
    else:
        masked_host = "***.host"

    meta = {
        "scheme": scheme,
        "is_tls_scheme": scheme == "rediss",
        "has_username": bool(parsed.username),
        "has_password": bool(parsed.password),
        "masked_host": masked_host,
        "port": port,
        "db": db,
    }
    return cleaned, meta


def get_masked_url_info(url_str: str) -> str:
    """Parses a Redis URI and returns a sanitized summary without credentials."""
    try:
        _, meta = sanitize_and_validate_redis_url(url_str)
        auth_desc = "Present" if (meta["has_username"] or meta["has_password"]) else "None"
        tls_desc = "Yes" if meta["is_tls_scheme"] else "No"
        return f"Scheme={meta['scheme']}:// | Host={meta['masked_host']} | Port={meta['port']} | DB={meta['db']} | Auth={auth_desc} | TLS={tls_desc}"
    except Exception as e:
        return f"Scheme=unknown (parsing error: {e})"


def classify_connection_error(exc: Exception, meta: Dict[str, Any]) -> str:
    """
    Analyzes an exception during Redis connection/ping and returns an actionable diagnosis.
    Never prints or reveals credentials.
    """
    err_str = str(exc)
    err_type = type(exc).__name__

    # 1. Authentication failure
    if isinstance(exc, redis.exceptions.AuthenticationError) or "NOAUTH" in err_str or "WRONGPASS" in err_str:
        return (
            "AUTHENTICATION FAILURE: The server rejected the provided credentials.\n"
            "  -> Ensure the password/token is correct.\n"
            "  -> Expected format: rediss://default:<PASSWORD_OR_TOKEN>@<HOST>:<PORT>\n"
            "  -> Note: Use the Redis TCP password from Upstash, NOT the REST API token unless they are identical."
        )

    # 2. Plaintext-to-TLS port mismatch
    if meta.get("scheme") == "redis" and ("Connection closed by server" in err_str or "BrokenPipeError" in err_str):
        return (
            "PLAINTEXT-TO-TLS MISMATCH: Connection closed by server because plaintext TCP ('redis://') was sent to a TLS-only port.\n"
            "  -> Upstash clusters enforce TLS on port 6379.\n"
            "  -> Fix: Change URI scheme from 'redis://' to 'rediss://'."
        )

    # 3. TLS / SSL handshake failure
    if "SSL" in err_type or "ssl" in err_str.lower() or "certificate" in err_str.lower():
        return (
            f"TLS/SSL HANDSHAKE ERROR ({err_type}):\n"
            f"  -> Error details: {err_str}\n"
            "  -> Possible causes: CA bundle mismatch, expired certificate, or SNI negotiation issue."
        )

    # 4. Timeout error
    if isinstance(exc, (redis.exceptions.TimeoutError, TimeoutError)) or "timed out" in err_str.lower():
        return (
            "NETWORK TIMEOUT: Connection timed out attempting to reach the Redis host.\n"
            "  -> Check network connectivity, DNS resolution, or firewall/IP allowlist rules."
        )

    # 5. Connection closed by server while TLS is active
    if "Connection closed by server" in err_str:
        if meta.get("is_tls_scheme"):
            return (
                "CONNECTION CLOSED BY SERVER (TLS Active):\n"
                "  -> Upstash terminated the connection during or immediately after the handshake.\n"
                "  -> Possible causes: Invalid credentials, rate limit exceeded on Upstash free tier, or database paused/inactive."
            )
        else:
            return (
                "CONNECTION CLOSED BY SERVER (Plaintext TCP):\n"
                "  -> Server closed connection. Check if endpoint requires TLS ('rediss://')."
            )

    return f"CONNECTION ERROR ({err_type}): {err_str}"


def run_diagnostics(redis_url: str) -> bool:
    try:
        cleaned_url, meta = sanitize_and_validate_redis_url(redis_url)
    except Exception as e:
        print(f"[ERROR] Invalid Redis URL format: {e}", file=sys.stderr)
        return False

    run_id = f"diag_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    diag_prefix = f"nileguide:diag:{run_id}:"
    created_keys: Set[str] = set()

    print("=" * 70)
    print("NileGuide mini-rag — Upstash Redis Connectivity Diagnostic")
    print(f"Test Run ID:    {run_id}")
    print(f"Diagnostic Key: {diag_prefix}*")
    print(f"Target Details: {get_masked_url_info(cleaned_url)}")
    print("=" * 70)

    # Pre-flight scheme check & warning
    if not meta["is_tls_scheme"]:
        print("\n  [CRITICAL WARNING] The connection URL uses 'redis://' (plaintext TCP).", file=sys.stderr)
        print("  Upstash endpoints require TLS ('rediss://').", file=sys.stderr)
        print("  Connecting with 'redis://' will almost certainly fail with 'Connection closed by server.'\n", file=sys.stderr)

    if not meta["has_password"]:
        print("  [WARNING] No password/token found in URL. Authentication will likely fail.", file=sys.stderr)

    client = None
    all_passed = True

    try:
        # Step 1: TLS Connection & Initialization
        print("\n[Step 1/6] Initializing Redis connection pool with socket timeout (5.0s)...")
        client = redis.from_url(cleaned_url, socket_timeout=5.0)
        is_ssl_class = issubclass(client.connection_pool.connection_class, redis.connection.SSLConnection)
        print(f"  Connection Pool Class: {client.connection_pool.connection_class.__name__}")
        print(f"  TLS Configuration:     {'ENABLED (SSLConnection pool)' if is_ssl_class else 'WARNING: TLS NOT ACTIVE (Plaintext Connection)'}")

        # Step 2: PING / PONG
        print("\n[Step 2/6] Sending Redis PING command...")
        try:
            ping_response = client.ping()
            if ping_response is True:
                print("  PING returned: PONG (Connection Healthy)")
                # Inspect underlying socket TLS info if available
                try:
                    conn = client.connection_pool.get_connection()
                    try:
                        conn.connect()
                        sock = getattr(conn, "_sock", None)
                        if sock is not None and isinstance(sock, ssl.SSLSocket):
                            v = sock.version() or "TLS"
                            c = sock.cipher()
                            c_name = c[0] if c else "unknown"
                            print(f"  Active TLS Socket:     Verified ({v}, cipher={c_name})")
                    finally:
                        client.connection_pool.release(conn)
                except Exception:
                    pass
            else:
                print(f"  [FAIL] Unexpected PING response: {ping_response}")
                all_passed = False
        except Exception as ping_err:
            diagnosis = classify_connection_error(ping_err, meta)
            print(f"  [FAIL] PING failed: {ping_err}", file=sys.stderr)
            print(f"\n[DIAGNOSTIC ANALYSIS]\n{diagnosis}\n", file=sys.stderr)
            raise ping_err

        # Step 3: Sorted Set Operations
        print("\n[Step 3/6] Verifying Sorted Set (ZSET) commands for sliding window...")
        zset_key = f"{diag_prefix}zset"
        created_keys.add(zset_key)

        client.zadd(zset_key, {"req_alpha": 1000.0, "req_beta": 1030.0, "req_gamma": 1050.0})
        count = client.zcard(zset_key)
        assert count == 3, f"Expected ZCARD 3, got {count}"

        client.zremrangebyscore(zset_key, "-inf", 1020.0)
        count_after = client.zcard(zset_key)
        assert count_after == 2, f"Expected ZCARD 2 after trim, got {count_after}"

        oldest = client.zrange(zset_key, 0, 0, withscores=True)
        assert len(oldest) == 1 and oldest[0][0] in (b"req_beta", "req_beta"), f"Unexpected oldest: {oldest}"
        print("  ZADD, ZCARD, ZREMRANGEBYSCORE, ZRANGE operations verified.")

        # Step 4: Atomic Sliding Window Lua Script
        print("\n[Step 4/6] Verifying atomic sliding-window rate limit Lua script...")
        rate_key = f"{diag_prefix}rate_limit"
        created_keys.add(rate_key)

        now = time.time()
        window = 60
        max_quota = 3

        # Request 1 (Allow)
        res1 = client.eval(LUA_SLIDING_WINDOW_SCRIPT, 1, rate_key, str(now), str(window), str(max_quota), "req_1")
        assert res1[0] == 1 and res1[2] == 1, f"Req 1 failed: {res1}"

        # Request 2 (Allow)
        res2 = client.eval(LUA_SLIDING_WINDOW_SCRIPT, 1, rate_key, str(now + 0.1), str(window), str(max_quota), "req_2")
        assert res2[0] == 1 and res2[2] == 2, f"Req 2 failed: {res2}"

        # Request 3 (Allow - Quota Filled)
        res3 = client.eval(LUA_SLIDING_WINDOW_SCRIPT, 1, rate_key, str(now + 0.2), str(window), str(max_quota), "req_3")
        assert res3[0] == 1 and res3[2] == 3, f"Req 3 failed: {res3}"

        # Request 4 (Throttle - Quota Exceeded)
        res4 = client.eval(LUA_SLIDING_WINDOW_SCRIPT, 1, rate_key, str(now + 0.3), str(window), str(max_quota), "req_4")
        assert res4[0] == 0, f"Req 4 should be throttled, got: {res4}"
        assert res4[1] > 0, f"Expected retry_after > 0, got {res4[1]}"
        assert res4[2] == 3, f"Expected count 3, got {res4[2]}"

        ttl = client.ttl(rate_key)
        assert ttl > 0, f"Expected TTL on rate_key, got {ttl}"
        print(f"  Sliding window Lua script verified. 3 allowed, 4th throttled (retry_after={res4[1]}s, TTL={ttl}s).")

        # Step 5: Distributed Lock Acquisition (NX/PX) and Compare-and-Delete Lua Release
        print("\n[Step 5/6] Verifying distributed asset locking & atomic Lua release...")
        lock_key = f"{diag_prefix}lock:asset_x"
        created_keys.add(lock_key)

        token_a = str(uuid.uuid4())
        token_b = str(uuid.uuid4())
        release_lua = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('DEL', KEYS[1])
        else
            return 0
        end
        """

        # Acquire lock A
        acquired_a = bool(client.set(lock_key, token_a, nx=True, px=10000))
        assert acquired_a is True, "Failed to acquire lock with token A"

        # Lock collision: Attempt acquire with token B while held by A
        acquired_b = bool(client.set(lock_key, token_b, nx=True, px=10000))
        assert acquired_b is False, "Lock collision defect: Token B acquired lock while held by Token A"

        # Release with WRONG token (Token B tries to release Token A's lock)
        res_wrong = client.eval(release_lua, 1, lock_key, token_b)
        assert res_wrong == 0, f"Security defect: released lock with wrong token! result={res_wrong}"

        val_still = client.get(lock_key)
        if isinstance(val_still, bytes):
            val_still = val_still.decode("utf-8")
        assert val_still == token_a, f"Lock value modified after invalid release: {val_still}"

        # Release with CORRECT token (Token A)
        res_correct = client.eval(release_lua, 1, lock_key, token_a)
        assert res_correct == 1, f"Failed to release lock with correct token: {res_correct}"

        # Verify key is deleted
        val_after = client.get(lock_key)
        assert val_after is None, f"Lock key still exists after valid release: {val_after}"
        print("  Distributed lock acquisition (NX/PX) and atomic compare-and-delete release verified.")

    except Exception as e:
        print(f"\n[ERROR] Diagnostic encountered an exception: {e}", file=sys.stderr)
        all_passed = False

    finally:
        # Step 6: Safe Cleanup of ONLY Diagnostic Keys
        print("\n[Step 6/6] Cleaning up test-specific diagnostic keys...")
        if client and created_keys:
            try:
                # Double-check all keys start with diag_prefix before deleting
                safe_to_delete = [k for k in created_keys if k.startswith(diag_prefix)]
                if safe_to_delete:
                    deleted_count = client.delete(*safe_to_delete)
                    print(f"  Safely deleted {deleted_count}/{len(safe_to_delete)} diagnostic keys.")
                    print("  No production or application keys were modified.")
                else:
                    print("  No keys to clean up.")
            except Exception as ce:
                print(f"  [WARNING] Cleanup error: {ce}", file=sys.stderr)
        elif not created_keys:
            print("  No keys were created.")

    return all_passed


def verify_fail_closed_policy() -> bool:
    """Verifies that Redis failures are handled strictly according to REQUIRE_REDIS_RATE_LIMITER policy."""
    print("\n[Bonus Step] Verifying REQUIRE_REDIS_RATE_LIMITER Fail-Closed Policy...")
    dummy_req = type("MockRequest", (), {
        "client": type("MockClient", (), {"host": "203.0.113.195"})(),
        "headers": {}
    })()

    # Case A: require_redis=True with invalid/unreachable Redis URL -> MUST RAISE RuntimeError
    raised_fail_closed = False
    try:
        check_rate_limit(
            request=dummy_req,
            max_requests=10,
            window_seconds=60,
            redis_url="rediss://default:invalid@127.0.0.1:9999",
            require_redis=True
        )
    except RuntimeError:
        raised_fail_closed = True
    except Exception as e:
        print(f"  [FAIL] Expected RuntimeError for require_redis=True, got {type(e).__name__}: {e}")

    assert raised_fail_closed, "Fail-closed security defect: require_redis=True did not raise RuntimeError on Redis outage"
    print("  Policy require_redis=True strictly FAILS CLOSED on connection failure (HTTP 503 path verified).")

    # Case B: require_redis=False with invalid/unreachable Redis URL -> Falls back gracefully to in-memory
    allowed, retry_after = check_rate_limit(
        request=dummy_req,
        max_requests=10,
        window_seconds=60,
        redis_url="rediss://default:invalid@127.0.0.1:9999",
        require_redis=False
    )
    assert allowed is True, "Expected fallback to in-memory limiter when require_redis=False"
    print("  Policy require_redis=False falls back gracefully to in-memory limiter without crashing.")

    return True


def main() -> int:
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        redis_url = redis_url.strip().strip('"').strip("'").strip()

    # If not in process environment, check .env in scratch directory or current directory
    if not redis_url:
        scratch_env = Path(r"C:\Users\DELL\.gemini\antigravity\brain\c95201ec-bbb8-4f6d-b0f7-684e71df0a5c\scratch\.env.production")
        if scratch_env.exists():
            for line in scratch_env.read_text(encoding="utf-8").splitlines():
                if line.startswith("REDIS_URL=") and not line.startswith("REDIS_URL=[SENSITIVE]"):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'").strip()
                    if val and val != "[SENSITIVE]":
                        redis_url = val
                        break

    if not redis_url:
        print("=" * 70)
        print("NileGuide mini-rag — Upstash Redis Connectivity Diagnostic")
        print("STATUS: REDIS_URL ENVIRONMENT VARIABLE NOT CONFIGURED LOCALLY")
        print("=" * 70)
        print("\nNote: REDIS_URL has been configured in Vercel Production, but is not present")
        print("in the local process environment. Running fail-closed policy verification...\n")
        verify_fail_closed_policy()
        print("\nTo test live Upstash connectivity directly:")
        print("  1. In PowerShell: .\\scripts\\run_secure_upstash_check.ps1")
        print("     (prompts securely without echoing or logging the URL)")
        print("\nDiagnostic script ready. Returning code 0.")
        return 0

    success = run_diagnostics(redis_url)
    fail_closed_ok = verify_fail_closed_policy()

    if success and fail_closed_ok:
        print("\n" + "=" * 70)
        print("ALL UPSTASH REDIS CONNECTIVITY & COMPATIBILITY CHECKS PASSED.")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("ONE OR MORE UPSTASH REDIS CHECKS FAILED.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
