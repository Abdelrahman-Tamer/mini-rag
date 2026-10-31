#!/usr/bin/env python3
"""
Pre-flight verification script for Redis packaging and dependency compatibility.
Used by CI and deployment readiness checks.
"""
import sys
import inspect
from pathlib import Path
from packaging.version import parse as parse_version


def main() -> int:
    print("[1/5] Verifying redis-py import and version bounds...")
    try:
        import redis
    except ImportError as e:
        print(f"FAILED: Could not import redis: {e}", file=sys.stderr)
        return 1

    ver_str = getattr(redis, "__version__", "")
    print(f"  Installed redis-py version: {ver_str}")
    ver = parse_version(ver_str)
    min_ver = parse_version("5.0.0")
    max_ver = parse_version("8.1.0")
    if not (min_ver <= ver <= max_ver):
        print(f"FAILED: Installed redis version {ver_str} is outside [{min_ver}, {max_ver}]", file=sys.stderr)
        return 1
    print("  Version check PASSED.")

    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    api_dir = repo_root / "api"

    print("[2/5] Verifying rate_limiter import and Lua sliding window script...")
    for directory in [src_dir, api_dir]:
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))

    try:
        from helpers.rate_limiter import check_rate_limit, LUA_SLIDING_WINDOW_SCRIPT, parse_and_validate_ip
    except ImportError as e:
        print(f"FAILED: Could not import rate_limiter helpers: {e}", file=sys.stderr)
        return 1

    if not LUA_SLIDING_WINDOW_SCRIPT:
        print("FAILED: LUA_SLIDING_WINDOW_SCRIPT is empty or None", file=sys.stderr)
        return 1

    required_commands = ["ZREMRANGEBYSCORE", "ZCARD", "ZADD", "EXPIRE", "ZRANGE"]
    for cmd in required_commands:
        if cmd not in LUA_SLIDING_WINDOW_SCRIPT:
            print(f"FAILED: Missing Lua Redis command '{cmd}' in LUA_SLIDING_WINDOW_SCRIPT", file=sys.stderr)
            return 1
    print("  Rate limiter and Lua sliding window script PASSED.")

    print("[3/5] Verifying NLPController distributed lock release script...")
    try:
        from controllers.NLPController import NLPController
        source = inspect.getsource(NLPController.asset_lock)
        has_get = "redis.call('GET', KEYS[1]) == ARGV[1]" in source or 'redis.call("GET", KEYS[1]) == ARGV[1]' in source
        has_del = "redis.call('DEL', KEYS[1])" in source or 'redis.call("DEL", KEYS[1])' in source
        if not (has_get and has_del):
            print("FAILED: Atomic compare-and-delete Lua script missing in NLPController.asset_lock", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"FAILED: Could not inspect NLPController.asset_lock: {e}", file=sys.stderr)
        return 1
    print("  Distributed lock release script PASSED.")

    print("[4/5] Verifying requirements manifests parity...")
    root_req = repo_root / "requirements.txt"
    api_req = repo_root / "api" / "requirements.txt"

    root_lines = [l.strip() for l in root_req.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    api_lines = [l.strip() for l in api_req.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]

    if root_lines != api_lines:
        print("FAILED: Root requirements.txt and api/requirements.txt do not match!", file=sys.stderr)
        print(f"  Root: {root_lines}", file=sys.stderr)
        print(f"  API:  {api_lines}", file=sys.stderr)
        return 1

    redis_entries = [l for l in root_lines if l.startswith("redis")]
    if len(redis_entries) != 1 or redis_entries[0] != "redis>=5.0.0,<=8.1.0":
        print(f"FAILED: Invalid redis entry in manifests: {redis_entries}", file=sys.stderr)
        return 1
    print("  Requirements manifests parity PASSED.")

    print("[5/5] Verifying rediss:// TLS connection parsing support...")
    try:
        client = redis.from_url("rediss://default:dummy-token@mock.upstash.io:6379/0", socket_timeout=1.5)
        kwargs = client.connection_pool.connection_kwargs
        is_ssl = kwargs.get("ssl") is True or "SSLConnection" in str(client.connection_pool.connection_class)
        if not is_ssl:
            print("FAILED: rediss:// did not configure SSL/TLS connection class", file=sys.stderr)
            return 1
        if kwargs.get("socket_timeout") != 1.5:
            print("FAILED: socket_timeout not 1.5", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"FAILED: Could not parse rediss:// connection: {e}", file=sys.stderr)
        return 1
    print("  rediss:// TLS connection parsing PASSED.")

    print("\nALL REDIS PACKAGING VERIFICATIONS PASSED SUCCESSFULLY.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
