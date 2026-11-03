import sys
import inspect
from pathlib import Path
import pytest
from packaging.version import parse as parse_version


def test_redis_import_and_version():
    """Confirms redis-py can be imported directly and its version matches bounded constraints."""
    import redis
    assert hasattr(redis, "__version__")
    ver_str = redis.__version__
    ver = parse_version(ver_str)
    assert parse_version("5.0.0") <= ver <= parse_version("8.1.0"), f"Installed redis version {ver_str} outside expected bound [5.0.0, 8.1.0]"


def test_rate_limiter_imports_and_lua_script_intact():
    """Confirms rate_limiter module imports cleanly and Lua sliding window script is intact."""
    from helpers.rate_limiter import check_rate_limit, LUA_SLIDING_WINDOW_SCRIPT, parse_and_validate_ip

    assert callable(check_rate_limit)
    assert callable(parse_and_validate_ip)

    assert LUA_SLIDING_WINDOW_SCRIPT is not None
    required_lua_commands = ["ZREMRANGEBYSCORE", "ZCARD", "ZADD", "EXPIRE", "ZRANGE"]
    for cmd in required_lua_commands:
        assert cmd in LUA_SLIDING_WINDOW_SCRIPT, f"Missing Redis command '{cmd}' in LUA_SLIDING_WINDOW_SCRIPT"


def test_distributed_lock_release_lua_script_intact():
    """Confirms NLPController's distributed asset lock uses the atomic compare-and-delete Lua script."""
    from controllers.NLPController import NLPController

    source = inspect.getsource(NLPController.asset_lock)
    assert "redis.call('GET', KEYS[1]) == ARGV[1]" in source or 'redis.call("GET", KEYS[1]) == ARGV[1]' in source
    assert "redis.call('DEL', KEYS[1])" in source or 'redis.call("DEL", KEYS[1])' in source


def test_requirements_manifests_parity():
    """Confirms root requirements.txt and api/requirements.txt both include bounded redis dependency and are identical."""
    repo_root = Path(__file__).resolve().parent.parent
    root_req_path = repo_root / "requirements.txt"
    api_req_path = repo_root / "api" / "requirements.txt"

    assert root_req_path.exists(), "Root requirements.txt missing"
    assert api_req_path.exists(), "api/requirements.txt missing"

    root_lines = [line.strip() for line in root_req_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    api_lines = [line.strip() for line in api_req_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]

    assert root_lines == api_lines, "Root requirements.txt and api/requirements.txt must be identical"

    redis_entries = [line for line in root_lines if line.startswith("redis")]
    assert len(redis_entries) == 1, f"Expected exactly one redis dependency in requirements.txt, got {redis_entries}"
    assert redis_entries[0] == "redis>=5.0.0,<=8.1.0", f"Unexpected redis requirement specification: {redis_entries[0]}"


def test_rediss_url_parsing_and_tls_configuration():
    """Confirms rediss:// (TLS Redis URI) is parsed correctly by redis-py with socket_timeout."""
    import redis

    client = redis.from_url("rediss://default:dummy-token@mock.upstash.io:6379/0", socket_timeout=1.5)
    connection_kwargs = client.connection_pool.connection_kwargs

    # rediss:// must activate SSL/TLS
    is_ssl = connection_kwargs.get("ssl") is True or "SSLConnection" in str(client.connection_pool.connection_class)
    assert is_ssl, "rediss:// scheme did not configure SSL/TLS connection class"
    assert connection_kwargs.get("socket_timeout") == 1.5, "socket_timeout not correctly configured on connection pool"
