import uuid
import time
import logging
import ipaddress
from typing import Optional, Tuple, List, Dict, Union
from collections import defaultdict
import threading
from fastapi import Request

logger = logging.getLogger("uvicorn.error")

# Atomic Lua script for sliding window rate limiting in Redis
LUA_SLIDING_WINDOW_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local max_requests = tonumber(ARGV[3])
local event_id = ARGV[4] or (now .. ":" .. math.random(100000, 999999))
local clear_before = now - window

redis.call('ZREMRANGEBYSCORE', key, '-inf', clear_before)
local current_requests = redis.call('ZCARD', key)

if current_requests < max_requests then
    redis.call('ZADD', key, now, event_id)
    redis.call('EXPIRE', key, math.ceil(window) + 1)
    return {1, 0, current_requests + 1}
else
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry_after = 1
    if oldest and #oldest >= 2 then
        local oldest_ts = tonumber(oldest[2])
        retry_after = math.max(1, math.ceil(oldest_ts + window - now))
    end
    return {0, retry_after, current_requests}
end
"""


class InMemorySlidingWindowLimiter:
    """
    Sliding window in-memory rate limiter with thread-safe lock and periodic eviction.
    Permitted strictly in development mode or single-instance setups.
    """
    def __init__(self):
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    def is_allowed(self, client_key: str, max_requests: int, window_seconds: int) -> Tuple[bool, int]:
        now = time.time()
        cutoff = now - window_seconds

        with self._lock:
            if now - self._last_cleanup > 60:
                self._cleanup(cutoff)
                self._last_cleanup = now

            timestamps = self._history[client_key]
            valid_timestamps = [ts for ts in timestamps if ts > cutoff]
            self._history[client_key] = valid_timestamps

            if len(valid_timestamps) >= max_requests:
                oldest_ts = valid_timestamps[0]
                retry_after = max(1, int(oldest_ts + window_seconds - now))
                return False, retry_after

            self._history[client_key].append(now)
            return True, 0

    def _cleanup(self, cutoff: float):
        keys_to_remove = []
        for key, timestamps in self._history.items():
            valid = [ts for ts in timestamps if ts > cutoff]
            if not valid:
                keys_to_remove.append(key)
            else:
                self._history[key] = valid
        for key in keys_to_remove:
            del self._history[key]


# Global in-memory instance
_in_memory_limiter = InMemorySlidingWindowLimiter()
_redis_client = None
_redis_lua_sha = None


def parse_and_validate_ip(ip_str: str) -> Optional[Union[ipaddress.IPv4Address, ipaddress.IPv6Address]]:
    """Safely parses and validates an IPv4 or IPv6 address string."""
    if not ip_str or not isinstance(ip_str, str):
        return None
    cleaned = ip_str.strip().strip("[]")
    if ":" in cleaned and cleaned.count(":") == 1 and "." in cleaned:
        cleaned = cleaned.split(":")[0]
    try:
        return ipaddress.ip_address(cleaned)
    except ValueError:
        return None


def is_ip_trusted(ip_obj: Union[ipaddress.IPv4Address, ipaddress.IPv6Address], trusted_list: List[str]) -> bool:
    """Checks whether an IP object belongs to an enumerated trusted list (IP or CIDR network)."""
    if not trusted_list:
        return False
    for trusted in trusted_list:
        try:
            trusted_clean = trusted.strip().strip("[]")
            if "/" in trusted_clean:
                network = ipaddress.ip_network(trusted_clean, strict=False)
                if ip_obj in network:
                    return True
            else:
                trusted_ip = ipaddress.ip_address(trusted_clean)
                if ip_obj == trusted_ip:
                    return True
        except ValueError:
            continue
    return False


def get_client_ip(
    request: Request,
    trusted_proxies: Optional[List[str]] = None,
    trusted_cloudflare_proxies: Optional[List[str]] = None
) -> str:
    """
    Safely resolves the genuine client IP using secure right-to-left proxy chain traversal.
    
    Security rules:
    1. Start with actual socket peer IP.
    2. Only trust forwarding headers if the direct peer is in TRUSTED_PROXIES.
    3. If peer is a trusted Cloudflare proxy, inspect and validate CF-Connecting-IP.
    4. Traverse X-Forwarded-For from RIGHT to LEFT, skipping trusted proxy hops.
    5. The first untrusted IP encountered is the real client IP.
    6. Discard malformed IP tokens.
    """
    raw_peer = request.client.host if request.client else "127.0.0.1"
    peer_ip = parse_and_validate_ip(raw_peer)
    if not peer_ip:
        return "127.0.0.1"

    effective_trusted = trusted_proxies or ["127.0.0.1", "::1"]
    
    # If the direct peer is NOT a trusted proxy, never trust forwarded headers
    if not is_ip_trusted(peer_ip, effective_trusted):
        return str(peer_ip)

    # If peer is an explicitly trusted Cloudflare edge proxy, inspect CF-Connecting-IP
    cf_trusted = trusted_cloudflare_proxies or effective_trusted
    if is_ip_trusted(peer_ip, cf_trusted):
        cf_ip_header = request.headers.get("CF-Connecting-IP")
        if cf_ip_header:
            cf_ip = parse_and_validate_ip(cf_ip_header)
            if cf_ip:
                return str(cf_ip)

    # Secure right-to-left traversal of X-Forwarded-For
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        raw_hops = [hop.strip() for hop in forwarded_for.split(",") if hop.strip()]
        for hop_str in reversed(raw_hops):
            hop_ip = parse_and_validate_ip(hop_str)
            if not hop_ip:
                continue
            if not is_ip_trusted(hop_ip, effective_trusted):
                return str(hop_ip)

    return str(peer_ip)


def check_rate_limit(
    request: Request,
    max_requests: int = 15,
    window_seconds: int = 60,
    trusted_proxies: Optional[List[str]] = None,
    trusted_cloudflare_proxies: Optional[List[str]] = None,
    redis_url: Optional[str] = None,
    require_redis: bool = False,
    prefix: str = "chat",
    custom_ip: Optional[str] = None,
    request_id: Optional[str] = None
) -> Tuple[bool, int]:
    """
    Evaluates sliding-window rate limit for incoming request.
    Returns (is_allowed: bool, retry_after_seconds: int).
    In production multi-replica environments (require_redis=True), fails closed if Redis is unavailable.
    """
    client_ip = custom_ip or get_client_ip(request, trusted_proxies, trusted_cloudflare_proxies)
    client_key = f"{prefix}:{client_ip}"

    # Distributed Redis sliding window evaluation
    if redis_url and redis_url.strip():
        try:
            import redis
            r = redis.from_url(redis_url, socket_timeout=1.5)
            now = time.time()
            event_id = str(request_id).strip() if request_id else f"{now}_{uuid.uuid4().hex}"
            res = r.eval(LUA_SLIDING_WINDOW_SCRIPT, 1, client_key, str(now), str(window_seconds), str(max_requests), event_id)
            is_allowed = bool(res[0] == 1)
            retry_after = int(res[1])
            return is_allowed, retry_after
        except Exception as e:
            if require_redis:
                logger.critical(f"FATAL: Distributed Redis rate limiter failed under require_redis policy: {e}")
                raise RuntimeError(f"Distributed Redis rate limiter failed: {e}")
            logger.warning(f"Distributed Redis rate limiter failed, falling back to in-memory: {e}")

    elif require_redis:
        raise RuntimeError("Distributed Redis rate limiting is mandatory in production, but REDIS_URL is not configured.")

    # In-memory sliding window fallback (development only)
    return _in_memory_limiter.is_allowed(client_key, max_requests, window_seconds)
