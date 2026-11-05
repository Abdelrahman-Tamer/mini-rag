import hmac
import hashlib
import time
import uuid
import json
import logging
from typing import Optional, Tuple, List, Dict, Any
from collections import defaultdict
import threading

logger = logging.getLogger("uvicorn.error")

# Atomic Lua sliding window script for Redis rate limiting and quotas
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


class InMemoryGuestLimiter:
    """
    Thread-safe in-memory sliding window limiter & session buffer for development and tests.
    Strictly forbidden in production environments where require_redis=True.
    """
    def __init__(self):
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._conversations: Dict[str, Tuple[List[Dict[str, Any]], float]] = {}
        self._lock = threading.Lock()

    def check_and_record(self, key: str, max_requests: int, window_seconds: int) -> Tuple[bool, int, int]:
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            timestamps = [ts for ts in self._history[key] if ts > cutoff]
            if len(timestamps) >= max_requests:
                oldest = timestamps[0]
                retry_after = max(1, int(oldest + window_seconds - now))
                self._history[key] = timestamps
                return False, retry_after, 0
            timestamps.append(now)
            self._history[key] = timestamps
            remaining = max(0, max_requests - len(timestamps))
            return True, 0, remaining

    def get_remaining(self, key: str, max_requests: int, window_seconds: int) -> int:
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            timestamps = [ts for ts in self._history.get(key, []) if ts > cutoff]
            return max(0, max_requests - len(timestamps))

    def get_conversation(self, session_id: str, ttl: int = 1800) -> List[Dict[str, Any]]:
        now = time.time()
        with self._lock:
            item = self._conversations.get(session_id)
            if not item:
                return []
            history, expiry = item
            if now > expiry:
                del self._conversations[session_id]
                return []
            self._conversations[session_id] = (history, now + ttl)
            return [dict(turn) for turn in history]

    def append_conversation(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ttl: int = 1800
    ) -> None:
        now = time.time()
        with self._lock:
            history: List[Dict[str, Any]] = []
            if session_id in self._conversations:
                item, expiry = self._conversations[session_id]
                if now <= expiry:
                    history = list(item)
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": assistant_msg, "sources": sources or []})
            self._conversations[session_id] = (history, now + ttl)

    def reset(self):
        with self._lock:
            self._history.clear()
            self._conversations.clear()


_in_memory_guest_limiter = InMemoryGuestLimiter()


def resolve_guest_secret(settings: Any) -> str:
    """Resolves cryptographic secret key for guest HMAC signing."""
    if getattr(settings, "GUEST_SECRET_KEY", None) and settings.GUEST_SECRET_KEY.strip():
        return settings.GUEST_SECRET_KEY.strip()
    if getattr(settings, "ADMIN_API_KEY", None) and settings.ADMIN_API_KEY.strip():
        return settings.ADMIN_API_KEY.strip()
    if getattr(settings, "SYNC_API_KEY", None) and settings.SYNC_API_KEY.strip():
        return settings.SYNC_API_KEY.strip()
    return "nileguide-guest-secret-2026-safe-dev"


def derive_guest_id(client_ip: str, secret_key: str) -> str:
    """Derives a deterministic, pseudonymous guest identity from client IP and HMAC secret."""
    raw = f"{secret_key}:nileguide-guest:{client_ip.strip()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def generate_guest_token(guest_id: str, session_id: str, secret_key: str) -> str:
    """Generates an HMAC-SHA256 authenticated guest token."""
    timestamp = int(time.time())
    payload = f"{guest_id}.{session_id}.{timestamp}"
    signature = hmac.new(
        secret_key.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    return f"{payload}.{signature}"


def verify_guest_token(
    token: str,
    secret_key: str,
    max_age_seconds: int = 86400 * 7
) -> Optional[Tuple[str, str, int]]:
    """
    Verifies HMAC-SHA256 signature and age of a guest token.
    Returns (guest_id, session_id, timestamp) if valid, otherwise None.
    """
    if not token or not isinstance(token, str):
        return None
    parts = token.strip().split(".")
    if len(parts) != 4:
        return None
    guest_id, session_id, ts_str, signature = parts
    try:
        timestamp = int(ts_str)
    except ValueError:
        return None

    now = time.time()
    # Reject timestamps more than 5 minutes in future (clock skew allowance)
    if timestamp > now + 300:
        return None
    # Reject expired tokens
    if timestamp < now - max_age_seconds:
        return None

    payload = f"{guest_id}.{session_id}.{timestamp}"
    expected_sig = hmac.new(
        secret_key.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(signature, expected_sig):
        return None

    return (guest_id, session_id, timestamp)


def get_guest_remaining_questions(
    guest_id: str,
    redis_url: Optional[str],
    require_redis: bool,
    env: str,
    daily_limit: int = 5,
    daily_window: int = 86400
) -> int:
    """Returns the number of remaining questions for the guest in the 24h window without incrementing."""
    quota_key = f"nileguide:{env}:guest_quota:{guest_id}"

    if redis_url and redis_url.strip():
        try:
            import redis
            r = redis.from_url(redis_url, socket_timeout=1.5)
            now = time.time()
            clear_before = now - daily_window
            r.zremrangebyscore(quota_key, "-inf", clear_before)
            used = r.zcard(quota_key)
            return max(0, daily_limit - int(used))
        except Exception as e:
            if require_redis:
                logger.critical(f"FATAL: Redis limiter error in get_guest_remaining_questions: {e}")
                raise RuntimeError(f"Distributed Redis rate limiter failed: {e}")
            logger.warning(f"Redis get_remaining failed, falling back to memory: {e}")

    elif require_redis:
        raise RuntimeError("Distributed Redis rate limiting is mandatory in production, but REDIS_URL is not configured.")

    return _in_memory_guest_limiter.get_remaining(quota_key, daily_limit, daily_window)


def check_guest_limits(
    guest_id: str,
    redis_url: Optional[str],
    require_redis: bool,
    env: str,
    daily_limit: int = 5,
    daily_window: int = 86400,
    burst_limit: int = 10,
    burst_window: int = 60,
    request_id: Optional[str] = None
) -> Tuple[bool, int, int, str]:
    """
    Evaluates guest burst rate limit (10 req/min) and daily quota (5 questions/24h).
    Returns: (is_allowed: bool, retry_after: int, remaining_questions: int, error_type: str)
    """
    burst_key = f"nileguide:{env}:guest_burst:{guest_id}"
    quota_key = f"nileguide:{env}:guest_quota:{guest_id}"

    if redis_url and redis_url.strip():
        try:
            import redis
            r = redis.from_url(redis_url, socket_timeout=1.5)
            now = time.time()
            req_tag = request_id or uuid.uuid4().hex

            # 1. Check Burst Limit
            burst_res = r.eval(
                LUA_SLIDING_WINDOW_SCRIPT,
                1,
                burst_key,
                str(now),
                str(burst_window),
                str(burst_limit),
                f"{now}_{req_tag}"
            )
            if burst_res[0] != 1:
                retry_after = int(burst_res[1])
                # Query daily remaining without consuming
                clear_before = now - daily_window
                r.zremrangebyscore(quota_key, "-inf", clear_before)
                current_used = int(r.zcard(quota_key))
                remaining = max(0, daily_limit - current_used)
                return False, retry_after, remaining, "burst_limit_exceeded"

            # 2. Check Daily Quota
            quota_res = r.eval(
                LUA_SLIDING_WINDOW_SCRIPT,
                1,
                quota_key,
                str(now),
                str(daily_window),
                str(daily_limit),
                f"{now}_{req_tag}"
            )
            if quota_res[0] != 1:
                retry_after = int(quota_res[1])
                return False, retry_after, 0, "daily_quota_exceeded"

            used = int(quota_res[2])
            remaining = max(0, daily_limit - used)
            return True, 0, remaining, ""

        except Exception as e:
            if require_redis:
                logger.critical(f"FATAL: Redis check_guest_limits failed under require_redis policy: {e}")
                raise RuntimeError(f"Distributed Redis rate limiter failed: {e}")
            logger.warning(f"Redis check_guest_limits failed, falling back to memory: {e}")

    elif require_redis:
        raise RuntimeError("Distributed Redis rate limiting is mandatory in production, but REDIS_URL is not configured.")

    # In-memory fallback (development & test mode only)
    burst_ok, burst_retry, _ = _in_memory_guest_limiter.check_and_record(burst_key, burst_limit, burst_window)
    if not burst_ok:
        remaining = _in_memory_guest_limiter.get_remaining(quota_key, daily_limit, daily_window)
        return False, burst_retry, remaining, "burst_limit_exceeded"

    quota_ok, quota_retry, remaining = _in_memory_guest_limiter.check_and_record(quota_key, daily_limit, daily_window)
    if not quota_ok:
        return False, quota_retry, 0, "daily_quota_exceeded"

    return True, 0, remaining, ""


def get_guest_conversation_history(
    session_id: str,
    redis_url: Optional[str],
    require_redis: bool,
    env: str,
    ttl: int = 1800
) -> List[Dict[str, Any]]:
    """
    Retrieves temporary guest conversation turns from Redis or in-memory buffer.
    Never queries or touches MongoDB.
    """
    session_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
    key = f"nileguide:{env}:guest_conv:{session_hash}"

    if redis_url and redis_url.strip():
        try:
            import redis
            r = redis.from_url(redis_url, socket_timeout=1.5)
            raw = r.get(key)
            if raw:
                r.expire(key, ttl)
                data = json.loads(raw.decode("utf-8"))
                if isinstance(data, list):
                    return data
            return []
        except Exception as e:
            if require_redis:
                logger.critical(f"FATAL: Redis get_guest_conversation_history failed: {e}")
                raise RuntimeError(f"Distributed Redis session store failed: {e}")
            logger.warning(f"Redis get_guest_conversation_history failed, falling back: {e}")

    elif require_redis:
        raise RuntimeError("Distributed Redis session store is mandatory in production, but REDIS_URL is not configured.")

    return _in_memory_guest_limiter.get_conversation(session_id, ttl)


def append_guest_conversation_turn(
    session_id: str,
    user_msg: str,
    assistant_msg: str,
    sources: Optional[List[Dict[str, Any]]],
    redis_url: Optional[str],
    require_redis: bool,
    env: str,
    ttl: int = 1800
) -> None:
    """
    Appends a new turn to the temporary guest session with sliding TTL (default 1800s).
    Never writes to MongoDB.
    """
    session_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
    key = f"nileguide:{env}:guest_conv:{session_hash}"

    if redis_url and redis_url.strip():
        try:
            import redis
            r = redis.from_url(redis_url, socket_timeout=1.5)
            raw = r.get(key)
            history = json.loads(raw.decode("utf-8")) if raw else []
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": assistant_msg, "sources": sources or []})
            r.set(key, json.dumps(history), ex=ttl)
            return
        except Exception as e:
            if require_redis:
                logger.critical(f"FATAL: Redis append_guest_conversation_turn failed: {e}")
                raise RuntimeError(f"Distributed Redis session store failed: {e}")
            logger.warning(f"Redis append_guest_conversation_turn failed, falling back: {e}")

    elif require_redis:
        raise RuntimeError("Distributed Redis session store is mandatory in production, but REDIS_URL is not configured.")

    _in_memory_guest_limiter.append_conversation(session_id, user_msg, assistant_msg, sources, ttl)
