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

# Standard atomic Lua sliding window script (used for burst limiting and backwards compatibility)
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

# Atomic Lua script to reserve daily quota slot with pending status (p:<req_id>)
LUA_GUEST_RESERVE_QUOTA_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local max_requests = tonumber(ARGV[3])
local req_id = ARGV[4]
local pending_ttl = tonumber(ARGV[5]) or 60
local clear_before = now - window

-- 1. Remove expired committed turns (older than daily_window)
redis.call('ZREMRANGEBYSCORE', key, '-inf', clear_before)

-- 2. Clean up stale pending reservations older than pending_ttl (e.g. 60s)
local stale_cutoff = now - pending_ttl
local stale_candidates = redis.call('ZRANGEBYSCORE', key, '-inf', stale_cutoff)
for _, member in ipairs(stale_candidates) do
    if string.sub(member, 1, 2) == 'p:' then
        redis.call('ZREM', key, member)
    end
end

-- 3. Idempotent check: if req_id is already in-flight or committed, allow without double-counting
local p_member = 'p:' .. req_id
local c_member = 'c:' .. req_id
local p_score = redis.call('ZSCORE', key, p_member)
local c_score = redis.call('ZSCORE', key, c_member)
if p_score or c_score then
    local current_requests = redis.call('ZCARD', key)
    local remaining = math.max(0, max_requests - current_requests)
    return {1, 0, remaining, p_member}
end

-- 4. Check capacity
local current_requests = redis.call('ZCARD', key)
if current_requests < max_requests then
    redis.call('ZADD', key, now, p_member)
    redis.call('EXPIRE', key, math.ceil(window) + 1)
    local remaining = math.max(0, max_requests - (current_requests + 1))
    return {1, 0, remaining, p_member}
else
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry_after = 1
    if oldest and #oldest >= 2 then
        local oldest_ts = tonumber(oldest[2])
        retry_after = math.max(1, math.ceil(oldest_ts + window - now))
    end
    return {0, retry_after, 0, ''}
end
"""

# Atomic Lua script to commit a reserved daily quota slot (p:<req_id> -> c:<req_id>)
LUA_GUEST_COMMIT_QUOTA_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local req_id = ARGV[3]
local p_member = 'p:' .. req_id
local c_member = 'c:' .. req_id

redis.call('ZREM', key, p_member)
redis.call('ZADD', key, now, c_member)
redis.call('EXPIRE', key, math.ceil(window) + 1)
local current_requests = redis.call('ZCARD', key)
return current_requests
"""

# Atomic Lua script to refund a reserved daily quota slot (removes p:<req_id>)
LUA_GUEST_REFUND_QUOTA_SCRIPT = """
local key = KEYS[1]
local req_id = ARGV[1]
local p_member = 'p:' .. req_id
local removed = redis.call('ZREM', key, p_member)
local current_requests = redis.call('ZCARD', key)
return {removed, current_requests}
"""


class InMemoryGuestLimiter:
    """
    Thread-safe in-memory sliding window limiter & session buffer for development and tests.
    Strictly forbidden in production environments where require_redis=True.
    """
    def __init__(self):
        self._quota_entries: Dict[str, Dict[str, Tuple[float, str]]] = defaultdict(dict)
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

    def get_remaining(self, key: str, max_requests: int, window_seconds: int, pending_ttl: int = 60) -> int:
        now = time.time()
        cutoff = now - window_seconds
        stale_pending = now - pending_ttl
        with self._lock:
            entries = self._quota_entries.get(key, {})
            cleaned = {}
            for r_id, (ts, status) in entries.items():
                if ts <= cutoff:
                    continue
                if status == "p" and ts <= stale_pending:
                    continue
                cleaned[r_id] = (ts, status)
            self._quota_entries[key] = cleaned

            if cleaned:
                return max(0, max_requests - len(cleaned))

            timestamps = [ts for ts in self._history.get(key, []) if ts > cutoff]
            return max(0, max_requests - len(timestamps))

    def reserve_quota(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
        req_id: str,
        pending_ttl: int = 60
    ) -> Tuple[bool, int, int]:
        now = time.time()
        cutoff = now - window_seconds
        stale_pending = now - pending_ttl
        with self._lock:
            entries = self._quota_entries[key]
            cleaned = {}
            for r_id, (ts, status) in entries.items():
                if ts <= cutoff:
                    continue
                if status == "p" and ts <= stale_pending:
                    continue
                cleaned[r_id] = (ts, status)
            self._quota_entries[key] = cleaned

            if req_id in cleaned:
                return True, 0, max(0, max_requests - len(cleaned))

            if len(cleaned) >= max_requests:
                oldest_ts = min((ts for ts, _ in cleaned.values()), default=now)
                retry_after = max(1, int(oldest_ts + window_seconds - now))
                return False, retry_after, 0

            cleaned[req_id] = (now, "p")
            self._quota_entries[key] = cleaned
            remaining = max(0, max_requests - len(cleaned))
            return True, 0, remaining

    def commit_quota(self, key: str, req_id: str) -> None:
        now = time.time()
        with self._lock:
            if key in self._quota_entries and req_id in self._quota_entries[key]:
                self._quota_entries[key][req_id] = (now, "c")

    def refund_quota(self, key: str, req_id: str) -> None:
        with self._lock:
            if key in self._quota_entries and req_id in self._quota_entries[key]:
                del self._quota_entries[key][req_id]

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
            self._quota_entries.clear()
            self._history.clear()
            self._conversations.clear()


_in_memory_guest_limiter = InMemoryGuestLimiter()


def resolve_guest_secret(settings: Any) -> str:
    """Resolves cryptographic secret key for guest HMAC signing with production fail-closed security."""
    if getattr(settings, "GUEST_SECRET_KEY", None) and settings.GUEST_SECRET_KEY.strip():
        return settings.GUEST_SECRET_KEY.strip()
    if getattr(settings, "ADMIN_API_KEY", None) and settings.ADMIN_API_KEY.strip():
        return settings.ADMIN_API_KEY.strip()
    if getattr(settings, "SYNC_API_KEY", None) and settings.SYNC_API_KEY.strip():
        return settings.SYNC_API_KEY.strip()

    env = (getattr(settings, "ENVIRONMENT", None) or "development").strip().lower()
    if env in ("production", "staging", "prod"):
        raise RuntimeError("FATAL: Guest HMAC secret key is mandatory in production/staging environments.")

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
    daily_window: int = 86400,
    pending_ttl: int = 60
) -> int:
    """Returns the number of remaining questions for the guest in the 24h window without incrementing."""
    clean_env = (env or "development").strip().lower()
    quota_key = f"nileguide:{clean_env}:guest_quota:{guest_id}"

    if redis_url and redis_url.strip():
        try:
            import redis
            r = redis.from_url(redis_url, socket_timeout=1.5)
            now = time.time()
            clear_before = now - daily_window
            r.zremrangebyscore(quota_key, "-inf", clear_before)

            # Clean up stale pending reservations older than pending_ttl
            stale_cutoff = now - pending_ttl
            stale_candidates = r.zrangebyscore(quota_key, "-inf", stale_cutoff)
            if stale_candidates:
                for member in stale_candidates:
                    m_str = member.decode("utf-8") if isinstance(member, bytes) else str(member)
                    if m_str.startswith("p:"):
                        r.zrem(quota_key, member)

            used = r.zcard(quota_key)
            return max(0, daily_limit - int(used))
        except Exception as e:
            if require_redis:
                logger.critical(f"FATAL: Redis limiter error in get_guest_remaining_questions: {e}")
                raise RuntimeError(f"Distributed Redis rate limiter failed: {e}")
            logger.warning(f"Redis get_remaining failed, falling back to memory: {e}")

    elif require_redis:
        raise RuntimeError("Distributed Redis rate limiting is mandatory in production, but REDIS_URL is not configured.")

    return _in_memory_guest_limiter.get_remaining(quota_key, daily_limit, daily_window, pending_ttl)


def check_guest_limits(
    guest_id: str,
    redis_url: Optional[str],
    require_redis: bool,
    env: str,
    daily_limit: int = 5,
    daily_window: int = 86400,
    burst_limit: int = 10,
    burst_window: int = 60,
    request_id: Optional[str] = None,
    pending_ttl: int = 60
) -> Tuple[bool, int, int, str]:
    """
    Evaluates guest burst rate limit (10 req/min) and atomically reserves a daily quota slot (p:<req_id>).
    Returns: (is_allowed: bool, retry_after: int, remaining_questions: int, error_type: str)
    """
    clean_env = (env or "development").strip().lower()
    burst_key = f"nileguide:{clean_env}:guest_burst:{guest_id}"
    quota_key = f"nileguide:{clean_env}:guest_quota:{guest_id}"
    req_tag = request_id or uuid.uuid4().hex[:12]

    if redis_url and redis_url.strip():
        try:
            import redis
            r = redis.from_url(redis_url, socket_timeout=1.5)
            now = time.time()

            # 1. Check Burst Limit (never consumes daily quota if burst limit is hit)
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
                clear_before = now - daily_window
                r.zremrangebyscore(quota_key, "-inf", clear_before)
                current_used = int(r.zcard(quota_key))
                remaining = max(0, daily_limit - current_used)
                return False, retry_after, remaining, "burst_limit_exceeded"

            # 2. Reserve Daily Quota Slot with Pending State
            quota_res = r.eval(
                LUA_GUEST_RESERVE_QUOTA_SCRIPT,
                1,
                quota_key,
                str(now),
                str(daily_window),
                str(daily_limit),
                req_tag,
                str(pending_ttl)
            )
            if quota_res[0] != 1:
                retry_after = int(quota_res[1])
                return False, retry_after, 0, "daily_quota_exceeded"

            remaining = int(quota_res[2])
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
        remaining = _in_memory_guest_limiter.get_remaining(quota_key, daily_limit, daily_window, pending_ttl)
        return False, burst_retry, remaining, "burst_limit_exceeded"

    quota_ok, quota_retry, remaining = _in_memory_guest_limiter.reserve_quota(quota_key, daily_limit, daily_window, req_tag, pending_ttl)
    if not quota_ok:
        return False, quota_retry, 0, "daily_quota_exceeded"

    return True, 0, remaining, ""


def commit_guest_quota(
    guest_id: str,
    req_id: str,
    redis_url: Optional[str],
    require_redis: bool,
    env: str,
    daily_window: int = 86400
) -> None:
    """
    Commits a reserved guest quota slot after successful answer generation.
    Transitions member from p:<req_id> to c:<req_id>.
    """
    if not req_id:
        return
    clean_env = (env or "development").strip().lower()
    quota_key = f"nileguide:{clean_env}:guest_quota:{guest_id}"

    if redis_url and redis_url.strip():
        try:
            import redis
            r = redis.from_url(redis_url, socket_timeout=1.5)
            now = time.time()
            r.eval(LUA_GUEST_COMMIT_QUOTA_SCRIPT, 1, quota_key, str(now), str(daily_window), req_id)
            return
        except Exception as e:
            if require_redis:
                logger.critical(f"FATAL: Redis commit_guest_quota failed under require_redis policy: {e}")
                raise RuntimeError(f"Distributed Redis rate limiter failed: {e}")
            logger.warning(f"Redis commit_guest_quota failed: {e}")
    elif require_redis:
        raise RuntimeError("Distributed Redis rate limiting is mandatory in production, but REDIS_URL is not configured.")

    _in_memory_guest_limiter.commit_quota(quota_key, req_id)


def refund_guest_quota(
    guest_id: str,
    req_id: str,
    redis_url: Optional[str],
    require_redis: bool,
    env: str
) -> None:
    """
    Refunds a reserved guest quota slot when generation fails or an error occurs.
    Removes p:<req_id> from the Redis ZSET or in-memory limiter.
    """
    if not req_id:
        return
    clean_env = (env or "development").strip().lower()
    quota_key = f"nileguide:{clean_env}:guest_quota:{guest_id}"

    if redis_url and redis_url.strip():
        try:
            import redis
            r = redis.from_url(redis_url, socket_timeout=1.5)
            r.eval(LUA_GUEST_REFUND_QUOTA_SCRIPT, 1, quota_key, req_id)
            return
        except Exception as e:
            if require_redis:
                logger.critical(f"FATAL: Redis refund_guest_quota failed under require_redis policy: {e}")
                raise RuntimeError(f"Distributed Redis rate limiter failed: {e}")
            logger.warning(f"Redis refund_guest_quota failed: {e}")
    elif require_redis:
        raise RuntimeError("Distributed Redis rate limiting is mandatory in production, but REDIS_URL is not configured.")

    _in_memory_guest_limiter.refund_quota(quota_key, req_id)


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
    clean_env = (env or "development").strip().lower()
    session_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
    key = f"nileguide:{clean_env}:guest_conv:{session_hash}"

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
    clean_env = (env or "development").strip().lower()
    session_hash = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:32]
    key = f"nileguide:{clean_env}:guest_conv:{session_hash}"

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
