import time
import hashlib
import threading
from typing import Optional, Any, Dict

class ResponseCache:
    """
    Thread-safe, bounded In-Memory LRU/TTL Response Cache.
    Designed for fast query caching without cross-user leakage.
    """
    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def generate_key(query: str, language: str = "en", city: Optional[str] = None, category: Optional[str] = None) -> str:
        norm_query = query.strip().lower()
        key_raw = f"{language}|{norm_query}|{str(city).lower()}|{str(category).lower()}"
        return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                self._misses += 1
                return None

            # Check expiration
            if time.time() > entry["expires_at"]:
                del self._cache[key]
                self._misses += 1
                return None

            # Update access time (LRU)
            entry["last_accessed"] = time.time()
            self._hits += 1
            return entry["data"]

    def set(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        duration = ttl if ttl is not None else self.ttl_seconds
        expires_at = time.time() + duration
        with self._lock:
            # Evict oldest entry if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["last_accessed"])
                del self._cache[oldest_key]

            self._cache[key] = {
                "data": data,
                "expires_at": expires_at,
                "last_accessed": time.time(),
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_ratio = round((self._hits / total) * 100, 2) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio_percent": hit_ratio,
            }

# Global singleton cache instance
global_response_cache = ResponseCache(max_size=500, ttl_seconds=3600)
