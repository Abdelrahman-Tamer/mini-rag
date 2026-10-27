import time
import threading
import pytest
from helpers.cache import ResponseCache


def test_cache_key_generation():
    key1 = ResponseCache.generate_key("Best places in Cairo", "en", "Cairo", "Historical")
    key2 = ResponseCache.generate_key("  best places in cairo  ", "en", "cairo", "historical")
    assert key1 == key2, "Cache key should be normalized and case-insensitive"

    key_ar = ResponseCache.generate_key("أفضل الأنشطة في القاهرة", "ar", "Cairo", None)
    assert key_ar != key1, "Different languages and queries should produce different keys"


def test_cache_set_and_get():
    cache = ResponseCache(max_size=10, ttl_seconds=60)
    key = ResponseCache.generate_key("Cairo activities", "en")
    data = {"answer": "Visit Pyramids and Egyptian Museum", "sources": []}

    assert cache.get(key) is None
    cache.set(key, data)

    cached = cache.get(key)
    assert cached is not None
    assert cached["answer"] == data["answer"]


def test_cache_ttl_expiration():
    cache = ResponseCache(max_size=10, ttl_seconds=1)
    key = ResponseCache.generate_key("Alexandria day trip", "en")
    data = {"answer": "Visit Citadel of Qaitbay"}

    cache.set(key, data, ttl=1)
    assert cache.get(key) is not None

    time.sleep(1.1)
    assert cache.get(key) is None, "Expired cache entry should return None"


def test_cache_lru_eviction():
    cache = ResponseCache(max_size=3, ttl_seconds=60)

    key1 = ResponseCache.generate_key("q1", "en")
    key2 = ResponseCache.generate_key("q2", "en")
    key3 = ResponseCache.generate_key("q3", "en")
    key4 = ResponseCache.generate_key("q4", "en")

    cache.set(key1, {"val": 1})
    cache.set(key2, {"val": 2})
    cache.set(key3, {"val": 3})

    # Access key1 to make key2 the least recently accessed
    _ = cache.get(key1)

    # Insert 4th element -> key2 should be evicted
    cache.set(key4, {"val": 4})

    assert cache.get(key1) is not None, "key1 was recently accessed and should remain"
    assert cache.get(key2) is None, "key2 should have been evicted by LRU"
    assert cache.get(key3) is not None, "key3 should remain"
    assert cache.get(key4) is not None, "key4 should be present"


def test_cache_stats_and_clear():
    cache = ResponseCache(max_size=10, ttl_seconds=60)
    key = ResponseCache.generate_key("Luxor tours", "en")

    _ = cache.get(key)  # Miss
    cache.set(key, {"answer": "Karnak temple"})
    _ = cache.get(key)  # Hit

    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1
    assert stats["hit_ratio_percent"] == 50.0

    cache.clear()
    assert cache.get(key) is None
    assert cache.get_stats()["size"] == 0


def test_cache_thread_safety():
    cache = ResponseCache(max_size=100, ttl_seconds=60)
    errors = []

    def worker(worker_id: int):
        try:
            for i in range(50):
                k = ResponseCache.generate_key(f"query_{i % 10}", "en")
                cache.set(k, {"worker": worker_id, "i": i})
                _ = cache.get(k)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Thread safety test encountered errors: {errors}"
    assert cache.get_stats()["size"] <= 100
