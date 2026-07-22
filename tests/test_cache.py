"""Unit tests for the ResponseCache module."""

from __future__ import annotations

import time

from copilot.core.cache import ResponseCache


class TestResponseCache:
    """Tests for the LRU response cache."""

    def test_get_miss(self) -> None:
        """Getting a non-existent key should return None."""
        cache = ResponseCache(maxsize=10, ttl_s=60.0)
        assert cache.get("unknown") is None

    def test_put_and_get(self) -> None:
        """Putting and getting should return the same value."""
        cache = ResponseCache(maxsize=10, ttl_s=60.0)
        cache.put("hello", {"answer": "Hi!"})
        result = cache.get("hello")
        assert result is not None
        assert result["answer"] == "Hi!"

    def test_ttl_expiry(self) -> None:
        """Entries should expire after TTL."""
        cache = ResponseCache(maxsize=10, ttl_s=0.1)
        cache.put("key", {"data": 42})
        time.sleep(0.15)
        assert cache.get("key") is None

    def test_lru_eviction(self) -> None:
        """When maxsize is exceeded, the oldest entry should be evicted."""
        cache = ResponseCache(maxsize=2, ttl_s=60.0)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        assert cache.get("a") is None  # evicted
        assert cache.get("b") is not None
        assert cache.get("c") is not None

    def test_clear(self) -> None:
        """Clearing the cache should remove all entries."""
        cache = ResponseCache(maxsize=10, ttl_s=60.0)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.size == 0

    def test_size_property(self) -> None:
        """Size should reflect the number of cached entries."""
        cache = ResponseCache(maxsize=10, ttl_s=60.0)
        assert cache.size == 0
        cache.put("a", 1)
        assert cache.size == 1
        cache.put("b", 2)
        assert cache.size == 2
