"""Lightweight LRU response cache for the RAG pipeline.

Caches ChatResponse objects keyed by query text to avoid redundant
LLM invocations for identical questions within a configurable TTL.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class ResponseCache:
    """Simple in-memory LRU cache with TTL for pipeline responses.

    Args:
        maxsize: Maximum number of cached entries.
        ttl_s: Time-to-live in seconds. Entries older than this are evicted.
    """

    def __init__(self, maxsize: int = 128, ttl_s: float = 300.0) -> None:
        self._maxsize = maxsize
        self._ttl = ttl_s
        self._store: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        logger.info("ResponseCache initialised: maxsize=%d ttl=%.0fs", maxsize, ttl_s)

    def get(self, key: str) -> dict[str, Any] | None:
        """Return cached response if present and not expired, else None."""
        if key not in self._store:
            return None
        ts, value = self._store[key]
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            logger.debug("Cache entry expired for key=%s", key[:40])
            return None
        # Move to end (most recently used).
        self._store.move_to_end(key)
        logger.debug("Cache hit for key=%s", key[:40])
        return value

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Store a response under the given key."""
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (time.monotonic(), value)
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)
        logger.debug("Cache put for key=%s", key[:40])

    def clear(self) -> None:
        """Evict all cached entries."""
        self._store.clear()
        logger.info("ResponseCache cleared")

    @property
    def size(self) -> int:
        return len(self._store)
