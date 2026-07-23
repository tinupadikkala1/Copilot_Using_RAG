"""API-key authentication and a simple in-memory sliding-window rate limiter."""

from __future__ import annotations

import os
import secrets
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request, status


def require_api_key(x_api_key: str = Header(default="")) -> None:
    """Validate the ``X-API-Key`` header against the configured key.

    Raises:
        HTTPException 503: If the API key is not configured on the server.
        HTTPException 401: If the provided key does not match.
    """
    expected = os.environ.get("COPILOT_API_KEY")
    if not expected:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "API key not configured")
    if not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")


class RateLimiter:
    """In-memory sliding-window rate limiter per client IP.

    Args:
        max_requests: Maximum requests allowed within the window.
        window_s: Duration of the sliding window in seconds.
    """

    def __init__(self, max_requests: int = 60, window_s: float = 60.0) -> None:
        self._max = max_requests
        self._window = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, request: Request) -> None:
        """Check and record a request, raising 429 if over the limit."""
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        q = self._hits[client]
        # Prune expired entries.
        while q and now - q[0] > self._window:
            q.popleft()
        if len(q) >= self._max:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Rate limit exceeded")
        q.append(now)
