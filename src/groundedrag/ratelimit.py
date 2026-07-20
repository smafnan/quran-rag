"""A tiny in-process rate limiter for the endpoints that cost money.

Every semantic search spends an embedding call and every explanation spends a
chat completion, both billed to whoever's API key the deployment carries. A
public deployment without a cap hands that quota to anyone who finds the URL.

Deliberately dependency-free and in-memory: it resets on restart and is per
process, which is the right trade for a single small instance. A multi-instance
deployment wants Redis (or the host's own edge rate limiting) instead.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """(allowed, retry_after_seconds). Records the hit when allowed."""
        if self.limit <= 0:                      # 0/negative disables the limit
            return True, 0
        now = time.monotonic()
        cutoff = now - self.window
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, max(1, int(hits[0] + self.window - now) + 1)
            hits.append(now)
            # opportunistic sweep so idle clients don't accumulate forever
            if len(self._hits) > 4096:
                for k in [k for k, v in self._hits.items() if not v or v[-1] < cutoff]:
                    del self._hits[k]
            return True, 0


def client_key(request, trusted_hops: int | None = None) -> str:
    """Client identity behind a reverse proxy.

    Each proxy *appends* the address it received the request from, so in
    ``X-Forwarded-For: a, b, c`` the rightmost entry was written by the last
    proxy and the leftmost is whatever the original client claimed.

    Reading the leftmost entry — the obvious-looking choice — lets a caller send
    a different fabricated value on every request and get a fresh bucket each
    time, which silently defeats the whole rate limit. So count back from the
    right by the number of proxies actually in front of this app
    (TRUSTED_PROXY_HOPS, default 1 for a single load balancer).
    """
    if trusted_hops is None:
        try:
            trusted_hops = int(os.environ.get("TRUSTED_PROXY_HOPS", "1"))
        except ValueError:
            trusted_hops = 1
    peer = request.client.host if request.client else "unknown"
    if trusted_hops <= 0:
        return peer                      # no proxy: trust the socket, not headers
    xff = request.headers.get("x-forwarded-for", "")
    if not xff:
        return peer
    parts = [p.strip() for p in xff.split(",") if p.strip()]
    if not parts:
        return peer
    # index from the right; clamp so a short header can't reach a spoofed entry
    return parts[max(0, len(parts) - trusted_hops)]
