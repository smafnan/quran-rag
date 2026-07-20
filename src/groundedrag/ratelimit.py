"""A tiny in-process rate limiter for the endpoints that cost money.

Every semantic search spends an embedding call and every explanation spends a
chat completion, both billed to whoever's API key the deployment carries. A
public deployment without a cap hands that quota to anyone who finds the URL.

Deliberately dependency-free and in-memory: it resets on restart and is per
process, which is the right trade for a single small instance. A multi-instance
deployment wants Redis (or the host's own edge rate limiting) instead.
"""

from __future__ import annotations

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


def client_key(request) -> str:
    """Best-effort client identity behind a proxy (Render/Fly set XFF)."""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
