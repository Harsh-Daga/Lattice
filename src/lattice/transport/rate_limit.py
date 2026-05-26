"""Per-provider rate-limit header tracking with TTL eviction."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx

RATE_LIMIT_TTL_S = 3600.0  # 1 hour


@dataclass
class RateLimitState:
    limit: int | None = None
    remaining: int | None = None
    reset: int | None = None
    retry_after: int | None = None
    last_seen_at: float = field(default_factory=time.time)


class RateLimitTracker:
    """Light-weight per-provider rate-limit state.

    Parses common ``x-ratelimit-*`` headers and tracks whether a provider
    is currently throttled. Entries older than ``RATE_LIMIT_TTL_S`` are
    evicted during periodic cleanup.
    """

    def __init__(self) -> None:
        self._limits: dict[str, RateLimitState] = {}
        self._last_cleanup_at: float = time.time()

    def update(self, provider: str, headers: httpx.Headers) -> None:
        """Parse rate-limit headers from a response."""
        limit = headers.get("x-ratelimit-limit")
        remaining = headers.get("x-ratelimit-remaining")
        reset = headers.get("x-ratelimit-reset")
        retry_after = headers.get("retry-after")
        if limit or remaining or reset or retry_after:
            self._limits[provider] = RateLimitState(
                limit=int(limit) if limit else None,
                remaining=int(remaining) if remaining else None,
                reset=int(reset) if reset else None,
                retry_after=int(retry_after) if retry_after else None,
                last_seen_at=time.time(),
            )
        elif provider in self._limits:
            self._limits[provider].last_seen_at = time.time()
        self._maybe_cleanup()

    def record(self, provider: str, headers: dict[str, str]) -> None:
        """Parse rate-limit headers from a plain header mapping."""
        self.update(provider, httpx.Headers(headers))

    def is_throttled(self, provider: str) -> bool:
        state = self._limits.get(provider)
        if not state:
            return False
        remaining = state.remaining
        return bool(remaining is not None and remaining <= 0)

    def retry_after(self, provider: str) -> float | None:
        state = self._limits.get(provider)
        if not state:
            return None
        ra = state.retry_after
        if ra is not None:
            return float(ra)
        return None

    def get(self, provider: str) -> RateLimitState | None:
        return self._limits.get(provider)

    def _maybe_cleanup(self) -> None:
        now = time.time()
        if now - self._last_cleanup_at < 300.0:
            return
        cutoff = now - RATE_LIMIT_TTL_S
        self._limits = {k: v for k, v in self._limits.items() if v.last_seen_at >= cutoff}
        self._last_cleanup_at = now
