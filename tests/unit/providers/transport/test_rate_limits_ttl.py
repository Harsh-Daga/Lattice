"""RateLimitTracker TTL eviction."""

from __future__ import annotations

import time

import httpx

from lattice.transport.rate_limit import RATE_LIMIT_TTL_S, RateLimitTracker


def test_rate_limit_evicts_stale(monkeypatch) -> None:
    tracker = RateLimitTracker()
    tracker.update(
        "openai",
        httpx.Headers({"x-ratelimit-remaining": "100", "x-ratelimit-limit": "100"}),
    )
    assert tracker.get("openai") is not None

    fake_now = time.time() + RATE_LIMIT_TTL_S + 1

    def fake_time() -> float:
        return fake_now

    monkeypatch.setattr(time, "time", fake_time)
    tracker._last_cleanup_at = 0.0
    tracker.update("anthropic", httpx.Headers({"x-ratelimit-remaining": "50"}))

    assert tracker.get("openai") is None
    assert tracker.get("anthropic") is not None
