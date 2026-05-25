"""Unit tests for Token-Aware Congestion Control (TACC)."""

from __future__ import annotations

import pytest

from lattice.transport.congestion import TACCController


@pytest.mark.asyncio
async def test_slow_start_window_growth() -> None:
    tacc = TACCController()

    assert tacc.window_size("openai") == 1
    assert await tacc.before_request("openai") is True
    await tacc.after_response("openai", latency_ms=100.0, tokens_generated=200, status_code=200)

    assert tacc.window_size("openai") == 2


@pytest.mark.asyncio
async def test_congestion_avoidance_growth() -> None:
    tacc = TACCController()

    # Grow through slow start until crossing ssthresh.
    for _ in range(16):
        assert await tacc.before_request("openai") is True
        await tacc.after_response("openai", latency_ms=100.0, tokens_generated=100, status_code=200)

    baseline = tacc.stats("openai")["window_size_float"]
    assert tacc.stats("openai")["in_slow_start"] is False

    assert await tacc.before_request("openai") is True
    await tacc.after_response("openai", latency_ms=100.0, tokens_generated=100, status_code=200)
    updated = tacc.stats("openai")["window_size_float"]

    # In congestion avoidance increase is fractional (<< 1 in most cases).
    assert updated > baseline
    assert updated < baseline + 1.0


@pytest.mark.asyncio
async def test_timeout_behavior_resets_window() -> None:
    tacc = TACCController()

    for _ in range(3):
        assert await tacc.before_request("openai") is True
        await tacc.after_response("openai", latency_ms=100.0, tokens_generated=100, status_code=200)

    assert tacc.window_size("openai") >= 2

    # Latency spike above 3x RTT EWMA should trigger timeout-like behavior.
    assert await tacc.before_request("openai") is True
    await tacc.after_response("openai", latency_ms=1000.0, tokens_generated=10, status_code=200)

    stats = tacc.stats("openai")
    assert stats["window_size"] == 1
    assert stats["in_slow_start"] is True
    assert stats["consecutive_failures"] >= 1


@pytest.mark.asyncio
async def test_429_backpressure_with_retry_after() -> None:
    tacc = TACCController()

    for _ in range(4):
        assert await tacc.before_request("openai") is True
        await tacc.after_response("openai", latency_ms=80.0, tokens_generated=60, status_code=200)

    assert await tacc.before_request("openai") is True
    await tacc.after_response(
        "openai",
        latency_ms=50.0,
        tokens_generated=0,
        status_code=429,
        retry_after=0.1,
    )

    stats = tacc.stats("openai")
    assert stats["window_size"] >= 1
    assert stats["consecutive_failures"] >= 1
    assert await tacc.before_request("openai") is False


@pytest.mark.asyncio
async def test_window_limits_concurrency() -> None:
    tacc = TACCController()

    # Move window to 3.
    for _ in range(2):
        assert await tacc.before_request("openai") is True
        await tacc.after_response("openai", latency_ms=100.0, tokens_generated=100, status_code=200)
    assert tacc.window_size("openai") == 3

    assert await tacc.before_request("openai") is True
    assert await tacc.before_request("openai") is True
    assert await tacc.before_request("openai") is True
    assert await tacc.before_request("openai") is False


@pytest.mark.asyncio
async def test_rtt_ewma_update() -> None:
    tacc = TACCController()

    assert await tacc.before_request("openai") is True
    await tacc.after_response("openai", latency_ms=100.0, tokens_generated=50, status_code=200)
    first = tacc.stats("openai")["rtt_estimate_ms"]
    assert first == pytest.approx(100.0)

    assert await tacc.before_request("openai") is True
    await tacc.after_response("openai", latency_ms=200.0, tokens_generated=50, status_code=200)
    second = tacc.stats("openai")["rtt_estimate_ms"]

    # EWMA should move toward 200, not jump to it.
    assert second > first
    assert second < 200.0


@pytest.mark.asyncio
async def test_multiple_providers_are_independent() -> None:
    tacc = TACCController()

    assert await tacc.before_request("openai") is True
    await tacc.after_response("openai", latency_ms=100.0, tokens_generated=100, status_code=200)
    assert tacc.window_size("openai") == 2
    assert tacc.window_size("anthropic") == 1

    assert await tacc.before_request("anthropic") is True
    await tacc.after_response("anthropic", latency_ms=300.0, tokens_generated=100, status_code=503)

    openai = tacc.stats("openai")
    anthropic = tacc.stats("anthropic")
    assert openai["window_size"] == 2
    assert anthropic["window_size"] == 1
    assert anthropic["consecutive_failures"] >= 1


@pytest.mark.asyncio
async def test_token_pressure_blocks_large_second_request() -> None:
    tacc = TACCController(token_budget_per_request=1000)

    assert await tacc.before_request("openai", estimated_tokens=900) is True
    assert await tacc.before_request("openai", estimated_tokens=900) is False

    stats = tacc.stats("openai")
    assert stats["active_token_pressure"] == 900
    assert stats["token_window_limit"] == 1000

    await tacc.after_response("openai", latency_ms=100.0, tokens_generated=50, status_code=200)
    assert await tacc.before_request("openai", estimated_tokens=900) is True


@pytest.mark.asyncio
async def test_priority_boost_allows_larger_request() -> None:
    tacc = TACCController(token_budget_per_request=1000)

    assert await tacc.before_request("openai", estimated_tokens=950) is True
    assert await tacc.before_request("openai", estimated_tokens=950, priority=10) is False

    await tacc.after_response("openai", latency_ms=100.0, tokens_generated=50, status_code=200)
    assert await tacc.before_request("openai", estimated_tokens=950, priority=10) is True
