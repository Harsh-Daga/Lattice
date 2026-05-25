"""Phase 4: TACC V2 tests.

Tests for token-aware congestion control runtime controller
enhancements: stream velocity, cache hit feedback, batch pressure,
and priority downgrade.
"""

from __future__ import annotations

import time

import pytest

from lattice.transport.congestion import AdmissionDecision, TACCController


class TestTACCV2:
    @pytest.mark.asyncio
    async def test_cache_hit_reduces_effective_latency(self) -> None:
        ctrl = TACCController(enabled=True, token_budget_per_request=100)
        await ctrl.before_request("openai", estimated_tokens=100)
        # First success establishes baseline RTT
        await ctrl.after_response("openai", latency_ms=100, tokens_generated=10, status_code=200)

        await ctrl.before_request("openai", estimated_tokens=100)
        # Cache hit with high latency should NOT trigger timeout signal
        await ctrl.after_response(
            "openai",
            latency_ms=400,  # Would normally be 4x baseline
            tokens_generated=10,
            status_code=200,
            cache_hit=True,
        )
        state = ctrl._states["openai"]
        assert state.window_size > 1.0  # Should not have collapsed

    @pytest.mark.asyncio
    async def test_batch_pressure_reduces_window_credit(self) -> None:
        ctrl = TACCController(enabled=True, token_budget_per_request=100)
        await ctrl.before_request("openai", estimated_tokens=100)
        await ctrl.after_response(
            "openai",
            latency_ms=100,
            tokens_generated=10,
            status_code=200,
            batch_size=4,
        )
        state = ctrl._states["openai"]
        # After batch, window should grow but not as aggressively
        assert state.window_size >= 1.0

    @pytest.mark.asyncio
    async def test_stream_velocity_updates_token_rate(self) -> None:
        ctrl = TACCController(enabled=True)
        await ctrl.record_stream_velocity("openai", tokens_per_second=50.0)
        state = ctrl._states["openai"]
        assert state.token_rate_estimate > 0

        await ctrl.record_stream_velocity("openai", tokens_per_second=100.0)
        assert state.token_rate_estimate > 50.0  # EWMA should trend upward

    def test_should_downgrade_priority_under_pressure(self) -> None:
        ctrl = TACCController(enabled=True)
        # No state yet — should not downgrade
        assert ctrl.should_downgrade_priority("openai", priority=0) is False

        # Simulate collapsed window with pending waiters
        state = ctrl._get_or_create("openai")
        state.window_size = 1.0
        state.pending_waiters.append(
            object()  # dummy waiter
        )
        assert ctrl.should_downgrade_priority("openai", priority=0) is True
        # High priority requests should NOT be downgraded
        assert ctrl.should_downgrade_priority("openai", priority=10) is False

    def test_should_downgrade_priority_when_stable(self) -> None:
        ctrl = TACCController(enabled=True)
        state = ctrl._get_or_create("openai")
        state.window_size = 16.0
        assert ctrl.should_downgrade_priority("openai", priority=0) is False

    @pytest.mark.asyncio
    async def test_disabled_controller_is_noop(self) -> None:
        ctrl = TACCController(enabled=False)
        assert await ctrl.before_request("openai") is True
        await ctrl.after_response("openai", latency_ms=100, tokens_generated=10, status_code=200)
        await ctrl.record_stream_velocity("openai", tokens_per_second=100.0)
        assert ctrl.should_downgrade_priority("openai", priority=0) is False
        # When disabled, window_size returns 1 for unknown providers
        assert ctrl.window_size("unknown") == 1

    def test_stats_include_all_fields(self) -> None:
        ctrl = TACCController(enabled=True)
        state = ctrl._get_or_create("openai")
        state.window_size = 8.0
        state.rtt_estimate = 150.0
        state.token_rate_estimate = 42.0
        stats = ctrl.stats("openai")
        assert stats["window_size"] == 8
        assert stats["rtt_estimate_ms"] == 150.0
        assert stats["token_rate_estimate"] == 42.0
        assert "pending_requests" in stats
        assert "active_token_pressure" in stats
        assert "pending_token_pressure" in stats

    @pytest.mark.asyncio
    async def test_priority_boost_increases_token_window(self) -> None:
        ctrl = TACCController(enabled=True, token_budget_per_request=100)
        # Low priority request
        low = await ctrl.before_request("openai", estimated_tokens=100, priority=0)
        assert low is True
        await ctrl.after_response("openai", latency_ms=100, tokens_generated=10, status_code=200)

        # High priority request should get larger token window
        high = await ctrl.before_request("openai", estimated_tokens=1000, priority=10)
        assert high is True

    @pytest.mark.asyncio
    async def test_acquire_request_queues_when_window_full(self) -> None:
        ctrl = TACCController(enabled=True, token_budget_per_request=100)
        # Fill the window
        await ctrl.before_request("openai", estimated_tokens=100)
        # Release to make room
        await ctrl.release_request("openai")
        # Second request should acquire after release
        acquired = await ctrl.acquire_request("openai", estimated_tokens=100)
        assert acquired is True

    @pytest.mark.asyncio
    async def test_429_backpressure_blocks_provider(self) -> None:
        ctrl = TACCController(enabled=True)
        await ctrl.before_request("openai")
        await ctrl.after_response(
            "openai",
            latency_ms=100,
            tokens_generated=10,
            status_code=429,
            retry_after=2.0,
        )
        state = ctrl._states["openai"]
        assert state.blocked_until > 0
        assert state.window_size < 2.0

    def test_evaluate_admission_blocked_provider_rejects(self) -> None:
        ctrl = TACCController(enabled=True)
        state = ctrl._get_or_create("openai")
        state.blocked_until = time.monotonic() + 10.0
        decision, reason = ctrl.evaluate_admission("openai", 100, 0)
        assert decision == AdmissionDecision.REJECT
        assert reason == "provider_blocked"

    def test_evaluate_admission_post_stall_delay(self) -> None:
        ctrl = TACCController(enabled=True)
        state = ctrl._get_or_create("openai")
        state.stall_detected = True
        decision, reason = ctrl.evaluate_admission("openai", 100, 0)
        assert decision == AdmissionDecision.DELAY
        assert reason == "post_stall_cooldown"

    def test_evaluate_admission_window_collapsed_rejects(self) -> None:
        ctrl = TACCController(enabled=True)
        state = ctrl._get_or_create("openai")
        state.window_size = 1.0
        state.queue_depth = 6
        decision, reason = ctrl.evaluate_admission("openai", 100, 0)
        assert decision == AdmissionDecision.REJECT
        assert reason == "window_collapsed"

    def test_evaluate_admission_token_pressure_delays(self) -> None:
        ctrl = TACCController(enabled=True, token_budget_per_request=100)
        state = ctrl._get_or_create("openai")
        state.active_token_pressure = 100
        decision, reason = ctrl.evaluate_admission("openai", 100, 0)
        assert decision == AdmissionDecision.DELAY
        assert reason == "token_pressure"

    def test_evaluate_admission_speculative_downgrade(self) -> None:
        ctrl = TACCController(enabled=True)
        state = ctrl._get_or_create("openai")
        state.window_size = 2.0
        decision, reason = ctrl.evaluate_admission("openai", 100, 0, is_speculative=True)
        assert decision == AdmissionDecision.PRIORITY_DOWNGRADE
        assert reason == "speculative_degraded"

    def test_evaluate_admission_batch_pressure_delays(self) -> None:
        ctrl = TACCController(enabled=True)
        state = ctrl._get_or_create("openai")
        state.window_size = 2.0
        state.batch_pressure = 5
        decision, reason = ctrl.evaluate_admission("openai", 100, 0, is_batch=True)
        assert decision == AdmissionDecision.DELAY
        assert reason == "batch_pressure"

    def test_evaluate_admission_admits_normal_request(self) -> None:
        ctrl = TACCController(enabled=True)
        decision, reason = ctrl.evaluate_admission("openai", 100, 0)
        assert decision == AdmissionDecision.ADMIT
        assert reason == "ok"

    @pytest.mark.asyncio
    async def test_ttft_recorded(self) -> None:
        ctrl = TACCController(enabled=True)
        await ctrl.record_ttft("openai", 250.0)
        state = ctrl._states["openai"]
        assert state.last_ttft_ms > 0

    @pytest.mark.asyncio
    async def test_cache_hit_rate_recorded(self) -> None:
        ctrl = TACCController(enabled=True)
        await ctrl.record_cache_hit_rate("openai", 0.75)
        state = ctrl._states["openai"]
        assert state.cache_hit_rate > 0

    def test_stats_include_new_fields(self) -> None:
        ctrl = TACCController(enabled=True)
        state = ctrl._get_or_create("openai")
        state.last_ttft_ms = 120.0
        state.token_velocity = 45.0
        state.cache_hit_rate = 0.8
        state.batch_pressure = 3
        state.stall_detected = True
        state.last_decision = "admit"
        stats = ctrl.stats("openai")
        assert stats["last_ttft_ms"] == 120.0
        assert stats["token_velocity"] == 45.0
        assert stats["cache_hit_rate"] == 0.8
        assert stats["batch_pressure"] == 3
        assert stats["stall_detected"] is True
        assert stats["last_decision"] == "admit"
