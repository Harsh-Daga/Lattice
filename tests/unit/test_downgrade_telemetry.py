"""Tests for formal downgrade telemetry and phase-aware stall detection."""

from __future__ import annotations

import pytest

from lattice.core.telemetry import DowngradeCategory, DowngradeTelemetry, TransportOutcome
from lattice.providers.stall_detector import StreamStallDetector


class TestDowngradeTelemetry:
    """Unit tests for DowngradeTelemetry."""

    def test_record_increments_count(self) -> None:
        dt = DowngradeTelemetry()
        dt.record(DowngradeCategory.HTTP2_TO_HTTP11, reason="h2_unavailable")
        assert dt.snapshot()["counts"]["http2_to_http11"] == 1

    def test_multiple_records_aggregate(self) -> None:
        dt = DowngradeTelemetry()
        dt.record(DowngradeCategory.DELTA_TO_FULL_PROMPT, reason="session_not_found")
        dt.record(DowngradeCategory.DELTA_TO_FULL_PROMPT, reason="version_mismatch")
        assert dt.snapshot()["counts"]["delta_to_full_prompt"] == 2

    def test_recent_reasons_capped(self) -> None:
        dt = DowngradeTelemetry()
        for i in range(15):
            dt.record(DowngradeCategory.BATCHING_BYPASSED, reason=f"r{i}")
        reasons = dt.snapshot()["recent_reasons"]["batching_bypassed"]
        assert len(reasons) == 5

    def test_total_events(self) -> None:
        dt = DowngradeTelemetry()
        dt.record(DowngradeCategory.HTTP2_TO_HTTP11)
        dt.record(DowngradeCategory.DELTA_TO_FULL_PROMPT)
        assert dt.total_events == 2

    def test_reset_clears(self) -> None:
        dt = DowngradeTelemetry()
        dt.record(DowngradeCategory.HTTP2_TO_HTTP11)
        dt.reset()
        assert dt.total_events == 0

    def test_snapshot_serializable(self) -> None:
        dt = DowngradeTelemetry()
        dt.record(DowngradeCategory.STREAM_STALL_DETECTED, reason="timeout")
        snap = dt.snapshot()
        assert "counts" in snap
        assert "recent_reasons" in snap
        assert snap["counts"]["stream_stall_detected"] == 1


class TestPhaseAwareStallDetection:
    """Tests for phase-aware stall thresholds."""

    def test_first_chunk_longer_tolerance(self) -> None:
        det = StreamStallDetector()
        # first_chunk phase gets 1.5x tolerance
        tol_streaming = det._get_tolerance_ms("openai", "streaming")
        tol_first = det._get_tolerance_ms("openai", "first_chunk")
        assert tol_first == tol_streaming * 1.5

    def test_thinking_longest_tolerance(self) -> None:
        det = StreamStallDetector()
        tol_streaming = det._get_tolerance_ms("openai", "streaming")
        tol_thinking = det._get_tolerance_ms("openai", "thinking")
        assert tol_thinking == tol_streaming * 2.0

    def test_tool_call_moderate_tolerance(self) -> None:
        det = StreamStallDetector()
        tol_streaming = det._get_tolerance_ms("openai", "streaming")
        tol_tool = det._get_tolerance_ms("openai", "tool_call")
        assert tol_tool == tol_streaming * 1.2

    def test_grace_period_uses_phase_tolerance(self) -> None:
        det = StreamStallDetector()
        det.start_stream("openai", stream_id="s1")
        # thinking phase: 30s * 2.0 = 60s tolerance
        # grace period: 60s * 0.5 = 30s
        # So 25s silence should NOT be stalled
        assert not det.is_stalled(
            "openai",
            since_last_chunk_ms=25000.0,
            fallback_timeout_ms=120000.0,
            stream_id="s1",
        )

    def test_thinking_phase_stalled_after_grace(self) -> None:
        det = StreamStallDetector()
        det.start_stream("openai", stream_id="s1")
        det.record_chunk("openai", "thinking", 100.0, tokens=0, stream_id="s1")
        # thinking phase: 60s tolerance
        # After grace (30s), 70s silence exceeds tolerance → stalled
        assert det.is_stalled(
            "openai",
            since_last_chunk_ms=70000.0,
            fallback_timeout_ms=120000.0,
            stream_id="s1",
        )

    def test_provider_specific_baselines(self) -> None:
        det = StreamStallDetector()
        assert det._get_tolerance_ms("groq", "streaming") == 15000.0
        assert det._get_tolerance_ms("anthropic", "streaming") == 45000.0
        assert det._get_tolerance_ms("unknown", "streaming") == 30000.0


class TestDowngradeCategoryCompleteness:
    """Ensure the taxonomy covers all planned categories."""

    def test_all_expected_categories_exist(self) -> None:
        expected = {
            "binary_to_json",
            "delta_to_full_prompt",
            "http2_to_http11",
            "stream_resume_to_full",
            "batching_bypassed",
            "speculation_bypassed",
            "cache_arbitrage_skipped",
            "transform_skipped",
            "semantic_cache_exact_hit",
            "semantic_cache_approximate_hit",
            "semantic_cache_miss",
            "semantic_cache_disabled",
            "provider_routing_failure",
            "provider_capability_mismatch",
            "stream_stall_detected",
            "tacc_delayed",
            "tacc_rejected",
        }
        actual = {c.value for c in DowngradeCategory}
        assert expected == actual


class TestTransportOutcome:
    """Tests for canonical TransportOutcome."""

    def test_default_outcome_is_json_bypassed(self) -> None:
        outcome = TransportOutcome()
        assert outcome.framing == "json"
        assert outcome.delta_mode == "bypassed"
        assert outcome.semantic_cache_status == ""

    def test_headers_from_outcome(self) -> None:
        outcome = TransportOutcome(
            framing="native",
            delta_mode="delta",
            http_version="http/2",
            semantic_cache_status="exact-hit",
            batching_status="batched",
            speculative_status="hit",
            stream_resumed=True,
        )
        headers = outcome.to_headers()
        assert headers["x-lattice-framing"] == "native"
        assert headers["x-lattice-delta"] == "delta"
        assert headers["x-lattice-http-version"] == "http/2"
        assert headers["x-lattice-semantic-cache"] == "exact-hit"
        assert headers["x-lattice-batching"] == "batched"
        assert headers["x-lattice-speculative-status"] == "hit"
        assert headers["x-lattice-stream-resumed"] == "true"

    def test_downgrade_categories_extracted(self) -> None:
        outcome = TransportOutcome(
            framing="json",
            framing_fallback_reason="crc_mismatch",
            delta_mode="bypassed",
            delta_fallback_reason="version_mismatch",
            http_version="http/1.1",
            http_fallback_reason="h2_unavailable",
            batching_status="bypassed",
        )
        cats = outcome.to_downgrade_categories()
        values = {c.value for c in cats}
        assert "binary_to_json" in values
        assert "delta_to_full_prompt" in values
        assert "http2_to_http11" in values
        assert "batching_bypassed" in values

    def test_cache_miss_without_fallback_reason(self) -> None:
        """Semantic cache miss should be classified without needing fallback_reason."""
        outcome = TransportOutcome(
            semantic_cache_status="miss",
        )
        cats = outcome.to_downgrade_categories()
        values = {c.value for c in cats}
        assert "semantic_cache_miss" in values

    def test_speculation_bypassed_without_fallback_reason(self) -> None:
        """Speculation bypass should be classified without needing fallback_reason."""
        outcome = TransportOutcome(
            speculative_status="bypassed",
        )
        cats = outcome.to_downgrade_categories()
        values = {c.value for c in cats}
        assert "speculation_bypassed" in values

    def test_stream_resume_with_dedicated_reason(self) -> None:
        """Stream resume fallback uses its own reason field."""
        outcome = TransportOutcome(
            stream_resumed=True,
            stream_resume_fallback_reason="token_expired",
        )
        cats = outcome.to_downgrade_categories()
        values = {c.value for c in cats}
        assert "stream_resume_to_full" in values

    def test_no_false_positive_when_no_reason(self) -> None:
        """A subsystem in bypassed state without an explicit reason should still classify."""
        outcome = TransportOutcome(
            batching_status="bypassed",
            speculative_status="bypassed",
            semantic_cache_status="miss",
        )
        cats = outcome.to_downgrade_categories()
        values = {c.value for c in cats}
        assert "batching_bypassed" in values
        assert "speculation_bypassed" in values
        assert "semantic_cache_miss" in values
        # Should NOT have generic categories without explicit reasons
        assert "binary_to_json" not in values
        assert "delta_to_full_prompt" not in values

    def test_stats_roundtrip(self) -> None:
        outcome = TransportOutcome(
            framing="native",
            delta_mode="delta",
            semantic_cache_status="approximate-hit",
        )
        stats = outcome.to_stats()
        assert stats["framing"] == "native"
        assert stats["delta_mode"] == "delta"
        assert stats["semantic_cache_status"] == "approximate-hit"


class TestTransportOutcomeOverrideSemantics:
    """Phase 0: Explicit False overrides on legacy booleans."""

    def test_false_suppresses_speculative_when_canonical_is_hit(self) -> None:
        from lattice.core.telemetry import TransportOutcome
        from lattice.gateway.compat import build_routing_headers

        outcome = TransportOutcome(speculative_status="hit")
        headers = build_routing_headers(
            "gpt-4",
            transport_outcome=outcome,
            used_speculative=False,
        )
        assert "x-lattice-speculative" not in headers
        assert "x-lattice-speculative-status" not in headers

    def test_false_suppresses_batched_when_canonical_is_batched(self) -> None:
        from lattice.core.telemetry import TransportOutcome
        from lattice.gateway.compat import build_routing_headers

        outcome = TransportOutcome(batching_status="batched")
        headers = build_routing_headers(
            "gpt-4",
            transport_outcome=outcome,
            batched=False,
        )
        assert "x-lattice-batched" not in headers
        assert "x-lattice-batching" not in headers

    def test_false_suppresses_stream_resumed_when_canonical_is_true(self) -> None:
        from lattice.core.telemetry import TransportOutcome
        from lattice.gateway.compat import build_routing_headers

        outcome = TransportOutcome(stream_resumed=True)
        headers = build_routing_headers(
            "gpt-4",
            transport_outcome=outcome,
            stream_resumed=False,
        )
        assert "x-lattice-stream-resumed" not in headers

    def test_none_preserves_canonical_value(self) -> None:
        from lattice.core.telemetry import TransportOutcome
        from lattice.gateway.compat import build_routing_headers

        outcome = TransportOutcome(
            speculative_status="hit",
            batching_status="batched",
            stream_resumed=True,
        )
        headers = build_routing_headers(
            "gpt-4",
            transport_outcome=outcome,
            used_speculative=None,
            batched=None,
            stream_resumed=None,
        )
        assert headers["x-lattice-speculative-status"] == "hit"
        assert headers["x-lattice-batching"] == "batched"
        assert headers["x-lattice-stream-resumed"] == "true"

    def test_true_overrides_canonical_speculative(self) -> None:
        from lattice.core.telemetry import TransportOutcome
        from lattice.gateway.compat import build_routing_headers

        outcome = TransportOutcome(speculative_status="miss")
        headers = build_routing_headers(
            "gpt-4",
            transport_outcome=outcome,
            used_speculative=True,
            prediction_hit=True,
        )
        # Legacy alias is set to hit even though canonical was miss
        assert headers["x-lattice-speculative"] == "hit"
        # Canonical still present
        assert headers["x-lattice-speculative-status"] == "miss"

    def test_stream_resume_fallback_reason_still_visible(self) -> None:
        from lattice.core.telemetry import TransportOutcome
        from lattice.gateway.compat import build_routing_headers

        outcome = TransportOutcome(
            stream_resumed=True,
            stream_resume_fallback_reason="token_expired",
        )
        headers = build_routing_headers("gpt-4", transport_outcome=outcome)
        assert headers["x-lattice-stream-resumed"] == "true"
        assert headers["x-lattice-stream-resume-fallback-reason"] == "token_expired"
