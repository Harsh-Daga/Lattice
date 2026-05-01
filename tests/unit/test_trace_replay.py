"""Tests for the trace replay benchmark runner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from benchmarks.evals.replay import (
    ReplayTrace,
    _render_markdown,
    _write_outputs,
    load_traces,
    run_feature_isolated_replay,
    run_trace_replay,
    select_traces,
)


def test_load_traces_from_jsonl() -> None:
    path = Path("benchmarks/datasets/replay_traces.jsonl")
    traces = load_traces(path)
    assert len(traces) >= 3
    assert traces[0].trace_id
    assert traces[0].messages


def test_select_traces() -> None:
    traces = [
        ReplayTrace(trace_id="t1", scenario="alpha", category="a"),
        ReplayTrace(trace_id="t2", scenario="beta", category="b"),
    ]
    selected = select_traces(traces, ["beta"])
    assert len(selected) == 1
    assert selected[0].trace_id == "t2"


@pytest.mark.asyncio
async def test_run_trace_replay_summary() -> None:
    traces = [
        ReplayTrace(
            trace_id="trace_test_001",
            scenario="test",
            category="reference_substitution",
            messages=[
                {"role": "system", "content": "Debug transaction failures."},
                {
                    "role": "user",
                    "content": "Transactions failed: 550e8400-e29b-41d4-a716-446655440000",
                },
            ],
            reference_response="Duplicate UUIDs caused a conflict.",
            optimized_response="Duplicate UUIDs caused a conflict.",
        )
    ]
    report = await run_trace_replay(
        traces, model="gpt-4", provider="openai", iterations=1, warmup=0
    )
    assert report.total_scenarios == 1
    assert report.passed_scenarios == 1
    assert report.total_token_savings >= 0
    assert report.scenarios[0].scenario_name == "trace_test_001"
    assert report.scenarios[0].avg_quality.passed is True


@pytest.mark.asyncio
async def test_trace_replay_outputs(tmp_path: Path) -> None:
    traces = [
        ReplayTrace(
            trace_id="trace_test_002",
            scenario="test",
            category="reference_substitution",
            messages=[{"role": "user", "content": "hello"}],
            reference_response="hello",
            optimized_response="hello",
        )
    ]
    report = await run_trace_replay(
        traces, model="gpt-4", provider="openai", iterations=1, warmup=0
    )
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    _write_outputs(report, output_json=str(json_path), output_md=str(md_path))
    assert json_path.exists()
    assert md_path.exists()
    assert "Trace Replay Benchmark" in md_path.read_text()
    assert _render_markdown(report).startswith("# Trace Replay Benchmark")


@pytest.mark.asyncio
async def test_feature_isolated_replay_runs_all_features() -> None:
    traces = [
        ReplayTrace(
            trace_id="t1",
            scenario="test",
            category="test",
            messages=[{"role": "user", "content": "hello world"}],
            reference_response="hello",
            optimized_response="hello",
        )
    ]
    reports = await run_feature_isolated_replay(
        traces, model="gpt-4", provider="openai", iterations=1, warmup=0
    )
    feature_names = [
        "semantic_cache",
        "cache_arbitrage",
        "stall_detection",
        "batching",
        "speculation",
        "tacc",
        "semantic_compress",
        "message_dedup",
        "reference_sub",
    ]
    assert "baseline" in reports
    assert "bare" in reports
    for name in feature_names:
        assert f"{name}_off" in reports, f"missing {name}_off"
        assert f"{name}_on" in reports, f"missing {name}_on"


def test_replay_detects_quality_regression() -> None:
    from benchmarks.evals.replay import FailureCategory, _classify_failure
    from benchmarks.framework.types import BenchmarkReport, QualityMeasurement, ScenarioResult

    baseline = BenchmarkReport(
        runner_name="baseline",
        provider="openai",
        model="gpt-4",
        scenarios=[
            ScenarioResult(
                scenario_name="s1",
                category="test",
                qualities=[QualityMeasurement(semantic_similarity=0.95)],
            )
        ],
    )
    degraded = BenchmarkReport(
        runner_name="feature_off",
        provider="openai",
        model="gpt-4",
        scenarios=[
            ScenarioResult(
                scenario_name="s1",
                category="test",
                qualities=[QualityMeasurement(semantic_similarity=0.80)],
            )
        ],
    )
    categories = _classify_failure(
        baseline, degraded, quality_threshold=0.05, latency_threshold=1.5
    )
    assert FailureCategory.QUALITY_DROP in categories


def test_replay_detects_latency_regression() -> None:
    from benchmarks.evals.replay import FailureCategory, _classify_failure
    from benchmarks.framework.types import BenchmarkReport, LatencyMeasurement, ScenarioResult

    baseline = BenchmarkReport(
        runner_name="baseline",
        provider="openai",
        model="gpt-4",
        scenarios=[
            ScenarioResult(
                scenario_name="s1",
                category="test",
                optimized_latencies=[LatencyMeasurement(pipeline_ms=10.0, total_ms=10.0)],
            )
        ],
    )
    slower = BenchmarkReport(
        runner_name="feature_off",
        provider="openai",
        model="gpt-4",
        scenarios=[
            ScenarioResult(
                scenario_name="s1",
                category="test",
                optimized_latencies=[LatencyMeasurement(pipeline_ms=20.0, total_ms=20.0)],
            )
        ],
    )
    categories = _classify_failure(baseline, slower, quality_threshold=0.05, latency_threshold=1.5)
    assert FailureCategory.TRANSPORT_DOWNGRADE in categories


def test_failure_categories_present() -> None:
    from benchmarks.evals.replay import FailureCategory, _classify_failure
    from benchmarks.framework.types import (
        BenchmarkReport,
        LatencyMeasurement,
        QualityMeasurement,
        ScenarioResult,
        TokenMeasurement,
    )

    baseline = BenchmarkReport(
        runner_name="baseline",
        provider="openai",
        model="gpt-4",
        scenarios=[
            ScenarioResult(
                scenario_name="s1",
                category="test",
                optimized_latencies=[LatencyMeasurement(pipeline_ms=10.0, total_ms=10.0)],
                optimized_tokens=[TokenMeasurement(before=100, after=90, saved=10, ratio=0.1)],
                qualities=[QualityMeasurement(semantic_similarity=0.95)],
            )
        ],
    )
    # Construct a feature report that triggers multiple categories
    feature = BenchmarkReport(
        runner_name="cache_off",
        provider="openai",
        model="gpt-4",
        scenarios=[
            ScenarioResult(
                scenario_name="s1",
                category="test",
                optimized_latencies=[LatencyMeasurement(pipeline_ms=20.0, total_ms=20.0)],
                optimized_tokens=[TokenMeasurement(before=100, after=95, saved=5, ratio=0.05)],
                qualities=[QualityMeasurement(semantic_similarity=0.80)],
                baseline_errors=["routing timeout"],
                optimized_errors=["stall detected"],
            )
        ],
    )
    categories = _classify_failure(baseline, feature, quality_threshold=0.05, latency_threshold=1.5)
    values = {c.value for c in categories}
    # Ensure at least 5 distinct failure categories appear
    assert FailureCategory.QUALITY_DROP.value in values
    assert FailureCategory.TRANSPORT_DOWNGRADE.value in values
    assert FailureCategory.ROUTING_FAILURE.value in values
    assert FailureCategory.STALL_MISCLASSIFICATION.value in values
    assert FailureCategory.CACHE_MISS.value in values


@pytest.mark.asyncio
async def test_governance_report_pass_fail(tmp_path: Path) -> None:
    from benchmarks.evals.runner import run_replay_governance

    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "trace_id": "t1",
                "scenario": "test",
                "category": "test",
                "messages": [{"role": "user", "content": "hello world"}],
                "reference_response": "hello",
                "optimized_response": "hello",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = await run_replay_governance(
        input_path=str(trace_path),
        model="gpt-4",
        provider="openai",
        iterations=1,
        warmup=0,
        regression_threshold_quality=0.05,
        regression_threshold_latency=1.5,
    )
    assert report.name == "replay_governance"
    assert "feature_count" in report.summary
    gov = report.details["governance"]
    for feature in gov:
        assert gov[feature]["status"] in ("pass", "fail")


@pytest.mark.asyncio
async def test_semantic_cache_exact_vs_approximate_hit() -> None:
    from lattice.core.semantic_cache import (
        CachedResponse,
        SemanticCache,
        compute_cache_key,
    )

    cache = SemanticCache(ttl_seconds=300, enabled=True)
    req1 = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello world"}],
    }
    key1 = compute_cache_key(req1)
    await cache.set(key1, CachedResponse(content="hi"), req1)

    # Exact hit
    hit1 = await cache.get(key1)
    assert hit1 is not None
    assert hit1.content == "hi"

    # Approximate hit (near-duplicate)
    req2 = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hello world!"}],
    }
    key2 = compute_cache_key(req2)
    hit2 = await cache.get(key2, req2)
    assert hit2 is not None
    assert hit2.content == "hi"

    stats = await cache.stats
    assert stats["exact_hits"] == 1
    assert stats["approximate_hits"] == 1


def test_stall_detection_signal_accuracy() -> None:
    from lattice.providers.stall_detector import StreamStallDetector

    detector = StreamStallDetector()
    detector.start_stream("openai", "stream-1")

    # Healthy fast stream
    for _ in range(5):
        detector.record_chunk("openai", "chunk", 100.0, tokens=50, stream_id="stream-1")

    # Short silence — not stalled
    assert not detector.is_stalled(
        "openai", since_last_chunk_ms=5000.0, fallback_timeout_ms=60000.0, stream_id="stream-1"
    )

    # Long silence — stalled
    assert detector.is_stalled(
        "openai", since_last_chunk_ms=35000.0, fallback_timeout_ms=60000.0, stream_id="stream-1"
    )


def test_transport_negotiation_success_vs_fallback() -> None:
    from lattice.core.delta_wire import DeltaWireEncoder
    from lattice.protocol.framing import BinaryFramer

    # Delta negotiation: success
    delta_enc = DeltaWireEncoder()
    outcome = delta_enc.encode_negotiation_outcome(True, "")
    assert outcome["delta_accepted"] is True
    assert outcome["delta_fallback_reason"] == ""

    # Delta negotiation: fallback
    outcome = delta_enc.encode_negotiation_outcome(False, "version_mismatch")
    assert outcome["delta_accepted"] is False
    assert outcome["delta_fallback_reason"] == "version_mismatch"

    # Framing negotiation: success
    framer = BinaryFramer()
    frame = framer.encode_negotiation_outcome(True, "")
    accepted, reason = framer.decode_negotiation_outcome(frame)
    assert accepted is True
    assert reason == ""

    # Framing negotiation: fallback
    frame = framer.encode_negotiation_outcome(False, "crc_mismatch")
    accepted, reason = framer.decode_negotiation_outcome(frame)
    assert accepted is False
    assert reason == "crc_mismatch"


@pytest.mark.asyncio
async def test_degraded_path_multiple_issues() -> None:
    from benchmarks.evals.replay import (
        FailureCategory,
        _classify_failure,
    )
    from benchmarks.framework.types import (
        BenchmarkReport,
        LatencyMeasurement,
        QualityMeasurement,
        ScenarioResult,
        TokenMeasurement,
    )

    baseline = BenchmarkReport(
        runner_name="baseline",
        provider="openai",
        model="gpt-4",
        scenarios=[
            ScenarioResult(
                scenario_name="s1",
                category="test",
                optimized_latencies=[LatencyMeasurement(pipeline_ms=10.0, total_ms=10.0)],
                optimized_tokens=[TokenMeasurement(before=100, after=80, saved=20, ratio=0.2)],
                qualities=[QualityMeasurement(semantic_similarity=0.95)],
            )
        ],
    )

    # Degraded: slower, less savings, worse quality, plus errors
    degraded = BenchmarkReport(
        runner_name="degraded",
        provider="openai",
        model="gpt-4",
        scenarios=[
            ScenarioResult(
                scenario_name="s1",
                category="test",
                optimized_latencies=[LatencyMeasurement(pipeline_ms=25.0, total_ms=25.0)],
                optimized_tokens=[TokenMeasurement(before=100, after=95, saved=5, ratio=0.05)],
                qualities=[QualityMeasurement(semantic_similarity=0.78)],
                baseline_errors=["routing timeout"],
                optimized_errors=["stall detected", "cache miss"],
            )
        ],
    )

    categories = _classify_failure(
        baseline, degraded, quality_threshold=0.05, latency_threshold=1.5
    )
    values = {c.value for c in categories}

    assert FailureCategory.QUALITY_DROP.value in values
    assert FailureCategory.TRANSPORT_DOWNGRADE.value in values
    assert FailureCategory.ROUTING_FAILURE.value in values
    assert FailureCategory.STALL_MISCLASSIFICATION.value in values
    assert FailureCategory.PIPELINE_ERROR.value in values

    # CACHE_MISS only triggered when runner_name contains "cache"
    cache_degraded = BenchmarkReport(
        runner_name="cache_off",
        provider="openai",
        model="gpt-4",
        scenarios=[
            ScenarioResult(
                scenario_name="s1",
                category="test",
                optimized_latencies=[LatencyMeasurement(pipeline_ms=25.0, total_ms=25.0)],
                optimized_tokens=[TokenMeasurement(before=100, after=95, saved=5, ratio=0.05)],
                qualities=[QualityMeasurement(semantic_similarity=0.78)],
            )
        ],
    )
    cache_categories = _classify_failure(
        baseline, cache_degraded, quality_threshold=0.05, latency_threshold=1.5
    )
    cache_values = {c.value for c in cache_categories}
    assert FailureCategory.CACHE_MISS.value in cache_values


def test_concurrent_stream_stall_regression() -> None:
    """Stall on one stream should not affect another concurrent stream."""
    from lattice.providers.stall_detector import StreamStallDetector

    detector = StreamStallDetector()
    detector.start_stream("openai", "stream-fast")
    detector.start_stream("openai", "stream-slow")

    # Fast stream gets regular chunks
    for _ in range(5):
        detector.record_chunk("openai", "chunk", 100.0, tokens=50, stream_id="stream-fast")

    # Slow stream gets no chunks and should be stalled
    assert detector.is_stalled(
        "openai",
        since_last_chunk_ms=35000.0,
        fallback_timeout_ms=60000.0,
        stream_id="stream-slow",
    )

    # Fast stream should NOT be stalled
    assert not detector.is_stalled(
        "openai",
        since_last_chunk_ms=5000.0,
        fallback_timeout_ms=60000.0,
        stream_id="stream-fast",
    )


def test_cache_arbitrage_manifest_provenance() -> None:
    from lattice.core.context import TransformContext
    from lattice.core.result import unwrap
    from lattice.core.transport import Message, Request
    from lattice.transforms.cache_arbitrage import CacheArbitrageOptimizer

    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="system", content="System prompt."),
            Message(role="user", content="Hello."),
        ]
    )
    context = TransformContext(provider="openai")
    result = transform.process(request, context)
    modified = unwrap(result)
    arb = modified.metadata.get("_cache_arbitrage", {})
    assert arb.get("manifest_source") == "reconstructed"
    assert arb.get("plan_applied") is True


def test_transport_negotiation_downgrade_visible() -> None:
    from lattice.core.delta_wire import DeltaWireEncoder
    from lattice.core.telemetry import DowngradeCategory, TransportOutcome

    # Delta fallback
    delta_enc = DeltaWireEncoder()
    outcome = delta_enc.encode_negotiation_outcome(
        accepted=False, fallback_reason="version_mismatch"
    )
    assert outcome["delta_accepted"] is False
    assert outcome["delta_fallback_reason"] == "version_mismatch"

    # TransportOutcome captures it
    to = TransportOutcome(
        delta_mode="bypassed",
        delta_fallback_reason="version_mismatch",
    )
    cats = to.to_downgrade_categories()
    assert DowngradeCategory.DELTA_TO_FULL_PROMPT in cats


def test_batching_speculation_tacc_telemetry() -> None:
    from lattice.core.telemetry import DowngradeCategory, TransportOutcome

    # Batching bypassed (no fallback_reason required)
    to = TransportOutcome(
        batching_status="bypassed",
    )
    assert DowngradeCategory.BATCHING_BYPASSED in to.to_downgrade_categories()

    # Speculation bypassed (no fallback_reason required)
    to = TransportOutcome(
        speculative_status="bypassed",
    )
    assert DowngradeCategory.SPECULATION_BYPASSED in to.to_downgrade_categories()

    # Stream resume fallback uses dedicated reason field
    to = TransportOutcome(
        stream_resumed=True,
        stream_resume_fallback_reason="resume_token_expired",
    )
    assert DowngradeCategory.STREAM_RESUME_TO_FULL in to.to_downgrade_categories()

    # Stream resumed without fallback_reason should NOT classify as downgrade
    to = TransportOutcome(
        stream_resumed=True,
    )
    assert DowngradeCategory.STREAM_RESUME_TO_FULL not in to.to_downgrade_categories()


@pytest.mark.asyncio
async def test_semantic_cache_maintenance_tracked_in_stats() -> None:
    from lattice.core.semantic_cache import CachedResponse, SemanticCache

    cache = SemanticCache(ttl_seconds=0, enabled=True)
    await cache.set("a", CachedResponse(content="x"))
    await asyncio.sleep(0.1)
    removed = await cache.expire_stale()
    assert removed == 1

    stats = await cache.stats
    assert stats["maintenance_runs"] == 1
    assert stats["stale_removed"] == 1


def test_transport_outcome_headers_and_categories_consistent() -> None:
    """Headers and downgrade categories should agree on the same state."""
    from lattice.core.telemetry import TransportOutcome

    to = TransportOutcome(
        framing="json",
        framing_fallback_reason="crc_mismatch",
        delta_mode="bypassed",
        delta_fallback_reason="version_mismatch",
        http_version="http/1.1",
        http_fallback_reason="h2_unavailable",
        semantic_cache_status="miss",
        batching_status="bypassed",
        speculative_status="bypassed",
        stream_resumed=True,
        stream_resume_fallback_reason="token_expired",
    )

    headers = to.to_headers()
    categories = to.to_downgrade_categories()
    category_values = {c.value for c in categories}

    # Headers exist for each subsystem
    assert headers["x-lattice-framing"] == "json"
    assert headers["x-lattice-delta"] == "bypassed"
    assert headers["x-lattice-http-version"] == "http/1.1"
    assert headers["x-lattice-semantic-cache"] == "miss"
    assert headers["x-lattice-batching"] == "bypassed"
    assert headers["x-lattice-speculative-status"] == "bypassed"
    assert headers["x-lattice-stream-resumed"] == "true"

    # Categories match
    assert "binary_to_json" in category_values
    assert "delta_to_full_prompt" in category_values
    assert "http2_to_http11" in category_values
    assert "semantic_cache_miss" in category_values
    assert "batching_bypassed" in category_values
    assert "speculation_bypassed" in category_values
    assert "stream_resume_to_full" in category_values


def test_transport_outcome_precedence_over_canonical() -> None:
    """Legacy params should override TransportOutcome values in headers."""
    from lattice.core.telemetry import TransportOutcome
    from lattice.gateway.compat import build_routing_headers

    outcome = TransportOutcome(
        framing="native",
        delta_mode="delta",
        http_version="http/2",
    )
    headers = build_routing_headers(
        "gpt-4",
        transport_outcome=outcome,
        framing="json",  # explicit override
        delta_mode="bypassed",  # explicit override
        http_version="http/1.1",  # explicit override
    )
    assert headers["x-lattice-framing"] == "json"
    assert headers["x-lattice-delta"] == "bypassed"
    assert headers["x-lattice-http-version"] == "http/1.1"


def test_stream_resume_fallback_reason_visible_in_headers() -> None:
    """stream_resume_fallback_reason should appear as its own header."""
    from lattice.core.telemetry import TransportOutcome

    to = TransportOutcome(
        stream_resumed=True,
        stream_resume_fallback_reason="resume_token_expired",
    )
    headers = to.to_headers()
    assert headers["x-lattice-stream-resumed"] == "true"
    assert headers["x-lattice-stream-resume-fallback-reason"] == "resume_token_expired"


def test_ignored_chunk_updates_observable() -> None:
    """Unknown stream IDs should increment ignored_chunk_count."""
    from lattice.providers.stall_detector import StreamStallDetector

    detector = StreamStallDetector()
    detector.start_stream("openai", "stream-known")
    detector.record_chunk("openai", "chunk", 100.0, tokens=50, stream_id="stream-unknown")
    assert detector.get_ignored_chunk_count() == 1

    # Known stream should not affect counter
    detector.record_chunk("openai", "chunk", 100.0, tokens=50, stream_id="stream-known")
    assert detector.get_ignored_chunk_count() == 1


@pytest.mark.asyncio
async def test_maintenance_coordinator_throttling() -> None:
    """MaintenanceCoordinator should not run more than once per interval."""
    from lattice.core.maintenance import MaintenanceCoordinator, MaintenanceResult

    coordinator = MaintenanceCoordinator(interval_seconds=60.0)
    call_count = 0

    async def _callback() -> MaintenanceResult:
        nonlocal call_count
        call_count += 1
        return MaintenanceResult()

    coordinator.register("test", _callback)

    await coordinator.tick()
    assert call_count == 1

    # Second tick within interval should be throttled
    result = await coordinator.tick()
    assert result == {}
    assert call_count == 1
