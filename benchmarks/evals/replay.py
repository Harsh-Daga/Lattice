"""Replay evaluation helpers for production evals."""

from __future__ import annotations

import argparse
import asyncio
import enum
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.framework.types import (
    BenchmarkReport,
    LatencyMeasurement,
    ScenarioResult,
    TokenMeasurement,
)
from benchmarks.metrics.quality import evaluate_response
from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.pipeline import CompressorPipeline
from lattice.core.pipeline_factory import build_default_pipeline
from lattice.core.result import unwrap
from lattice.core.serialization import message_from_dict
from lattice.core.transport import Request
from lattice.transforms.batching import BatchingTransform
from lattice.transforms.cache_arbitrage import CacheArbitrageOptimizer
from lattice.transforms.format_conv import FormatConverter
from lattice.transforms.message_dedup import MessageDeduplicator
from lattice.transforms.output_cleanup import OutputCleanup
from lattice.transforms.prefix_opt import PrefixOptimizer
from lattice.transforms.reference_sub import ReferenceSubstitution
from lattice.transforms.speculative import SpeculativeTransform
from lattice.transforms.tool_filter import ToolOutputFilter
from lattice.utils.token_count import count_message_tokens


REPLAY_FEATURE_FLAGS: list[tuple[str, str]] = [
    ("semantic_cache", "semantic_cache_enabled"),
    ("cache_arbitrage", "transform_cache_arbitrage"),
    ("stall_detection", "provider_stall_detection_enabled"),
    ("batching", "transform_batching"),
    ("speculation", "transform_speculation"),
    ("tacc", "tacc_enabled"),
    ("semantic_compress", "transform_semantic_compress"),
    ("message_dedup", "transform_message_dedup"),
    ("reference_sub", "transform_reference_sub"),
]


@dataclass(slots=True)
class ReplayTrace:
    """A single replayable trace entry."""

    trace_id: str
    scenario: str
    category: str
    provider: str = "openai"
    model: str = "gpt-4"
    description: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    reference_response: str = ""
    optimized_response: str = ""
    expect_json: bool = False
    json_schema: dict[str, Any] | None = None
    baseline_tool_calls: list[dict[str, Any]] | None = None
    optimized_tool_calls: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayTrace:
        return cls(
            trace_id=str(data["trace_id"]),
            scenario=str(data.get("scenario", data["trace_id"])),
            category=str(data.get("category", "replay")),
            provider=str(data.get("provider", "openai")),
            model=str(data.get("model", "gpt-4")),
            description=str(data.get("description", "")),
            messages=list(data.get("messages", [])),
            reference_response=str(data.get("reference_response", data.get("baseline_response", ""))),
            optimized_response=str(data.get("optimized_response", data.get("reference_response", data.get("baseline_response", "")))),
            expect_json=bool(data.get("expect_json", False)),
            json_schema=data.get("json_schema"),
            baseline_tool_calls=data.get("baseline_tool_calls"),
            optimized_tool_calls=data.get("optimized_tool_calls"),
            metadata=dict(data.get("metadata", {})),
        )


def load_traces(path: str | Path) -> list[ReplayTrace]:
    """Load replay traces from JSON or JSONL."""
    raw_path = Path(path)
    text = raw_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("trace replay file must contain a JSON array or JSONL records")
        return [ReplayTrace.from_dict(item) for item in data]
    traces: list[ReplayTrace] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        traces.append(ReplayTrace.from_dict(json.loads(line)))
    return traces


def select_traces(traces: list[ReplayTrace], names: list[str]) -> list[ReplayTrace]:
    """Filter traces by scenario or trace ID."""
    if not names:
        return traces
    wanted = {name.strip() for name in names if name.strip()}
    return [trace for trace in traces if trace.trace_id in wanted or trace.scenario in wanted]


def _build_pipeline(config: LatticeConfig) -> Any:
    """Build a compression pipeline that respects all feature flags."""
    pipeline = build_default_pipeline(
        config,
        include_execution_transforms=False,
    )
    if config.transform_batching:
        pipeline.register(BatchingTransform())
    if config.transform_speculation:
        pipeline.register(SpeculativeTransform())
    return pipeline


async def _measure_latency(run_fn: Any, iterations: int, warmup: int = 3) -> list[float]:
    for _ in range(warmup):
        await run_fn()
    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        await run_fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return samples


class FailureCategory(enum.Enum):
    """Categories of regression failures detected during replay governance."""

    ROUTING_FAILURE = "routing_failure"
    CACHE_MISS = "cache_miss"
    QUALITY_DROP = "quality_drop"
    TRANSPORT_DOWNGRADE = "transport_downgrade"
    STALL_MISCLASSIFICATION = "stall_misclassification"
    PIPELINE_ERROR = "pipeline_error"


def _classify_failure(
    baseline_report: BenchmarkReport,
    feature_report: BenchmarkReport,
    *,
    quality_threshold: float = 0.05,
    latency_threshold: float = 1.5,
) -> list[FailureCategory]:
    """Compare a feature report against baseline and flag regression categories."""
    categories: list[FailureCategory] = []

    # Quality drop
    baseline_quality = baseline_report.avg_quality_score
    feature_quality = feature_report.avg_quality_score
    if baseline_quality > 0 and (baseline_quality - feature_quality) > quality_threshold:
        categories.append(FailureCategory.QUALITY_DROP)

    # Latency increase (transport downgrade)
    baseline_latency = baseline_report.avg_pipeline_latency_ms
    feature_latency = feature_report.avg_pipeline_latency_ms
    if baseline_latency > 0 and feature_latency > baseline_latency * latency_threshold:
        categories.append(FailureCategory.TRANSPORT_DOWNGRADE)

    # Pipeline errors
    baseline_errors = sum(
        len(s.baseline_errors) + len(s.optimized_errors) for s in baseline_report.scenarios
    )
    feature_errors = sum(
        len(s.baseline_errors) + len(s.optimized_errors) for s in feature_report.scenarios
    )
    if feature_errors > baseline_errors:
        categories.append(FailureCategory.PIPELINE_ERROR)

    # Routing failure (heuristic: error messages mentioning routing)
    for s in feature_report.scenarios:
        for err in s.baseline_errors + s.optimized_errors:
            if "routing" in err.lower() or "route" in err.lower():
                categories.append(FailureCategory.ROUTING_FAILURE)
                break

    # Stall misclassification (heuristic: error messages mentioning stall)
    for s in feature_report.scenarios:
        for err in s.baseline_errors + s.optimized_errors:
            if "stall" in err.lower():
                categories.append(FailureCategory.STALL_MISCLASSIFICATION)
                break

    # Cache miss (for cache-related features, token savings regression)
    if "cache" in feature_report.runner_name and feature_report.total_token_savings < baseline_report.total_token_savings:
        categories.append(FailureCategory.CACHE_MISS)

    return categories


async def run_trace_replay(
    traces: list[ReplayTrace],
    *,
    model: str,
    provider: str,
    iterations: int = 3,
    warmup: int = 1,
    config: LatticeConfig | None = None,
) -> BenchmarkReport:
    """Replay traces through the compression pipeline and score the result."""
    pipeline = _build_pipeline(config or LatticeConfig(graceful_degradation=True))
    results: list[ScenarioResult] = []

    for trace in traces:
        trace_id = trace.trace_id
        trace_provider = trace.provider
        trace_model = trace.model or model
        trace_messages = list(trace.messages)
        request = Request(
            messages=[message_from_dict(m) for m in trace_messages],
            model=trace_model,
        )
        baseline_tokens = count_message_tokens(trace_messages, model=request.model)

        async def _run_pipeline(
            trace_id: str = trace_id,
            trace_provider: str = trace_provider,
            trace_model: str = trace_model,
            trace_messages: list[dict[str, Any]] = trace_messages,
        ) -> TransformContext:
            ctx = TransformContext(
                request_id=f"replay-{trace_id}",
                provider=trace_provider,
                model=trace_model,
            )
            await pipeline.process(
                Request(
                    messages=[message_from_dict(m) for m in trace_messages],
                    model=trace_model,
                ),
                ctx,
            )
            return ctx

        lat_samples = await _measure_latency(_run_pipeline, iterations=iterations, warmup=warmup)
        latency = LatencyMeasurement(
            pipeline_ms=statistics.mean(lat_samples) if lat_samples else 0.0,
            total_ms=statistics.mean(lat_samples) if lat_samples else 0.0,
        )

        ctx = await _run_pipeline()
        compressed = unwrap(
            await pipeline.process(
                Request(
                    messages=[message_from_dict(m) for m in trace_messages],
                    model=trace_model,
                ),
                ctx,
            )
        )
        optimized_tokens = count_message_tokens(
            [{"role": str(m.role), "content": m.content} for m in compressed.messages],
            model=request.model,
        )

        token_measurement = TokenMeasurement(
            before=baseline_tokens,
            after=optimized_tokens,
            saved=max(0, baseline_tokens - optimized_tokens),
            ratio=(baseline_tokens - optimized_tokens) / baseline_tokens if baseline_tokens > 0 else 0.0,
        )

        reference_response = trace.reference_response or trace.optimized_response or ""
        candidate_response = trace.optimized_response or reference_response
        quality = evaluate_response(
            reference_response,
            candidate_response,
            expect_json=trace.expect_json,
            json_schema=trace.json_schema,
            baseline_tool_calls=trace.baseline_tool_calls,
            optimized_tool_calls=trace.optimized_tool_calls,
        )

        baseline_response = trace.reference_response or candidate_response
        optimized_response = candidate_response
        scenario = ScenarioResult(
            scenario_name=trace_id,
            category=trace.category,
            description=trace.description,
            provider=trace_provider,
            model=trace_model,
            baseline_latencies=[latency],
            optimized_latencies=[latency],
            baseline_tokens=[token_measurement],
            optimized_tokens=[token_measurement],
            qualities=[quality],
            baseline_response_sample=baseline_response,
            optimized_response_sample=optimized_response,
            transform_breakdown=await _transform_breakdown(trace_messages, trace_model, config),
            baseline_errors=[],
            optimized_errors=[],
            telemetry={**ctx.metrics, "transforms_applied": ctx.transforms_applied},
        )
        results.append(scenario)

    return BenchmarkReport(
        runner_name="trace_replay",
        provider=provider,
        model=model,
        scenarios=results,
        config={
            "iterations": iterations,
            "warmup": warmup,
            "trace_count": len(traces),
            "semantic_cache_enabled": (config or LatticeConfig()).semantic_cache_enabled,
            "transform_cache_arbitrage": (config or LatticeConfig()).transform_cache_arbitrage,
            "provider_stall_detection_enabled": (config or LatticeConfig()).provider_stall_detection_enabled,
            "transform_batching": (config or LatticeConfig()).transform_batching,
            "transform_speculation": (config or LatticeConfig()).transform_speculation,
            "tacc_enabled": (config or LatticeConfig()).tacc_enabled,
            "transform_semantic_compress": (config or LatticeConfig()).transform_semantic_compress,
            "transform_message_dedup": (config or LatticeConfig()).transform_message_dedup,
            "transform_reference_sub": (config or LatticeConfig()).transform_reference_sub,
        },
    )


# =============================================================================
# Feature-isolated replay — Phase 8 requirement
# =============================================================================

async def run_feature_isolated_replay(
    traces: list[ReplayTrace],
    *,
    model: str,
    provider: str,
    iterations: int = 3,
    warmup: int = 1,
) -> dict[str, BenchmarkReport]:
    """Replay traces through multiple isolated feature configurations.

    Configurations (one feature changed at a time from a common baseline):
    - baseline: all features enabled
    - For each feature: feature_off (only that feature disabled)
    - For each feature: feature_on (only that feature enabled from a minimal base)

    Returns a dict mapping config name -> BenchmarkReport.
    """
    # Common baseline with all 9 features ON
    base_all_on = LatticeConfig(
        compression_mode="aggressive",
        semantic_cache_enabled=True,
        transform_cache_arbitrage=True,
        provider_stall_detection_enabled=True,
        transform_batching=True,
        transform_speculation=True,
        tacc_enabled=True,
        transform_semantic_compress=True,
        transform_message_dedup=True,
        transform_reference_sub=True,
        transform_prefix_opt=True,
        transform_tool_filter=True,
        transform_output_cleanup=True,
        transform_format_conversion=True,
        transform_content_profiler=True,
        graceful_degradation=True,
    )

    # Minimal base with all 9 features OFF
    base_all_off = LatticeConfig(
        compression_mode="safe",
        semantic_cache_enabled=False,
        transform_cache_arbitrage=False,
        provider_stall_detection_enabled=False,
        transform_batching=False,
        transform_speculation=False,
        tacc_enabled=False,
        transform_semantic_compress=False,
        transform_message_dedup=False,
        transform_reference_sub=False,
        transform_prefix_opt=False,
        transform_tool_filter=False,
        transform_output_cleanup=False,
        transform_format_conversion=False,
        transform_content_profiler=False,
        graceful_degradation=True,
    )

    configs: dict[str, LatticeConfig] = {
        "baseline": base_all_on,
        "bare": base_all_off,
    }

    for name, field in REPLAY_FEATURE_FLAGS:
        # Off variant: all on except this feature
        off_config = base_all_on.model_copy(update={field: False})
        configs[f"{name}_off"] = off_config

        # On variant: all off except this feature
        on_config = base_all_off.model_copy(update={field: True})
        configs[f"{name}_on"] = on_config

    reports: dict[str, BenchmarkReport] = {}
    for name, config in configs.items():
        report = await run_trace_replay(
            traces,
            model=model,
            provider=provider,
            iterations=iterations,
            warmup=warmup,
            config=config,
        )
        # Override runner name to reflect the config
        report.runner_name = f"trace_replay_{name}"
        reports[name] = report

    return reports


async def _transform_breakdown(
    messages: list[dict[str, Any]], model: str, config: LatticeConfig | None = None
) -> dict[str, dict[str, Any]]:
    """Measure isolated savings for the replay transforms."""
    baseline_tokens = count_message_tokens(messages, model=model)
    breakdown: dict[str, dict[str, Any]] = {}
    cfg = config or LatticeConfig(graceful_degradation=True)
    transforms_to_measure: list[tuple[str, Any]] = [
        ("prefix_opt", PrefixOptimizer()),
        ("reference_sub", ReferenceSubstitution()),
        ("tool_filter", ToolOutputFilter()),
        ("output_cleanup", OutputCleanup()),
        ("format_conv", FormatConverter(validate_roundtrip=False)),
    ]
    if cfg.transform_cache_arbitrage:
        transforms_to_measure.append(("cache_arbitrage", CacheArbitrageOptimizer()))
    if cfg.transform_message_dedup:
        transforms_to_measure.append(("message_dedup", MessageDeduplicator()))
    for name, transform in transforms_to_measure:
        pipeline = CompressorPipeline(config=cfg)
        pipeline.register(transform)
        request = Request(messages=[message_from_dict(m) for m in messages], model=model)
        compressed = unwrap(
            await pipeline.process(
                request,
                TransformContext(model=model, provider="openai"),
            )
        )
        optimized_tokens = count_message_tokens(
            [{"role": str(m.role), "content": m.content} for m in compressed.messages],
            model=model,
        )
        breakdown[name] = {
            "before": baseline_tokens,
            "after": optimized_tokens,
            "saved": max(0, baseline_tokens - optimized_tokens),
        }
    return breakdown


def _render_markdown(report: BenchmarkReport) -> str:
    payload = report.to_dict()
    summary = payload["summary"]
    lines = [
        "# Trace Replay Benchmark",
        "",
        f"- Runner: `{payload['runner']}`",
        f"- Provider: `{payload['provider']}`",
        f"- Model: `{payload['model']}`",
        f"- Total scenarios: `{summary['total_scenarios']}`",
        f"- Passed scenarios: `{summary['passed_scenarios']}`",
        f"- Total token savings: `{summary['total_token_savings']}`",
        f"- Avg reduction ratio: `{summary['avg_reduction_ratio']:.4f}`",
        f"- Avg pipeline latency: `{summary['avg_pipeline_latency_ms']:.4f} ms`",
        f"- Avg quality score: `{summary['avg_quality_score']:.4f}`",
        "",
        "| Scenario | Category | Tokens Before | Tokens After | Saved | Ratio | Quality |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in payload["scenarios"]:
        tokens = scenario["tokens"]["savings"]
        # Task-equivalence is the source of truth; semantic similarity is legacy
        te = scenario["quality"].get("task_equivalence", {})
        quality = te.get("composite", scenario["quality"]["semantic_similarity"])
        lines.append(
            f"| {scenario['scenario']} | {scenario['category']} | "
            f"{tokens['before']} | {tokens['after']} | {tokens['saved']} | "
            f"{tokens['ratio']:.4f} | {quality:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _write_outputs(
    report: BenchmarkReport,
    *,
    output_json: str = "",
    output_md: str = "",
) -> None:
    payload = report.to_dict()
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if output_md:
        path = Path(output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_render_markdown(report), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay captured traces through the LATTICE pipeline")
    parser.add_argument("--input", required=True, help="Path to a JSON or JSONL trace file")
    parser.add_argument("--model", default="gpt-4", help="Model name for token accounting")
    parser.add_argument("--provider", default="openai", help="Provider name for the report")
    parser.add_argument("--iterations", type=int, default=3, help="Latency samples per trace")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations before timing")
    parser.add_argument("--prompts", default="", help="Comma-separated trace IDs or scenario names")
    parser.add_argument("--output-json", default="", help="Optional JSON output path")
    parser.add_argument("--output-md", default="", help="Optional markdown output path")
    parser.add_argument("--json-only", action="store_true", help="Print JSON only")
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    traces = load_traces(args.input)
    if args.prompts:
        traces = select_traces(traces, [name.strip() for name in args.prompts.split(",")])
    report = await run_trace_replay(
        traces,
        model=args.model,
        provider=args.provider,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    _write_outputs(report, output_json=args.output_json, output_md=args.output_md)
    if args.json_only:
        print(json.dumps(report.to_dict(), indent=2))
        return
    print(_render_markdown(report))


if __name__ == "__main__":
    asyncio.run(main())
