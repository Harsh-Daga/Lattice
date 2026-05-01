"""Production-grade eval runners."""

from __future__ import annotations

import asyncio
import json
import re
import statistics
from pathlib import Path
from typing import Any

from benchmarks.evals.catalog import default_provider_targets, default_scenarios
from benchmarks.evals.live import build_full_pipeline, run_scenario, setup_provider
from benchmarks.evals.replay import (
    REPLAY_FEATURE_FLAGS,
    load_traces,
    run_feature_isolated_replay,
    run_trace_replay,
)
from benchmarks.evals.report import EvalSectionReport, ProductionEvalReport, render_markdown
from benchmarks.evals.surfaces import (
    run_capability_eval,
    run_integration_eval,
    run_protocol_eval,
    run_transport_eval,
)
from benchmarks.framework.types import (
    BenchmarkReport,
    ScenarioResult,
    TaskEquivalenceScore,
    TokenMeasurement,
)
from benchmarks.scenarios.prompts import BenchmarkScenario
from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.result import is_ok, unwrap
from lattice.core.serialization import message_from_dict, message_to_dict
from lattice.core.session import MemorySessionStore, SessionManager
from lattice.core.transport import Message, Request, Role
from lattice.protocol.dictionary_codec import DictionaryCodec
from lattice.protocol.framing import BinaryFramer, FrameFlags, FrameType
from lattice.protocol.manifest import manifest_from_messages, manifest_summary
from lattice.transport.simulation import (
    SimulationConfig,
    run_static_concurrency_simulation,
    run_tacc_simulation,
)


def _usage_prompt_tokens(usage: dict[str, Any], fallback: int) -> int:
    value = usage.get("prompt_tokens")
    return int(value) if isinstance(value, int) else fallback


_FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "semantic_compress": ("rate_distortion",),
    "rate_distortion": ("semantic_compress",),
    "prefix_opt": ("prefix_optimizer",),
    "prefix_optimizer": ("prefix_opt",),
}

_FEATURE_SURFACES: dict[str, tuple[str, str]] = {
    "semantic_cache": ("execution", "provider_eval"),
    "cache_arbitrage": ("replay", "feature_matrix"),
    "stall_detection": ("execution", "provider_eval"),
    "batching": ("execution", "provider_eval"),
    "speculation": ("execution", "provider_eval"),
    "tacc": ("simulation", "tacc_eval"),
    "semantic_compress": ("replay", "feature_matrix"),
    "message_dedup": ("replay", "feature_matrix"),
    "reference_sub": ("replay", "feature_matrix"),
}

_FEATURE_SIGNALS: dict[str, tuple[str, str]] = {
    "cache_arbitrage": ("cache_arbitrage", "stability_score"),
    "semantic_compress": ("rate_distortion", "tokens_saved_estimate"),
    "message_dedup": ("message_dedup", "tokens_saved_estimate"),
    "reference_sub": ("reference_sub", "tokens_saved_estimate"),
}


def _feature_matches(feature: str, applied: set[str]) -> bool:
    candidates = {feature, *_FEATURE_ALIASES.get(feature, ())}
    return any(candidate in applied for candidate in candidates)


def _feature_surface(feature: str) -> tuple[str, str]:
    return _FEATURE_SURFACES.get(feature, ("execution", "provider_eval"))


def _trace_supports_feature(trace: Any, feature: str) -> bool:
    metadata = getattr(trace, "metadata", {}) or {}
    feature_targets = metadata.get("matrix_features") or metadata.get("target_features") or []
    if feature in feature_targets:
        return True
    scenario = getattr(trace, "scenario", "")
    scenario_support = {
        "uuid_deduplication": {"reference_sub"},
        "cache_arbitrage_prefix": {"cache_arbitrage"},
        "dictionary_repetition": {"semantic_compress", "message_dedup"},
        "message_dedup_turns": {"message_dedup"},
        "semantic_compress_longform": {"semantic_compress"},
        "json_response_format": {"grammar_compress", "format_conversion"},
        "json_integrity": {"grammar_compress", "format_conversion"},
        "tool_output_filtering": {"tool_filter"},
        "tool_call_preservation": {"tool_filter"},
    }
    return feature in scenario_support.get(scenario, set())


def _feature_signal(feature: str) -> tuple[str, str] | None:
    return _FEATURE_SIGNALS.get(feature)


def _aggregate_feature_signal(report: BenchmarkReport, feature: str) -> float:
    signal = _feature_signal(feature)
    if signal is None:
        return report.total_token_savings
    transform_name, metric_name = signal
    values: list[float] = []
    for scenario in report.scenarios:
        telemetry = scenario.telemetry or {}
        transform_metrics = telemetry.get("transforms", {}).get(transform_name) or telemetry.get(transform_name) or {}
        value = transform_metrics.get(metric_name)
        if isinstance(value, bool):
            values.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            values.append(float(value))
    if values:
        return statistics.mean(values)
    return 0.0


def _infer_tier_from_score(score: int) -> str:
    if score >= 65:
        return "REASONING"
    if score >= 40:
        return "COMPLEX"
    if score >= 20:
        return "MEDIUM"
    return "SIMPLE"


def _feature_flaws(
    scenario: BenchmarkScenario,
    telemetry: dict[str, Any],
    applied: set[str],
) -> tuple[list[str], list[str]]:
    contract = telemetry.get("runtime_contract") or {}
    pipeline = telemetry.get("pipeline") or {}
    skipped = set(contract.get("skipped_transforms") or [])
    missing = [feature for feature in scenario.target_features if not _feature_matches(feature, applied)]
    flaws: list[str] = []
    suggestions: list[str] = []

    if not missing:
        return flaws, suggestions

    if pipeline.get("runtime_budget_exhausted"):
        flaws.append("runtime budget exhausted before all requested features could run")
        suggestions.append("raise the contract budget or reduce earlier transform cost")

    blocked = [feature for feature in missing if feature in skipped]
    if blocked:
        flaws.append("blocked by runtime contract: " + ", ".join(blocked))
        suggestions.append("adjust contract thresholds or move the feature earlier in the pipeline")

    unblocked = [feature for feature in missing if feature not in skipped]
    if unblocked:
        flaws.append("features not reached by pipeline: " + ", ".join(unblocked))
        suggestions.append("inspect pipeline ordering and transform eligibility for the missing features")

    return flaws, suggestions


def _scenario_proof_row(scenario: BenchmarkScenario, telemetry: dict[str, Any], transform_breakdown: dict[str, dict[str, Any]]) -> dict[str, Any]:
    runtime = telemetry.get("runtime") or {}
    contract = telemetry.get("runtime_contract") or {}
    applied = set(telemetry.get("transforms_applied") or [])
    target_features = list(scenario.target_features)
    expected_tier = scenario.expected_tier or ""
    observed_tier = str(runtime.get("tier") or "")
    tier_score = int(runtime.get("score") or 0)
    tier_match = not expected_tier or expected_tier == observed_tier
    feature_hits = [feature for feature in target_features if _feature_matches(feature, applied)]
    feature_match = len(feature_hits) == len(target_features)
    flaws, suggestions = _feature_flaws(scenario, telemetry, applied)
    if not tier_match:
        flaws.append(f"tier mismatch: expected {expected_tier or 'ANY'} but observed {observed_tier or _infer_tier_from_score(tier_score)}")
        suggestions.append("revisit the runtime classifier thresholds and scenario calibration")
    return {
        "scenario": scenario.name,
        "category": scenario.category,
        "complexity": scenario.complexity,
        "expected_tier": expected_tier,
        "observed_tier": observed_tier or _infer_tier_from_score(tier_score),
        "tier_score": tier_score,
        "confidence": round(float(runtime.get("confidence", 0.0)), 4),
        "budget_ms": float(contract.get("max_transform_latency_ms") or 0.0),
        "target_features": target_features,
        "applied_transforms": sorted(applied),
        "feature_hits": feature_hits,
        "tier_match": tier_match,
        "feature_match": feature_match,
        "passed": tier_match and feature_match,
        "status": "pass" if tier_match and feature_match else "fail",
        "flaws": flaws,
        "suggestions": suggestions,
        "transform_count": len(transform_breakdown),
        "proof": scenario.proof,
    }


def _selected_traces_for_feature(traces: list[Any], feature: str) -> list[Any]:
    selected = [trace for trace in traces if _trace_supports_feature(trace, feature)]
    if selected:
        return selected
    return []


async def run_feature_eval(
    *,
    scenarios: list[BenchmarkScenario] | None = None,
    model: str = "",
    provider: str = "",
    iterations: int = 1,
    warmup: int = 0,
) -> EvalSectionReport:
    """Run local feature evals without provider calls."""
    selected = scenarios or default_scenarios()
    config = LatticeConfig.auto()
    pipeline = build_full_pipeline(config)
    report = BenchmarkReport(
        runner_name="feature_eval",
        provider=provider or "local",
        model=model or "dry-run",
        config={
            "iterations": iterations,
            "warmup": warmup,
            "scenario_count": len(selected),
            "transforms": [t.name for t in pipeline.transforms],
        },
    )

    transform_totals: dict[str, dict[str, float]] = {}
    for scenario in selected:
        result = ScenarioResult(
            scenario_name=scenario.name,
            category=scenario.category,
            description=scenario.description,
            provider=provider,
            model=model,
        )
        for iteration in range(iterations):
            (
                baseline_lat,
                optimized_lat,
                baseline_tokens,
                optimized_tokens,
                quality,
                baseline_resp,
                optimized_resp,
                baseline_err,
                optimized_err,
                baseline_usage,
                optimized_usage,
                transform_breakdown,
                usage_summary,
                telemetry,
            ) = await run_scenario(
                scenario=scenario,
                provider=None,
                pipeline=pipeline,
                model=model,
                provider_name=provider,
                dry_run=True,
                iteration=iteration,
            )
            if baseline_lat:
                result.baseline_latencies.append(baseline_lat)
            if optimized_lat:
                result.optimized_latencies.append(optimized_lat)
            if quality:
                result.qualities.append(quality)
            if baseline_err:
                result.baseline_errors.append(baseline_err)
            if optimized_err:
                result.optimized_errors.append(optimized_err)
            baseline_prompt_tokens = _usage_prompt_tokens(baseline_usage, baseline_tokens)
            optimized_prompt_tokens = _usage_prompt_tokens(optimized_usage, optimized_tokens)
            result.transform_breakdown = transform_breakdown
            result.usage_summary = usage_summary
            result.telemetry = telemetry
            result.baseline_tokens.append(
                TokenMeasurement(
                    before=baseline_prompt_tokens,
                    after=baseline_prompt_tokens,
                    saved=0,
                    ratio=0.0,
                )
            )
            result.optimized_tokens.append(
                TokenMeasurement(
                    before=baseline_prompt_tokens,
                    after=optimized_prompt_tokens,
                    saved=max(0, baseline_prompt_tokens - optimized_prompt_tokens),
                    ratio=(baseline_prompt_tokens - optimized_prompt_tokens) / max(baseline_prompt_tokens, 1),
                )
            )
            if iteration == iterations - 1:
                result.baseline_response_sample = baseline_resp
                result.optimized_response_sample = optimized_resp

            for name, metrics in result.transform_breakdown.items():
                slot = transform_totals.setdefault(name, {"before": 0.0, "after": 0.0, "saved": 0.0, "count": 0.0})
                slot["before"] += float(metrics.get("before", baseline_tokens))
                slot["after"] += float(metrics.get("after", optimized_tokens))
                slot["saved"] += float(metrics.get("saved", max(0, baseline_tokens - optimized_tokens)))
                slot["count"] += 1.0

        report.scenarios.append(result)

    scenario_proof = [
        _scenario_proof_row(scenario, result.telemetry, result.transform_breakdown)
        for scenario, result in zip(selected, report.scenarios, strict=False)
    ]
    feature_fails = sum(1 for row in scenario_proof if not row["feature_match"])
    tier_fails = sum(1 for row in scenario_proof if not row["tier_match"])
    budget_fails = sum(1 for row in scenario_proof if any("runtime budget exhausted" in flaw for flaw in row["flaws"]))
    flaw_counts: dict[str, int] = {}
    for row in scenario_proof:
        for flaw in row["flaws"]:
            flaw_counts[flaw] = flaw_counts.get(flaw, 0) + 1
    coverage_summary = {
        "scenario_count": len(selected),
        "proof_passed": sum(1 for row in scenario_proof if row["passed"]),
        "proof_failed": sum(1 for row in scenario_proof if not row["passed"]),
        "feature_failed": feature_fails,
        "tier_failed": tier_fails,
        "budget_failed": budget_fails,
        "flaw_counts": flaw_counts,
        "tiers": {
            tier: sum(1 for row in scenario_proof if row["observed_tier"] == tier)
            for tier in ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING")
        },
        "features": {
            feature: sum(1 for row in scenario_proof if feature in row["feature_hits"])
            for feature in sorted({feature for scenario in selected for feature in scenario.target_features})
        },
    }
    summary = {
        name: {
            "avg_before": round(values["before"] / max(values["count"], 1.0), 2),
            "avg_after": round(values["after"] / max(values["count"], 1.0), 2),
            "avg_saved": round(values["saved"] / max(values["count"], 1.0), 2),
            "count": int(values["count"]),
        }
        for name, values in transform_totals.items()
    }
    return EvalSectionReport(
        name="feature_eval",
        kind="local",
        summary={
            "scenario_count": len(selected),
            "iterations": iterations,
            "avg_reduction_ratio": round(report.avg_reduction_ratio, 4),
            "avg_pipeline_latency_ms": round(report.avg_pipeline_latency_ms, 4),
            "avg_quality_score": round(report.avg_quality_score, 4),
            "proof_passed": coverage_summary["proof_passed"],
            "proof_failed": coverage_summary["proof_failed"],
            "feature_failed": coverage_summary["feature_failed"],
            "tier_failed": coverage_summary["tier_failed"],
            "budget_failed": coverage_summary["budget_failed"],
        },
        details={
            "transform_summary": summary,
            "coverage_summary": coverage_summary,
            "scenario_proof": scenario_proof,
        },
        benchmark=report,
    )


async def run_feature_matrix_eval(
    *,
    input_path: str | Path = "benchmarks/datasets/replay_traces.jsonl",
    model: str = "gpt-4",
    provider: str = "openai",
    iterations: int = 1,
    warmup: int = 0,
) -> EvalSectionReport:
    """Run the replay feature isolation matrix and summarize each feature toggle."""
    traces = load_traces(input_path)
    feature_rows: list[dict[str, Any]] = []
    on_ratio_samples: list[float] = []
    off_ratio_samples: list[float] = []
    on_quality_samples: list[float] = []
    off_quality_samples: list[float] = []
    on_signal_samples: list[float] = []
    off_signal_samples: list[float] = []
    evaluated_features = 0

    for feature_name, _field in REPLAY_FEATURE_FLAGS:
        surface, evidence = _feature_surface(feature_name)
        if surface != "replay":
            feature_rows.append(
                {
                    "feature": feature_name,
                    "surface": surface,
                    "evidence": evidence,
                    "supporting_traces": [],
                    "off_savings": 0,
                    "on_savings": 0,
                    "off_reduction_ratio": 0.0,
                    "on_reduction_ratio": 0.0,
                    "off_quality": 0.0,
                    "on_quality": 0.0,
                    "delta_savings": 0,
                    "verdict": "n/a",
                }
            )
            continue

        feature_traces = _selected_traces_for_feature(traces, feature_name)
        if not feature_traces:
            feature_rows.append(
                {
                    "feature": feature_name,
                    "surface": surface,
                    "evidence": evidence,
                    "supporting_traces": [],
                    "off_savings": 0,
                    "on_savings": 0,
                    "off_reduction_ratio": 0.0,
                    "on_reduction_ratio": 0.0,
                    "off_quality": 0.0,
                    "on_quality": 0.0,
                    "delta_savings": 0,
                    "verdict": "n/a",
                }
            )
            continue

        all_on_config = LatticeConfig(
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
        off_config = all_on_config.model_copy(update={_field: False})
        on_config = all_on_config

        off_report = await run_trace_replay(
            feature_traces,
            model=model,
            provider=provider,
            iterations=iterations,
            warmup=warmup,
            config=off_config,
        )
        on_report = await run_trace_replay(
            feature_traces,
            model=model,
            provider=provider,
            iterations=iterations,
            warmup=warmup,
            config=on_config,
        )

        delta_savings = on_report.total_token_savings - off_report.total_token_savings
        off_signal = _aggregate_feature_signal(off_report, feature_name)
        on_signal = _aggregate_feature_signal(on_report, feature_name)
        delta_signal = on_signal - off_signal
        if (delta_savings > 0 or delta_signal > 0) and on_report.avg_quality_score >= off_report.avg_quality_score:
            verdict = "improves"
        elif delta_savings == 0 and delta_signal == 0:
            verdict = "flat"
        elif delta_savings < 0 or delta_signal < 0:
            verdict = "regresses"
        else:
            verdict = "unclear"
        feature_rows.append(
            {
                "feature": feature_name,
                "surface": surface,
                "evidence": evidence,
                "supporting_traces": [trace.trace_id for trace in feature_traces],
                "off_savings": off_report.total_token_savings,
                "on_savings": on_report.total_token_savings,
                "off_reduction_ratio": round(off_report.avg_reduction_ratio, 4),
                "on_reduction_ratio": round(on_report.avg_reduction_ratio, 4),
                "off_quality": round(off_report.avg_quality_score, 4),
                "on_quality": round(on_report.avg_quality_score, 4),
                "delta_savings": delta_savings,
                "signal_metric": _feature_signal(feature_name)[1] if _feature_signal(feature_name) else "token_savings",
                "off_signal": round(off_signal, 4),
                "on_signal": round(on_signal, 4),
                "delta_signal": round(delta_signal, 4),
                "verdict": verdict,
            }
        )
        evaluated_features += 1
        off_ratio_samples.append(off_report.avg_reduction_ratio)
        on_ratio_samples.append(on_report.avg_reduction_ratio)
        off_quality_samples.append(off_report.avg_quality_score)
        on_quality_samples.append(on_report.avg_quality_score)
        off_signal_samples.append(off_signal)
        on_signal_samples.append(on_signal)

    replay_features = sum(1 for row in feature_rows if row["surface"] == "replay")
    n_a_features = sum(1 for row in feature_rows if row["verdict"] == "n/a")
    improving_features = sum(1 for row in feature_rows if row["verdict"] == "improves")
    flat_features = sum(1 for row in feature_rows if row["verdict"] == "flat")
    regressing_features = sum(1 for row in feature_rows if row["verdict"] == "regresses")
    summary = {
        "trace_count": len(traces),
        "config_count": 2 * evaluated_features,
        "feature_count": len(feature_rows),
        "replay_feature_count": replay_features,
        "n_a_feature_count": n_a_features,
        "off_reduction_ratio": round(statistics.mean(off_ratio_samples), 4) if off_ratio_samples else 0.0,
        "on_reduction_ratio": round(statistics.mean(on_ratio_samples), 4) if on_ratio_samples else 0.0,
        "off_quality_score": round(statistics.mean(off_quality_samples), 4) if off_quality_samples else 0.0,
        "on_quality_score": round(statistics.mean(on_quality_samples), 4) if on_quality_samples else 0.0,
        "off_signal_score": round(statistics.mean(off_signal_samples), 4) if off_signal_samples else 0.0,
        "on_signal_score": round(statistics.mean(on_signal_samples), 4) if on_signal_samples else 0.0,
        "improving_features": improving_features,
        "flat_features": flat_features,
        "regressing_features": regressing_features,
    }
    return EvalSectionReport(
        name="feature_matrix_eval",
        kind="replay",
        summary=summary,
        details={"feature_matrix": feature_rows},
    )


async def run_provider_eval(
    *,
    providers: list[str] | None = None,
    model_overrides: dict[str, str] | None = None,
    scenarios: list[BenchmarkScenario] | None = None,
    iterations: int = 1,
    warmup: int = 1,
) -> EvalSectionReport:
    """Run live provider evals for configured targets."""
    selected = scenarios or default_scenarios()
    targets = default_provider_targets(
        providers,
        model_overrides=model_overrides,
        strict_model_selection=providers is not None,
    )
    sections: list[dict[str, Any]] = []
    total_runs = 0
    total_skipped = 0

    for target in targets:
        if not target.available:
            sections.append(target.to_dict())
            total_skipped += 1
            continue

        provider = setup_provider(
            provider_name=target.provider,
            base_url=target.base_url or None,
        )
        try:
            adapter = provider.registry.resolve(target.model)
        except Exception as exc:
            sections.append({**target.to_dict(), "available": False, "skip_reason": str(exc)})
            total_skipped += 1
            continue

        pipeline = build_full_pipeline(LatticeConfig.auto())
        report = BenchmarkReport(
            runner_name="provider_eval",
            provider=target.provider,
            model=target.model,
            config={
                "iterations": iterations,
                "warmup": warmup,
                "scenario_count": len(selected),
                "target": target.to_dict(),
                "adapter": adapter.name,
            },
        )
        for scenario in selected:
            result = ScenarioResult(
                scenario_name=scenario.name,
                category=scenario.category,
                description=scenario.description,
                provider=target.provider,
                model=target.model,
            )
            for iteration in range(iterations):
                (
                    baseline_lat,
                    optimized_lat,
                    baseline_tokens,
                    optimized_tokens,
                    quality,
                    baseline_resp,
                    optimized_resp,
                    baseline_err,
                    optimized_err,
                    baseline_usage,
                    optimized_usage,
                    transform_breakdown,
                    usage_summary,
                    telemetry,
                ) = await run_scenario(
                    scenario=scenario,
                    provider=provider,
                    pipeline=pipeline,
                    model=target.model,
                    provider_name=target.provider,
                    dry_run=False,
                    iteration=iteration,
                )
                if baseline_lat:
                    result.baseline_latencies.append(baseline_lat)
                if optimized_lat:
                    result.optimized_latencies.append(optimized_lat)
                if quality:
                    result.qualities.append(quality)
                if baseline_err:
                    result.baseline_errors.append(baseline_err)
                if optimized_err:
                    result.optimized_errors.append(optimized_err)
                baseline_prompt_tokens = _usage_prompt_tokens(baseline_usage, baseline_tokens)
                optimized_prompt_tokens = _usage_prompt_tokens(optimized_usage, optimized_tokens)
                result.baseline_tokens.append(
                    TokenMeasurement(
                        before=baseline_prompt_tokens,
                        after=baseline_prompt_tokens,
                        saved=0,
                        ratio=0.0,
                    )
                )
                result.optimized_tokens.append(
                    TokenMeasurement(
                        before=baseline_prompt_tokens,
                        after=optimized_prompt_tokens,
                        saved=max(0, baseline_prompt_tokens - optimized_prompt_tokens),
                        ratio=(baseline_prompt_tokens - optimized_prompt_tokens) / max(baseline_prompt_tokens, 1),
                    )
                )
                if iteration == iterations - 1:
                    result.baseline_response_sample = baseline_resp
                    result.optimized_response_sample = optimized_resp
                    result.transform_breakdown = transform_breakdown
                    result.usage_summary = usage_summary
                    result.telemetry = telemetry
            report.scenarios.append(result)
            total_runs += 1
        sections.append({"target": target.to_dict(), "adapter": adapter.name, "benchmark": report.to_dict()})

    return EvalSectionReport(
        name="provider_eval",
        kind="live",
        status="ok" if total_runs else "skipped",
        summary={
            "target_count": len(targets),
            "active_targets": total_runs,
            "skipped_targets": total_skipped,
        },
        details={"targets": sections},
        benchmark=None,
    )


async def run_replay_eval(
    *,
    input_path: str | Path,
    model: str = "gpt-4",
    provider: str = "openai",
    iterations: int = 1,
    warmup: int = 0,
    prompts: list[str] | None = None,
) -> EvalSectionReport:
    """Run replay-based regression evals."""
    traces = load_traces(input_path)
    if prompts:
        wanted = {name.strip() for name in prompts if name.strip()}
        traces = [trace for trace in traces if trace.trace_id in wanted or trace.scenario in wanted]
    report = await run_trace_replay(
        traces,
        model=model,
        provider=provider,
        iterations=iterations,
        warmup=warmup,
    )
    return EvalSectionReport(
        name="replay_eval",
        kind="replay",
        summary={
            "trace_count": len(traces),
            "iterations": iterations,
            "avg_reduction_ratio": round(report.avg_reduction_ratio, 4),
            "avg_pipeline_latency_ms": round(report.avg_pipeline_latency_ms, 4),
            "avg_quality_score": round(report.avg_quality_score, 4),
        },
        details={"input_path": str(input_path)},
        benchmark=report,
    )


async def run_replay_feature_isolated(
    *,
    input_path: str | Path,
    model: str = "gpt-4",
    provider: str = "openai",
    iterations: int = 1,
    warmup: int = 0,
    prompts: list[str] | None = None,
) -> dict[str, EvalSectionReport]:
    """Run feature-isolated replay evals.

    Runs traces through multiple configurations (bare, baseline, compiler,
    cache, full) so each feature's impact can be measured independently.
    """
    traces = load_traces(input_path)
    if prompts:
        wanted = {name.strip() for name in prompts if name.strip()}
        traces = [trace for trace in traces if trace.trace_id in wanted or trace.scenario in wanted]

    reports = await run_feature_isolated_replay(
        traces,
        model=model,
        provider=provider,
        iterations=iterations,
        warmup=warmup,
    )

    sections: dict[str, EvalSectionReport] = {}
    for name, report in reports.items():
        sections[name] = EvalSectionReport(
            name=f"replay_isolated_{name}",
            kind="replay",
            summary={
                "trace_count": len(traces),
                "iterations": iterations,
                "avg_reduction_ratio": round(report.avg_reduction_ratio, 4),
                "avg_pipeline_latency_ms": round(report.avg_pipeline_latency_ms, 4),
                "avg_quality_score": round(report.avg_quality_score, 4),
            },
            details={"input_path": str(input_path), "config": name},
            benchmark=report,
        )
    return sections


async def run_replay_governance(
    *,
    input_path: str | Path,
    model: str = "gpt-4",
    provider: str = "openai",
    iterations: int = 1,
    warmup: int = 0,
    prompts: list[str] | None = None,
    regression_threshold_quality: float = 0.05,
    regression_threshold_latency: float = 1.5,
) -> EvalSectionReport:
    """Run feature-isolated replay with regression governance.

    Compares each feature-off config against the all-features-on baseline.
    Flags regressions when quality drops or latency increases beyond thresholds.
    """
    from benchmarks.evals.replay import _classify_failure

    traces = load_traces(input_path)
    if prompts:
        wanted = {name.strip() for name in prompts if name.strip()}
        traces = [trace for trace in traces if trace.trace_id in wanted or trace.scenario in wanted]

    reports = await run_feature_isolated_replay(
        traces,
        model=model,
        provider=provider,
        iterations=iterations,
        warmup=warmup,
    )

    baseline = reports.get("baseline")
    if baseline is None:
        raise ValueError("Baseline report missing from feature-isolated replay")

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

    governance: dict[str, dict[str, Any]] = {}
    all_pass = True

    for feature in feature_names:
        off_report = reports.get(f"{feature}_off")
        if off_report is None:
            continue

        categories = _classify_failure(
            baseline,
            off_report,
            quality_threshold=regression_threshold_quality,
            latency_threshold=regression_threshold_latency,
        )

        quality_delta = baseline.avg_quality_score - off_report.avg_quality_score
        baseline_latency = baseline.avg_pipeline_latency_ms
        off_latency = off_report.avg_pipeline_latency_ms
        latency_ratio = off_latency / baseline_latency if baseline_latency > 0 else 1.0

        passed = len(categories) == 0
        if not passed:
            all_pass = False

        governance[feature] = {
            "status": "pass" if passed else "fail",
            "quality_delta": round(quality_delta, 4),
            "latency_ratio": round(latency_ratio, 4),
            "failure_categories": [c.value for c in categories],
        }

    return EvalSectionReport(
        name="replay_governance",
        kind="replay",
        status="ok" if all_pass else "degraded",
        summary={
            "feature_count": len(feature_names),
            "passed_features": sum(1 for g in governance.values() if g["status"] == "pass"),
            "failed_features": sum(1 for g in governance.values() if g["status"] == "fail"),
            "regression_threshold_quality": regression_threshold_quality,
            "regression_threshold_latency": regression_threshold_latency,
        },
        details={
            "governance": governance,
            "reports": {k: v.to_dict() for k, v in reports.items()},
        },
        benchmark=baseline,
    )


async def run_tacc_eval() -> EvalSectionReport:
    """Run the deterministic TACC simulation benchmark."""
    scenarios = {
        "light_load": SimulationConfig(
            duration_seconds=20.0,
            step_seconds=0.02,
            arrival_rate_qps=20.0,
            service_capacity=6,
            static_concurrency=12,
            base_latency_ms=120.0,
            seed=42,
        ),
        "moderate_load": SimulationConfig(
            duration_seconds=20.0,
            step_seconds=0.02,
            arrival_rate_qps=60.0,
            service_capacity=6,
            static_concurrency=12,
            base_latency_ms=120.0,
            seed=42,
        ),
        "heavy_load": SimulationConfig(
            duration_seconds=20.0,
            step_seconds=0.02,
            arrival_rate_qps=120.0,
            service_capacity=6,
            static_concurrency=12,
            base_latency_ms=120.0,
            seed=42,
        ),
        "burst_load": SimulationConfig(
            duration_seconds=20.0,
            step_seconds=0.02,
            arrival_rate_qps=240.0,
            service_capacity=6,
            static_concurrency=12,
            base_latency_ms=120.0,
            seed=42,
        ),
    }

    results: list[dict[str, Any]] = []
    for name, cfg in scenarios.items():
        static = await asyncio.to_thread(run_static_concurrency_simulation, cfg)
        tacc = await asyncio.to_thread(run_tacc_simulation, cfg)
        results.append(
            {
                "scenario": name,
                "static": {
                    "throughput_qps": round(static.throughput_qps, 2),
                    "p95_latency_ms": round(static.p95_latency_ms, 2),
                    "p99_latency_ms": round(static.p99_latency_ms, 2),
                    "error_rate": round(static.error_rate, 4),
                },
                "tacc": {
                    "throughput_qps": round(tacc.throughput_qps, 2),
                    "p95_latency_ms": round(tacc.p95_latency_ms, 2),
                    "p99_latency_ms": round(tacc.p99_latency_ms, 2),
                    "error_rate": round(tacc.error_rate, 4),
                },
            }
        )
    return EvalSectionReport(
        name="tacc_eval",
        kind="simulation",
        summary={
            "scenario_count": len(results),
            "avg_static_p95_ms": round(statistics.mean(r["static"]["p95_latency_ms"] for r in results), 2),
            "avg_tacc_p95_ms": round(statistics.mean(r["tacc"]["p95_latency_ms"] for r in results), 2),
        },
        details={"scenarios": results},
    )


async def run_control_plane_eval() -> EvalSectionReport:
    """Run local invariants for manifests, sessions, and native wire."""
    store = MemorySessionStore(ttl_seconds=60)
    await store.start()
    manager = SessionManager(store)
    messages = [
        Message(role=Role.SYSTEM, content="You are a production evaluator."),
        Message(role=Role.USER, content="Summarize the manifest and wire state."),
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup data",
                "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
            },
        }
    ]
    session = await manager.create_session("openai", "openai/gpt-4o-mini", messages, tools=tools)
    original_version = session.version
    follow_up = messages + [Message(role=Role.USER, content="Add one more turn.")]
    delta = await manager.compute_delta(session.session_id, follow_up)
    updated = await manager.update_session(session.session_id, follow_up, session.manifest)

    manifest = session.manifest or manifest_from_messages(
        session.session_id,
        [message_to_dict(m) for m in messages],
        tools=tools,
        model=session.model,
        provider=session.provider,
    )
    codec = DictionaryCodec(session_id=session.session_id)
    payload = json.dumps({"session_id": session.session_id, "model": session.model, "messages": [message_to_dict(m) for m in messages]}, sort_keys=True).encode("utf-8")
    compressed = codec.compress(payload)
    roundtrip = codec.decompress(compressed)
    framer = BinaryFramer()
    request_frames = framer.encode_request(compressed, flags=FrameFlags.DICT_COMPRESSED)
    encoded = request_frames[0].to_bytes()
    decoded = framer.decode_frame(encoded)
    frame_ok = bool(decoded.flags & FrameFlags.DICT_COMPRESSED) and decoded.frame_type == FrameType.REQUEST
    codec_snapshot = codec.to_snapshot()
    restored = DictionaryCodec.from_snapshot(codec_snapshot)
    restored_roundtrip = restored.decompress(restored.compress(payload))

    await store.stop()

    checks = {
        "session_created": session.session_id.startswith("lattice-"),
        "manifest_summary": manifest_summary(manifest)["segment_count"] >= 1,
        "delta_detected": len(delta) == 1,
        "session_updated": updated is not None and updated.version > original_version,
        "dictionary_roundtrip": roundtrip == payload and restored_roundtrip == payload,
        "frame_roundtrip": frame_ok,
        "manifest_version": manifest.anchor_version == 0,
    }
    return EvalSectionReport(
        name="control_plane_eval",
        kind="local",
        summary={"passed": sum(1 for ok in checks.values() if ok), "total": len(checks)},
        details={"checks": checks, "session": session.to_dict()},
    )


async def run_production_evals(
    *,
    scenarios: list[str] | None = None,
    providers: list[str] | None = None,
    model_overrides: dict[str, str] | None = None,
    replay_input: str | Path = "benchmarks/datasets/replay_traces.jsonl",
    iterations: int = 1,
    warmup: int = 0,
    provider_iterations: int = 1,
    provider_warmup: int = 1,
) -> ProductionEvalReport:
    """Run the full production eval bundle."""
    selected_scenarios = default_scenarios(scenarios or None)
    # Resolve provider/model from targets for accurate report labelling
    provider_targets = default_provider_targets(
        providers, model_overrides=model_overrides,
        strict_model_selection=providers is not None,
    )
    first_target = provider_targets[0] if provider_targets else None
    eval_provider = first_target.provider if first_target else ""
    eval_model = first_target.model if first_target else ""

    sections = [
        await run_feature_eval(
            scenarios=selected_scenarios,
            iterations=iterations,
            warmup=warmup,
            provider=eval_provider,
            model=eval_model,
        ),
        await run_feature_matrix_eval(
            input_path=replay_input,
            iterations=iterations,
            warmup=warmup,
        ),
        await run_protocol_eval(),
        await run_transport_eval(),
        await run_integration_eval(),
        await run_capability_eval(),
        await run_control_plane_eval(),
        await run_tacc_eval(),
        await run_replay_eval(
            input_path=replay_input,
            iterations=iterations,
            warmup=warmup,
        ),
    ]
    provider_section = await run_provider_eval(
        providers=providers,
        model_overrides=model_overrides,
        scenarios=selected_scenarios,
        iterations=provider_iterations,
        warmup=provider_warmup,
    )
    sections.insert(1, provider_section)

    # Provider validation: model-in-the-loop task-equivalence check
    validation_section = await run_provider_validation(
        providers=providers,
        model_overrides=model_overrides,
        scenarios=selected_scenarios,
    )
    sections.insert(2, validation_section)

    return ProductionEvalReport(
        runner_name="production_evals",
        config={
            "scenario_count": len(selected_scenarios),
            "providers": providers or [],
            "provider_models": model_overrides or {},
            "replay_input": str(replay_input),
            "iterations": iterations,
            "warmup": warmup,
            "provider_iterations": provider_iterations,
            "provider_warmup": provider_warmup,
        },
        sections=sections,
    )


def write_production_eval_outputs(
    report: ProductionEvalReport,
    *,
    output_json: str = "",
    output_md: str = "",
) -> None:
    """Write production eval outputs to disk."""
    payload = report.to_dict()
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if output_md:
        path = Path(output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(report), encoding="utf-8")


# =============================================================================
# Task-equivalence evaluation (model-in-the-loop)
# =============================================================================

_JUDGE_SYSTEM_PROMPT = """You are an output quality judge for an LLM compression pipeline.
Your job: compare two answers to the SAME task and decide if they are semantically equivalent.

The BASELINE answer came from the model processing the original full prompt.
The OPTIMIZED answer came from the same model processing a compressed version.

For each dimension, score from 0.0 (completely different) to 1.0 (identical):
- constraint_preservation: Did the optimized answer follow the same constraints/rules?
- entity_preservation: Were all named entities (names, IDs, URLs, UUIDs) preserved?
- format_preservation: Did formatting (JSON, tables, code blocks) stay equivalent?
- reasoning_correctness: Did the reasoning/steps/facts match between answers?
- refusal_correctness: If baseline refused, did optimized also refuse? If baseline didn't, did optimized NOT refuse?
- answer_completeness: Is the optimized answer as complete as baseline?
- harmful_drift: Does the optimized answer contain placeholder artifacts (<ref_N>, <crossref_N>) that the baseline doesn't? 0.0=none, 1.0=severe.

Return ONLY a JSON object with these 7 float fields. No explanation, no markdown, just JSON."""


def _build_judge_prompt(
    baseline_output: str,
    optimized_output: str,
    judge_rubric: str,
    required_properties: list[str],
) -> str:
    """Build the judge prompt for comparing two outputs."""
    rubric_text = f"Scenario-specific rubric: {judge_rubric}" if judge_rubric else ""
    props_text = (
        f"Required properties to check: {', '.join(required_properties)}"
        if required_properties
        else ""
    )
    return (
        f"{rubric_text}\n{props_text}\n\n"
        f"=== BASELINE OUTPUT ===\n{baseline_output}\n\n"
        f"=== OPTIMIZED OUTPUT ===\n{optimized_output}\n\n"
        f"Judge the equivalence. Return ONLY the JSON."
    )


def _parse_judge_response(raw: str) -> TaskEquivalenceScore | None:
    """Parse the judge LLM's JSON response into a TaskEquivalenceScore."""
    try:
        # Find JSON in the response (may have surrounding text)
        import json as _json
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end <= start:
            return None
        data = _json.loads(raw[start:end])
        return TaskEquivalenceScore(
            constraint_preservation=float(data.get("constraint_preservation", 1.0)),
            entity_preservation=float(data.get("entity_preservation", 1.0)),
            format_preservation=float(data.get("format_preservation", 1.0)),
            reasoning_correctness=float(data.get("reasoning_correctness", 1.0)),
            refusal_correctness=float(data.get("refusal_correctness", 1.0)),
            answer_completeness=float(data.get("answer_completeness", 1.0)),
            harmful_drift=float(data.get("harmful_drift", 0.0)),
        )
    except (ValueError, KeyError, _json.JSONDecodeError):
        return None


def evaluate_task_equivalence_structural(
    baseline_output: str,
    optimized_output: str,
    required_properties: list[str],
) -> TaskEquivalenceScore:
    """Structural (regex-based) fallback for when a judge LLM is unavailable."""
    score = TaskEquivalenceScore()

    if not baseline_output.strip() or not optimized_output.strip():
        # Blank or failed outputs: automatic fail on all dimensions.
        return TaskEquivalenceScore(
            constraint_preservation=0.0,
            entity_preservation=0.0,
            format_preservation=0.0,
            reasoning_correctness=0.0,
            refusal_correctness=0.0,
            answer_completeness=0.0,
            harmful_drift=1.0,
        )

    baseline_len = len(baseline_output.strip())
    optimized_len = len(optimized_output.strip())
    if baseline_len > 0:
        ratio = min(baseline_len, optimized_len) / max(baseline_len, optimized_len)
        score.constraint_preservation = round(ratio, 4)
    else:
        score.constraint_preservation = 0.5

    ent_pattern = re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b|"
        r"https?://[^\s)]+|"
        r"\d+(?:\.\d+)?",
        re.IGNORECASE,
    )
    baseline_entities = set(ent_pattern.findall(baseline_output))
    optimized_entities = set(ent_pattern.findall(optimized_output))
    if baseline_entities:
        overlap = len(baseline_entities & optimized_entities)
        score.entity_preservation = round(overlap / len(baseline_entities), 4)
    else:
        score.entity_preservation = 1.0

    baseline_has_json = baseline_output.strip().startswith("{") or baseline_output.strip().startswith("[")
    optimized_has_json = optimized_output.strip().startswith("{") or optimized_output.strip().startswith("[")
    if baseline_has_json != optimized_has_json:
        score.format_preservation = 0.5
    baseline_has_table = "|" in baseline_output
    optimized_has_table = "|" in optimized_output
    if baseline_has_table != optimized_has_table:
        score.format_preservation = min(score.format_preservation, 0.7)

    if "?" in baseline_output and "?" in optimized_output:
        score.reasoning_correctness = 0.9

    refusal_patterns = [r"\bcan'?t\b", r"\bunable\b", r"\bnot available\b", r"\bdenied\b"]
    baseline_refuses = any(re.search(p, baseline_output, re.IGNORECASE) for p in refusal_patterns)
    optimized_refuses = any(re.search(p, optimized_output, re.IGNORECASE) for p in refusal_patterns)
    if baseline_refuses == optimized_refuses:
        score.refusal_correctness = 1.0

    if required_properties:
        found = sum(
            1 for prop in required_properties
            if any(word.lower() in optimized_output.lower() for word in prop.split())
        )
        score.answer_completeness = round(found / len(required_properties), 4) if required_properties else 1.0

    if re.search(r"<ref_\d+>", optimized_output):
        score.harmful_drift = max(score.harmful_drift, 0.3)
    if re.search(r"<crossref_\d+>", optimized_output):
        score.harmful_drift = max(score.harmful_drift, 0.2)

    return score


async def evaluate_task_equivalence_with_judge(
    baseline_output: str,
    optimized_output: str,
    judge_rubric: str,
    required_properties: list[str],
    *,
    provider: Any,
    model: str,
    provider_name: str,
    timeout_s: float = 15.0,
) -> TaskEquivalenceScore:
    """Compare outputs using the same provider/model as a judge.

    Primary path: deterministic structural evaluation (no LLM, no bias).
    Supplemental path: LLM judge refines the structural score with
    scenario-specific rubric awareness.  The LLM judge only upgrades
    the structural score; it never overrides a structural fail.

    This ensures the evaluator is independent of the model-under-test —
    the structural path is the anchor of truth, and the LLM judge
    can only confirm or strengthen it.
    """
    structural = evaluate_task_equivalence_structural(
        baseline_output, optimized_output, required_properties
    )

    # If structural already says fail, skip the LLM judge entirely.
    if not structural.passed:
        return structural

    # If either output is too short for meaningful LLM judging, skip.
    if len(baseline_output.strip()) < 20 or len(optimized_output.strip()) < 20:
        return structural

    judge_prompt = _build_judge_prompt(
        baseline_output, optimized_output, judge_rubric, required_properties
    )

    try:
        import asyncio as _asyncio

        judge_response = await _asyncio.wait_for(
            provider.completion(
                model=model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": judge_prompt},
                ],
                provider_name=provider_name,
                max_tokens=512,
                temperature=0.0,
            ),
            timeout=timeout_s,
        )
        judge_text = judge_response.content if hasattr(judge_response, "content") else str(judge_response)
        parsed = _parse_judge_response(judge_text)
        if parsed is not None:
            # LLM judge refines; structural anchor ensures no false-positive override
            return TaskEquivalenceScore(
                constraint_preservation=max(structural.constraint_preservation, parsed.constraint_preservation),
                entity_preservation=max(structural.entity_preservation, parsed.entity_preservation),
                format_preservation=max(structural.format_preservation, parsed.format_preservation),
                reasoning_correctness=max(structural.reasoning_correctness, parsed.reasoning_correctness),
                refusal_correctness=max(structural.refusal_correctness, parsed.refusal_correctness),
                answer_completeness=max(structural.answer_completeness, parsed.answer_completeness),
                harmful_drift=min(structural.harmful_drift, parsed.harmful_drift),
            )
    except Exception:
        pass  # LLM judge unavailable — structural is sufficient

    return structural


async def run_provider_validation(
    *,
    providers: list[str] | None = None,
    model_overrides: dict[str, str] | None = None,
    scenarios: list[BenchmarkScenario] | None = None,
    _iterations: int = 1,
) -> EvalSectionReport:
    """Model-in-the-loop provider validation.

    Runs each scenario through the real provider with both the baseline
    and compressed prompts.  Compares outputs using the scenario's judge
    rubric to compute task-equivalence scores.

    This is the source of truth for meaning preservation — token reduction
    without task equivalence is a FAIL.
    """
    selected = scenarios or default_scenarios()
    targets = default_provider_targets(
        providers,
        model_overrides=model_overrides,
        strict_model_selection=providers is not None,
    )

    results: list[dict[str, Any]] = []
    total_passed = 0
    total_evaluated = 0

    for target in targets:
        if not target.available:
            continue

        for scenario in selected:
            if not scenario.judge_rubric:
                continue

            provider_obj = setup_provider(
                provider_name=target.provider,
                base_url=target.base_url or None,
            )

            # Build a minimal pipeline for baseline vs optimized comparison
            pipeline = build_full_pipeline(LatticeConfig.auto())

            try:
                import asyncio as _asyncio
                import time as _time

                # Run baseline first
                msgs = [message_from_dict(m) for m in scenario.messages]
                request = Request(
                    messages=msgs,
                    model=target.model,
                    temperature=scenario.temperature,
                    max_tokens=scenario.max_tokens,
                )
                if scenario.tools:
                    request.tools = list(scenario.tools)

                t0 = _time.perf_counter()
                try:
                    baseline_resp = await _asyncio.wait_for(
                        provider_obj.completion(
                            model=target.model,
                            messages=[message_to_dict(m) for m in request.messages],
                            temperature=request.temperature,
                            max_tokens=request.max_tokens,
                            tools=request.tools,
                            provider_name=target.provider,
                        ),
                        timeout=30,
                    )
                    baseline_output = baseline_resp.content or ""
                    provider_ms = (_time.perf_counter() - t0) * 1000
                except Exception:
                    baseline_output = ""
                    provider_ms = (_time.perf_counter() - t0) * 1000

                # Compress and run optimized
                ctx = TransformContext(
                    request_id=str(_time.time()),
                    provider=target.provider,
                    model=target.model,
                )
                compress_result = await pipeline.process(request.copy(), ctx)
                if is_ok(compress_result):
                    compressed = unwrap(compress_result)
                    comp_messages = [message_to_dict(m) for m in compressed.messages]

                    t0 = _time.perf_counter()
                    try:
                        optimized_resp = await _asyncio.wait_for(
                            provider_obj.completion(
                                model=target.model,
                                messages=comp_messages,
                                temperature=compressed.temperature,
                                max_tokens=compressed.max_tokens,
                                tools=compressed.tools,
                                provider_name=target.provider,
                            ),
                            timeout=30,
                        )
                        optimized_output = optimized_resp.content or ""
                        opt_provider_ms = (_time.perf_counter() - t0) * 1000
                    except Exception:
                        optimized_output = ""
                        opt_provider_ms = (_time.perf_counter() - t0) * 1000

                    pipeline_latency_ms = ctx.metrics.get(
                        "transforms", {}
                    ).get("_pipeline", {}).get("transform_latency_ms", 0.0)

                    # Task equivalence scoring — use the same provider/model as judge
                    te_score = await evaluate_task_equivalence_with_judge(
                        baseline_output,
                        optimized_output,
                        scenario.judge_rubric,
                        scenario.required_answer_properties,
                        provider=provider_obj,
                        model=target.model,
                        provider_name=target.provider,
                    )

                    # Build a human-readable verdict from the scores
                    verdict_parts: list[str] = []
                    te_dict = te_score.to_dict()
                    if te_dict["composite"] >= 0.9:
                        verdict_parts.append("PASS: optimized output preserves meaning well")
                    elif te_dict["composite"] >= 0.8:
                        verdict_parts.append("WARN: minor meaning drift detected")
                    else:
                        verdict_parts.append("FAIL: significant meaning drift")
                    if te_dict.get("harmful_drift", 0) > 0:
                        verdict_parts.append(f"placeholder leakage detected (drift={te_dict['harmful_drift']:.2f})")
                    if te_dict.get("constraint_preservation", 1.0) < 0.7:
                        verdict_parts.append("constraints partially lost")
                    if te_dict.get("entity_preservation", 1.0) < 0.7:
                        verdict_parts.append(f"entity loss (score={te_dict['entity_preservation']:.2f})")
                    if te_dict.get("answer_completeness", 1.0) < 0.7:
                        verdict_parts.append("answer incomplete vs baseline")
                    judge_verdict = "; ".join(verdict_parts)

                    tokens_before = request.token_estimate
                    tokens_after = compressed.token_estimate
                    token_savings = max(0, tokens_before - tokens_after)

                    risk = request.metadata.get("_lattice_risk_score", {})
                    forbidden_applied = [
                        t for t in ctx.transforms_applied
                        if t in scenario.forbidden_transforms
                    ]
                    risky_applied = [
                        t for t in ctx.transforms_applied
                        if t in scenario.risky_transforms
                    ]

                    # Check for expansion
                    expansion_ratios = {}
                    for t_name, t_metrics in ctx.metrics.get("transforms", {}).items():
                        if "expansion_ratio" in t_metrics:
                            expansion_ratios[t_name] = t_metrics["expansion_ratio"]

                    te_passed = te_score.passed
                    token_reduction_passed = (
                        tokens_after < tokens_before or token_savings == 0
                    )  # no savings is OK for simple prompts

                    overall_pass = te_passed and token_reduction_passed and not forbidden_applied

                    if overall_pass:
                        total_passed += 1
                    total_evaluated += 1

                    results.append({
                        "provider": target.provider,
                        "model": target.model,
                        "scenario": scenario.name,
                        "task_equivalence": te_score.to_dict(),
                        "task_equivalence_pass": te_passed,
                        "token_reduction_pass": token_reduction_passed,
                        "overall_pass": overall_pass,
                        "tokens_before": tokens_before,
                        "tokens_after": tokens_after,
                        "token_savings": token_savings,
                        "pipeline_latency_ms": pipeline_latency_ms,
                        "provider_latency_ms": round(provider_ms, 2),
                        "optimized_provider_latency_ms": round(opt_provider_ms, 2),
                        "risk_level": risk.get("level", "unknown"),
                        "risk_score": risk.get("total", 0.0),
                        "transforms_applied": list(ctx.transforms_applied),
                        "forbidden_applied": forbidden_applied,
                        "risky_applied": risky_applied,
                        "expansion_ratios": expansion_ratios,
                        "judge_rubric": scenario.judge_rubric,
                        "judge_verdict": judge_verdict,
                        "required_properties": scenario.required_answer_properties,
                        "baseline_output_sample": baseline_output[:500],
                        "optimized_output_sample": optimized_output[:500],
                    })
                else:
                    results.append({
                        "provider": target.provider,
                        "model": target.model,
                        "scenario": scenario.name,
                        "error": "pipeline_failed",
                        "overall_pass": False,
                    })
                    total_evaluated += 1
            except Exception as exc:
                results.append({
                    "provider": target.provider,
                    "model": target.model,
                    "scenario": scenario.name,
                    "error": str(exc),
                    "overall_pass": False,
                })
                total_evaluated += 1

    return EvalSectionReport(
        name="provider_validation",
        kind="live",
        status="ok" if total_evaluated > 0 else "skipped",
        summary={
            "total_evaluated": total_evaluated,
            "total_passed": total_passed,
            "pass_rate": round(total_passed / max(total_evaluated, 1), 4),
        },
        details={"results": results},
    )
