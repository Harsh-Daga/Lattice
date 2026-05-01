"""Benchmark framework data types.

Standardized data models for benchmark results across all runners.
"""

from __future__ import annotations

import dataclasses
import statistics
import time
from typing import Any


@dataclasses.dataclass(slots=True)
class LatencyMeasurement:
    """A single latency observation with phase breakdown."""

    pipeline_ms: float = 0.0       # Transform pipeline wall time
    provider_ms: float = 0.0       # Time spent waiting on the LLM provider
    network_ms: float = 0.0        # Network round-trip time
    queueing_ms: float = 0.0       # Time spent in batching/speculation queues
    retry_ms: float = 0.0          # Cumulative retry/fallback time
    total_ms: float = 0.0          # Total end-to-end wall time
    time_to_first_byte_ms: float | None = None
    timestamp: float = dataclasses.field(default_factory=time.time)


@dataclasses.dataclass(slots=True)
class TokenMeasurement:
    """Token counts for a single run."""

    before: int = 0
    after: int = 0
    saved: int = 0
    ratio: float = 0.0


@dataclasses.dataclass(slots=True)
class TaskEquivalenceScore:
    """Per-scenario rubric scoring for task-equivalence evaluation.

    Each field is 0.0–1.0 where 1.0 = perfect equivalence.
    Composite is task-aware: reasoning/debugging tasks weight correctness
    and key facts higher; structured tasks weight schema and numerics higher.
    """

    correctness: float = 1.0
    completeness: float = 1.0
    reasoning_equivalence: float = 1.0
    key_fact_preservation: float = 1.0
    numeric_preservation: float = 1.0
    refusal_correctness: float | None = None
    schema_validity: float = 1.0
    harmful_drift: float = 0.0
    failure_reasons: list[str] = dataclasses.field(default_factory=list)

    # Legacy fields kept for backward compatibility
    constraint_preservation: float = 1.0
    entity_preservation: float = 1.0
    format_preservation: float = 1.0
    reasoning_correctness: float = 1.0
    answer_completeness: float = 1.0

    @property
    def composite(self) -> float:
        """Task-agnostic composite (unweighted average of all active dims)."""
        dims = [
            self.correctness,
            self.completeness,
            self.reasoning_equivalence,
            self.key_fact_preservation,
            self.numeric_preservation,
            self.schema_validity,
            1.0 - self.harmful_drift,
        ]
        if self.refusal_correctness is not None:
            dims.append(self.refusal_correctness)
        return round(sum(dims) / len(dims), 4)

    def composite_for(self, task_class: str) -> float:
        """Task-aware composite score using appropriate weights.

        Reasoning/debugging: correctness and key facts dominate.
        Structured/JSON: schema and numerics dominate.
        Default: balanced weighting.
        """
        if task_class in ("reasoning", "debugging", "analysis"):
            return round(
                0.35 * self.correctness
                + 0.25 * self.key_fact_preservation
                + 0.15 * self.numeric_preservation
                + 0.20 * self.reasoning_equivalence
                + 0.05 * self.completeness,
                4,
            )
        if task_class in ("structured", "json", "table"):
            return round(
                0.35 * self.schema_validity
                + 0.30 * self.key_fact_preservation
                + 0.20 * self.numeric_preservation
                + 0.15 * self.completeness,
                4,
            )
        return round(
            0.45 * self.correctness
            + 0.25 * self.completeness
            + 0.20 * self.key_fact_preservation
            + 0.10 * self.reasoning_equivalence,
            4,
        )

    @property
    def passed(self) -> bool:
        """Task equivalent if composite >= 0.85."""
        return self.composite >= 0.85

    def passed_for(self, task_class: str) -> bool:
        return self.composite_for(task_class) >= 0.85

    def to_dict(self) -> dict[str, float]:
        return {
            "correctness": self.correctness,
            "completeness": self.completeness,
            "reasoning_equivalence": self.reasoning_equivalence,
            "key_fact_preservation": self.key_fact_preservation,
            "numeric_preservation": self.numeric_preservation,
            "refusal_correctness": self.refusal_correctness,
            "schema_validity": self.schema_validity,
            "harmful_drift": self.harmful_drift,
            "composite": self.composite,
            "failure_reasons": "; ".join(self.failure_reasons),
        }


@dataclasses.dataclass(slots=True)
class QualityMeasurement:
    """Quality metrics comparing baseline vs optimized responses.

    Includes both legacy semantic_similarity (backward-compat) and
    task-equivalence rubric (source of truth for meaning preservation).
    """

    semantic_similarity: float = 0.0  # 0-1, legacy compat
    exact_match: bool = False
    json_valid: bool | None = None  # None if not applicable
    json_schema_valid: bool | None = None
    tool_calls_equivalent: bool | None = None
    reasoning_equivalent: bool | None = None
    pass_threshold: float = 0.7  # minimum semantic similarity to pass (legacy)
    task_equivalence: TaskEquivalenceScore | None = None  # source of truth

    @property
    def passed(self) -> bool:
        if self.task_equivalence is not None:
            return self.task_equivalence.passed
        return self.semantic_similarity >= self.pass_threshold

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "semantic_similarity": self.semantic_similarity,
            "exact_match": self.exact_match,
            "passed": self.passed,
        }
        if self.json_valid is not None:
            d["json_valid"] = self.json_valid
        if self.json_schema_valid is not None:
            d["json_schema_valid"] = self.json_schema_valid
        if self.tool_calls_equivalent is not None:
            d["tool_calls_equivalent"] = self.tool_calls_equivalent
        if self.reasoning_equivalent is not None:
            d["reasoning_equivalent"] = self.reasoning_equivalent
        if self.task_equivalence is not None:
            d["task_equivalence"] = self.task_equivalence.to_dict()
        return d


@dataclasses.dataclass(slots=True)
class ScenarioResult:
    """Results for a single scenario across N iterations."""

    scenario_name: str
    category: str
    description: str = ""
    provider: str = ""
    model: str = ""

    # Per-iteration measurements
    baseline_latencies: list[LatencyMeasurement] = dataclasses.field(default_factory=list)
    optimized_latencies: list[LatencyMeasurement] = dataclasses.field(default_factory=list)
    baseline_tokens: list[TokenMeasurement] = dataclasses.field(default_factory=list)
    optimized_tokens: list[TokenMeasurement] = dataclasses.field(default_factory=list)
    qualities: list[QualityMeasurement] = dataclasses.field(default_factory=list)

    # Response samples (last iteration only, for inspection)
    baseline_response_sample: str = ""
    optimized_response_sample: str = ""

    # Pipeline metrics (per-transform breakdown from last iteration)
    transform_breakdown: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)

    # Errors
    baseline_errors: list[str] = dataclasses.field(default_factory=list)
    optimized_errors: list[str] = dataclasses.field(default_factory=list)

    # Telemetry captured during replay (transform metrics, cache hits, etc.)
    telemetry: dict[str, Any] = dataclasses.field(default_factory=dict)

    # Provider usage and billing detail (live provider evals only)
    usage_summary: dict[str, Any] = dataclasses.field(default_factory=dict)

    def _mean(self, values: list[float] | list[int]) -> float:
        return statistics.mean(values) if values else 0.0

    def _stdev(self, values: list[float] | list[int]) -> float:
        return statistics.stdev(values) if len(values) > 1 else 0.0

    def _median(self, values: list[float] | list[int]) -> float:
        return statistics.median(values) if values else 0.0

    @property
    def avg_baseline_tokens(self) -> TokenMeasurement:
        before_vals = [t.before for t in self.baseline_tokens]
        after_vals = [t.after for t in self.baseline_tokens]
        return TokenMeasurement(
            before=int(self._mean(before_vals)),
            after=int(self._mean(after_vals)),
            saved=int(self._mean([t.saved for t in self.baseline_tokens])),
            ratio=self._mean([t.ratio for t in self.baseline_tokens]),
        )

    @property
    def avg_optimized_tokens(self) -> TokenMeasurement:
        before_vals = [t.before for t in self.optimized_tokens]
        after_vals = [t.after for t in self.optimized_tokens]
        return TokenMeasurement(
            before=int(self._mean(before_vals)),
            after=int(self._mean(after_vals)),
            saved=int(self._mean([t.saved for t in self.optimized_tokens])),
            ratio=self._mean([t.ratio for t in self.optimized_tokens]),
        )

    @property
    def avg_quality(self) -> QualityMeasurement:
        sims = [q.semantic_similarity for q in self.qualities]
        exacts = [q.exact_match for q in self.qualities]
        task_scores = [q.task_equivalence for q in self.qualities if q.task_equivalence is not None]
        avg_task = None
        if task_scores:
            avg_task = TaskEquivalenceScore(
                correctness=self._mean([t.correctness for t in task_scores]),
                completeness=self._mean([t.completeness for t in task_scores]),
                reasoning_equivalence=self._mean([t.reasoning_equivalence for t in task_scores]),
                key_fact_preservation=self._mean([t.key_fact_preservation for t in task_scores]),
                numeric_preservation=self._mean([t.numeric_preservation for t in task_scores]),
                schema_validity=self._mean([t.schema_validity for t in task_scores]),
                harmful_drift=self._mean([t.harmful_drift for t in task_scores]),
            )
        return QualityMeasurement(
            semantic_similarity=round(self._mean(sims), 4) if sims else 0.0,
            exact_match=sum(exacts) / len(exacts) >= 0.5 if exacts else False,
            pass_threshold=self.qualities[0].pass_threshold if self.qualities else 0.7,
            task_equivalence=avg_task,
        )

    @property
    def avg_baseline_latency(self) -> dict[str, float]:
        totals = [lat.total_ms for lat in self.baseline_latencies]
        networks = [lat.network_ms for lat in self.baseline_latencies]
        return {
            "total_ms": self._mean(totals),
            "total_stdev": self._stdev(totals),
            "network_ms": self._mean(networks),
            "network_stdev": self._stdev(networks),
        }

    @property
    def avg_optimized_latency(self) -> dict[str, float]:
        totals = [lat.total_ms for lat in self.optimized_latencies]
        networks = [lat.network_ms for lat in self.optimized_latencies]
        pipelines = [lat.pipeline_ms for lat in self.optimized_latencies]
        return {
            "total_ms": self._mean(totals),
            "total_stdev": self._stdev(totals),
            "network_ms": self._mean(networks),
            "network_stdev": self._stdev(networks),
            "pipeline_ms": self._mean(pipelines),
            "pipeline_stdev": self._stdev(pipelines),
        }

    @property
    def latency_delta_ms(self) -> float:
        """Difference in average total latency (optimized - baseline). Negative = faster."""
        return self.avg_optimized_latency["total_ms"] - self.avg_baseline_latency["total_ms"]

    @property
    def token_savings(self) -> TokenMeasurement:
        """Net token savings from optimization."""
        # Baseline tokens are the "before" count; optimized tokens are the "after" count
        # But baseline_tokens list contains measurements where before=after (no compression)
        # So we use the optimized measurement's before/after
        opt = self.optimized_tokens[0] if self.optimized_tokens else TokenMeasurement()
        return TokenMeasurement(
            before=opt.before,
            after=opt.after,
            saved=opt.saved,
            ratio=opt.ratio,
        )

    @property
    def cost_savings_usd(self) -> float:
        """Estimated cost savings per 1K requests (input tokens only).
        Uses average pricing: $0.50/1M input tokens for small models,
        $3.00/1M for large models. Conservative estimate.
        """
        saved = self.token_savings.saved
        # Conservative blended rate: $1.50 per 1M tokens
        rate_per_token = 1.50 / 1_000_000
        return saved * rate_per_token * 1000  # per 1K requests

    @property
    def all_passed(self) -> bool:
        return all(q.passed for q in self.qualities) and len(self.qualities) > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "category": self.category,
            "provider": self.provider,
            "model": self.model,
            "iterations": len(self.qualities),
            "tokens": {
                "baseline": dataclasses.asdict(self.avg_baseline_tokens),
                "optimized": dataclasses.asdict(self.avg_optimized_tokens),
                "savings": dataclasses.asdict(self.token_savings),
            },
            "latency_ms": {
                "baseline": self.avg_baseline_latency,
                "optimized": self.avg_optimized_latency,
                "delta": round(self.latency_delta_ms, 3),
            },
            "quality": {
                "semantic_similarity": round(self.avg_quality.semantic_similarity, 4),
                "exact_match": self.avg_quality.exact_match,
                "json_valid": self.avg_quality.json_valid,
                "tool_calls_equivalent": self.avg_quality.tool_calls_equivalent,
                "passed": self.all_passed,
            },
            "cost": {
                "savings_per_1k_requests_usd": round(self.cost_savings_usd, 4),
            },
            "errors": {
                "baseline": len(self.baseline_errors),
                "optimized": len(self.optimized_errors),
            },
            "telemetry": self.telemetry,
            "usage": self.usage_summary,
            "transform_breakdown": self.transform_breakdown,
            "response_samples": {
                "baseline": self.baseline_response_sample[:500],
                "optimized": self.optimized_response_sample[:500],
            },
        }


@dataclasses.dataclass(slots=True)
class BenchmarkReport:
    """Complete benchmark report across all scenarios."""

    runner_name: str
    provider: str
    model: str
    timestamp: float = dataclasses.field(default_factory=time.time)
    scenarios: list[ScenarioResult] = dataclasses.field(default_factory=list)
    config: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def total_scenarios(self) -> int:
        return len(self.scenarios)

    @property
    def passed_scenarios(self) -> int:
        return sum(1 for s in self.scenarios if s.all_passed)

    @property
    def total_token_savings(self) -> int:
        return sum(s.token_savings.saved for s in self.scenarios)

    @property
    def avg_reduction_ratio(self) -> float:
        ratios = [s.token_savings.ratio for s in self.scenarios if s.token_savings.before > 0]
        return statistics.mean(ratios) if ratios else 0.0

    @property
    def avg_pipeline_latency_ms(self) -> float:
        lats = [s.avg_optimized_latency["pipeline_ms"] for s in self.scenarios]
        return statistics.mean(lats) if lats else 0.0

    @property
    def avg_quality_score(self) -> float:
        # Source of truth: task_equivalence composite when available;
        # falls back to semantic_similarity only when no task_equivalence exists.
        te_scores: list[float] = []
        sim_scores: list[float] = []
        for s in self.scenarios:
            if s.avg_quality.task_equivalence is not None:
                te_scores.append(s.avg_quality.task_equivalence.composite)
            else:
                sim_scores.append(s.avg_quality.semantic_similarity)
        # Prefer task_equivalence average if any scenarios have it
        if te_scores:
            return round(statistics.mean(te_scores), 4)
        return round(statistics.mean(sim_scores), 4) if sim_scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "runner": self.runner_name,
            "provider": self.provider,
            "model": self.model,
            "timestamp": self.timestamp,
            "summary": {
                "total_scenarios": self.total_scenarios,
                "passed_scenarios": self.passed_scenarios,
                "total_token_savings": self.total_token_savings,
                "avg_reduction_ratio": round(self.avg_reduction_ratio, 4),
                "avg_pipeline_latency_ms": round(self.avg_pipeline_latency_ms, 4),
                "avg_quality_score": round(self.avg_quality_score, 4),
            },
            "config": self.config,
            "scenarios": [s.to_dict() for s in self.scenarios],
        }
