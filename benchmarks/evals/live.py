"""Live-provider helpers for production evals.

Orchestrates provider calls and local quality comparison.
For the source-of-truth live task-equivalence validation, see
benchmarks/evals/runner.py: evaluate_task_equivalence_with_judge()."""

from __future__ import annotations

import time
from typing import Any

from benchmarks.framework.types import LatencyMeasurement, QualityMeasurement, TaskEquivalenceScore
from benchmarks.metrics.quality import evaluate_response  # legacy compat / JSON+tool checks
from benchmarks.scenarios.prompts import BenchmarkScenario
from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.cost_estimator import CostEstimator
from lattice.core.credentials import CredentialResolver
from lattice.core.pipeline import CompressorPipeline
from lattice.core.pipeline_factory import build_default_pipeline
from lattice.core.result import is_err, unwrap, unwrap_err
from lattice.core.serialization import message_from_dict, message_to_dict
from lattice.core.transport import Request
from lattice.providers.transport import DirectHTTPProvider, ProviderRegistry
from lattice.utils.validation import lossy_transform_allowed, request_safety_profile, structure_signature


def build_full_pipeline(config: LatticeConfig | None = None) -> CompressorPipeline:
    """Build the complete LATTICE pipeline with all production transforms."""
    config = config or LatticeConfig.auto()
    return build_default_pipeline(config, include_execution_transforms=False)


def setup_provider(provider_name: str, base_url: str | None = None, api_key: str | None = None) -> DirectHTTPProvider:
    """Set up DirectHTTPProvider with resolved credentials."""
    registry = ProviderRegistry()
    credentials = CredentialResolver()
    resolved_api_key = credentials.resolve(provider_name).api_key
    key = api_key or resolved_api_key
    provider = DirectHTTPProvider(
        registry=registry,
        default_api_key=key,
        provider_base_urls={} if not base_url else {provider_name: base_url},
        credentials=credentials,
    )
    return provider


async def run_scenario(
    *,
    scenario: BenchmarkScenario,
    provider: DirectHTTPProvider | None,
    pipeline: CompressorPipeline,
    model: str,
    provider_name: str,
    dry_run: bool,
    iteration: int,
) -> tuple[
    LatencyMeasurement | None,
    LatencyMeasurement | None,
    int,
    int,
    QualityMeasurement | None,
    str,
    str,
    str | None,
    str | None,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Run a single scenario through baseline and optimized paths."""
    messages = [message_from_dict(m) for m in scenario.messages]
    request = Request(
        model=model,
        messages=messages,
        temperature=scenario.temperature,
        max_tokens=scenario.max_tokens,
        tools=scenario.tools,
    )
    prompt_text = "\n".join(msg.content for msg in request.messages)
    prompt_profile = request_safety_profile(request)

    baseline_resp_text = ""
    baseline_error = None
    baseline_latency = None
    baseline_usage: dict[str, Any] = {}
    if not dry_run and provider is not None:
        t0_total = time.perf_counter()
        try:
            baseline_resp = await provider.completion(
                model=model,
                messages=[message_to_dict(m) for m in messages],
                temperature=scenario.temperature,
                max_tokens=scenario.max_tokens,
                tools=scenario.tools,
            )
            baseline_resp_text = baseline_resp.content or ""
            baseline_usage = dict(baseline_resp.usage or {})
            baseline_latency = LatencyMeasurement(
                pipeline_ms=0.0,
                network_ms=(time.perf_counter() - t0_total) * 1000,
                total_ms=(time.perf_counter() - t0_total) * 1000,
            )
        except Exception as exc:
            baseline_error = str(exc)

    ctx = TransformContext(
        request_id=f"bench-{scenario.name}-{iteration}",
        provider=provider_name,
        model=model,
    )
    pipeline_start = time.perf_counter()
    result = await pipeline.process(request, ctx)
    pipeline_ms = (time.perf_counter() - pipeline_start) * 1000
    if is_err(result):
        err = unwrap_err(result)
        return (
            baseline_latency,
            None,
            request.token_estimate,
            request.token_estimate,
            None,
            baseline_resp_text,
            "",
            baseline_error,
            f"Pipeline failed: {err.message}",
            baseline_usage,
            {},
            {},
            {},
        )

    compressed = unwrap(result)
    optimized_resp_text = ""
    optimized_error = None
    optimized_latency = None
    optimized_usage: dict[str, Any] = {}

    if not dry_run and provider is not None:
        t0_total = time.perf_counter()
        try:
            compressed_messages = [message_to_dict(m) for m in compressed.messages]
            optimized_resp = await provider.completion(
                model=model,
                messages=compressed_messages,
                temperature=scenario.temperature,
                max_tokens=scenario.max_tokens,
                tools=scenario.tools,
            )
            optimized_resp_text = optimized_resp.content or ""
            optimized_usage = dict(optimized_resp.usage or {})
            optimized_latency = LatencyMeasurement(
                pipeline_ms=pipeline_ms,
                network_ms=(time.perf_counter() - t0_total) * 1000,
                total_ms=(time.perf_counter() - t0_total) * 1000 + pipeline_ms,
            )
        except Exception as exc:
            optimized_error = str(exc)

    if optimized_latency is None:
        optimized_latency = LatencyMeasurement(
            pipeline_ms=pipeline_ms,
            network_ms=0.0,
            total_ms=pipeline_ms,
        )

    baseline_tokens = request.token_estimate
    optimized_tokens = compressed.token_estimate
    quality: QualityMeasurement | None = None
    if not dry_run and baseline_resp_text and optimized_resp_text:
        te = _compute_task_equivalence(
            baseline_output=baseline_resp_text,
            optimized_output=optimized_resp_text,
            scenario=scenario,
        )
        quality = QualityMeasurement(
            task_equivalence=te,
            semantic_similarity=te.composite,  # align legacy field with authoritative score
        )
    # Dry-run / no outputs: quality is explicitly unavailable — not 0.0.
    # A misleading 0.0 placeholder would contaminate avg_quality_score.

    usage_summary: dict[str, Any] = {}
    if baseline_usage or optimized_usage:
        estimator = CostEstimator()
        baseline_cost = estimator.compute_actual(provider=provider_name, model=model, usage=baseline_usage) if baseline_usage else None
        optimized_cost = estimator.compute_actual(provider=provider_name, model=model, usage=optimized_usage) if optimized_usage else None
        usage_summary = {
            "baseline": baseline_usage,
            "optimized": optimized_usage,
            "costs": {
                "baseline_total_usd": getattr(baseline_cost, "total_cost_usd", 0.0) if baseline_cost else 0.0,
                "optimized_total_usd": getattr(optimized_cost, "total_cost_usd", 0.0) if optimized_cost else 0.0,
                "delta_usd": (
                    (getattr(optimized_cost, "total_cost_usd", 0.0) if optimized_cost else 0.0)
                    - (getattr(baseline_cost, "total_cost_usd", 0.0) if baseline_cost else 0.0)
                ),
            },
        }

    transform_breakdown = {
        name: {
            "latency_ms": metrics.get("latency_ms", 0),
            "tokens_after": metrics.get("tokens_after", 0),
            **{k: v for k, v in metrics.items() if k not in {"latency_ms", "tokens_after"}},
        }
        for name, metrics in ctx.metrics.get("transforms", {}).items()
        if isinstance(metrics, dict)
    }

    runtime_metadata = compressed.metadata if isinstance(compressed.metadata, dict) else {}
    telemetry: dict[str, Any] = {
        "scenario": {
            "name": scenario.name,
            "category": scenario.category,
            "complexity": scenario.complexity,
            "expected_tier": scenario.expected_tier,
            "target_features": list(scenario.target_features),
            "proof": scenario.proof,
        },
        "prompt": {
            "safety_profile": {
                "has_structured_content": prompt_profile.has_structured_content,
                "has_code_blocks": prompt_profile.has_code_blocks,
                "has_strict_instructions": prompt_profile.has_strict_instructions,
                "has_tool_calls": prompt_profile.has_tool_calls,
                "has_high_stakes_entities": prompt_profile.has_high_stakes_entities,
                "long_form": prompt_profile.long_form,
            },
            "lossy_transform_allowed": lossy_transform_allowed(request),
            "structure_signature": structure_signature(prompt_text),
        },
        "runtime": runtime_metadata.get("_lattice_runtime", {}),
        "runtime_contract": runtime_metadata.get("_lattice_runtime_contract", {}),
        "transforms": ctx.metrics.get("transforms", {}),
        "transforms_applied": ctx.transforms_applied,
        "pipeline": {
            "tokens_in": ctx.metrics.get("tokens_in", request.token_estimate),
            "tokens_out": ctx.metrics.get("tokens_out", compressed.token_estimate),
            "latency_ms": ctx.metrics.get("latency_ms", 0.0),
            "transform_latency_ms": ctx.metrics.get("transform_latency_ms", 0.0),
        },
    }

    return (
        baseline_latency,
        optimized_latency,
        baseline_tokens,
        optimized_tokens,
        quality,
        baseline_resp_text,
        optimized_resp_text,
        baseline_error,
        optimized_error,
        baseline_usage,
        optimized_usage,
        transform_breakdown,
        usage_summary,
        telemetry,
    )


def _compute_task_equivalence(
    baseline_output: str,
    optimized_output: str,
    scenario: BenchmarkScenario,
) -> TaskEquivalenceScore:
    """Compute task equivalence using the structural evaluator.

    The structural evaluator is deterministic (no LLM dependency) and
    checks: correctness (length ratio), key fact preservation (entity
    overlap), numeric preservation, schema validity, reasoning signals,
    and placeholder leakage.
    """
    from benchmarks.evals.runner import evaluate_task_equivalence_structural

    return evaluate_task_equivalence_structural(
        baseline_output=baseline_output,
        optimized_output=optimized_output,
        required_properties=getattr(scenario, "required_answer_properties", []) or [],
    )


def provider_name_from_model(model: str) -> str:
    """Backward-compatible helper for legacy callers."""
    if "/" in model:
        return model.split("/", 1)[0]
    return "openai"
