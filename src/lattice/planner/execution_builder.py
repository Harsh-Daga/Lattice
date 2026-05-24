"""ExecutionPlan builder — single entry point for creating an ExecutionPlan.

Phase 10 — End-to-end wiring.

This module centralizes the logic for building an ExecutionPlan from:
  1. Request classification (task, risk, content shape)
  2. Provider strategy (cache mode, preferred optimizers, max context)
  3. Transport planning (delta, framing, codec, resume)
  4. Cache planning (provider-specific breakpoints)
  5. Fallback planning (provider fallback, retry)

Usage::
    from lattice.planner.execution_builder import build_execution_plan
    plan = build_execution_plan(
        request=request,
        provider_name="openai",
        model="gpt-4",
        session_id="sess_abc",
        config=config,
    )
    request.metadata["_lattice_execution_plan"] = plan.to_dict()
    context.session_state["_lattice_execution_plan"] = plan
"""

from __future__ import annotations

from lattice.core.config import LatticeConfig
from lattice.planner.execution_plan import (
    CachePlanEntry,
    ExecutionPlan,
    FallbackPlan,
    TransportPlanEntry,
)
from lattice.planner.provider_strategy import (
    build_cache_plan_for_provider,
    simulate_provider_cache,
)
from lattice.planner.request_classifier import RequestClassifier
from lattice.planner.task_classifier import TaskClass
from lattice.planner.transport_planner import build_transport_plan
from lattice.planner.unified_planner import SemanticProfile, UnifiedPlanner
from lattice.transport.types import Request


def build_execution_plan(
    *,
    request: Request,
    provider_name: str,
    model: str,
    session_id: str | None = None,
    base_sequence: int = 0,
    config: LatticeConfig | None = None,
    is_streaming: bool = False,
    is_multiplex: bool = False,
    estimated_tokens: int | None = None,
    fallback_strategy: str = "retry",
    allowed_optimizers: list[str] | None = None,
) -> ExecutionPlan:
    """Build a complete ExecutionPlan from request + provider + session context.

    Steps:
      1. Classify the request (task, risk, quality_floor, budget).
      2. Resolve provider strategy (preferred optimizers, cache mode).
      3. Build transport plan (delta, framing, codec, resume).
      4. Build cache plan (provider-specific breakpoints).
      5. Create ExecutionPlan with all fields.

    Args:
        request: The incoming request.
        provider_name: Resolved provider name.
        model: Target model identifier.
        session_id: Optional session for delta/resume.
        base_sequence: Sequence number for delta mode.
        config: LatticeConfig for defaults.
        is_streaming: Whether the request uses streaming.
        is_multiplex: Whether multiplex framing is desired.
        estimated_tokens: Estimated tokens (defaults to request.token_estimate).
        fallback_strategy: Retry strategy for transport.
        allowed_optimizers: Override optimizers from classification.

    Returns:
        A fully-populated ExecutionPlan.
    """
    config = config or LatticeConfig()
    classifier = RequestClassifier()
    classification = classifier.classify(request)

    task_class = classification["task_class"]
    risk_level = classification["risk_level"]
    context_length = request.token_estimate

    profile = SemanticProfile(
        task_class=TaskClass(task_class),
        task_label=classification.get("preferred_strategy", ""),
        risk_total=classification["risk_total"],
        context_length=context_length,
        has_tool_calls=request.is_tool_conversation,
        is_streaming=is_streaming,
        is_conservative=classification["is_conservative"],
        provider=provider_name,
        model=model,
    )

    planner = UnifiedPlanner()
    plan = planner.plan(request, profile)

    plan_transforms = list(plan.transforms)
    if allowed_optimizers is not None:
        allowed_set = set(allowed_optimizers)
        constituent_optimizers = {
            "structure_optimizer",
            "reference_optimizer",
            "tool_optimizer",
            "context_optimizer",
            "diagnostic_optimizer",
        }
        if allowed_set & constituent_optimizers:
            allowed_set.add("representation_optimizer")
        plan_transforms = [
            name
            for name in plan_transforms
            if not name.endswith("_optimizer") or name in allowed_set
        ]
    allowed_optimizers_final = [name for name in plan_transforms if name.endswith("_optimizer")]

    # Build transport plan
    tokens = estimated_tokens if estimated_tokens is not None else request.token_estimate
    transport = build_transport_plan(
        provider=provider_name,
        model=model,
        session_id=session_id,
        base_sequence=base_sequence,
        is_multiplex=is_multiplex,
        is_streaming=is_streaming,
        estimated_tokens=tokens,
        fallback_strategy=fallback_strategy,
    )

    # Build cache plan
    segment_count = len(request.messages)
    cache_plan_raw = build_cache_plan_for_provider(
        provider_name,
        segment_count=segment_count,
        estimated_tokens=tokens,
    )
    cache_plan = [
        CachePlanEntry(
            segment_index=entry["segment_index"],
            provider_mode=entry["provider_mode"],
            expected_cached_tokens=entry["expected_cached_tokens"],
            annotations=entry.get("annotations", {}),
        )
        for entry in cache_plan_raw
    ]

    # Representation plan: canonical execution order from UnifiedPlanner
    representation_plan = plan_transforms

    # Fallback plan
    fallback = FallbackPlan(
        fallback_provider=None,
        fallback_model=None,
        disable_optimizers=risk_level == "critical",
        retry_count=3 if fallback_strategy == "retry" else 0,
    )

    # Write cache plan into request metadata so provider adapters can read it.
    request.metadata["_lattice_cache_plan"] = [e.to_dict() for e in cache_plan]
    request.metadata["_lattice_execution_plan"] = plan.to_dict()
    request.metadata["_lattice_cache_simulation"] = simulate_provider_cache(
        provider_name,
        model,
        estimated_tokens=tokens,
        cache_plan=request.metadata["_lattice_cache_plan"],
        prefix_manifest=request.metadata.get("_prefix_manifest"),
    ).to_dict()

    return ExecutionPlan(
        request_id="",
        session_id=session_id,
        provider=provider_name,
        model=model,
        task_class=task_class,
        risk_level=risk_level,
        latency_budget_ms=plan.latency_budget_ms,
        quality_floor=plan.quality_floor,
        allowed_optimizers=allowed_optimizers_final,
        blocked_optimizers={},
        representation_plan=representation_plan,
        transport_plan=TransportPlanEntry(
            use_delta=transport.delta_mode,
            use_multiplex=transport.use_framing and is_multiplex,
            use_framing=transport.use_framing,
            resume_enabled=transport.resume_enabled,
            compression_codec=transport.compression_codec,
            wire_format=transport.wire_format,
        ),
        cache_plan=cache_plan,
        fallback_plan=fallback,
    )


__all__ = ["build_execution_plan"]
