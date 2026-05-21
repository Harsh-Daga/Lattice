"""Shared runtime-state helpers for canonical execution-plan propagation."""

from __future__ import annotations

from typing import Any

from lattice.core.context import TransformContext
from lattice.transport.types import Request


def coerce_execution_plan(plan: Any) -> Any | None:
    """Accept dict payloads and concrete plan objects.

    The runtime still carries both legacy and v2 plan objects through
    compatibility surfaces. This helper normalizes the common cases without
    forcing callers to care which concrete implementation they received.
    """
    if plan is None:
        return None
    if _is_core_execution_plan(plan):
        return plan
    normalized = _normalize_legacy_execution_plan(plan)
    if normalized is not None:
        return normalized
    if isinstance(plan, dict):
        try:
            from lattice.ir.primitives import ExecutionPlan as CoreExecutionPlan

            if "transforms" in plan:
                return CoreExecutionPlan.from_dict(plan)
            return _normalize_legacy_execution_plan(plan)
        except Exception:
            try:
                from lattice.planner.execution_plan import ExecutionPlan as LegacyExecutionPlan

                legacy = LegacyExecutionPlan.from_dict(plan)
                return _normalize_legacy_execution_plan(legacy)
            except Exception:
                return None
    return plan


def get_ir_metadata(context: TransformContext) -> dict[str, Any]:
    """Return canonical PromptIR metadata when available.

    The v2 runtime stores the canonical IR in session_state["_lattice_ir_v2"].
    Consumers should prefer this over ad-hoc session or request side channels.
    """
    ir_v2 = context.session_state.get("_lattice_ir_v2")
    metadata = getattr(ir_v2, "metadata", None)
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    try:
        return dict(metadata)
    except Exception:
        return {}


def get_ir_metadata_value(context: TransformContext, key: str, default: Any = None) -> Any:
    """Read a single value from canonical PromptIR metadata."""
    metadata = get_ir_metadata(context)
    return metadata.get(key, default)


def get_canonical_state_value(context: TransformContext, key: str, default: Any = None) -> Any:
    """Read a runtime value from PromptIR metadata first, then session state.

    This keeps PromptIR as the primary source of truth while preserving
    compatibility with older surfaces that still mirror state into session
    storage.
    """
    metadata = get_ir_metadata(context)
    if key in metadata:
        return thaw_value(metadata.get(key, default))
    return thaw_value(context.session_state.get(key, default))


def get_canonical_request_value(
    request: Request,
    context: TransformContext | None,
    key: str,
    default: Any = None,
) -> Any:
    """Read a runtime value from PromptIR metadata first, then request/session state.

    This is for compatibility surfaces that still accept request.metadata as an
    input source while keeping PromptIR as the primary canonical storage.
    """
    if context is not None:
        metadata = get_ir_metadata(context)
        if key in metadata:
            return thaw_value(metadata.get(key, default))
    if key in request.metadata:
        return thaw_value(request.metadata.get(key, default))
    if context is not None:
        return thaw_value(context.session_state.get(key, default))
    return default


def thaw_value(value: Any) -> Any:
    """Best-effort thaw for frozen IR metadata payloads.

    `PromptIRV2.add_metadata()` stores nested dict/list/set payloads as
    immutable containers. This helper converts the common shapes back into
    plain Python containers for read-only consumers.
    """
    if isinstance(value, frozenset):
        if all(isinstance(item, tuple) and len(item) == 2 for item in value):
            return {k: thaw_value(v) for k, v in value}
        return [thaw_value(item) for item in value]
    if isinstance(value, tuple):
        return [thaw_value(item) for item in value]
    if isinstance(value, dict):
        return {k: thaw_value(v) for k, v in value.items()}
    return value


def _is_core_execution_plan(plan: Any) -> bool:
    return hasattr(plan, "transforms") and not hasattr(plan, "representation_plan")


def _normalize_legacy_execution_plan(plan: Any) -> Any | None:
    """Convert the legacy planner execution plan into the core immutable plan."""
    if not hasattr(plan, "representation_plan"):
        return None
    try:
        from lattice.ir.primitives import CachePlan as CoreCachePlan
        from lattice.ir.primitives import ExecutionPlan as CoreExecutionPlan
        from lattice.ir.primitives import TransportPlan as CoreTransportPlan
    except Exception:
        return None

    cache_plan = getattr(plan, "cache_plan", None) or []
    first_cache = cache_plan[0] if cache_plan else None
    cache_hint = None
    stable_hash = None
    use_delta = False
    compression_codec = None
    if first_cache is not None:
        if isinstance(first_cache, dict):
            cache_hint = first_cache.get("provider_mode")
            stable_hash = first_cache.get("annotations", {}).get("stable_prefix_hash")
        else:
            cache_hint = getattr(first_cache, "provider_mode", None)
            annotations = getattr(first_cache, "annotations", {}) or {}
            if isinstance(annotations, dict):
                stable_hash = annotations.get("stable_prefix_hash")
    transport = getattr(plan, "transport_plan", None)
    if transport is not None:
        use_delta = bool(
            getattr(transport, "use_delta", getattr(transport, "delta_encoding", False))
        )
        compression_codec = getattr(transport, "compression_codec", None)
        if cache_hint is None:
            cache_hint = getattr(transport, "provider", None)
        if stable_hash is None:
            stable_hash = getattr(transport, "stable_prefix_hash", None)

    return CoreExecutionPlan(
        transforms=tuple(getattr(plan, "representation_plan", ())),
        quality_floor=float(getattr(plan, "quality_floor", 0.85)),
        latency_budget_ms=float(getattr(plan, "latency_budget_ms", 100.0)),
        utility_score=float(getattr(plan, "utility_score", 0.0)),
        beam_width=5,
        max_depth=6,
        cache_plan=None
        if cache_hint is None
        and stable_hash is None
        and not use_delta
        and compression_codec is None
        else CoreCachePlan(
            use_delta=use_delta,
            compression_codec=compression_codec,
            provider_cache_hint=cache_hint,
            stable_prefix_hash=stable_hash,
        ),
        transport_plan=None
        if cache_hint is None
        and stable_hash is None
        and not use_delta
        and compression_codec is None
        else CoreTransportPlan(
            use_delta=use_delta,
            compression_codec=compression_codec,
            provider_cache_hint=cache_hint,
            stable_prefix_hash=stable_hash,
        ),
        provider=str(getattr(plan, "provider", "generic")),
        model=str(getattr(plan, "model", "")),
    )


def normalize_cache_plan_entries(cache_plan: Any) -> list[dict[str, Any]]:
    """Convert cache-plan entries into JSON-safe dicts."""
    if not cache_plan:
        return []

    normalized: list[dict[str, Any]] = []
    for entry in cache_plan:
        if isinstance(entry, dict):
            normalized.append(dict(entry))
            continue
        if hasattr(entry, "to_dict"):
            normalized.append(entry.to_dict())
            continue
        normalized.append(
            {
                "segment_index": getattr(entry, "segment_index", 0),
                "provider_mode": getattr(entry, "provider_mode", ""),
                "expected_cached_tokens": getattr(entry, "expected_cached_tokens", 0),
                "annotations": getattr(entry, "annotations", {}),
            }
        )
    return normalized


def sum_expected_cached_tokens(cache_plan: Any) -> int:
    """Return the total expected cached tokens for mixed cache-plan shapes."""
    total = 0
    for entry in cache_plan or []:
        if isinstance(entry, dict):
            total += int(entry.get("expected_cached_tokens", 0))
        else:
            total += int(getattr(entry, "expected_cached_tokens", 0))
    return total


def persist_execution_plan_state(
    request: Request,
    context: TransformContext,
    plan: Any,
    *,
    cache_plan: Any | None = None,
    cache_simulation: Any | None = None,
) -> None:
    """Write the canonical plan and cache state to request + context.

    This is the shared mutation point for the proxy/gateway/profile surfaces so
    each entrypoint writes the same runtime state shape.
    """
    normalized_plan = plan.to_dict() if hasattr(plan, "to_dict") else plan
    request.metadata["_lattice_execution_plan"] = normalized_plan
    context.session_state["_lattice_execution_plan"] = coerce_execution_plan(plan)

    if cache_plan is not None:
        normalized_cache = normalize_cache_plan_entries(cache_plan)
        request.metadata["_lattice_cache_plan"] = normalized_cache
        context.session_state["_lattice_cache_plan"] = {"plan": normalized_cache}
        context.session_state["cache_plan_entries"] = cache_plan

    if cache_simulation is not None:
        normalized_sim = (
            cache_simulation.to_dict() if hasattr(cache_simulation, "to_dict") else cache_simulation
        )
        request.metadata["_lattice_cache_simulation"] = normalized_sim
        context.session_state["_lattice_cache_simulation"] = normalized_sim


def persist_session_plan_state(
    session_metadata: dict[str, Any],
    plan: Any,
    *,
    cache_plan: Any | None = None,
    cache_simulation: Any | None = None,
) -> None:
    """Write canonical execution-plan state into session metadata."""
    session_metadata["_lattice_execution_plan"] = (
        plan.to_dict() if hasattr(plan, "to_dict") else plan
    )
    if cache_plan is not None:
        normalized_cache = normalize_cache_plan_entries(cache_plan)
        session_metadata["_lattice_cache_plan"] = normalized_cache
        session_metadata["_lattice_cache_plan_state"] = {"plan": normalized_cache}
    if cache_simulation is not None:
        session_metadata["_lattice_cache_simulation"] = (
            cache_simulation.to_dict() if hasattr(cache_simulation, "to_dict") else cache_simulation
        )
