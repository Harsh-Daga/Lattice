"""Safety gates extracted from v1 CompressorPipeline (Phase 3).

Each gate is a pure helper that consumes request + context state and returns
a decision (skip/veto/rollback/None). The Pipeline.compress() driver applies
them in the order v1 used so behavior is bit-for-bit preserved.

Gate inventory (matches docs/refactor/STATUS.md §5.2):
    1. Policy Allow/Skip/Reject (lives in pipeline.policy.OptimizationPolicy)
    2. Runtime-budget skip — budget-sensitive transforms vs cumulative wallclock
    3. Semantic-risk gate — transform_allowed_at_risk(name, risk)
    4. Protected-span DANGEROUS veto — bucket==DANGEROUS + non-empty protected spans
    5. Scheduler blocking — _lattice_schedule blocked/allowed sets (with optimizer alias)
    6. MILV post-transform validation — runtime judge on lossy transforms
    7. Transform reputation tracking — record per-transform success/failure
    8. Expansion guard / rollback — bloat threshold; plus negative-savings,
       compression-limit (tier+task), placeholder-leakage, PSG numeric/entity/
       format/signal preservation, aggregate debugging compression cap.

All helpers are sync. Pipeline.compress drives them around each transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lattice.core.context import TransformContext
from lattice.planner.runtime_state import get_canonical_request_value
from lattice.transport.types import Request

# Per-task compression limits (formerly core/scheduler.py).
_TASK_COMPRESSION_LIMITS: dict[str, float] = {
    "reasoning": 0.10,
    "debugging": 0.40,
    "analysis": 0.25,
    "structured": 0.30,
    "retrieval": 0.40,
    "summarization": 0.35,
    "simple": 0.50,
}

# Transform classification sets (lifted verbatim from v1 CompressorPipeline).
QUALITY_ONLY_TRANSFORMS: frozenset[str] = frozenset(
    {
        "content_profiler",
        "runtime_contract",
        "strategy_selector",
        "causal_chain",
    }
)

BUDGET_SENSITIVE_TRANSFORMS: frozenset[str] = frozenset({"rate_distortion"})

IRREVERSIBLE_TRANSFORMS: frozenset[str] = frozenset({"message_dedup", "rate_distortion"})

PLACEHOLDER_USING_TRANSFORMS: frozenset[str] = frozenset(
    {
        "reference_sub",
        "path_prefix",
        "crossref_substitution",
        "crossref_compressor",
        "representation_optimizer",
        "reference_optimizer",
    }
)


@dataclass(frozen=True)
class GateDecision:
    """Decision returned by a pre-execution gate.

    `skip=True` → transform should not run; record metric + continue.
    `veto=True` → same as skip but logged differently (DANGEROUS bucket).
    `reason` → metric/log payload.
    """

    skip: bool = False
    veto: bool = False
    reason: str = ""


# ---------------------------------------------------------------------------
# Runtime budget
# ---------------------------------------------------------------------------


def runtime_budget_ms(request: Request, context: TransformContext) -> float:
    """Read the per-request runtime-contract budget (ms)."""
    contract = get_canonical_request_value(request, context, "_lattice_runtime_contract")
    if not isinstance(contract, dict):
        return 0.0
    value = contract.get("max_transform_latency_ms")
    if isinstance(value, int | float):
        return max(0.0, float(value))
    return 0.0


def runtime_budget_blocks(
    name: str,
    request: Request,
    context: TransformContext,
    cumulative_ms: float,
) -> GateDecision:
    budget_ms = runtime_budget_ms(request, context)
    if (
        budget_ms > 0
        and name != "runtime_contract"
        and name in BUDGET_SENSITIVE_TRANSFORMS
        and cumulative_ms >= budget_ms
    ):
        return GateDecision(skip=True, reason="runtime_budget_exhausted")
    return GateDecision()


# ---------------------------------------------------------------------------
# Semantic-risk gate
# ---------------------------------------------------------------------------


def risk_gate_blocks(
    name: str,
    request: Request,
    context: TransformContext,
    profiler_present: bool,
) -> GateDecision:
    """Block transforms disallowed by the per-request risk score.

    When content_profiler is registered but no risk data is present, only
    SAFE transforms run (conservative fallback). When profiler is absent,
    no gating (caller opted out of risk-aware compression).
    """
    if not profiler_present:
        return GateDecision()

    from lattice.utils.validation import (
        SemanticRiskScore,
        TransformSafetyBucket,
        get_transform_safety_bucket,
        transform_allowed_at_risk,
    )

    risk_data = get_canonical_request_value(request, context, "_lattice_risk_score", {})
    if risk_data and isinstance(risk_data, dict):
        risk = SemanticRiskScore(
            strict_instructions=float(risk_data.get("strict_instructions", 0)),
            sensitive_domain=float(risk_data.get("sensitive_domain", 0)),
            structured_output=float(risk_data.get("structured_output", 0)),
            high_stakes_entities=float(risk_data.get("high_stakes_entities", 0)),
            reasoning_heavy=float(risk_data.get("reasoning_heavy", 0)),
            intentional_repetition=float(risk_data.get("intentional_repetition", 0)),
            tool_call_dependency=float(risk_data.get("tool_call_dependency", 0)),
            formatting_constraints=float(risk_data.get("formatting_constraints", 0)),
        )
        allowed, reason = transform_allowed_at_risk(name, risk)
    else:
        bucket = get_transform_safety_bucket(name)
        allowed = bucket == TransformSafetyBucket.SAFE
        reason = "safe_transform" if allowed else "profiler_failed_no_risk_data"

    if not allowed:
        return GateDecision(skip=True, reason=reason)
    return GateDecision()


# ---------------------------------------------------------------------------
# Protected-span DANGEROUS veto
# ---------------------------------------------------------------------------


def protected_span_veto(name: str, request: Request, context: TransformContext) -> GateDecision:
    from lattice.utils.validation import TransformSafetyBucket, get_transform_safety_bucket

    bucket = get_transform_safety_bucket(name)
    if bucket != TransformSafetyBucket.DANGEROUS:
        return GateDecision()
    protected = get_canonical_request_value(request, context, "_lattice_protected_spans", [])
    if protected:
        return GateDecision(veto=True, reason="protected_spans_present")
    return GateDecision()


# ---------------------------------------------------------------------------
# Scheduler blocking
# ---------------------------------------------------------------------------


def scheduler_blocks(name: str, request: Request, context: TransformContext) -> GateDecision:
    """Enforce _lattice_schedule blocked/allowed sets (with optimizer alias handling)."""
    schedule = get_canonical_request_value(request, context, "_lattice_schedule", {})
    if not (schedule and isinstance(schedule, dict)):
        return GateDecision()

    blocked_names = set(schedule.get("blocked", []))
    allowed_names = set(schedule.get("allowed", []))
    allowed_optimizers = set(schedule.get("allowed_optimizers", []))

    from lattice.transforms.registry import get_transform_spec

    spec = get_transform_spec(name)
    canonical = spec.canonical_name if spec else name

    if canonical in blocked_names:
        reason = "scheduler_blocked"
        for entry in schedule.get("schedule", []):
            if isinstance(entry, dict) and entry.get("name") == canonical:
                reason = entry.get("reason", "scheduler_blocked")
                break
        return GateDecision(skip=True, reason=reason)

    is_optimizer = canonical.endswith("_optimizer")
    if is_optimizer and canonical in allowed_optimizers:
        return GateDecision()
    if allowed_names and canonical not in allowed_names:
        return GateDecision(skip=True, reason="not_in_allowed_list")
    return GateDecision()


# ---------------------------------------------------------------------------
# Post-execution guards
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PostCheckResult:
    rollback: bool = False
    fail: bool = False
    code: str = ""
    reason: str = ""
    extra: dict[str, Any] | None = None


def check_expansion(
    name: str,
    tokens_before: int,
    tokens_after: int,
    max_ratio: float,
) -> PostCheckResult:
    if tokens_before <= 0:
        return PostCheckResult()
    ratio = tokens_after / tokens_before
    if ratio > max_ratio:
        return PostCheckResult(
            rollback=True,
            reason="expansion_exceeded",
            extra={"expansion_ratio": round(ratio, 2)},
        )
    return PostCheckResult()


def check_negative_savings(name: str, tokens_before: int, tokens_after: int) -> PostCheckResult:
    if (
        tokens_before > 0
        and tokens_after > tokens_before
        and name not in PLACEHOLDER_USING_TRANSFORMS
        and name not in QUALITY_ONLY_TRANSFORMS
    ):
        return PostCheckResult(
            rollback=True,
            reason="negative_savings",
            extra={"tokens_delta": tokens_after - tokens_before},
        )
    return PostCheckResult()


def check_compression_limit(
    name: str,
    request: Request,
    context: TransformContext,
    tokens_before: int,
    tokens_after: int,
) -> PostCheckResult:
    """REASONING and debugging tier per-transform compression caps."""
    if tokens_before <= 0 or name in ("content_profiler", "runtime_contract"):
        return PostCheckResult()

    task_data = get_canonical_request_value(request, context, "_lattice_task_classification", {})
    if not isinstance(task_data, dict):
        return PostCheckResult()

    tier = task_data.get("execution_tier", "")
    task_class_value = task_data.get("task_class", "")
    compression_ratio = (tokens_before - tokens_after) / tokens_before

    if tier in ("REASONING", "REASONING_SAFE"):
        max_compression = _TASK_COMPRESSION_LIMITS.get(task_class_value, 0.10)
        if compression_ratio > max_compression:
            return PostCheckResult(
                rollback=True,
                fail=True,
                code="PSG_REASONING_COMPRESSION_LIMIT",
                reason=(
                    f"Compression ratio {compression_ratio:.2f} exceeds "
                    f"{task_class_value} tier limit ({max_compression})"
                ),
            )

    if task_class_value == "debugging":
        max_compression = _TASK_COMPRESSION_LIMITS.get("debugging", 0.40)
        if compression_ratio > max_compression:
            return PostCheckResult(
                rollback=True,
                fail=True,
                code="PSG_DEBUGGING_COMPRESSION_LIMIT",
                reason=(
                    f"Debugging compression {compression_ratio:.2f} exceeds limit ({max_compression})"
                ),
            )
    return PostCheckResult()


def check_placeholder_leakage(name: str, text_before: str, text_after: str) -> PostCheckResult:
    if name in PLACEHOLDER_USING_TRANSFORMS or text_before == text_after:
        return PostCheckResult()
    from lattice.pipeline.guardrails import check_placeholder_leakage as _check

    decision = _check(text_before, text_after)
    if decision.action.value == "rollback":
        return PostCheckResult(
            rollback=True,
            fail=True,
            code="PSG_PLACEHOLDER_LEAKAGE",
            reason=f"Placeholder leakage: {decision.reason}",
        )
    return PostCheckResult()


def check_psg_preservation(
    name: str,
    request: Request,
    context: TransformContext,
    text_before: str,
    text_after: str,
) -> PostCheckResult:
    """Numeric loss + (for irreversible only) entity/format/signal preservation."""
    if text_before == text_after:
        return PostCheckResult()
    from lattice.pipeline.guardrails import (
        _check_numeric_preservation,
        check_critical_signal_loss,
        check_entity_preservation,
        check_format_preservation,
    )

    num_decision = _check_numeric_preservation(text_before, text_after)
    if num_decision.action.value == "rollback":
        return PostCheckResult(
            rollback=True,
            fail=True,
            code="PSG_NUMERIC_LOSS",
            reason=f"Numeric preservation failed: {num_decision.reason}",
        )

    if name not in IRREVERSIBLE_TRANSFORMS:
        return PostCheckResult()
    protected_spans = get_canonical_request_value(request, context, "_lattice_protected_spans", [])
    if not protected_spans:
        return PostCheckResult()

    for fn, code in (
        (check_entity_preservation, "PSG_ENTITY_LOSS"),
        (check_format_preservation, "PSG_FORMAT_LOSS"),
        (check_critical_signal_loss, "PSG_CRITICAL_SIGNAL_LOSS"),
    ):
        decision = fn(text_before, text_after)
        if decision.action.value == "rollback":
            return PostCheckResult(
                rollback=True,
                fail=True,
                code=code,
                reason=f"{code.split('_', 1)[1].replace('_', ' ').title()}: {decision.reason}",
            )
    return PostCheckResult()


def should_run_milv(
    name: str,
    text_before: str,
    text_after: str,
    tokens_before: int,
    has_task_classification: bool,
) -> bool:
    return (
        text_before != text_after
        and tokens_before > 0
        and name not in PLACEHOLDER_USING_TRANSFORMS
        and has_task_classification
    )


def check_aggregate_debugging_cap(
    request: Request,
    context: TransformContext,
    original_tokens: int,
    final_tokens: int,
) -> PostCheckResult:
    """Post-pipeline cap: aggregate debugging compression must not exceed 40%."""
    if original_tokens <= 0:
        return PostCheckResult()
    task_data = get_canonical_request_value(request, context, "_lattice_task_classification", {})
    tc = task_data.get("task_class", "") if isinstance(task_data, dict) else ""
    if tc != "debugging":
        return PostCheckResult()
    aggregate = (original_tokens - final_tokens) / original_tokens
    if aggregate > 0.40:
        return PostCheckResult(
            rollback=True,
            reason="aggregate_debugging_compression_exceeded",
            extra={"aggregate_compression": round(aggregate, 3)},
        )
    return PostCheckResult()
