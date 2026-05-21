"""DiagnosticOptimizer — wraps diagnostic_rle.

Purpose: logs, repeated errors, stack traces, debugging signal preservation.
Only runs when debug context is detected.
"""

from __future__ import annotations

import dataclasses
import re
import time
from typing import Any

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result, is_ok
from lattice.core.runtime_state import (
    get_canonical_state_value,
    get_ir_metadata_value,
    thaw_value,
)
from lattice.core.transport import Request, Response

# Import constituent transforms
try:
    from lattice.transforms.diagnostic_rle import DiagnosticRLE
except Exception:
    DiagnosticRLE = None  # type: ignore[misc,assignment]


@dataclasses.dataclass(slots=True)
class _Candidate:
    request: Request
    latency_ms: float
    tokens_before: int
    tokens_after: int
    transforms_used: list[str]
    quality_estimate: float = 1.0

    @property
    def score(self) -> float:
        savings = max(0, self.tokens_before - self.tokens_after)
        return (
            self.quality_estimate
            + (savings / 50.0)
            - (self.latency_ms / 100.0)
        )


class DiagnosticOptimizer(ReversibleSyncTransform):
    """Unified diagnostic signal preservation optimizer.

    Only activates when the request contains debugging/error signals.
    """

    name = "diagnostic_optimizer"
    priority = 17
    transform_class = ReversibleSyncTransform.transform_class  # LOSSLESS_SAFE

    def __init__(self) -> None:
        self._constituents: list[tuple[str, Any]] = []
        if DiagnosticRLE is not None:
            self._constituents.append(("diagnostic_rle", DiagnosticRLE()))

    def can_process(self, request: Request, context: TransformContext) -> bool:
        text = "\n".join(msg.content or "" for msg in request.messages)
        lowered = text.lower()
        debug_signals = [
            r"\berror\b",
            r"\btraceback\b",
            r"\bexception\b",
            r"\bfailed\b",
            r"\bcrash\b",
            r"\blog\b",
            r"\bstack\b",
        ]
        for pat in debug_signals:
            if re.search(pat, lowered):
                return True
        # Check task classification
        tc = thaw_value(get_ir_metadata_value(context, "_lattice_task_classification", {}))
        if not tc:
            tc = get_canonical_state_value(context, "_lattice_task_classification", {})
        if isinstance(tc, dict) and tc.get("task_class") == "debugging":
            return True
        return False

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        original = request.copy()
        original_tokens = original.token_estimate
        quality_floor = _get_quality_floor(context)

        candidates: list[_Candidate] = []

        # diagnostic_rle is primary — run first
        for t_name, t_instance in self._constituents:
            if not getattr(t_instance, "enabled", True):
                continue
            if not t_instance.can_process(original, context):
                continue
            start = time.perf_counter()
            result = t_instance.process(original.copy(), context)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if is_ok(result):
                candidate_req = result.unwrap()
                candidate = _Candidate(
                    request=candidate_req,
                    latency_ms=elapsed_ms,
                    tokens_before=original_tokens,
                    tokens_after=candidate_req.token_estimate,
                    transforms_used=[t_name],
                )
                if _validate_candidate(candidate, quality_floor, context, t_name):
                    candidates.append(candidate)
            else:
                context.record_metric(t_name, "optimizer_skipped", True)

        # Baseline
        baseline = _Candidate(
            request=original,
            latency_ms=0.0,
            tokens_before=original_tokens,
            tokens_after=original_tokens,
            transforms_used=[],
            quality_estimate=1.0,
        )
        candidates.append(baseline)

        if not candidates:
            return Ok(original)
        best = max(candidates, key=lambda c: c.score)

        if best.transforms_used:
            context.session_state[self.name] = {
                "transforms_used": best.transforms_used,
                "tokens_before": best.tokens_before,
                "tokens_after": best.tokens_after,
            }
            for t_name in best.transforms_used:
                context.mark_transform_applied(t_name)
                context.record_metric(t_name, "optimizer_applied", True)
            context.record_metric(
                self.name,
                "tokens_saved",
                best.tokens_before - best.tokens_after,
            )
            return Ok(best.request)

        return Ok(original)

    def reverse(self, response: Response, context: TransformContext) -> Response:
        state = context.session_state.get(self.name, {})
        transforms_used = state.get("transforms_used", [])
        for t_name in reversed(transforms_used):
            for name, instance in self._constituents:
                if name == t_name:
                    response = instance.reverse(response, context)
                    break
        return response


def _get_quality_floor(context: TransformContext) -> float:
    plan = thaw_value(get_ir_metadata_value(context, "_lattice_execution_plan"))
    if plan is not None:
        qf = getattr(plan, "quality_floor", None)
        if qf is None and isinstance(plan, dict):
            qf = plan.get("quality_floor")
        if qf is not None:
            return float(qf)
    tc = thaw_value(get_ir_metadata_value(context, "_lattice_task_classification", {}))
    if isinstance(tc, dict):
        return tc.get("quality_floor", 0.85)
    return 0.85


def _validate_candidate(
    candidate: _Candidate,
    quality_floor: float,
    context: TransformContext,
    label: str,
) -> bool:
    tokens_before = candidate.tokens_before
    tokens_after = candidate.tokens_after

    # Diagnostic optimizer should not expand tokens — compression only
    if tokens_after >= tokens_before:
        context.record_metric(label, "rejected_expansion", True)
        return False

    if candidate.quality_estimate < quality_floor:
        context.record_metric(label, "rejected_quality", True)
        return False

    return True
