"""ToolOptimizer — merged tool_projection and tool_filter.

Phase 3 — Collapse overlapping transforms.

Purpose: safe tool-output schema projection and cleanup.
Runs constituent transforms, scores candidates, applies the best.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result, is_ok, unwrap
from lattice.core.runtime_state import get_ir_metadata_value, thaw_value
from lattice.optimizer._dispatch import run_constituent
from lattice.transport.types import Request, Response

# Import constituent transforms
try:
    from lattice.transforms.tool_projection import QueryAwareProjection
except Exception:
    QueryAwareProjection = None  # type: ignore[misc,assignment]

try:
    from lattice.transforms.tool_filter import ToolOutputFilter
except Exception:
    ToolOutputFilter = None  # type: ignore[misc,assignment]


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
        return self.quality_estimate + (savings / 50.0) - (self.latency_ms / 100.0)


class ToolOptimizer(ReversibleSyncTransform):
    """Unified tool output projection and cleanup optimizer."""

    name = "tool_optimizer"
    priority = 30
    transform_class = ReversibleSyncTransform.transform_class  # LOSSLESS_SAFE

    def __init__(self) -> None:
        self._constituents: list[tuple[str, Any]] = []
        if ToolOutputFilter is not None:
            self._constituents.append(("tool_filter", ToolOutputFilter()))
        if QueryAwareProjection is not None:
            self._constituents.append(("tool_projection", QueryAwareProjection()))

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        original = request.copy()
        original_tokens = original.token_estimate
        quality_floor = _get_quality_floor(context)

        candidates: list[_Candidate] = []

        # Run each constituent independently
        for t_name, t_instance in self._constituents:
            if not getattr(t_instance, "enabled", True):
                continue
            if not t_instance.can_process(original, context):
                continue
            start = time.perf_counter()
            result = run_constituent(t_name, t_instance, original.copy(), context)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if is_ok(result):
                candidate_req = unwrap(result)
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

        # Try tool_filter + tool_projection combo (common)
        tf_cand = next((c for c in candidates if "tool_filter" in c.transforms_used), None)
        tp_cand = next((c for c in candidates if "tool_projection" in c.transforms_used), None)
        if tf_cand and tp_cand:
            combo = self._try_combo(original, context, [tf_cand, tp_cand], quality_floor)
            if combo:
                candidates.append(combo)

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

    def _try_combo(
        self,
        original: Request,
        context: TransformContext,
        candidates: list[_Candidate],
        quality_floor: float,
    ) -> _Candidate | None:
        working = original.copy()
        transforms_used: list[str] = []
        start = time.perf_counter()
        for cand in candidates:
            for t_name, t_inst in self._constituents:
                if t_name in cand.transforms_used:
                    result = run_constituent(t_name, t_inst, working, context)
                    if is_ok(result):
                        working = unwrap(result)
                        transforms_used.append(t_name)
                    else:
                        return None
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        combo = _Candidate(
            request=working,
            latency_ms=elapsed_ms,
            tokens_before=original.token_estimate,
            tokens_after=working.token_estimate,
            transforms_used=transforms_used,
        )
        if _validate_candidate(combo, quality_floor, context, "tool_combo"):
            return combo
        return None


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

    if tokens_after >= tokens_before:
        context.record_metric(label, "rejected_expansion", True)
        return False

    if candidate.quality_estimate < quality_floor:
        context.record_metric(label, "rejected_quality", True)
        return False

    return True
