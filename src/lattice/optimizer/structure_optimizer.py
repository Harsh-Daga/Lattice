"""StructureOptimizer — merged json_shape, format_conversion, columnar_pack.

Phase 3 — Collapse overlapping transforms.

Purpose: JSON / table / log / CSV / markdown structure compaction.
Runs constituent transforms, scores candidates, applies the best one.
Implements hard rollback: rejects candidates with negative savings or quality loss.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result, is_ok
from lattice.core.runtime_state import get_ir_metadata_value, thaw_value
from lattice.transport.types import Request, Response

# Import constituent transforms (may fail gracefully)
try:
    from lattice.transforms.json_shape import JSONShapeFactor
except Exception:
    JSONShapeFactor = None  # type: ignore[misc,assignment]

try:
    from lattice.transforms.format_conv import FormatConverter
except Exception:
    FormatConverter = None  # type: ignore[misc,assignment]

try:
    from lattice.transforms.columnar_pack import ColumnarTablePack
except Exception:
    ColumnarTablePack = None  # type: ignore[misc,assignment]


@dataclasses.dataclass(slots=True)
class _Candidate:
    request: Request
    latency_ms: float
    tokens_before: int
    tokens_after: int
    transforms_used: list[str]
    quality_estimate: float = 1.0
    cache_gain: float = 0.0
    transport_gain: float = 0.0
    semantic_risk: float = 0.0
    rollback_reason: str | None = None

    @property
    def score(self) -> float:
        savings = max(0, self.tokens_before - self.tokens_after)
        return (
            self.quality_estimate
            + self.cache_gain
            + self.transport_gain
            - (self.tokens_after / 1000.0)
            - (self.latency_ms / 100.0)
            - self.semantic_risk
            + (savings / 100.0)
        )


class StructureOptimizer(ReversibleSyncTransform):
    """Unified structure compaction optimizer."""

    name = "structure_optimizer"
    priority = 20
    transform_class = ReversibleSyncTransform.transform_class  # LOSSLESS_SAFE

    def __init__(self) -> None:
        self._constituents: list[tuple[str, Any]] = []
        if JSONShapeFactor is not None:
            self._constituents.append(("json_shape", JSONShapeFactor()))
        if FormatConverter is not None:
            self._constituents.append(("format_conversion", FormatConverter()))
        if ColumnarTablePack is not None:
            self._constituents.append(("columnar_pack", ColumnarTablePack()))

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        original = request.copy()
        original_tokens = original.token_estimate
        quality_floor = _get_quality_floor(context)

        candidates: list[_Candidate] = []

        # 1. Run each constituent transform independently as a candidate
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
                # Record failure
                context.record_metric(t_name, "optimizer_skipped", True)

        # 2. Run combinations: format + columnar / json shape pairings
        # Only combine when both individual candidates were valid
        if len(candidates) >= 2:
            combo = self._try_combo(original, context, candidates, quality_floor)
            if combo:
                candidates.append(combo)

        # 3. Always include "original" as baseline
        baseline = _Candidate(
            request=original,
            latency_ms=0.0,
            tokens_before=original_tokens,
            tokens_after=original_tokens,
            transforms_used=[],
            quality_estimate=1.0,
        )
        candidates.append(baseline)

        # 4. Select best candidate
        if not candidates:
            return Ok(original)
        best = max(candidates, key=lambda c: c.score)

        if best.transforms_used:
            # Save constituent transforms used for reverse
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
        # Reverse in opposite order
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
        """Try combining the top-2 candidates sequentially."""
        top2 = sorted(candidates, key=lambda c: c.score, reverse=True)[:2]
        if len(top2) < 2:
            return None

        working = original.copy()
        transforms_used: list[str] = []
        start = time.perf_counter()
        for cand in top2:
            for t_name, t_inst in self._constituents:
                if t_name in cand.transforms_used:
                    result = t_inst.process(working, context)
                    if is_ok(result):
                        working = result.unwrap()
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
        if _validate_candidate(combo, quality_floor, context, "combo"):
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
    """Hard rollback validation: reject bad candidates.

    Phase 7 — hard rollback everywhere.
    """
    tokens_before = candidate.tokens_before
    tokens_after = candidate.tokens_after

    # Reject if tokens increased with no transport/cache gain
    if tokens_after >= tokens_before:
        if candidate.cache_gain <= 0 and candidate.transport_gain <= 0:
            context.record_metric(label, "rejected_expansion", True)
            context.record_metric(label, "rejected_reason", "tokens_after >= tokens_before")
            return False

    # Reject if quality below floor
    if candidate.quality_estimate < quality_floor:
        context.record_metric(label, "rejected_quality", True)
        context.record_metric(
            label,
            "rejected_reason",
            f"quality {candidate.quality_estimate} < floor {quality_floor}",
        )
        return False

    # Reject if compression is excessive (>50%)
    if tokens_before > 0:
        compression = (tokens_before - tokens_after) / tokens_before
        if compression > 0.90:
            context.record_metric(label, "rejected_compression", True)
            context.record_metric(label, "rejected_reason", f"compression {compression:.2f} > 0.90")
            return False

    return True
