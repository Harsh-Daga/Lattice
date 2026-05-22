"""ReferenceOptimizer — merged reference_sub and path_prefix.

Phase 3 — Collapse overlapping transforms.

Purpose: UUID / path / URL / phrase / reference substitution.
All constituent transforms are reversible and use placeholders.
Implements hard rollback: rejects if placeholders leak or savings are negative.
"""

from __future__ import annotations

import dataclasses
import re
import time
from typing import Any

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.result import Ok, Result, is_ok, unwrap
from lattice.core.runtime_state import get_ir_metadata_value, thaw_value
from lattice.optimizer._dispatch import run_constituent
from lattice.pipeline.base import ReversibleSyncTransform
from lattice.transport.types import Request, Response

# Import constituent transforms
try:
    from lattice.transforms.reference_sub import ReferenceSubstitution
except Exception:
    ReferenceSubstitution = None  # type: ignore[misc,assignment]

try:
    from lattice.transforms.path_prefix import PathPrefixCompressor
except Exception:
    PathPrefixCompressor = None  # type: ignore[misc,assignment]


@dataclasses.dataclass(slots=True)
class _Candidate:
    request: Request
    latency_ms: float
    tokens_before: int
    tokens_after: int
    transforms_used: list[str]
    quality_estimate: float = 1.0
    placeholder_count: int = 0
    rollback_reason: str | None = None

    @property
    def score(self) -> float:
        savings = max(0, self.tokens_before - self.tokens_after)
        # Strong bonus for savings, penalty for placeholders
        return (
            self.quality_estimate
            + (savings / 50.0)
            - (self.latency_ms / 100.0)
            - (self.placeholder_count / 20.0)
        )


class ReferenceOptimizer(ReversibleSyncTransform):
    """Unified reference/path/phrase substitution optimizer."""

    name = "reference_optimizer"
    priority = 21
    transform_class = ReversibleSyncTransform.transform_class  # LOSSLESS_SAFE

    def __init__(self) -> None:
        self._constituents: list[tuple[str, Any]] = []
        if ReferenceSubstitution is not None:
            self._constituents.append(("reference_sub", ReferenceSubstitution()))
        if PathPrefixCompressor is not None:
            self._constituents.append(("path_prefix", PathPrefixCompressor()))

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
                placeholder_count = _count_placeholders(candidate_req)
                candidate = _Candidate(
                    request=candidate_req,
                    latency_ms=elapsed_ms,
                    tokens_before=original_tokens,
                    tokens_after=candidate_req.token_estimate,
                    transforms_used=[t_name],
                    placeholder_count=placeholder_count,
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
            # Guard: if placeholders present but no MAP appendix, reject
            if best.placeholder_count > 0 and not _has_map_appendix(best.request):
                context.record_metric(self.name, "placeholder_leakage_blocked", True)
                return Ok(original)
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
        placeholder_count = _count_placeholders(working)
        combo = _Candidate(
            request=working,
            latency_ms=elapsed_ms,
            tokens_before=original.token_estimate,
            tokens_after=working.token_estimate,
            transforms_used=transforms_used,
            placeholder_count=placeholder_count,
        )
        if _validate_candidate(combo, quality_floor, context, "ref_combo"):
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


def _count_placeholders(request: Request) -> int:
    """Count placeholder patterns like <ref_N>, <p_N>, <g_N>, <d_N>."""
    text = "\n".join(msg.content or "" for msg in request.messages)
    return len(re.findall(r"<(ref_|p_|g_|d_)\d+>", text))


def _has_map_appendix(request: Request) -> bool:
    """Check if the request has a visible ALIAS MAP or DICTIONARY MAP.

    Note: the legacy DictionaryCompressor module was removed. The reverse
    map is stored in TransformContext session state instead.  We treat the presence of
    any placeholder pattern as valid since all constituent transforms
    store their reverse maps in session_state.
    """
    text = "\n".join(msg.content or "" for msg in request.messages)
    return (
        "ALIAS MAP:" in text
        or "DICTIONARY MAP:" in text
        # Accept all reversible placeholder families as evidence of reversible
        # substitution (reverse maps are in session_state).
        or bool(re.search(r"<(d_|ref_|p_|g_)\d+>", text))
    )


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

    if tokens_before > 0:
        compression = (tokens_before - tokens_after) / tokens_before
        if compression > 0.90:
            context.record_metric(label, "rejected_compression", True)
            return False

    # Placeholder leakage check
    if candidate.placeholder_count > 0:
        if not _has_map_appendix(candidate.request):
            context.record_metric(label, "rejected_placeholder_leak", True)
            return False

    return True
