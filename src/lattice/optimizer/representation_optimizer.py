"""RepresentationOptimizer — global optimizer with true state-space beam search.

Phase 1 cutover — true beam search.

Algorithm:
    beam = [baseline]
    for optimizer in allowed:
        expanded = []
        for candidate in beam:
            expanded.append(candidate)                       # don't apply
            result = optimizer(candidate.request.copy())       # apply
            if valid:
                expanded.append(new_candidate)
        beam = top_k(expanded)

This means:
    reference_optimizer sees structure-optimized data
    structure_optimizer sees reference-optimized data
    combinations emerge naturally from the search graph

Core scoring:
    score = quality_estimate + cache_gain + transport_gain
            - token_cost/1000 - latency_cost/100 - semantic_risk
"""

from __future__ import annotations

from typing import Any

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result, is_ok
from lattice.core.runtime_state import get_canonical_state_value, thaw_value
from lattice.core.transport import Request, Response
from lattice.ir.primitives import PromptIRV2, prompt_ir_v2_from_legacy
from lattice.ir.transform import (
    CandidateSearch,
    IRTransform,
    LegacyRequestTransformAdapter,
)
from lattice.optimizer import _OPTIMIZER_CLASSES


class RepresentationOptimizer(ReversibleSyncTransform):
    """Global immutable candidate-graph optimizer.

    Applies the best combination of constituent optimizers via the shared
    CandidateSearch engine. The optimizer now branches on immutable PromptIRV2
    candidates instead of mutating request state in place.
    """

    name = "representation_optimizer"
    priority = 19
    transform_class = ReversibleSyncTransform.transform_class

    def __init__(
        self,
        beam_width: int = 5,
        max_candidates: int = 12,
    ) -> None:
        self.beam_width = beam_width
        self.max_candidates = max_candidates
        self._optimizer_instances: dict[str, Any] = {}

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        original = request.copy()
        original_tokens = original.token_estimate
        quality_floor = _get_quality_floor(context)
        budget_ms = _get_budget_ms(context)

        allowed = _get_allowed_optimizers(context)
        if not allowed:
            return Ok(original)

        optimizer_transforms: list[IRTransform] = []
        for name in allowed:
            inst = self._optimizer_instances.get(name)
            if inst is None:
                cls = _OPTIMIZER_CLASSES.get(name)
                if cls is None:
                    continue
                inst = cls()
                self._optimizer_instances[name] = inst
            optimizer_transforms.append(LegacyRequestTransformAdapter(inst))

        if not optimizer_transforms:
            return Ok(original)

        initial_ir = _get_initial_ir(original, context)
        if initial_ir is None:
            return Ok(original)

        search = CandidateSearch(
            transforms=optimizer_transforms,
            beam_width=self.beam_width,
            max_depth=max(1, min(self.max_candidates, len(optimizer_transforms))),
        )

        try:
            import time

            search_start = time.perf_counter()
            best = search.search(
                initial_ir,
                request=original,
                quality_floor=quality_floor,
                budget_ms=budget_ms,
                context=context,
            )
            search_ms = (time.perf_counter() - search_start) * 1000.0
        except Exception:
            return Ok(original)

        if not best.applied:
            return Ok(original)

        replay_start = time.perf_counter()
        modified = original.copy()
        replayed: list[str] = []
        for opt_name in best.applied:
            inst = self._optimizer_instances.get(opt_name)
            if inst is None:
                continue
            try:
                result = inst.process(modified, context)
            except Exception:
                context.record_metric(opt_name, "replay_error", True)
                continue
            if is_ok(result):
                replayed_request = result.unwrap()
                if replayed_request is not None:
                    modified = replayed_request
                replayed.append(opt_name)
                continue
            context.record_metric(opt_name, "replay_error", True)

        replay_ms = (time.perf_counter() - replay_start) * 1000.0

        canonical_ir = initial_ir or _compile_request_to_ir_v2(modified, context)
        if canonical_ir is not None:
            modified.metadata.setdefault("_lattice_ir_v2", canonical_ir)
            context.session_state["_lattice_ir_v2"] = canonical_ir

        context.session_state[self.name] = {
            "optimizers_applied": replayed,
            "tokens_before": original_tokens,
            "tokens_after": modified.token_estimate,
            "beam_candidates": len(optimizer_transforms),
        }
        context.record_metric(self.name, "tokens_saved", original_tokens - modified.token_estimate)
        context.record_metric(self.name, "beam_candidates", len(optimizer_transforms))
        context.record_metric(self.name, "search_latency_ms", round(search_ms, 3))
        context.record_metric(self.name, "replay_latency_ms", round(replay_ms, 3))
        context.record_metric(
            self.name,
            "latency_ms",
            round(search_ms + replay_ms, 3),
        )
        context.record_metric(self.name, "immutable_candidate_graph", True)

        for opt_name in replayed:
            context.mark_transform_applied(opt_name)
            context.record_metric(opt_name, "optimizer_applied", True)

        return Ok(modified)

    def reverse(self, response: Response, context: TransformContext) -> Response:
        state = context.session_state.get(self.name, {})
        optimizers_applied = state.get("optimizers_applied", [])
        for opt_name in reversed(optimizers_applied):
            inst = self._optimizer_instances.get(opt_name)
            if inst is not None:
                response = inst.reverse(response, context)
        return response


def _get_initial_ir(request: Request, context: TransformContext) -> PromptIRV2 | None:
    """Load canonical PromptIRV2 from request metadata, session, or compiler."""
    ir_v2 = get_canonical_state_value(context, "_lattice_ir_v2")

    if isinstance(ir_v2, PromptIRV2):
        return ir_v2
    if isinstance(ir_v2, dict):
        try:
            return PromptIRV2.from_dict(ir_v2)
        except Exception:
            return None

    try:
        from lattice.ir.builder import build_ir
        from lattice.ir.normalizer import normalize_ir

        legacy_ir = normalize_ir(build_ir(request))
        ir_v2 = prompt_ir_v2_from_legacy(legacy_ir)
        context.session_state["_lattice_ir_v2"] = ir_v2
        return ir_v2
    except Exception:
        return None


def _compile_request_to_ir_v2(
    request: Request,
    context: TransformContext,
) -> PromptIRV2 | None:
    """Compile a live request back into canonical PromptIRV2."""
    try:
        from lattice.ir.builder import build_ir
        from lattice.ir.normalizer import normalize_ir

        legacy_ir = normalize_ir(build_ir(request))
        return prompt_ir_v2_from_legacy(legacy_ir)
    except Exception:
        return None


def _get_allowed_optimizers(context: TransformContext) -> list[str]:
    """Read allowed optimizers from multiple sources in priority order."""
    ir_meta = _ir_v2_metadata(context)
    if ir_meta:
        opt_sched = ir_meta.get("_lattice_optimizer_schedule")
        if isinstance(opt_sched, dict):
            allowed = opt_sched.get("allowed_optimizers", [])
            if allowed:
                return list(allowed)

        plan = ir_meta.get("_lattice_execution_plan")
        if isinstance(plan, dict):
            allowed = plan.get("allowed_optimizers", [])
            if allowed:
                return list(allowed)

        seg_summary = ir_meta.get("_lattice_segment_summary")
        if isinstance(seg_summary, dict):
            seg_types = seg_summary.get("segment_types", [])
            if seg_types:
                selected = _select_optimizers_from_segments(seg_types)
                if selected:
                    return selected

    # 1. OptimizerSchedule (highest priority for Phase 1 cutover)
    opt_sched = get_canonical_state_value(context, "_lattice_optimizer_schedule")
    if opt_sched is not None:
        allowed = getattr(opt_sched, "allowed_optimizers", None)
        if allowed:
            return list(allowed)

    # 2. ExecutionPlan
    plan = get_canonical_state_value(context, "_lattice_execution_plan")
    if plan is not None:
        allowed = getattr(plan, "allowed_optimizers", None)
        if allowed:
            return list(allowed)

    # 3. Phase 2: semantic segment-based selection
    seg_summary = thaw_value(get_canonical_state_value(context, "_lattice_segment_summary"))
    if seg_summary is not None:
        seg_types = seg_summary.get("segment_types", [])
        if seg_types:
            selected = _select_optimizers_from_segments(seg_types)
            if selected:
                return selected

    # 4. Legacy scheduler decision
    sched = thaw_value(get_canonical_state_value(context, "_lattice_schedule", {}))
    if isinstance(sched, dict):
        allowed = sched.get("allowed_optimizers", [])
        if allowed:
            return list(allowed)

    # 5. Fallback: content_profile
    profile = get_canonical_state_value(context, "_lattice_profile")
    profile_to_optimizers: dict[str, list[str]] = {
        "table_heavy":       ["structure_optimizer", "reference_optimizer"],
        "tool_output":         ["tool_optimizer", "reference_optimizer"],
        "code_heavy":        ["reference_optimizer", "structure_optimizer"],
        "log_output":        ["diagnostic_optimizer", "reference_optimizer"],
        "diff_output":       ["reference_optimizer"],
        "stack_trace":       ["reference_optimizer", "diagnostic_optimizer"],
        "grep_output":       ["structure_optimizer", "reference_optimizer"],
        "file_tree":         ["reference_optimizer"],
        "mcp_output":        ["tool_optimizer", "reference_optimizer"],
        "narrative_long":    ["context_optimizer", "reference_optimizer"],
    }
    if profile and profile in profile_to_optimizers:
        return profile_to_optimizers[profile]

    # Default production optimizers
    from lattice.optimizer import PRODUCTION_OPTIMIZERS
    return list(PRODUCTION_OPTIMIZERS)


_SEG_OPTIMIZER_MAP: dict[str, list[str]] = {
    "code":       ["structure_optimizer", "reference_optimizer"],
    "json":       ["structure_optimizer", "ir_structure_optimizer", "reference_optimizer"],
    "table":      ["structure_optimizer", "ir_structure_optimizer", "reference_optimizer"],
    "log":        ["diagnostic_optimizer", "reference_optimizer"],
    "tool_output": ["tool_optimizer", "ir_structure_optimizer", "reference_optimizer"],
    "reasoning":  ["structure_optimizer", "reference_optimizer"],
    "narrative":  ["context_optimizer", "reference_optimizer"],
    "instructions": ["reference_optimizer"],
    "short":       [],
}


def _select_optimizers_from_segments(segment_types: list[str]) -> list[str] | None:
    """Map segment types to optimizer recommendations and return unique list."""
    selected: set[str] = set()
    for st in segment_types:
        optimizers = _SEG_OPTIMIZER_MAP.get(st, [])
        selected.update(optimizers)
    if not selected:
        return None
    return sorted(selected)


def _get_quality_floor(context: TransformContext) -> float:
    """Read quality floor from OptimizerSchedule first, then ExecutionPlan, then legacy paths."""
    ir_meta = _ir_v2_metadata(context)
    if ir_meta:
        opt_sched = ir_meta.get("_lattice_optimizer_schedule")
        if isinstance(opt_sched, dict):
            qf = opt_sched.get("quality_floor")
            if qf is not None:
                return float(qf)
        plan = ir_meta.get("_lattice_execution_plan")
        if isinstance(plan, dict):
            qf = plan.get("quality_floor")
            if qf is not None:
                return float(qf)

    opt_sched = get_canonical_state_value(context, "_lattice_optimizer_schedule")
    if opt_sched is not None:
        qf = getattr(opt_sched, "quality_floor", None)
        if qf is not None:
            return float(qf)

    plan = get_canonical_state_value(context, "_lattice_execution_plan")
    if plan is not None:
        qf = getattr(plan, "quality_floor", None)
        if qf is not None:
            return float(qf)
    tc = get_canonical_state_value(context, "_lattice_task_classification", {})
    if isinstance(tc, dict):
        return tc.get("quality_floor", 0.85)
    return 0.85


def _get_budget_ms(context: TransformContext) -> float:
    """Read latency budget from OptimizerSchedule first, then ExecutionPlan, then legacy paths."""
    ir_meta = _ir_v2_metadata(context)
    if ir_meta:
        opt_sched = ir_meta.get("_lattice_optimizer_schedule")
        if isinstance(opt_sched, dict):
            budget = opt_sched.get("latency_budget_ms")
            if budget is not None:
                return float(budget)
        plan = ir_meta.get("_lattice_execution_plan")
        if isinstance(plan, dict):
            budget = plan.get("latency_budget_ms")
            if budget is not None:
                return float(budget)

    opt_sched = get_canonical_state_value(context, "_lattice_optimizer_schedule")
    if opt_sched is not None:
        budget = getattr(opt_sched, "latency_budget_ms", None)
        if budget is not None:
            return float(budget)

    plan = get_canonical_state_value(context, "_lattice_execution_plan")
    if plan is not None:
        budget = getattr(plan, "latency_budget_ms", None)
        if budget is not None:
            return float(budget)
    tc = get_canonical_state_value(context, "_lattice_task_classification", {})
    if isinstance(tc, dict):
        return tc.get("budget_ms", 100.0)
    return 100.0


def _validate_beam_candidate(
    candidate: Any,
    quality_floor: float,
    context: TransformContext,
) -> bool:
    """Compatibility validation for legacy tests and callers.

    The new production path uses CandidateSearch + CandidateScorer. This helper
    remains as a lightweight compatibility gate for tests and older code paths.
    """
    tokens_before = candidate.tokens_before
    tokens_after = candidate.tokens_after
    quality_estimate = candidate.quality_estimate
    cache_gain = candidate.cache_gain
    transport_gain = candidate.transport_gain
    transport_penalty = getattr(candidate, "transport_penalty", 0.0)

    if quality_estimate < quality_floor:
        return False

    if tokens_after > tokens_before:
        if cache_gain <= 0 and transport_gain <= 0:
            return False
        expansion = (tokens_after - tokens_before) / max(1, tokens_before)
        transport_penalty += expansion * 1.5

    if tokens_before > 0:
        compression = (tokens_before - tokens_after) / tokens_before
        if compression > 0.70:
            overage = compression - 0.70
            transport_penalty += overage * overage * 5.0

    if hasattr(candidate, "transport_penalty"):
        candidate.transport_penalty = transport_penalty
    return True


def _ir_v2_metadata(context: TransformContext) -> dict[str, Any]:
    """Extract canonical metadata from the v2 IR when available."""
    ir_v2 = get_canonical_state_value(context, "_lattice_ir_v2")
    metadata = getattr(ir_v2, "metadata", None)
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    try:
        return dict(metadata)
    except Exception:
        return {}
