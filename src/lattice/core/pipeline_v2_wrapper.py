"""PipelineV2 wrapper — adapts PipelineV2 into CompressorPipeline-compatible transform."""

from __future__ import annotations

from typing import Any

from lattice.core.context import TransformContext
from lattice.core.pipeline import ReversibleSyncTransform, TransformClass
from lattice.core.pipeline_v2 import PipelineV2, TransformRegistryV2
from lattice.core.result import Ok
from lattice.core.runtime_state import coerce_execution_plan, get_canonical_state_value
from lattice.core.task_classifier import TaskClass
from lattice.core.unified_planner import SemanticProfile, UnifiedPlanner, profile_from_legacy
from lattice.transport.types import Request, Response


class PipelineV2Wrapper(ReversibleSyncTransform):
    """Adapts PipelineV2 into CompressorPipeline-compatible transform.

    Reads ExecutionPlan from context (set by content_profiler when
    UnifiedPlanner is attached) and executes it verbatim via PipelineV2.
    """

    name = "pipeline_v2"
    priority = 19
    transform_class = TransformClass.LOSSLESS_SAFE
    enabled = True

    def __init__(self) -> None:
        self._pipeline_v2 = PipelineV2(registry=TransformRegistryV2())

    def process(self, request: Request, context: TransformContext) -> Any:
        context.session_state["_lattice_last_request"] = request.copy()
        plan = _coerce_execution_plan(get_canonical_state_value(context, "_lattice_execution_plan"))
        if plan is None:
            # Fallback: read canonical task/risk state and produce a plan
            profile = _build_profile_from_context(request, context)
            if profile is not None:
                planner = UnifiedPlanner()
                plan = planner.plan(request, profile)
            else:
                # No plan available — passthrough
                return Ok(request)

        context.session_state["_lattice_execution_plan"] = plan
        return self._pipeline_v2.process(request, plan, context)

    def reverse(self, response: Response, context: TransformContext) -> Response:
        plan = _coerce_execution_plan(get_canonical_state_value(context, "_lattice_execution_plan"))
        if plan is None:
            last_request = get_canonical_state_value(context, "_lattice_last_request")
            profile = _build_profile_from_context(last_request, context)
            if profile is not None:
                planner = UnifiedPlanner()
                if last_request is None:
                    return response
                plan = planner.plan(last_request, profile)
            else:
                return response

        if plan is None:
            return response

        return self._pipeline_v2.reverse(response, plan, context)


def _coerce_execution_plan(plan: Any) -> Any | None:
    """Accept both dict payloads and concrete plan objects."""
    return coerce_execution_plan(plan)


def _build_profile_from_context(
    request: Request | None, context: TransformContext
) -> SemanticProfile | None:
    """Build a SemanticProfile from canonical session state when possible."""
    if request is None:
        return None

    task_data = get_canonical_state_value(context, "_lattice_task_classification", {})
    if isinstance(task_data, dict) and task_data:
        task_cls = task_data.get("task_class", TaskClass.SIMPLE)
        if isinstance(task_cls, str):
            try:
                task_cls = TaskClass(task_cls)
            except ValueError:
                task_cls = TaskClass.SIMPLE

        risk_data = get_canonical_state_value(context, "_lattice_risk_score", {})
        risk_total = 0
        if isinstance(risk_data, dict):
            risk_total = int(risk_data.get("total", 0))

        return SemanticProfile(
            task_class=task_cls,
            task_label=task_data.get("preferred_strategy", ""),
            risk_total=risk_total,
            context_length=request.token_estimate,
            has_tool_calls=request.is_tool_conversation,
            is_streaming=request.stream,
            is_conservative=bool(task_data.get("is_conservative", False)),
            provider=context.provider or "generic",
            model=request.model,
        )

    legacy = get_canonical_state_value(context, "_lattice_schedule", {})
    return profile_from_legacy(legacy)


# NOTE: _lattice_schedule fallback above is a compatibility bridge. When
# content_profiler runs with the v2 pipeline (use_v2_pipeline=True), it
# stores _lattice_execution_plan in session state. The canonical path
# reads that plan directly; this legacy fallback only fires when the v2
# plan is absent, which happens only in compatibility mode.
