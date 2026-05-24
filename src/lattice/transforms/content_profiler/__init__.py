"""Content profiler — priority 1 transform that runs FIRST and populates IR metadata.

Splits the historic monolith into focused submodules. The public class
ContentProfiler still satisfies ReversibleSyncTransform.
"""

from __future__ import annotations

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.result import Ok, Result
from lattice.ir.primitives import PromptIRV2
from lattice.pipeline.base import ReversibleSyncTransform, TransformClass
from lattice.planner.runtime_state import get_canonical_state_value
from lattice.transforms.content_profiler.classifier import (
    ClassifierConfig,
    ContentProfile,
    classify_by_signals,
    select_compression_strategy,
)
from lattice.transforms.content_profiler.planner_bridge import (
    _build_importance_graph,
    apply_content_profiling,
    build_importance_graph,
    coerce_execution_plan,
    derive_optimizer_schedule_from_plan,
    derive_schedule_from_plan,
)
from lattice.transforms.content_profiler.risk_scorer import (
    SemanticRiskScore,
    compute_risk_score,
    score_request_risk,
)
from lattice.transforms.content_profiler.task_classifier_bridge import (
    TaskClassification,
    bridge_task_classification,
    classify_task,
)
from lattice.transport.types import Request, Response


class ContentProfiler(ReversibleSyncTransform):
    """Profile request content and recommend compression strategy."""

    name = "content_profiler"
    transform_class = TransformClass.OBSERVABILITY_ONLY
    priority = 1

    def __init__(
        self,
        enable_adaptive: bool = True,
        short_threshold_tokens: int = 50,
        code_block_weight: float = 3.0,
        table_row_weight: float = 2.0,
        narrative_length_weight: float = 1.0,
    ) -> None:
        self.enable_adaptive = enable_adaptive
        self.short_threshold_tokens = short_threshold_tokens
        self.code_block_weight = code_block_weight
        self.table_row_weight = table_row_weight
        self.narrative_length_weight = narrative_length_weight

    @property
    def _classifier_config(self) -> ClassifierConfig:
        return ClassifierConfig(
            short_threshold_tokens=self.short_threshold_tokens,
            code_block_weight=self.code_block_weight,
            table_row_weight=self.table_row_weight,
            narrative_length_weight=self.narrative_length_weight,
        )

    def _run_profiling(
        self,
        request: Request,
        context: TransformContext,
    ) -> PromptIRV2 | None:
        """Shared metadata + IR seeding used by process() and optimize()."""
        if not self.enable_adaptive:
            return None

        profile = classify_by_signals(request, self._classifier_config)
        task = bridge_task_classification(request)
        strategy = select_compression_strategy(profile, task)
        risk_score = score_request_risk(request)
        return apply_content_profiling(
            self,
            request,
            context,
            profile=profile,
            task=task,
            strategy=strategy,
            risk_score=risk_score,
        )

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        """Profile request content, compute risk, build SIG, and set strategy."""
        if not self.enable_adaptive:
            return Ok(request)
        self._run_profiling(request, context)
        return Ok(request)

    def optimize(
        self,
        ir: PromptIRV2,
        request: Request,
        context: TransformContext,
    ) -> Result[PromptIRV2, TransformError]:
        """Populate metadata and return canonical IR from context."""
        if not self.enable_adaptive:
            return Ok(ir)
        ir_v2 = self._run_profiling(request, context)
        if ir_v2 is None:
            return Ok(ir)
        stored = get_canonical_state_value(context, "_lattice_ir_v2")
        return Ok(stored if stored is not None else ir_v2)

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        """No-op."""
        return response


__all__ = [
    "ContentProfiler",
    "ContentProfile",
    "ClassifierConfig",
    "SemanticRiskScore",
    "TaskClassification",
    "_build_importance_graph",
    "apply_content_profiling",
    "bridge_task_classification",
    "build_importance_graph",
    "classify_by_signals",
    "classify_task",
    "coerce_execution_plan",
    "compute_risk_score",
    "derive_optimizer_schedule_from_plan",
    "derive_schedule_from_plan",
    "score_request_risk",
    "select_compression_strategy",
]
