"""Runtime optimization contract transform.

Classifies the request and writes a per-request optimization contract into
metadata. The contract is consumed by ``OptimizationPolicy`` to skip transforms
whose latency/cost is not justified for the workload tier.
"""

from __future__ import annotations

from lattice.core.context import (
    METADATA_KEY_SCHEDULE,
    METADATA_KEY_TASK_CLASSIFICATION,
    TransformContext,
)
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.scheduler import decide_schedule
from lattice.core.task_classifier import ExecutionTier, TaskClass, TaskClassification
from lattice.core.transport import Request, Response
from lattice.runtime.router import RuntimeRouter
from lattice.utils.validation import SemanticRiskScore


class RuntimeContractTransform(ReversibleSyncTransform):
    """Attach a runtime optimization contract to each request.

    Also reconciles the scheduler when the RuntimeRouter tier differs
    from the task classifier tier set by content_profiler (priority 1).
    """

    name = "runtime_contract"
    priority = 2

    def __init__(self, router: RuntimeRouter | None = None) -> None:
        self.router = router or RuntimeRouter()

    def process(
        self,
        request: Request,
        context: TransformContext,
    ) -> Result[Request, TransformError]:
        decision = self.router.classify(request)
        request.metadata["_lattice_runtime"] = decision.to_dict()
        request.metadata["_lattice_runtime_contract"] = decision.contract
        context.record_metric(self.name, "tier_score", decision.score)
        context.record_metric(self.name, "confidence", decision.confidence)
        context.record_metric(
            self.name, "transform_budget_ms", decision.contract["max_transform_latency_ms"]
        )

        # Reconcile task classification if RuntimeRouter tier diverges from
        # the content_profiler's task_classifier. The RuntimeRouter is the
        # authoritative source for tier decisions because it includes
        # debugging/reasoning context awareness.
        existing_tc = request.metadata.get(METADATA_KEY_TASK_CLASSIFICATION)
        if existing_tc and isinstance(existing_tc, dict):
            classifier_tier = existing_tc.get("execution_tier", "")
            if classifier_tier != decision.tier:
                # Update task classification to match RuntimeRouter tier
                tier_map: dict[str, tuple[TaskClass, ExecutionTier]] = {
                    "SIMPLE": (TaskClass.SIMPLE, ExecutionTier.SIMPLE),
                    "MEDIUM": (TaskClass.ANALYSIS, ExecutionTier.MEDIUM),
                    "COMPLEX": (TaskClass.STRUCTURED, ExecutionTier.COMPLEX),
                    "REASONING": (TaskClass.REASONING, ExecutionTier.REASONING),
                    "REASONING_SAFE": (TaskClass.REASONING, ExecutionTier.REASONING_SAFE),
                }
                new_tc_tuple = tier_map.get(decision.tier)
                if new_tc_tuple:
                    new_tc = TaskClassification(
                        task_class=new_tc_tuple[0],
                        execution_tier=new_tc_tuple[1],
                        score=decision.score,
                        confidence=decision.confidence,
                        hard_override=True,
                        preferred_strategy=decision.contract.get("preferred_strategy", "balanced"),
                    )
                    request.metadata[METADATA_KEY_TASK_CLASSIFICATION] = new_tc.to_dict()

                    # Rebuild schedule with corrected tier
                    risk_data = request.metadata.get("_lattice_risk_score")
                    risk = None
                    if risk_data and isinstance(risk_data, dict):
                        risk = SemanticRiskScore(
                            strict_instructions=float(risk_data.get("strict_instructions", 0)),
                            sensitive_domain=float(risk_data.get("sensitive_domain", 0)),
                            structured_output=float(risk_data.get("structured_output", 0)),
                            high_stakes_entities=float(risk_data.get("high_stakes_entities", 0)),
                            reasoning_heavy=float(risk_data.get("reasoning_heavy", 0)),
                            intentional_repetition=float(
                                risk_data.get("intentional_repetition", 0)
                            ),
                            tool_call_dependency=float(risk_data.get("tool_call_dependency", 0)),
                            formatting_constraints=float(
                                risk_data.get("formatting_constraints", 0)
                            ),
                        )

                    sig_summary = request.metadata.get("_lattice_sig_summary", {})
                    protected_count = (
                        sig_summary.get("protected_count", 0)
                        if isinstance(sig_summary, dict)
                        else 0
                    )

                    schedule = decide_schedule(
                        transform_names=decision.contract.get("skipped_transforms", []),
                        task=new_tc,
                        risk=risk,
                        protected_span_count=protected_count,
                        total_budget_ms=new_tc.budget_ms,
                    )
                    request.metadata[METADATA_KEY_SCHEDULE] = schedule.to_dict()
                    context.record_metric(self.name, "schedule_reconciled", decision.tier)

        return Ok(request)

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        return response


__all__ = ["RuntimeContractTransform"]
