"""Runtime optimization contract transform.

Sets per-request optimization limits that the scheduler consumes.
Does NOT veto transforms — the scheduler is the decision maker.
"""

from __future__ import annotations

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform, TransformClass
from lattice.core.primitives import PromptIRV2
from lattice.core.result import Ok, Result
from lattice.core.transport import Request, Response
from lattice.runtime.router import RuntimeRouter


class RuntimeContractTransform(ReversibleSyncTransform):
    """Attach a runtime optimization contract to each request.

    Returns limits (max_tokens, allow_lossy, preserve_entities, max_latency)
    that the scheduler reads to gate transforms. Does NOT skip transforms
    directly — that is the scheduler's job.
    """

    name = "runtime_contract"
    transform_class = TransformClass.OBSERVABILITY_ONLY
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
        return Ok(request)

    def optimize(
        self,
        ir: PromptIRV2,
        request: Request,
        context: TransformContext,
    ) -> Result[PromptIRV2, TransformError]:
        """Attach runtime contract metadata to immutable IR."""
        decision = self.router.classify(request)
        updated = ir.add_metadata(
            _lattice_runtime=decision.to_dict(),
            _lattice_runtime_contract=decision.contract,
        )
        request.metadata["_lattice_runtime"] = decision.to_dict()
        request.metadata["_lattice_runtime_contract"] = decision.contract
        context.session_state["_lattice_ir_v2"] = updated
        context.record_metric(self.name, "tier_score", decision.score)
        context.record_metric(self.name, "confidence", decision.confidence)
        context.record_metric(
            self.name, "transform_budget_ms", decision.contract["max_transform_latency_ms"]
        )
        return Ok(updated)

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        return response


__all__ = ["RuntimeContractTransform"]
