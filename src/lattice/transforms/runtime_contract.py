"""Runtime optimization contract transform.

Classifies the request and writes a per-request optimization contract into
metadata. The contract is consumed by ``OptimizationPolicy`` to skip transforms
whose latency/cost is not justified for the workload tier.
"""

from __future__ import annotations

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.transport import Request, Response
from lattice.runtime.router import RuntimeRouter


class RuntimeContractTransform(ReversibleSyncTransform):
    """Attach a runtime optimization contract to each request."""

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
        return Ok(request)

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        return response


__all__ = ["RuntimeContractTransform"]
