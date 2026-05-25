"""core/ir_transform.py — New transform base class operating on PromptIR.

This is the v2 transform interface. It operates on PromptIRV2 (immutable) and
returns new PromptIRV2 instances. It does NOT mutate Request.messages directly.

Legacy transforms (ReversibleSyncTransform) are wrapped by IRTransformAdapter.
"""

from __future__ import annotations

from typing import Any, Protocol

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.result import Ok, Result
from lattice.ir.primitives import (
    Candidate,
    CandidateGraph,
    CandidateScore,
    PromptIRV2,
)
from lattice.transport.types import Request, Response


class IRTransform(Protocol):
    """Protocol for v2 transforms operating on immutable PromptIRV2.

    Every v2 transform MUST:
    1. Accept PromptIRV2 as input
    2. Return a NEW PromptIRV2 (never mutate in place)
    3. Never touch Request.messages directly
    4. Validate its output via the canonical CandidateScorer
    """

    name: str
    priority: int
    enabled: bool = True

    def can_process(self, ir: PromptIRV2, context: TransformContext) -> bool:
        """Check whether this transform applies to the given IR."""
        return self.enabled

    def optimize(
        self, ir: PromptIRV2, request: Request, context: TransformContext
    ) -> Result[PromptIRV2, TransformError]:
        """Optimize IR. Returns NEW PromptIRV2."""
        ...

    def reverse(self, response: Response, context: TransformContext) -> Response:
        """Reverse any transformations. Default: no-op (lossless)."""
        return response


class LegacyRequestTransformAdapter:
    """Adapt a legacy request-mutating transform into an IRTransform.

    The adapter runs the legacy transform on a request copy, then rebuilds the
    immutable PromptIRV2 from the modified request so CandidateSearch can keep
    branching on immutable state.
    """

    def __init__(self, legacy_transform: Any) -> None:
        self._tx = legacy_transform
        self.name = getattr(legacy_transform, "name", legacy_transform.__class__.__name__)
        self.priority = getattr(legacy_transform, "priority", 50)
        self.enabled = getattr(legacy_transform, "enabled", True)

    def can_process(self, _ir: PromptIRV2, _context: TransformContext) -> bool:
        return self.enabled

    def optimize(
        self, ir: PromptIRV2, request: Request, context: TransformContext
    ) -> Result[PromptIRV2, TransformError]:
        from lattice.ir.builder import compile_request_ir
        from lattice.ir.primitives import prompt_ir_v2_from_legacy

        req_copy = request.copy()
        before = req_copy.copy()
        result = self._tx.process(req_copy, context)
        if isinstance(result, Ok):
            modified = result.unwrap()
            if modified is not None and modified != before:
                try:
                    legacy_ir = compile_request_ir(modified)
                    return Ok(prompt_ir_v2_from_legacy(legacy_ir))
                except Exception:
                    return Ok(ir)
        return Ok(ir)

    def reverse(self, response: Response, context: TransformContext) -> Response:
        result = self._tx.reverse(response, context)
        if isinstance(result, Response):
            return result
        return response


class CandidateScorer:
    """Single source of truth for scoring candidates.

    Replaces:
    - optimizer/quality_estimator.py
    - optimizer/validation.py (partial)
    - guardrails.py safety checks (partial)
    - Per-optimizer _Candidate.score

    Formula:
        expected_utility =
            cost_reduction (token savings)
            + cache_gain (prefix stability)
            + transport_gain (wire savings)
            - semantic_risk (task disruption)
            - latency_cost (transform overhead)
            - instability_penalty (non-determinism)
    """

    @staticmethod
    def score(candidate: Candidate) -> CandidateScore:
        """Compute canonical score from candidate metrics."""
        from lattice.ir.scoring import composite_score

        return composite_score(dict(candidate.metrics))

    @staticmethod
    def validate(candidate: Candidate, quality_floor: float = 0.85) -> tuple[bool, str]:
        """Validate candidate against hard rules.

        Returns (passed, reason).
        """
        m = dict(candidate.metrics)
        tokens_before = m.get("tokens_before", 0)
        tokens_after = m.get("tokens_after", tokens_before)
        quality_estimate = m.get("quality_estimate", 1.0)
        cache_gain = m.get("cache_gain", 0.0)
        transport_gain = m.get("transport_gain", 0.0)

        # Rule 1: Expansion without gain → REJECT
        if tokens_after > tokens_before:
            if cache_gain <= 0 and transport_gain <= 0:
                return (
                    False,
                    f"tokens_after ({tokens_after}) > tokens_before ({tokens_before}) with no gain",
                )

        # Rule 2: Quality below floor → REJECT
        if quality_estimate < quality_floor:
            return False, f"quality {quality_estimate:.2f} < floor {quality_floor:.2f}"

        # Rule 3: Excessive compression → PENALIZE (but not reject)
        if tokens_before > 0:
            compression = (tokens_before - tokens_after) / tokens_before
            if compression > 0.90:
                return True, f"warning: compression {compression:.2f} > 0.90"

        return True, ""


class CandidateSearch:
    """Immutable state-space search over candidates.

    Replaces:
    - optimizer/representation_optimizer.py beam search
    - optimizer/*_optimizer.py internal candidate selection
    """

    def __init__(
        self,
        transforms: list[IRTransform],
        beam_width: int = 5,
        max_depth: int = 6,
        min_score: float = -10.0,
    ) -> None:
        self.transforms = transforms
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.min_score = min_score

    def search(
        self,
        initial_ir: PromptIRV2,
        request: Request,
        quality_floor: float,
        budget_ms: float,
        context: TransformContext,
    ) -> Candidate:
        """Run immutable beam search and return best candidate."""
        initial = Candidate(ir=initial_ir)
        graph = CandidateGraph(beam=(initial,))
        total_latency = 0.0

        for step in range(self.max_depth):
            expanded: list[Candidate] = []

            for cand in graph.beam:
                # Path 1: carry forward (skip this step)
                expanded.append(cand)

                # Path 2: apply eligible transforms
                for tx in self.transforms:
                    if not tx.can_process(cand.ir, context):
                        continue
                    if tx.name in cand.applied:
                        continue

                    import time

                    start = time.perf_counter()
                    search_context = context.copy()
                    result = tx.optimize(cand.ir, request, search_context)
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    total_latency += elapsed_ms

                    if total_latency > budget_ms:
                        context.record_metric(self.__class__.__name__, "budget_exceeded", True)
                        break

                    if isinstance(result, Ok):
                        new_ir = result.unwrap()
                        if isinstance(new_ir, PromptIRV2) and new_ir is not cand.ir:
                            new_cand = self._make_candidate(
                                cand, tx.name, new_ir, elapsed_ms, context
                            )
                            passed, reason = CandidateScorer.validate(new_cand, quality_floor)
                            if passed:
                                expanded.append(new_cand)
                            elif reason:
                                context.record_metric(tx.name, "rejected", reason)

                if total_latency > budget_ms:
                    break

            if total_latency > budget_ms:
                break

            # Prune to beam_width, keeping highest-scoring
            scored = sorted(expanded, key=lambda c: c.score().expected_utility, reverse=True)
            graph = graph.expand(tuple(scored[: self.beam_width]))

        if graph.best is None:
            return initial

        return graph.best

    @staticmethod
    def _make_candidate(
        parent: Candidate,
        tx_name: str,
        new_ir: PromptIRV2,
        latency_ms: float,
        context: TransformContext,
    ) -> Candidate:
        """Create a new candidate with metrics from the optimization."""
        new_cand = parent.apply(tx_name, new_ir)

        # Compute token metrics (simplified — real version uses tokenizer)
        tokens_before = sum(len(sp.text) for sec in parent.ir.sections for sp in sec.spans)
        tokens_after = sum(len(sp.text) for sec in new_ir.sections for sp in sec.spans)

        return (
            new_cand.with_metric("tokens_before", tokens_before)
            .with_metric("tokens_after", tokens_after)
            .with_metric("latency_ms", latency_ms)
            .with_metric("quality_estimate", 1.0)  # Simplified
        )
