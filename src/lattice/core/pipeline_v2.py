"""core/pipeline_v2.py — Verbatim execution of ExecutionPlan with optional beam search.

PipelineV2 receives an ExecutionPlan from UnifiedPlanner and executes it
without runtime re-decision. When the plan includes optimizer transforms,
it runs CandidateSearch (immutable beam search) over those transforms.

Key design:
  1. process(request, plan, context) → apply core transforms verbatim,
     then run CandidateSearch over optimizer transforms, then serialize
  2. Track budget. Break if exceeded.
  3. Validate each result via CandidateScorer.
  4. Return best Request.
  5. reverse(response, context) → apply reverse transforms in reverse order.

This replaces pipeline.py's 13 gate layers with clean execution.
"""
from __future__ import annotations

import time
from typing import Any

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.result import Ok, Result, is_ok, unwrap, unwrap_err
from lattice.core.runtime_state import get_canonical_state_value
from lattice.core.transport import Request, Response
from lattice.ir.primitives import Candidate, ExecutionPlan, PromptIRV2
from lattice.ir.transform import (
    CandidateSearch,
    IRTransform,
    LegacyRequestTransformAdapter,
)

_RESPONSE_ONLY_TRANSFORMS = {"output_cleanup"}


def _serialize_ir_to_messages(ir: PromptIRV2, working: Request) -> None:
    """Map serialized IR sections back to working Request.messages.

    Strategy:
    - If there's exactly one message and multiple IR sections,
      serialize the whole IR into that message.
    - If IR has fewer sections than messages, only update the first N messages.
    - If IR has more sections than messages, coalesce extra sections into the last message.
    """
    if not ir.sections or not working.messages:
        return

    # Compute text per section
    section_texts = ["\n".join(sp.text for sp in sec.spans if sp.text) for sec in ir.sections]

    if len(section_texts) == 1 and working.messages:
        working.messages[0].content = section_texts[0]
        return

    if len(section_texts) >= len(working.messages):
        for i, msg in enumerate(working.messages):
            if i < len(section_texts) - 1:
                msg.content = section_texts[i]
            else:
                # Last message gets all remaining sections
                msg.content = "\n\n".join(section_texts[i:])
    else:
        for i, text in enumerate(section_texts):
            working.messages[i].content = text


class TransformRegistryV2:
    """Lazy transform registry — loads by canonical name."""

    # Map canonical_name → (module_path, class_name)
    _FACTORIES: dict[str, tuple[str, str]] = {
        "content_profiler": ("lattice.transforms.content_profiler", "ContentProfiler"),
        "runtime_contract": ("lattice.transforms.runtime_contract", "RuntimeContractTransform"),
        "constraint_lifting": ("lattice.transforms.constraint_lifting", "ConstraintLiftingTransform"),
        "message_dedup": ("lattice.transforms.message_dedup", "MessageDeduplicator"),
        "cache_arbitrage": ("lattice.transforms.cache_arbitrage", "CacheArbitrageOptimizer"),
        "causal_chain": ("lattice.transforms.causal_chain", "CausalChainExtractor"),
        "prefix_optimizer": ("lattice.transforms.prefix_opt", "PrefixOptimizer"),
        "strategy_selector": ("lattice.transforms.strategy_selector", "StrategySelector"),
        "format_conversion": ("lattice.transforms.format_conv", "FormatConverter"),
        "rate_distortion": ("lattice.transforms.rate_distortion", "RateDistortionCompressor"),
        "path_prefix": ("lattice.transforms.path_prefix", "PathPrefixCompressor"),
        "tool_projection": ("lattice.transforms.tool_projection", "QueryAwareProjection"),
        "reference_optimizer": ("lattice.optimizer.reference_optimizer", "ReferenceOptimizer"),
        "structure_optimizer": ("lattice.optimizer.structure_optimizer", "StructureOptimizer"),
        "ir_structure_optimizer": ("lattice.optimizer.ir_structure_optimizer", "IRStructureOptimizer"),
        "diagnostic_optimizer": ("lattice.optimizer.diagnostic_optimizer", "DiagnosticOptimizer"),
        "context_optimizer": ("lattice.optimizer.context_optimizer", "ContextOptimizer"),
        "tool_optimizer": ("lattice.optimizer.tool_optimizer", "ToolOptimizer"),
        "output_cleanup": ("lattice.transforms.output_cleanup", "OutputCleanup"),
        "reference_sub": ("lattice.transforms.reference_sub", "ReferenceSubstitution"),
        "tool_filter": ("lattice.transforms.tool_filter", "ToolOutputFilter"),
    }

    _instances: dict[str, Any]

    def __init__(self) -> None:
        self._instances = {}

    def get(self, name: str) -> Any | None:
        """Lazy-load transform instance by canonical name."""
        if name in self._instances:
            return self._instances[name]

        spec = self._FACTORIES.get(name)
        if spec is None:
            return None

        mod_path, cls_name = spec
        try:
            import importlib

            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            inst = cls()
            self._instances[name] = inst
            return inst
        except Exception:
            return None

    def get_transform_names(self) -> list[str]:
        """Return all registered canonical names."""
        return sorted(self._FACTORIES.keys())


class PipelineV2:
    """Verbatim pipeline — executes plan.transforms in order without re-decision."""

    # Explicit allowlist: transforms that run through native IR `optimize(ir, ...)`.
    # Every transform on the canonical v2 request-side path runs via this path.
    # Response-only transforms (output_cleanup) and optimizers without native
    # IR support go through the legacy adapter.
    _IR_NATIVE_TRANSFORMS = {
        "runtime_contract",
        "prefix_optimizer",
        "message_dedup",
        "strategy_selector",
        "cache_arbitrage",
        "constraint_lifting",
        "causal_chain",
        "format_conversion",
        "rate_distortion",
        "path_prefix",
        "tool_projection",
        "reference_sub",
        "tool_filter",
    }

    def __init__(self, registry: TransformRegistryV2 | None = None) -> None:
        self.registry = registry or TransformRegistryV2()

    def process(
        self,
        request: Request,
        plan: ExecutionPlan,
        context: TransformContext,
    ) -> Result[Request, TransformError]:
        """Execute ExecutionPlan transforms in order.

        Algorithm:
            1. Split transforms into core (verbatim) and optimizers (beam search)
            2. Execute core transforms verbatim
            3. If optimizers present, run CandidateSearch over them
            4. Serialize best IR back to Request
            5. Execute remaining transforms verbatim
        """
        working = request.copy()
        tokens_before = sum(len(str(m.content or "")) for m in working.messages)
        total_latency_ms = 0.0

        # Start with IR compiled by content_profiler (v2 path)
        ir_v2 = get_canonical_state_value(context, "_lattice_ir_v2")
        if ir_v2 is None:
            ir_v2 = PromptIRV2()
        working.metadata["_lattice_ir_v2"] = ir_v2
        context.session_state["_lattice_ir_v2"] = ir_v2

        candidate = Candidate(
            ir=ir_v2,
            metrics=frozenset({("tokens_before", tokens_before), ("quality_estimate", 1.0)}),
        )

        # Split: core transforms (verbatim) vs optimizers (beam search)
        core_transforms: list[str] = []
        optimizer_transforms: list[str] = []
        for tx_name in plan.transforms:
            if tx_name in _RESPONSE_ONLY_TRANSFORMS:
                continue
            if tx_name.endswith("_optimizer") and tx_name != "pipeline_v2":
                optimizer_transforms.append(tx_name)
            else:
                core_transforms.append(tx_name)

        # Phase 1: Execute core transforms verbatim (skip already-applied + self)
        for tx_name in core_transforms:
            if tx_name == "pipeline_v2":
                continue
            if tx_name in context.transforms_applied:
                continue
            if total_latency_ms > plan.latency_budget_ms:
                context.record_metric("pipeline_v2", "budget_exceeded", True)
                break

            inst = self.registry.get(tx_name)
            if inst is None:
                context.record_metric("missing_transform", tx_name, True)
                continue

            start = time.perf_counter()
            try:
                if tx_name in self._IR_NATIVE_TRANSFORMS:
                    result = inst.optimize(ir_v2, working, context)
                    if is_ok(result):
                        ir_v2 = unwrap(result)
                        candidate = candidate.apply(tx_name, ir_v2)
                        working.metadata["_lattice_ir_v2"] = ir_v2
                        context.session_state["_lattice_ir_v2"] = ir_v2
                        context.mark_transform_applied(tx_name)
                        continue
                result = inst.process(working, context)
            except Exception:
                context.record_metric(tx_name, "exception", True)
                continue
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            total_latency_ms += elapsed_ms

            if is_ok(result):
                modified = unwrap(result)
                if modified is not None:
                    working = modified
                    context.mark_transform_applied(tx_name)
                    if tx_name in self._IR_NATIVE_TRANSFORMS:
                        ir_v2 = get_canonical_state_value(context, "_lattice_ir_v2") or ir_v2
                        candidate = candidate.apply(tx_name, ir_v2)
            else:
                context.record_metric(tx_name, "error", str(unwrap_err(result)))
                continue
        # Phase 2: Beam search over optimizer transforms (if any)
        if optimizer_transforms and total_latency_ms <= plan.latency_budget_ms:
            # Build IRTransform list using the canonical allowlist. Everything
            # else is wrapped in the legacy request adapter.
            ir_transforms: list[IRTransform] = []
            for tx_name in optimizer_transforms:
                inst = self.registry.get(tx_name)
                if inst is not None:
                    if tx_name in self._IR_NATIVE_TRANSFORMS:
                        ir_transforms.append(inst)
                    else:
                        ir_transforms.append(LegacyRequestTransformAdapter(inst))

            if ir_transforms:
                remaining_budget = plan.latency_budget_ms - total_latency_ms
                search = CandidateSearch(
                    transforms=ir_transforms,
                    beam_width=plan.beam_width,
                    max_depth=min(plan.max_depth, len(optimizer_transforms)),
                )
                start = time.perf_counter()
                try:
                    best_candidate = search.search(
                        candidate.ir,
                        request=working,
                        quality_floor=plan.quality_floor,
                        budget_ms=remaining_budget,
                        context=context,
                    )
                    search_ms = (time.perf_counter() - start) * 1000.0
                    total_latency_ms += search_ms

                    # Update working request from best candidate IR only if the
                    # optimizer block actually changed the IR. Otherwise we must
                    # preserve the output produced by the verbatim core transforms
                    # that already ran on the request.
                    old_ir = candidate.ir
                    candidate = best_candidate
                    ir_changed = best_candidate.ir != old_ir
                    if ir_changed and best_candidate.applied and best_candidate.ir.sections:
                        _serialize_ir_to_messages(best_candidate.ir, working)

                    context.record_metric(
                        "pipeline_v2", "beam_search_latency_ms", round(search_ms, 3)
                    )
                    context.record_metric(
                        "pipeline_v2", "beam_candidates", len(search.transforms)
                    )
                    context.record_metric(
                        "pipeline_v2",
                        "ir_sections",
                        len(best_candidate.ir.sections),
                    )
                except Exception:
                    context.record_metric("pipeline_v2", "beam_search_error", True)

        # Phase 3: Execute any remaining non-optimizer transforms verbatim
        # (already handled in core_transforms)

        final_tokens = sum(len(str(m.content or "")) for m in working.messages)
        context.record_metric("pipeline_v2", "tokens_before", tokens_before)
        context.record_metric("pipeline_v2", "tokens_after", final_tokens)
        context.record_metric("pipeline_v2", "tokens_saved", tokens_before - final_tokens)
        context.record_metric("pipeline_v2", "latency_ms", total_latency_ms)
        context.record_metric("pipeline_v2", "optimizers", len(optimizer_transforms))
        context.record_metric("pipeline_v2", "core_transforms", len(core_transforms))

        return Ok(working)

    def reverse(
        self,
        response: Response,
        plan: ExecutionPlan,
        context: TransformContext,
    ) -> Response:
        """Reverse transforms in reverse order of the plan.

        Response-only transforms (output_cleanup) run on every response
        regardless of whether they appear in the request-side plan.
        """
        for tx_name in reversed(plan.transforms):
            inst = self.registry.get(tx_name)
            if inst is None:
                continue
            try:
                response = inst.reverse(response, context)
            except Exception:
                pass

        oc = self.registry.get("output_cleanup")
        if oc is not None:
            try:
                response = oc.reverse(response, context)
            except Exception:
                pass

        return response
