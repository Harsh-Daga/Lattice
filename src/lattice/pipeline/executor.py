"""Pipeline verbatim execution — process() and transform registry."""

from __future__ import annotations

import time
from typing import Any

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.result import Ok, Result, is_ok, unwrap, unwrap_err
from lattice.ir.primitives import Candidate, ExecutionPlan, PromptIRV2
from lattice.ir.transform import CandidateSearch, IRTransform, LegacyRequestTransformAdapter
from lattice.pipeline._generated_factories import DEFAULT_TRANSFORM_FACTORIES
from lattice.planner.runtime_state import get_canonical_state_value
from lattice.transport.types import Request

__all__ = ["PipelineTransformRegistry", "_serialize_ir_to_messages", "IR_NATIVE_TRANSFORMS", "pipeline_process"]


def _serialize_ir_to_messages(ir: PromptIRV2, working: Request) -> None:
    """Map serialized IR sections back to working Request.messages."""
    if not ir.sections or not working.messages:
        return

    section_texts = ["\n".join(sp.text for sp in sec.spans if sp.text) for sec in ir.sections]

    if len(section_texts) == 1 and working.messages:
        working.messages[0].content = section_texts[0]
        return

    if len(section_texts) >= len(working.messages):
        for i, msg in enumerate(working.messages):
            if i < len(section_texts) - 1:
                msg.content = section_texts[i]
            else:
                msg.content = "\n\n".join(section_texts[i:])
    else:
        for i, text in enumerate(section_texts):
            working.messages[i].content = text


class PipelineTransformRegistry:
    """Lazy transform registry — loads by canonical name."""

    _FACTORIES: dict[str, tuple[str, str]] = DEFAULT_TRANSFORM_FACTORIES

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

    def get_instance_names(self) -> list[str]:
        """Return names with materialized instances from :meth:`register_instance`."""
        return sorted(self._instances.keys())

    def register_instance(self, name: str, instance: Any) -> None:
        """Inject a pre-built instance under ``name``."""
        self._instances[name] = instance


# Explicit allowlist: transforms that run through native IR `optimize(ir, ...)`.
IR_NATIVE_TRANSFORMS = frozenset(
    {
        "content_profiler",
        "runtime_contract",
        "message_dedup",
        "cache_arbitrage",
        "causal_chain",
        "format_conversion",
        "rate_distortion",
        "path_prefix",
        "tool_projection",
        "reference_sub",
        "tool_filter",
    }
)


def pipeline_process(
    pipeline: Any,
    request: Request,
    plan: ExecutionPlan,
    context: TransformContext,
) -> Result[Request, TransformError]:
    """Execute ExecutionPlan transforms in order (verbatim + beam search)."""
    working = request.copy()
    tokens_before = sum(len(str(m.content or "")) for m in working.messages)
    total_latency_ms = 0.0

    ir_v2 = get_canonical_state_value(context, "_lattice_ir_v2")
    if ir_v2 is None:
        ir_v2 = PromptIRV2()
    working.metadata["_lattice_ir_v2"] = ir_v2
    context.session_state["_lattice_ir_v2"] = ir_v2

    candidate = Candidate(
        ir=ir_v2,
        metrics=frozenset({("tokens_before", tokens_before), ("quality_estimate", 1.0)}),
    )

    from lattice.transforms.registry import is_legacy_only, is_response_side

    core_transforms: list[str] = []
    optimizer_transforms: list[str] = []
    for tx_name in plan.transforms:
        if is_response_side(tx_name):
            continue
        if is_legacy_only(tx_name):
            continue
        if tx_name.endswith("_optimizer"):
            optimizer_transforms.append(tx_name)
        else:
            core_transforms.append(tx_name)

    for tx_name in core_transforms:
        if tx_name in context.transforms_applied:
            continue
        if total_latency_ms > plan.latency_budget_ms:
            context.record_metric("pipeline", "budget_exceeded", True)
            break

        inst = pipeline.registry.get(tx_name)
        if inst is None:
            context.record_metric("missing_transform", tx_name, True)
            continue

        start = time.perf_counter()
        try:
            if tx_name in IR_NATIVE_TRANSFORMS:
                result = inst.optimize(ir_v2, working, context)
                if is_ok(result):
                    ir_v2 = unwrap(result)
                    candidate = candidate.apply(tx_name, ir_v2)
                    working.metadata["_lattice_ir_v2"] = ir_v2
                    context.session_state["_lattice_ir_v2"] = ir_v2
                    context.mark_transform_applied(tx_name)
                else:
                    context.record_metric(tx_name, "error", str(unwrap_err(result)))
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
                if tx_name in IR_NATIVE_TRANSFORMS:
                    ir_v2 = get_canonical_state_value(context, "_lattice_ir_v2") or ir_v2
                    candidate = candidate.apply(tx_name, ir_v2)
        else:
            context.record_metric(tx_name, "error", str(unwrap_err(result)))
            continue

    if optimizer_transforms and total_latency_ms <= plan.latency_budget_ms:
        ir_transforms: list[IRTransform] = []
        for tx_name in optimizer_transforms:
            inst = pipeline.registry.get(tx_name)
            if inst is not None:
                if tx_name in IR_NATIVE_TRANSFORMS:
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

                old_ir = candidate.ir
                candidate = best_candidate
                ir_changed = best_candidate.ir != old_ir
                if ir_changed and best_candidate.applied and best_candidate.ir.sections:
                    _serialize_ir_to_messages(best_candidate.ir, working)

                context.record_metric("pipeline", "beam_search_latency_ms", round(search_ms, 3))
                context.record_metric("pipeline", "beam_candidates", len(search.transforms))
                context.record_metric(
                    "pipeline",
                    "ir_sections",
                    len(best_candidate.ir.sections),
                )
            except Exception:
                context.record_metric("pipeline", "beam_search_error", True)

    final_tokens = sum(len(str(m.content or "")) for m in working.messages)
    context.record_metric("pipeline", "tokens_before", tokens_before)
    context.record_metric("pipeline", "tokens_after", final_tokens)
    context.record_metric("pipeline", "tokens_saved", tokens_before - final_tokens)
    context.record_metric("pipeline", "latency_ms", total_latency_ms)
    context.record_metric("pipeline", "optimizers", len(optimizer_transforms))
    context.record_metric("pipeline", "core_transforms", len(core_transforms))

    return Ok(working)
