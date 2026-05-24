"""Unified pipeline runner — verbatim plan execution + gated compress entry.

Pipeline exposes two entry points on a single class (not parallel paths):

  - Pipeline.process(request, plan, context) — low-level verbatim executor
    used by benchmarks and tests. Applies plan.transforms in order with
    beam search over optimizer transforms. No safety gates, no rollback.

  - Pipeline.compress(request, context, *, config=None) — high-level entry
    that ports v1 CompressorPipeline's safety machinery into the v2 path:
    runs content_profiler to seed context, reads ExecutionPlan, then walks
    plan transforms applying the 8 v1 gates around each (see pipeline.gates).

  - Pipeline.reverse(response, plan, context) — apply reverse() to applied
    transforms in reverse order.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.result import Err, Ok, Result, is_err, is_ok, unwrap, unwrap_err
from lattice.ir.primitives import Candidate, ExecutionPlan, PromptIRV2
from lattice.ir.transform import (
    CandidateSearch,
    IRTransform,
    LegacyRequestTransformAdapter,
)
from lattice.pipeline import gates as _gates
from lattice.pipeline.base import ReversibleSyncTransform, TransformClass
from lattice.pipeline.policy import OptimizationPolicy, Reject, Skip
from lattice.planner.runtime_state import (
    coerce_execution_plan,
    get_canonical_request_value,
    get_canonical_state_value,
)

# Re-exports — keeps ``from lattice.pipeline.runner import ReversibleSyncTransform``
# working for callers that prefer the runner module path.
__all__ = ["Pipeline", "PipelineTransformRegistry", "ReversibleSyncTransform", "TransformClass"]
from lattice.transport.types import Request, Response

_logger = structlog.get_logger()


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


class PipelineTransformRegistry:
    """Lazy transform registry — loads by canonical name."""

    # Map canonical_name → (module_path, class_name)
    _FACTORIES: dict[str, tuple[str, str]] = {
        "content_profiler": ("lattice.transforms.content_profiler", "ContentProfiler"),
        "runtime_contract": ("lattice.transforms.runtime_contract", "RuntimeContractTransform"),
        "message_dedup": ("lattice.transforms.message_dedup", "MessageDeduplicator"),
        "cache_arbitrage": ("lattice.transforms.cache_arbitrage", "CacheArbitrageOptimizer"),
        "causal_chain": ("lattice.transforms.causal_chain", "CausalChainExtractor"),
        "strategy_selector": ("lattice.transforms.strategy_selector", "StrategySelector"),
        "format_conversion": ("lattice.transforms.format_converter", "FormatConverter"),
        "rate_distortion": ("lattice.transforms.rate_distortion", "RateDistortionCompressor"),
        "path_prefix": ("lattice.transforms.path_prefix", "PathPrefixCompressor"),
        "tool_projection": ("lattice.transforms.tool_projection", "QueryAwareProjection"),
        "reference_optimizer": (
            "lattice.transforms.optimizers.reference_optimizer",
            "ReferenceOptimizer",
        ),
        "ir_structure_optimizer": (
            "lattice.transforms.optimizers.ir_structure_optimizer",
            "IRStructureOptimizer",
        ),
        "diagnostic_optimizer": (
            "lattice.transforms.optimizers.diagnostic_optimizer",
            "DiagnosticOptimizer",
        ),
        "context_optimizer": (
            "lattice.transforms.optimizers.context_optimizer",
            "ContextOptimizer",
        ),
        "tool_optimizer": ("lattice.transforms.optimizers.tool_optimizer", "ToolOptimizer"),
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

    def get_instance_names(self) -> list[str]:
        """Return names with materialized instances from :meth:`register_instance`."""
        return sorted(self._instances.keys())

    def register_instance(self, name: str, instance: Any) -> None:
        """Inject a pre-built instance under ``name``.

        Used by the proxy bootstrap to install execution-only transforms
        (delta_encoder, batching, speculative) that need session-scoped
        dependencies the lazy factory cannot supply.
        """
        self._instances[name] = instance


class Pipeline:
    """Verbatim pipeline — executes plan.transforms in order without re-decision."""

    # Explicit allowlist: transforms that run through native IR `optimize(ir, ...)`.
    # Every transform on the canonical v2 request-side path runs via this path.
    # Response-only transforms (output_cleanup) and optimizers without native
    # IR support go through the legacy adapter.
    _IR_NATIVE_TRANSFORMS = {
        "content_profiler",
        "runtime_contract",
        "message_dedup",
        "strategy_selector",
        "cache_arbitrage",
        "causal_chain",
        "format_conversion",
        "rate_distortion",
        "path_prefix",
        "tool_projection",
        "reference_sub",
        "tool_filter",
    }

    def __init__(
        self,
        registry: PipelineTransformRegistry | None = None,
        config: LatticeConfig | None = None,
        policy: OptimizationPolicy | None = None,
    ) -> None:
        self.registry = registry or PipelineTransformRegistry()
        self.config = config or LatticeConfig()
        self.policy = policy or OptimizationPolicy(self.config)

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

        # Phase 1: Execute core transforms verbatim (skip already-applied + self)
        for tx_name in core_transforms:
            if tx_name in context.transforms_applied:
                continue
            if total_latency_ms > plan.latency_budget_ms:
                context.record_metric("pipeline", "budget_exceeded", True)
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

                    context.record_metric("pipeline", "beam_search_latency_ms", round(search_ms, 3))
                    context.record_metric("pipeline", "beam_candidates", len(search.transforms))
                    context.record_metric(
                        "pipeline",
                        "ir_sections",
                        len(best_candidate.ir.sections),
                    )
                except Exception:
                    context.record_metric("pipeline", "beam_search_error", True)

        # Phase 3: Execute any remaining non-optimizer transforms verbatim
        # (already handled in core_transforms)

        final_tokens = sum(len(str(m.content or "")) for m in working.messages)
        context.record_metric("pipeline", "tokens_before", tokens_before)
        context.record_metric("pipeline", "tokens_after", final_tokens)
        context.record_metric("pipeline", "tokens_saved", tokens_before - final_tokens)
        context.record_metric("pipeline", "latency_ms", total_latency_ms)
        context.record_metric("pipeline", "optimizers", len(optimizer_transforms))
        context.record_metric("pipeline", "core_transforms", len(core_transforms))

        return Ok(working)

    @property
    def transforms(self) -> list[Any]:
        """Materialized lazy-registry transforms (health, tests, debugging)."""
        instances: list[Any] = []
        for name in self.registry.get_transform_names():
            inst = self.registry.get(name)
            if inst is not None:
                instances.append(inst)
        return instances

    def compress(
        self,
        request: Request,
        context: TransformContext,
        *,
        config: LatticeConfig | None = None,
    ) -> Result[Request, TransformError]:
        """High-level entry: gate-orchestrated compression (replaces v1 CompressorPipeline.process).

        Algorithm:
          1. Global policy check_request_limits — Reject aborts pipeline.
          2. Run content_profiler.process(req, ctx) to populate context with
             task classification, risk score, and ExecutionPlan.
          3. Read plan from context; if absent, fall back to a default plan
             (all default_pipeline transforms in priority order).
          4. For each transform in plan.transforms, apply v1 gates in order:
             enabled → can_process → policy → runtime-budget → risk →
             protected-span → scheduler → execute → expansion → negative-savings →
             compression-limit → placeholder-leakage → PSG preservation → MILV →
             record reputation + mark applied.
          5. Apply aggregate debugging compression cap.
          6. Return Ok(working) or Err on hard failure (graceful_degradation=False).
        """
        cfg = config or self.config
        log = _logger.bind(module="pipeline_compress", request_id=context.request_id)

        working = request.copy()
        original_token_estimate = working.token_estimate

        # ---- Global request limits (policy gate 0) ----
        limits = self.policy.check_request_limits(working)
        if isinstance(limits, Reject):
            log.warning(
                "request_rejected_by_policy",
                code=limits.code,
                message=limits.message,
            )
            return Err(
                TransformError(
                    transform="policy",
                    code=limits.code,
                    message=limits.message,
                    detail=limits.detail,
                )
            )

        # ---- Seed context: run content_profiler if available + not already applied ----
        from lattice.transforms.registry import is_legacy_only, is_response_side

        profiler = self.registry.get("content_profiler")
        profiler_present = profiler is not None
        if (
            profiler_present
            and profiler is not None
            and "content_profiler" not in context.transforms_applied
        ):
            try:
                ir_seed = get_canonical_state_value(context, "_lattice_ir_v2") or PromptIRV2()
                if "content_profiler" in self._IR_NATIVE_TRANSFORMS and hasattr(
                    profiler, "optimize"
                ):
                    result = profiler.optimize(ir_seed, working, context)
                    if is_ok(result):
                        ir_v2 = unwrap(result)
                        working.metadata["_lattice_ir_v2"] = ir_v2
                        context.session_state["_lattice_ir_v2"] = ir_v2
                else:
                    result = profiler.process(working, context)
                    if is_ok(result):
                        working = unwrap(result)
                if is_ok(result):
                    context.mark_transform_applied("content_profiler")
            except Exception as exc:
                log.warning("content_profiler_failed", error=str(exc))

        # ---- Read plan from context; fallback to a synthetic default plan ----
        plan = coerce_execution_plan(get_canonical_state_value(context, "_lattice_execution_plan"))
        if plan is None:
            from lattice.transforms.registry import BUILTIN_TRANSFORMS

            names = [
                spec.canonical_name
                for spec in BUILTIN_TRANSFORMS
                if spec.default_pipeline and not spec.legacy_only
            ]
            plan = ExecutionPlan(
                transforms=tuple(n for n in names if n != "content_profiler"),
                latency_budget_ms=1000.0,
                quality_floor=0.85,
            )

        # ---- Per-transform gated loop ----
        backup = working.copy()
        original_backup = working.copy()
        cumulative_transform_ms = 0.0
        rollback_reasons: dict[str, str] = {}
        ir_v2 = get_canonical_state_value(context, "_lattice_ir_v2") or PromptIRV2()

        for tx_name in plan.transforms:
            if tx_name in context.transforms_applied:
                continue
            if is_response_side(tx_name):
                continue
            if is_legacy_only(tx_name):
                continue
            if not cfg.is_transform_enabled(tx_name):
                continue

            inst = self.registry.get(tx_name)
            if inst is None:
                context.record_metric("missing_transform", tx_name, True)
                continue
            if hasattr(inst, "can_process") and not inst.can_process(working, context):
                continue

            # Gate 1: Policy decision
            decision = self.policy.should_run(tx_name, working, context)
            if isinstance(decision, Skip):
                continue
            if isinstance(decision, Reject):
                if cfg.graceful_degradation:
                    working = backup.copy()
                    continue
                return Err(
                    TransformError(
                        transform=tx_name,
                        code=decision.code,
                        message=decision.message,
                        detail=decision.detail,
                    )
                )

            # Gate 2: Runtime budget
            g = _gates.runtime_budget_blocks(tx_name, working, context, cumulative_transform_ms)
            if g.skip:
                context.record_metric(tx_name, "deferred", True)
                context.record_metric(tx_name, "deferred_reason", g.reason)
                continue

            # Gate 3: Semantic risk
            g = _gates.risk_gate_blocks(tx_name, working, context, profiler_present)
            if g.skip:
                context.record_metric(tx_name, "risk_blocked", True)
                context.record_metric(tx_name, "risk_block_reason", g.reason)
                blocked = context.metrics.setdefault("risk_blocked_transforms", [])
                if isinstance(blocked, list):
                    blocked.append(tx_name)
                continue

            # Gate 4: Protected-span DANGEROUS veto
            g = _gates.protected_span_veto(tx_name, working, context)
            if g.veto:
                context.record_metric(tx_name, "spans_vetoed", True)
                continue

            # Gate 5: Scheduler blocking
            g = _gates.scheduler_blocks(tx_name, working, context)
            if g.skip:
                context.record_metric(tx_name, "scheduler_blocked", True)
                context.record_metric(tx_name, "deferred", True)
                context.record_metric(tx_name, "deferred_reason", g.reason)
                continue

            # ---- Execute (IR-native optimize() preferred; fall back to process()) ----
            tokens_before = working.token_estimate
            text_before = "\n".join(m.content for m in backup.messages)
            start = time.perf_counter()
            try:
                if tx_name in self._IR_NATIVE_TRANSFORMS and hasattr(inst, "optimize"):
                    result = inst.optimize(ir_v2, working, context)
                    if is_ok(result):
                        ir_v2 = unwrap(result)
                        working.metadata["_lattice_ir_v2"] = ir_v2
                        context.session_state["_lattice_ir_v2"] = ir_v2
                else:
                    result = inst.process(working, context)
            except Exception as exc:
                log.warning("transform_exception", transform=tx_name, error=str(exc))
                if cfg.graceful_degradation:
                    working = backup.copy()
                    continue
                return Err(
                    TransformError(
                        transform=tx_name,
                        code="PIPELINE_EXECUTION_ERROR",
                        message=f"Transform raised exception: {exc}",
                        detail={"exception": str(exc)},
                    )
                )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            cumulative_transform_ms += elapsed_ms

            if is_err(result):
                err = unwrap_err(result)
                if cfg.graceful_degradation:
                    working = backup.copy()
                    continue
                return Err(err)

            # IR-native: optimize returns PromptIRV2; serialize back to messages.
            unwrapped = unwrap(result)
            if isinstance(unwrapped, PromptIRV2):
                if unwrapped.sections:
                    _serialize_ir_to_messages(unwrapped, working)
                ir_v2 = unwrapped
                working.metadata["_lattice_ir_v2"] = ir_v2
                context.session_state["_lattice_ir_v2"] = ir_v2
            elif isinstance(unwrapped, Request):
                working = unwrapped
            tokens_after = working.token_estimate
            text_after = "\n".join(m.content for m in working.messages)

            # ---- Post-execution guards (rollback on violation) ----
            max_ratio = getattr(cfg, "max_transform_expansion_ratio", 1.5)

            def _rollback(reason: str, code: str = "", extra: dict[str, Any] | None = None) -> None:
                nonlocal working
                context.record_metric(tx_name, "safety_rollback", True)
                context.record_metric(tx_name, "rollback_reason", reason)
                if extra:
                    for k, v in extra.items():
                        context.record_metric(tx_name, k, v)
                working = backup.copy()
                working.metadata["_lattice_rollback_reason"] = reason
                rollback_reasons[tx_name] = reason
                from lattice.transforms.reputation import get_reputation_registry

                get_reputation_registry().record(
                    tx_name, quality=0.0, compression=0.0, rolled_back=True
                )

            chk = _gates.check_expansion(tx_name, tokens_before, tokens_after, max_ratio)
            if chk.rollback:
                _rollback(chk.reason, chk.code, chk.extra)
                continue

            chk = _gates.check_negative_savings(tx_name, tokens_before, tokens_after)
            if chk.rollback:
                _rollback(chk.reason, chk.code, chk.extra)
                continue

            chk = _gates.check_compression_limit(
                tx_name, working, context, tokens_before, tokens_after
            )
            if chk.rollback:
                _rollback(chk.reason, chk.code, chk.extra)
                if chk.fail and not cfg.graceful_degradation:
                    return Err(TransformError(transform=tx_name, code=chk.code, message=chk.reason))
                continue

            chk = _gates.check_placeholder_leakage(tx_name, text_before, text_after)
            if chk.rollback:
                _rollback(chk.reason, chk.code, chk.extra)
                if chk.fail and not cfg.graceful_degradation:
                    return Err(TransformError(transform=tx_name, code=chk.code, message=chk.reason))
                continue

            chk = _gates.check_psg_preservation(tx_name, working, context, text_before, text_after)
            if chk.rollback:
                _rollback(chk.reason, chk.code, chk.extra)
                if chk.fail and not cfg.graceful_degradation:
                    return Err(TransformError(transform=tx_name, code=chk.code, message=chk.reason))
                continue

            # MILV runtime judge
            task_data = get_canonical_request_value(
                working, context, "_lattice_task_classification", {}
            )
            has_task = (
                bool(task_data)
                and isinstance(task_data, dict)
                and bool(task_data.get("task_class"))
            )
            if _gates.should_run_milv(tx_name, text_before, text_after, tokens_before, has_task):
                from lattice.pipeline.milv import should_trigger_milv, validate_transform
                from lattice.planner.task_classifier import TaskClass, TaskClassification

                tc_str = (
                    task_data.get("task_class", "simple")
                    if isinstance(task_data, dict)
                    else "simple"
                )
                tc = TaskClassification(
                    task_class=getattr(TaskClass, tc_str.upper(), TaskClass.SIMPLE),
                )
                compression = (tokens_before - tokens_after) / tokens_before
                if should_trigger_milv(
                    tx_name,
                    tc,
                    compression_ratio=compression,
                    placeholder_aliasing_used=False,
                ):
                    milv_result = validate_transform(
                        text_before,
                        text_after,
                        tc,
                        placeholder_aliasing_used=False,
                    )
                    context.record_metric(tx_name, "milv_triggered", True)
                    context.record_metric(tx_name, "milv_score", milv_result.score)
                    context.record_metric(tx_name, "milv_passed", milv_result.passed)
                    if not milv_result.passed:
                        _rollback(
                            f"milv_rejected:{milv_result.reason}",
                            code="MILV_REJECTED",
                        )
                        if not cfg.graceful_degradation:
                            return Err(
                                TransformError(
                                    transform=tx_name,
                                    code="MILV_REJECTED",
                                    message=(
                                        f"MILV rejected: score={milv_result.score:.2f}, "
                                        f"{milv_result.reason}"
                                    ),
                                )
                            )
                        continue

            # ---- Accept transform ----
            backup = working.copy()
            context.mark_transform_applied(tx_name)
            context.record_metric(tx_name, "tokens_before", tokens_before)
            context.record_metric(tx_name, "tokens_after", tokens_after)
            context.record_metric(tx_name, "latency_ms", round(elapsed_ms, 3))

            if tokens_before > 0:
                compression = (tokens_before - tokens_after) / tokens_before
                from lattice.transforms.reputation import get_reputation_registry

                get_reputation_registry().record(
                    tx_name, quality=1.0, compression=compression, rolled_back=False
                )

        # ---- Aggregate debugging cap ----
        final_tokens = working.token_estimate
        agg = _gates.check_aggregate_debugging_cap(
            working, context, original_token_estimate, final_tokens
        )
        if agg.rollback:
            working = original_backup.copy()
            working.metadata["_lattice_rollback_reason"] = agg.reason
            context.record_metric("pipeline", "aggregate_rollback", True)
            if agg.extra:
                for k, v in agg.extra.items():
                    context.record_metric("pipeline", k, v)
            final_tokens = original_token_estimate

        # ---- Telemetry ----
        context.metrics["tokens_in"] = original_token_estimate
        context.metrics["tokens_out"] = final_tokens
        context.metrics["latency_ms"] = context.elapsed_ms
        context.metrics["transform_latency_ms"] = round(cumulative_transform_ms, 3)
        applied_list = list(context.transforms_applied)
        working.metadata["_lattice_safety_decision"] = {
            "applied": applied_list,
            "rollback_reasons": rollback_reasons,
        }
        working.metadata["_lattice_reachability"] = {
            "reached": applied_list,
            "activated": applied_list,
            "useful": applied_list,
            "reached_count": len(applied_list),
            "activated_count": len(applied_list),
            "useful_count": len(applied_list),
        }
        runtime_contract_md = working.metadata.get("_lattice_runtime_contract") or {}
        runtime_budget_ms = (
            runtime_contract_md.get("max_transform_latency_ms", 0.0)
            if isinstance(runtime_contract_md, dict)
            else 0.0
        )
        working.metadata["_lattice_runtime_budget"] = {
            "exhausted": False,
            "skipped_count": 0,
            "skipped_transforms": [],
            "actual_transform_ms": round(cumulative_transform_ms, 3),
            "budget_ms": runtime_budget_ms,
        }

        log.info(
            "pipeline_complete",
            transforms_applied=list(context.transforms_applied),
            tokens_before=original_token_estimate,
            tokens_after=final_tokens,
        )
        return Ok(working)

    def reverse(
        self,
        response: Response,
        context: TransformContext,
        *,
        plan: ExecutionPlan | None = None,
    ) -> Response:
        """Reverse transforms in reverse order.

        Source of transforms (in order of preference):
          1. ``plan`` argument when supplied;
          2. ``context.session_state["_lattice_execution_plan"]`` (coerced);
          3. ``context.transforms_applied`` (what compress actually ran).

        Response-side transforms (``is_response_side`` in the registry) run here.
        """
        from lattice.transforms.registry import is_response_side

        if plan is None:
            plan = coerce_execution_plan(
                get_canonical_state_value(context, "_lattice_execution_plan")
            )

        if plan is not None and plan.transforms:
            tx_names: list[str] = list(plan.transforms)
        else:
            tx_names = list(context.transforms_applied)

        for tx_name in reversed(tx_names):
            if is_response_side(tx_name):
                continue
            inst = self.registry.get(tx_name)
            if inst is None or not hasattr(inst, "reverse"):
                continue
            try:
                response = inst.reverse(response, context)
            except Exception:
                pass

        for tx_name in reversed(tx_names):
            if not is_response_side(tx_name):
                continue
            inst = self.registry.get(tx_name)
            if inst is None or not hasattr(inst, "reverse"):
                continue
            try:
                response = inst.reverse(response, context)
            except Exception:
                pass

        # Ensure response-side transforms run even if omitted from plan/applied list.
        for tx_name in self.registry.get_transform_names():
            if not is_response_side(tx_name) or tx_name in tx_names:
                continue
            inst = self.registry.get(tx_name)
            if inst is None or not hasattr(inst, "reverse"):
                continue
            try:
                response = inst.reverse(response, context)
            except Exception:
                pass

        return response
