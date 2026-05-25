"""Unified pipeline runner — verbatim plan execution + gated compress entry."""

from __future__ import annotations

import time
from typing import Any

import structlog

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.result import Err, Ok, Result, is_err, is_ok, unwrap, unwrap_err
from lattice.ir.primitives import ExecutionPlan, PromptIRV2
from lattice.pipeline import gates as _gates
from lattice.pipeline.base import ReversibleSyncTransform, TransformClass
from lattice.pipeline.executor import (
    IR_NATIVE_TRANSFORMS,
    PipelineTransformRegistry,
    pipeline_process,
)
from lattice.pipeline.policy import OptimizationPolicy, Reject, Skip
from lattice.pipeline.reverse import pipeline_reverse
from lattice.planner.runtime_state import (
    coerce_execution_plan,
    get_canonical_request_value,
    get_canonical_state_value,
)
from lattice.transport.types import Request, Response

__all__ = ["Pipeline", "PipelineTransformRegistry", "ReversibleSyncTransform", "TransformClass"]

_logger = structlog.get_logger()


class Pipeline:
    """Verbatim pipeline — executes plan.transforms in order without re-decision."""

    _IR_NATIVE_TRANSFORMS = IR_NATIVE_TRANSFORMS

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
        """Execute ExecutionPlan transforms in order."""
        return pipeline_process(self, request, plan, context)

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
        """High-level entry: gate-orchestrated compression."""
        from lattice.pipeline.executor import _serialize_ir_to_messages

        cfg = config or self.config
        log = _logger.bind(module="pipeline_compress", request_id=context.request_id)

        working = request.copy()
        original_token_estimate = working.token_estimate

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
                result = profiler.optimize(ir_seed, working, context)
                if is_ok(result):
                    ir_v2 = unwrap(result)
                    working.metadata["_lattice_ir_v2"] = ir_v2
                    context.session_state["_lattice_ir_v2"] = ir_v2
                if is_ok(result):
                    context.mark_transform_applied("content_profiler")
            except Exception as exc:
                log.warning("content_profiler_failed", error=str(exc))

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

            g = _gates.runtime_budget_blocks(tx_name, working, context, cumulative_transform_ms)
            if g.skip:
                context.record_metric(tx_name, "deferred", True)
                context.record_metric(tx_name, "deferred_reason", g.reason)
                continue

            g = _gates.risk_gate_blocks(tx_name, working, context, profiler_present)
            if g.skip:
                context.record_metric(tx_name, "risk_blocked", True)
                context.record_metric(tx_name, "risk_block_reason", g.reason)
                blocked = context.metrics.setdefault("risk_blocked_transforms", [])
                if isinstance(blocked, list):
                    blocked.append(tx_name)
                continue

            g = _gates.protected_span_veto(tx_name, working, context)
            if g.veto:
                context.record_metric(tx_name, "spans_vetoed", True)
                continue

            g = _gates.scheduler_blocks(tx_name, working, context)
            if g.skip:
                context.record_metric(tx_name, "scheduler_blocked", True)
                context.record_metric(tx_name, "deferred", True)
                context.record_metric(tx_name, "deferred_reason", g.reason)
                continue

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

            task_data = get_canonical_request_value(
                working, context, "_lattice_task_classification", {}
            )
            has_task = (
                bool(task_data)
                and isinstance(task_data, dict)
                and bool(task_data.get("task_class"))
            )
            if _gates.should_run_post_transform_guard(
                tx_name, text_before, text_after, tokens_before, has_task
            ):
                from lattice.pipeline.post_transform_guard import (
                    evaluate_post_transform,
                    should_trigger_post_transform_guard,
                )
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
                if should_trigger_post_transform_guard(
                    tx_name,
                    tc,
                    compression_ratio=compression,
                    placeholder_aliasing_used=False,
                ):
                    guard_result = evaluate_post_transform(
                        text_before,
                        text_after,
                        tc,
                        placeholder_aliasing_used=False,
                    )
                    context.record_metric(tx_name, "post_transform_guard_triggered", True)
                    context.record_metric(tx_name, "post_transform_guard_score", guard_result.score)
                    context.record_metric(tx_name, "post_transform_guard_passed", guard_result.passed)
                    if not guard_result.passed:
                        _rollback(
                            f"post_guard_rejected:{guard_result.reason}",
                            code="POST_TRANSFORM_GUARD_REJECTED",
                        )
                        if not cfg.graceful_degradation:
                            return Err(
                                TransformError(
                                    transform=tx_name,
                                    code="POST_TRANSFORM_GUARD_REJECTED",
                                    message=(
                                        f"Post-transform guard rejected: "
                                        f"score={guard_result.score:.2f}, {guard_result.reason}"
                                    ),
                                )
                            )
                        continue

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
        """Reverse transforms in reverse order."""
        return pipeline_reverse(self, response, context, plan=plan)
