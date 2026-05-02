"""Compressor pipeline — orchestrates transforms.

The pipeline receives a Request, applies enabled transforms in priority order,
and returns the modified Request or an error. It handles:
- Transform ordering by priority
- Metrics collection per transform
- Graceful degradation on transform failure
- Timeout enforcement via cancellation
"""

from __future__ import annotations

import inspect
import time
from typing import Any

import structlog

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.policy import OptimizationPolicy, Reject, Skip
from lattice.core.result import Err, Ok, Result, is_err, unwrap, unwrap_err
from lattice.core.transport import Request, Response

logger = structlog.get_logger()


# =============================================================================
# ReversibleSyncTransform base class
# =============================================================================


class ReversibleSyncTransform:
    """Base class for transforms that can be reversed.

    Subclasses should implement `process` (forward) and `reverse` (backward).
    The pipeline will call reverse for each applied transform in reverse order.

    This is an ABC, NOT a Protocol, because it carries shared behavior.
    """

    name: str = ""
    enabled: bool = True
    priority: int = 50

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        """Forward pass: compress/optimize the request."""
        raise NotImplementedError

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        """Reverse pass: restore original values in the response.

        Only called if this transform was applied during the forward pass.
        The implementation should use context.session_state[own_name] to
        retrieve its saved state.
        """
        return response

    def can_process(self, _request: Request, _context: TransformContext) -> bool:
        """Check whether this transform can handle the request."""
        return self.enabled


# =============================================================================
# CompressorPipeline
# =============================================================================


class CompressorPipeline:
    """Orchestrates transform execution.

    The pipeline holds a registry of transforms. When `process` is called,
    it sorts transforms by priority (ascending = lower numbers first),
    runs them sequentially, and collects metrics.

    Attributes:
        transforms: Ordered list of transforms (maintained sorted).
        config: Lattice configuration for graceful degradation, timeouts, etc.
    """

    transforms: list[ReversibleSyncTransform]
    config: LatticeConfig
    _budget_sensitive_transforms: frozenset[str] = frozenset(
        {
            "strategy_selector",
            "structural_fingerprint",
            "self_information",
            "context_selector",
            "information_theoretic_selector",
            "rate_distortion",
            "hierarchical_summary",
        }
    )
    _irreversible_transforms: frozenset[str] = frozenset(
        {
            "message_dedup",
            "rate_distortion",
            "semantic_compress",
            "structural_fingerprint",
            "hierarchical_summary",
        }
    )

    def __init__(
        self,
        transforms: list[ReversibleSyncTransform] | None = None,
        config: LatticeConfig | None = None,
        policy: OptimizationPolicy | None = None,
    ) -> None:
        """Initialize pipeline with transforms, config, and optional policy.

        Args:
            transforms: Initial list of transforms (optional).
            config: LatticeConfig for settings. Defaults to auto().
            policy: OptimizationPolicy for runtime decisions. Defaults to
                a new policy using the provided config.
        """
        self.transforms = list(transforms) if transforms else []
        self.config = config or LatticeConfig()
        # Policy: provided or built from config. Central brain for decisions.
        self.policy = policy or OptimizationPolicy(self.config)
        # Ensure sorted by priority
        self.transforms.sort(key=lambda t: t.priority)
        self._log = logger.bind(module="compressor_pipeline")

    def register(self, transform: ReversibleSyncTransform) -> None:
        """Register a new transform and re-sort by priority.

        Args:
            transform: Instance to add to the pipeline.
        """
        self.transforms.append(transform)
        self.transforms.sort(key=lambda t: t.priority)
        self._log.debug(
            "transform_registered",
            name=transform.name,
            priority=transform.priority,
            total_transforms=len(self.transforms),
        )

    def unregister(self, name: str) -> bool:
        """Remove a transform by name.

        Returns:
            True if removed, False if not found.
        """
        before = len(self.transforms)
        self.transforms = [t for t in self.transforms if t.name != name]
        removed = len(self.transforms) != before
        if removed:
            self._log.debug("transform_unregistered", name=name)
        return removed

    async def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        """Run all enabled transforms on the request.

        Algorithm:
        1. Deep copy the request (so we can rollback if a transform fails).
        2. Sort transforms by priority.
        3. For each transform:
           a. Skip if not enabled in config.
           b. Skip if can_process returns False.
           c. Record start time.
           d. Call transform.process.
           e. If error:
              - If graceful degradation: log warning, rollback, continue.
              - If strict mode: return error immediately.
           f. Record metrics.
        4. Return modified request.

        Args:
            request: The incoming request.
            context: Mutable per-request state.

        Returns:
            Ok(Request) with transforms applied, or Err(TransformError) on failure.
        """
        working = request.copy()
        original_token_estimate = working.token_estimate

        # ---- Global request limits (policy gate) ----
        limits = self.policy.check_request_limits(working)
        if isinstance(limits, Reject):
            self._log.warning(
                "request_rejected_by_policy",
                request_id=context.request_id,
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

        # Track pre-transform state for rollback
        backup = working.copy()
        cumulative_transform_ms = 0.0

        for transform in self.transforms:
            # Skip if not enabled
            if not getattr(transform, "enabled", True):
                continue

            # Skip if config disables this transform
            if not self.config.is_transform_enabled(transform.name):
                self._log.debug(
                    "transform_skipped_config",
                    request_id=context.request_id,
                    transform=transform.name,
                )
                continue

            # Skip if transform says it can't handle this request
            if not transform.can_process(working, context):
                continue

            # ---- Policy engine decision ----
            decision = self.policy.should_run(transform.name, working, context)
            if isinstance(decision, Skip):
                self._log.debug(
                    "transform_skipped_policy",
                    request_id=context.request_id,
                    transform=transform.name,
                    reason=decision.reason,
                )
                continue
            elif isinstance(decision, Reject):
                self._log.warning(
                    "transform_rejected_policy",
                    request_id=context.request_id,
                    transform=transform.name,
                    code=decision.code,
                    message=decision.message,
                )
                if self.config.graceful_degradation:
                    working = backup.copy()
                    continue
                return Err(
                    TransformError(
                        transform=transform.name,
                        code=decision.code,
                        message=decision.message,
                        detail=decision.detail,
                    )
                )

            runtime_budget_ms = self._runtime_budget_ms(working)
            if (
                runtime_budget_ms > 0
                and transform.name != "runtime_contract"
                and transform.name in self._budget_sensitive_transforms
                and cumulative_transform_ms >= runtime_budget_ms
            ):
                skipped = context.metrics.setdefault("runtime_budget_skipped", [])
                if isinstance(skipped, list):
                    skipped.append(transform.name)
                context.record_metric("pipeline", "runtime_budget_exhausted", True)
                context.record_metric(
                    "pipeline",
                    "runtime_budget_ms",
                    runtime_budget_ms,
                )
                context.record_metric(
                    "pipeline",
                    "runtime_transform_ms",
                    round(cumulative_transform_ms, 3),
                )
                self._log.debug(
                    "transform_skipped_runtime_budget",
                    request_id=context.request_id,
                    transform=transform.name,
                    runtime_budget_ms=runtime_budget_ms,
                    cumulative_transform_ms=round(cumulative_transform_ms, 3),
                )
                continue

            # ---- Semantic risk gate ----
            # Only gate when content_profiler is registered in this pipeline.
            # If registered but no risk data: conservative fallback (SAFE only).
            # If not registered: no gating (user chose to run without it).
            from lattice.utils.validation import (
                SemanticRiskScore,
                TransformSafetyBucket,
                get_transform_safety_bucket,
                transform_allowed_at_risk,
            )

            profiler_present = any(t.name == "content_profiler" for t in self.transforms)
            if profiler_present:
                risk_data = working.metadata.get("_lattice_risk_score")

                if risk_data and isinstance(risk_data, dict):
                    risk = SemanticRiskScore(
                        strict_instructions=float(risk_data.get("strict_instructions", 0)),
                        sensitive_domain=float(risk_data.get("sensitive_domain", 0)),
                        structured_output=float(risk_data.get("structured_output", 0)),
                        high_stakes_entities=float(risk_data.get("high_stakes_entities", 0)),
                        reasoning_heavy=float(risk_data.get("reasoning_heavy", 0)),
                        intentional_repetition=float(risk_data.get("intentional_repetition", 0)),
                        tool_call_dependency=float(risk_data.get("tool_call_dependency", 0)),
                        formatting_constraints=float(risk_data.get("formatting_constraints", 0)),
                    )
                    allowed, reason = transform_allowed_at_risk(transform.name, risk)
                else:
                    # Profiler registered but no risk data — only SAFE transforms
                    bucket = get_transform_safety_bucket(transform.name)
                    allowed = bucket == TransformSafetyBucket.SAFE
                    reason = "safe_transform" if allowed else "profiler_failed_no_risk_data"

                if not allowed:
                    self._log.warning(
                        "transform_blocked_by_risk_gate",
                        request_id=context.request_id,
                        transform=transform.name,
                        risk_level=risk.level
                        if risk_data and isinstance(risk_data, dict)
                        else "unknown",
                        reason=reason,
                    )
                    context.record_metric(transform.name, "risk_blocked", True)
                    context.record_metric(transform.name, "risk_block_reason", reason)
                    blocked = context.metrics.setdefault("risk_blocked_transforms", [])
                    if isinstance(blocked, list):
                        blocked.append(transform.name)
                    continue
            # ---- End risk gate ----

            # ---- Protected-span pre-execution veto ----
            # DANGEROUS transforms are vetoed when protected spans exist.
            # CONDITIONAL transforms are handled by the risk gate + scheduler;
            # a universal veto on CONDITIONAL requires more precise SIG span
            # protection (future: per-transform span-safety declarations).
            bucket_at_veto = get_transform_safety_bucket(transform.name)
            if bucket_at_veto == TransformSafetyBucket.DANGEROUS:
                protected = working.metadata.get("_lattice_protected_spans", [])
                if protected:
                    self._log.info(
                        "transform_vetoed_by_protected_spans",
                        request_id=context.request_id,
                        transform=transform.name,
                        bucket=bucket_at_veto.value,
                        protected_count=len(protected),
                    )
                    context.record_metric(transform.name, "spans_vetoed", True)
                    continue
            # ---- End protected-span veto ----

            # ---- Span-aware gating (SIG) ----
            # Only gate when scheduler has explicitly built a schedule with
            # allowed/blocked transforms. Empty/missing schedule = no gating.
            schedule = working.metadata.get("_lattice_schedule")
            if schedule and isinstance(schedule, dict):
                blocked_names = schedule.get("blocked", [])
                if blocked_names and transform.name in blocked_names:
                    self._log.info(
                        "transform_blocked_by_scheduler",
                        request_id=context.request_id,
                        transform=transform.name,
                    )
                    context.record_metric(transform.name, "scheduler_blocked", True)
                    # Record deferred reason for reachability — important
                    # for rate_distortion/context_selector which may be
                    # reached but deferred by scheduler
                    context.record_metric(transform.name, "deferred", True)
                    for entry in schedule.get("schedule", []):
                        if isinstance(entry, dict) and entry.get("name") == transform.name:
                            context.record_metric(
                                transform.name,
                                "deferred_reason",
                                entry.get("reason", "scheduler_blocked"),
                            )
                            break
                    continue
            # ---- End span-aware gating ----

            # Execute
            start = time.perf_counter()
            try:
                # Handle both sync and async transforms
                if inspect.iscoroutinefunction(transform.process):
                    result = await transform.process(working, context)
                else:
                    result = transform.process(working, context)
            except Exception as exc:
                self._log.warning(
                    "transform_exception",
                    request_id=context.request_id,
                    transform=transform.name,
                    error=str(exc),
                )
                if self.config.graceful_degradation:
                    working = backup.copy()
                    continue
                return Err(
                    TransformError(
                        transform=transform.name,
                        code="PIPELINE_EXECUTION_ERROR",
                        message=f"Transform raised exception: {exc}",
                        detail={"exception": str(exc)},
                    )
                )

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            cumulative_transform_ms += elapsed_ms

            if is_err(result):
                err = unwrap_err(result)
                self._log.warning(
                    "transform_failed",
                    request_id=context.request_id,
                    transform=transform.name,
                    code=err.code,
                    message=err.message,
                )
                if self.config.graceful_degradation:
                    working = backup.copy()
                    continue
                return Err[Request, TransformError](err)

            # Success — check for intermediate expansion before accepting
            working = unwrap(result)
            working_tokens = working.token_estimate

            # ---- Intermediate expansion guardrail ----
            tokens_before = backup.token_estimate
            if tokens_before > 0:
                expansion_ratio = working_tokens / tokens_before
                max_ratio = getattr(self.config, "max_transform_expansion_ratio", 1.5)
                if expansion_ratio > max_ratio:
                    self._log.warning(
                        "transform_expansion_aborted",
                        request_id=context.request_id,
                        transform=transform.name,
                        tokens_before=tokens_before,
                        tokens_after=working_tokens,
                        expansion_ratio=round(expansion_ratio, 2),
                        max_ratio=max_ratio,
                    )
                    context.record_metric(transform.name, "expansion_aborted", True)
                    context.record_metric(
                        transform.name, "expansion_ratio", round(expansion_ratio, 2)
                    )
                    working = backup.copy()
                    continue
            # ---- End expansion guardrail ----

            # ---- Compression limit guard (REASONING tier) ----
            task_data = working.metadata.get("_lattice_task_classification", {})
            tier = task_data.get("execution_tier", "") if isinstance(task_data, dict) else ""
            if tier in ("REASONING", "REASONING_SAFE") and tokens_before > 0:
                compression_ratio = (tokens_before - working_tokens) / tokens_before
                if compression_ratio > 0.10 and transform.name not in (
                    "content_profiler",
                    "runtime_contract",
                ):
                    self._log.warning(
                        "transform_compression_limit_exceeded",
                        request_id=context.request_id,
                        transform=transform.name,
                        compression_ratio=round(compression_ratio, 2),
                        tier=tier,
                    )
                    context.record_metric(transform.name, "compression_limited", True)
                    if self.config.graceful_degradation:
                        working = backup.copy()
                        continue
                    return Err(
                        TransformError(
                            transform=transform.name,
                            code="PSG_REASONING_COMPRESSION_LIMIT",
                            message=(
                                f"Compression ratio {compression_ratio:.2f} exceeds "
                                f"reasoning tier limit (0.10)"
                            ),
                        )
                    )
            # ---- End compression limit guard ----

            # ---- PSG safety check ----
            # Entity/format checks only for irreversible transforms that
            # genuinely discard content. Reversible transforms (reference_sub,
            # dictionary_compress, grammar_compress) store referent mappings.
            if transform.name in self._irreversible_transforms:
                from lattice.core.guardrails import (
                    check_critical_signal_loss,
                    check_entity_preservation,
                    check_format_preservation,
                )

                protected_spans: list[int] = working.metadata.get("_lattice_protected_spans", [])
                if protected_spans:
                    text_before = "\n".join(m.content for m in backup.messages)
                    text_after = "\n".join(m.content for m in working.messages)
                    entity_decision = check_entity_preservation(text_before, text_after)
                    if entity_decision.action.value == "rollback":
                        self._log.warning(
                            "transform_entity_loss_rollback",
                            request_id=context.request_id,
                            transform=transform.name,
                            reason=entity_decision.reason,
                        )
                        context.record_metric(transform.name, "safety_rollback", True)
                        context.record_metric(
                            transform.name, "rollback_reason", entity_decision.reason
                        )
                        working = backup.copy()
                        working.metadata["_lattice_rollback_reason"] = entity_decision.reason
                        if self.config.graceful_degradation:
                            continue
                        return Err(
                            TransformError(
                                transform=transform.name,
                                code="PSG_ENTITY_LOSS",
                                message=f"Entity preservation failed: {entity_decision.reason}",
                            )
                        )
                    fmt_decision = check_format_preservation(text_before, text_after)
                    if fmt_decision.action.value == "rollback":
                        self._log.warning(
                            "transform_format_loss_rollback",
                            request_id=context.request_id,
                            transform=transform.name,
                            reason=fmt_decision.reason,
                        )
                        context.record_metric(transform.name, "safety_rollback", True)
                        context.record_metric(
                            transform.name, "rollback_reason", fmt_decision.reason
                        )
                        working = backup.copy()
                        working.metadata["_lattice_rollback_reason"] = fmt_decision.reason
                        if self.config.graceful_degradation:
                            continue
                        return Err(
                            TransformError(
                                transform=transform.name,
                                code="PSG_FORMAT_LOSS",
                                message=f"Format preservation failed: {fmt_decision.reason}",
                            )
                        )
                    sig_decision = check_critical_signal_loss(text_before, text_after)
                    if sig_decision.action.value == "rollback":
                        self._log.warning(
                            "transform_critical_signal_loss",
                            request_id=context.request_id,
                            transform=transform.name,
                            reason=sig_decision.reason,
                        )
                        context.record_metric(transform.name, "safety_rollback", True)
                        context.record_metric(
                            transform.name, "rollback_reason", sig_decision.reason
                        )
                        working = backup.copy()
                        working.metadata["_lattice_rollback_reason"] = sig_decision.reason
                        if self.config.graceful_degradation:
                            continue
                        return Err(
                            TransformError(
                                transform=transform.name,
                                code="PSG_CRITICAL_SIGNAL_LOSS",
                                message=f"Critical signal loss: {sig_decision.reason}",
                            )
                        )
            # ---- End PSG safety check ----

            backup = working.copy()
            context.mark_transform_applied(transform.name)

            # Record metrics
            context.record_metric(transform.name, "tokens_before", tokens_before)
            context.record_metric(transform.name, "tokens_after", working_tokens)
            context.record_metric(transform.name, "latency_ms", round(elapsed_ms, 3))

            self._log.debug(
                "transform_applied",
                request_id=context.request_id,
                transform=transform.name,
                latency_ms=round(elapsed_ms, 3),
            )

        # Final token count after all transforms
        final_tokens = working.token_estimate
        budget_skipped = context.metrics.get("runtime_budget_skipped", [])
        budget_skipped_count = len(budget_skipped) if isinstance(budget_skipped, list) else 0
        risk_blocked = context.metrics.get("risk_blocked_transforms", [])
        runtime_budget_ms = self._runtime_budget_ms(working)

        # ---- PSG explainability ----
        safety_reasons: dict[str, Any] = {}
        for t_name in dir(context.metrics.get("transforms", {})):
            pass  # collected below
        transforms_metrics = context.metrics.get("transforms", {})
        if isinstance(transforms_metrics, dict):
            for t_name, t_metrics in transforms_metrics.items():
                if isinstance(t_metrics, dict):
                    if t_metrics.get("safety_rollback"):
                        safety_reasons[t_name] = t_metrics.get("rollback_reason", "unknown")
                    if t_metrics.get("expansion_aborted"):
                        safety_reasons[t_name] = (
                            f"expansion_{t_metrics.get('expansion_ratio', '?')}x"
                        )
                    if t_metrics.get("risk_blocked"):
                        safety_reasons[t_name] = t_metrics.get("risk_block_reason", "risk_gate")
                    if t_metrics.get("scheduler_blocked"):
                        safety_reasons[t_name] = "scheduler_blocked"

        # Record PSG outcomes in metadata for reports
        working.metadata["_lattice_safety_decision"] = {
            "applied": list(context.transforms_applied),
            "skipped": list(budget_skipped) if isinstance(budget_skipped, list) else [],
            "risk_blocked": list(risk_blocked) if isinstance(risk_blocked, list) else [],
            "rollback_reasons": safety_reasons,
        }
        # ---- End PSG explainability ----

        # ---- Reached / Activated / Useful telemetry ----
        # reached: transform passed all gates (config, policy, risk, scheduler, span veto)
        # activated: transform changed the request (tokens changed)
        # useful: activated AND no safety rollback AND tokens saved
        transforms_reached: list[str] = []
        transforms_activated: list[str] = []
        transforms_useful: list[str] = []
        for t_name, t_metrics in (
            transforms_metrics.items() if isinstance(transforms_metrics, dict) else {}
        ):
            if isinstance(t_metrics, dict):
                tokens_before = t_metrics.get("tokens_before", 0)
                tokens_after = t_metrics.get("tokens_after", 0)
                changed = tokens_before != tokens_after and tokens_after > 0
                has_safety_issue = t_metrics.get("safety_rollback") or t_metrics.get(
                    "expansion_aborted"
                )
                was_blocked = t_metrics.get("risk_blocked") or t_metrics.get("scheduler_blocked")
                if not was_blocked:
                    transforms_reached.append(t_name)
                if not was_blocked and changed:
                    transforms_activated.append(t_name)
                if (
                    not was_blocked
                    and changed
                    and not has_safety_issue
                    and tokens_after < tokens_before
                ):
                    transforms_useful.append(t_name)

        working.metadata["_lattice_reachability"] = {
            "reached": transforms_reached,
            "activated": transforms_activated,
            "useful": transforms_useful,
            "reached_count": len(transforms_reached),
            "activated_count": len(transforms_activated),
            "useful_count": len(transforms_useful),
        }
        # ---- End reachability telemetry ----
        budget_exhausted = budget_skipped_count > 0
        working.metadata["_lattice_runtime_budget"] = {
            "exhausted": budget_exhausted,
            "skipped_count": budget_skipped_count,
            "skipped_transforms": list(budget_skipped) if isinstance(budget_skipped, list) else [],
            "actual_transform_ms": round(cumulative_transform_ms, 3),
            "budget_ms": runtime_budget_ms,
        }
        context.metrics["tokens_in"] = original_token_estimate
        context.metrics["tokens_out"] = final_tokens
        context.metrics["latency_ms"] = context.elapsed_ms
        context.metrics["transform_latency_ms"] = round(cumulative_transform_ms, 3)

        self._log.info(
            "pipeline_complete",
            request_id=context.request_id,
            transforms_applied=context.transforms_applied,
            tokens_before=original_token_estimate,
            tokens_after=final_tokens,
            compression_ratio=round(
                (original_token_estimate - final_tokens) / max(original_token_estimate, 1), 4
            ),
        )

        return Ok[Request, TransformError](working)

    @staticmethod
    def _runtime_budget_ms(request: Request) -> float:
        contract = request.metadata.get("_lattice_runtime_contract")
        if not isinstance(contract, dict):
            return 0.0
        value = contract.get("max_transform_latency_ms")
        if isinstance(value, int | float):
            return max(0.0, float(value))
        return 0.0

    async def reverse(self, response: Response, context: TransformContext) -> Response:
        """Reverse compress transforms on the response.

        Iterates `transforms_applied` in reverse order and calls reverse()
        on any ReversibleSyncTransform.

        Args:
            response: The provider's response.
            context: Mutable per-request state.

        Returns:
            The response with references expanded, if applicable.
        """
        for name in reversed(context.transforms_applied):
            # Find the transform by name
            for transform in self.transforms:
                if transform.name == name and isinstance(transform, ReversibleSyncTransform):
                    try:
                        response = transform.reverse(response, context)
                    except Exception as exc:
                        self._log.warning(
                            "reverse_transform_failed",
                            transform=name,
                            error=str(exc),
                            request_id=context.request_id,
                        )
                        # Reverse failures are non-fatal — passthrough
                    break
        return response
