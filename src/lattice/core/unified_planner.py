"""core/unified_planner.py — Single source of truth for execution planning.

Replaces:
- core/scheduler.py (RATS)
- core/optimizer_scheduler.py
- planner/execution_builder.py

This is the ONLY module that decides "what runs, in what order, with what budget."

The pipeline executes ExecutionPlan verbatim — no runtime re-decision.
No committee. No committee override. No hidden logic.

Usage:
    plan = UnifiedPlanner().plan(request, profile, config)
    pipeline.execute(plan, context)  # executes verbatim
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any

from lattice.core.task_classifier import TaskClass
from lattice.core.transport import Request
from lattice.ir.primitives import ExecutionPlan


class Tier(enum.Enum):
    """Execution tier derived from task complexity + risk."""

    FAST = "fast"  # Simple requests, low latency
    STANDARD = "standard"  # Normal requests
    SAFE = "safe"  # Conservative, no lossy
    REASONING = "reasoning"  # Reasoning/debugging, extra conservative
    EMERGENCY = "emergency"  # High risk, transforms disabled


@dataclasses.dataclass(frozen=True, slots=True)
class SemanticProfile:
    """Immutable profile produced by semantic analysis.

    This is the SINGLE input to the planner.
    No other module should produce scheduling decisions.
    """

    task_class: TaskClass
    task_label: str = ""
    risk_total: int = 0
    context_length: int = 0
    has_tool_calls: bool = False
    is_streaming: bool = False
    is_conservative: bool = False
    provider: str = "generic"
    model: str = ""

    @property
    def is_debugging(self) -> bool:
        return self.task_class == TaskClass.DEBUGGING

    @property
    def is_reasoning(self) -> bool:
        return self.task_class == TaskClass.REASONING

    @property
    def is_high_risk(self) -> bool:
        return self.risk_total > 50 or self.is_conservative


# ──────────────────────────────────────────────────────────────────
# Unified planner
# ──────────────────────────────────────────────────────────────────


class UnifiedPlanner:
    """Produces ONE ExecutionPlan per request.

    The plan includes:
    - Ordered list of transforms
    - Quality floor
    - Latency budget
    - Cache plan (optional)
    - Transport plan (optional)
    """

    # Canonical request-side transform ordering (response-only excluded).
    _TRANSFORM_ORDER: tuple[str, ...] = (
        "content_profiler",  # 1 — always runs first
        "runtime_contract",  # 2
        "constraint_lifting",  # 6
        "cache_arbitrage",  # 9
        "causal_chain",  # 9
        "prefix_optimizer",  # 10
        "message_dedup",  # 15
        "diagnostic_optimizer",  # 17
        "strategy_selector",  # 19 (bandit arms)
        "representation_optimizer",  # 19 (beam search orchestrator)
        "structure_optimizer",  # 20
        "ir_structure_optimizer",  # 20 (IR-native pair)
        "reference_optimizer",  # 21
        "rate_distortion",  # 22
        "path_prefix",  # 23
        "format_conversion",  # 25
        "tool_projection",  # 29
        "context_optimizer",  # 22 (lossy, gated)
        "tool_optimizer",  # 30
        "reference_sub",  # 20
        "tool_filter",  # 30
    )

    # Safety mapping: which transforms are safe at which tier.
    # output_cleanup is response-only and excluded from request-side plans.
    _TIER_ALLOWED: dict[Tier, set[str]] = {
        Tier.FAST: {
            "content_profiler",
            "runtime_contract",
            "message_dedup",
            "strategy_selector",
            "cache_arbitrage",
            "prefix_optimizer",
            "reference_optimizer",
            "structure_optimizer",
            "ir_structure_optimizer",
            "tool_optimizer",
            "reference_sub",
            "tool_filter",
        },
        Tier.STANDARD: {
            "content_profiler",
            "runtime_contract",
            "message_dedup",
            "strategy_selector",
            "cache_arbitrage",
            "prefix_optimizer",
            "diagnostic_optimizer",
            "representation_optimizer",
            "structure_optimizer",
            "ir_structure_optimizer",
            "reference_optimizer",
            "rate_distortion",
            "path_prefix",
            "format_conversion",
            "tool_projection",
            "tool_optimizer",
            "reference_sub",
            "tool_filter",
        },
        Tier.SAFE: {
            "content_profiler",
            "runtime_contract",
            "message_dedup",
            "strategy_selector",
            "cache_arbitrage",
            "prefix_optimizer",
            "diagnostic_optimizer",
            "representation_optimizer",
            "structure_optimizer",
            "ir_structure_optimizer",
            "reference_optimizer",
            "rate_distortion",
            "format_conversion",
            "context_optimizer",
            "tool_optimizer",
            "reference_sub",
            "tool_filter",
        },
        Tier.REASONING: {
            "content_profiler",
            "runtime_contract",
            "strategy_selector",
            "cache_arbitrage",
            "prefix_optimizer",
            "diagnostic_optimizer",
            "representation_optimizer",
            "structure_optimizer",
            "ir_structure_optimizer",
            "reference_optimizer",
            "format_conversion",
            "tool_optimizer",
            "reference_sub",
            "tool_filter",
        },
        Tier.EMERGENCY: {
            "content_profiler",
            "runtime_contract",
        },
    }

    _TIER_BUDGET_MS: dict[Tier, float] = {
        Tier.FAST: 30.0,
        Tier.STANDARD: 100.0,
        Tier.SAFE: 80.0,
        Tier.REASONING: 60.0,
        Tier.EMERGENCY: 10.0,
    }

    _QUALITY_FLOOR: dict[TaskClass, float] = {
        TaskClass.REASONING: 0.92,
        TaskClass.DEBUGGING: 0.90,
        TaskClass.ANALYSIS: 0.88,
        TaskClass.STRUCTURED: 0.87,
        TaskClass.RETRIEVAL: 0.85,
        TaskClass.SUMMARIZATION: 0.85,
        TaskClass.SIMPLE: 0.80,
    }

    # Mapping: which profiles trigger which tier
    def classify_tier(self, profile: SemanticProfile) -> Tier:
        """Map profile to execution tier."""
        if profile.risk_total > 60 or profile.is_conservative and profile.risk_total > 30:
            return Tier.EMERGENCY
        if profile.is_reasoning or profile.is_debugging:
            return Tier.REASONING
        if profile.context_length < 500 and not profile.is_high_risk:
            return Tier.FAST
        if profile.is_high_risk:
            return Tier.SAFE
        return Tier.STANDARD

    def plan(
        self,
        request: Request,
        profile: SemanticProfile,
        config: Any | None = None,
    ) -> ExecutionPlan:
        """Produce ONE ExecutionPlan.

        This is the ONLY scheduling decision maker in the entire system.
        """
        tier = self.classify_tier(profile)
        allowed = self._TIER_ALLOWED[tier]

        # Filter to those registered in our ordered list
        transforms = tuple(name for name in self._TRANSFORM_ORDER if name in allowed)

        # Contextually add conditional transforms
        if profile.has_tool_calls and "tool_optimizer" in allowed:
            transforms = self._insert_after(transforms, "reference_optimizer", "tool_optimizer")

        if profile.context_length > 4000:
            if not profile.is_reasoning and not profile.is_debugging:
                transforms = self._insert_after(
                    transforms, "reference_optimizer", "context_optimizer"
                )

        # Derive budget and quality floor
        quality_floor = self._QUALITY_FLOOR.get(profile.task_class, 0.85)
        budget_ms = self._TIER_BUDGET_MS[tier]

        # Streaming adjusts budget
        if profile.is_streaming:
            budget_ms = min(budget_ms, 50.0)

        utility_score = self._estimate_utility(profile, transforms, budget_ms)

        return ExecutionPlan(
            transforms=transforms,
            quality_floor=quality_floor,
            latency_budget_ms=budget_ms,
            utility_score=utility_score,
            provider=profile.provider,
            model=profile.model,
        )

    @staticmethod
    def _insert_after(transforms: tuple[str, ...], after: str, what: str) -> tuple[str, ...]:
        """Insert *what* after the first occurrence of *after*.

        If *after* is not in the transforms list, *what* is NOT inserted.
        This prevents orphaned transforms when the anchor is tier-excluded.
        """
        lst = list(transforms)
        if what in lst:
            return transforms
        if after not in lst:
            # Anchor not present — don't insert orphan
            return transforms
        idx = lst.index(after)
        lst.insert(idx + 1, what)
        return tuple(lst)

    def _estimate_utility(
        self,
        profile: SemanticProfile,
        transforms: tuple[str, ...],
        budget_ms: float,
    ) -> float:
        """Estimate expected utility for the selected execution plan."""
        utility = 0.0

        if "representation_optimizer" in transforms:
            utility += 0.15
        if "structure_optimizer" in transforms or "ir_structure_optimizer" in transforms:
            utility += 0.12
        if "reference_optimizer" in transforms:
            utility += 0.10
        if "tool_optimizer" in transforms:
            utility += 0.08
        if "context_optimizer" in transforms:
            utility += 0.06
        if "diagnostic_optimizer" in transforms:
            utility += 0.04

        if profile.has_tool_calls:
            utility += 0.05
        if profile.context_length > 4000:
            utility += 0.05
        if profile.is_streaming:
            utility += 0.03

        if profile.is_high_risk:
            utility -= 0.12
        if profile.is_conservative:
            utility -= 0.08

        utility += min(0.08, max(0.0, 100.0 - budget_ms) / 1000.0)
        return round(utility, 4)


# ──────────────────────────────────────────────────────────────────
# Compatibility: bridge from legacy profile to SemanticProfile
# ──────────────────────────────────────────────────────────────────


def profile_from_legacy(legacy: Any) -> SemanticProfile | None:
    """Convert a legacy profile dict/object to SemanticProfile."""
    if legacy is None:
        return None

    if isinstance(legacy, dict):
        getter = legacy.get
    else:

        def getter(key: str, default: Any = None) -> Any:
            return getattr(legacy, key, default)

    task_cls = getter("task_class", None)
    task_label = getter("task_label", "")
    risk_total = getter("risk_total", 0)
    context_length = getter("context_length", 0)
    has_tool_calls = getter("has_tool_calls", False)
    is_streaming = getter("is_streaming", False)
    is_conservative = getter("is_conservative", False)
    provider = getter("provider", "generic")
    model = getter("model", "")

    if isinstance(task_cls, dict):
        nested = task_cls
        task_cls = nested.get("task_class", nested.get("task", None))
        task_label = nested.get("task_label", task_label)
        risk_total = nested.get("risk_total", risk_total)
        context_length = nested.get("context_length", context_length)
        has_tool_calls = nested.get("has_tool_calls", has_tool_calls)
        is_streaming = nested.get("is_streaming", is_streaming)
        is_conservative = nested.get("is_conservative", is_conservative)
        provider = nested.get("provider", provider)
        model = nested.get("model", model)

    if task_cls is None:
        task_cls = TaskClass.SIMPLE
    elif isinstance(task_cls, str):
        try:
            task_cls = TaskClass(task_cls)
        except ValueError:
            task_cls = TaskClass.SIMPLE

    return SemanticProfile(
        task_class=task_cls,
        task_label=task_label,
        risk_total=risk_total,
        context_length=context_length,
        has_tool_calls=has_tool_calls,
        is_streaming=is_streaming,
        is_conservative=is_conservative,
        provider=provider,
        model=model,
    )
