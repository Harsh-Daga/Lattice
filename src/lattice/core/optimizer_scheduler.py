"""OptimizerScheduler — compatibility wrapper over UnifiedPlanner.

The production planner is UnifiedPlanner. This module remains as a thin
adapter for older consumers that still expect optimizer-level allow/block
metadata.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

from lattice.core.task_classifier import TaskClass, TaskClassification
from lattice.core.unified_planner import SemanticProfile, UnifiedPlanner
from lattice.planner.execution_plan import tier_budget_ms


@dataclasses.dataclass(slots=True)
class OptimizerSchedule:
    """Simplified scheduler output at optimizer granularity."""

    tier: str
    latency_budget_ms: float
    quality_floor: float
    allowed_optimizers: list[str]
    blocked_optimizers: dict[str, str]
    transport_enabled: bool
    cache_enabled: bool


def decide_optimizer_schedule(
    task: TaskClassification,
    *,
    risk_total: float = 0,
    context_length: int = 0,
    has_tool_calls: bool = False,
    is_streaming: bool = False,
    provider: str = "generic",
) -> OptimizerSchedule:
    """Return optimizer allow/block metadata derived from UnifiedPlanner."""
    profile = SemanticProfile(
        task_class=task.task_class,
        task_label=task.execution_tier.value,
        risk_total=int(risk_total),
        context_length=context_length,
        has_tool_calls=has_tool_calls,
        is_streaming=is_streaming,
        is_conservative=task.is_conservative,
        provider=provider,
    )
    plan = UnifiedPlanner().plan(cast(Any, RequestShim(profile)), profile)

    tier = task.execution_tier.value
    budget = tier_budget_ms(tier)
    quality = plan.quality_floor

    allowed = [name for name in plan.transforms if name.endswith("_optimizer")]
    blocked: dict[str, str] = {}

    all_optimizers = {
        "representation_optimizer",
        "ir_structure_optimizer",
        "structure_optimizer",
        "reference_optimizer",
        "tool_optimizer",
        "diagnostic_optimizer",
        "context_optimizer",
    }
    for name in sorted(all_optimizers - set(allowed)):
        blocked[name] = "plan_excludes"

    return OptimizerSchedule(
        tier=tier,
        latency_budget_ms=budget,
        quality_floor=quality,
        allowed_optimizers=allowed,
        blocked_optimizers=blocked,
        transport_enabled=is_streaming or context_length > 2000,
        cache_enabled=provider not in ("ollama", "ollama-cloud"),
    )


def _task_quality_floor(task_class: str) -> float:
    floors: dict[str, float] = {
        TaskClass.REASONING.value: 0.92,
        TaskClass.DEBUGGING.value: 0.90,
        TaskClass.ANALYSIS.value: 0.88,
        TaskClass.STRUCTURED.value: 0.87,
        TaskClass.RETRIEVAL.value: 0.85,
        TaskClass.SUMMARIZATION.value: 0.85,
        TaskClass.SIMPLE.value: 0.80,
    }
    return floors.get(task_class, 0.85)


@dataclasses.dataclass(slots=True)
class RequestShim:
    """Minimal shim Request for UnifiedPlanner compatibility."""

    profile: SemanticProfile

    @property
    def token_estimate(self) -> int:
        return self.profile.context_length

    @property
    def is_tool_conversation(self) -> bool:
        return self.profile.has_tool_calls

    @property
    def stream(self) -> bool:
        return self.profile.is_streaming


def schedule_to_dict(schedule: OptimizerSchedule) -> dict[str, Any]:
    return {
        "tier": schedule.tier,
        "latency_budget_ms": schedule.latency_budget_ms,
        "quality_floor": schedule.quality_floor,
        "allowed_optimizers": schedule.allowed_optimizers,
        "blocked_optimizers": schedule.blocked_optimizers,
        "transport_enabled": schedule.transport_enabled,
        "cache_enabled": schedule.cache_enabled,
    }


__all__ = [
    "OptimizerSchedule",
    "decide_optimizer_schedule",
    "schedule_to_dict",
]
