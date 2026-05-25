"""Planner auxiliary types — cache/transport/fallback entries and tier budgets."""

from __future__ import annotations

import dataclasses
import enum
from typing import Any

from lattice.transport.types import Request


class ExecutionTier(enum.Enum):
    """Tiered latency budgets for optimization."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    REASONING = "reasoning"
    DEBUGGING = "debugging"


class RiskLevel(enum.Enum):
    """Semantic risk classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclasses.dataclass(slots=True)
class RepresentationCandidate:
    """A candidate representation produced by an optimizer."""

    optimizer_name: str
    request: Request
    token_cost: int
    latency_ms: float
    quality_estimate: float
    cache_gain: float
    transport_gain: float
    semantic_risk: float
    rollback_reason: str | None = None

    @property
    def score(self) -> float:
        return (
            self.quality_estimate
            + self.cache_gain
            + self.transport_gain
            - (self.token_cost / 1000.0)
            - (self.latency_ms / 100.0)
            - self.semantic_risk
        )


@dataclasses.dataclass(slots=True)
class OptimizerDecision:
    optimizer_name: str
    allowed: bool
    reason: str = ""
    budget_ms: float = 0.0
    candidates: list[RepresentationCandidate] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class CachePlanEntry:
    segment_index: int
    provider_mode: str
    expected_cached_tokens: int
    annotations: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "provider_mode": self.provider_mode,
            "expected_cached_tokens": self.expected_cached_tokens,
            "annotations": self.annotations,
        }


@dataclasses.dataclass(slots=True)
class TransportPlanEntry:
    use_delta: bool
    use_multiplex: bool
    use_framing: bool
    resume_enabled: bool
    compression_codec: str | None
    wire_format: str


@dataclasses.dataclass(slots=True)
class FallbackPlan:
    fallback_provider: str | None = None
    fallback_model: str | None = None
    disable_optimizers: bool = False
    retry_count: int = 0


TIER_BUDGETS_MS: dict[str, float] = {
    ExecutionTier.SIMPLE.value: 10.0,
    ExecutionTier.MEDIUM.value: 50.0,
    ExecutionTier.COMPLEX.value: 100.0,
    ExecutionTier.REASONING.value: 150.0,
    ExecutionTier.DEBUGGING.value: 150.0,
}


def tier_budget_ms(tier: str) -> float:
    return TIER_BUDGETS_MS.get(tier.lower(), 100.0)


__all__ = [
    "CachePlanEntry",
    "ExecutionTier",
    "FallbackPlan",
    "OptimizerDecision",
    "RepresentationCandidate",
    "RiskLevel",
    "TransportPlanEntry",
    "TIER_BUDGETS_MS",
    "tier_budget_ms",
]
