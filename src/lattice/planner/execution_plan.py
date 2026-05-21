"""ExecutionPlan — single source of truth for the entire request lifecycle.

Every subsystem reads from the ExecutionPlan instead of making independent
decisions. This prevents drift between scheduler, pipeline, transport, and provider.

Phase 9 from the architecture refactor.
"""

from __future__ import annotations

import dataclasses
import enum
import uuid
from typing import Any

from lattice.core.transport import Request


class ExecutionTier(enum.Enum):
    """Tiered latency budgets for optimization."""

    SIMPLE = "simple"          # 10ms
    MEDIUM = "medium"          # 50ms
    COMPLEX = "complex"        # 100ms
    REASONING = "reasoning"    # 150ms
    DEBUGGING = "debugging"    # 150ms, no lossy transforms


class RiskLevel(enum.Enum):
    """Semantic risk classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclasses.dataclass(slots=True)
class OptimizerDecision:
    """Decision for a single optimizer."""

    optimizer_name: str
    allowed: bool
    reason: str = ""
    budget_ms: float = 0.0
    # Candidate representations this optimizer generated (for beam search)
    candidates: list[RepresentationCandidate] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class RepresentationCandidate:
    """A candidate representation produced by an optimizer."""

    optimizer_name: str
    request: Request
    token_cost: int
    latency_ms: float
    quality_estimate: float   # 0-1, higher is better
    cache_gain: float         # estimated cache hit improvement
    transport_gain: float     # estimated wire/transport savings
    semantic_risk: float      # 0-1, lower is better
    rollback_reason: str | None = None

    @property
    def score(self) -> float:
        """Global objective score: quality + cache + transport - cost - risk."""
        return (
            self.quality_estimate
            + self.cache_gain
            + self.transport_gain
            - (self.token_cost / 1000.0)
            - (self.latency_ms / 100.0)
            - self.semantic_risk
        )


@dataclasses.dataclass(slots=True)
class CachePlanEntry:
    """Per-segment cache plan."""

    segment_index: int
    provider_mode: str         # auto_prefix, explicit_breakpoint, etc.
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
    """Per-request transport optimization plan."""

    use_delta: bool
    use_multiplex: bool
    use_framing: bool
    resume_enabled: bool
    compression_codec: str | None  # dictionary, gzip, etc.
    wire_format: str               # binary, json, sse


@dataclasses.dataclass(slots=True)
class FallbackPlan:
    """What to do when the primary path fails."""

    fallback_provider: str | None = None
    fallback_model: str | None = None
    disable_optimizers: bool = False
    retry_count: int = 0


@dataclasses.dataclass(slots=True)
class ExecutionPlan:
    """Single object passed through the entire LATTICE system.

    Attributes:
        request_id: Unique identifier for this request.
        session_id: Optional session for multi-turn tracking.
        provider: Resolved provider name (openai, anthropic, ollama, etc.).
        model: Target model identifier.
        task_class: Classification from request_classifier.
        risk_level: Semantic risk from content_profiler.
        latency_budget_ms: Total budget for ALL optimization work.
        quality_floor: Minimum quality score (0-1) for any representation.
        allowed_optimizers: Optimizers permitted to run.
        blocked_optimizers: Optimizers blocked with reasons.
        representation_plan: Ordered list of optimizer names for representation.
        transport_plan: Wire/transport decisions.
        cache_plan: Provider-specific cache annotations.
        fallback_plan: What to do on failure.
        metrics: Mutable dict for telemetry (not frozen).
    """

    request_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str | None = None
    provider: str = ""
    model: str = ""

    task_class: str = "simple"
    risk_level: str = "low"
    latency_budget_ms: float = 100.0
    quality_floor: float = 0.85

    allowed_optimizers: list[str] = dataclasses.field(default_factory=list)
    blocked_optimizers: dict[str, str] = dataclasses.field(default_factory=dict)

    representation_plan: list[str] = dataclasses.field(default_factory=list)
    transport_plan: TransportPlanEntry = dataclasses.field(
        default_factory=lambda: TransportPlanEntry(
            use_delta=False,
            use_multiplex=False,
            use_framing=True,
            resume_enabled=False,
            compression_codec=None,
            wire_format="json",
        )
    )
    cache_plan: list[CachePlanEntry] = dataclasses.field(default_factory=list)
    fallback_plan: FallbackPlan = dataclasses.field(default_factory=FallbackPlan)

    # Mutable telemetry — not part of equality/hash
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionPlan:
        """Rebuild an ExecutionPlan from a JSON-serializable dict."""
        transport = data.get("transport_plan", {})
        fallback = data.get("fallback_plan", {})
        cache = data.get("cache_plan", [])
        return cls(
            request_id=data.get("request_id", str(uuid.uuid4())),
            session_id=data.get("session_id"),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            task_class=data.get("task_class", "simple"),
            risk_level=data.get("risk_level", "low"),
            latency_budget_ms=float(data.get("latency_budget_ms", 100.0)),
            quality_floor=float(data.get("quality_floor", 0.85)),
            allowed_optimizers=list(data.get("allowed_optimizers", [])),
            blocked_optimizers=dict(data.get("blocked_optimizers", {})),
            representation_plan=list(data.get("representation_plan", [])),
            transport_plan=TransportPlanEntry(
                use_delta=bool(transport.get("use_delta", False)),
                use_multiplex=bool(transport.get("use_multiplex", False)),
                use_framing=bool(transport.get("use_framing", True)),
                resume_enabled=bool(transport.get("resume_enabled", False)),
                compression_codec=transport.get("compression_codec"),
                wire_format=str(transport.get("wire_format", "json")),
            ),
            cache_plan=[
                CachePlanEntry(
                    segment_index=e["segment_index"],
                    provider_mode=e["provider_mode"],
                    expected_cached_tokens=e["expected_cached_tokens"],
                    annotations=e.get("annotations", {}),
                )
                for e in cache
            ],
            fallback_plan=FallbackPlan(
                fallback_provider=fallback.get("fallback_provider"),
                fallback_model=fallback.get("fallback_model"),
                disable_optimizers=bool(fallback.get("disable_optimizers", False)),
                retry_count=int(fallback.get("retry_count", 0)),
            ),
            metrics=dict(data.get("metrics", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "task_class": self.task_class,
            "risk_level": self.risk_level,
            "latency_budget_ms": self.latency_budget_ms,
            "quality_floor": self.quality_floor,
            "allowed_optimizers": self.allowed_optimizers,
            "blocked_optimizers": self.blocked_optimizers,
            "representation_plan": self.representation_plan,
            "transport_plan": {
                "use_delta": self.transport_plan.use_delta,
                "use_multiplex": self.transport_plan.use_multiplex,
                "use_framing": self.transport_plan.use_framing,
                "resume_enabled": self.transport_plan.resume_enabled,
                "compression_codec": self.transport_plan.compression_codec,
                "wire_format": self.transport_plan.wire_format,
            },
            "cache_plan": [
                {
                    "segment_index": e.segment_index,
                    "provider_mode": e.provider_mode,
                    "expected_cached_tokens": e.expected_cached_tokens,
                }
                for e in self.cache_plan
            ],
            "fallback_plan": {
                "fallback_provider": self.fallback_plan.fallback_provider,
                "fallback_model": self.fallback_plan.fallback_model,
                "disable_optimizers": self.fallback_plan.disable_optimizers,
                "retry_count": self.fallback_plan.retry_count,
            },
            "metrics": self.metrics,
        }


# Tiered budgets — Phase 6
TIER_BUDGETS_MS: dict[str, float] = {
    ExecutionTier.SIMPLE.value: 10.0,
    ExecutionTier.MEDIUM.value: 50.0,
    ExecutionTier.COMPLEX.value: 100.0,
    ExecutionTier.REASONING.value: 150.0,
    ExecutionTier.DEBUGGING.value: 150.0,
}


def tier_budget_ms(tier: str) -> float:
    """Return the latency budget for a tier."""
    return TIER_BUDGETS_MS.get(tier.lower(), 100.0)


__all__ = [
    "ExecutionPlan",
    "OptimizerDecision",
    "RepresentationCandidate",
    "CachePlanEntry",
    "TransportPlanEntry",
    "FallbackPlan",
    "ExecutionTier",
    "RiskLevel",
    "tier_budget_ms",
]
