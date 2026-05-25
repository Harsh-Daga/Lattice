"""Mutable session-scoped plan for gateway and execution_builder (not the IR ExecutionPlan)."""

from __future__ import annotations

import dataclasses
import uuid
from typing import Any

from lattice.planner.plan_types import (
    CachePlanEntry,
    FallbackPlan,
    TransportPlanEntry,
)

__all__ = ["SessionExecutionPlan"]


@dataclasses.dataclass(slots=True)
class SessionExecutionPlan:
    """Rich plan persisted on request metadata for adapters and fallback execution."""

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
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionExecutionPlan:
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
