"""Per-request mutable context for transforms.

TransformContext is the single mutable scratchpad that carries state
across the transform pipeline. Each request gets a fresh context.
"""

from __future__ import annotations

import dataclasses
import time
import uuid
from typing import Any


@dataclasses.dataclass(slots=True)
class TransformContext:
    """Per-request mutable state that transforms can read and write.

    TransformContext carries state across the pipeline. For example:
    - ReferenceSubstitution stores its mapping table here.
    - PrefixOptimizer stores prefix hash here.
    - MetricsCollector stores per-transform timing here.

    Attributes:
        request_id: UUIDv4, immutable, used for logging and tracing.
        session_id: Optional session identifier for multi-turn tracking.
        started_at: Unix timestamp when request processing began.
        transforms_applied: Ordered list of transform names that ran.
        session_state: Arbitrary key-value store. Survives across requests
            in the same session. If session_id is None, it's a throwaway.
        metrics: Structured metrics collected during processing.
        provider: Target provider name (openai, anthropic, etc.).
        model: Target model identifier.
    """

    request_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str | None = None
    started_at: float = dataclasses.field(default_factory=time.time)
    transforms_applied: list[str] = dataclasses.field(default_factory=list)
    session_state: dict[str, Any] = dataclasses.field(default_factory=dict)
    metrics: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "tokens_in": 0,
            "tokens_out": 0,
            "latency_ms": 0.0,
            "transforms": {},
        }
    )
    provider: str = "openai"
    model: str = ""

    def mark_transform_applied(self, name: str) -> None:
        """Record that a transform was applied.

        Args:
            name: The transform's name attribute.
        """
        self.transforms_applied.append(name)

    def record_metric(self, transform: str, key: str, value: Any) -> None:
        """Record a metric for a specific transform.

        Metrics are stored in a nested dict under `self.metrics["transforms"]`.
        This structure is consumed by MetricsCollector to produce reports.

        Args:
            transform: Transform name (e.g., "reference_sub").
            key: Metric key (e.g., "tokens_in", "latency_ms").
            value: Metric value.
        """
        if "transforms" not in self.metrics:
            self.metrics["transforms"] = {}
        if transform not in self.metrics["transforms"]:
            self.metrics["transforms"][transform] = {}
        self.metrics["transforms"][transform][key] = value

    def get_transform_state(self, transform_name: str) -> dict[str, Any]:
        """Get mutable state dict for a transform.

        This is the preferred way for transforms to persist state across
        the compress/decompress cycle.

        Returns:
            A mutable dict (created if not present). The transform is
            responsible for the key names within this dict.
        """
        state: dict[str, Any] = self.session_state.setdefault(transform_name, {})
        return state

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time since processing started.

        Returns:
            Milliseconds since `started_at`.
        """
        return (time.time() - self.started_at) * 1000.0

    def get_summary(self) -> dict[str, Any]:
        """Return a human-readable summary of this context.

        Useful for logging and debugging.
        """
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "transforms_applied": self.transforms_applied,
            "metrics": self.metrics,
        }
