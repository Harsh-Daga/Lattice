"""Per-request transport telemetry surfaced on responses and /healthz."""

from __future__ import annotations

from dataclasses import dataclass

from lattice.transport.circuit_breaker import BreakerState


@dataclass(frozen=True, slots=True)
class TransportTelemetry:
    provider: str
    model: str
    rtt_ms: float
    attempt: int
    pool_utilization: float
    was_resumed: bool = False
    breaker_state: BreakerState = BreakerState.CLOSED

    def to_headers(self) -> dict[str, str]:
        return {
            "x-lattice-transport-rtt-ms": f"{self.rtt_ms:.2f}",
            "x-lattice-transport-attempt": str(self.attempt),
            "x-lattice-transport-pool-utilization": f"{self.pool_utilization:.3f}",
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "rtt_ms": self.rtt_ms,
            "attempt": self.attempt,
            "pool_utilization": self.pool_utilization,
            "was_resumed": self.was_resumed,
            "breaker_state": self.breaker_state.value,
        }
