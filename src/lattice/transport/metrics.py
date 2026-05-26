"""Transport-layer gauges and per-request telemetry."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

from lattice.transport.circuit_breaker import BreakerState


@dataclass(frozen=True, slots=True)
class TransportMetricsSnapshot:
    in_flight: int
    queue_depth: int
    pool_active_connections: dict[str, int]
    breaker_state: dict[tuple[str, str], BreakerState]
    retry_count_last_minute: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class RequestTelemetry:
    rtt: float
    attempt: int = 1


# Backward-compatible alias
_Telemetry = RequestTelemetry


class TransportMetrics:
    def __init__(self) -> None:
        self._rtt_samples: dict[str, list[float]] = defaultdict(list)
        self._retries: dict[str, int] = defaultdict(int)
        self._failures: dict[str, int] = defaultdict(int)
        self._last_attempt: dict[str, int] = {}

    def record_success(self, provider: str, telemetry: _Telemetry) -> None:
        self._rtt_samples[provider].append(telemetry.rtt)
        if len(self._rtt_samples[provider]) > 1000:
            self._rtt_samples[provider] = self._rtt_samples[provider][-500:]
        self._last_attempt[provider] = telemetry.attempt

    def record_failure(self, provider: str, exc: Exception) -> None:
        del exc
        self._failures[provider] += 1

    def record_retry(self, provider: str) -> None:
        self._retries[provider] += 1

    def last_attempt(self, provider: str) -> int:
        return self._last_attempt.get(provider, 1)

    def snapshot(
        self,
        *,
        in_flight: int,
        queue_depth: int,
        pool_counts: dict[str, int],
        breaker_states: dict[tuple[str, str], BreakerState],
    ) -> TransportMetricsSnapshot:
        return TransportMetricsSnapshot(
            in_flight=in_flight,
            queue_depth=queue_depth,
            pool_active_connections=dict(pool_counts),
            breaker_state=dict(breaker_states),
            retry_count_last_minute=dict(self._retries),
        )

    def snapshot_dict(
        self,
        *,
        in_flight: int,
        queue_depth: int,
        pool_counts: dict[str, int],
        breaker_states: dict[tuple[str, str], BreakerState],
    ) -> dict[str, object]:
        snap = self.snapshot(
            in_flight=in_flight,
            queue_depth=queue_depth,
            pool_counts=pool_counts,
            breaker_states=breaker_states,
        )
        return {
            "in_flight": snap.in_flight,
            "queue_depth": snap.queue_depth,
            "pool_active_connections": snap.pool_active_connections,
            "breaker_state": {f"{p}/{m}": s.value for (p, m), s in snap.breaker_state.items()},
            "retry_count_last_minute": snap.retry_count_last_minute,
        }


def monotonic_rtt(start: float) -> _Telemetry:
    return _Telemetry(rtt=time.perf_counter() - start)
