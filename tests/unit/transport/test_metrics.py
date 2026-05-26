"""Transport metrics snapshot."""

from __future__ import annotations

import time

from lattice.transport.circuit_breaker import BreakerState
from lattice.transport.metrics import TransportMetrics, monotonic_rtt


def test_record_success_and_snapshot() -> None:
    m = TransportMetrics()
    start = time.perf_counter()
    m.record_success("openai", monotonic_rtt(start))
    snap = m.snapshot(
        in_flight=2,
        queue_depth=2,
        pool_counts={"openai": 1},
        breaker_states={("openai", "gpt-4"): BreakerState.CLOSED},
    )
    assert snap.in_flight == 2
    assert snap.pool_active_connections["openai"] == 1
