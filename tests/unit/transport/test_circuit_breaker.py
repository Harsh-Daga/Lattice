"""Circuit breaker state machine."""

from __future__ import annotations

from lattice.transport.circuit_breaker import BreakerState, CircuitBreaker


def test_opens_after_threshold() -> None:
    cb = CircuitBreaker(failure_threshold=3, window_seconds=60, cooldown_seconds=120)
    assert cb.allow()
    for _ in range(3):
        cb.on_failure()
    assert cb.state() == BreakerState.OPEN
    assert not cb.allow()


def test_half_open_success_closes() -> None:
    cb = CircuitBreaker(failure_threshold=1, window_seconds=60, cooldown_seconds=0)
    cb.on_failure()
    assert cb.state() == BreakerState.OPEN
    assert cb.allow()
    cb.on_success()
    assert cb.state() == BreakerState.CLOSED
