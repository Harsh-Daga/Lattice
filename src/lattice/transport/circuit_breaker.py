"""Per-(provider, model) circuit breaker."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field


class BreakerState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    def __init__(self, *, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        super().__init__(f"Circuit open for {provider}/{model}")


@dataclass
class CircuitBreaker:
    failure_threshold: int = 10
    window_seconds: int = 60
    cooldown_seconds: int = 30

    _state: BreakerState = field(default=BreakerState.CLOSED, init=False)
    _failures: list[float] = field(default_factory=list, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _half_open_in_flight: int = field(default=0, init=False)

    def allow(self) -> bool:
        now = time.monotonic()
        self._prune(now)
        match self._state:
            case BreakerState.CLOSED:
                return True
            case BreakerState.OPEN:
                if now - self._opened_at >= self.cooldown_seconds:
                    self._state = BreakerState.HALF_OPEN
                    self._half_open_in_flight = 0
                    return True
                return False
            case BreakerState.HALF_OPEN:
                return self._half_open_in_flight == 0

    def on_success(self) -> None:
        self._state = BreakerState.CLOSED
        self._failures.clear()
        self._half_open_in_flight = 0

    def on_failure(self, error_class: str = "unknown") -> None:
        del error_class  # reserved for metrics
        now = time.monotonic()
        self._failures.append(now)
        self._prune(now)
        if self._state == BreakerState.HALF_OPEN:
            self._state = BreakerState.OPEN
            self._opened_at = now
            return
        if len(self._failures) >= self.failure_threshold:
            self._state = BreakerState.OPEN
            self._opened_at = now

    def state(self) -> BreakerState:
        return self._state

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._failures = [t for t in self._failures if t >= cutoff]


class CircuitBreakerRegistry:
    def __init__(
        self,
        *,
        failure_threshold: int = 10,
        window_seconds: int = 60,
        cooldown_seconds: int = 30,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._window_seconds = window_seconds
        self._cooldown_seconds = cooldown_seconds
        self._breakers: dict[tuple[str, str], CircuitBreaker] = {}

    def for_(self, provider: str, model: str) -> CircuitBreaker:
        key = (provider, model)
        if key not in self._breakers:
            self._breakers[key] = CircuitBreaker(
                failure_threshold=self._failure_threshold,
                window_seconds=self._window_seconds,
                cooldown_seconds=self._cooldown_seconds,
            )
        return self._breakers[key]

    def all_states(self) -> dict[tuple[str, str], BreakerState]:
        return {key: br.state() for key, br in self._breakers.items()}
