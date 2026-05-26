"""Transport-layer defaults (pool, retry, breaker, backpressure)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ProviderTransportConfig:
    http2: bool = True
    pool_size: int = 64
    connect_timeout: float = 10.0
    read_timeout: float = 300.0
    write_timeout: float = 30.0
    pool_acquire_timeout: float = 5.0


@dataclass(frozen=True, slots=True)
class TransportConfig:
    """Root transport configuration."""

    default_timeout: float = 120.0
    max_in_flight: int = 100
    failure_threshold: int = 10
    breaker_window_seconds: int = 60
    breaker_cooldown_seconds: int = 30
    providers: dict[str, ProviderTransportConfig] = field(default_factory=dict)

    def for_provider(self, provider: str) -> ProviderTransportConfig:
        return self.providers.get(provider, ProviderTransportConfig())
