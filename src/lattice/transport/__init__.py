"""Protocol-level transport: wire types, serialization, congestion, delta wire.

HTTP provider dispatch (``TransportDispatcher``, pools, retry) lives in submodules;
import via ``lattice.transport.dispatcher`` or ``lattice.providers`` re-exports.
"""

from __future__ import annotations

from typing import Any

from lattice.transport.congestion import ProviderCongestionState, TACCController
from lattice.transport.delta_wire import (
    DeltaWireDecoder,
    DeltaWireEncoder,
    compute_wire_savings,
    delta_wire_bytes,
)
from lattice.transport.serialization import (
    message_from_dict,
    message_to_dict,
    request_from_dict,
    request_to_dict,
    response_to_dict,
)
from lattice.transport.simulation import (
    SimulationConfig,
    SimulationMetrics,
    run_static_concurrency_simulation,
    run_tacc_simulation,
)
from lattice.transport.types import (
    Message,
    Request,
    Response,
    Role,
    SyncTransform,
    Transform,
)

_LAZY_HTTP_EXPORTS = {
    "TransportDispatcher",
    "DirectHTTPProvider",
    "ConnectionPoolManager",
    "ProviderRegistry",
    "RateLimitTracker",
    "RateLimitState",
    "RATE_LIMIT_TTL_S",
    "StreamStallDetector",
    "_resolve_provider_name",
    "_PROVIDER_ALIASES",
    "should_retry",
    "RetryEngine",
    "RetryPolicy",
    "RetryRule",
    "Backoff",
    "policy_from_retry_config",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitOpenError",
    "BreakerState",
    "Backpressure",
    "QueueFullError",
    "StreamResumer",
    "TransportMetrics",
    "TransportMetricsSnapshot",
    "TransportConfig",
    "ProviderTransportConfig",
    "TimeoutResolver",
    "TimeoutPolicy",
    "httpx",
}


def __getattr__(name: str) -> Any:
    if name == "httpx":
        import httpx as _httpx

        return _httpx
    if name in _LAZY_HTTP_EXPORTS:
        from lattice.transport import (
            backpressure,
            circuit_breaker,
            config,
            dispatcher,
            metrics,
            pool,
            rate_limit,
            registry,
            retry,
            retry_policy,
            stall_detector,
            stream_resume,
            timeout,
        )

        _map = {
            "TransportDispatcher": dispatcher.TransportDispatcher,
            "DirectHTTPProvider": dispatcher.DirectHTTPProvider,
            "ConnectionPoolManager": pool.ConnectionPoolManager,
            "ProviderRegistry": registry.ProviderRegistry,
            "RateLimitTracker": rate_limit.RateLimitTracker,
            "RateLimitState": rate_limit.RateLimitState,
            "RATE_LIMIT_TTL_S": rate_limit.RATE_LIMIT_TTL_S,
            "StreamStallDetector": stall_detector.StreamStallDetector,
            "_resolve_provider_name": registry._resolve_provider_name,
            "_PROVIDER_ALIASES": registry._PROVIDER_ALIASES,
            "should_retry": registry.should_retry,
            "RetryEngine": retry.RetryEngine,
            "RetryPolicy": retry_policy.RetryPolicy,
            "RetryRule": retry_policy.RetryRule,
            "Backoff": retry_policy.Backoff,
            "policy_from_retry_config": retry_policy.policy_from_retry_config,
            "CircuitBreaker": circuit_breaker.CircuitBreaker,
            "CircuitBreakerRegistry": circuit_breaker.CircuitBreakerRegistry,
            "CircuitOpenError": circuit_breaker.CircuitOpenError,
            "BreakerState": circuit_breaker.BreakerState,
            "Backpressure": backpressure.Backpressure,
            "QueueFullError": backpressure.QueueFullError,
            "StreamResumer": stream_resume.StreamResumer,
            "TransportMetrics": metrics.TransportMetrics,
            "TransportMetricsSnapshot": metrics.TransportMetricsSnapshot,
            "TransportConfig": config.TransportConfig,
            "ProviderTransportConfig": config.ProviderTransportConfig,
            "TimeoutResolver": timeout.TimeoutResolver,
            "TimeoutPolicy": timeout.TimeoutPolicy,
        }
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Message",
    "Request",
    "Response",
    "Role",
    "Transform",
    "SyncTransform",
    "message_to_dict",
    "message_from_dict",
    "request_to_dict",
    "request_from_dict",
    "response_to_dict",
    "DeltaWireDecoder",
    "DeltaWireEncoder",
    "delta_wire_bytes",
    "compute_wire_savings",
    "ProviderCongestionState",
    "TACCController",
    "SimulationConfig",
    "SimulationMetrics",
    "run_static_concurrency_simulation",
    "run_tacc_simulation",
    *sorted(_LAZY_HTTP_EXPORTS),
]
