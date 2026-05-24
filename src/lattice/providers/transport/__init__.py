"""HTTP transport layer for provider dispatch."""

import httpx  # noqa: F401 — tests patch ``lattice.providers.transport.httpx``

from lattice.providers.transport.completion import DirectHTTPProvider
from lattice.providers.transport.pool import ConnectionPoolManager
from lattice.providers.transport.rate_limits import RateLimitState, RateLimitTracker
from lattice.providers.transport.registry import (
    _PROVIDER_ALIASES,
    ProviderRegistry,
    _resolve_provider_name,
)
from lattice.providers.transport.stall_detector import StreamStallDetector

__all__ = [
    "DirectHTTPProvider",
    "ProviderRegistry",
    "ConnectionPoolManager",
    "RateLimitTracker",
    "RateLimitState",
    "StreamStallDetector",
    "_resolve_provider_name",
    "_PROVIDER_ALIASES",
]
