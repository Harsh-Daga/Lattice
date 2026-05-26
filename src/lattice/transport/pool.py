"""Per-provider httpx connection pooling."""

from __future__ import annotations

from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


class ConnectionPoolManager:
    """Manages one persistent ``httpx.AsyncClient`` per provider (HTTP/2 when available)."""

    def __init__(self, http2: bool = True, downgrade_telemetry: Any = None) -> None:
        self._http2 = http2
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._http2_fallback_reason: dict[str, str] = {}
        self._downgrade_telemetry = downgrade_telemetry
        self._log = logger.bind(module="connection_pool")

    def get_client(self, provider: str, base_url: str = "") -> httpx.AsyncClient:
        # One pool per provider; full URL is built per request on the adapter endpoint.
        key = provider
        if key not in self._clients:
            limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
            timeout = httpx.Timeout(120.0, connect=10.0)
            try:
                client = httpx.AsyncClient(
                    timeout=timeout,
                    limits=limits,
                    http1=True,
                    http2=self._http2,
                )
            except ImportError:
                client = httpx.AsyncClient(
                    timeout=timeout,
                    limits=limits,
                    http1=True,
                    http2=False,
                )
                self._http2 = False
                self._http2_fallback_reason[key] = "h2_unavailable"
                self._log.warning("http2_unavailable", provider=provider, fallback="http1.1")
                if self._downgrade_telemetry is not None:
                    from lattice.telemetry.downgrade import DowngradeCategory

                    self._downgrade_telemetry.record(
                        DowngradeCategory.HTTP2_TO_HTTP11,
                        reason="h2_unavailable",
                    )
            self._clients[key] = client
            self._log.info(
                "pool_created",
                provider=provider,
                base_url=base_url,
                http2=self._http2,
            )
        return self._clients[key]

    def get_http_version(self, provider: str, base_url: str = "") -> str:
        del base_url
        if provider in self._http2_fallback_reason:
            return "http/1.1"
        return "http/2" if self._http2 else "http/1.1"

    def get_fallback_reason(self, provider: str, base_url: str = "") -> str | None:
        del base_url
        return self._http2_fallback_reason.get(provider)

    async def close(self) -> None:
        for provider, client in list(self._clients.items()):
            await client.aclose()
            self._log.info("pool_closed", provider=provider)
        self._clients.clear()

    async def recycle_client(self, provider: str, base_url: str = "") -> None:
        del base_url
        client = self._clients.pop(provider, None)
        if client is not None:
            await client.aclose()
            self._log.info("pool_recycled", provider=provider)

    def clients_by_provider(self) -> dict[str, httpx.AsyncClient]:
        return dict(self._clients)

    @property
    def pool_count(self) -> int:
        return len(self._clients)
