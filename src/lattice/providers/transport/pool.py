"""Per-provider httpx connection pooling."""

from __future__ import annotations

from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


class ConnectionPoolManager:
    """Manages persistent ``httpx.AsyncClient`` instances per provider."""

    def __init__(self, http2: bool = True, downgrade_telemetry: Any = None) -> None:
        self._http2 = http2
        self._clients: dict[tuple[str, str], httpx.AsyncClient] = {}
        self._http2_fallback_reason: dict[tuple[str, str], str] = {}
        self._downgrade_telemetry = downgrade_telemetry
        self._log = logger.bind(module="connection_pool")

    def get_client(self, provider: str, base_url: str) -> httpx.AsyncClient:
        key = (provider, base_url)
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
                    from lattice.core.telemetry import DowngradeCategory

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

    def get_http_version(self, provider: str, base_url: str) -> str:
        key = (provider, base_url)
        if key in self._http2_fallback_reason:
            return "http/1.1"
        return "http/2" if self._http2 else "http/1.1"

    def get_fallback_reason(self, provider: str, base_url: str) -> str | None:
        return self._http2_fallback_reason.get((provider, base_url))

    async def close(self) -> None:
        for key, client in list(self._clients.items()):
            await client.aclose()
            self._log.info("pool_closed", provider=key[0])
        self._clients.clear()

    async def recycle_client(self, provider: str, base_url: str) -> None:
        key = (provider, base_url)
        client = self._clients.pop(key, None)
        if client is not None:
            await client.aclose()
            self._log.info("pool_recycled", provider=provider, base_url=base_url)

    @property
    def pool_count(self) -> int:
        return len(self._clients)
