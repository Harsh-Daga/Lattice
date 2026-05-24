"""Health check endpoints for the LATTICE proxy.

``HealthManager`` is the single coordinator for ``/healthz``, ``/readyz``,
``/startupz``, ``/metrics``, and ``/stats``. Route handlers in
``proxy/routes.register_health_routes`` delegate to these methods only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import status

from lattice._version import __version__

if TYPE_CHECKING:
    from lattice.gateway.compat import OperationalRouteDeps


class HealthManager:
    """Health, readiness, metrics, and stats for the proxy."""

    def __init__(
        self,
        *,
        ops: OperationalRouteDeps | None = None,
        pipeline_transform_count: int = 0,
        store_ready: bool = True,
        provider_base_url: str | None = None,
    ) -> None:
        self._ops = ops
        self.pipeline_transform_count = pipeline_transform_count
        self.store_ready = store_ready
        self.provider_base_url = provider_base_url

    @classmethod
    def from_operational(cls, ops: OperationalRouteDeps) -> HealthManager:
        """Build a manager wired to the live proxy runtime."""
        transform_count = len(ops.pipeline.registry.get_transform_names())
        return cls(
            ops=ops,
            pipeline_transform_count=transform_count,
            store_ready=True,
            provider_base_url=ops.config.provider_base_url,
        )

    def healthz(self) -> dict[str, Any]:
        """Liveness probe — process is up."""
        if self._ops is not None:
            return {
                "status": "healthy",
                "version": self._ops.version,
                "provider": "direct_http",
                "adapters": ", ".join(self._ops.provider.registry.list_adapters()),
            }
        return {"status": "healthy", "version": __version__}

    def readyz(self) -> tuple[dict[str, Any], int]:
        """Readiness probe — returns ``(body, http_status_code)``."""
        if self._ops is not None:
            live, detail = self._ops.provider.health_check()
            body = {
                "status": "ready" if live else "not_ready",
                "checks": {
                    "config": True,
                    "pipeline": len(self._ops.pipeline.registry.get_transform_names()) > 0,
                    "provider": live,
                    "provider_detail": detail,
                    "http2_pools": self._ops.provider.pool.pool_count,
                    "sessions": self._ops.store.session_count
                    if hasattr(self._ops.store, "session_count")
                    else 0,
                },
            }
            code = status.HTTP_200_OK if live else status.HTTP_503_SERVICE_UNAVAILABLE
            return body, code

        checks: dict[str, Any] = {
            "config": True,
            "pipeline": self.pipeline_transform_count > 0,
            "session_store": self.store_ready,
            "provider_url": bool(self.provider_base_url),
        }
        all_pass = all(checks.values())
        body = {
            "status": "ready" if all_pass else "not_ready",
            "checks": checks,
        }
        code = status.HTTP_200_OK if all_pass else status.HTTP_503_SERVICE_UNAVAILABLE
        return body, code

    def startupz(self) -> dict[str, str]:
        """Startup probe."""
        return {"status": "started", "version": __version__}

    def metrics(self) -> str:
        """Prometheus metrics exposition."""
        if self._ops is not None:
            return str(self._ops.metrics.prometheus_output())
        from lattice.telemetry.metrics import get_metrics

        return get_metrics().prometheus_output()

    async def stats(self) -> dict[str, Any]:
        """Full proxy statistics snapshot."""
        if self._ops is not None:
            from lattice.gateway.compat import build_proxy_stats_payload

            return await build_proxy_stats_payload(self._ops)

        return {
            "version": __version__,
            "transforms": [],
            "sessions": 0,
            "config": {
                "provider_base_url": self.provider_base_url or "default",
                "session_store": "memory",
                "log_level": "INFO",
            },
        }

    def stats_minimal(self, transform_names: list[str], session_count: int) -> dict[str, Any]:
        """Compact stats for unit tests without a full ``OperationalRouteDeps``."""
        return {
            "version": __version__,
            "transforms": transform_names,
            "sessions": session_count,
            "config": {
                "provider_base_url": self.provider_base_url or "default",
                "session_store": "memory",
                "log_level": "INFO",
            },
        }
