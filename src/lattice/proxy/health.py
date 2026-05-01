"""Health check endpoints for the LATTICE proxy.

Provides Kubernetes-compatible probes:
- /healthz     → liveness
- /readyz      → readiness
- /startupz    → startup
- /metrics     → Prometheus exposition

All endpoints return JSON except /metrics which returns Prometheus text.
"""

from __future__ import annotations

from typing import Any

from lattice._version import __version__
from lattice.core.config import LatticeConfig
from lattice.core.metrics import get_metrics


class HealthManager:
    """Health and readiness check coordinator.

    Attributes:
        config: LatticeConfig for checking provider settings.
        pipeline_transform_count: Number of registered transforms.
        store_ready: Is the session store accessible.
    """

    def __init__(
        self,
        config: LatticeConfig,
        pipeline_transform_count: int = 0,
        store_ready: bool = True,
    ) -> None:
        self.config = config
        self.pipeline_transform_count = pipeline_transform_count
        self.store_ready = store_ready
        self._metrics = get_metrics()

    # ------------------------------------------------------------------
    # Liveness
    # ------------------------------------------------------------------

    def healthz(self) -> dict[str, str]:
        """Liveness probe.

        Returns 200 as long as the process is alive. Fast and cheap.
        """
        return {"status": "healthy", "version": __version__}

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    def readyz(self) -> dict[str, Any]:
        """Readiness probe.

        Checks:
        - Config loaded
        - Pipeline has transforms
        - Session store accessible
        - Provider base URL configured
        """
        checks: dict[str, Any] = {
            "config": True,
            "pipeline": self.pipeline_transform_count > 0,
            "session_store": self.store_ready,
            "provider_url": bool(self.config.provider_base_url),
        }
        all_pass = all(checks.values())
        return {
            "status": "ready" if all_pass else "not_ready",
            "checks": checks,
        }

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def startupz(self) -> dict[str, str]:
        """Startup probe.

        Returns 200 once the application has finished initializing
        (e.g., config loaded, session store connected).
        """
        return {"status": "started", "version": __version__}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self, transform_names: list[str], session_count: int) -> dict[str, Any]:
        """Human-readable proxy statistics."""
        return {
            "version": __version__,
            "transforms": transform_names,
            "sessions": session_count,
            "config": {
                "provider_base_url": self.config.provider_base_url or "default",
                "session_store": self.config.session_store,
                "log_level": self.config.log_level,
            },
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def metrics(self) -> str:
        """Prometheus metrics exposition."""
        return self._metrics.prometheus_output()
