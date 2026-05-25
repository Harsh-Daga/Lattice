from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import status
from fastapi.responses import JSONResponse

from lattice.gateway.compat.translation import is_local_origin
from lattice.pipeline.factory import pipeline_summary
from lattice.protocol.manifest import manifest_summary
from lattice.providers.capabilities import Capability, get_capability_registry
from lattice.proxy.middleware import stash_lattice_response_headers

Handler = Callable[..., Awaitable[Any]]


class OperationalRouteDeps:
    """Dependencies for middleware and operational routes."""

    __slots__ = (
        "config",
        "metrics",
        "provider",
        "pipeline",
        "store",
        "batching_engine",
        "speculative_executor",
        "agent_stats",
        "semantic_cache",
        "cost_estimator",
        "logger",
        "version",
        "downgrade_telemetry",
        "maintenance",
    )

    def __init__(
        self,
        config: Any,
        metrics: Any,
        provider: Any,
        pipeline: Any,
        store: Any,
        batching_engine: Any,
        speculative_executor: Any,
        agent_stats: Any,
        semantic_cache: Any,
        cost_estimator: Any,
        logger: Any,
        version: str,
        downgrade_telemetry: Any = None,
        maintenance: Any = None,
    ) -> None:
        self.config = config
        self.metrics = metrics
        self.provider = provider
        self.pipeline = pipeline
        self.store = store
        self.batching_engine = batching_engine
        self.speculative_executor = speculative_executor
        self.agent_stats = agent_stats
        self.semantic_cache = semantic_cache
        self.cost_estimator = cost_estimator
        self.logger = logger
        self.version = version
        self.downgrade_telemetry = downgrade_telemetry
        self.maintenance = maintenance


async def build_proxy_stats_payload(deps: OperationalRouteDeps) -> dict[str, Any]:
    """Async wrapper for stats payload (store/cache/batching need await)."""
    from lattice.transport.delta_wire import DeltaWireDecoder

    capability_registry = get_capability_registry()
    cache_stats: dict[str, Any] = {}
    if deps.semantic_cache is not None:
        cache_stats = await deps.semantic_cache.stats

    result: dict[str, Any] = {
        "version": deps.version,
        "transforms": deps.pipeline.registry.get_transform_names(),
        "pipeline": pipeline_summary(deps.pipeline),
        "sessions": deps.store.session_count if hasattr(deps.store, "session_count") else 0,
        "provider": "direct_http",
        "adapters": deps.provider.registry.list_adapters(),
        "capabilities": {
            provider: {
                "cache_mode": capability_registry.cache_mode(provider).value,
                "supports_prompt_caching": capability_registry.supports(
                    provider, Capability.PROMPT_CACHING
                ),
                "default_base_url": capability_registry.get(provider).default_base_url  # type: ignore[union-attr]
                if capability_registry.get(provider)
                else "",
            }
            for provider in capability_registry.list_providers()
        },
        "pools": deps.provider.pool.pool_count,
        "batching": await deps.batching_engine.stats(),
        "speculation": deps.speculative_executor.stats,
        "tacc": deps.provider.tacc.all_stats() if hasattr(deps.provider, "tacc") else {},
    }
    manifest_stats: dict[str, Any] = {
        "sessions_with_manifest": 0,
        "anchor_version_max": 0,
        "token_estimate_total": 0,
        "segment_counts": {},
    }
    if hasattr(deps.store, "keys"):
        try:
            session_ids = await deps.store.keys()
            for session_id in session_ids:
                session = await deps.store.get(session_id)
                if session is None or session.manifest is None:
                    continue
                summary = manifest_summary(session.manifest)
                manifest_stats["sessions_with_manifest"] += 1
                manifest_stats["anchor_version_max"] = max(
                    int(manifest_stats["anchor_version_max"]),
                    int(summary["anchor_version"]),
                )
                manifest_stats["token_estimate_total"] += int(summary["token_estimate"])
                segment_counts = manifest_stats["segment_counts"]
                if isinstance(segment_counts, dict):
                    for seg_type, count in summary["segment_counts"].items():
                        segment_counts[seg_type] = segment_counts.get(seg_type, 0) + int(count)
        except (TypeError, AttributeError, KeyError) as exc:
            deps.logger.warning("maintenance_manifest_summary_failed", error=str(exc))
            manifest_stats = {
                "sessions_with_manifest": 0,
                "anchor_version_max": 0,
                "token_estimate_total": 0,
                "segment_counts": {},
            }
    result["manifest"] = manifest_stats
    if deps.agent_stats:
        result["agents"] = deps.agent_stats.global_summary()

    delta_fallback_stats = DeltaWireDecoder.get_fallback_stats()
    result["fallbacks"] = {
        "http2_to_http11_count": len(
            {
                k
                for k, v in (deps.provider.pool._http2_fallback_reason.items())
                if v == "h2_unavailable"
            }
        ),
        "delta_to_full_prompt_count": delta_fallback_stats.get("fallback_count", 0),
        "native_framing_to_json_count": 0,
        "stream_resume_fallback_reason_count": (
            deps.downgrade_telemetry._counts.get("stream_resume_to_full", 0)
            if deps.downgrade_telemetry is not None
            else 0
        ),
        "semantic_cache_approximate_hits": cache_stats.get("semantic_hits", 0),
        "semantic_cache_misses": cache_stats.get("misses", 0)
        + cache_stats.get("semantic_misses", 0),
    }
    if deps.downgrade_telemetry is not None:
        result["downgrades"] = deps.downgrade_telemetry.snapshot()
        result["transport_outcome_rollup"] = {
            k: v
            for k, v in deps.downgrade_telemetry._counts.items()
            if k
            in (
                "binary_to_json",
                "delta_to_full_prompt",
                "http2_to_http11",
                "stream_resume_to_full",
                "batching_bypassed",
                "speculation_bypassed",
            )
        }
    result["transport"] = {
        "pools": {
            f"{provider}:{base_url}": {
                "http_version": deps.provider.pool.get_http_version(provider, base_url),
                "fallback_reason": deps.provider.pool.get_fallback_reason(provider, base_url),
            }
            for (provider, base_url) in deps.provider.pool._clients
        }
    }
    if hasattr(deps.provider, "stall_detector"):
        result["ignored_chunks"] = deps.provider.stall_detector.get_ignored_chunk_stats()
    if deps.maintenance is not None:
        result["maintenance"] = deps.maintenance.stats()
    return result


def register_operational_routes(app: Any, deps: OperationalRouteDeps) -> None:
    """Register request middleware and operational health/stats routes."""

    @app.middleware("http")
    async def _request_middleware(request: Any, call_next: Any) -> Any:
        request_id = request.headers.get("x-request-id", str(time.time()))
        import structlog

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        host = request.headers.get("host", "")
        if deps.config.proxy_host == "127.0.0.1" and not is_local_origin(request):
            deps.logger.warning("reject_non_local_request", host=host, path=request.url.path)
            return JSONResponse(
                {
                    "error": "forbidden",
                    "message": "Local daemon mode rejects non-local requests",
                },
                status_code=status.HTTP_403_FORBIDDEN,
            )

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        response.headers["x-request-id"] = request_id
        stash_lattice_response_headers(request, {"x-lattice-version": deps.version})
        deps.metrics.increment("lattice_requests_total")
        deps.metrics.record_latency("lattice_request_latency_ms", elapsed_ms)
        if hasattr(deps.provider, "tacc"):
            for provider_name, state in deps.provider.tacc.all_stats().items():
                deps.metrics.tacc_metrics(provider_name, state)
        deps.logger.info(
            "proxy_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            elapsed_ms=round(elapsed_ms, 3),
        )
        return response

    @app.get("/providers/capabilities")
    async def _provider_capabilities() -> dict[str, Any]:
        registry = get_capability_registry()
        return {
            "providers": registry.to_dict(),
            "cache_modes": {
                provider: registry.cache_mode(provider).value
                for provider in registry.list_providers()
            },
        }

    @app.get("/cache/stats")
    async def _cache_stats() -> dict[str, Any]:
        if deps.semantic_cache is None:
            return {"enabled": False, "reason": "semantic_cache_not_configured"}
        cache_stats = await deps.semantic_cache.stats
        return {
            "enabled": deps.semantic_cache.enabled,
            "entries": cache_stats.get("entries", 0),
            "max_entries": cache_stats.get("max_entries", 0),
            "ttl_seconds": cache_stats.get("ttl_seconds", 0),
            "hits": cache_stats.get("hits", 0),
            "misses": cache_stats.get("misses", 0),
            "hit_rate": cache_stats.get("hit_rate", 0.0),
            "evictions": cache_stats.get("evictions", 0),
            "rejects": cache_stats.get("rejects", 0),
        }

    @app.post("/cache/clear")
    async def _cache_clear() -> dict[str, Any]:
        if deps.semantic_cache is None:
            return {"cleared": 0, "enabled": False}
        count = await deps.semantic_cache.clear()
        deps.logger.info("semantic_cache_cleared", entries_removed=count)
        return {"cleared": count, "enabled": deps.semantic_cache.enabled}
