"""LATTICE proxy server — Production-grade Transport Layer.

This is the HTTP proxy that sits between AI coding agents and LLM providers.
It receives OpenAI-compatible chat completion requests, runs LATTICE's
compression pipeline, then DIRECTLY dispatches the optimized request to the
correct LLM provider via our own HTTP/2 transport layer.

Architecture
------------
┌──────────────┐
│    Agent     │ ← OpenCode, Claude, Cursor, Codex...
│(OpenAI API)  │
└──────┬───────┘
       │ POST /v1/chat/completions
       │
┌──────▼───────┐
│    Proxy     │ ← FastAPI (this file)
│              │
│  1. Deserialize request (OpenAI JSON → internal Request)
│  2. Session management (create / lookup / persist)
│  3. Run compression pipeline
│     • reference_sub      (deduplicate repeated content)
│     • tool_filter        (strip unused tool output)
│     • content_profiler   (profile + prefix canonicalization)
│     • output_cleanup     (trim whitespace, normalize JSON)
│     • semantic_dict      (HPACK-style shared dictionary)
│     • delta_encoder      (session-based delta reconstruction)
│     • cache_planner      (provider-aware prefix packing)
│  4. Route via ProviderRegistry → adapter → DirectHTTPProvider
│  5. HTTP/2 connection pool to upstream provider
│  6. Deserialize provider-native response → internal Response
│  7. Run decompression pipeline (expand references back)
│  8. Persist session with updated messages
│  9. Return OpenAI-shaped JSON response + session headers
└──────────────┘

Design Decisions
----------------
• **NO LiteLLM in hot path.** Own the transport, own the adapters, own the
  connection pool. This is what makes LATTICE a transport layer.

• **Provider adapters are explicit.** Each provider gets its own adapter
  (OpenAI, Anthropic, Ollama, etc.) that handles request serialization,
  response deserialization, and streaming chunk normalization.

• **HTTP/2 connection pooling** is real and active. Every provider base URL
  gets a persistent ``httpx.AsyncClient(http2=True)``.

• **Session persistence** is now production-grade:
  - Sessions are created on turn 1 and persisted after every turn.
  - Manifest-based canonicalization for provider cache alignment.
  - Optimistic concurrency (CAS versioning) prevents lost updates.
  - Redis/KeyDB backend for multi-process deployments.

• **Semantic dictionary** is integrated into the hot path via
  ReferenceSubstitution, which learns from repeated content and encodes
  it as short integer references.

• The proxy is intentionally focused (~400 lines). Heavy logic lives in:
  • ``lattice.core.pipeline``        — compression / decompression
  • ``lattice.core.transport``       — Request / Response data model
  • ``lattice.core.session``         — session management
  • ``lattice.protocol.manifest``    — canonical segments
  • ``lattice.protocol.cache_planner`` — provider cache optimization
  • ``lattice.providers.transport``  — DirectHTTPProvider + connection pools
  • ``lattice.providers.*``          — per-provider adapters
  • ``lattice.integrations.agents``  — agent config injection

Streaming
---------
Streaming is supported end-to-end:
1. Client sends ``stream: true``.
2. Pipeline compresses the request.
3. DirectHTTPProvider opens an SSE stream to the provider.
4. Provider-specific adapter normalizes each chunk to OpenAI delta format.
5. Server sends SSE to client.
6. Session is updated when stream completes.

Error Handling
--------------
All HTTP/provider exceptions are mapped to HTTP status codes:

    httpx.TimeoutException      → 504
    ProviderError               → uses status_code field
    any other exception         → 502

Environment Variables
---------------------
``LATTICE_PROVIDER_BASE_URL`` — default upstream base URL
``LATTICE_PROVIDER_API_KEY``  — default upstream API key
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import FastAPI

from lattice._version import __version__
from lattice.core.config import LatticeConfig
from lattice.core.errors import ProviderError, ProviderTimeoutError
from lattice.gateway import compat as _gateway_compat
from lattice.gateway.compat import OperationalRouteDeps, register_operational_routes
from lattice.gateway.compat import anthropic_passthrough as compat_anthropic_passthrough
from lattice.gateway.compat import build_routing_headers as _build_routing_headers
from lattice.gateway.compat import (
    chat_completions_websocket_passthrough as compat_ws_chat,
)
from lattice.gateway.compat import deserialize_openai_request as _deserialize_openai_request
from lattice.gateway.compat import detect_new_messages as _detect_new_messages
from lattice.gateway.compat import responses_passthrough as compat_responses_passthrough
from lattice.gateway.compat import (
    responses_websocket_passthrough as compat_responses_websocket_passthrough,
)
from lattice.gateway.compat import serialize_messages as _serialize_messages
from lattice.gateway.compat import serialize_openai_response as _serialize_openai_response
from lattice.protocol.cache_planner import get_cache_planner
from lattice.proxy.bootstrap import build_proxy_runtime, configure_cors, configure_lifecycle
from lattice.proxy.middleware import install_middleware
from lattice.proxy.routes import (
    ProviderCompatRouteDeps,
    register_health_routes,
    register_native_lattice_routes,
    register_provider_compat_routes,
)
from lattice.transport.types import Message

logger = structlog.get_logger()

_deserialize_anthropic_request = _gateway_compat.deserialize_anthropic_request
_serialize_anthropic_response = _gateway_compat.serialize_anthropic_response
_is_local_origin = _gateway_compat.is_local_origin

# =============================================================================
# Constants
# =============================================================================

_SSE_KEEP_ALIVE = "\n"
_SSE_DONE = "[DONE]\n\n"

# =============================================================================
# App factory
# =============================================================================


def create_app(config: LatticeConfig | None = None) -> FastAPI:
    """Create a LATTICE proxy application.

    All state is injected via the factory — no global variables.
    """
    config = config or LatticeConfig.auto()

    app = FastAPI(
        title="LATTICE Proxy",
        description="LLM Transport & Efficiency Layer",
        version=__version__,
    )

    runtime = build_proxy_runtime(config)
    store = runtime.store
    session_manager = runtime.session_manager
    pipeline = runtime.pipeline
    provider = runtime.provider
    gateway = runtime.gateway
    compat = runtime.compat
    batching_engine = runtime.batching_engine
    speculative_executor = runtime.speculative_executor
    metrics = runtime.metrics

    # ------------------------------------------------------------------
    # Shared maintenance coordinator (created before lifecycle so the
    # background loop can be started/stopped alongside the app).
    # ------------------------------------------------------------------
    from lattice.core.maintenance import MaintenanceCoordinator, MaintenanceResult

    maintenance = MaintenanceCoordinator(interval_seconds=60.0)

    async def _maintain_stall_detector() -> MaintenanceResult:
        cleaned = provider.cleanup_stale_streams(max_age_ms=300000.0)
        return MaintenanceResult(
            stale_streams_removed=cleaned,
            did_work=cleaned > 0,
        )

    async def _maintain_semantic_cache() -> MaintenanceResult:
        if runtime.semantic_cache is not None and runtime.semantic_cache.enabled:
            removed = await runtime.semantic_cache.expire_stale()
            return MaintenanceResult(
                stale_cache_entries_removed=removed,
                did_work=removed > 0,
            )
        return MaintenanceResult()

    maintenance.register("stall_detector", _maintain_stall_detector)
    maintenance.register("semantic_cache", _maintain_semantic_cache)

    configure_lifecycle(
        app,
        store=store,
        provider=provider,
        logger=logger,
        version=__version__,
        semantic_cache=runtime.semantic_cache,
        maintenance=maintenance,
    )
    configure_cors(app, config)
    install_middleware(app)

    operational_deps = OperationalRouteDeps(
        config=config,
        metrics=metrics,
        provider=provider,
        pipeline=pipeline,
        store=store,
        batching_engine=batching_engine,
        speculative_executor=speculative_executor,
        agent_stats=runtime.agent_stats,
        semantic_cache=runtime.semantic_cache,
        cost_estimator=runtime.cost_estimator,
        logger=logger,
        version=__version__,
        downgrade_telemetry=runtime.downgrade_telemetry,
        maintenance=maintenance,
    )

    register_health_routes(app, runtime.health_manager, operational_deps)

    # ------------------------------------------------------------------
    # Request middleware + operational routes
    # ------------------------------------------------------------------
    register_operational_routes(app, operational_deps)

    @app.middleware("http")
    async def _transport_metadata_middleware(request: Any, call_next: Any) -> Any:
        from lattice.proxy.middleware import stash_lattice_response_headers

        stash_lattice_response_headers(
            request,
            {
                "x-lattice-framing": "json",
                "x-lattice-delta": "bypassed",
                "x-lattice-http-version": "2",
            },
        )
        return await call_next(request)

    register_native_lattice_routes(app, gateway)

    async def _anthropic_passthrough_with_logger(
        method: str,
        path: str,
        body: bytes,
        fastapi_request: Any,
        provider_obj: Any,
        **kwargs: Any,
    ) -> Any:
        return await compat_anthropic_passthrough(
            method,
            path,
            body,
            fastapi_request,
            provider_obj,
            logger=logger,
            **kwargs,
        )

    async def _responses_passthrough_with_logger(
        method: str,
        path: str,
        body: bytes,
        fastapi_request: Any,
        provider_obj: Any,
        **kwargs: Any,
    ) -> Any:
        return await compat_responses_passthrough(
            method,
            path,
            body,
            fastapi_request,
            provider_obj,
            logger=logger,
            **kwargs,
        )

    async def _responses_websocket_passthrough_with_logger(websocket: Any) -> None:
        await compat_responses_websocket_passthrough(websocket, logger=logger)

    async def _chat_completions_ws_passthrough(websocket: Any) -> None:
        await compat_ws_chat(websocket, logger=logger)

    register_provider_compat_routes(
        app,
        compat,
        ProviderCompatRouteDeps(
            config=config,
            pipeline=pipeline,
            provider=provider,
            session_manager=session_manager,
            batching_engine=batching_engine,
            speculative_executor=speculative_executor,
            semantic_cache=runtime.semantic_cache,
            cost_estimator=runtime.cost_estimator,
            auto_continuation=runtime.auto_continuation,
            agent_stats=runtime.agent_stats,
            metrics=metrics,
            logger=logger,
            deserialize_openai_request=_deserialize_openai_request,
            serialize_messages=_serialize_messages,
            serialize_openai_response=_serialize_openai_response,
            build_routing_headers=_build_routing_headers,
            detect_new_messages=_detect_new_messages,
            get_cache_planner=get_cache_planner,
            message_cls=Message,
            provider_timeout_error=ProviderTimeoutError,
            provider_error=ProviderError,
            sse_done=_SSE_DONE,
            anthropic_passthrough=_anthropic_passthrough_with_logger,
            responses_passthrough=_responses_passthrough_with_logger,
            responses_websocket_passthrough=_responses_websocket_passthrough_with_logger,
            chat_completions_websocket_passthrough=_chat_completions_ws_passthrough,
            maintenance=maintenance,
        ),
    )

    return app


# Compatibility helpers are imported directly from `gateway.compat`.


# =============================================================================
# CLI entry point
# =============================================================================


def main() -> None:
    """CLI entry point: ``lattice-proxy``."""
    import uvicorn

    config = LatticeConfig.auto()
    uvicorn.run(
        "lattice.proxy.server:create_app",
        host=config.proxy_host,
        port=config.proxy_port,
        workers=config.worker_count,
        reload=config.proxy_reload,
    )
