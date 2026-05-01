"""Runtime bootstrap helpers for the proxy app factory."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lattice.core.agent_stats import AgentStatsCollector
from lattice.core.auto_continuation import AutoContinuation
from lattice.core.config import LatticeConfig
from lattice.core.cost_estimator import CostEstimator
from lattice.core.credentials import CredentialResolver
from lattice.core.metrics import get_metrics
from lattice.core.pipeline import CompressorPipeline
from lattice.core.pipeline_factory import build_default_pipeline
from lattice.core.semantic_cache import SemanticCache
from lattice.core.serialization import message_to_dict
from lattice.core.session import MemorySessionStore, SessionManager
from lattice.core.store import RedisSessionStore
from lattice.core.telemetry import DowngradeTelemetry
from lattice.core.transport import Message, Request, Response
from lattice.gateway.compat import HTTPCompatHandler, detect_provider, serialize_messages
from lattice.gateway.server import LLMTPGateway
from lattice.protocol.framing import BinaryFramer
from lattice.protocol.resume import StreamManager
from lattice.providers.transport import DirectHTTPProvider
from lattice.transforms.batching import BatchingEngine
from lattice.transforms.speculative import SpeculativeExecutor, SpeculativeTransform


@dataclass(slots=True)
class ProxyRuntime:
    """Container for configured proxy runtime dependencies."""

    store: Any
    session_manager: SessionManager
    pipeline: CompressorPipeline
    provider: DirectHTTPProvider
    gateway: LLMTPGateway
    compat: HTTPCompatHandler
    batching_engine: BatchingEngine
    speculative_executor: SpeculativeExecutor
    semantic_cache: SemanticCache
    cost_estimator: CostEstimator
    auto_continuation: AutoContinuation
    agent_stats: AgentStatsCollector
    metrics: Any
    downgrade_telemetry: DowngradeTelemetry


def configure_lifecycle(
    app: FastAPI,
    *,
    store: Any,
    provider: Any,
    logger: Any,
    version: str,
    semantic_cache: SemanticCache | None = None,
    maintenance: Any = None,
) -> None:
    """Attach startup/shutdown lifecycle hooks to app router."""

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        await store.start()
        if semantic_cache is not None and hasattr(semantic_cache._backend, "start"):
            await semantic_cache._backend.start()
        if maintenance is not None:
            await maintenance.start()
        logger.info("proxy_startup", version=version, store_type=type(store).__name__)
        yield
        if maintenance is not None:
            await maintenance.stop()
        if semantic_cache is not None and hasattr(semantic_cache._backend, "stop"):
            await semantic_cache._backend.stop()
        await store.stop()
        await provider.pool.close()
        logger.info("proxy_shutdown", version=version)

    app.router.lifespan_context = _lifespan


def configure_cors(app: FastAPI, config: LatticeConfig) -> None:
    """Configure proxy CORS policy from runtime config."""
    is_local = config.proxy_host in ("127.0.0.1", "localhost")
    cors_origins = ["*"] if is_local else ["http://localhost", "http://127.0.0.1"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def build_proxy_runtime(config: LatticeConfig) -> ProxyRuntime:
    """Build proxy runtime dependencies from configuration."""
    if config.session_store == "redis" and config.redis_url:
        store: Any = RedisSessionStore(
            url=config.redis_url,
            ttl_seconds=config.session_ttl_seconds,
        )
    else:
        store = MemorySessionStore(
            ttl_seconds=config.session_ttl_seconds,
            max_sessions=10000,
        )

    session_manager = SessionManager(store, ttl_seconds=config.session_ttl_seconds)

    pipeline = build_default_pipeline(
        config,
        include_execution_transforms=True,
        session_manager=session_manager,
    )

    credentials = CredentialResolver()
    if config.provider_api_key:
        credentials.register("openai", api_key=config.provider_api_key)

    downgrade_telemetry = DowngradeTelemetry()

    provider = DirectHTTPProvider(
        default_api_base=config.provider_base_url or None,
        default_api_key=config.provider_api_key or None,
        provider_base_urls=config.provider_base_urls,
        timeout=config.request_timeout_seconds,
        credentials=credentials,
        tacc_enabled=config.tacc_enabled,
        downgrade_telemetry=downgrade_telemetry,
    )
    provider.configure_resilience(stall_timeout=config.provider_stall_timeout_seconds)

    framer = BinaryFramer()
    stream_manager = StreamManager()
    gateway = LLMTPGateway(
        session_manager=session_manager,
        pipeline=pipeline,
        provider=provider,
        framer=framer,
        stream_manager=stream_manager,
        store=store,
    )
    compat = HTTPCompatHandler(gateway=gateway)

    async def _batched_provider_call(batched: Any) -> Any:
        from lattice.transforms.batching import BatchedResponse

        async def _single_call(
            idx: int, _cid: str, messages: list[Message]
        ) -> tuple[dict[str, Any], dict[str, int]]:
            serialized = [message_to_dict(m) for m in messages]
            provider_name = detect_provider(batched.key.model)
            resp = await provider.completion(
                model=batched.key.model,
                messages=serialized,
                temperature=batched.key.temperature,
                max_tokens=batched.key.max_tokens,
                top_p=batched.key.top_p,
                tools=batched.tools,
                tool_choice=batched.metadata.get("tool_choice"),
                stream=False,
                stop=batched.metadata.get("stop"),
                provider_name=provider_name,
                api_key=batched.metadata.get("_lattice_client_api_key"),
                metadata=batched.metadata.get("request_metadata", {}),
                extra_headers=batched.metadata.get("extra_headers"),
                extra_body=batched.metadata.get("extra_body"),
            )
            choice = {
                "index": idx,
                "message": {"role": resp.role, "content": resp.content},
                "finish_reason": resp.finish_reason or "stop",
            }
            return choice, resp.usage

        cid_groups: dict[str, list[Message]] = {}
        for cid, msg in batched.messages:
            cid_groups.setdefault(cid, []).append(msg)
        tasks = [
            asyncio.create_task(_single_call(i, cid, msgs))
            for i, (cid, msgs) in enumerate(cid_groups.items())
        ]
        results = await asyncio.gather(*tasks)
        results.sort(key=lambda r: r[0]["index"])
        choices = [r[0] for r in results]

        total_usage: dict[str, int] = {}
        for _choice, usage in results:
            for k, v in usage.items():
                total_usage[k] = total_usage.get(k, 0) + v

        return BatchedResponse(
            choices=choices,
            usage=total_usage,
            model=batched.key.model,
        )

    batching_engine = BatchingEngine(
        max_batch_size=8,
        max_wait_ms=10.0,
        provider_caller=_batched_provider_call,
        tacc=provider.tacc,
    )

    async def _speculative_provider_call(req: Request) -> Response:
        messages = serialize_messages(req)
        provider_name = detect_provider(req.model)
        return await provider.completion(
            model=req.model,
            messages=messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            top_p=req.top_p,
            tools=req.tools,
            tool_choice=req.tool_choice,
            stream=False,
            stop=req.stop,
            provider_name=provider_name,
            api_key=req.metadata.get("_lattice_client_api_key"),
            metadata=req.metadata,
            extra_headers=req.extra_headers,
            extra_body=req.extra_body,
        )

    speculative_executor = SpeculativeExecutor(
        max_speculative_tokens=256,
        confidence_threshold=0.7,
        provider_caller=_speculative_provider_call,
    )
    pipeline.unregister("speculative")
    pipeline.register(SpeculativeTransform(executor=speculative_executor))

    cache_backend = None
    if config.semantic_cache_backend == "redis":
        from lattice.core.semantic_cache import RedisCacheBackend

        cache_url = (
            config.semantic_cache_backend_url or config.redis_url or "redis://localhost:6379/0"
        )
        cache_backend = RedisCacheBackend(url=cache_url)

    semantic_cache = SemanticCache(
        backend=cache_backend,
        ttl_seconds=config.semantic_cache_ttl_seconds,
        max_entries=config.semantic_cache_max_entries,
        max_entry_size_kb=config.semantic_cache_max_entry_size_kb,
        enabled=config.semantic_cache_enabled,
    )

    cost_estimator = CostEstimator()
    auto_continuation = AutoContinuation(max_turns=3)
    agent_stats = AgentStatsCollector(metrics=get_metrics(), cost_estimator=cost_estimator)

    metrics = get_metrics()
    return ProxyRuntime(
        store=store,
        session_manager=session_manager,
        pipeline=pipeline,
        provider=provider,
        gateway=gateway,
        compat=compat,
        batching_engine=batching_engine,
        speculative_executor=speculative_executor,
        semantic_cache=semantic_cache,
        cost_estimator=cost_estimator,
        auto_continuation=auto_continuation,
        agent_stats=agent_stats,
        metrics=metrics,
        downgrade_telemetry=downgrade_telemetry,
    )
