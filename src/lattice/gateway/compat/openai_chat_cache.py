"""Semantic cache serving for chat completions."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse

from lattice.cache.semantic import compute_cache_key
from lattice.gateway.compat.headers import (
    _runtime_header_values,
    _usage_total_tokens,
)
from lattice.gateway.compat.openai_chat_deps import ChatCompatDeps
from lattice.proxy.middleware import attach_routing_headers
from lattice.telemetry.agent_stats import identify_agent
from lattice.telemetry.downgrade import TransportOutcome
from lattice.transport.types import Response


async def try_semantic_cache_hit(
    deps: ChatCompatDeps,
    *,
    fastapi_request: Any,
    request: Any,
    compressed_request: Any,
    ctx: Any,
    session: Any,
    provider_name: str,
    delta_mode: str,
    http_version: str,
    delta_savings_bytes: int,
    x_lattice_client_profile: str | None,
    x_lattice_disable_transforms: str | None,
) -> Any | None:
    """Return a Starlette response on cache hit, else None."""
    disable_cache = request.extra_headers.get("x-lattice-disable-cache", "")
    if not (deps.semantic_cache and deps.semantic_cache.enabled and not disable_cache):
        return None

    cache_key = compute_cache_key(compressed_request)
    cached = await deps.semantic_cache.get(cache_key, compressed_request)
    if cached is None:
        return None

    deps.metrics.increment("lattice_semantic_cache_hit")
    deps.logger.info(
        "semantic_cache_hit",
        key=cache_key[:16],
        model=cached.model,
        content_len=len(cached.content),
    )
    cached_tokens = _usage_total_tokens(cached.usage)
    if cached_tokens > 0:
        session.record_cache_hit(cached_tokens)
    cache_savings_usd = 0.0
    if deps.cost_estimator:
        cache_cost = deps.cost_estimator.compute_actual(
            provider=provider_name,
            model=cached.model,
            usage=cached.usage,
        )
        cache_savings_usd = cache_cost.total_cost_usd

    if compressed_request.stream:
        from lattice.cache.semantic import generate_sse_chunks

        sse_chunks = generate_sse_chunks(
            cached,
            request_id=ctx.request_id,
            session_id=session.session_id,
        )

        async def _cached_stream() -> AsyncIterator[str]:
            for chunk in sse_chunks:
                yield chunk
            yield f"data: {deps.sse_done.strip()}\n\n"
            if cached.content or cached.tool_calls:
                msg = deps.message_cls(
                    role="assistant",
                    content=cached.content,
                    tool_calls=cached.tool_calls,
                )
                session.messages.append(msg)
                await deps.session_manager.update_session(session.session_id, session.messages)
            if deps.agent_stats:
                await deps.agent_stats.record_request(
                    agent=identify_agent(
                        request.extra_headers.get("user-agent"),
                        x_lattice_client_profile,
                    ),
                    provider=provider_name,
                    model=cached.model,
                    prompt_tokens=0,
                    completion_tokens=0,
                    compressed_tokens=compressed_request.token_estimate or 0,
                    original_tokens=request.token_estimate or 0,
                    cached_tokens=cached_tokens,
                    cache_hit=cached_tokens > 0,
                    cost_usd=0.0,
                )

        transport_outcome = TransportOutcome(
            semantic_cache_status="hit",
            delta_mode=delta_mode,
            http_version=http_version,
        )
        attach_routing_headers(
            fastapi_request,
            ctx,
            deps.build_routing_headers(
                cached.model,
                session_id=session.session_id,
                cache_hit=True,
                cached_tokens=cached_tokens,
                delta_savings_bytes=delta_savings_bytes,
                cache_savings_usd=cache_savings_usd,
                transport_outcome=transport_outcome,
                **_runtime_header_values(compressed_request),
            ),
        )
        return StreamingResponse(_cached_stream(), media_type="text/event-stream")

    cached_response = Response(
        content=cached.content,
        tool_calls=cached.tool_calls,
        usage=cached.usage,
        model=cached.model,
        finish_reason=cached.finish_reason,
    )
    if not x_lattice_disable_transforms:
        cached_response = deps.pipeline.reverse(cached_response, ctx)
    response_body = deps.serialize_openai_response(cached_response, compressed_request)
    response = JSONResponse(content=response_body)
    if deps.agent_stats:
        await deps.agent_stats.record_request(
            agent=identify_agent(
                request.extra_headers.get("user-agent"),
                x_lattice_client_profile,
            ),
            provider=provider_name,
            model=cached.model,
            prompt_tokens=0,
            completion_tokens=0,
            compressed_tokens=compressed_request.token_estimate or 0,
            original_tokens=request.token_estimate or 0,
            cached_tokens=cached_tokens,
            cache_hit=cached_tokens > 0,
            cost_usd=0.0,
        )
    transport_outcome = TransportOutcome(
        semantic_cache_status="hit",
        delta_mode=delta_mode,
        http_version=http_version,
    )
    attach_routing_headers(
        fastapi_request,
        ctx,
        deps.build_routing_headers(
            cached.model,
            session_id=session.session_id,
            compressed_tokens=compressed_request.token_estimate,
            original_tokens=request.token_estimate,
            cache_hit=True,
            cached_tokens=cached_tokens,
            delta_savings_bytes=delta_savings_bytes,
            cache_savings_usd=cache_savings_usd,
            transport_outcome=transport_outcome,
            **_runtime_header_values(compressed_request),
        ),
    )
    if cached.content or cached.tool_calls:
        msg = deps.message_cls(
            role="assistant",
            content=cached.content,
            tool_calls=cached.tool_calls,
        )
        session.messages.append(msg)
        await deps.session_manager.update_session(session.session_id, session.messages)
    return response
