"""Streaming chat completion branch."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse

from lattice.cache.semantic import assemble_cached_response
from lattice.gateway.compat.headers import (
    _extract_cached_tokens,
    _runtime_header_values,
)
from lattice.gateway.compat.openai_chat_deps import ChatCompatDeps
from lattice.planner.runtime_state import sum_expected_cached_tokens
from lattice.proxy.middleware import attach_routing_headers
from lattice.telemetry.agent_stats import identify_agent
from lattice.telemetry.cost_estimator import normalize_usage
from lattice.telemetry.downgrade import TransportOutcome


async def handle_chat_stream(
    deps: ChatCompatDeps,
    *,
    fastapi_request: Any,
    request: Any,
    compressed_request: Any,
    ctx: Any,
    session: Any,
    execution_plan: Any,
    messages: list[Any],
    provider_name: str,
    requested_model: str,
    client_api_key: str | None,
    cache_key: str | None,
    delta_mode: str,
    http_version: str,
    delta_savings_bytes: int,
    start_llm: float,
    x_lattice_client_profile: str | None,
) -> StreamingResponse:
    """Run streaming provider path and return StreamingResponse."""
    model_used = requested_model

    async def _stream_response() -> AsyncIterator[str]:
        from lattice.planner.fallback_executor import execute_with_fallback_stream

        first_chunk = True
        full_content = ""
        stream_meta: dict[str, Any] = {}
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        sse_chunks: list[str] = []
        try:
            stream_kwargs = dict(
                messages=messages,
                temperature=compressed_request.temperature,
                max_tokens=compressed_request.max_tokens,
                top_p=compressed_request.top_p,
                tools=compressed_request.tools,
                tool_choice=compressed_request.tool_choice,
                stop=compressed_request.stop,
                api_key=client_api_key if provider_name == "openai" else None,
                metadata=compressed_request.metadata,
                extra_headers=compressed_request.extra_headers,
                extra_body=compressed_request.extra_body,
            )
            stream = execute_with_fallback_stream(
                deps.provider.completion_stream_with_stall_detect,
                execution_plan=execution_plan,
                provider_name=provider_name,
                model=model_used,
                logger=deps.logger,
                metrics=deps.metrics,
                **stream_kwargs,
            )
            async for chunk in stream:
                if chunk is None:
                    continue
                if first_chunk:
                    chunk.setdefault("choices", [{}])
                    if chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        delta["_lattice"] = {
                            "model": model_used,
                            "session_id": session.session_id,
                        }
                        chunk["choices"][0]["delta"] = delta
                    first_chunk = False

                choices = chunk.get("choices", [])
                delta = choices[0].get("delta", {}) if choices else {}
                content = delta.get("content", "")
                if content:
                    full_content += content

                delta_tool_calls = delta.get("tool_calls")
                if delta_tool_calls:
                    for tc in delta_tool_calls:
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc.get("id", ""),
                                "type": tc.get("type", "function"),
                                "function": {"name": "", "arguments": ""},
                            }
                        if "function" in tc:
                            fn = tc["function"]
                            if "name" in fn:
                                tool_calls_acc[idx]["function"]["name"] = fn["name"]
                            if "arguments" in fn:
                                tool_calls_acc[idx]["function"]["arguments"] += fn["arguments"]

                lat_meta = chunk.pop("_lattice_metadata", None)
                if lat_meta:
                    stream_meta.update(lat_meta)

                if chunk.get("done"):
                    sse_chunks.append("data: [DONE]\n\n")
                    yield "data: [DONE]\n\n"
                    break
                sse_line = f"data: {json.dumps(chunk)}\n\n"
                sse_chunks.append(sse_line)
                yield sse_line
        except Exception as exc:
            deps.logger.error("stream_error", error=str(exc))
            error_payload = {"error": {"message": str(exc), "type": "stream_error"}}
            sse_chunks.append(f"data: {json.dumps(error_payload)}\n\n")
            yield f"data: {json.dumps(error_payload)}\n\n"
        finally:
            yield f"data: {deps.sse_done.strip()}\n\n"
            elapsed_ms = (time.perf_counter() - start_llm) * 1000
            deps.metrics.record_latency("lattice_llm_latency_ms", elapsed_ms)
            acc_tool_calls = list(tool_calls_acc.values()) if tool_calls_acc else None
            if full_content or acc_tool_calls:
                msg = deps.message_cls(
                    role="assistant",
                    content=full_content,
                    tool_calls=acc_tool_calls,
                )
                if stream_meta:
                    msg.metadata.update(stream_meta)
                session.messages.append(msg)
                await deps.session_manager.update_session(session.session_id, session.messages)
            # Feed cache telemetry back for observability
            stream_cached_tokens = _extract_cached_tokens(stream_meta.get("usage", {}))
            if stream_cached_tokens > 0:
                session.record_cache_hit(stream_cached_tokens)
            cache_plan_stream = ctx.session_state.get("cache_plan")
            if cache_plan_stream is None:
                cache_plan_stream = ctx.session_state.get("cache_plan_entries")
            if cache_plan_stream is not None and stream_meta is not None:
                if isinstance(cache_plan_stream, list):
                    total_expected = sum_expected_cached_tokens(cache_plan_stream)
                    breakpoints = len(cache_plan_stream)
                else:
                    total_expected = getattr(cache_plan_stream, "expected_cached_tokens", 0)
                    breakpoints_int: int = getattr(  # type: ignore[assignment]
                        cache_plan_stream, "breakpoints", []
                    )
                    breakpoints = breakpoints_int
                stream_meta["_cache_arbitrage_actual"] = {
                    "expected_cached_tokens": total_expected,
                    "actual_cached_tokens": stream_cached_tokens,
                    "breakpoints": breakpoints,
                    "provider": provider_name,
                }

            # Store in semantic cache
            if cache_key and deps.semantic_cache:
                await deps.semantic_cache.set(
                    cache_key,
                    assemble_cached_response(
                        model=model_used,
                        content=full_content,
                        tool_calls=list(tool_calls_acc.values()) if tool_calls_acc else None,
                        usage=stream_meta.get("usage", {}),
                        finish_reason="stop",
                        sse_chunks=sse_chunks,
                    ),
                    compressed_request,
                )

            # Record per-agent stats for streaming
            if deps.agent_stats:
                stream_usage = stream_meta.get("usage", {})
                normalized_usage = normalize_usage(stream_usage)
                stream_prompt = (
                    normalized_usage["prompt_tokens"] or compressed_request.token_estimate or 0
                )
                stream_completion = normalized_usage["completion_tokens"]
                stream_cached = normalized_usage["cached_tokens"]
                await deps.agent_stats.record_request(
                    agent=identify_agent(
                        request.extra_headers.get("user-agent"),
                        x_lattice_client_profile,
                    ),
                    provider=provider_name,
                    model=model_used,
                    prompt_tokens=stream_prompt,
                    completion_tokens=stream_completion,
                    compressed_tokens=compressed_request.token_estimate or 0,
                    original_tokens=request.token_estimate or 0,
                    cached_tokens=stream_cached,
                    cache_hit=stream_cached > 0,
                    cost_usd=stream_cost_usd,
                )

    # Pre-compute estimated cost for streaming responses
    stream_cost_usd = 0.0
    if deps.cost_estimator:
        est = deps.cost_estimator.estimate_request(
            provider=provider_name,
            model=model_used,
            prompt_tokens=compressed_request.token_estimate or 0,
            completion_tokens=0,  # unknown until stream finishes
        )
        stream_cost_usd = est.total_cost_usd

    transport_outcome = TransportOutcome(
        semantic_cache_status="miss" if cache_key is not None else "",
        delta_mode=delta_mode,
        http_version=http_version,
    )
    attach_routing_headers(
        fastapi_request,
        ctx,
        deps.build_routing_headers(
            model_used,
            session_id=session.session_id,
            delta_savings_bytes=delta_savings_bytes,
            cost_usd=stream_cost_usd,
            transport_outcome=transport_outcome,
            **_runtime_header_values(compressed_request),
        ),
    )
    return StreamingResponse(
        _stream_response(),
        media_type="text/event-stream",
    )
