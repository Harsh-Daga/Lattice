"""HTTP compatibility handler for proxy route delegation."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx
from fastapi import status
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.responses import Response as StarletteResponse

from lattice.core.agent_stats import identify_agent
from lattice.core.context import TransformContext
from lattice.core.cost_estimator import normalize_usage
from lattice.core.pipeline_factory import pipeline_summary
from lattice.core.result import is_err, unwrap
from lattice.core.semantic_cache import assemble_cached_response, compute_cache_key
from lattice.core.serialization import message_to_dict, request_from_dict, response_to_dict
from lattice.core.telemetry import TransportOutcome
from lattice.core.transport import Message, Request, Response
from lattice.gateway.server import LLMTPGateway
from lattice.protocol.manifest import manifest_summary
from lattice.providers.capabilities import Capability, get_capability_registry

Handler = Callable[..., Awaitable[Any]]


class HTTPCompatHandler:
    """Handles OpenAI/Anthropic/Responses compatibility routes.

    The proxy can register concrete handlers and keep this class as a
    thin, testable delegation layer while extraction is in progress.
    """

    def __init__(
        self,
        gateway: LLMTPGateway,
        *,
        chat_completion_handler: Handler | None = None,
        anthropic_handler: Handler | None = None,
        responses_handler: Handler | None = None,
        models_handler: Handler | None = None,
    ) -> None:
        self.gateway = gateway
        self.chat_completion_handler = chat_completion_handler
        self.anthropic_handler = anthropic_handler
        self.responses_handler = responses_handler
        self.models_handler = models_handler

    async def handle_chat_completion(self, *args: Any, **kwargs: Any) -> Any:
        if self.chat_completion_handler is None:
            raise RuntimeError("chat_completion_handler is not configured")
        return await self.chat_completion_handler(*args, **kwargs)

    async def handle_anthropic_message(self, *args: Any, **kwargs: Any) -> Any:
        if self.anthropic_handler is None:
            raise RuntimeError("anthropic_handler is not configured")
        return await self.anthropic_handler(*args, **kwargs)

    async def handle_responses_api(self, *args: Any, **kwargs: Any) -> Any:
        if self.responses_handler is None:
            raise RuntimeError("responses_handler is not configured")
        return await self.responses_handler(*args, **kwargs)

    async def handle_models(self, *args: Any, **kwargs: Any) -> Any:
        if self.models_handler is None:
            raise RuntimeError("models_handler is not configured")
        return await self.models_handler(*args, **kwargs)


# =============================================================================
# Shared proxy compatibility helpers
# =============================================================================

_SUPPORTED_PROVIDERS: tuple[str, ...] = (
    "openai",
    "anthropic",
    "ollama",
    "ollama-cloud",
    "azure",
    "bedrock",
    "groq",
    "together",
    "deepseek",
    "perplexity",
    "mistral",
    "fireworks",
    "openrouter",
    "cohere",
    "ai21",
    "gemini",
    "google",
    "vertex",
)


class ProviderDetectionError(Exception):
    """Raised when provider cannot be determined from request data."""


def build_routing_headers(
    model_used: str,
    *,
    compressed_tokens: int = 0,
    original_tokens: int = 0,
    session_id: str = "",
    anchor_version: int = 0,
    anchor_hash: str = "",
    used_speculative: bool | None = None,
    prediction_hit: bool | None = None,
    batched: bool | None = None,
    delta_savings_bytes: int = 0,
    cache_hit: bool | None = None,
    cached_tokens: int = 0,
    cost_usd: float = 0.0,
    cache_savings_usd: float = 0.0,
    runtime_tier: str = "",
    runtime_mode: str = "",
    runtime_budget_ms: float = 0.0,
    runtime_actual_ms: float = 0.0,
    runtime_budget_exhausted: bool = False,
    runtime_skipped_count: int = 0,
    framing: str = "",
    delta_mode: str = "",
    http_version: str = "",
    semantic_cache_status: str = "",
    batching_status: str = "",
    speculative_status: str = "",
    fallback_reason: str = "",
    stream_resumed: bool | None = None,
    transport_outcome: TransportOutcome | None = None,
) -> dict[str, str]:
    """Build response routing headers exposed by proxy/gateway.

    When *transport_outcome* is provided, transport-related headers are
    seeded from its canonical ``to_headers()`` output.  Individual legacy
    parameters are then applied on top, so an explicit legacy argument
    always overrides the canonical value.  This lets callers pin a header
    value without reconstructing the entire ``TransportOutcome``.

    Override semantics
    ------------------
    The legacy boolean parameters (used_speculative, batched, cache_hit,
    stream_resumed) use tri-state (None = use canonical, True/False =
    explicitly set).  An explicit ``False`` will **suppress** a canonical
    ``True``, resolving the previous ambiguity where ``False`` was
    indistinguishable from ``not set``.
    """
    compression = (
        f"{round(1 - compressed_tokens / max(original_tokens, 1), 4):.2%}"
        if original_tokens != compressed_tokens and original_tokens > 0
        else "0%"
    )
    headers: dict[str, str] = {
        "x-lattice-model": model_used,
        "x-lattice-compression": compression,
    }
    if session_id:
        headers["x-lattice-session-id"] = session_id
    if anchor_version > 0:
        headers["x-lattice-anchor-version"] = str(anchor_version)
    if anchor_hash:
        headers["x-lattice-anchor-hash"] = anchor_hash
    if delta_savings_bytes > 0:
        headers["x-lattice-delta-savings-bytes"] = str(delta_savings_bytes)
    if cached_tokens > 0:
        headers["x-lattice-cached-tokens"] = str(cached_tokens)
    if cost_usd > 0:
        headers["x-lattice-cost-usd"] = f"{cost_usd:.6f}"
    if cache_savings_usd > 0:
        headers["x-lattice-cache-savings-usd"] = f"{cache_savings_usd:.6f}"
    if runtime_tier:
        headers["x-lattice-runtime-tier"] = runtime_tier
    if runtime_mode:
        headers["x-lattice-runtime-mode"] = runtime_mode
    if runtime_budget_ms > 0:
        headers["x-lattice-runtime-budget-ms"] = f"{runtime_budget_ms:.2f}"
    if runtime_actual_ms > 0:
        headers["x-lattice-runtime-actual-ms"] = f"{runtime_actual_ms:.2f}"
    if runtime_budget_exhausted:
        headers["x-lattice-runtime-budget-exhausted"] = "true"
    if runtime_skipped_count > 0:
        headers["x-lattice-runtime-skipped-transforms"] = str(runtime_skipped_count)

    # Transport-related headers: start from canonical object, then let
    # legacy parameters override (explicit args win over defaults).
    if transport_outcome is not None:
        for k, v in transport_outcome.to_headers().items():
            headers.setdefault(k, v)

    # Tri-state legacy boolean overrides (applied AFTER canonical so
    # explicit False can suppress canonical True values).
    if used_speculative is not None:
        if used_speculative:
            headers["x-lattice-speculative"] = "hit" if prediction_hit else "miss"
        else:
            headers.pop("x-lattice-speculative", None)
            headers.pop("x-lattice-speculative-status", None)
    if batched is not None:
        if batched:
            headers["x-lattice-batched"] = "true"
        else:
            headers.pop("x-lattice-batched", None)
            headers.pop("x-lattice-batching", None)
    if cache_hit is not None:
        if cache_hit:
            headers["x-lattice-cache-hit"] = "true"
        else:
            headers.pop("x-lattice-cache-hit", None)
    if stream_resumed is not None:
        if stream_resumed:
            headers["x-lattice-stream-resumed"] = "true"
        else:
            headers.pop("x-lattice-stream-resumed", None)

    # Legacy individual-parameter path — always applied last so explicit
    # arguments override the canonical object.
    if framing:
        headers["x-lattice-framing"] = framing
    if delta_mode:
        headers["x-lattice-delta"] = delta_mode
    if http_version:
        headers["x-lattice-http-version"] = http_version
    if semantic_cache_status:
        headers["x-lattice-semantic-cache"] = semantic_cache_status
    if batching_status:
        headers["x-lattice-batching"] = batching_status
    if speculative_status:
        headers["x-lattice-speculative-status"] = speculative_status
    if fallback_reason:
        headers["x-lattice-fallback-reason"] = fallback_reason
    return headers


def _runtime_header_values(request: Request) -> dict[str, Any]:
    runtime = request.metadata.get("_lattice_runtime")
    contract = request.metadata.get("_lattice_runtime_contract")
    budget = request.metadata.get("_lattice_runtime_budget")
    if not isinstance(runtime, dict):
        runtime = {}
    if not isinstance(contract, dict):
        contract = {}
    if not isinstance(budget, dict):
        budget = {}
    return {
        "runtime_tier": str(runtime.get("tier") or ""),
        "runtime_mode": str(contract.get("mode") or ""),
        "runtime_budget_ms": float(contract.get("max_transform_latency_ms") or 0.0),
        "runtime_actual_ms": float(budget.get("actual_transform_ms") or 0.0),
        "runtime_budget_exhausted": bool(budget.get("exhausted") or False),
        "runtime_skipped_count": int(budget.get("skipped_count") or 0),
    }


def _extract_cached_tokens(usage: dict[str, Any]) -> int:
    """Extract cached-token count from provider usage dict."""
    return normalize_usage(usage).get("cached_tokens", 0)


def _usage_total_tokens(usage: dict[str, Any]) -> int:
    """Return total logical tokens represented by a usage dict."""
    normalized = normalize_usage(usage)
    total = usage.get("total_tokens") if isinstance(usage, dict) else None
    if isinstance(total, int):
        return total
    return normalized["prompt_tokens"] + normalized["completion_tokens"]


def detect_provider(model: str, provider_hint: str | None = None) -> str:
    """Determine provider from explicit hint or model prefix."""
    if provider_hint:
        hint = provider_hint.lower().strip()
        if hint in _SUPPORTED_PROVIDERS:
            return hint
        raise ProviderDetectionError(
            f"Unknown provider hint '{provider_hint}'. "
            f"Supported: {', '.join(_SUPPORTED_PROVIDERS)}"
        )
    if "/" in model:
        prefix = model.split("/", 1)[0].lower()
        if prefix in _SUPPORTED_PROVIDERS:
            return prefix
        raise ProviderDetectionError(
            f"Unknown provider prefix '{prefix}' in model '{model}'. "
            f"Supported prefixes: {', '.join(_SUPPORTED_PROVIDERS)}"
        )
    raise ProviderDetectionError(
        f"Provider not specified. Use either: "
        f"1) x-lattice-provider header, or "
        f"2) model prefix like 'groq/llama-3b' (got model='{model}')"
    )


def detect_new_messages(existing: list[Message], incoming: list[Message]) -> list[Message]:
    """Detect newly appended messages against existing conversation state."""
    if len(incoming) <= len(existing):
        return []
    if len(existing) == 0:
        return list(incoming)
    for _i, (a, b) in enumerate(zip(existing, incoming, strict=False)):
        if a.content != b.content or a.role != b.role:
            return list(incoming)
    return list(incoming[len(existing):])


def deserialize_openai_request(body: dict[str, Any]) -> Request:
    """Convert OpenAI JSON request body into internal request."""
    return request_from_dict(body)


def serialize_messages(request: Request) -> list[dict[str, Any]]:
    """Convert internal request messages into OpenAI list format."""
    return [message_to_dict(m) for m in request.messages]


def serialize_openai_response(response: Response, request: Request) -> dict[str, Any]:
    """Convert internal response into OpenAI-compatible response body."""
    return response_to_dict(response, request_model=request.model)


def is_local_origin(request: Any) -> bool:
    """Check whether request appears to come from localhost."""
    host = request.headers.get("host", "")
    remote = request.client.host if request.client else ""
    forwarded = request.headers.get("x-forwarded-for", "")
    return (
        host.startswith("127.0.0.1")
        or host.startswith("localhost")
        or remote in ("127.0.0.1", "::1")
        or forwarded.startswith("127.0.0.1")
    )


def deserialize_anthropic_request(body: dict[str, Any]) -> Request:
    """Convert Anthropic Messages API JSON body into internal request."""
    messages: list[Message] = []

    system = body.get("system")
    if system is not None:
        if isinstance(system, str):
            messages.append(Message(role="system", content=system))
        elif isinstance(system, list):
            system_texts: list[str] = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    system_texts.append(block.get("text", ""))
            if system_texts:
                messages.append(Message(role="system", content="\n".join(system_texts)))

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] | None = None
            tool_call_id: str | None = None
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append(
                        {
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        }
                    )
                elif btype == "tool_result":
                    tool_call_id = block.get("tool_use_id", "")
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_texts = [
                            b.get("text", "")
                            for b in result_content
                            if isinstance(b, dict) and b.get("type") == "text"
                        ]
                        text_parts.append("\n".join(result_texts))
                    elif isinstance(result_content, str):
                        text_parts.append(result_content)
                elif btype == "image":
                    text_parts.append(
                        f"[Image: {block.get('source', {}).get('type', 'unknown')}]"
                    )
            content = "\n".join(text_parts)
        else:
            content = str(content)
            tool_calls = None
            tool_call_id = None

        messages.append(
            Message(
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
            )
        )

    tools = body.get("tools")
    if tools is not None:
        remapped_tools: list[dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "custom" and "custom" in tool:
                tool = tool["custom"]
            if "input_schema" in tool:
                remapped_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.get("name", ""),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("input_schema", {}),
                        },
                    }
                )
            else:
                remapped_tools.append(tool)
        tools = remapped_tools

    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "tool":
        tool_choice = {"type": "function", "function": {"name": tool_choice.get("name", "")}}

    req = Request(
        messages=messages,
        model=body.get("model", ""),
        temperature=body.get("temperature"),
        max_tokens=body.get("max_tokens"),
        top_p=body.get("top_p"),
        tools=tools,
        tool_choice=tool_choice,
        stream=body.get("stream", False),
        stop=body.get("stop_sequences"),
    )
    if "thinking" in body:
        req.metadata["thinking"] = body["thinking"]
    if "metadata" in body:
        req.metadata["anthropic_metadata"] = body["metadata"]
    return req


def serialize_anthropic_response(response: Response, request: Request) -> dict[str, Any]:
    """Convert internal response into Anthropic Messages API response body."""
    content_blocks: list[dict[str, Any]] = []
    if response.content:
        content_blocks.append({"type": "text", "text": response.content})
    if response.tool_calls:
        for tc in response.tool_calls:
            func = tc.get("function", {})
            args = func.get("arguments", "{}")
            try:
                input_data = json.loads(args) if isinstance(args, str) else args
            except json.JSONDecodeError:
                input_data = {}
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "input": input_data,
                }
            )

    stop_reason: str | None = None
    if response.finish_reason == "stop":
        stop_reason = "end_turn"
    elif response.finish_reason == "length":
        stop_reason = "max_tokens"
    elif response.finish_reason == "tool_calls":
        stop_reason = "tool_use"

    return {
        "id": response.metadata.get("anthropic_message_id", f"msg_{int(time.time())}"),
        "type": "message",
        "role": "assistant",
        "model": response.model or request.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": response.metadata.get("stop_sequence"),
        "usage": response.usage or {},
    }


def extract_anthropic_text_blocks(body: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    """Extract mutable Anthropic text blocks for selective compression."""
    blocks: list[tuple[dict[str, Any], str]] = []

    system = body.get("system")
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                blocks.append((block, block.get("text", "")))

    for msg in body.get("messages", []):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            blocks.append(({"_msg_content": msg, "type": "_string"}, content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    blocks.append((block, block.get("text", "")))
                elif isinstance(block, dict) and block.get("type") == "tool_result":
                    result_content = block.get("content")
                    if isinstance(result_content, str):
                        blocks.append(
                            ({"_tool_result_content": block, "type": "_string"}, result_content)
                        )
                    elif isinstance(result_content, list):
                        for inner in result_content:
                            if isinstance(inner, dict) and inner.get("type") == "text":
                                blocks.append((inner, inner.get("text", "")))
    return blocks


def replace_anthropic_text_blocks(
    blocks: list[tuple[dict[str, Any], str]], compressed_texts: list[str]
) -> None:
    """Write compressed text back into extracted Anthropic block references."""
    for (block, _original), compressed in zip(blocks, compressed_texts, strict=True):
        if block.get("type") == "_string":
            if "_msg_content" in block:
                block["_msg_content"]["content"] = compressed
            elif "_tool_result_content" in block:
                block["_tool_result_content"]["content"] = compressed
        else:
            block["text"] = compressed


async def compress_anthropic_body(
    body: dict[str, Any],
    pipeline: Any,
    config: Any,
    provider_name: str,
    model: str,
    logger: Any,
) -> tuple[dict[str, Any], TransformContext, int, int]:
    """Compress Anthropic request text blocks while preserving shape."""
    blocks = extract_anthropic_text_blocks(body)
    if not blocks:
        return body, TransformContext(
            request_id=str(time.time()), provider=provider_name, model=model
        ), 0, 0

    pseudo_messages = [
        Message(role="user" if i % 2 == 0 else "assistant", content=text)
        for i, (_block, text) in enumerate(blocks)
    ]
    pseudo_request = Request(messages=pseudo_messages, model=model)
    original_tokens = pseudo_request.token_estimate

    ctx = TransformContext(request_id=str(time.time()), provider=provider_name, model=model)
    result = await pipeline.process(pseudo_request, ctx)
    if is_err(result):
        if config.graceful_degradation:
            logger.warning("anthropic_content_compression_degraded", error=str(result))
            return body, ctx, original_tokens, original_tokens
        raise Exception(f"Anthropic content compression failed: {result}")

    compressed_request = unwrap(result)
    compressed_texts = [m.content for m in compressed_request.messages]
    replace_anthropic_text_blocks(blocks, compressed_texts)
    return body, ctx, original_tokens, compressed_request.token_estimate


def extract_responses_text_blocks(body: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    """Extract mutable text blocks from OpenAI Responses request payloads."""
    blocks: list[tuple[dict[str, Any], str]] = []

    instructions = body.get("instructions")
    if isinstance(instructions, str):
        blocks.append(({"_instructions": body, "type": "_string"}, instructions))

    for item in body.get("input", []):
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str):
            blocks.append(({"_item_content": item, "type": "_string"}, content))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    blocks.append((part, part.get("text", "")))
    return blocks


def replace_responses_text_blocks(
    blocks: list[tuple[dict[str, Any], str]], compressed_texts: list[str]
) -> None:
    """Write compressed text back into extracted Responses text blocks."""
    for (block, _original), compressed in zip(blocks, compressed_texts, strict=True):
        if block.get("type") == "_string":
            if "_instructions" in block:
                block["_instructions"]["instructions"] = compressed
            elif "_item_content" in block:
                block["_item_content"]["content"] = compressed
        else:
            block["text"] = compressed


async def compress_responses_body(
    body: dict[str, Any],
    pipeline: Any,
    config: Any,
    model: str,
    logger: Any,
) -> tuple[dict[str, Any], TransformContext, int, int]:
    """Compress OpenAI Responses request text blocks while preserving shape."""
    blocks = extract_responses_text_blocks(body)
    if not blocks:
        return body, TransformContext(
            request_id=str(time.time()), provider="openai", model=model
        ), 0, 0

    pseudo_messages = [
        Message(role="user" if i % 2 == 0 else "assistant", content=text)
        for i, (_block, text) in enumerate(blocks)
    ]
    pseudo_request = Request(messages=pseudo_messages, model=model)
    original_tokens = pseudo_request.token_estimate

    ctx = TransformContext(request_id=str(time.time()), provider="openai", model=model)
    result = await pipeline.process(pseudo_request, ctx)
    if is_err(result):
        if config.graceful_degradation:
            logger.warning("responses_content_compression_degraded", error=str(result))
            return body, ctx, original_tokens, original_tokens
        raise Exception(f"Responses content compression failed: {result}")

    compressed_request = unwrap(result)
    compressed_texts = [m.content for m in compressed_request.messages]
    replace_responses_text_blocks(blocks, compressed_texts)
    return body, ctx, original_tokens, compressed_request.token_estimate


async def responses_passthrough(
    method: str,
    path: str,
    body: bytes,
    fastapi_request: Any,
    provider: Any,
    *,
    compressed_tokens: int = 0,
    original_tokens: int = 0,
    logger: Any,
    session_id: str | None = None,
) -> Any:
    """Forward OpenAI Responses API requests directly to configured upstream."""
    provider_name = "openai"
    base_url = provider.provider_base_urls.get(provider_name)
    if not base_url:
        raise ValueError(f"No upstream base URL configured for provider '{provider_name}'")

    upstream_url = f"{base_url.rstrip('/')}{path}"
    query = str(fastapi_request.query_params)
    if query:
        upstream_url = f"{upstream_url}?{query}"

    headers: dict[str, str] = {}
    for k, v in fastapi_request.headers.items():
        kl = k.lower()
        if kl in ("host", "content-length", "connection", "keep-alive"):
            continue
        headers[k] = v

    is_streaming = False
    if body:
        with contextlib.suppress(Exception):
            body_json = json.loads(body)
            is_streaming = body_json.get("stream", False)

    logger.info(
        "responses_passthrough_start",
        url=upstream_url,
        is_streaming=is_streaming,
        has_auth=bool(headers.get("authorization") or headers.get("Authorization")),
        body_bytes=len(body),
    )
    client = provider.pool.get_client(provider_name, base_url)
    http_version = provider.pool.get_http_version(provider_name, base_url)

    if is_streaming:

        async def _stream_relay() -> Any:
            try:
                async with client.stream(
                    method, upstream_url, content=body, headers=headers
                ) as resp:
                    if not resp.is_success:
                        error_body = await resp.aread()
                        logger.error(
                            "responses_passthrough_stream_error",
                            status_code=resp.status_code,
                            error_body=error_body.decode("utf-8", errors="replace")[:500],
                        )
                        yield (
                            f'event: error\ndata: '
                            f'{{"error":"upstream_error","message":"HTTP {resp.status_code}"}}\n\n'
                        )
                        return
                    async for chunk in resp.aiter_text():
                        yield chunk
            except Exception as exc:
                logger.error(
                    "responses_passthrough_stream_exception",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                yield (
                    f'event: error\ndata: '
                    f'{{"error":"stream_error","message":"{str(exc)}"}}\n\n'
                )

        transport_outcome = TransportOutcome(
            http_version=http_version,
        )
        return StreamingResponse(
            _stream_relay(),
            media_type="text/event-stream",
            headers=build_routing_headers(
                model_used="gpt",
                compressed_tokens=compressed_tokens,
                original_tokens=original_tokens,
                session_id=session_id or "",
                transport_outcome=transport_outcome,
            ),
        )

    try:
        http_resp = await client.request(method, upstream_url, content=body, headers=headers)
    except httpx.TimeoutException:
        return JSONResponse(
            {"error": "upstream_timeout"},
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        )
    except Exception as exc:
        logger.error(
            "responses_passthrough_error",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return JSONResponse(
            {"error": "upstream_error", "message": str(exc)},
            status_code=status.HTTP_502_BAD_GATEWAY,
        )

    response_headers = {
        k: v
        for k, v in http_resp.headers.items()
        if k.lower()
        in (
            "content-type",
            "x-request-id",
            "x-ratelimit-limit-requests",
            "x-ratelimit-limit-tokens",
            "x-ratelimit-remaining-requests",
            "x-ratelimit-remaining-tokens",
        )
    }
    transport_outcome = TransportOutcome(
        http_version=http_version,
    )
    response_headers.update(
        build_routing_headers(
            model_used="gpt",
            compressed_tokens=compressed_tokens,
            original_tokens=original_tokens,
            session_id=session_id or "",
            transport_outcome=transport_outcome,
        )
    )
    return StarletteResponse(
        content=http_resp.content,
        status_code=http_resp.status_code,
        headers=response_headers,
    )


async def responses_websocket_passthrough(websocket: Any, *, logger: Any) -> None:
    """Relay websocket traffic between client and OpenAI Responses upstream.

    Supports both standard OpenAI Responses and Codex-specific routing:
    * Standard → ``wss://api.openai.com/v1/responses``
    * Codex     → ``wss://chatgpt.com/backend-api/codex/responses``
      (detected via JWT ``chatgpt_account_id`` claim)
    """
    import websockets as _ws_lib

    from lattice.integrations.codex.auth import (
        _is_codex_jwt,
        _resolve_codex_routing_headers,
    )

    await websocket.accept()
    auth = websocket.headers.get("authorization", "")
    openai_beta = websocket.headers.get("openai-beta", "")

    if _is_codex_jwt(auth):
        upstream_uri = "wss://chatgpt.com/backend-api/codex/responses"
        upstream_headers = _resolve_codex_routing_headers(auth, openai_beta)
        logger.info("codex_websocket_routing", upstream=upstream_uri)
    else:
        upstream_uri = "wss://api.openai.com/v1/responses"
        upstream_headers: dict[str, str] = {}
        if auth:
            upstream_headers["Authorization"] = auth
        if openai_beta:
            upstream_headers["OpenAI-Beta"] = openai_beta

    upstream_ws = None
    try:
        upstream_ws = await _ws_lib.connect(upstream_uri, additional_headers=upstream_headers)
    except Exception as exc:
        logger.warning("websocket_upstream_connect_failed", error=str(exc))
        await websocket.close(code=1011, reason=f"Upstream connect failed: {exc}")
        return

    async def client_to_upstream() -> None:
        try:
            while True:
                msg = await websocket.receive_text()
                await upstream_ws.send(msg)
        except (
            _ws_lib.exceptions.ConnectionClosed,
            _ws_lib.exceptions.ConnectionClosedOK,
            _ws_lib.exceptions.ConnectionClosedError,
            RuntimeError,
        ):
            # Client or upstream closed — expected on normal teardown
            pass

    async def upstream_to_client() -> None:
        try:
            while True:
                msg = await upstream_ws.recv()
                if isinstance(msg, bytes):
                    await websocket.send_bytes(msg)
                else:
                    await websocket.send_text(msg)
        except (
            _ws_lib.exceptions.ConnectionClosed,
            _ws_lib.exceptions.ConnectionClosedOK,
            _ws_lib.exceptions.ConnectionClosedError,
            RuntimeError,
        ):
            # Client or upstream closed — expected on normal teardown
            pass

    try:
        await asyncio.gather(client_to_upstream(), upstream_to_client())
    finally:
        if upstream_ws:
            await upstream_ws.close()
        with contextlib.suppress(Exception):
            await websocket.close()


async def anthropic_passthrough(
    method: str,
    path: str,
    body: bytes,
    fastapi_request: Any,
    provider: Any,
    *,
    compressed_tokens: int = 0,
    original_tokens: int = 0,
    logger: Any,
    session_id: str | None = None,
) -> Any:
    """Forward Anthropic Messages API requests directly to configured upstream."""
    provider_name = "anthropic"
    base_url = provider.provider_base_urls.get(provider_name)
    if not base_url:
        raise ValueError(f"No upstream base URL configured for provider '{provider_name}'")

    upstream_url = f"{base_url.rstrip('/')}{path}"
    query = str(fastapi_request.query_params)
    if query:
        upstream_url = f"{upstream_url}?{query}"

    headers: dict[str, str] = {}
    for k, v in fastapi_request.headers.items():
        kl = k.lower()
        if kl in ("host", "content-length", "connection", "keep-alive"):
            continue
        headers[k] = v

    is_streaming = False
    if body:
        with contextlib.suppress(Exception):
            body_json = json.loads(body)
            is_streaming = body_json.get("stream", False)

    logger.info(
        "anthropic_passthrough_start",
        url=upstream_url,
        is_streaming=is_streaming,
        has_auth=bool(headers.get("authorization") or headers.get("Authorization")),
        body_bytes=len(body),
    )

    client = provider.pool.get_client(provider_name, base_url)
    http_version = provider.pool.get_http_version(provider_name, base_url)
    if is_streaming:

        async def _stream_relay() -> Any:
            try:
                async with client.stream(
                    method, upstream_url, content=body, headers=headers
                ) as resp:
                    if not resp.is_success:
                        error_body = await resp.aread()
                        logger.error(
                            "anthropic_passthrough_stream_error",
                            status_code=resp.status_code,
                            error_body=error_body.decode("utf-8", errors="replace")[:500],
                        )
                        yield (
                            f'event: error\ndata: '
                            f'{{"type":"error","error":{{"type":"upstream_error",'
                            f'"message":"HTTP {resp.status_code}"}}}}\n\n'
                        )
                        return
                    async for chunk in resp.aiter_text():
                        yield chunk
            except Exception as exc:
                logger.error(
                    "anthropic_passthrough_stream_exception",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                yield (
                    f'event: error\ndata: '
                    f'{{"type":"error","error":{{"type":"stream_error",'
                    f'"message":"{str(exc)}"}}}}\n\n'
                )

        transport_outcome = TransportOutcome(
            http_version=http_version,
        )
        return StreamingResponse(
            _stream_relay(),
            media_type="text/event-stream",
            headers=build_routing_headers(
                model_used="claude",
                compressed_tokens=compressed_tokens,
                original_tokens=original_tokens,
                session_id=session_id or "",
                transport_outcome=transport_outcome,
            ),
        )

    try:
        http_resp = await client.request(method, upstream_url, content=body, headers=headers)
    except httpx.TimeoutException:
        return JSONResponse(
            {
                "type": "error",
                "error": {
                    "type": "timeout_error",
                    "message": "Upstream request timed out",
                },
            },
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        )
    except Exception as exc:
        logger.error(
            "anthropic_passthrough_error",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return JSONResponse(
            {
                "type": "error",
                "error": {
                    "type": "upstream_error",
                    "message": str(exc),
                },
            },
            status_code=status.HTTP_502_BAD_GATEWAY,
        )

    response_headers = {
        k: v
        for k, v in http_resp.headers.items()
        if k.lower()
        in (
            "content-type",
            "x-request-id",
            "anthropic-ratelimit-requests-limit",
            "anthropic-ratelimit-tokens-limit",
            "anthropic-ratelimit-requests-remaining",
            "anthropic-ratelimit-tokens-remaining",
        )
    }
    transport_outcome = TransportOutcome(
        http_version=http_version,
    )
    response_headers.update(
        build_routing_headers(
            model_used="claude",
            compressed_tokens=compressed_tokens,
            original_tokens=original_tokens,
            session_id=session_id or "",
            transport_outcome=transport_outcome,
        )
    )
    return StarletteResponse(
        content=http_resp.content,
        status_code=http_resp.status_code,
        headers=response_headers,
    )


@dataclasses.dataclass(slots=True)
class ChatCompatDeps:
    """Dependencies required by chat completion compatibility handler."""

    config: Any
    pipeline: Any
    provider: Any
    session_manager: Any
    batching_engine: Any
    speculative_executor: Any
    semantic_cache: Any
    cost_estimator: Any
    auto_continuation: Any
    agent_stats: Any
    metrics: Any
    logger: Any
    deserialize_openai_request: Callable[[dict[str, Any]], Any]
    serialize_messages: Callable[[Any], list[dict[str, Any]]]
    serialize_openai_response: Callable[[Any, Any], dict[str, Any]]
    build_routing_headers: Callable[..., dict[str, str]]
    detect_provider: Callable[[str, str | None], str]
    detect_new_messages: Callable[[list[Any], list[Any]], list[Any]]
    get_cache_planner: Callable[[str], Any]
    message_cls: Any
    provider_timeout_error: type[Exception]
    provider_error: type[Exception]
    sse_done: str
    maintenance: Any = None


def make_chat_completion_handler(deps: ChatCompatDeps) -> Handler:
    """Create OpenAI-compatible chat completions handler."""

    async def _handle_chat_completion(
        body: dict[str, Any],
        x_lattice_session_id: str | None = None,
        x_lattice_disable_transforms: str | None = None,
        x_lattice_client_profile: str | None = None,
        x_lattice_provider: str | None = None,
        authorization: str | None = None,
        x_api_key: str | None = None,
    ) -> Any:
        request = deps.deserialize_openai_request(body)
        request.extra_headers["x-lattice-session-id"] = x_lattice_session_id or ""

        client_api_key = None
        if authorization and authorization.startswith("Bearer "):
            client_api_key = authorization[7:]

        # Preserve additional auth headers so upstream provider adapters can use them
        if x_api_key:
            request.extra_headers["x-api-key"] = x_api_key

        try:
            provider_name = deps.detect_provider(
                request.model or body.get("model", "gpt-4"),
                x_lattice_provider,
            )
        except Exception as exc:
            return JSONResponse(
                {"error": "provider_detection_failed", "message": str(exc)},
                status_code=status.HTTP_400_BAD_REQUEST,
            )

        delta_mode = "delta" if request.metadata.get("_delta_wire") else ""
        base_url = deps.provider.provider_base_urls.get(provider_name, "")
        http_version = deps.provider.pool.get_http_version(provider_name, base_url)

        # Throttled maintenance: clean up abandoned stream state and stale cache entries
        if deps.maintenance is not None:
            try:
                maintenance_results = await deps.maintenance.tick()
                for name, result in maintenance_results.items():
                    if result.did_work:
                        deps.logger.debug(
                            "maintenance_tick_did_work",
                            name=name,
                            stale_streams_removed=result.stale_streams_removed,
                            stale_cache_entries_removed=result.stale_cache_entries_removed,
                        )
            except Exception:
                pass  # Never block the request path for maintenance

        # Auto-generate session ID if absent
        if not x_lattice_session_id:
            import secrets

            x_lattice_session_id = f"sess_{secrets.token_hex(8)}"

        ctx = TransformContext(
            request_id=str(time.time()),
            session_id=x_lattice_session_id,
            provider=provider_name,
            model=request.model or body.get("model", "gpt-4"),
        )
        ctx.session_state["client_profile"] = x_lattice_client_profile or "default"

        session, was_created = await deps.session_manager.get_or_create_session(
            session_id=x_lattice_session_id,
            provider=provider_name,
            model=request.model or body.get("model", "gpt-4"),
            messages=request.messages,
            tools=request.tools,
        )

        # Compute delta savings against prior session state
        delta_savings_bytes = 0
        if not was_created:
            existing = session.messages
            new_msgs = deps.detect_new_messages(existing, request.messages)
            if new_msgs:
                full_messages = existing + new_msgs
                request.messages = full_messages
                await deps.session_manager.update_session(session.session_id, full_messages)
            else:
                await deps.session_manager.update_session(session.session_id, request.messages)
            # Calculate wire savings if client had used delta encoding
            from lattice.core.delta_wire import delta_wire_bytes

            full_msgs = deps.serialize_messages(request)
            new_raw = [msg.to_dict() if hasattr(msg, "to_dict") else msg for msg in new_msgs]
            if new_raw:
                try:
                    full_bytes, delta_bytes = delta_wire_bytes(
                        full_msgs, new_raw, session.session_id, len(existing)
                    )
                    delta_savings_bytes = max(0, full_bytes - delta_bytes)
                except (TypeError, ValueError) as exc:
                    deps.logger.warning("delta_wire_bytes_failed", error=str(exc), session_id=session.session_id)

        if session.manifest:
            cache_planner = deps.get_cache_planner(provider_name)
            cache_plan = cache_planner.plan(session.manifest)
            ctx.session_state["cache_plan"] = cache_plan
            ctx.record_metric(
                "cache_planner",
                "expected_cached_tokens",
                cache_plan.expected_cached_tokens,
            )
            ctx.record_metric("cache_planner", "breakpoints", len(cache_plan.breakpoints))

        if x_lattice_disable_transforms:
            compressed_request = request
        else:
            result = await deps.pipeline.process(request, ctx)
            if is_err(result):
                if deps.config.graceful_degradation:
                    compressed_request = request
                    deps.logger.warning("pipeline_degraded", error=str(result))
                else:
                    return JSONResponse(
                        {
                            "error": "pipeline_failed",
                            "message": "Transform error — set graceful_degradation=true to continue",
                        },
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    )
            else:
                compressed_request = unwrap(result)

        if client_api_key:
            compressed_request.metadata["_lattice_client_api_key"] = client_api_key

        # ------------------------------------------------------------------
        # Semantic cache check (after transforms, before provider call)
        # ------------------------------------------------------------------
        cache_key = None
        disable_cache = request.extra_headers.get("x-lattice-disable-cache", "")
        if deps.semantic_cache and deps.semantic_cache.enabled and not disable_cache:
            cache_key = compute_cache_key(compressed_request)
            cached = await deps.semantic_cache.get(cache_key, compressed_request)
            if cached is not None:
                deps.metrics.increment("lattice_semantic_cache_hit")
                deps.logger.info(
                    "semantic_cache_hit",
                    key=cache_key[:16],
                    model=cached.model,
                    content_len=len(cached.content),
                )
                # Serve cached response as either streaming or non-streaming
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
                    from lattice.core.semantic_cache import generate_sse_chunks

                    sse_chunks = generate_sse_chunks(
                        cached,
                        request_id=ctx.request_id,
                        session_id=session.session_id,
                    )

                    async def _cached_stream() -> AsyncIterator[str]:
                        for chunk in sse_chunks:
                            yield chunk
                        yield f"data: {deps.sse_done.strip()}\n\n"
                        # Update session with cached assistant message
                        if cached.content or cached.tool_calls:
                            msg = deps.message_cls(
                                role="assistant",
                                content=cached.content,
                                tool_calls=cached.tool_calls,
                            )
                            session.messages.append(msg)
                            await deps.session_manager.update_session(
                                session.session_id, session.messages
                            )
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
                    return StreamingResponse(
                        _cached_stream(),
                        media_type="text/event-stream",
                        headers=deps.build_routing_headers(
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
                else:
                    # Rebuild a Response-like object for serialization
                    from lattice.core.transport import Response

                    cached_response = Response(
                        content=cached.content,
                        tool_calls=cached.tool_calls,
                        usage=cached.usage,
                        model=cached.model,
                        finish_reason=cached.finish_reason,
                    )
                    # Run reverse transforms on cached response
                    if not x_lattice_disable_transforms:
                        cached_response = await deps.pipeline.reverse(
                            cached_response, ctx
                        )
                    response_body = deps.serialize_openai_response(
                        cached_response, compressed_request
                    )
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
                    for k, v in deps.build_routing_headers(
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
                    ).items():
                        response.headers[k] = v
                    # Update session with cached assistant message
                    if cached.content or cached.tool_calls:
                        msg = deps.message_cls(
                            role="assistant",
                            content=cached.content,
                            tool_calls=cached.tool_calls,
                        )
                        session.messages.append(msg)
                        await deps.session_manager.update_session(
                            session.session_id, session.messages
                        )
                    return response

        start_llm = time.perf_counter()
        try:
            messages = deps.serialize_messages(compressed_request)
            stream = compressed_request.stream
            requested_model = compressed_request.model or "gpt-4"

            if stream:
                model_used = requested_model

                async def _stream_response() -> AsyncIterator[str]:
                    first_chunk = True
                    full_content = ""
                    stream_meta: dict[str, Any] = {}
                    tool_calls_acc: dict[int, dict[str, Any]] = {}
                    sse_chunks: list[str] = []
                    try:
                        async for chunk in deps.provider.completion_stream_with_stall_detect(
                            model=model_used,
                            messages=messages,
                            temperature=compressed_request.temperature,
                            max_tokens=compressed_request.max_tokens,
                            top_p=compressed_request.top_p,
                            tools=compressed_request.tools,
                            tool_choice=compressed_request.tool_choice,
                            stop=compressed_request.stop,
                            provider_name=provider_name,
                            api_key=client_api_key,
                            metadata=compressed_request.metadata,
                            extra_headers=compressed_request.extra_headers,
                            extra_body=compressed_request.extra_body,
                        ):
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
                            await deps.session_manager.update_session(
                                session.session_id, session.messages
                            )
                        # Feed cache telemetry back for observability
                        stream_cached_tokens = _extract_cached_tokens(stream_meta.get("usage", {}))
                        if stream_cached_tokens > 0:
                            session.record_cache_hit(stream_cached_tokens)
                        cache_plan_stream = ctx.session_state.get("cache_plan")
                        if cache_plan_stream is not None and stream_meta is not None:
                            stream_meta["_cache_arbitrage_actual"] = {
                                "expected_cached_tokens": cache_plan_stream.expected_cached_tokens,
                                "actual_cached_tokens": stream_cached_tokens,
                                "breakpoints": cache_plan_stream.breakpoints,
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
                            stream_prompt = normalized_usage["prompt_tokens"] or compressed_request.token_estimate or 0
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
                return StreamingResponse(
                    _stream_response(),
                    media_type="text/event-stream",
                    headers=deps.build_routing_headers(
                        model_used,
                        session_id=session.session_id,
                        delta_savings_bytes=delta_savings_bytes,
                        cost_usd=stream_cost_usd,
                        transport_outcome=transport_outcome,
                        **_runtime_header_values(compressed_request),
                    ),
                )

            used_speculative = False
            prediction_hit = False
            batching_eligible = ctx.metrics.get("transforms", {}).get("batching", {}).get("eligible", False)
            if batching_eligible and not x_lattice_disable_transforms:
                try:
                    compressed_request.metadata["_lattice_is_batch"] = True
                    internal_response = await deps.batching_engine.submit(compressed_request, ctx)
                    model_used = requested_model
                    deps.metrics.increment("lattice_batching_dispatched")
                    if hasattr(deps.provider, "tacc"):
                        await deps.provider.tacc.record_batch_pressure(provider_name, 1)
                except Exception as batch_exc:
                    deps.logger.warning("batching_failed", error=str(batch_exc))
                    batching_eligible = False

            if not batching_eligible:
                prediction = ctx.session_state.get("speculative_prediction")
                if prediction and not x_lattice_disable_transforms:
                    compressed_request.metadata["_lattice_is_speculative"] = True
                    speculative_task = asyncio.create_task(
                        deps.speculative_executor.run_speculative(compressed_request, prediction)
                    )
                    real_start = time.perf_counter()
                    internal_response = await deps.provider.completion(
                        model=requested_model,
                        messages=messages,
                        temperature=compressed_request.temperature,
                        max_tokens=compressed_request.max_tokens,
                        top_p=compressed_request.top_p,
                        tools=compressed_request.tools,
                        tool_choice=compressed_request.tool_choice,
                        stream=False,
                        stop=compressed_request.stop,
                        provider_name=provider_name,
                        api_key=client_api_key,
                        metadata=compressed_request.metadata,
                        extra_headers=compressed_request.extra_headers,
                        extra_body=compressed_request.extra_body,
                    )
                    model_used = requested_model
                    real_latency_ms = (time.perf_counter() - real_start) * 1000

                    speculative_response = None
                    if not speculative_task.done():
                        with contextlib.suppress(asyncio.TimeoutError):
                            speculative_response = await asyncio.wait_for(speculative_task, timeout=0.5)
                    else:
                        speculative_response = speculative_task.result()

                    if speculative_response is not None:
                        actual = deps.speculative_executor.extract_actual_step(internal_response)
                        hit = deps.speculative_executor.is_hit(prediction, actual)
                        deps.speculative_executor.record_result(
                            hit=hit,
                            _predicted=prediction,
                            _actual=actual,
                            latency_ms=real_latency_ms,
                        )
                        if hit:
                            internal_response = speculative_response
                            used_speculative = True
                            prediction_hit = True
                            deps.metrics.increment("lattice_speculative_hit")
                        else:
                            deps.metrics.increment("lattice_speculative_miss")
                else:
                    internal_response = await deps.provider.completion(
                        model=requested_model,
                        messages=messages,
                        temperature=compressed_request.temperature,
                        max_tokens=compressed_request.max_tokens,
                        top_p=compressed_request.top_p,
                        tools=compressed_request.tools,
                        tool_choice=compressed_request.tool_choice,
                        stream=False,
                        stop=compressed_request.stop,
                        provider_name=provider_name,
                        api_key=client_api_key,
                        metadata=compressed_request.metadata,
                        extra_headers=compressed_request.extra_headers,
                        extra_body=compressed_request.extra_body,
                    )
                    model_used = requested_model

            # Auto-continuation for truncated responses (non-streaming only)
            cont_result = None
            if (
                deps.auto_continuation
                and not stream
                and internal_response.finish_reason == "length"
            ):
                deps.logger.info(
                    "auto_continuation_triggered",
                    session_id=session.session_id,
                    turns=deps.auto_continuation.max_turns,
                )
                cont_result = await deps.auto_continuation.continue_if_needed(
                    request=compressed_request,
                    initial_response=internal_response,
                    provider_caller=deps.provider.completion,
                    session_manager=deps.session_manager,
                    message_cls=deps.message_cls,
                    provider_name=provider_name,
                )
                if cont_result.was_continued:
                    internal_response = cont_result.response
                    deps.metrics.increment("lattice_auto_continuation_turns", cont_result.turns)
                    deps.logger.info(
                        "auto_continuation_complete",
                        session_id=session.session_id,
                        turns=cont_result.turns,
                        final_length=len(internal_response.content or ""),
                    )

            if internal_response.content or internal_response.tool_calls:
                msg = deps.message_cls(
                    role="assistant",
                    content=internal_response.content or "",
                    tool_calls=internal_response.tool_calls,
                )
                session.messages.append(msg)
                await deps.session_manager.update_session(session.session_id, session.messages)

            # Record cache telemetry from provider usage
            cached_tokens = _extract_cached_tokens(internal_response.usage)
            if cached_tokens > 0:
                session.record_cache_hit(cached_tokens)

            # Feed actual cache result back into request metadata for observability
            cache_plan = ctx.session_state.get("cache_plan")
            if cache_plan is not None:
                internal_response.metadata["_cache_arbitrage_actual"] = {
                    "expected_cached_tokens": cache_plan.expected_cached_tokens,
                    "actual_cached_tokens": cached_tokens,
                    "breakpoints": cache_plan.breakpoints,
                    "provider": provider_name,
                }

            # Store non-streaming response in semantic cache
            if cache_key and deps.semantic_cache:
                await deps.semantic_cache.set(
                    cache_key,
                    assemble_cached_response(
                        model=model_used,
                        content=internal_response.content or "",
                        tool_calls=internal_response.tool_calls,
                        usage=internal_response.usage or {},
                        finish_reason=internal_response.finish_reason or "stop",
                    ),
                    compressed_request,
                )

        except deps.provider_timeout_error:
            return JSONResponse(
                {"error": "provider_timeout"},
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            )
        except deps.provider_error as exc:
            return JSONResponse(
                {"error": "provider_error", "message": str(exc)},
                status_code=getattr(exc, "status_code", None) or status.HTTP_502_BAD_GATEWAY,
            )
        except Exception as exc:
            deps.logger.error("provider_unexpected_error", error=str(exc))
            return JSONResponse(
                {"error": "provider_error", "message": str(exc)},
                status_code=status.HTTP_502_BAD_GATEWAY,
            )

        if not x_lattice_disable_transforms:
            internal_response = await deps.pipeline.reverse(internal_response, ctx)

        elapsed_ms = (time.perf_counter() - start_llm) * 1000
        deps.metrics.record_latency("lattice_llm_latency_ms", elapsed_ms)
        response_body = deps.serialize_openai_response(internal_response, compressed_request)
        response = JSONResponse(content=response_body)
        cached_tokens = session.metadata.get("last_cache_hit_tokens", 0)
        # Compute actual cost
        cost_usd = 0.0
        cache_savings_usd = 0.0
        if deps.cost_estimator:
            cost_estimate = deps.cost_estimator.compute_actual(
                provider=provider_name,
                model=model_used,
                usage=internal_response.usage or {},
            )
            cost_usd = cost_estimate.total_cost_usd
            cache_savings_usd = cost_estimate.cached_savings_usd
            deps.metrics.record_metric("cost_estimator", "request_cost_usd", cost_usd)
            deps.metrics.record_metric(
                "cost_estimator",
                "cache_savings_usd",
                cache_savings_usd,
            )

        # Record per-agent stats
        if deps.agent_stats:
            agent_name = identify_agent(
                request.extra_headers.get("user-agent"),
                x_lattice_client_profile,
            )
            normalized_usage = normalize_usage(internal_response.usage or {})
            await deps.agent_stats.record_request(
                agent=agent_name,
                provider=provider_name,
                model=model_used,
                prompt_tokens=normalized_usage["prompt_tokens"],
                completion_tokens=normalized_usage["completion_tokens"],
                compressed_tokens=compressed_request.token_estimate or 0,
                original_tokens=request.token_estimate or 0,
                cached_tokens=cached_tokens,
                cache_hit=cached_tokens > 0,
                cost_usd=cost_usd,
                speculative_hit=used_speculative and prediction_hit,
                batched=batching_eligible,
                auto_continuation_turns=cont_result.turns if cont_result else 0,
            )

        transport_outcome = TransportOutcome(
            semantic_cache_status="miss" if cache_key is not None else "",
            batching_status="batched" if batching_eligible else "",
            speculative_status="hit" if (used_speculative and prediction_hit) else ("miss" if used_speculative else ""),
            delta_mode=delta_mode,
            http_version=http_version,
        )
        for k, v in deps.build_routing_headers(
            model_used,
            session_id=session.session_id,
            compressed_tokens=compressed_request.token_estimate,
            original_tokens=request.token_estimate,
            used_speculative=used_speculative,
            prediction_hit=prediction_hit,
            batched=batching_eligible,
            delta_savings_bytes=delta_savings_bytes,
            cache_hit=cached_tokens > 0,
            cached_tokens=cached_tokens,
            cost_usd=cost_usd,
            cache_savings_usd=cache_savings_usd,
            transport_outcome=transport_outcome,
            **_runtime_header_values(compressed_request),
        ).items():
            response.headers[k] = v
        return response

    return _handle_chat_completion


@dataclasses.dataclass(slots=True)
class AnthropicCompatDeps:
    """Dependencies for Anthropic messages passthrough handler."""

    anthropic_passthrough: Callable[..., Awaitable[Any]]
    provider: Any


def make_anthropic_handler(deps: AnthropicCompatDeps) -> Handler:
    """Create Anthropic messages passthrough handler."""

    async def _handle_anthropic_message(
        fastapi_request: Any,
        x_lattice_session_id: str | None = None,
    ) -> Any:
        raw_body = await fastapi_request.body()
        return await deps.anthropic_passthrough(
            "POST",
            "/v1/messages",
            raw_body,
            fastapi_request,
            deps.provider,
            session_id=x_lattice_session_id or "",
        )

    return _handle_anthropic_message


@dataclasses.dataclass(slots=True)
class ResponsesCompatDeps:
    """Dependencies for OpenAI Responses passthrough handlers."""

    responses_passthrough: Callable[..., Awaitable[Any]]
    provider: Any


def make_models_handler(deps: ResponsesCompatDeps) -> Handler:
    """Create /v1/models passthrough handler."""

    async def _handle_models(
        request: Any,
        x_lattice_session_id: str | None = None,
    ) -> Any:
        return await deps.responses_passthrough(
            "GET", "/v1/models", b"", request, deps.provider,
            session_id=x_lattice_session_id or "",
        )

    return _handle_models


def make_responses_handler(deps: ResponsesCompatDeps) -> Handler:
    """Create /v1/responses* passthrough handler."""

    async def _handle_responses(
        method: str,
        request: Any,
        response_id: str | None = None,
        x_lattice_session_id: str | None = None,
    ) -> Any:
        path = "/v1/responses" if response_id is None else f"/v1/responses/{response_id}"
        body = await request.body() if method == "POST" else b""
        return await deps.responses_passthrough(
            method, path, body, request, deps.provider,
            session_id=x_lattice_session_id or "",
        )

    return _handle_responses


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
        response.headers["x-lattice-version"] = deps.version
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

    @app.get("/healthz", tags=["health"])
    async def healthz() -> dict[str, str]:
        return {
            "status": "healthy",
            "version": deps.version,
            "provider": "direct_http",
            "adapters": ", ".join(deps.provider.registry.list_adapters()),
        }

    @app.get("/readyz", tags=["health"])
    async def readyz() -> dict[str, Any]:
        live, detail = deps.provider.health_check()
        return {
            "status": "ready" if live else "not_ready",
            "checks": {
                "config": True,
                "pipeline": len(deps.pipeline.transforms) > 0,
                "provider": live,
                "provider_detail": detail,
                "http2_pools": deps.provider.pool.pool_count,
                "sessions": deps.store.session_count
                if hasattr(deps.store, "session_count")
                else 0,
            },
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def _metrics() -> str:
        return str(deps.metrics.prometheus_output())

    @app.get("/stats")
    async def _stats() -> dict[str, Any]:
        from lattice.core.delta_wire import DeltaWireDecoder

        capability_registry = get_capability_registry()
        cache_stats: dict[str, Any] = {}
        if deps.semantic_cache is not None:
            cache_stats = await deps.semantic_cache.stats

        result: dict[str, Any] = {
            "version": deps.version,
            "transforms": [t.name for t in deps.pipeline.transforms],
            "pipeline": pipeline_summary(deps.pipeline),
            "sessions": deps.store.session_count
            if hasattr(deps.store, "session_count")
            else 0,
            "provider": "direct_http",
            "adapters": deps.provider.registry.list_adapters(),
            "capabilities": {
                provider: {
                    "cache_mode": capability_registry.cache_mode(provider).value,
                    "supports_prompt_caching": capability_registry.supports(
                        provider, Capability.PROMPT_CACHING
                    ),
                    "default_base_url": capability_registry.get(provider).default_base_url
                    if capability_registry.get(provider)
                    else "",
                }
                for provider in capability_registry.list_providers()
            },
            "pools": deps.provider.pool.pool_count,
            "batching": await deps.batching_engine.stats(),
            "speculation": deps.speculative_executor.stats,
            "tacc": deps.provider.tacc.all_stats()
            if hasattr(deps.provider, "tacc")
            else {},
        }
        manifest_stats: dict[str, Any] = {
            "sessions_with_manifest": 0,
            "anchor_version_max": 0,
            "token_estimate_total": 0,
            "segment_counts": {},
        }
        if hasattr(deps.store, "keys"):
            with contextlib.suppress(Exception):
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
            # Transport outcome rollup derived from canonical telemetry
            result["transport_outcome_rollup"] = {
                k: v for k, v in deps.downgrade_telemetry._counts.items()
                if k in (
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
