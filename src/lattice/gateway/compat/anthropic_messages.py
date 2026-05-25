from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

from lattice.core.context import TransformContext
from lattice.core.result import is_err, unwrap
from lattice.transport.types import Message, Request, Response

Handler = Callable[..., Awaitable[Any]]


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
                    text_parts.append(f"[Image: {block.get('source', {}).get('type', 'unknown')}]")
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


def deserialize_anthropic_response(body: dict[str, Any]) -> Response:
    """Convert Anthropic Messages API JSON response into internal Response."""
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] | None = None
    for block in body.get("content", []):
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "text":
            content_parts.append(block.get("text", ""))
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
    content = "\n".join(content_parts) if content_parts else ""

    stop_reason = body.get("stop_reason")
    finish_reason = "stop"
    if stop_reason == "end_turn":
        finish_reason = "stop"
    elif stop_reason == "max_tokens":
        finish_reason = "length"
    elif stop_reason == "tool_use":
        finish_reason = "tool_calls"

    resp = Response(
        content=content,
        tool_calls=tool_calls,
        model=body.get("model", ""),
        usage=body.get("usage", {}),
        finish_reason=finish_reason,
    )
    if body.get("stop_sequence"):
        resp.metadata["stop_sequence"] = body["stop_sequence"]
    if body.get("id"):
        resp.metadata["anthropic_message_id"] = body["id"]
    return resp


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
        return (
            body,
            TransformContext(request_id=str(time.time()), provider=provider_name, model=model),
            0,
            0,
        )

    pseudo_messages = [
        Message(role="user" if i % 2 == 0 else "assistant", content=text)
        for i, (_block, text) in enumerate(blocks)
    ]
    pseudo_request = Request(messages=pseudo_messages, model=model)
    original_tokens = pseudo_request.token_estimate

    ctx = TransformContext(request_id=str(time.time()), provider=provider_name, model=model)
    result = pipeline.compress(pseudo_request, ctx)
    if is_err(result):
        if config.graceful_degradation:
            logger.warning("anthropic_content_compression_degraded", error=str(result))
            return body, ctx, original_tokens, original_tokens
        raise Exception(f"Anthropic content compression failed: {result}")

    compressed_request = unwrap(result)
    compressed_texts = [m.content for m in compressed_request.messages]
    replace_anthropic_text_blocks(blocks, compressed_texts)
    return body, ctx, original_tokens, compressed_request.token_estimate
