from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from lattice.core.context import TransformContext
from lattice.core.result import is_err, unwrap
from lattice.transport.types import Message, Request

Handler = Callable[..., Awaitable[Any]]

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
        return (
            body,
            TransformContext(request_id=str(time.time()), provider="openai", model=model),
            0,
            0,
        )

    pseudo_messages = [
        Message(role="user" if i % 2 == 0 else "assistant", content=text)
        for i, (_block, text) in enumerate(blocks)
    ]
    pseudo_request = Request(messages=pseudo_messages, model=model)
    original_tokens = pseudo_request.token_estimate

    ctx = TransformContext(request_id=str(time.time()), provider="openai", model=model)
    result = pipeline.compress(pseudo_request, ctx)
    if is_err(result):
        if config.graceful_degradation:
            logger.warning("responses_content_compression_degraded", error=str(result))
            return body, ctx, original_tokens, original_tokens
        raise Exception(f"Responses content compression failed: {result}")

    compressed_request = unwrap(result)
    compressed_texts = [m.content for m in compressed_request.messages]
    replace_responses_text_blocks(blocks, compressed_texts)
    return body, ctx, original_tokens, compressed_request.token_estimate


