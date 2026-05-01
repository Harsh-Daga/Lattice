"""Shared serialization helpers for LATTICE Request / Message / Response.

Ensures consistent conversion between:
- OpenAI-compatible JSON dicts  ↔  internal Message / Request objects
- Preserves multimodal content, tool calls, reasoning, and metadata

This module is the single source of truth for serialization. All SDK,
proxy, and transport code should use these helpers instead of ad-hoc
dict construction.
"""

from __future__ import annotations

from typing import Any

from lattice.core.transport import Message, Request, Response
from lattice.protocol.content import (
    ContentPart,
    ImagePart,
    ImageSource,
    ImageSourceType,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    parts_from_dict_list,
)

# =============================================================================
# Message ↔ dict
# =============================================================================

def message_to_dict(msg: Message) -> dict[str, Any]:
    """Serialize a Message to an OpenAI-compatible dict.

    Preserves:
    - Multimodal content (as array of content blocks)
    - tool_calls and tool_call_id
    - name
    - metadata (as _lattice_metadata to avoid colliding with provider fields)
    """
    d: dict[str, Any] = {"role": str(msg.role)}

    # Content: prefer content_parts if multimodal, else string
    content_parts = getattr(msg, "_content_parts", None)
    if content_parts is not None and len(content_parts) > 1:
        # Multimodal — serialize as array
        d["content"] = _content_parts_to_openai(content_parts)
    elif content_parts is not None and len(content_parts) == 1:
        part = content_parts[0]
        if isinstance(part, TextPart):
            d["content"] = part.text
        elif isinstance(part, ImagePart):
            d["content"] = [{"type": "image_url", "image_url": {"url": part.source.data}}]
        else:
            d["content"] = part.to_dict()
    else:
        d["content"] = msg.content

    if msg.name:
        d["name"] = msg.name
    if msg.tool_calls:
        d["tool_calls"] = msg.tool_calls
    if msg.tool_call_id:
        d["tool_call_id"] = msg.tool_call_id
    if msg.metadata:
        d["_lattice_metadata"] = dict(msg.metadata)

    return d


def message_from_dict(d: dict[str, Any]) -> Message:
    """Deserialize an OpenAI-compatible dict to a Message.

    Handles:
    - String content (standard)
    - Array content (multimodal: text + image_url blocks)
    - content_parts field (internal format)
    - tool_calls and tool_call_id
    """
    raw_content = d.get("content", "")
    content_parts: list[ContentPart] | None = None

    if isinstance(raw_content, list):
        # Multimodal OpenAI format → content_parts
        content_parts = _openai_content_to_parts(raw_content)
        # String content is concatenated text
        text_parts = [p for p in content_parts if isinstance(p, TextPart)]
        text_content = "\n".join(p.text for p in text_parts)
    else:
        text_content = raw_content
        # Also check for explicit content_parts field
        if "content_parts" in d:
            content_parts = parts_from_dict_list(d["content_parts"])

    msg = Message(
        role=d.get("role", "user"),
        content=text_content,
        name=d.get("name"),
        tool_calls=d.get("tool_calls"),
        tool_call_id=d.get("tool_call_id"),
    )
    if content_parts is not None:
        msg.content_parts = content_parts

    # Restore metadata
    meta = d.get("_lattice_metadata")
    if isinstance(meta, dict):
        msg.metadata.update(meta)

    return msg


# =============================================================================
# Request ↔ dict
# =============================================================================

def request_to_dict(req: Request) -> dict[str, Any]:
    """Serialize a Request to an OpenAI-compatible dict."""
    d: dict[str, Any] = {
        "model": req.model,
        "messages": [message_to_dict(m) for m in req.messages],
    }
    if req.temperature is not None:
        d["temperature"] = req.temperature
    if req.max_tokens is not None:
        d["max_tokens"] = req.max_tokens
    if req.top_p is not None:
        d["top_p"] = req.top_p
    if req.tools:
        d["tools"] = req.tools
    if req.tool_choice is not None:
        d["tool_choice"] = req.tool_choice
    if req.stream:
        d["stream"] = True
    if req.stop:
        d["stop"] = req.stop
    if req.metadata:
        d["_lattice_metadata"] = dict(req.metadata)
    if req.extra_headers:
        d["_lattice_extra_headers"] = dict(req.extra_headers)
    if req.extra_body:
        d["_lattice_extra_body"] = dict(req.extra_body)
    return d


def request_from_dict(d: dict[str, Any]) -> Request:
    """Deserialize an OpenAI-compatible dict to a Request."""
    messages = [message_from_dict(m) for m in d.get("messages", [])]
    req = Request(
        messages=messages,
        model=d.get("model", ""),
        temperature=d.get("temperature"),
        max_tokens=d.get("max_tokens"),
        top_p=d.get("top_p"),
        tools=d.get("tools"),
        tool_choice=d.get("tool_choice"),
        stream=d.get("stream", False),
        stop=d.get("stop"),
    )
    # Restore extra_headers (support both _lattice_extra_headers and raw extra_headers)
    extra_headers = d.get("_lattice_extra_headers") or d.get("extra_headers")
    if isinstance(extra_headers, dict):
        req.extra_headers.update(extra_headers)
    # Restore extra_body (support both _lattice_extra_body and raw extra_body)
    extra_body = d.get("_lattice_extra_body") or d.get("extra_body")
    if isinstance(extra_body, dict):
        req.extra_body.update(extra_body)
    # Restore metadata (strip fields that are direct Request attributes)
    reserved = {
        "model", "messages", "temperature", "max_tokens", "top_p",
        "tools", "tool_choice", "stream", "stop",
        "_lattice_metadata", "_lattice_extra_headers", "_lattice_extra_body",
        "extra_headers", "extra_body",
    }
    meta = {k: v for k, v in d.items() if k not in reserved}
    # Also merge explicit _lattice_metadata
    lattice_meta = d.get("_lattice_metadata")
    if isinstance(lattice_meta, dict):
        meta.update(lattice_meta)
    req.metadata.update(meta)
    return req


# =============================================================================
# Response → dict
# =============================================================================

def response_to_dict(resp: Response, request_model: str = "") -> dict[str, Any]:
    """Serialize a Response to an OpenAI-compatible response dict."""
    import time as time_mod
    body: dict[str, Any] = {
        "id": "chatcmpl-lattice",
        "object": "chat.completion",
        "created": int(time_mod.time()),
        "model": resp.model or request_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": resp.role,
                    "content": resp.content or "",
                },
                "finish_reason": resp.finish_reason or "stop",
            }
        ],
    }
    if resp.tool_calls:
        body["choices"][0]["message"]["tool_calls"] = resp.tool_calls
    reasoning = resp.metadata.get("reasoning") or resp.metadata.get("thinking")
    if reasoning:
        body["choices"][0]["message"]["reasoning_content"] = reasoning
    refusal = resp.metadata.get("refusal")
    if refusal:
        body["choices"][0]["message"]["refusal"] = refusal
    body["usage"] = resp.usage or {}
    return body


# =============================================================================
# Internal helpers
# =============================================================================

def _content_parts_to_openai(parts: list[ContentPart]) -> list[dict[str, Any]] | str:
    """Convert internal ContentPart list to OpenAI content array format."""
    out: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, TextPart):
            out.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            if part.source.type == ImageSourceType.URL:
                out.append({"type": "image_url", "image_url": {"url": part.source.data}})
            else:
                out.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{part.source.media_type};base64,{part.source.data}",
                    },
                })
        elif isinstance(part, ToolCallPart):
            out.append({
                "type": "tool_call",
                "id": part.id,
                "function": {"name": part.name, "arguments": part.arguments},
            })
        elif isinstance(part, ToolResultPart):
            out.append({
                "type": "tool_result",
                "tool_call_id": part.tool_call_id,
                "content": part.content,
            })
        else:
            # Fallback: serialize as dict
            out.append(part.to_dict())
    return out


def _openai_content_to_parts(content: list[dict[str, Any]]) -> list[ContentPart]:
    """Convert OpenAI content array to internal ContentPart list."""
    parts: list[ContentPart] = []
    for block in content:
        btype = block.get("type", "text")
        if btype == "text":
            parts.append(TextPart(text=block.get("text", "")))
        elif btype == "image_url":
            url_data = block.get("image_url", {}).get("url", "")
            if url_data.startswith("data:"):
                # Parse data URL: data:<media_type>;base64,<data>
                try:
                    header, b64_data = url_data.split(",", 1)
                    media_type = header.split(";")[0].split(":", 1)[1]
                except (ValueError, IndexError):
                    media_type = "image/jpeg"
                    b64_data = url_data
                parts.append(
                    ImagePart(
                        source=ImageSource(
                            type=ImageSourceType.BASE64, data=b64_data, media_type=media_type
                        )
                    )
                )
            else:
                parts.append(
                    ImagePart(source=ImageSource(type=ImageSourceType.URL, data=url_data))
                )
        elif btype == "tool_call":
            fn = block.get("function", {})
            parts.append(ToolCallPart(
                id=block.get("id", ""),
                name=fn.get("name", ""),
                arguments=fn.get("arguments", ""),
            ))
        elif btype == "tool_result":
            parts.append(ToolResultPart(
                tool_call_id=block.get("tool_call_id", ""),
                content=block.get("content", ""),
            ))
        else:
            # Unknown block type — store as text
            import json
            parts.append(TextPart(text=json.dumps(block)))
    return parts
