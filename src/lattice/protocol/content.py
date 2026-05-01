"""Typed content parts for LATTICE messages.

Replaces the str-centric Message.content with a union of content part types
that can represent text, images, tool calls, tool results, and reasoning.

This is the foundation for multimodal support, tool block canonicalization,
and provider-aware prefix packing.

Design Principles
-----------------
- **Backward compatible**: Message.content still works as a string fast-path.
- **Provider-agnostic**: ContentPart types map cleanly to OpenAI, Anthropic,
  and Ollama native formats.
- **Hashable**: Each part can be deterministically hashed for segment
  deduplication and manifest stability.
- **Immutable**: Frozen dataclasses to prevent accidental mutation.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from typing import Any

# =============================================================================
# Image source types
# =============================================================================

class ImageSourceType(str, enum.Enum):
    """How image data is encoded."""

    URL = "url"
    BASE64 = "base64"


@dataclasses.dataclass(frozen=True, slots=True)
class ImageSource:
    """Image data reference."""

    type: ImageSourceType
    data: str
    media_type: str = "image/jpeg"

    def to_dict(self) -> dict[str, Any]:
        if self.type == ImageSourceType.URL:
            return {"url": self.data}
        return {
            "type": "base64",
            "media_type": self.media_type,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageSource:
        if "url" in data:
            return cls(type=ImageSourceType.URL, data=data["url"], media_type=data.get("media_type", "image/jpeg"))
        return cls(
            type=ImageSourceType.BASE64,
            data=data.get("data", ""),
            media_type=data.get("media_type", "image/jpeg"),
        )


# =============================================================================
# ContentPart union
# =============================================================================

@dataclasses.dataclass(frozen=True, slots=True)
class TextPart:
    """Plain text content."""

    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextPart:
        return cls(text=data.get("text", ""))


@dataclasses.dataclass(frozen=True, slots=True)
class ImagePart:
    """Image content (multimodal)."""

    source: ImageSource
    detail: str = "auto"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "image", "source": self.source.to_dict()}
        if self.detail != "auto":
            d["detail"] = self.detail
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImagePart:
        return cls(
            source=ImageSource.from_dict(data.get("source", {})),
            detail=data.get("detail", "auto"),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ToolCallPart:
    """An assistant-generated tool call."""

    id: str
    name: str
    arguments: str  # JSON string

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_call",
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCallPart:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            arguments=data.get("arguments", ""),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ToolResultPart:
    """Result of a tool execution."""

    tool_call_id: str
    content: str
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
            "is_error": self.is_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolResultPart:
        return cls(
            tool_call_id=data.get("tool_call_id", ""),
            content=data.get("content", ""),
            is_error=data.get("is_error", False),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ReasoningPart:
    """Explicit reasoning / thinking content (e.g. Claude thinking blocks)."""

    text: str
    signature: str | None = None  # Anthropic thinking signature

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "reasoning", "text": self.text}
        if self.signature:
            d["signature"] = self.signature
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReasoningPart:
        return cls(text=data.get("text", ""), signature=data.get("signature"))


ContentPart = TextPart | ImagePart | ToolCallPart | ToolResultPart | ReasoningPart


# =============================================================================
# Content helpers
# =============================================================================

def content_to_parts(content: str | list[ContentPart] | list[dict[str, Any]] | None) -> list[ContentPart]:
    """Normalize content to a list of ContentPart.

    Args:
        content: Raw content (str, list of parts, list of dicts, or None).

    Returns:
        List of ContentPart objects.
    """
    if content is None:
        return []
    if isinstance(content, str):
        if not content:
            return []
        return [TextPart(text=content)]
    if content and isinstance(content[0], dict):
        return parts_from_dict_list(content)  # type: ignore[arg-type]
    return list(content)  # type: ignore[arg-type]


def parts_to_str(parts: list[ContentPart]) -> str:
    """Convert parts to a plain string (text parts concatenated)."""
    texts: list[str] = []
    for part in parts:
        if isinstance(part, TextPart):
            texts.append(part.text)
        elif isinstance(part, ToolResultPart):
            texts.append(part.content)
        elif isinstance(part, ReasoningPart):
            texts.append(part.text)
    return "\n".join(texts)


def parts_to_dict_list(parts: list[ContentPart]) -> list[dict[str, Any]]:
    """Serialize parts to a list of dicts."""
    return [part.to_dict() for part in parts]


def parts_from_dict_list(data: list[dict[str, Any]]) -> list[ContentPart]:
    """Deserialize a list of dicts to ContentPart objects."""
    parts: list[ContentPart] = []
    for item in data:
        part_type = item.get("type", "text")
        if part_type == "text":
            parts.append(TextPart.from_dict(item))
        elif part_type == "image":
            parts.append(ImagePart.from_dict(item))
        elif part_type == "tool_call":
            parts.append(ToolCallPart.from_dict(item))
        elif part_type == "tool_result":
            parts.append(ToolResultPart.from_dict(item))
        elif part_type == "reasoning":
            parts.append(ReasoningPart.from_dict(item))
        else:
            # Fallback: treat unknown as text
            parts.append(TextPart(text=json.dumps(item)))
    return parts


def content_part_hash(part: ContentPart) -> str:
    """Deterministic hash of a content part for segment deduplication."""
    data = json.dumps(part.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def content_parts_hash(parts: list[ContentPart]) -> str:
    """Deterministic hash of a list of content parts."""
    if not parts:
        return ""
    hashes = [content_part_hash(p) for p in parts]
    combined = "|".join(hashes)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]
