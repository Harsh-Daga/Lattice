"""Canonical segment types for LATTICE manifests.

A conversation is decomposed into typed, hash-addressed segments:
- system: System prompt block
- tools: Tool schema definitions
- docs: Static project/documentation context
- messages: Rolling conversation suffix
- artifacts: Verified outputs (tool results, summaries)

Each segment has a content hash and a version. Segments are immutable;
updates create new segment versions. This enables deterministic reuse,
partial invalidation, and cross-session deduplication.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import time
from typing import Any

from lattice.protocol.content import (
    ContentPart,
    content_parts_hash,
    parts_from_dict_list,
    parts_to_dict_list,
)

# =============================================================================
# Segment type enum
# =============================================================================


class SegmentType(str, enum.Enum):
    """Classification of a canonical segment."""

    SYSTEM = "system"
    TOOLS = "tools"
    DOCS = "docs"
    MESSAGES = "messages"
    ARTIFACTS = "artifacts"


# =============================================================================
# Segment
# =============================================================================


@dataclasses.dataclass(frozen=True, slots=True)
class Segment:
    """A typed, hash-addressed chunk of prompt context.

    Attributes:
        type: Segment classification.
        version: Monotonic version counter (0 = initial).
        hash: SHA-256 content hash (truncated to 16 chars).
        parts: Content parts that make up this segment.
        metadata: Provider-specific annotations (e.g. cache_control).
        created_at: Unix timestamp.
    """

    type: SegmentType
    version: int
    hash: str
    parts: list[ContentPart]
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    created_at: float = dataclasses.field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "version": self.version,
            "hash": self.hash,
            "parts": parts_to_dict_list(self.parts),
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Segment:
        return cls(
            type=SegmentType(data.get("type", "messages")),
            version=data.get("version", 0),
            hash=data.get("hash", ""),
            parts=parts_from_dict_list(data.get("parts", [])),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", 0.0),
        )

    @property
    def token_estimate(self) -> int:
        """Rough token estimate for this segment."""
        text_len = sum(len(getattr(p, "text", getattr(p, "content", ""))) for p in self.parts)
        return max(1, text_len // 4)


# =============================================================================
# Segment builder
# =============================================================================


def build_segment(
    segment_type: SegmentType,
    parts: list[ContentPart],
    version: int = 0,
    metadata: dict[str, Any] | None = None,
) -> Segment:
    """Build a segment with computed hash.

    Args:
        segment_type: Classification of the segment.
        parts: Content parts.
        version: Version counter.
        metadata: Optional provider annotations.

    Returns:
        A Segment with deterministic hash.
    """
    h = content_parts_hash(parts)
    if not h:
        # Empty parts still need a stable hash
        h = hashlib.sha256(f"{segment_type.value}:{version}:empty".encode()).hexdigest()[:16]
    return Segment(
        type=segment_type,
        version=version,
        hash=h,
        parts=parts,
        metadata=metadata or {},
    )


def build_system_segment(text: str, metadata: dict[str, Any] | None = None) -> Segment:
    """Convenience builder for system segments."""
    from lattice.protocol.content import TextPart

    return build_segment(SegmentType.SYSTEM, [TextPart(text=text)], metadata=metadata)


def build_tools_segment(
    tools: list[dict[str, Any]], metadata: dict[str, Any] | None = None
) -> Segment:
    """Convenience builder for tool schema segments."""
    from lattice.protocol.content import TextPart

    # Tools are stored as JSON text for stable hashing
    tools_json = json.dumps(tools, sort_keys=True, ensure_ascii=False)
    return build_segment(SegmentType.TOOLS, [TextPart(text=tools_json)], metadata=metadata)


def build_messages_segment(messages: list[ContentPart], version: int = 0) -> Segment:
    """Convenience builder for message segments."""
    return build_segment(SegmentType.MESSAGES, messages, version=version)
