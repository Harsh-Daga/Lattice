"""Manifest protocol for LATTICE sessions.

A manifest is a Merkle-like DAG of typed segments that represents the
complete state of a conversation at a point in time.

Key concepts:
- **Manifest**: Ordered list of segments + metadata.
- **Anchor version**: Monotonic counter incremented on every change.
- **Anchor hash**: SHA-256 of the canonical manifest serialization.
- **Delta operations**: APPEND, INVALIDATE, REPLACE for updating manifests.

Provider cache alignment:
- OpenAI rewards stable exact prefixes → system/tools first, then messages.
- Anthropic rewards cache_control at breakpoints → tools → system → messages.
- The manifest canonicalizer orders segments to maximize cache hit rates.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import secrets
import time
from typing import Any

from lattice.protocol.content import ContentPart, TextPart
from lattice.protocol.segments import (
    Segment,
    SegmentType,
    build_messages_segment,
    build_segment,
    build_tools_segment,
)

# =============================================================================
# Manifest
# =============================================================================


@dataclasses.dataclass(slots=True)
class Manifest:
    """Complete prompt state as an ordered list of segments.

    Attributes:
        manifest_id: Unique manifest identifier.
        session_id: Parent session identifier.
        anchor_version: Monotonic version counter.
        anchor_hash: SHA-256 of canonical serialization (32 chars).
        segments: Ordered list of segments.
        created_at: Unix timestamp.
        metadata: Session-level metadata (model, provider, etc.).
    """

    manifest_id: str
    session_id: str
    anchor_version: int
    anchor_hash: str
    segments: list[Segment]
    created_at: float = dataclasses.field(default_factory=time.time)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "session_id": self.session_id,
            "anchor_version": self.anchor_version,
            "anchor_hash": self.anchor_hash,
            "segments": [s.to_dict() for s in self.segments],
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        return cls(
            manifest_id=data["manifest_id"],
            session_id=data["session_id"],
            anchor_version=data.get("anchor_version", 0),
            anchor_hash=data.get("anchor_hash", ""),
            segments=[Segment.from_dict(s) for s in data.get("segments", [])],
            created_at=data.get("created_at", 0.0),
            metadata=data.get("metadata", {}),
        )

    @property
    def token_estimate(self) -> int:
        """Total token estimate across all segments."""
        return sum(s.token_estimate for s in self.segments)

    def get_segment(self, segment_type: SegmentType) -> Segment | None:
        """Return the first segment of the given type."""
        for seg in self.segments:
            if seg.type == segment_type:
                return seg
        return None

    def get_segments(self, segment_type: SegmentType) -> list[Segment]:
        """Return all segments of the given type."""
        return [seg for seg in self.segments if seg.type == segment_type]

    def summary(self) -> dict[str, Any]:
        """Return a compact, human-readable manifest summary."""
        return manifest_summary(self)


# =============================================================================
# Manifest builder / canonicalizer
# =============================================================================


def compute_anchor_hash(segments: list[Segment], metadata: dict[str, Any]) -> str:
    """Compute deterministic anchor hash from segments and metadata.

    The hash covers segment types, versions, and hashes in order.
    It does NOT cover created_at (volatile) or full part content
    (those are covered by per-segment hashes).
    """
    canonical = {
        "segments": [
            {"type": s.type.value, "version": s.version, "hash": s.hash} for s in segments
        ],
        "metadata": _sort_dict(metadata),
    }
    data = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def build_manifest(
    session_id: str,
    segments: list[Segment],
    anchor_version: int = 0,
    metadata: dict[str, Any] | None = None,
    manifest_id: str | None = None,
) -> Manifest:
    """Build a manifest with computed anchor hash.

    Args:
        session_id: Parent session ID.
        segments: Ordered list of segments.
        anchor_version: Version counter.
        metadata: Session-level metadata.
        manifest_id: Optional manifest ID (auto-generated if None).

    Returns:
        Manifest with canonical anchor hash.
    """
    meta = metadata or {}
    canonical_segments = list(segments)
    anchor_hash = compute_anchor_hash(canonical_segments, meta)
    return Manifest(
        manifest_id=manifest_id or f"mf-{secrets.token_hex(8)}",
        session_id=session_id,
        anchor_version=anchor_version,
        anchor_hash=anchor_hash,
        segments=canonical_segments,
        metadata=meta,
    )


def manifest_summary(manifest: Manifest) -> dict[str, Any]:
    """Summarize a manifest for telemetry and user-facing stats."""
    segment_counts: dict[str, int] = {}
    segment_types: list[str] = []
    for seg in manifest.segments:
        segment_types.append(seg.type.value)
        segment_counts[seg.type.value] = segment_counts.get(seg.type.value, 0) + 1
    return {
        "manifest_id": manifest.manifest_id,
        "session_id": manifest.session_id,
        "anchor_version": manifest.anchor_version,
        "anchor_hash": manifest.anchor_hash,
        "token_estimate": manifest.token_estimate,
        "segment_count": len(manifest.segments),
        "segment_types": segment_types,
        "segment_counts": segment_counts,
        "metadata": _sort_dict(manifest.metadata),
    }


def canonicalize_segments(segments: list[Segment]) -> list[Segment]:
    """Reorder segments for optimal provider cache hit rates.

    Order:
    1. TOOLS (stable across turns)
    2. SYSTEM (stable across turns)
    3. DOCS (stable project context)
    4. ARTIFACTS (verified outputs)
    5. MESSAGES (rolling conversation)

    This matches OpenAI's exact-prefix caching and Anthropic's
    tools → system → messages ordering.
    """
    order = {
        SegmentType.TOOLS: 0,
        SegmentType.SYSTEM: 1,
        SegmentType.DOCS: 2,
        SegmentType.ARTIFACTS: 3,
        SegmentType.MESSAGES: 4,
    }
    return sorted(segments, key=lambda s: order.get(s.type, 99))


def manifest_from_messages(
    session_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str = "",
    provider: str = "openai",
) -> Manifest:
    """Build a manifest from a raw OpenAI-style message list.

    Args:
        session_id: Session identifier.
        messages: List of message dicts.
        tools: Optional tool definitions.
        model: Model identifier.
        provider: Provider name.

    Returns:
        Manifest with canonical segments.
    """
    from lattice.protocol.content import content_to_parts

    segments: list[Segment] = []

    # Extract system message
    system_texts: list[str] = []
    conversation_parts: list[ContentPart] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_texts.append(content if isinstance(content, str) else json.dumps(content))
        else:
            parts = content_to_parts(content)
            # Tag parts with role metadata by wrapping in text prefix
            if parts:
                role_prefix = f"[{role.upper()}] "
                if isinstance(parts[0], TextPart):
                    # Prepend role to first text part
                    new_parts = [TextPart(text=role_prefix + parts[0].text)] + parts[1:]
                    conversation_parts.extend(new_parts)
                else:
                    conversation_parts.append(TextPart(text=role_prefix))
                    conversation_parts.extend(parts)

    if system_texts:
        segments.append(
            build_segment(
                SegmentType.SYSTEM,
                [TextPart(text="\n".join(system_texts))],
            )
        )

    if tools:
        segments.append(build_tools_segment(tools))

    if conversation_parts:
        segments.append(build_messages_segment(conversation_parts))

    segments = canonicalize_segments(segments)

    return build_manifest(
        session_id=session_id,
        segments=segments,
        metadata={"model": model, "provider": provider},
    )


def manifest_to_messages(manifest: Manifest) -> list[dict[str, Any]]:
    """Flatten a manifest back to OpenAI-style messages.

    This is the inverse of manifest_from_messages. It reconstructs
    the full message list from segments.
    """
    messages: list[dict[str, Any]] = []

    system_seg = manifest.get_segment(SegmentType.SYSTEM)
    if system_seg and system_seg.parts:
        text = "\n".join(p.text for p in system_seg.parts if isinstance(p, TextPart))
        if text:
            messages.append({"role": "system", "content": text})

    # Messages segment
    msg_seg = manifest.get_segment(SegmentType.MESSAGES)
    if msg_seg and msg_seg.parts:
        current_role = "user"
        current_texts: list[str] = []

        for part in msg_seg.parts:
            if isinstance(part, TextPart) and part.text.startswith("["):
                # Flush previous
                if current_texts:
                    messages.append({"role": current_role, "content": "\n".join(current_texts)})
                    current_texts = []
                # Extract role from prefix like [USER] ...
                end_bracket = part.text.find("] ")
                if end_bracket != -1:
                    current_role = part.text[1:end_bracket].lower()
                    current_texts.append(part.text[end_bracket + 2 :])
                else:
                    current_texts.append(part.text)
            else:
                current_texts.append(str(part))

        if current_texts:
            messages.append({"role": current_role, "content": "\n".join(current_texts)})

    return messages


# =============================================================================
# Delta operations on manifests
# =============================================================================


def apply_delta(
    manifest: Manifest,
    new_segments: list[Segment] | None = None,
    invalidate_hashes: list[str] | None = None,
    replace_messages: list[ContentPart] | None = None,
) -> Manifest:
    """Apply a delta to an existing manifest, producing a new manifest.

    Args:
        manifest: Base manifest.
        new_segments: Segments to append or replace.
        invalidate_hashes: Segment hashes to remove.
        replace_messages: New MESSAGES segment content.

    Returns:
        New manifest with incremented version and recomputed hash.
    """
    segments = list(manifest.segments)

    # Remove invalidated segments
    if invalidate_hashes:
        segments = [s for s in segments if s.hash not in invalidate_hashes]

    # Replace messages segment if provided
    if replace_messages:
        segments = [s for s in segments if s.type != SegmentType.MESSAGES]
        segments.append(
            build_segment(
                SegmentType.MESSAGES, replace_messages, version=manifest.anchor_version + 1
            )
        )

    # Append new segments
    if new_segments:
        for seg in new_segments:
            # If same type exists, replace it
            existing_idx = next(
                (
                    i
                    for i, s in enumerate(segments)
                    if s.type == seg.type and s.type != SegmentType.MESSAGES
                ),
                None,
            )
            if existing_idx is not None:
                segments[existing_idx] = seg
            else:
                segments.append(seg)

    segments = canonicalize_segments(segments)

    return build_manifest(
        session_id=manifest.session_id,
        segments=segments,
        anchor_version=manifest.anchor_version + 1,
        metadata=manifest.metadata,
    )


# =============================================================================
# Internal helpers
# =============================================================================


def _sort_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively sort dict keys for canonical serialization."""
    if isinstance(d, dict):
        return {k: _sort_dict(v) for k, v in sorted(d.items())}
    if isinstance(d, list):
        return [_sort_dict(v) for v in d]
    return d
