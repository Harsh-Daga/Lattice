"""Semantic Segmenter — break prompts into typed segments for optimizer routing.

Phase 2 — segmentation-driven optimization.

Segments a request into semantic categories. The representation_optimizer
uses segment types to pick the best optimizer combination per segment type.
"""

from __future__ import annotations

import dataclasses
import enum
import re
from typing import Any

from lattice.core.transport import Request


class SegmentKind(enum.Enum):
    """Semantic classification of a message region."""

    INSTRUCTIONS = "instructions"       # System prompts, user commands
    REASONING = "reasoning"           # Chain-of-thought, causal analysis
    CODE = "code"                    # Code blocks, diffs
    JSON = "json"                    # JSON blobs, structured data
    TABLE = "table"                  # Markdown tables, CSV
    LOG = "log"                      # Timestamped logs, stack traces
    TOOL_OUTPUT = "tool_output"      # Tool/function results
    NARRATIVE = "narrative"          # Long-form text
    SHORT = "short"                  # Minimal content (< 50 tokens)


@dataclasses.dataclass(slots=True)
class SemanticSegment:
    """A slice of a message with a semantic type."""

    kind: SegmentKind
    text: str
    start_char: int
    end_char: int
    token_estimate: int
    message_index: int

    def copy(self) -> "SemanticSegment":
        return SemanticSegment(
            kind=self.kind,
            text=self.text,
            start_char=self.start_char,
            end_char=self.end_char,
            token_estimate=self.token_estimate,
            message_index=self.message_index,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "token_estimate": self.token_estimate,
            "message_index": self.message_index,
        }


def segment_request(request: Request) -> list[SemanticSegment]:
    """Segment a request into semantic regions.

    Algorithm:
      1. Split each message by structural boundaries (code fences, JSON blocks)
      2. Classify each region using regex heuristics
      3. Merge adjacent regions of the same type within a message
      4. Return ordered list of segments with metadata
    """
    segments: list[SemanticSegment] = []

    for i, msg in enumerate(request.messages):
        text = msg.content or ""
        if not text:
            continue

        msg_segments = _segment_text(text, i)
        # Merge adjacent segments of same kind
        merged = _merge_adjacent(msg_segments)
        segments.extend(merged)

    return segments


def _segment_text(text: str, message_index: int) -> list[SemanticSegment]:
    """Split a single message text into raw segments by structure."""
    segments: list[SemanticSegment] = []
    pos = 0

    # Primary split: code blocks (preserving delimiters)
    parts = re.split(r"(```[\w]*\n.*?```)", text, flags=re.DOTALL)
    for part in parts:
        if not part:
            pos += len(part)
            continue
        if part.startswith("```"):
            kind = _classify_code_block(part)
            segments.append(
                SemanticSegment(
                    kind=kind,
                    text=part,
                    start_char=pos,
                    end_char=pos + len(part),
                    token_estimate=len(part.split()),
                    message_index=message_index,
                )
            )
        else:
            # Split non-code by JSON and table boundaries
            sub_segments = _segment_non_code(part, pos, message_index)
            segments.extend(sub_segments)
        pos += len(part)

    # Deduplicate by adjusting positions
    return segments


def _segment_non_code(text: str, offset: int, message_index: int) -> list[SemanticSegment]:
    """Split non-code text into JSON blocks, tables, and narrative."""
    segments: list[SemanticSegment] = []
    pos = offset

    # Split by JSON blocks first (greedy)
    json_pattern = re.compile(r"(\{[\s\S]*?\}|\[[\s\S]*?\])")
    parts = json_pattern.split(text)

    for part in parts:
        if not part:
            pos += len(part)
            continue
        stripped = part.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            segments.append(
                SemanticSegment(
                    kind=SegmentKind.JSON,
                    text=part,
                    start_char=pos,
                    end_char=pos + len(part),
                    token_estimate=len(part.split()),
                    message_index=message_index,
                )
            )
        else:
            # Split by table rows
            table_pattern = re.compile(r"(^(\|[^\n]*\|)\s*$)", re.MULTILINE)
            table_parts = table_pattern.split(part)
            for tp in table_parts:
                if not tp:
                    pos += len(tp)
                    continue
                if "|" in tp:
                    segments.append(
                        SemanticSegment(
                            kind=SegmentKind.TABLE,
                            text=tp,
                            start_char=pos,
                            end_char=pos + len(tp),
                            token_estimate=len(tp.split()),
                            message_index=message_index,
                        )
                    )
                else:
                    # Remaining: classify as narrative, log, or reasoning
                    kind = _classify_narrative(tp)
                    segments.append(
                        SemanticSegment(
                            kind=kind,
                            text=tp,
                            start_char=pos,
                            end_char=pos + len(tp),
                            token_estimate=len(tp.split()),
                            message_index=message_index,
                        )
                    )
                pos += len(tp)
        pos += len(part)

    return segments


def _classify_code_block(text: str) -> SegmentKind:
    """Classify a ```-delimited block."""
    # Extract content after opening fence
    first_newline = text.find("\n")
    if first_newline < 0:
        return SegmentKind.CODE
    content = text[first_newline + 1 : -3]  # strip ```\n and ```

    # JSON inside code block
    stripped = content.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return SegmentKind.JSON

    # Table inside code block
    if "|" in content.split("\n")[0]:
        return SegmentKind.TABLE

    # Log / stack trace patterns
    if re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}|\b(ERROR|WARN|INFO|DEBUG|FATAL)\b", content):
        return SegmentKind.LOG

    return SegmentKind.CODE


def _classify_narrative(text: str) -> SegmentKind:
    """Classify narrative text."""
    # Tool output
    if re.search(r'"tool_call_id"|"function"\s*:|"is_error"|"type"\s*:\s*"tool"', text):
        return SegmentKind.TOOL_OUTPUT

    # Log patterns
    if re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}|\b(ERROR|WARN|INFO|DEBUG|FATAL)\b", text):
        return SegmentKind.LOG

    # Reasoning markers
    if re.search(r"\b(therefore|thus|hence|consequently|because|as a result|root cause|determined that)\b", text, re.IGNORECASE):
        return SegmentKind.REASONING

    # Too short
    if len(text.split()) < 20:
        return SegmentKind.SHORT

    return SegmentKind.NARRATIVE


def _merge_adjacent(segments: list[SemanticSegment]) -> list[SemanticSegment]:
    """Merge adjacent segments of the same kind within a message."""
    if not segments:
        return []

    merged: list[SemanticSegment] = []
    current = segments[0].copy()

    for seg in segments[1:]:
        if seg.kind == current.kind and seg.message_index == current.message_index:
            # Merge
            current.text = current.text + seg.text
            current.end_char = seg.end_char
            current.token_estimate += seg.token_estimate
        else:
            merged.append(current)
            current = seg.copy()

    merged.append(current)
    return merged


def segment_summary(segments: list[SemanticSegment]) -> dict[str, Any]:
    """Summarize segments for context/storage."""
    counts: dict[str, int] = {}
    tokens: dict[str, int] = {}
    for seg in segments:
        k = seg.kind.value
        counts[k] = counts.get(k, 0) + 1
        tokens[k] = tokens.get(k, 0) + seg.token_estimate

    return {
        "segment_count": len(segments),
        "segment_types": sorted(set(s.kind.value for s in segments)),
        "segment_counts": counts,
        "segment_tokens": tokens,
        "total_tokens": sum(seg.token_estimate for seg in segments),
    }
