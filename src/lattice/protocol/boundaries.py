"""Semantic boundary detection for LATTICE token streams.

Aligns protocol frames with meaningful content boundaries such as sentence
endings, tool call markers, and code block boundaries.
"""

from __future__ import annotations

import enum


class BoundaryType(enum.Enum):
    """Categories of semantic boundaries."""

    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    SENTENCE_END = "sentence_end"
    CODE_BLOCK = "code_block"
    CONTINUATION = "continuation"


class SemanticBoundaryDetector:
    """Detect semantic boundaries in token / text streams.

    Uses simple heuristics over plain text chunks to classify boundaries.
    """

    SENTENCE_ENDINGS = {'.', '!', '?', '\n'}
    TOOL_CALL_START_MARKER = '<|tool_call|>'
    TOOL_CALL_END_MARKER = '<|/tool_call|>'

    @classmethod
    def classify_chunk(cls, text: str) -> BoundaryType:
        """Determine what kind of boundary ``text`` represents.

        Checks are performed in the following order:
        1. Tool call start / end markers
        2. Sentence-ending punctuation or newline
        3. Code fence start (`` ``` ``)
        4. Fallback to CONTINUATION
        """
        if cls.TOOL_CALL_START_MARKER in text:
            return BoundaryType.TOOL_START
        if cls.TOOL_CALL_END_MARKER in text:
            return BoundaryType.TOOL_END
        if any(c in text for c in cls.SENTENCE_ENDINGS):
            return BoundaryType.SENTENCE_END
        if text.strip().startswith('```'):
            return BoundaryType.CODE_BLOCK
        return BoundaryType.CONTINUATION
