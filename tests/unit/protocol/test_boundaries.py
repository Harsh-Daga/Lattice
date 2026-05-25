"""Tests for lattice.protocol.boundaries.

Covers:
- BoundaryType enum
- SemanticBoundaryDetector.classify_chunk for each category
"""

from __future__ import annotations

from lattice.protocol.boundaries import BoundaryType, SemanticBoundaryDetector


class TestBoundaryType:
    def test_members(self) -> None:
        assert BoundaryType.TOOL_START.value == "tool_start"
        assert BoundaryType.TOOL_END.value == "tool_end"
        assert BoundaryType.SENTENCE_END.value == "sentence_end"
        assert BoundaryType.CODE_BLOCK.value == "code_block"
        assert BoundaryType.CONTINUATION.value == "continuation"


class TestSemanticBoundaryDetector:
    def test_tool_start(self) -> None:
        text = "Calling tool <|tool_call|> now"
        assert SemanticBoundaryDetector.classify_chunk(text) == BoundaryType.TOOL_START

    def test_tool_start_at_beginning(self) -> None:
        text = "<|tool_call|>"
        assert SemanticBoundaryDetector.classify_chunk(text) == BoundaryType.TOOL_START

    def test_tool_end(self) -> None:
        text = "Done with <|/tool_call|>"
        assert SemanticBoundaryDetector.classify_chunk(text) == BoundaryType.TOOL_END

    def test_tool_end_trumps_sentence_end(self) -> None:
        """If both tool-end and sentence-ending punctuation are present,
        tool-end should win because it is checked first."""
        text = "Finished. <|/tool_call|>"
        assert SemanticBoundaryDetector.classify_chunk(text) == BoundaryType.TOOL_END

    def test_sentence_end_period(self) -> None:
        text = "Hello world."
        assert SemanticBoundaryDetector.classify_chunk(text) == BoundaryType.SENTENCE_END

    def test_sentence_end_exclamation(self) -> None:
        text = "Wow!"
        assert SemanticBoundaryDetector.classify_chunk(text) == BoundaryType.SENTENCE_END

    def test_sentence_end_question(self) -> None:
        text = "Really?"
        assert SemanticBoundaryDetector.classify_chunk(text) == BoundaryType.SENTENCE_END

    def test_sentence_end_newline(self) -> None:
        text = "line one\nline two"
        assert SemanticBoundaryDetector.classify_chunk(text) == BoundaryType.SENTENCE_END

    def test_code_block(self) -> None:
        text = "  ```python"
        assert SemanticBoundaryDetector.classify_chunk(text) == BoundaryType.CODE_BLOCK

    def test_code_block_no_indent(self) -> None:
        text = "```"
        assert SemanticBoundaryDetector.classify_chunk(text) == BoundaryType.CODE_BLOCK

    def test_continuation(self) -> None:
        text = "This is just more text"
        assert SemanticBoundaryDetector.classify_chunk(text) == BoundaryType.CONTINUATION

    def test_empty_string_is_continuation(self) -> None:
        assert SemanticBoundaryDetector.classify_chunk("") == BoundaryType.CONTINUATION

    def test_tool_start_trumps_all(self) -> None:
        text = "<|tool_call|> and ```code```"
        assert SemanticBoundaryDetector.classify_chunk(text) == BoundaryType.TOOL_START
