"""Tests for lattice.providers.tool_sanitizer."""

from __future__ import annotations

import pytest

from lattice.providers.tool_sanitizer import (
    AnthropicToolSanitizer,
    BedrockToolSanitizer,
    IdentitySanitizer,
    ToolSanitizer,
    get_sanitizer,
    restore_tool_call_ids,
    sanitize_tool_ids,
)

# =============================================================================
# AnthropicToolSanitizer
# =============================================================================


class TestAnthropicToolSanitizer:
    """Every tool ID pattern that FreeRouter's e2e tests validate."""

    def setup_method(self) -> None:
        self.s = AnthropicToolSanitizer()

    @pytest.mark.parametrize(
        "raw,expected",
        [
            # Already valid -- pass through
            ("call_123", "call_123"),
            ("get_weather", "get_weather"),
            ("calc-v2", "calc-v2"),
            # Colons → underscore
            ("call:with:colons", "call_with_colons"),
            # Dots → underscore
            ("call.with.dots", "call_with_dots"),
            # Slashes → underscore
            ("call/with/slashes", "call_with_slashes"),
            # @ symbols → underscore
            ("call@with@at", "call_with_at"),
            # Spaces → underscore
            ("call with spaces", "call_with_spaces"),
            # Hash → underscore
            ("call#with#hash", "call_with_hash"),
            # Mixed
            ("tool.name:v2/1", "tool_name_v2_1"),
        ],
    )
    def test_sanitize(self, raw: str, expected: str) -> None:
        result = self.s.sanitize(raw)
        assert result == expected, f"sanitize({raw!r}) = {result!r}, expected {expected!r}"
        # Result must match Anthropic pattern
        assert self.s.needs_sanitization(result) is False

    def test_sanitize_empty_fallback(self) -> None:
        """When every character is illegal we must not return ''."""
        result = self.s.sanitize("@#$")
        assert result.startswith("tool_")
        assert len(result) > 5

    def test_needs_sanitization(self) -> None:
        assert self.s.needs_sanitization("call_123") is False
        assert self.s.needs_sanitization("call:with:colons") is True

    def test_fast_path_identical(self) -> None:
        """Already-valid IDs must pass through without mutation."""
        raw = "get_weather_v2"
        assert self.s.sanitize(raw) is raw  # identity, we can check exact obj


# =============================================================================
# BedrockToolSanitizer
# =============================================================================


class TestBedrockToolSanitizer:
    def setup_method(self) -> None:
        self.s = BedrockToolSanitizer()

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("get_weather", "get_weather"),
            ("call.with.dots", "call_with_dots"),
        ],
    )
    def test_sanitize(self, raw: str, expected: str) -> None:
        assert self.s.sanitize(raw) == expected


# =============================================================================
# IdentitySanitizer
# =============================================================================


class TestIdentitySanitizer:
    def setup_method(self) -> None:
        self.s = IdentitySanitizer()

    def test_never_changes(self) -> None:
        assert self.s.sanitize("anything@#$goes") == "anything@#$goes"
        assert self.s.needs_sanitization("whatever") is False


# =============================================================================
# ToolSanitizer.register / unsanitize
# =============================================================================


class TestSanitizerMapping:
    def test_register_skips_identical(self) -> None:
        mapping: dict[str, str] = {}
        ToolSanitizer.register(mapping, "same", "same")
        assert "same" not in mapping

    def test_register_records_translation(self) -> None:
        mapping: dict[str, str] = {}
        ToolSanitizer.register(mapping, "raw", "sanitised")
        assert mapping == {"raw": "sanitised"}

    def test_unsanitize_happy(self) -> None:
        mapping = {"original": "clean"}
        assert ToolSanitizer.unsanitize("clean", mapping) == "original"

    def test_unsanitize_missing(self) -> None:
        assert ToolSanitizer.unsanitize("unknown", {}) is None

    def test_unsanitize_no_sanitization(self) -> None:
        """If id was never sanitised (not in mapping), return None."""
        mapping = {"a": "b"}
        assert ToolSanitizer.unsanitize("untouched", mapping) is None


# =============================================================================
# sanitize_tool_ids / restore helpers
# =============================================================================


class TestSanitizeToolIds:
    def test_no_tools(self) -> None:
        mapping: dict[str, str] = {}
        assert sanitize_tool_ids([], mapping) == []

    def test_valid_tools_unchanged(self) -> None:
        mapping: dict[str, str] = {}
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        result = sanitize_tool_ids(tools, mapping)
        assert len(mapping) == 0  # no translation recorded
        assert result[0]["function"]["name"] == "get_weather"

    def test_invalid_tools_sanitized(self) -> None:
        mapping: dict[str, str] = {}
        tools = [{"type": "function", "function": {"name": "call:with:colons"}}]
        result = sanitize_tool_ids(tools, mapping, provider="anthropic")
        assert result[0]["function"]["name"] == "call_with_colons"
        assert mapping == {"call:with:colons": "call_with_colons"}

    def test_preserves_description(self) -> None:
        mapping: dict[str, str] = {}
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "bad.name",
                    "description": "A bad name",
                    "parameters": {"type": "object"},
                },
            }
        ]
        result = sanitize_tool_ids(tools, mapping)
        fn = result[0]["function"]
        assert fn["name"] == "bad_name"
        assert fn["description"] == "A bad name"
        assert fn["parameters"] == {"type": "object"}


class TestRestoreToolCallIds:
    def test_restore_tool_calls(self) -> None:
        mapping = {"raw_name": "clean_name", "raw_id": "clean_id"}
        tool_calls = [
            {
                "id": "clean_id",
                "type": "function",
                "function": {"name": "clean_name", "arguments": "{}"},
            }
        ]
        result = restore_tool_call_ids(tool_calls, mapping)
        assert result[0]["id"] == "raw_id"
        assert result[0]["function"]["name"] == "raw_name"

    def test_restore_tool_calls_unknown(self) -> None:
        """Unknown IDs stay unchanged."""
        result = restore_tool_call_ids(
            [{"id": "x", "type": "function", "function": {"name": "y"}}],
            {},
        )
        assert result[0]["id"] == "x"
        assert result[0]["function"]["name"] == "y"


class TestGetSanitizer:
    def test_known(self) -> None:
        assert get_sanitizer("anthropic").name == "anthropic"

    def test_unknown_falls_back(self) -> None:
        assert get_sanitizer("openrouter").name == "identity"
