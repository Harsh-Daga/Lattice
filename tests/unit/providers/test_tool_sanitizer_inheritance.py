"""Tool sanitizer base-class inheritance."""

from __future__ import annotations

from lattice.providers.tool_sanitizer import (
    AnthropicToolSanitizer,
    BedrockToolSanitizer,
    ToolSanitizer,
)


def test_sanitizers_share_base() -> None:
    assert issubclass(AnthropicToolSanitizer, ToolSanitizer)
    assert issubclass(BedrockToolSanitizer, ToolSanitizer)


def test_anthropic_sanitizer_validates() -> None:
    sanitizer = AnthropicToolSanitizer()
    assert sanitizer.validate_tool_id("good_tool-name")
    assert not sanitizer.validate_tool_id("bad tool!")
    safe = sanitizer.sanitize("bad tool!")
    assert sanitizer.validate_tool_id(safe)
    mapping: dict[str, str] = {}
    ToolSanitizer.register(mapping, "bad tool!", safe)
    assert ToolSanitizer.unsanitize(safe, mapping) == "bad tool!"
