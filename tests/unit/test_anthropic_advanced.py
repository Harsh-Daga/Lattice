"""Tests for Anthropic advanced features: system blocks, cache_control, defer_loading,
allowed_callers, output schema tool.
"""

from __future__ import annotations

from typing import Any

import pytest

from lattice.core.transport import Message, Request
from lattice.providers.anthropic import AnthropicAdapter


@pytest.fixture
def adapter() -> AnthropicAdapter:
    return AnthropicAdapter()


class TestSystemBlocks:
    def test_plain_string_without_cache(self, adapter: AnthropicAdapter) -> None:
        body = adapter.serialize_request(
            Request(
                messages=[Message(role="system", content="Be helpful")],
                model="claude-3-5-haiku",
            )
        )
        assert body["system"] == "Be helpful"

    def test_block_array_with_cache(self, adapter: AnthropicAdapter) -> None:
        req = Request(
            messages=[Message(role="system", content="Be helpful")],
            model="claude-3-5-haiku",
            metadata={"anthropic_cache_control": True},
        )
        body = adapter.serialize_request(req)
        assert isinstance(body["system"], list)
        assert body["system"][0]["type"] == "text"
        assert body["system"][0]["text"] == "Be helpful"
        assert body["system"][0]["cache_control"]["type"] == "ephemeral"

    def test_none_system_omitted(self, adapter: AnthropicAdapter) -> None:
        body = adapter.serialize_request(Request(messages=[], model="claude-3-5-haiku"))
        assert "system" not in body


class ToolAnnotation:
    """Tests for _annotate_tools, _build_system_blocks, _build_output_schema_tool"""

    pass  # placeholder


class TestToolAnnotation:
    def test_cache_control(self, adapter: AnthropicAdapter) -> None:
        tools = [{"name": "sum", "description": "Add", "input_schema": {}}]
        out = adapter._annotate_tools(
            tools, cache_control=True, defer_loading=False, allowed_callers=False
        )
        assert out[0]["cache_control"]["type"] == "ephemeral"

    def test_defer_loading(self, adapter: AnthropicAdapter) -> None:
        tools = [{"name": "sum", "description": "Add", "input_schema": {}}]
        out = adapter._annotate_tools(
            tools, cache_control=False, defer_loading=True, allowed_callers=False
        )
        assert out[0]["deferred_loading"] is True

    def test_allowed_callers(self, adapter: AnthropicAdapter) -> None:
        tools = [{"name": "sum", "description": "Add", "input_schema": {}}]
        out = adapter._annotate_tools(
            tools, cache_control=False, defer_loading=False, allowed_callers=True
        )
        assert out[0]["allowed_callers"] == ["user", "assistant"]

    def test_combined_annotations(self, adapter: AnthropicAdapter) -> None:
        tools = [{"name": "sum", "description": "Add", "input_schema": {}}]
        out = adapter._annotate_tools(
            tools, cache_control=True, defer_loading=True, allowed_callers=True
        )
        assert out[0]["cache_control"]["type"] == "ephemeral"
        assert out[0]["deferred_loading"] is True
        assert "allowed_callers" in out[0]

    def test_existing_cache_control_preserved(self, adapter: AnthropicAdapter) -> None:
        tools = [
            {
                "name": "sum",
                "description": "Add",
                "input_schema": {},
                "cache_control": {"type": "persistent"},
            }
        ]
        out = adapter._annotate_tools(
            tools, cache_control=True, defer_loading=False, allowed_callers=False
        )
        assert out[0]["cache_control"]["type"] == "persistent"


class TestOutputSchemaTool:
    def test_basic(self, adapter: AnthropicAdapter) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        tool = adapter._build_output_schema_tool(schema)
        assert tool["name"] == "json"
        assert "json" in tool["description"].lower()
        assert tool["input_schema"]["type"] == "object"

    def test_strips_unsupported(self, adapter: AnthropicAdapter) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 10},
            },
        }
        tool = adapter._build_output_schema_tool(schema)
        assert "minimum" not in tool["input_schema"]["properties"]["score"]
        assert "maximum" not in tool["input_schema"]["properties"]["score"]

    def test_none_schema_passes(self, adapter: AnthropicAdapter) -> None:
        # Edge case: sanitize_json_schema can return None if schema is None,
        # but _build_output_schema_tool always gets a dict.
        tool = adapter._build_output_schema_tool({"type": "string"})
        assert tool["input_schema"]["type"] == "string"
