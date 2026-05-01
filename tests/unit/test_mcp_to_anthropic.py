"""Tests for lattice.providers.mcp_to_anthropic."""

from __future__ import annotations

import pytest

from lattice.providers.mcp_to_anthropic import (
    MCPServerTool,
    convert_mcp_to_anthropic,
    is_mcp_tool,
)

# =============================================================================
# MCPServerTool dataclass
# =============================================================================


class TestMCPServerTool:
    def test_basic_fields(self) -> None:
        tool = MCPServerTool(name="get_weather", url="https://api.example.com/weather")
        assert tool.name == "get_weather"
        assert tool.url == "https://api.example.com/weather"
        assert tool.description == ""
        assert tool.authorization is None
        assert tool.input_schema is None

    def test_all_fields(self) -> None:
        tool = MCPServerTool(
            name="search",
            description="Web search",
            url="https://search.example.com",
            authorization="Bearer abc123",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        assert tool.authorization == "Bearer abc123"
        assert tool.input_schema == {"type": "object", "properties": {"q": {"type": "string"}}}

    def test_frozen(self) -> None:
        tool = MCPServerTool(name="x", url="https://x.com")
        with pytest.raises(AttributeError):
            tool.name = "y"


# =============================================================================
# is_mcp_tool detection
# =============================================================================


class TestIsMCPTool:
    def test_top_level_url(self) -> None:
        assert is_mcp_tool({"name": "get_weather", "url": "https://api.example.com"})

    def test_function_url(self) -> None:
        assert is_mcp_tool(
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "url": "https://api.example.com",
                },
            }
        )

    def test_function_metadata_url(self) -> None:
        assert is_mcp_tool(
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "metadata": {"url": "https://api.example.com"},
                },
            }
        )

    def test_regular_function_not_mcp(self) -> None:
        assert not is_mcp_tool(
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                },
            }
        )

    def test_non_string_url_ignored(self) -> None:
        assert not is_mcp_tool({"name": "x", "url": 123})
        assert not is_mcp_tool(
            {"type": "function", "function": {"name": "x", "url": None}}
        )
        assert not is_mcp_tool(
            {"type": "function", "function": {"name": "x", "metadata": {"url": 456}}}
        )

    def test_empty_dict(self) -> None:
        assert not is_mcp_tool({})


# =============================================================================
# convert_mcp_to_anthropic — single tool
# =============================================================================


class TestConvertSingleMCPTool:
    def test_top_level_url(self) -> None:
        tools = [
            {"name": "get_weather", "description": "Get weather", "url": "https://api.example.com"}
        ]
        result = convert_mcp_to_anthropic(tools)
        assert result == [
            {
                "type": "url",
                "name": "get_weather",
                "description": "Get weather",
                "url": "https://api.example.com",
                "input_schema": None,
            }
        ]

    def test_function_with_url(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Web search",
                    "url": "https://search.example.com",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ]
        result = convert_mcp_to_anthropic(tools)
        assert result[0]["type"] == "url"
        assert result[0]["name"] == "search"
        assert result[0]["description"] == "Web search"
        assert result[0]["url"] == "https://search.example.com"
        assert result[0]["input_schema"] == {
            "type": "object",
            "properties": {"q": {"type": "string"}},
        }

    def test_metadata_url(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calc",
                    "metadata": {"url": "https://calc.example.com"},
                },
            }
        ]
        result = convert_mcp_to_anthropic(tools)
        assert result[0]["type"] == "url"
        assert result[0]["url"] == "https://calc.example.com"

    def test_authorization_from_function(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "secure",
                    "url": "https://secure.example.com",
                    "authorization": "Bearer tok123",
                },
            }
        ]
        result = convert_mcp_to_anthropic(tools)
        assert result[0]["authorization"] == "Bearer tok123"

    def test_authorization_from_metadata(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "secure",
                    "url": "https://secure.example.com",
                    "metadata": {"authorization": "ApiKey abc"},
                },
            }
        ]
        result = convert_mcp_to_anthropic(tools)
        assert result[0]["authorization"] == "ApiKey abc"

    def test_authorization_top_level(self) -> None:
        tools = [
            {
                "name": "secure",
                "url": "https://secure.example.com",
                "authorization": "Token xyz",
            }
        ]
        result = convert_mcp_to_anthropic(tools)
        assert result[0]["authorization"] == "Token xyz"

    def test_no_authorization_omitted(self) -> None:
        tools = [
            {
                "name": "open",
                "url": "https://open.example.com",
            }
        ]
        result = convert_mcp_to_anthropic(tools)
        assert "authorization" not in result[0]

    def test_cache_control_propagated(self) -> None:
        tools = [
            {
                "name": "cached",
                "url": "https://cached.example.com",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        result = convert_mcp_to_anthropic(tools)
        assert result[0]["cache_control"] == {"type": "ephemeral"}


# =============================================================================
# convert_mcp_to_anthropic — mixed and edge cases
# =============================================================================


class TestConvertMixedTools:
    def test_mcp_plus_regular(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Weather",
                    "parameters": {"type": "object"},
                },
            },
            {
                "name": "mcp_search",
                "url": "https://search.example.com",
                "description": "Search the web",
                "input_schema": {"type": "object"},
            },
        ]
        result = convert_mcp_to_anthropic(tools)
        assert len(result) == 2
        # first stays function
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        # second becomes url
        assert result[1]["type"] == "url"
        assert result[1]["name"] == "mcp_search"

    def test_pass_through_non_mcp(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calc",
                    "description": "Calculate",
                    "parameters": {"type": "object"},
                },
            },
        ]
        result = convert_mcp_to_anthropic(tools)
        assert result[0] == tools[0]
        assert result[0] is not tools[0]  # copy, not alias

    def test_empty_tools(self) -> None:
        result = convert_mcp_to_anthropic([])
        assert result == []

    def test_both_tools_and_mcp(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get time",
                    "parameters": {"type": "object"},
                },
            },
            {
                "name": "mcp_search",
                "url": "https://search.example.com",
                "description": "Web search",
                "input_schema": {"type": "object"},
            },
            {
                "type": "function",
                "function": {
                    "name": "code_interp",
                    "description": "Interpret code",
                    "parameters": {"type": "object"},
                },
            },
        ]
        result = convert_mcp_to_anthropic(tools)
        assert result[0]["type"] == "function"
        assert result[1]["type"] == "url"
        assert result[2]["type"] == "function"

    def test_input_schema_from_parameters(self) -> None:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "reverse",
                    "url": "https://reverse.example.com",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                },
            }
        ]
        result = convert_mcp_to_anthropic(tools)
        assert result[0]["input_schema"] == {
            "type": "object",
            "properties": {"text": {"type": "string"}},
        }

    def test_input_schema_from_top_level(self) -> None:
        tools = [
            {
                "name": "upper",
                "url": "https://upper.example.com",
                "input_schema": {"type": "object"},
            }
        ]
        result = convert_mcp_to_anthropic(tools)
        assert result[0]["input_schema"] == {"type": "object"}

    def test_mutability_safety(self) -> None:
        """Input must not be modified."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get time",
                    "parameters": {"type": "object"},
                },
            },
        ]
        original = tools[0]
        convert_mcp_to_anthropic(tools)
        assert tools[0] is original
        assert tools[0]["type"] == "function"

    def test_authorization_priority(self) -> None:
        """Priority: tool-level > function-level > metadata-level."""
        tools = [
            {
                "name": "conflict",
                "url": "https://example.com",
                "authorization": "tool-level",
                "function": {
                    "authorization": "function-level",
                    "metadata": {"authorization": "metadata-level"},
                },
            }
        ]
        result = convert_mcp_to_anthropic(tools)
        assert result[0]["authorization"] == "tool-level"
