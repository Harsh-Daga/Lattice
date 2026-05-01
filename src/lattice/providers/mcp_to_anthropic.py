"""Translate OpenAI MCP server tools to Anthropic ``url`` tools.

Anthropic supports a special ``type: "url"`` tool for MCP (Model Context Protocol)
server tools.  OpenAI-format MCP tools carry a ``url`` field (and optionally an
``authorization`` token).  This module detects and converts them while leaving
regular ``function`` tools untouched.

References
----------
- Anthropic Messages API docs — tool types (2024-06-01).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ============================================================================
# Data model
# ============================================================================

@dataclass(frozen=True)
class MCPServerTool:
    """Normalised representation of an MCP server tool in OpenAI format.

    Attributes
    ----------
    name:
        Tool name (maps to Anthropic ``name``).
    description:
        Human-readable description.
    url:
        MCP server endpoint URL.
    authorization:
        Optional bearer token or API key for the MCP server.
    input_schema:
        JSON Schema describing the tool's inputs (OpenAI ``parameters``).
    """

    name: str
    description: str = ""
    url: str = ""
    authorization: str | None = None
    input_schema: dict[str, Any] | None = None


# ============================================================================
# Detection
# ============================================================================

def _extract_url(tool: dict[str, Any]) -> str | None:
    """Return the URL string if *tool* carries one, else ``None``."""
    if "url" in tool and isinstance(tool["url"], str):
        return tool["url"]
    fn = tool.get("function", {})
    if isinstance(fn, dict):
        if "url" in fn and isinstance(fn["url"], str):
            return fn["url"]
        meta = fn.get("metadata", {})
        if isinstance(meta, dict) and "url" in meta and isinstance(meta["url"], str):
            return meta["url"]
    return None


def is_mcp_tool(tool: dict[str, Any]) -> bool:
    """Return ``True`` when *tool* is an MCP server tool (has a ``url`` field).

    The ``url`` may live at the top level, inside ``function.url``, or inside
    ``function.metadata.url``.
    """
    return _extract_url(tool) is not None


# ============================================================================
# Conversion helpers
# ============================================================================

def _extract_authorization(tool: dict[str, Any]) -> str | None:
    """Locate the authorization token in all supported nesting levels."""
    if "authorization" in tool and isinstance(tool["authorization"], str):
        return tool["authorization"]
    fn = tool.get("function", {})
    if isinstance(fn, dict):
        if "authorization" in fn and isinstance(fn["authorization"], str):
            return fn["authorization"]
        meta = fn.get("metadata", {})
        if (
            isinstance(meta, dict)
            and "authorization" in meta
            and isinstance(meta["authorization"], str)
        ):
            return meta["authorization"]
    return None


def _extract_input_schema(tool: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the JSON Schema from OpenAI ``function.parameters`` or top-level ``input_schema``."""
    if "input_schema" in tool and isinstance(tool["input_schema"], dict):
        return tool["input_schema"]
    fn = tool.get("function", {})
    if isinstance(fn, dict) and "parameters" in fn and isinstance(fn["parameters"], dict):
        return fn["parameters"]
    return None


def _extract_description(tool: dict[str, Any]) -> str:
    """Return the description from ``function.description`` or the tool itself."""
    fn = tool.get("function", {})
    if isinstance(fn, dict) and "description" in fn:
        desc = fn["description"]
        if isinstance(desc, str):
            return desc
    raw = tool.get("description", "")
    return raw if isinstance(raw, str) else ""


def _extract_name(tool: dict[str, Any]) -> str:
    """Return the name from ``function.name`` or the tool itself."""
    fn = tool.get("function", {})
    if isinstance(fn, dict) and "name" in fn:
        name = fn["name"]
        if isinstance(name, str):
            return name
    raw = tool.get("name", "")
    return raw if isinstance(raw, str) else ""


def _build_anthropic_url_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert a single OpenAI-format MCP tool to Anthropic ``type: "url"``."""
    url = _extract_url(tool)
    assert url is not None  # guarded by is_mcp_tool

    anthropic: dict[str, Any] = {
        "type": "url",
        "name": _extract_name(tool),
        "description": _extract_description(tool),
        "url": url,
        "input_schema": _extract_input_schema(tool),
    }

    auth = _extract_authorization(tool)
    if auth is not None:
        anthropic["authorization"] = auth

    cache = tool.get("cache_control")
    if cache is not None:
        anthropic["cache_control"] = cache

    return anthropic


# ============================================================================
# Public API
# ============================================================================

def convert_mcp_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert MCP tools in an OpenAI-format ``tools`` array to Anthropic ``url`` tools.

    Non-MCP tools (regular ``function`` tools without a ``url`` field) are
    passed through unchanged.  A new list is always returned; the input is never
    modified.

    Parameters
    ----------
    tools:
        OpenAI ``tools`` array, potentially containing MCP and regular tools.

    Returns
    -------
    A new ``tools`` array with MCP items translated to ``type: "url"`` and
    all other items left untouched.
    """
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if is_mcp_tool(tool):
            converted.append(_build_anthropic_url_tool(tool))
        else:
            converted.append(dict(tool))  # shallow copy for immutability
    return converted
