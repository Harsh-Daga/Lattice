"""Tests for lattice.providers.schema_filter.

Covers every path in sanitize_json_schema and sanitize_tool_definitions.
"""

from __future__ import annotations

from typing import Any

import pytest

from lattice.providers.schema_filter import (
    _UNSUPPORTED_KEYS,
    _build_annotation,
    sanitize_json_schema,
    sanitize_tool_definitions,
)


class TestSanitizeJsonSchema:
    def test_none_returns_none(self) -> None:
        assert sanitize_json_schema(None) is None

    def test_primitive_types_preserved(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "description": "A test object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "description": "User age"},
            },
        }
        out = sanitize_json_schema(schema)
        assert out == schema

    def test_unsupported_keys_stripped(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "exclusiveMinimum": -1,
                    "multipleOf": 5,
                }
            },
        }
        out = sanitize_json_schema(schema)
        prop = out["properties"]["count"]
        assert "minimum" not in prop
        assert "maximum" not in prop
        assert prop["type"] == "integer"

    def test_injected_description(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "rating": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Star rating",
                }
            },
        }
        out = sanitize_json_schema(schema)
        desc = out["properties"]["rating"]["description"]
        assert "Star rating" in desc
        assert "minimum=1" in desc or "maximum=5" in desc

    def test_no_description_before_injection(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "pattern": "^[A-Z]{3}$",
                }
            },
        }
        out = sanitize_json_schema(schema)
        desc = out["properties"]["code"]["description"]
        assert "Constraints:" in desc
        assert "pattern=" in desc

    def test_nested_object_recurse(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {"zip": {"type": "string", "minLength": 5, "maxLength": 10}},
                }
            },
        }
        out = sanitize_json_schema(schema)
        zip_prop = out["properties"]["address"]["properties"]["zip"]
        assert "minLength" not in zip_prop
        assert "maxLength" not in zip_prop
        assert "type" in zip_prop

    def test_array_items_recurse(self) -> None:
        schema: dict[str, Any] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer", "minimum": 0}},
            },
        }
        out = sanitize_json_schema(schema)
        assert "minimum" not in out["items"]["properties"]["id"]
        assert out["items"]["properties"]["id"]["type"] == "integer"

    def test_any_of_one_of_all_of_recurse(self) -> None:
        schema: dict[str, Any] = {
            "anyOf": [
                {"type": "string", "minLength": 1},
                {"type": "integer", "minimum": 0},
            ],
            "oneOf": [{"type": "boolean"}],
            "allOf": [{"type": "object", "additionalProperties": False}],
        }
        out = sanitize_json_schema(schema)
        assert "minLength" not in out["anyOf"][0]
        assert "minimum" not in out["anyOf"][1]
        assert out["allOf"][0]["type"] == "object"
        assert "additionalProperties" not in out["allOf"][0]

    def test_ref_dropped(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"link": {"$ref": "#/definitions/Node"}},
        }
        out = sanitize_json_schema(schema)
        assert "$ref" not in out["properties"]["link"]
        # $ref should become an injected description annotation
        assert "link" in out["properties"]

    def test_root_schema_meta_preserved(self) -> None:
        schema: dict[str, Any] = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "definitions": {"foo": {"type": "string"}},
        }
        out = sanitize_json_schema(schema)
        assert out["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert out["definitions"] == {"foo": {"type": "string"}}

    def test_inject_disabled(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"x": {"type": "integer", "minimum": 0}},
        }
        out = sanitize_json_schema(schema, inject_descriptions=False)
        assert "description" not in out["properties"]["x"]
        assert "minimum" not in out["properties"]["x"]

    def test_cap_truncates_long_annotation(self) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "big": {
                    "type": "string",
                    "default": "x" * 200,
                }
            },
        }
        out = sanitize_json_schema(schema, max_description_len=50)
        desc = out["properties"]["big"]["description"]
        assert len(desc) <= 50

    @pytest.mark.parametrize("bad_key", list(_UNSUPPORTED_KEYS))
    def test_all_unsupported_keys_stripped(self, bad_key: str) -> None:
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {"p": {"type": "string", bad_key: "whatever"}},
        }
        out = sanitize_json_schema(schema, inject_descriptions=False)
        assert bad_key not in out["properties"]["p"]

    def test_non_dict_pass_through(self) -> None:
        assert sanitize_json_schema("not a dict") == "not a dict"  # type: ignore[arg-type]


class TestSanitizeToolDefinitions:
    def test_empty_tools(self) -> None:
        assert sanitize_tool_definitions([]) == []
        assert sanitize_tool_definitions(None) is None

    def test_openai_tools_sanitized(self) -> None:
        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "sum",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer", "minimum": 0},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]
        out = sanitize_tool_definitions(tools)
        params = out[0]["function"]["parameters"]
        assert "minimum" not in params["properties"]["a"]
        assert params["properties"]["a"]["type"] == "integer"

    def test_anthropic_format_tools(self) -> None:
        tools: list[dict[str, Any]] = [
            {
                "name": "weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "pattern": "^[a-zA-Z ]+$"}},
                },
            }
        ]
        out = sanitize_tool_definitions(tools)
        params = out[0]["input_schema"]["properties"]["city"]
        assert "pattern" not in params


class TestBuildAnnotation:
    def test_empty(self) -> None:
        assert _build_annotation({}, cap=1000) == ""

    def test_simple(self) -> None:
        assert _build_annotation({"min": 0}, cap=100) == "min=0"

    def test_long_value_truncated(self) -> None:
        long_val = "x" * 200
        ann = _build_annotation({"default": long_val}, cap=1000)
        assert ann.endswith("...")
        assert len(ann) < 250

    def test_cap_cutoff(self) -> None:
        ann = _build_annotation({"a": "1", "b": "2", "c": "3"}, cap=10)
        assert len(ann) <= 10
