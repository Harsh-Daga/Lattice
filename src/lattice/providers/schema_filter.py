"""JSON Schema filter for Anthropic tool definitions.

Anthropic's tool `input_schema` only supports a strict subset of JSON Schema
draft-07.  Passing unsupported fields causes **silent truncation** on some
models and **400 errors** on newer models (Claude 4.6+).  This module strips
unsupported keywords **and** injects them into the ``description`` field so
no semantic information is lost.

Unsupported keywords:
    ``maxItems``, ``minItems``, ``minimum``, ``maximum``, ``exclusiveMinimum``,
    ``exclusiveMaximum``, ``multipleOf``, ``minLength``, ``maxLength``,
    ``pattern``, ``format``, ``default``, ``additionalProperties``,
    ``uniqueItems``, ``propertyNames``, ``readOnly``, ``writeOnly``,
    ``deprecated``, ``const``, ``if``, ``then``, ``else``

Supported keywords:
    ``type``, ``description``, ``enum``, ``required``, ``properties``, ``items``,
    ``anyOf``, ``oneOf``, ``allOf``, ``title``

References
----------
- Anthropic Messages API docs — Tool use / Structured output (2024-06-01).
"""

from __future__ import annotations

from typing import Any

# Anthropic does not guarantee support for these JSON Schema keywords.
_UNSUPPORTED_KEYS: frozenset[str] = frozenset(
    {
        "maxItems",
        "minItems",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "default",
        "additionalProperties",
        "uniqueItems",
        "propertyNames",
        "readOnly",
        "writeOnly",
        "deprecated",
        "const",
        "if",
        "then",
        "else",
    }
)

# Keywords that are ALWAYS allowed.
_SUPPORTED_KEYS: frozenset[str] = frozenset(
    {
        "type",
        "description",
        "enum",
        "required",
        "properties",
        "items",
        "anyOf",
        "oneOf",
        "allOf",
        "title",
    }
)


def sanitize_json_schema(
    schema: dict[str, Any] | None,
    *,
    inject_descriptions: bool = True,
    max_description_len: int = 1024,
) -> dict[str, Any] | None:
    """Return a schema safe for Anthropic ``input_schema``.

    Unsupported fields are **stripped** and, when ``inject_descriptions`` is
    ``True``, appended to the owning property's ``description`` as a flat
    annotation string so the model still sees the constraint.

    Parameters
    ----------
    schema: dict | None
        The raw JSON schema (usually ``function.parameters`` from an OpenAI
        tool definition).
    inject_descriptions: bool
        If ``True``, write stripped constraints into the ``description``
        field of the property that contained them.
    max_description_len: int
        Cap injected annotations to avoid exploding prompt size.

    Returns
    -------
    dict | None
        Sanitized schema, or ``None`` if input was ``None``.
    """
    if schema is None:
        return None
    if not isinstance(schema, dict):
        # Edge case: some providers pass a string or list. Pass-through.
        return schema

    return _filter_node(schema, _root=True, inject=inject_descriptions, cap=max_description_len)


def _filter_node(
    node: dict[str, Any],
    *,
    _root: bool = False,
    inject: bool,
    cap: int,
) -> dict[str, Any]:
    """Recursively filter a schema node.

    For each property/definition we:
    1. Collect stripped constraints.
    2. Annotate ``description`` with stripped info (if room).
    3. Recurse into ``properties``, ``items``, ``anyOf``/``oneOf``/``allOf``.
    """
    out: dict[str, Any] = {}
    stripped: dict[str, Any] = {}

    for key, value in node.items():
        if key in _SUPPORTED_KEYS:
            # Recurse into container nodes
            if key == "properties" and isinstance(value, dict):
                # Keep property names, filter each property's schema
                out[key] = {
                    prop_name: _filter_node(prop_schema, _root=False, inject=inject, cap=cap)
                    if isinstance(prop_schema, dict) else prop_schema
                    for prop_name, prop_schema in value.items()
                }
            elif key == "items" and isinstance(value, dict):
                out[key] = _filter_node(value, _root=False, inject=inject, cap=cap)
            elif key == "items" and isinstance(value, list) or key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
                out[key] = [_filter_node(v, _root=False, inject=inject, cap=cap) if isinstance(v, dict) else v for v in value]
            else:
                out[key] = value
        elif key in _UNSUPPORTED_KEYS:
            stripped[key] = value
        elif key == "$ref":
            # Drop $ref — Anthropic doesn't support it.
            stripped["$ref"] = value
        elif _root and key in ("$schema", "definitions"):
            # Keep root-level metadata (not part of input_schema, but harmless)
            out[key] = value
        else:
            # Unknown key — strip defensively
            stripped[key] = value

    # Inject stripped constraints into description
    if inject and stripped:
        annotation = _build_annotation(stripped, cap=cap)
        if annotation:
            desc = str(out.get("description", ""))
            desc = f"{desc} (Constraints: {annotation})" if desc else f"Constraints: {annotation}"
            out["description"] = desc[:cap]

    return out


def _build_annotation(stripped: dict[str, Any], *, cap: int) -> str:
    """Build a human-readable annotation of stripped constraints."""
    parts: list[str] = []
    for k, v in stripped.items():
        # Truncate long values (nested schemas, large arrays)
        val_str = str(v)
        if len(val_str) > 80:
            val_str = val_str[:77] + "..."
        parts.append(f"{k}={val_str}")
    annotation = ", ".join(parts)
    return annotation[:cap]


def sanitize_tool_definitions(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Sanitize every tool's schema in a list.

    Handles **both** OpenAI format (``function.parameters``) and Anthropic
    format (``input_schema`` top-level dict) tools.

    This is the public entry point called by adapters before serialising
    tools to Anthropic format.
    """
    if not tools:
        return tools
    out: list[dict[str, Any]] = []
    for tool in tools:
        t = dict(tool)
        # OpenAI format: function.parameters
        func = t.get("function")
        if isinstance(func, dict):
            params = func.get("parameters")
            if isinstance(params, dict):
                sanitized = sanitize_json_schema(params, inject_descriptions=True)
                if sanitized is not None:
                    func["parameters"] = sanitized
        # Anthropic format: input_schema is the schema itself
        input_schema = t.get("input_schema")
        if isinstance(input_schema, dict):
            sanitized = sanitize_json_schema(input_schema, inject_descriptions=True)
            if sanitized is not None:
                t["input_schema"] = sanitized
        out.append(t)
    return out
