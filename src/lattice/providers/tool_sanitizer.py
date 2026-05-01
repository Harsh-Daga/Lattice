"""Tool ID sanitization for provider compatibility.

Anthropic requires tool IDs to match ``^[a-zA-Z0-9_-]+$``.
Other providers may enforce different rules.  This module centralises
sanitisation logic and maintains bidirectional mappings so responses
can be faithfully reversed.

Design
------
- A single ``ToolSanitizer`` class for each provider with known rules.
- Stateless sanitise/unsanitise -- mappings passed in/out by caller
  (typically stored in ``TransformContext``).
- Fast-path when no sanitisation is needed (avoids dict overhead).

References
----------
- LiteLLM litellm/llms/anthropic/chat/transformation.py -- tool handling.
"""

from __future__ import annotations

import re
from typing import Any

# ============================================================================
# Patterns
# ============================================================================

# Anthropic: ids for tool_use.id must be ^[a-zA-Z0-9_-]+$
ANTHROPIC_TOOL_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

# General replacement of *any* character that is NOT in the allowed set.
_ANTHROPIC_SANITISE_RE = re.compile(r"[^a-zA-Z0-9_-]+")

# AWS Bedrock toolSpec name also has limited allowed set.
_BEDROCK_TOOL_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")


# ============================================================================
# ToolSanitizer base
# ============================================================================

class ToolSanitizer:
    """Sanitize tool IDs so they comply with a provider's validation rules.

    Attributes
    ----------
    name: str
        Human-readable identifier (e.g. ``"anthropic"``).
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def sanitize(self, raw_id: str) -> str:
        """Return a provider-safe ID.  No side effects."""
        raise NotImplementedError

    def needs_sanitization(self, raw_id: str) -> bool:
        """Return ``True`` if *raw_id* would be changed by ``sanitize``."""
        raise NotImplementedError

    @staticmethod
    def register(
        mapping: dict[str, str], raw_id: str, sanitized_id: str
    ) -> None:
        """Record ``raw <-> sanitized`` in a caller-supplied dict.

        The dict is updated in-place so that later ``unsanitize`` calls
        can look up the original value.
        """
        if raw_id == sanitized_id:
            # Fast path: no translation needed; skip dict churn.
            return
        mapping[raw_id] = sanitized_id

    @staticmethod
    def unsanitize(sanitized_id: str, mapping: dict[str, str]) -> str | None:
        """Look up an original ID from a caller-supplied mapping.

        Returns ``None`` when the ID was never sanitised (i.e. the
        sanitised and raw values are identical).
        """
        for raw, sanitised in mapping.items():
            if sanitised == sanitized_id:
                return raw
        return None


# ============================================================================
# AnthropicToolSanitizer
# ============================================================================

class AnthropicToolSanitizer(ToolSanitizer):
    """Sanitise tool and tool_use IDs for Anthropic Messages API.

    Anthropic tool_use ids must match ``^[a-zA-Z0-9_-]+$``.
    FreeRouter tests showed failures with colons, dots, slashes, spaces,
    @ symbols, and # signs.

    Strategy
    --------
    Replace every run of illegal characters with an underscore.  Collapse
    multiple consecutive underscores to a single one.  Strip leading or
    trailing underscores.

    Example
    -------
    >>> s = AnthropicToolSanitizer()
    >>> s.sanitize("call:with:colons")
    'call_with_colons'
    >>> s.sanitize("call.with.dots")
    'call_with_dots'
    >>> s.sanitize("call/with/slashes")
    'call_with_slashes'
    """

    def __init__(self) -> None:
        super().__init__(name="anthropic")

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def sanitize(self, raw_id: str) -> str:
        if ANTHROPIC_TOOL_ID_PATTERN.match(raw_id):
            return raw_id
        cleaned = _ANTHROPIC_SANITISE_RE.sub("_", raw_id)
        # Collapse multiple underscores
        cleaned = re.sub(r"_+", "_", cleaned)
        # Strip leading/trailing underscores
        cleaned = cleaned.strip("_")
        # Guard against total sanitisation producing empty string
        if not cleaned:
            # Fall back to generic hash-based name
            import hashlib
            h = hashlib.sha256(raw_id.encode()).hexdigest()[:16]
            cleaned = f"tool_{h}"
        return cleaned

    def needs_sanitization(self, raw_id: str) -> bool:
        return bool(not ANTHROPIC_TOOL_ID_PATTERN.match(raw_id))


# ============================================================================
# BedrockToolSanitizer
# ============================================================================

class BedrockToolSanitizer(ToolSanitizer):
    """Sanitise tool names for AWS Bedrock Converse ``toolSpec``.

    Bedrock enforces ``^[a-zA-Z0-9_-]+$`` for tool names.  The rules are
    effectively identical to Anthropic's so we reuse the same strategy.
    """

    _instance: BedrockToolSanitizer | None = None

    def __init__(self) -> None:
        super().__init__(name="bedrock")

    def sanitize(self, raw_id: str) -> str:
        if ANTHROPIC_TOOL_ID_PATTERN.match(raw_id):
            return raw_id
        cleaned = _BEDROCK_TOOL_NAME_RE.sub("_", raw_id)
        cleaned = re.sub(r"_+", "_", cleaned)
        cleaned = cleaned.strip("_")
        if not cleaned:
            import hashlib
            h = hashlib.sha256(raw_id.encode()).hexdigest()[:16]
            cleaned = f"tool_{h}"
        return cleaned

    def needs_sanitization(self, raw_id: str) -> bool:
        return bool(not ANTHROPIC_TOOL_ID_PATTERN.match(raw_id))

    @classmethod
    def sanitize_tool_name(cls, raw_name: str) -> str:
        """Convenience static method for one-off sanitization."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance.sanitize(raw_name)


# ============================================================================
# IdentitySanitizer (pass-through)
# ============================================================================

class IdentitySanitizer(ToolSanitizer):
    """No-op sanitizer for providers with no tool ID restrictions."""

    def __init__(self) -> None:
        super().__init__(name="identity")

    def sanitize(self, raw_id: str) -> str:
        return raw_id

    def needs_sanitization(self, _raw_id: str) -> bool:
        return False


# ============================================================================
# Global registry
# ============================================================================

_SANITISERS: dict[str, ToolSanitizer] = {
    "anthropic": AnthropicToolSanitizer(),
    "bedrock": BedrockToolSanitizer(),
    "identity": IdentitySanitizer(),
}


def get_sanitizer(provider: str) -> ToolSanitizer:
    """Return the default sanitizer for *provider*.

    Falls back to ``IdentitySanitizer`` when no rule is known.
    """
    return _SANITISERS.get(provider, _SANITISERS["identity"])


def sanitize_tool_ids(
    tools: list[dict[str, Any]],
    mapping: dict[str, str],
    provider: str = "anthropic",
) -> list[dict[str, Any]]:
    """Sanitise every tool name/id in a list of OpenAI-style tools.

    Mutates a deep copy -- the original list is **not** modified.
    Records ``raw -> sanitized`` mappings in *mapping*.

    Parameters
    ----------
    tools:
        OpenAI ``tools`` array.
    mapping:
        Dict updated in-place with bidirectional mappings.
    provider:
        Which provider rules to apply.

    Returns
    -------
    Sanitised deep copy of *tools*.
    """
    sanitizer = get_sanitizer(provider)
    cleaned: list[dict[str, Any]] = []
    for tool in tools:
        tool_copy: dict[str, Any] = dict(tool)
        fn = tool_copy.get("function", {})
        raw_name = fn.get("name", "")
        if raw_name and sanitizer.needs_sanitization(raw_name):
            safe_name = sanitizer.sanitize(raw_name)
            fn = dict(fn)
            fn["name"] = safe_name
            tool_copy["function"] = fn
            ToolSanitizer.register(mapping, raw_name, safe_name)
        cleaned.append(tool_copy)
    return cleaned


def restore_tool_id(sanitized_id: str, mapping: dict[str, str]) -> str:
    """Return original tool ID, falling back to *sanitized_id* unchanged."""
    original = ToolSanitizer.unsanitize(sanitized_id, mapping)
    return original if original is not None else sanitized_id


def restore_tool_call_ids(
    tool_calls: list[dict[str, Any]], mapping: dict[str, str]
) -> list[dict[str, Any]]:
    """Reverse sanitisation on a list of OpenAI-style tool_calls."""
    restored: list[dict[str, Any]] = []
    for tc in tool_calls:
        tc_copy = dict(tc)
        raw_id = restore_tool_id(tc_copy.get("id", ""), mapping)
        tc_copy["id"] = raw_id
        fn = tc_copy.get("function", {})
        if fn:
            raw_name = restore_tool_id(fn.get("name", ""), mapping)
            fn = dict(fn)
            fn["name"] = raw_name
            tc_copy["function"] = fn
        restored.append(tc_copy)
    return restored
