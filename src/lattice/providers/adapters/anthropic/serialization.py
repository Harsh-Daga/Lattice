"""Anthropic adapter — full Messages API support.

Implements every nuance discovered in FreeRouter ``provider.ts`` and
LiteLLM ``anthropic/chat/transformation.py``:

1. **OAuth mode** — detects ``sk-ant-oat`` tokens and sends the full
   Claude Code header stack (``Authorization: Bearer``, ``anthropic-beta``,
   ``user-agent``, etc.).
2. **System prompt blocks** — supports ``cache_control`` on system blocks
   (required for Claude Code OAuth) and plain-string fallback.
3. **Tool ID sanitisation** — validates ``^[a-zA-Z0-9_-]+$`` before sending;
   bidirectional mapping so response tool IDs are faithfully restored.
4. **Tool result merging** — consecutive ``role: tool`` messages collapse
   into a single user message with an array of ``tool_result`` blocks.
5. **Tool argument parsing** — explicitly ``json.loads`` string arguments
   (OpenAI tool_calls often have ``arguments`` as a JSON string).
6. **Thinking config** — supports ``adaptive`` (Opus 4.6+) and ``enabled``
   with ``budget_tokens``; temperature is **omitted** when thinking is on.
7. **max_tokens arithmetic** — when thinking is enabled, budget is added
   to the user's ``max_tokens`` so Claude can both think and answer.
 8. **Streaming** — full state machine via
    :mod:`AnthropicStreamState <lattice.providers.stream_state>`:
   text deltas, tool_use start + argument streaming, and **thinking skipped**.
9. **Finish reason mapping** — ``tool_use`` → ``tool_calls``,
   ``end_turn`` → ``stop``, ``max_tokens`` → ``length``.

Thread Safety
-------------
Tool-ID mappings are stored in a :class:`contextvars.ContextVar` so
concurrent async requests never collide.

References
----------
- LiteLLM ``litellm/llms/anthropic/chat/transformation.py`` (~2100 LOC).
- Anthropic Messages API docs (2024-06-01 version).
"""

from __future__ import annotations

import json
from typing import Any

from lattice.planner.runtime_state import get_canonical_request_value
from lattice.providers.adapters.anthropic.context import _ctx_tool_id_mapping
from lattice.providers.adapters.base import _pop_system, _remap_tool_choice, _remap_tools
from lattice.providers.mcp_to_anthropic import convert_mcp_to_anthropic, is_mcp_tool
from lattice.providers.schema_filter import sanitize_json_schema, sanitize_tool_definitions
from lattice.providers.tool_sanitizer import (
    sanitize_tool_ids,
)
from lattice.transport.types import Request


class AnthropicSerializeMixin:
    """Anthropic adapter segment."""

    def serialize_request(self, request: Request) -> dict[str, Any]:
        """Internal Request → Anthropic Messages API JSON body."""
        mapping: dict[str, str] = {}
        _ctx_tool_id_mapping.set(mapping)

        # 0. Determine flags from request metadata (set by proxy or SDK)
        use_cache_control = self._cache_control_enabled(request)
        cache_ttl_seconds = self._cache_ttl_seconds(request)
        use_defer_loading = request.metadata.get("anthropic_defer_loading", False)
        use_allowed_callers = request.metadata.get("anthropic_allowed_callers", False)
        use_output_schema = request.metadata.get("anthropic_output_schema", False)

        # 1. Extract system (plain string or block array with cache_control)
        messages_raw: list[dict[str, Any]] = []
        for msg in request.messages:
            m: dict[str, Any] = {
                "role": str(msg.role),
                "content": msg.content,
            }
            if msg.metadata.get("cache_control"):
                m["cache_control"] = msg.metadata["cache_control"]
            if msg.name:
                m["name"] = msg.name
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                m["tool_calls"] = msg.tool_calls
            messages_raw.append(m)

        system_text, messages = _pop_system(messages_raw)

        # 2. Sanitise tool names FIRST (while still in OpenAI format)
        sanitized_tools = list(request.tools) if request.tools else None
        if sanitized_tools:
            sanitized_tools = sanitize_tool_ids(sanitized_tools, mapping, provider="anthropic")

        # 2. Detect MCP tools and separate them from regular tools
        mcp_tools: list[dict[str, Any]] = []
        regular_tools: list[dict[str, Any]] = [] if sanitized_tools else []
        if sanitized_tools:
            for t in sanitized_tools:
                if is_mcp_tool(t):
                    mcp_tools.append(t)
                else:
                    regular_tools.append(t)

        # 2a. Convert MCP tools to Anthropic url format (before general remap)
        converted_mcp = convert_mcp_to_anthropic(mcp_tools) if mcp_tools else []

        # 2b. Strip unsupported JSON Schema keywords from regular tools
        regular_tools = sanitize_tool_definitions(regular_tools) or regular_tools

        # 3. Remap regular tools (OpenAI → Anthropic format)
        tools = _remap_tools(regular_tools) if regular_tools else []
        if converted_mcp:
            tools = (tools or []) + converted_mcp

        # 3a. Attach tool-level metadata (cache_control, defer_loading, allowed_callers)
        if tools:
            tools = self._annotate_tools(
                tools,
                cache_control=use_cache_control,
                defer_loading=use_defer_loading,
                allowed_callers=use_allowed_callers,
                cache_ttl_seconds=cache_ttl_seconds,
            )

        # 4. Remap messages (tool results merged, tool arguments parsed)
        anthropic_messages = self._remap_messages(messages)

        # 5. Thinking / reasoning config
        thinking = self._get_thinking_config(request.model)

        # 6. Build body
        computed_max_tokens = self._compute_max_tokens(request, thinking)
        body: dict[str, Any] = {
            "model": request.model,
            "messages": anthropic_messages,
        }
        if computed_max_tokens is not None:
            body["max_tokens"] = computed_max_tokens

        # 6a. System — plain string or block array with cache_control
        system_blocks = self._build_system_blocks(
            system_text,
            use_cache=use_cache_control,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        if system_blocks:
            body["system"] = system_blocks

        # Temperature — OMITTED when thinking is active (FreeRouter: provider.ts:306)
        if request.temperature is not None and not thinking:
            body["temperature"] = request.temperature

        # top_p — Anthropic supports this (was missing)
        if request.top_p is not None:
            body["top_p"] = request.top_p

        if request.stream:
            body["stream"] = True
        if request.stop:
            body["stop_sequences"] = request.stop
        if tools:
            body["tools"] = tools
        if request.tool_choice is not None:
            body["tool_choice"] = _remap_tool_choice(request.tool_choice)

        # Thinking config
        if thinking:
            body["thinking"] = thinking

        # 6b. Output format / structured output (JSON schema)
        output_schema = request.metadata.get("output_schema")
        if output_schema and use_output_schema:
            body["tool_choice"] = {"type": "any"}
            tools = tools or []
            tools.append(self._build_output_schema_tool(output_schema))
            body["tools"] = tools

        # Store mapping for later deserialization
        request.metadata["_anthropic_tool_id_mapping"] = dict(mapping)
        return body

    # ------------------------------------------------------------------
    # Tool annotation helpers (cache_control, defer_loading, allowed_callers)
    # ------------------------------------------------------------------

    @staticmethod
    def _annotate_tools(
        tools: list[dict[str, Any]],
        *,
        cache_control: bool,
        defer_loading: bool,
        allowed_callers: bool,
        cache_ttl_seconds: int | None = None,
    ) -> list[dict[str, Any]]:
        """Add Anthropic-specific metadata fields to tool definitions.

        These fields are **not** part of the standard Messages API but are
        supported by Claude Code / Claude 4+ for performance and security:

        * ``cache_control`` — mark tool descriptions for prompt caching.
        * ``deferred_loading`` — lazy-load tool descriptions when needed.
        * ``allowed_callers`` — restrict which callers can invoke the tool.
        """
        out: list[dict[str, Any]] = []
        for tool in tools:
            t = dict(tool)
            if cache_control:
                t.setdefault(
                    "cache_control", AnthropicSerializeMixin._cache_control_block(cache_ttl_seconds)
                )
            if defer_loading:
                t["deferred_loading"] = True
            if allowed_callers:
                # Default: allow self + assistant (Claude itself)
                t.setdefault("allowed_callers", ["user", "assistant"])
            out.append(t)
        return out

    @staticmethod
    def _build_system_blocks(
        system_text: str | None, *, use_cache: bool, cache_ttl_seconds: int | None = None
    ) -> list[dict[str, Any]] | str | None:
        """Build system prompt as blocks (with cache_control) or plain string."""
        if not system_text:
            return None
        if not use_cache:
            return system_text
        # Block array with cache_control (Claude Code OAuth mode)
        return [
            {
                "type": "text",
                "text": system_text,
                "cache_control": AnthropicSerializeMixin._cache_control_block(cache_ttl_seconds),
            }
        ]

    @staticmethod
    def _cache_control_block(ttl_seconds: int | None = None) -> dict[str, Any]:
        block: dict[str, Any] = {"type": "ephemeral"}
        if ttl_seconds == 3600:
            block["ttl"] = "1h"
        return block

    @staticmethod
    def _cache_arbitrage_annotations(request: Request) -> dict[str, Any]:
        cache_arbitrage = get_canonical_request_value(request, None, "_cache_arbitrage")
        if not isinstance(cache_arbitrage, dict):
            return {}
        annotations = cache_arbitrage.get("annotations")
        return annotations if isinstance(annotations, dict) else {}

    @classmethod
    def _cache_control_enabled(cls, request: Request) -> bool:
        """Return True if Anthropic cache_control should be injected."""
        if get_canonical_request_value(request, None, "anthropic_cache_control"):
            return True
        if any(msg.metadata.get("cache_control") for msg in request.messages):
            return True
        # Check _lattice_cache_plan from ExecutionPlan
        exec_plan_cache = get_canonical_request_value(request, None, "_lattice_cache_plan")
        if isinstance(exec_plan_cache, list):
            provider = getattr(request, "provider", "")
            for entry in exec_plan_cache:
                mode = entry.get("provider_mode", "")
                if mode == "explicit_breakpoint" and provider in (None, "", "anthropic"):
                    return True
        annotations = cls._cache_arbitrage_annotations(request)
        cache = annotations.get("cache")
        provider = annotations.get("provider")
        if isinstance(cache, dict) and cache.get("mode") == "explicit_breakpoint":
            return provider in (None, "anthropic")
        return False

    @classmethod
    def _cache_ttl_seconds(cls, request: Request) -> int | None:
        ttl = get_canonical_request_value(request, None, "anthropic_cache_ttl_seconds")
        if isinstance(ttl, int):
            return ttl
        annotations = cls._cache_arbitrage_annotations(request)
        cache = annotations.get("cache")
        if isinstance(cache, dict) and isinstance(cache.get("default_ttl_seconds"), int):
            return cache["default_ttl_seconds"]
        return None

    @staticmethod
    def _build_output_schema_tool(output_schema: dict[str, Any]) -> dict[str, Any]:
        """Build a synthetic tool for structured JSON output.

        Anthropic does not natively support ``response_format: {type:"json_object"}``.
        The standard workaround is to inject a special ``json`` tool with the
        schema as its ``input_schema`` and force ``tool_choice: any`` so Claude
        must call it.  The result is then extracted from the tool_use response.

        References
        ----------
        - LiteLLM ``anthropic/transformation.py:252-363`` structured output path.
        - Anthropic blog: "Structured JSON output with tool_use" (2024-10).
        """
        sanitized = sanitize_json_schema(output_schema, inject_descriptions=False) or output_schema
        return {
            "name": "json",
            "description": "Respond with a valid JSON object matching the requested schema.",
            "input_schema": sanitized,
        }

    # ------------------------------------------------------------------
    # Message remapping (OpenAI → Anthropic)
    # ------------------------------------------------------------------

    @staticmethod
    def _remap_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI messages to Anthropic format with **tool merging**.

        Key transformations (FreeRouter provider.ts:149-211):
        * ``role: tool`` → ``role: user`` with ``content: [{type:"tool_result"}]``.
        * Consecutive ``role: tool`` messages are **merged** into one user
          message whose content array holds every ``tool_result`` block.
        * ``role: assistant`` + ``tool_calls`` → content blocks with
          ``tool_use`` (with argument objects, not strings).
        * Regular text → plain string ``content``.
        """
        out: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            # System/developer handled upstream
            if role in ("system", "developer"):
                continue

            # Tool result → merge into previous user message if it only
            # contains tool_results.
            if role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                tool_block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": str(content) if content else "",
                }
                last = out[-1] if out else None
                if (
                    last
                    and last.get("role") == "user"
                    and isinstance(last.get("content"), list)
                    and all(b.get("type") == "tool_result" for b in last["content"])
                ):
                    last["content"].append(tool_block)
                else:
                    out.append({"role": "user", "content": [tool_block]})
                continue

            # Assistant with tool_calls → content blocks
            if role == "assistant" and msg.get("tool_calls"):
                blocks: list[dict[str, Any]] = []

                # Text content first
                if content:
                    text = str(content)
                    if text:
                        blocks.append({"type": "text", "text": text})

                # tool_use blocks
                for tc in msg["tool_calls"]:
                    args = tc.get("function", {}).get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": tc.get("function", {}).get("name", ""),
                            "input": args,
                        }
                    )
                out.append({"role": "assistant", "content": blocks})
                continue

            # Regular user / assistant message
            out.append(
                {
                    "role": role if role in ("user", "assistant") else "user",
                    "content": str(content) if content else "",
                }
            )

        return out

    # ------------------------------------------------------------------
    # Thinking config
    # ------------------------------------------------------------------

    @staticmethod
    def _get_thinking_config(model: str) -> dict[str, Any] | None:
        """Return Anthropic ``thinking`` payload ONLY for models known
        to support it (Opus 4.6+, Sonnet 4.5+).

        Phase 1: simple heuristic from model string.  Phase 3 will
        support explicit ``thinking`` param on Request.
        """
        lower = model.lower()
        if any(v in lower for v in ("opus-4-6", "opus_4_6", "opus-4.6", "opus_4.6")):
            return {"type": "adaptive"}
        if any(v in lower for v in ("opus-4-7", "opus_4_7", "opus-4.7", "opus_4.7")):
            return {"type": "adaptive"}
        # Sonnet 4.5 with budget – but only very recent models
        if any(v in lower for v in ("sonnet-4-5", "sonnet_4_5")):
            return {"type": "enabled", "budget_tokens": 4096}
        # Older models (claude-3-opus, claude-3-sonnet, etc) do NOT auto-enable
        return None

    # Model-specific default max_tokens when user does not specify.
    _DEFAULT_MAX_TOKENS: dict[str, int] = {
        "opus": 8192,
        "sonnet": 8192,
        "haiku": 4096,
        "claude-3": 4096,
        "claude-4": 8192,
    }

    @classmethod
    def _default_max_tokens(cls, model: str) -> int:
        """Return a sensible default max_tokens for the model."""
        lower = model.lower()
        for key, value in cls._DEFAULT_MAX_TOKENS.items():
            if key in lower:
                return value
        return 4096

    @classmethod
    def _compute_max_tokens(cls, request: Request, thinking: dict[str, Any] | None) -> int | None:
        """Compute ``max_tokens`` for Anthropic API.

        Design decisions:
        - If user does not specify ``max_tokens``, apply a model-aware default
          (4096–8192) so Anthropic never receives an unbounded request that
          could silently truncate long outputs.
        - When thinking is enabled with a ``budget_tokens``, ``max_tokens``
          MUST be strictly greater than the budget.  We enforce this by
          returning ``max(budget + 1, user_value, default)``.
        """
        base = request.max_tokens
        default = cls._default_max_tokens(request.model)

        # Determine thinking budget
        budget = 0
        if thinking and thinking.get("type") == "enabled":
            budget = thinking.get("budget_tokens", 0)

        if base is None:
            # No user value — use default, but ensure it exceeds thinking budget
            return max(default, budget + 1)

        # User specified a value — ensure it exceeds thinking budget
        if budget > 0 and base <= budget:
            return max(budget + 1, default)

        # Enforce absolute minimum for meaningful responses
        if base < 1:
            base = 1
        return base

    # ------------------------------------------------------------------
    # Deserialization (Anthropic JSON → internal Response)
    # ------------------------------------------------------------------

