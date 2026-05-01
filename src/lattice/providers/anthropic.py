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

import contextvars
import json
from typing import Any

from lattice.core.transport import Request, Response
from lattice.providers.base import _pop_system, _remap_tool_choice, _remap_tools
from lattice.providers.mcp_to_anthropic import convert_mcp_to_anthropic, is_mcp_tool
from lattice.providers.schema_filter import sanitize_json_schema, sanitize_tool_definitions
from lattice.providers.stream_state import AnthropicStreamState
from lattice.providers.tool_sanitizer import (
    AnthropicToolSanitizer,
    restore_tool_call_ids,
    sanitize_tool_ids,
)

# =============================================================================
# Context-local storage (async-safe)
# =============================================================================

# Maps original tool ID ↔ sanitised tool ID per in-flight request.
_ctx_tool_id_mapping: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "_anthropic_tool_id_mapping", default=None
)


# =============================================================================
# AnthropicAdapter
# =============================================================================

class AnthropicAdapter:
    """Anthropic Messages API adapter with full Claude Code parity."""

    name = "anthropic"
    _PREFIXES = {"anthropic", "claude"}

    # Beta headers required for Claude Code OAuth (FreeRouter: provider.ts:322)
    _OAUTH_BETA_HEADERS: str = (
        "claude-code-20250219,"
        "oauth-2025-04-20,"
        "fine-grained-tool-streaming-2025-05-14,"
        "interleaved-thinking-2025-05-14"
    )

    def __init__(self) -> None:
        self._sanitizer = AnthropicToolSanitizer()
        self._stop_reason: str | None = None

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def supports(self, model: str) -> bool:
        """Matches ``anthropic/...`` or bare ``claude-...`` names."""
        prefix = (
            model.split("/", 1)[0].lower() if "/" in model else model.split("-", 1)[0].lower()
        )
        return prefix in self._PREFIXES

    def chat_endpoint(self, _model: str, base_url: str) -> str:
        return f"{base_url.rstrip('/')}/v1/messages"

    def map_model_name(self, model: str) -> str:
        """Strip prefix; Anthropic has no aliases."""
        if "/" in model:
            prefix, rest = model.split("/", 1)
            if prefix.lower() in self._PREFIXES:
                return rest
        return model

    def extra_headers(self, _request: Any) -> dict[str, str]:
        return {}

    def retry_config(self) -> dict[str, Any]:
        return {
            "max_retries": 3,
            "backoff_factor": 1.0,
            "retry_on": (429, 502, 503, 504),
        }

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def auth_headers(self, api_key: str | None) -> dict[str, str]:
        """Build provider-specific auth headers.

        * API key (``sk-ant-api``) → ``x-api-key`` only.
        * OAuth (``sk-ant-oat``) → ``Authorization: Bearer`` + beta stack.
        """
        h: dict[str, str] = {"anthropic-version": "2023-06-01"}
        if not api_key:
            return h

        is_oauth = api_key.startswith("sk-ant-oat")
        if is_oauth:
            h["Authorization"] = (
                api_key if api_key.startswith("Bearer ") else f"Bearer {api_key}"
            )
            h["anthropic-beta"] = self._OAUTH_BETA_HEADERS
            h["user-agent"] = "claude-cli/2.1.2 (external, cli)"
            h["x-app"] = "cli"
            h["anthropic-dangerous-direct-browser-access"] = "true"
        else:
            h["x-api-key"] = api_key

        return h

    def is_oauth(self, api_key: str | None) -> bool:
        """Return ``True`` if the key is an Anthropic OAuth token."""
        return bool(api_key and api_key.startswith("sk-ant-oat"))

    # ------------------------------------------------------------------
    # Serialization (Request → Anthropic JSON)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Serialization (Request → Anthropic JSON)
    # ------------------------------------------------------------------

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
                    "cache_control", AnthropicAdapter._cache_control_block(cache_ttl_seconds)
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
                "cache_control": AnthropicAdapter._cache_control_block(cache_ttl_seconds),
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
        cache_arbitrage = request.metadata.get("_cache_arbitrage")
        if not isinstance(cache_arbitrage, dict):
            return {}
        annotations = cache_arbitrage.get("annotations")
        return annotations if isinstance(annotations, dict) else {}

    @classmethod
    def _cache_control_enabled(cls, request: Request) -> bool:
        if request.metadata.get("anthropic_cache_control"):
            return True
        if any(msg.metadata.get("cache_control") for msg in request.messages):
            return True
        annotations = cls._cache_arbitrage_annotations(request)
        cache = annotations.get("cache")
        provider = annotations.get("provider")
        if isinstance(cache, dict) and cache.get("mode") == "explicit_breakpoint":
            return provider in (None, "anthropic")
        return False

    @classmethod
    def _cache_ttl_seconds(cls, request: Request) -> int | None:
        ttl = request.metadata.get("anthropic_cache_ttl_seconds")
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
                    and all(
                        b.get("type") == "tool_result"
                        for b in last["content"]
                    )
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
    def _compute_max_tokens(
        cls, request: Request, thinking: dict[str, Any] | None
    ) -> int | None:
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

    def deserialize_response(self, data: dict[str, Any]) -> Response:
        """Anthropic JSON → internal Response."""
        content_blocks = data.get("content", [])
        content = self._extract_content_from_blocks(content_blocks)
        thinking = self._extract_thinking_from_blocks(content_blocks)
        usage = self._extract_usage(data)
        tool_calls = self._extract_tool_calls(content_blocks)

        # Restore original tool IDs
        mapping = _ctx_tool_id_mapping.get()
        if mapping and tool_calls:
            tool_calls = restore_tool_call_ids(tool_calls, mapping)

        resp = Response(
            content=content,
            role=data.get("role", "assistant"),
            model=data.get("model", ""),
            usage=usage,
            finish_reason=self._map_finish_reason(data.get("stop_reason")),
            tool_calls=tool_calls or None,
        )
        if thinking:
            resp.metadata["thinking"] = thinking
        return resp

    @staticmethod
    def _extract_content_from_blocks(content: list[dict[str, Any]]) -> str:
        """Concatenate only ``text`` blocks; skip ``tool_use`` and ``thinking``."""
        parts: list[str] = []
        for block in content:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)

    @staticmethod
    def _extract_thinking_from_blocks(content: list[dict[str, Any]]) -> str:
        """Extract thinking content from blocks for metadata preservation."""
        parts: list[str] = []
        for block in content:
            if block.get("type") == "thinking":
                parts.append(block.get("thinking", ""))
            elif block.get("type") == "redacted_thinking":
                parts.append("[redacted_thinking]")
        return "\n".join(parts) if parts else ""

    @staticmethod
    def _extract_usage(data: dict[str, Any]) -> dict[str, int]:
        u = data.get("usage", {})
        prompt_tokens = u.get("input_tokens", u.get("prompt_tokens", 0))
        completion_tokens = u.get("output_tokens", u.get("completion_tokens", 0))
        cache_read = u.get("cache_read_input_tokens", 0)
        cache_create = u.get("cache_creation_input_tokens", 0)
        usage: dict[str, int] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if cache_read:
            usage["cache_read_input_tokens"] = cache_read
            usage["cached_tokens"] = cache_read
        if cache_create:
            usage["cache_creation_input_tokens"] = cache_create
        return usage

    @staticmethod
    def _extract_tool_calls(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract ``tool_use`` blocks as OpenAI-style tool_calls."""
        calls: list[dict[str, Any]] = []
        for block in content:
            if block.get("type") == "tool_use":
                args = block.get("input", {})
                calls.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                        },
                    }
                )
        return calls

    @staticmethod
    def _map_finish_reason(reason: str | None) -> str | None:
        if reason is None:
            return None
        mappings = {
            "tool_use": "tool_calls",
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        return mappings.get(reason, reason)

    # ------------------------------------------------------------------
    # Streaming normalization
    # ------------------------------------------------------------------

    def normalize_sse_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize Anthropic SSE to OpenAI delta format.

        **Deprecated for streaming tool_use** — use
        :meth:`normalize_sse_stream` to get a stateful machine.
        This method handles text-only streams correctly.
        """
        etype = chunk.get("type", "")
        delta_text = ""
        finish: str | None = None

        if etype == "content_block_delta":
            delta = chunk.get("delta", {})
            if delta.get("type") == "text_delta":
                delta_text = delta.get("text", "")
            elif delta.get("type") == "thinking_delta":
                # skip thinking in legacy path
                return None
        elif etype == "message_delta":
            stop_reason = chunk.get("delta", {}).get("stop_reason")
            if stop_reason:
                self._stop_reason = stop_reason
                finish = self._map_finish_reason(stop_reason)
        elif etype == "message_stop":
            finish = self._map_finish_reason(self._stop_reason or "stop")
            if finish:
                return {
                    "choices": [
                        {"delta": {}, "finish_reason": finish}
                    ]
                }
            return None
        else:
            return None

        if not delta_text and finish is None:
            return None

        return {
            "choices": [
                {
                    "delta": {"content": delta_text} if delta_text else {},
                    "finish_reason": finish,
                }
            ]
        }

    def normalize_sse_stream(self, model: str) -> AnthropicStreamState:
        """Return a fresh stream state machine for Anthropic SSE chunks."""
        return AnthropicStreamState(model=model)

    # ------------------------------------------------------------------
    # Content extraction
    # ------------------------------------------------------------------

    def extract_content(self, msg: dict[str, Any]) -> str:
        content = msg.get("content")
        if isinstance(content, list):
            return self._extract_content_from_blocks(content)
        return str(content) if content else ""
