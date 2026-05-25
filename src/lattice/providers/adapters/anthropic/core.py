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

from typing import Any

from lattice.providers.tool_sanitizer import (
    AnthropicToolSanitizer,
)


class AnthropicAdapterCore:
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
        prefix = model.split("/", 1)[0].lower() if "/" in model else model.split("-", 1)[0].lower()
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

    def detect(self, signals: Any) -> Any:
        """Detect Anthropic from strong, provider-specific signals.

        Anthropic has several **unambiguous** signals:

        1. **Explicit** — body field ``provider=anthropic`` or header
           ``x-lattice-provider=anthropic``.
        2. **Auth** — ``Authorization: Bearer sk-ant-*`` (Anthropic-specific key
           prefix; no other provider uses ``sk-ant-``).
        3. **Header** — ``anthropic-version`` (Anthropic-specific API version
           header; no other provider sends this).
        4. **Path** — ``/v1/messages`` or ``/v1/messages/count_tokens``
           (Anthropic Messages API endpoints).
        5. **Model prefix** — ``anthropic/...`` or bare ``claude-...``
           (the ``claude-`` family is exclusive to Anthropic).

        Returns
        -------
        DetectionResult with confidence level.  ``NONE`` when no signal matches.
        """
        import re

        from lattice.gateway.detect_helpers import (
            detect_auth_pattern,
            detect_explicit,
            detect_header_present,
            detect_model_prefix,
            detect_path,
            highest_confidence,
        )
        from lattice.gateway.routing import DetectionConfidence, DetectionResult

        # Auth: sk-ant-* is unique to Anthropic
        auth_result = detect_auth_pattern(
            signals,
            self.name,
            re.compile(r"Bearer\s+sk-ant-"),
            "Authorization header matches Anthropic sk-ant-* pattern",
        )

        # Path: /v1/messages is Anthropic-specific
        path_result = detect_path(
            signals,
            self.name,
            {"/v1/messages", "/v1/messages/count_tokens"},
            "request path is Anthropic Messages API endpoint",
        )

        # Header: anthropic-version is Anthropic-specific
        version_result = detect_header_present(
            signals,
            self.name,
            "anthropic-version",
            "anthropic-version header is Anthropic-specific",
        )

        # Header: x-api-key is used by Anthropic
        x_api_key_result = detect_header_present(
            signals,
            self.name,
            "x-api-key",
            "x-api-key header is Anthropic-specific",
        )

        # Model: anthropic/ prefix or bare claude- name
        model_result = detect_model_prefix(signals, self.name, aliases=self._PREFIXES)
        if (
            model_result.confidence == DetectionConfidence.NONE
            and signals.model
            and signals.model.lower().startswith("claude-")
        ):
            model_result = DetectionResult(
                provider=self.name,
                confidence=DetectionConfidence.MODEL,
                reason="bare 'claude-' model name is exclusive to Anthropic",
                detail={"model": signals.model},
            )

        return highest_confidence(
            self.name,
            detect_explicit(signals, self.name, aliases=self._PREFIXES),
            auth_result,
            path_result,
            version_result,
            x_api_key_result,
            model_result,
        )

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
            h["Authorization"] = api_key if api_key.startswith("Bearer ") else f"Bearer {api_key}"
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

