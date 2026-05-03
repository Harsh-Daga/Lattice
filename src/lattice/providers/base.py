"""Provider adapter protocol for the LATTICE transport layer.

Every supported LLM provider implements `ProviderAdapter`. The adapter
is responsible for translating between LATTICE's internal `Request` /
`Response` objects and the provider's native HTTP wire format.

Design Principles
-----------------
* **Explicit over magic:** Each provider's quirks are encoded in its own
  adapter, not hidden behind a universal dispatcher.
* **Testable in isolation:** Adapters are pure functions (no network calls).
  The `DirectHTTPProvider` manages connections and delegates to adapters.
* **Streaming-aware:** Adapters normalize SSE / NDJSON / event-stream chunks
  into a uniform OpenAI-compatible delta format.
* **Auth-agnostic:** Each adapter declares its own header injection strategy.

Reference Implementations
-------------------------
* OpenRouter normalizes everything to OpenAI chat and translates at the edge.
* FreeRouter owns tool-call translation Anthropic ↔ OpenAI in ~200 LOC.

Our approach: each adapter is ~50–150 LOC, fully typed, unit-tested,
and directly wired into `DirectHTTPProvider`.
"""

from __future__ import annotations

from typing import Any, Protocol

from lattice.core.transport import Request, Response

# =============================================================================
# ProviderAdapter protocol
# =============================================================================


class ProviderAdapter(Protocol):
    """Transform OpenAI-shaped Request ↔ provider-native HTTP.

    Routing Philosophy
    ------------------
    * **Explicit provider wins:** Callers pass ``provider_name`` explicitly.
    * **Prefix requirement:** If no provider is given, parse ``provider/model``.
    * **No bare-model heuristics:** We NEVER guess a provider from a bare
      model name (e.g. ``gpt-4`` does NOT mean OpenAI — it could be on
      OpenRouter, Together, Groq, etc.).
    * **Alias mapping:** ``map_model_name`` strips the provider prefix and
      applies provider-native aliases (e.g. ``gpt-4o`` → ``gpt-4o-2024-08-06``).
    """

    @property
    def name(self) -> str: ...

    def supports(self, model: str) -> bool:
        """Return True if this adapter can handle the model string.

        Matches the ``provider/`` prefix (e.g. ``groq/llama-3.1``).
        OpenAIAdapter does not match bare model names.
        """
        ...

    def map_model_name(self, model: str) -> str:
        """Map a user-facing model name to the provider's native name.

        Steps:
        1. Strip the ``provider/`` prefix if present.
        2. Apply provider-specific aliases (sub-class responsibility).

        This method must be called by the transport layer before
        ``serialize_request`` so the provider sees its native model ID.
        """
        ...

    def chat_endpoint(self, model: str, base_url: str) -> str:
        """Return the full URL path for chat completions."""
        ...

    def auth_headers(self, api_key: str | None) -> dict[str, str]:
        """Provider-specific auth headers."""
        ...

    def extra_headers(self, request: Request) -> dict[str, str]:
        """Provider-specific extra headers for a request.

        Called after ``auth_headers`` to inject headers like
        ``HTTP-Referer`` (OpenRouter), routing hints, etc.
        """
        ...

    def serialize_request(self, request: Request) -> dict[str, Any]:
        """Internal Request → provider-native JSON body."""
        ...

    def deserialize_response(self, data: dict[str, Any]) -> Response:
        """Provider JSON body → internal Response."""
        ...

    def normalize_sse_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize one streaming chunk to OpenAI delta format.

        Returns ``None`` when the chunk is a provider-side control event
        (e.g. SSE comment ``: ping``).
        """
        ...

    def extract_content(self, msg: dict[str, Any]) -> str:
        """Best-effort content extraction from a provider message dict.

        Handles field-name differences such as ``content`` vs ``reasoning``
        vs ``text``.
        """
        ...

    def detect(self, signals: Any) -> Any:
        """Detect whether this adapter should handle *signals*.

        Returns a :class:`~lattice.gateway.routing.DetectionResult` with a
        confidence level.  :data:`~lattice.gateway.routing.DetectionConfidence.NONE`
        means "this adapter does not believe the request belongs to me".

        Every concrete adapter MUST implement this method.  There are no
        defaults — a missing ``detect()`` raises :exc:`AttributeError`.
        """
        ...

    def retry_config(self) -> dict[str, Any]:
        """Return retry policy for this provider.

        Keys:
        * ``max_retries`` — integer (default 3)
        * ``backoff_factor`` — float multiplier (default 1.0)
        * ``retry_on`` — tuple of HTTP status codes
        """
        ...


# =============================================================================
# Shared helpers (all adapters can import these)
# =============================================================================


def _pop_system(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract the first system message from an OpenAI message list.

    Returns ``(system_text, remaining_messages)``.
    Used by Anthropic, Gemini, and Bedrock adapters.
    """
    system_text: str | None = None
    remaining: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system" and system_text is None:
            system_text = msg.get("content", "")
        else:
            remaining.append(msg)
    return system_text, remaining


def _remap_tool_choice(tool_choice: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
    """Remap OpenAI ``tool_choice`` to Anthropic-style if needed.

    * ``"auto"`` → ``"auto"``
    * ``"none"`` → ``"none"``
    * ``{"type": "function", "function": {"name": "x"}}`` →
      ``{"type": "tool", "name": "x"}``
    """
    if tool_choice is None or tool_choice in ("auto", "none"):
        return tool_choice
    if isinstance(tool_choice, dict):
        func = tool_choice.get("function", {})
        if func and "name" in func:
            return {"type": "tool", "name": func["name"]}
        return tool_choice
    return tool_choice


def _remap_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remap OpenAI ``tools[].function.parameters`` to
    ``tools[].input_schema`` (Anthropic-style).

    Other adapters may apply additional remapping afterwards.
    """
    out: list[dict[str, Any]] = []
    for t in tools:
        copy: dict[str, Any] = dict(t)  # shallow copy
        if "function" in copy:
            fn = copy.pop("function")
            copy["name"] = fn.get("name", "")
            copy["description"] = fn.get("description", "")
            copy["input_schema"] = fn.get("parameters", {})
            if "input_schema" not in copy or not copy["input_schema"]:
                # fallback — some providers require the key
                copy["input_schema"] = {"type": "object", "properties": {}}
        out.append(copy)
    return out


def _strip_provider_prefix(model: str, prefixes: set[str]) -> str:
    """Strip a provider prefix from a model string.

    Examples::

        >>> _strip_provider_prefix("groq/llama-3.1", {"groq"})
        'llama-3.1'
        >>> _strip_provider_prefix("gpt-4", {"groq"})
        'gpt-4'
    """
    if "/" not in model:
        return model
    prefix, rest = model.split("/", 1)
    if prefix.lower() in prefixes:
        return rest
    return model


def _format_sse_event(event: str | None, data: str) -> str:
    """Format a single SSE event line.

    If *event* is given its ``event:`` line is emitted before ``data:``.
    """
    lines: list[str] = []
    if event:
        lines.append(f"event: {event}")
    # Split data on newlines so multi-line JSON gets multiple data: lines
    for line in data.splitlines():
        lines.append(f"data: {line}")
    lines.append("")
    return "\n".join(lines)
