"""Ollama adapter.

Ollama serves models locally via an OpenAI-compatible endpoint at
``/v1/chat/completions`` *and* a native endpoint at ``/api/chat``.

We use the **native** endpoint because it is the source of truth for:
- ``thinking`` / reasoning content (Ollama 0.5+ with think models)
- native tool calling (Ollama 0.4+)
- ``images`` for multimodal
- ``format: json`` for structured output
- ``keep_alive`` for model persistence
- ``options`` passthrough

References
----------
- https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import json
from typing import Any

from lattice.core.transport import Message, Request, Response
from lattice.providers.openai import OpenAIAdapter


class OllamaAdapter:
    """Ollama direct adapter (native ``/api/chat`` endpoint)."""

    name = "ollama"
    _PREFIX = "ollama"

    def supports(self, model: str) -> bool:
        """Matches ``ollama/...`` but NOT ``ollama-cloud/...``."""
        prefix = model.split("/", 1)[0].lower() if "/" in model else ""
        return prefix == self._PREFIX

    def chat_endpoint(self, _model: str, base_url: str) -> str:
        # Ollama Cloud (and other hosted instances) expose an OpenAI-compatible
        # endpoint at /v1/chat/completions.  Local Ollama uses the native
        # /api/chat endpoint.  We decide by looking at the base URL.
        if "ollama.com" in base_url or base_url.startswith("https://"):
            return f"{base_url.rstrip('/')}/v1/chat/completions"
        return f"{base_url.rstrip('/')}/api/chat"

    def auth_headers(self, _api_key: str | None) -> dict[str, str]:
        """Ollama usually requires no auth (local)."""
        return {}

    def map_model_name(self, model: str) -> str:
        """Strip the ``ollama/`` prefix."""
        return self._strip_prefix(model)

    def extra_headers(self, _request: Any) -> dict[str, str]:
        return {}

    def detect(self, signals: Any) -> Any:
        """Detect Ollama from explicit signals.

        Ollama is a **local** inference server.  It has no universal auth
        pattern, no unique endpoint paths, and no exclusive model names.
        The only reliable signals are:

        1. Explicit body/header naming the provider.
        2. Model prefix ``ollama/...``.
        3. The ``x-lattice-provider: ollama`` header (low confidence).
        """
        from lattice.gateway.detect_helpers import (
            detect_explicit,
            detect_model_prefix,
            highest_confidence,
        )

        return highest_confidence(
            self.name,
            detect_explicit(signals, self.name, aliases={self._PREFIX}),
            detect_model_prefix(signals, self.name, aliases={self._PREFIX}),
        )

    def retry_config(self) -> dict[str, Any]:
        return {
            "max_retries": 3,
            "backoff_factor": 1.0,
            "retry_on": (429, 502, 503, 504),
        }

    # ------------------------------------------------------------------
    # Model name
    # ------------------------------------------------------------------

    def _strip_prefix(self, model: str) -> str:
        """Remove the ``ollama/`` provider prefix for the wire."""
        if model.lower().startswith("ollama/"):
            return model[len("ollama/") :]
        return model

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize_request(self, request: Request) -> dict[str, Any]:
        """Internal Request → Ollama native JSON body."""
        body: dict[str, Any] = {
            "model": self._strip_prefix(request.model),
            "messages": self._build_messages(request.messages),
            "stream": request.stream,
        }

        # Options
        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.stop:
            options["stop"] = request.stop
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        # Deterministic sampling via seed (useful for benchmarking)
        seed = request.metadata.get("seed") or request.extra_body.get("seed")
        if seed is not None:
            options["seed"] = int(seed)
        if options:
            body["options"] = options

        # Tools
        if request.tools:
            body["tools"] = self._build_tools(request.tools)

        # Format mapping: json_object → "json", json_schema → schema dict
        fmt = self._resolve_format(request)
        if fmt is not None:
            body["format"] = fmt

        # keep_alive (from metadata or extra_body)
        keep = request.extra_body.get("keep_alive")
        if keep is None:
            keep = request.metadata.get("keep_alive")
        if keep is not None:
            body["keep_alive"] = keep

        # reasoning_effort → think (Ollama gpt-oss style)
        think = request.metadata.get("think") or request.extra_body.get("think")
        if think is not None:
            if isinstance(think, str):
                body["think"] = think in {"low", "medium", "high"}
            else:
                body["think"] = bool(think)

        return body

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            m: dict[str, Any] = {"role": str(msg.role)}

            # Extract reasoning/thinking content if present
            reasoning = msg.metadata.get("thinking") or msg.metadata.get("reasoning_content")
            content = msg.content
            if reasoning:
                m["thinking"] = str(reasoning)
            if content:
                m["content"] = str(content)

            # Images (base64 extracted from content blocks)
            images = _extract_image_urls(msg)
            if images:
                m["images"] = images

            # Tool calls
            if msg.tool_calls:
                m["tool_calls"] = _build_ollama_tool_calls(msg.tool_calls)

            out.append(m)

        # Content cleanup: if any message has no "content", Ollama still
        # expects the key with empty string.
        for m in out:
            if "content" not in m:
                m["content"] = ""

        return out

    # ------------------------------------------------------------------
    # Format resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_format(request: Request) -> str | dict[str, Any] | None:
        rf = request.metadata.get("response_format")
        if rf is None:
            return None
        t = rf.get("type")
        if t == "json_object":
            return "json"
        if t == "json_schema":
            schema = rf.get("json_schema", {}).get("schema")
            return schema or "json"
        return None

    # ------------------------------------------------------------------
    # Tool building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Map OpenAI tools to Ollama native tool format."""
        out: list[dict[str, Any]] = []
        for t in tools:
            if t.get("type") != "function":
                continue
            fn = t.get("function", {})
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {"type": "object"}),
                    },
                }
            )
        return out

    # ------------------------------------------------------------------
    # Deserialization
    # ------------------------------------------------------------------

    def deserialize_response(self, data: dict[str, Any]) -> Response:
        """Ollama native JSON → internal Response."""
        msg = data.get("message", {})
        content = self.extract_content(msg)
        thinking = msg.get("thinking") or ""

        usage: dict[str, int] = {}
        if data.get("prompt_eval_count"):
            usage["prompt_tokens"] = data["prompt_eval_count"]
        if data.get("eval_count"):
            usage["completion_tokens"] = data["eval_count"]
        total = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        if total:
            usage["total_tokens"] = total

        # Tool calls from message
        tool_calls: list[dict[str, Any]] | None = None
        raw_tool_calls = msg.get("tool_calls")
        if raw_tool_calls:
            tool_calls = _normalize_tool_calls(raw_tool_calls)

        resp = Response(
            content=content,
            role=msg.get("role", "assistant"),
            model=data.get("model", ""),
            usage=usage,
            finish_reason="stop" if data.get("done") else None,
            tool_calls=tool_calls,
        )
        if thinking:
            resp.metadata["thinking"] = thinking
        return resp

    # ------------------------------------------------------------------
    # Streaming normalisation
    # ------------------------------------------------------------------

    def normalize_sse_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """Normalise Ollama NDJSON stream chunk to OpenAI delta format."""
        msg = chunk.get("message", {})
        content = msg.get("content") if msg else None
        msg.get("thinking") if msg else None
        done = chunk.get("done", False)
        done_reason = chunk.get("done_reason")

        # Skip thinking content (user should not see reasoning tokens)
        text = content if isinstance(content, str) and content else ""
        if not text and not done:
            return None

        delta: dict[str, Any] = {}
        if text:
            delta["content"] = text

        # Map finish reason — always emit finish_reason on done=true
        finish: str | None = None
        if done:
            finish = _map_ollama_done_reason(done_reason)
            # Ollama may send no done_reason but done=true on normal stop
            if finish is None:
                finish = "stop"

        return {
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish,
                }
            ],
        }

    # ------------------------------------------------------------------
    # Content extraction
    # ------------------------------------------------------------------

    def extract_content(self, msg: dict[str, Any]) -> str:
        content = msg.get("content") or ""
        if not content:
            # Ollama 0.5+ stores reasoning in a separate "thinking" key
            # during streaming or when the model thinks but produces no output
            thinking = msg.get("thinking") or ""
            if thinking:
                return thinking
        return content


# =============================================================================
# Shared helpers
# =============================================================================


def _extract_image_urls(msg: Message) -> list[str]:
    """Extract base64 image URLs from a Message.

    Handles:
    1. Message.metadata["images"] = ["base64_1", "base64_2"]
    2. Message.content with image_url blocks (OpenAI multimodal format)
    """
    if msg.metadata.get("images"):
        images = msg.metadata.get("images")
        if isinstance(images, list):
            return [str(i) for i in images if isinstance(i, (str, bytes))]

    # OpenAI multimodal format: content = [ {"type":"text","text":"..."},
    #                                      {"type":"image_url","image_url":{"url":"data:..."}} ]
    # We support this by probing content for dict-like blocks.
    content = msg.content
    if not isinstance(content, str):
        try:
            blocks = list(content) if content else []
            extracted: list[str] = []
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    if url and url.startswith("data:"):
                        extracted.append(url)
            return extracted
        except Exception:
            pass
    return []


def _build_ollama_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map OpenAI tool_calls to Ollama's native tool call format."""
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        if tc.get("type") == "function":
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": fn.get("name", ""),
                        "arguments": args,
                    },
                }
            )
    return out


def _normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalise Ollama tool_calls to OpenAI format."""
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        if tc.get("type") == "function":
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, dict):
                args = json.dumps(args)
            out.append(
                {
                    "id": "",  # Ollama doesn't provide tool call ids
                    "type": "function",
                    "function": {
                        "name": fn.get("name", ""),
                        "arguments": args,
                    },
                }
            )
    return out or None  # type: ignore[return-value]


def _map_ollama_done_reason(reason: str | None) -> str | None:
    if not reason:
        return "stop"
    mappings = {
        "stop": "stop",
        "length": "length",
        "load": None,
    }
    return mappings.get(reason, reason)


class OllamaCloudAdapter(OpenAIAdapter):
    """Ollama Cloud — OpenAI-compatible API format with ollama-cloud base URL."""

    name = "ollama-cloud"
    _PREFIXES = {"ollama-cloud"}

    def supports(self, model: str) -> bool:
        """Matches ``ollama-cloud/...``."""
        prefix = model.split("/", 1)[0].lower() if "/" in model else ""
        return prefix == "ollama-cloud"

    def detect(self, signals: Any) -> Any:
        """Detect Ollama Cloud from explicit signals only.

        Ollama Cloud has no unique auth pattern or endpoint path.
        The only reliable signals are explicit declarations and the
        ``ollama-cloud/`` model prefix.
        """
        from lattice.gateway.detect_helpers import (
            detect_explicit,
            detect_model_prefix,
            highest_confidence,
        )

        return highest_confidence(
            self.name,
            detect_explicit(signals, self.name, aliases=self._PREFIXES),
            detect_model_prefix(signals, self.name, aliases=self._PREFIXES),
        )
