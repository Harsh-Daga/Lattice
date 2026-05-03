"""OpenAI adapter — identity pass-through.

OpenAI is the reference API: our internal `Request` / `Response` objects
are already shaped like OpenAI Chat Completions, so this adapter is
mostly a no-op.

Reference: https://platform.openai.com/docs/api-reference/chat
"""

from __future__ import annotations

from typing import Any

from lattice.core.transport import Request, Response


class OpenAIAdapter:
    """OpenAI-compatible adapter (also usable for any service that
    speaks the OpenAI chat-completions schema, e.g. custom proxies).
    """

    name = "openai"

    # Provider prefixes that route to this adapter
    _PREFIXES = {"openai"}

    def supports(self, model: str) -> bool:
        """Matches explicit OpenAI-compatible prefixes only."""
        if "/" not in model:
            return False
        prefix = model.split("/", 1)[0].lower()
        return prefix in self._PREFIXES

    def map_model_name(self, model: str) -> str:
        """Strip the ``openai/`` or ``azure/`` prefix if present.

        The caller is responsible for providing the exact model ID the
        provider expects.  We do not maintain provider-native aliases here.
        """
        if "/" in model:
            prefix, rest = model.split("/", 1)
            if prefix.lower() in self._PREFIXES:
                return rest
        return model

    def chat_endpoint(self, _model: str, base_url: str) -> str:
        """Path for chat completions."""
        return f"{base_url.rstrip('/')}/v1/chat/completions"

    def auth_headers(self, api_key: str | None) -> dict[str, str]:
        h: dict[str, str] = {}
        if api_key:
            h["Authorization"] = api_key if api_key.startswith("Bearer ") else f"Bearer {api_key}"
        return h

    def extra_headers(self, _request: Request) -> dict[str, str]:
        return {}

    def retry_config(self) -> dict[str, Any]:
        return {
            "max_retries": 3,
            "backoff_factor": 1.0,
            "retry_on": (429, 502, 503, 504),
        }

    def detect(self, signals: Any) -> Any:
        """Detect OpenAI from explicit signals only.

        OpenAI's auth format (``Bearer sk-...``) is shared with dozens of
        OpenAI-compatible providers, so we NEVER match on auth alone.
        Paths such as ``/v1/chat/completions`` are also shared, so we never
        match on path alone.  Only explicit body/header fields or the
        ``openai/`` model prefix are considered reliable signals.
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

    def serialize_request(self, request: Request) -> dict[str, Any]:
        """Internal Request → OpenAI JSON body."""
        body: dict[str, Any] = {
            "model": request.model,
            "messages": self._serialize_messages(request.messages),
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.tools:
            body["tools"] = request.tools
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice
        if request.stream:
            body["stream"] = True
        if request.stop:
            body["stop"] = request.stop

        # JSON mode / structured output
        response_format = request.metadata.get("response_format") or request.extra_body.get(
            "response_format"
        )
        if response_format is not None:
            body["response_format"] = response_format

        # Reasoning effort for o1/o3
        reasoning_effort = request.metadata.get("reasoning_effort") or request.extra_body.get(
            "reasoning_effort"
        )
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort

        # Prompt caching controls. These are explicit provider knobs, not
        # inferred by the serializer: callers/proxy planning must provide the
        # stable cache key or retention policy.
        prompt_cache_key = (
            request.metadata.get("prompt_cache_key")
            or request.metadata.get("openai_prompt_cache_key")
            or request.extra_body.get("prompt_cache_key")
        )
        if prompt_cache_key is not None:
            body["prompt_cache_key"] = prompt_cache_key

        prompt_cache_retention = (
            request.metadata.get("prompt_cache_retention")
            or request.metadata.get("openai_prompt_cache_retention")
            or request.extra_body.get("prompt_cache_retention")
        )
        if prompt_cache_retention is not None:
            body["prompt_cache_retention"] = prompt_cache_retention

        # Vision / image detail
        image_detail = request.metadata.get("image_detail")
        if image_detail is not None:
            body["image_detail"] = image_detail

        return body

    @staticmethod
    def _serialize_messages(messages: list[Any]) -> list[dict[str, Any]]:
        """Serialize messages, preserving multimodal content blocks."""
        from lattice.protocol.content import ImagePart, TextPart

        out: list[dict[str, Any]] = []
        for msg in messages:
            m: dict[str, Any] = {
                "role": str(msg.role),
            }
            # If content_parts has image blocks, serialize as multimodal
            content_parts = getattr(msg, "content_parts", None)
            if content_parts and len(content_parts) > 1:
                parts: list[dict[str, Any]] = []
                for part in content_parts:
                    if isinstance(part, TextPart):
                        parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImagePart):
                        if part.source.type.value == "url":
                            parts.append(
                                {"type": "image_url", "image_url": {"url": part.source.data}}
                            )
                        else:
                            parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{part.source.media_type};base64,{part.source.data}",
                                    },
                                }
                            )
                    else:
                        # Fallback: serialize as dict
                        parts.append(part.to_dict())
                m["content"] = parts
            elif content_parts and len(content_parts) == 1:
                part = content_parts[0]
                if isinstance(part, ImagePart):
                    if part.source.type.value == "url":
                        m["content"] = [
                            {"type": "image_url", "image_url": {"url": part.source.data}}
                        ]
                    else:
                        m["content"] = [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{part.source.media_type};base64,{part.source.data}",
                                },
                            }
                        ]
                elif isinstance(part, TextPart):
                    m["content"] = part.text
                else:
                    m["content"] = part.to_dict()
            else:
                m["content"] = msg.content

            if msg.name:
                m["name"] = msg.name
            if msg.tool_calls:
                m["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            out.append(m)
        return out

    def deserialize_response(self, data: dict[str, Any]) -> Response:
        """OpenAI JSON → internal Response.

        Handles standard content, reasoning content (o1/o3/DeepSeek),
        refusal messages, and tool calls.
        """
        choice = data.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = self.extract_content(msg)
        usage = self._normalize_usage(data.get("usage", {}))
        resp = Response(
            content=content,
            role=msg.get("role", "assistant"),
            model=data.get("model", ""),
            usage=usage,
            finish_reason=choice.get("finish_reason"),
            tool_calls=msg.get("tool_calls"),
        )
        # Preserve reasoning content in metadata
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if reasoning:
            resp.metadata["reasoning"] = reasoning
        # Preserve refusal in metadata
        refusal = msg.get("refusal")
        if refusal:
            resp.metadata["refusal"] = refusal
        return resp

    @staticmethod
    def _normalize_usage(usage: dict[str, Any]) -> dict[str, Any]:
        out = dict(usage) if isinstance(usage, dict) else {}
        details = out.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached_tokens = details.get("cached_tokens")
            if isinstance(cached_tokens, int) and cached_tokens:
                out.setdefault("cached_tokens", cached_tokens)
        return out

    def normalize_sse_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """Pass-through — OpenAI chunks are already in target format."""
        # Filter out the final empty-chunk-with-usage
        if chunk.get("choices") == []:
            return None
        return chunk

    def extract_content(self, msg: dict[str, Any]) -> str:
        """Extract text from an OpenAI message dict.

        Handles ``content`` (standard), ``reasoning_content`` (OpenAI o1/o3),
        and ``reasoning`` (DeepSeek / Ollama thinking models).
        Falls back to reasoning only when content is empty.
        """
        content = msg.get("content") or ""
        if not content:
            content = msg.get("reasoning_content") or ""
        if not content and "reasoning" in msg:
            content = msg["reasoning"] or ""
        return content
