"""AWS Bedrock adapter.

Bedrock uses the AWS **Converse** API. Because it requires boto3 / AWS SigV4
signing, a full implementation is substantially complex.  We provide a
production-grade adapter that serializes to Converse format and expects the
caller to handle signing (or we handle SigV4 in a future phase).

Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
"""

from __future__ import annotations

from typing import Any

from lattice.core.transport import Request, Response
from lattice.providers.tool_sanitizer import BedrockToolSanitizer


class BedrockAdapter:
    """AWS Bedrock Converse adapter.

    Serializes to the ``Converse`` request format.  SigV4 signing is NOT
    implemented here — callers should use ``boto3`` or an AWS-signed
    HTTP client in production.
    """

    name = "bedrock"
    _PREFIXES = {"bedrock"}

    def supports(self, model: str) -> bool:
        prefix = model.split("/", 1)[0].lower() if "/" in model else ""
        return prefix in self._PREFIXES

    def chat_endpoint(self, model: str, base_url: str) -> str:
        """Converse endpoint.

        ``model`` is the Bedrock model ARN, e.g.
        ``bedrock/anthropic.claude-3-sonnet-20240229-v1:0``
        """
        model_id = model.split("/", 1)[1] if "/" in model else model
        return f"{base_url.rstrip('/')}/model/{model_id}/converse"

    def auth_headers(self, _api_key: str | None) -> dict[str, str]:
        """SigV4 signing handled externally."""
        return {}

    def map_model_name(self, model: str) -> str:
        """Strip the ``bedrock/`` prefix."""
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

    def serialize_request(self, request: Request) -> dict[str, Any]:
        """OpenAI Request → Bedrock Converse format.

        References:
        https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        """
        messages_raw: list[dict[str, Any]] = [
            {
                "role": str(msg.role),
                "content": [{"text": msg.content}],
            }
            for msg in request.messages
        ]
        # Extract system message
        system_text = None
        messages = []
        for msg in messages_raw:
            if msg["role"] == "system" and system_text is None:
                system_text = msg["content"][0]["text"]
            else:
                # Bedrock uses "user" and "assistant" only
                role = msg["role"]
                if role == "tool":
                    role = "user"  # tool results sent as user content
                messages.append({"role": role, "content": msg["content"]})

        body: dict[str, Any] = {
            "messages": messages,
            "inferenceConfig": {},
        }
        if system_text:
            body["system"] = [{"text": system_text}]
            if self._cache_point_enabled(request):
                body["system"].append(self._cache_point_block())
        if request.temperature is not None:
            body["inferenceConfig"]["temperature"] = request.temperature
        if request.top_p is not None:
            body["inferenceConfig"]["topP"] = request.top_p
        if request.max_tokens is not None:
            body["inferenceConfig"]["maxTokens"] = request.max_tokens
        if request.stop:
            body["inferenceConfig"]["stopSequences"] = request.stop
        if request.tools:
            body["toolConfig"] = {
                "tools": [
                    {"toolSpec": self._remap_tool(t)} for t in request.tools
                ]
            }
            if request.tool_choice is not None:
                body["toolConfig"]["toolChoice"] = self._remap_tool_choice(request.tool_choice)

        # Bedrock Claude models support thinking via additionalModelRequestFields
        thinking = request.metadata.get("thinking") or request.extra_body.get("thinking")
        if thinking is not None:
            body["additionalModelRequestFields"] = {"thinking": thinking}

        return body

    @staticmethod
    def _remap_tool(tool: dict[str, Any]) -> dict[str, Any]:
        """OpenAI tool → Bedrock toolSpec with sanitized names."""
        fn = tool.get("function", {})
        name = BedrockToolSanitizer.sanitize_tool_name(fn.get("name", ""))
        return {
            "name": name,
            "description": fn.get("description", ""),
            "inputSchema": {"json": fn.get("parameters", {})},
        }

    @staticmethod
    def _remap_tool_choice(tool_choice: str | dict[str, Any]) -> dict[str, Any]:
        """Map OpenAI tool_choice to Bedrock toolChoice."""
        if tool_choice == "none":
            return {"none": {}}
        if tool_choice == "required":
            return {"any": {}}
        if isinstance(tool_choice, dict):
            # Specific function
            fn_name = tool_choice.get("function", {}).get("name", "")
            if fn_name:
                return {"tool": {"name": BedrockToolSanitizer.sanitize_tool_name(fn_name)}}
        # Default auto
        return {"auto": {}}

    def deserialize_response(self, data: dict[str, Any]) -> Response:
        """Bedrock Converse response → internal Response."""
        output = data.get("output", {})
        msg = output.get("message", {})
        content_blocks = msg.get("content", [])
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] | None = None
        for block in content_blocks:
            if "text" in block:
                text_parts.append(block["text"])
            if "toolUse" in block:
                if tool_calls is None:
                    tool_calls = []
                tu = block["toolUse"]
                tool_calls.append(
                    {
                        "id": tu.get("toolUseId", ""),
                        "type": "function",
                        "function": {
                            "name": tu.get("name", ""),
                            "arguments": tu.get("input", {}),
                        },
                    }
                )

        usage = data.get("usage", {})
        model_id = data.get("modelId", "")
        prompt_tokens = usage.get("inputTokens", 0)
        completion_tokens = usage.get("outputTokens", 0)
        total_tokens = usage.get("totalTokens", 0)
        cached_tokens = usage.get("cacheReadInputTokens", 0)
        cache_creation_tokens = usage.get("cacheWriteInputTokens", 0)
        response_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        if cached_tokens:
            response_usage["cached_tokens"] = cached_tokens
        if cache_creation_tokens:
            response_usage["cache_creation_input_tokens"] = cache_creation_tokens
        return Response(
            content="".join(text_parts),
            role=msg.get("role", "assistant"),
            model=model_id,
            usage=response_usage,
            finish_reason=self._map_finish_reason(data.get("stopReason")),
            tool_calls=tool_calls,
        )

    @staticmethod
    def _cache_point_block() -> dict[str, Any]:
        return {"cachePoint": {"type": "default"}}

    @staticmethod
    def _cache_point_enabled(request: Request) -> bool:
        if request.metadata.get("bedrock_prompt_caching"):
            return True
        cache_arbitrage = request.metadata.get("_cache_arbitrage")
        if not isinstance(cache_arbitrage, dict):
            return False
        annotations = cache_arbitrage.get("annotations")
        if not isinstance(annotations, dict):
            return False
        cache = annotations.get("cache")
        provider = annotations.get("provider")
        return (
            provider == "bedrock"
            and isinstance(cache, dict)
            and cache.get("mode") == "explicit_breakpoint"
        )

    @staticmethod
    def _map_finish_reason(stop_reason: str | None) -> str | None:
        """Map Bedrock stopReason to OpenAI finish_reason values."""
        if stop_reason is None:
            return None
        mapping = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
            "content_filtered": "content_filter",
        }
        return mapping.get(stop_reason, stop_reason)

    def normalize_sse_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize Bedrock ConverseStream chunk to OpenAI delta format.

        Bedrock ConverseStream produces events like:
        - {"messageStart": {"role": "assistant"}}
        - {"contentBlockDelta": {"delta": {"text": "..."}}}
        - {"messageStop": {"stopReason": "end_turn"}}
        """
        if "contentBlockDelta" in chunk:
            delta = chunk["contentBlockDelta"].get("delta", {})
            if "text" in delta:
                return {
                    "choices": [{"delta": {"content": delta["text"]}, "index": 0}],
                }
            if "toolUse" in delta:
                tu = delta["toolUse"]
                return {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": chunk["contentBlockDelta"].get("contentBlockIndex", 0),
                                "id": tu.get("toolUseId", ""),
                                "type": "function",
                                 "function": {
                                     "name": tu.get("name", ""),
                                     "arguments": tu.get("input", ""),
                                 },
                            }]
                        },
                        "index": 0,
                    }],
                }
        if "messageStop" in chunk:
            stop_reason = chunk["messageStop"].get("stopReason")
            return {
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": self._map_finish_reason(stop_reason),
                        "index": 0,
                    }
                ],
            }
        if "metadata" in chunk:
            usage = chunk["metadata"].get("usage", {})
            normalized_usage = {
                "prompt_tokens": usage.get("inputTokens", 0),
                "completion_tokens": usage.get("outputTokens", 0),
                "total_tokens": usage.get("totalTokens", 0),
            }
            cached_tokens = usage.get("cacheReadInputTokens", 0)
            cache_creation_tokens = usage.get("cacheWriteInputTokens", 0)
            if cached_tokens:
                normalized_usage["cached_tokens"] = cached_tokens
            if cache_creation_tokens:
                normalized_usage["cache_creation_input_tokens"] = cache_creation_tokens
            return {
                "choices": [],
                "usage": normalized_usage,
            }
        # Ignore other event types (messageStart, contentBlockStart, etc.)
        return None

    def extract_content(self, msg: dict[str, Any]) -> str:
        content = msg.get("content") or ""
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
            return "".join(parts)
        return content
