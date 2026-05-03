"""End-to-end flow tests for SDK, Proxy, and MCP modes.

Verifies that data flows correctly through the entire system:
- Provider name propagates from caller → compression → transport
- Multimodal content, tool_calls, metadata survive all boundaries
- Streaming mode preserves provider-specific fields
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from lattice.core.transport import Message, Request, Response
from lattice.integrations.mcp import LatticeMCPTools
from lattice.providers.transport import DirectHTTPProvider
from lattice.sdk import LatticeClient

# =============================================================================
# SDK Direct Mode — Compression API
# =============================================================================


class TestSDKCompressionAPI:
    """Unified compress() API replaces removed compress_request_async."""

    def test_compress_returns_result(self) -> None:
        client = LatticeClient()
        messages = [{"role": "user", "content": "Hello"}]
        result = client.compress(
            messages=messages,
            model="openai/gpt-4",
        )
        assert result.compressed_messages
        assert result.original_tokens > 0
        assert isinstance(result.transforms_applied, list)

    def test_compress_respects_mode_override(self) -> None:
        client = LatticeClient()
        messages = [
            {"role": "user", "content": "Hello world, this is a test message for compression."}
        ]
        result = client.compress(
            messages=messages,
            model="openai/gpt-4",
            mode="safe",
        )
        # safe mode should have fewer transforms than balanced/aggressive
        assert "reference_sub" in result.transforms_applied

    def test_compress_preserves_tool_calls(self) -> None:
        client = LatticeClient()
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "Sunny", "tool_call_id": "call_1"},
        ]
        result = client.compress(messages=messages, model="openai/gpt-4")
        assert len(result.compressed_messages) == 2
        assert result.compressed_messages[0].get("tool_calls") == messages[0]["tool_calls"]
        assert result.compressed_messages[1].get("tool_call_id") == "call_1"

    def test_compress_preserves_multimodal(self) -> None:
        client = LatticeClient()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's this?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                ],
            }
        ]
        result = client.compress(messages=messages, model="openai/gpt-4o")
        assert isinstance(result.compressed_messages[0]["content"], list)
        assert result.compressed_messages[0]["content"][1]["type"] == "image_url"

    def test_health_returns_summary(self) -> None:
        client = LatticeClient()
        health = client.health()
        assert health["status"] == "healthy"
        assert "transforms" in health


# =============================================================================
# MCP Mode — Serialization Integrity
# =============================================================================


class TestMCPSerialization:
    """MCP lattice_compress preserves all message fields."""

    def test_tool_calls_preserved(self) -> None:
        tools = LatticeMCPTools()
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "Sunny", "tool_call_id": "call_1"},
        ]
        result = tools.lattice_compress(messages, model="openai/gpt-4")
        assert "error" not in result
        assert result["runtime"]["tier"] == "SIMPLE"
        compressed = result["compressed_messages"]
        assert len(compressed) == 2
        assert compressed[0].get("tool_calls") == messages[0]["tool_calls"]
        assert compressed[1].get("tool_call_id") == "call_1"

    def test_multimodal_preserved(self) -> None:
        tools = LatticeMCPTools()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                ],
            }
        ]
        result = tools.lattice_compress(messages, model="openai/gpt-4o")
        assert "error" not in result
        compressed = result["compressed_messages"]
        assert isinstance(compressed[0]["content"], list)
        assert compressed[0]["content"][1]["type"] == "image_url"

    def test_provider_set_in_context(self) -> None:
        tools = LatticeMCPTools()
        messages = [{"role": "user", "content": "Hello"}]
        # Patch the pipeline to capture context.
        # lattice_compress uses a thread-pool when a running loop is detected
        # (common under pytest-asyncio).  Force the simple asyncio.run path
        # so the mock is visible.
        captured_context = None
        original_process = tools.pipeline.process

        async def _capture_process(request: Request, ctx: Any) -> Any:
            nonlocal captured_context
            captured_context = ctx
            return await original_process(request, ctx)

        with (
            patch.object(tools.pipeline, "process", side_effect=_capture_process),
            patch("asyncio.get_running_loop", side_effect=RuntimeError("no loop")),
        ):
            tools.lattice_compress(messages, model="groq/llama-3.1-70b")

        assert captured_context is not None
        assert captured_context.provider == "groq"
        assert captured_context.model == "groq/llama-3.1-70b"


# =============================================================================
# Proxy Mode — Provider Detection & Metadata
# =============================================================================


class TestProxyProviderDetection:
    """Proxy correctly detects provider from header and model prefix."""

    def test_detect_provider_from_header(self) -> None:
        from lattice.providers.transport import _resolve_provider_name

        assert _resolve_provider_name("gpt-4", provider_name="groq") == "groq"
        assert _resolve_provider_name("llama-3.1-70b", provider_name="groq") == "groq"

    def test_detect_provider_from_prefix(self) -> None:
        from lattice.providers.transport import _resolve_provider_name

        assert _resolve_provider_name("groq/llama-3.1-70b") == "groq"
        assert _resolve_provider_name("anthropic/claude-3-opus") == "anthropic"

    def test_detect_provider_fallback_raises(self) -> None:
        from lattice.core.errors import ProviderError
        from lattice.providers.transport import _resolve_provider_name

        with pytest.raises(ProviderError, match="Provider not specified"):
            _resolve_provider_name("gpt-4")
        with pytest.raises(ProviderError, match="Provider not specified"):
            _resolve_provider_name("llama-3.1-70b")


class TestProxySerialization:
    """Proxy request/response serialization preserves all fields."""

    def test_deserialize_multimodal_request(self) -> None:
        from lattice.proxy.server import _deserialize_openai_request

        body = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                    ],
                }
            ],
            "reasoning_effort": "high",
        }
        req = _deserialize_openai_request(body)
        assert len(req.messages) == 1
        parts = req.messages[0].content_parts
        assert len(parts) == 2
        assert req.metadata["reasoning_effort"] == "high"

    def test_serialize_messages_preserves_tool_calls(self) -> None:
        from lattice.proxy.server import _serialize_messages

        req = Request(
            messages=[
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "foo", "arguments": "{}"},
                        }
                    ],
                )
            ],
            model="gpt-4",
        )
        out = _serialize_messages(req)
        assert out[0]["tool_calls"] == req.messages[0].tool_calls

    def test_serialize_openai_response_preserves_reasoning(self) -> None:
        from lattice.proxy.server import _serialize_openai_response

        resp = Response(content="", metadata={"reasoning": "Let me think..."})
        req = Request(model="o1")
        body = _serialize_openai_response(resp, req)
        assert body["choices"][0]["message"]["reasoning_content"] == "Let me think..."


# =============================================================================
# Transport — Streaming Metadata Propagation
# =============================================================================


class TestTransportStreamingMetadata:
    """Streaming methods forward metadata to adapter serialization."""

    def test_completion_stream_accepts_metadata(self) -> None:
        import inspect

        sig = inspect.signature(DirectHTTPProvider.completion_stream)
        assert "metadata" in sig.parameters

    def test_completion_stream_with_stall_detect_accepts_metadata(self) -> None:
        import inspect

        sig = inspect.signature(DirectHTTPProvider.completion_stream_with_stall_detect)
        assert "metadata" in sig.parameters

    def test_build_request_preserves_metadata(self) -> None:
        req = DirectHTTPProvider._build_request(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            metadata={"reasoning_effort": "high"},
        )
        assert req.metadata["reasoning_effort"] == "high"


# =============================================================================
# Proxy — Non-streaming metadata forwarding
# =============================================================================


class TestProxyNonStreamingMetadata:
    """Proxy non-streaming path forwards compressed request metadata to transport."""

    def test_proxy_completion_calls_forward_metadata(self) -> None:
        """Verify metadata forwarding in proxy/compat non-streaming paths."""
        import inspect

        from lattice.gateway import compat as compat_mod
        from lattice.proxy import bootstrap as bootstrap_mod
        from lattice.proxy import server as server_mod

        source = (
            inspect.getsource(server_mod)
            + inspect.getsource(compat_mod)
            + inspect.getsource(bootstrap_mod)
        )
        # Verify that metadata=compressed_request.metadata appears in active implementation paths
        assert "metadata=compressed_request.metadata" in source
        # Verify speculative path also forwards metadata
        assert "metadata=req.metadata" in source


# =============================================================================
# Batching — Serialization integrity
# =============================================================================


class TestBatchingSerialization:
    """Batching engine preserves tool_calls, multimodal, and metadata."""

    @pytest.mark.asyncio
    async def test_batched_request_stores_tool_choice_and_stop(self) -> None:
        import asyncio

        from lattice.core.context import TransformContext
        from lattice.transforms.batching import BatchKey, PendingRequest

        key = BatchKey(model="gpt-4", temperature=0.7, max_tokens=100, top_p=1.0, stream=False)
        req = Request(
            messages=[Message(role="user", content="Hello")],
            model="gpt-4",
            tool_choice="auto",
            stop=["END"],
            metadata={"reasoning_effort": "high"},
        )
        pending = PendingRequest(
            request=req,
            context=TransformContext(),
            future=asyncio.get_event_loop().create_future(),
            enqueued_at=0.0,
        )

        from lattice.transforms.batching import BatchingEngine

        batched = BatchingEngine._build_batched_request(key, [pending])
        assert batched.metadata.get("tool_choice") == "auto"
        assert batched.metadata.get("stop") == ["END"]
        assert batched.metadata.get("request_metadata") == {"reasoning_effort": "high"}
