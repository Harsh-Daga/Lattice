"""End-to-end serialization tests.

Verifies that data flows correctly through the entire pipeline:
- dict → Message → dict roundtrips
- multimodal content is preserved
- tool_calls and tool_call_id are preserved
- SDK → DirectHTTPProvider → Adapter serialization chain works
"""

from __future__ import annotations

from lattice.core.serialization import (
    message_from_dict,
    message_to_dict,
    request_from_dict,
    request_to_dict,
    response_to_dict,
)
from lattice.core.transport import Message, Request, Response
from lattice.protocol.content import ImagePart, ImageSource, ImageSourceType, TextPart
from lattice.providers.transport import DirectHTTPProvider

# =============================================================================
# Message roundtrip
# =============================================================================


class TestMessageRoundtrip:
    """dict → Message → dict preserves all fields."""

    def test_simple_text(self) -> None:
        d = {"role": "user", "content": "Hello"}
        msg = message_from_dict(d)
        assert msg.role == "user"
        assert msg.content == "Hello"
        out = message_to_dict(msg)
        assert out["role"] == "user"
        assert out["content"] == "Hello"

    def test_with_name(self) -> None:
        d = {"role": "user", "content": "hi", "name": "alice"}
        msg = message_from_dict(d)
        assert msg.name == "alice"
        out = message_to_dict(msg)
        assert out["name"] == "alice"

    def test_tool_calls(self) -> None:
        d = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                }
            ],
        }
        msg = message_from_dict(d)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        out = message_to_dict(msg)
        assert out["tool_calls"] == d["tool_calls"]

    def test_tool_call_id(self) -> None:
        d = {"role": "tool", "content": "42", "tool_call_id": "call_1"}
        msg = message_from_dict(d)
        assert msg.tool_call_id == "call_1"
        out = message_to_dict(msg)
        assert out["tool_call_id"] == "call_1"

    def test_multimodal_text_image(self) -> None:
        d = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
            ],
        }
        msg = message_from_dict(d)
        # Content should be concatenated text
        assert msg.content == "What's in this image?"
        # content_parts should have both parts
        parts = msg.content_parts
        assert len(parts) == 2
        assert isinstance(parts[0], TextPart)
        assert parts[0].text == "What's in this image?"
        assert isinstance(parts[1], ImagePart)
        assert parts[1].source.data == "https://example.com/img.jpg"

        # Roundtrip back to dict
        out = message_to_dict(msg)
        assert isinstance(out["content"], list)
        assert out["content"][0]["type"] == "text"
        assert out["content"][1]["type"] == "image_url"

    def test_multimodal_base64_image(self) -> None:
        d = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc123"},
                },
            ],
        }
        msg = message_from_dict(d)
        parts = msg.content_parts
        assert len(parts) == 2
        assert isinstance(parts[0], TextPart)
        assert isinstance(parts[1], ImagePart)
        assert parts[1].source.type.value == "base64"
        assert parts[1].source.data == "abc123"
        assert parts[1].source.media_type == "image/png"

        # Roundtrip back to dict — base64 must be preserved
        out = message_to_dict(msg)
        assert isinstance(out["content"], list)
        assert out["content"][1]["type"] == "image_url"
        assert out["content"][1]["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_metadata_preserved(self) -> None:
        d = {"role": "user", "content": "hi", "_lattice_metadata": {"foo": "bar"}}
        msg = message_from_dict(d)
        assert msg.metadata.get("foo") == "bar"
        out = message_to_dict(msg)
        assert out["_lattice_metadata"]["foo"] == "bar"


# =============================================================================
# Request roundtrip
# =============================================================================


class TestRequestRoundtrip:
    """dict → Request → dict preserves all fields."""

    def test_basic_request(self) -> None:
        d = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 100,
        }
        req = request_from_dict(d)
        assert req.model == "gpt-4"
        assert req.temperature == 0.7
        assert req.max_tokens == 100
        assert len(req.messages) == 1
        out = request_to_dict(req)
        assert out["model"] == "gpt-4"
        assert out["temperature"] == 0.7
        assert out["max_tokens"] == 100

    def test_tools_and_tool_choice(self) -> None:
        d = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": "auto",
        }
        req = request_from_dict(d)
        assert req.tools is not None
        assert len(req.tools) == 1
        assert req.tool_choice == "auto"
        out = request_to_dict(req)
        assert out["tools"] == d["tools"]
        assert out["tool_choice"] == "auto"

    def test_metadata_fields(self) -> None:
        d = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "reasoning_effort": "high",
            "response_format": {"type": "json_object"},
        }
        req = request_from_dict(d)
        assert req.metadata["reasoning_effort"] == "high"
        assert req.metadata["response_format"] == {"type": "json_object"}
        out = request_to_dict(req)
        assert out["_lattice_metadata"]["reasoning_effort"] == "high"

    def test_multimodal_request(self) -> None:
        d = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's this?"},
                        {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}},
                    ],
                }
            ],
        }
        req = request_from_dict(d)
        assert len(req.messages) == 1
        parts = req.messages[0].content_parts
        assert len(parts) == 2
        out = request_to_dict(req)
        assert isinstance(out["messages"][0]["content"], list)


# =============================================================================
# Response serialization
# =============================================================================


class TestResponseSerialization:
    """Response → OpenAI dict."""

    def test_basic_response(self) -> None:
        resp = Response(
            content="Hello", model="gpt-4", usage={"prompt_tokens": 10, "completion_tokens": 5}
        )
        d = response_to_dict(resp, request_model="gpt-4")
        assert d["choices"][0]["message"]["content"] == "Hello"
        assert d["model"] == "gpt-4"
        assert d["usage"]["completion_tokens"] == 5

    def test_reasoning_preserved(self) -> None:
        resp = Response(content="", metadata={"reasoning": "Let me think..."})
        d = response_to_dict(resp)
        assert d["choices"][0]["message"]["reasoning_content"] == "Let me think..."

    def test_refusal_preserved(self) -> None:
        resp = Response(content="", metadata={"refusal": "I cannot help with that."})
        d = response_to_dict(resp)
        assert d["choices"][0]["message"]["refusal"] == "I cannot help with that."

    def test_tool_calls_preserved(self) -> None:
        resp = Response(
            content="",
            tool_calls=[
                {"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}
            ],
        )
        d = response_to_dict(resp)
        assert d["choices"][0]["message"]["tool_calls"] == resp.tool_calls


# =============================================================================
# DirectHTTPProvider _build_request
# =============================================================================


class TestDirectHTTPProviderBuildRequest:
    """DirectHTTPProvider correctly reconstructs Request from dicts."""

    def test_simple_messages(self) -> None:
        req = DirectHTTPProvider._build_request(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert req.model == "gpt-4"
        assert len(req.messages) == 1
        assert req.messages[0].content == "Hello"

    def test_multimodal_messages(self) -> None:
        req = DirectHTTPProvider._build_request(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's this?"},
                        {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}},
                    ],
                }
            ],
        )
        assert len(req.messages) == 1
        msg = req.messages[0]
        parts = msg.content_parts
        assert len(parts) == 2
        assert isinstance(parts[0], TextPart)
        assert isinstance(parts[1], ImagePart)

    def test_tool_messages(self) -> None:
        req = DirectHTTPProvider._build_request(
            model="gpt-4",
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "foo", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "content": "42", "tool_call_id": "call_1"},
            ],
        )
        assert len(req.messages) == 2
        assert req.messages[0].tool_calls is not None
        assert req.messages[1].tool_call_id == "call_1"

    def test_stop_as_string(self) -> None:
        req = DirectHTTPProvider._build_request(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            stop="END",
        )
        assert req.stop == ["END"]

    def test_metadata_passed(self) -> None:
        req = DirectHTTPProvider._build_request(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            metadata={"reasoning_effort": "high"},
        )
        assert req.metadata["reasoning_effort"] == "high"


# =============================================================================
# SDK compress_request preserves multimodal content
# =============================================================================


class TestSDKCompressPreservesMultimodal:
    """LatticeClient.compress does not corrupt multimodal messages."""

    def test_compress_multimodal(self) -> None:
        from lattice.sdk import LatticeClient

        client = LatticeClient()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image:"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
                ],
            }
        ]
        result = client.compress(messages=messages, model="openai/gpt-4o")
        assert len(result.compressed_messages) == 1
        msg = result.compressed_messages[0]
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][1]["type"] == "image_url"

    def test_compress_tool_conversation(self) -> None:
        from lattice.sdk import LatticeClient

        client = LatticeClient()
        messages = [
            {"role": "user", "content": "What's the weather?"},
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
        assert len(result.compressed_messages) == 3
        assert result.compressed_messages[1].get("tool_calls") is not None
        assert result.compressed_messages[2].get("tool_call_id") == "call_1"


# =============================================================================
# OpenAI adapter serialization with content_parts
# =============================================================================


class TestOpenAIAdapterSerialization:
    """OpenAIAdapter correctly serializes messages with content_parts."""

    def test_serialize_multimodal(self) -> None:
        from lattice.providers.openai import OpenAIAdapter

        adapter = OpenAIAdapter()
        msg = Message(role="user", content="")
        msg.content_parts = [
            TextPart(text="What's this?"),
            ImagePart(
                source=ImageSource(type=ImageSourceType.URL, data="https://example.com/img.jpg")
            ),
        ]
        req = Request(messages=[msg], model="gpt-4o")
        body = adapter.serialize_request(req)
        assert body["model"] == "gpt-4o"
        serialized_msg = body["messages"][0]
        assert isinstance(serialized_msg["content"], list)
        assert serialized_msg["content"][0]["type"] == "text"
        assert serialized_msg["content"][1]["type"] == "image_url"

    def test_serialize_simple_text(self) -> None:
        from lattice.providers.openai import OpenAIAdapter

        adapter = OpenAIAdapter()
        req = Request(messages=[Message(role="user", content="Hello")], model="gpt-4")
        body = adapter.serialize_request(req)
        assert body["messages"][0]["content"] == "Hello"

    def test_serialize_tool_calls(self) -> None:
        from lattice.providers.openai import OpenAIAdapter

        adapter = OpenAIAdapter()
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
        body = adapter.serialize_request(req)
        assert body["messages"][0]["tool_calls"] == req.messages[0].tool_calls
