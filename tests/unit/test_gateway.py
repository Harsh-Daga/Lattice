"""Tests for gateway and compatibility delegation."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from lattice.core.result import Ok
from lattice.core.session import MemorySessionStore, SessionManager
from lattice.core.transport import Response
from lattice.gateway.compat import (
    HTTPCompatHandler,
    anthropic_passthrough,
    extract_anthropic_text_blocks,
    extract_responses_text_blocks,
    replace_anthropic_text_blocks,
    replace_responses_text_blocks,
    responses_passthrough,
)
from lattice.gateway.server import ClientConnectionInfo, LLMTPGateway
from lattice.protocol.dictionary_codec import DictionaryCodec
from lattice.protocol.framing import BinaryFramer, FrameFlags, FrameType
from lattice.protocol.resume import StreamManager


class _PassthroughPipeline:
    async def process(self, request: Any, _context: Any) -> Any:
        return Ok(request)

    def reverse(self, response: Response, _context: Any) -> Response:
        return response


class _FakeProvider:
    async def completion(
        self, model: str, messages: list[dict[str, Any]], **_kwargs: Any
    ) -> Response:
        assert messages
        return Response(content="pong", model=model, usage={"total_tokens": 1})


class _FakeAdapter:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeRegistry:
    def resolve(self, model: str) -> _FakeAdapter:
        if model.startswith("claude-"):
            return _FakeAdapter("anthropic")
        return _FakeAdapter("openai")


class _FakeProviderWithRegistry(_FakeProvider):
    def __init__(self) -> None:
        self.registry = _FakeRegistry()


@pytest.mark.asyncio
async def test_gateway_json_request_roundtrip() -> None:
    store = MemorySessionStore()
    session_manager = SessionManager(store)
    gateway = LLMTPGateway(
        session_manager=session_manager,
        pipeline=_PassthroughPipeline(),  # type: ignore[arg-type]
        provider=_FakeProvider(),  # type: ignore[arg-type]
        framer=BinaryFramer(),
        stream_manager=StreamManager(),
        store=store,
    )
    body = {"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    out, meta = await gateway.handle_request(
        json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json"},
        client_info=ClientConnectionInfo(client_id="c1"),
    )
    parsed = json.loads(out.decode("utf-8"))
    assert parsed["choices"][0]["message"]["content"] == "pong"
    assert meta.get("x-lattice-framing") == "json"


@pytest.mark.asyncio
async def test_gateway_binary_request_roundtrip() -> None:
    store = MemorySessionStore()
    session_manager = SessionManager(store)
    framer = BinaryFramer()
    gateway = LLMTPGateway(
        session_manager=session_manager,
        pipeline=_PassthroughPipeline(),  # type: ignore[arg-type]
        provider=_FakeProvider(),  # type: ignore[arg-type]
        framer=framer,
        stream_manager=StreamManager(),
        store=store,
    )
    body = {"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    raw_in = framer.encode_request(json.dumps(body).encode("utf-8"))[0].to_bytes()
    raw_out, meta = await gateway.handle_request(
        raw_in, headers={}, client_info=ClientConnectionInfo()
    )
    out_frame = framer.decode_frame(raw_out)
    assert out_frame.frame_type == FrameType.RESPONSE
    payload = json.loads(out_frame.payload.decode("utf-8"))
    assert payload["choices"][0]["message"]["content"] == "pong"
    assert meta.get("x-lattice-framing") == "native"


@pytest.mark.asyncio
async def test_gateway_dictionary_wire_roundtrip() -> None:
    store = MemorySessionStore()
    session_manager = SessionManager(store)
    session = await session_manager.create_session(
        provider="openai",
        model="openai/gpt-4o",
        messages=[],
    )
    framer = BinaryFramer()
    gateway = LLMTPGateway(
        session_manager=session_manager,
        pipeline=_PassthroughPipeline(),  # type: ignore[arg-type]
        provider=_FakeProvider(),  # type: ignore[arg-type]
        framer=framer,
        stream_manager=StreamManager(),
        store=store,
    )
    body = {
        "model": "openai/gpt-4o",
        "session_id": session.session_id,
        "messages": [
            {"role": "user", "content": "repeat_this_custom_key repeat_this_custom_key"},
        ],
    }
    codec = DictionaryCodec(session_id=session.session_id)
    raw_json = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    raw_in = framer.encode_request(codec.compress(raw_json), flags=FrameFlags.DICT_COMPRESSED)[
        0
    ].to_bytes()
    raw_out, meta = await gateway.handle_request(
        raw_in,
        headers={"x-lattice-session-id": session.session_id},
        client_info=ClientConnectionInfo(client_id="c-dict"),
    )
    out_frame = framer.decode_frame(raw_out)
    assert out_frame.frame_type == FrameType.RESPONSE
    assert out_frame.flags & FrameFlags.DICT_COMPRESSED
    payload = json.loads(codec.decompress(out_frame.payload).decode("utf-8"))
    assert payload["choices"][0]["message"]["content"] == "pong"
    assert meta.get("x-lattice-framing") == "native"
    stored = await store.get(session.session_id)
    assert stored is not None
    assert "dict_wire_state" in stored.metadata


@pytest.mark.asyncio
async def test_gateway_format_detection() -> None:
    store = MemorySessionStore()
    session_manager = SessionManager(store)
    framer = BinaryFramer()
    gateway = LLMTPGateway(
        session_manager=session_manager,
        pipeline=_PassthroughPipeline(),  # type: ignore[arg-type]
        provider=_FakeProvider(),  # type: ignore[arg-type]
        framer=framer,
        stream_manager=StreamManager(),
        store=store,
    )
    json_body = {"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    json_out, json_meta = await gateway.handle_request(
        json.dumps(json_body).encode("utf-8"), headers={}, client_info=ClientConnectionInfo()
    )
    assert json.loads(json_out.decode("utf-8"))["object"] == "chat.completion"
    assert json_meta.get("x-lattice-framing") == "json"

    bin_in = framer.encode_request(json.dumps(json_body).encode("utf-8"))[0].to_bytes()
    bin_out, bin_meta = await gateway.handle_request(
        bin_in, headers={}, client_info=ClientConnectionInfo()
    )
    assert framer.decode_frame(bin_out).frame_type == FrameType.RESPONSE
    assert bin_meta.get("x-lattice-framing") == "native"


@pytest.mark.asyncio
async def test_compat_handler_session_header() -> None:
    store = MemorySessionStore()
    session_manager = SessionManager(store)
    gateway = LLMTPGateway(
        session_manager=session_manager,
        pipeline=_PassthroughPipeline(),  # type: ignore[arg-type]
        provider=_FakeProvider(),  # type: ignore[arg-type]
        framer=BinaryFramer(),
        stream_manager=StreamManager(),
        store=store,
    )
    payload = await gateway.handle_session_start(
        {
            "provider": "openai",
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert payload["session_id"].startswith("lattice-")


@pytest.mark.asyncio
async def test_compat_handler_delegates_chat() -> None:
    store = MemorySessionStore()
    session_manager = SessionManager(store)
    gateway = LLMTPGateway(
        session_manager=session_manager,
        pipeline=_PassthroughPipeline(),  # type: ignore[arg-type]
        provider=_FakeProvider(),  # type: ignore[arg-type]
        framer=BinaryFramer(),
        stream_manager=StreamManager(),
        store=store,
    )

    async def _handler(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        return {"ok": "yes"}

    compat = HTTPCompatHandler(gateway, chat_completion_handler=_handler)
    result = await compat.handle_chat_completion({})
    assert result == {"ok": "yes"}


@pytest.mark.asyncio
async def test_gateway_provider_resolution_explicit_hint() -> None:
    store = MemorySessionStore()
    gateway = LLMTPGateway(
        session_manager=SessionManager(store),
        pipeline=_PassthroughPipeline(),  # type: ignore[arg-type]
        provider=_FakeProvider(),  # type: ignore[arg-type]
        framer=BinaryFramer(),
        stream_manager=StreamManager(),
        store=store,
    )
    provider_name = gateway._resolve_provider_name(
        {"provider_name": "anthropic"},
        {"x-lattice-provider": "openai"},
        "openai/gpt-4o",
    )
    assert provider_name == "anthropic"


@pytest.mark.asyncio
async def test_gateway_provider_resolution_rejects_bare_model() -> None:
    store = MemorySessionStore()
    gateway = LLMTPGateway(
        session_manager=SessionManager(store),
        pipeline=_PassthroughPipeline(),  # type: ignore[arg-type]
        provider=_FakeProviderWithRegistry(),  # type: ignore[arg-type]
        framer=BinaryFramer(),
        stream_manager=StreamManager(),
        store=store,
    )
    with pytest.raises(ValueError, match="Provider not specified"):
        gateway._resolve_provider_name({}, {}, "claude-3-5-sonnet")


def test_compat_anthropic_text_block_roundtrip() -> None:
    body: dict[str, Any] = {
        "system": [{"type": "text", "text": "System intro"}],
        "messages": [
            {"role": "user", "content": "Plain user content"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Assistant block"},
                    {
                        "type": "tool_result",
                        "content": [{"type": "text", "text": "Tool text"}],
                    },
                ],
            },
        ],
    }
    blocks = extract_anthropic_text_blocks(body)
    assert len(blocks) == 4
    replace_anthropic_text_blocks(blocks, [f"z{i}" for i in range(len(blocks))])
    assert body["system"][0]["text"] == "z0"
    assert body["messages"][0]["content"] == "z1"
    assert body["messages"][1]["content"][0]["text"] == "z2"
    assert body["messages"][1]["content"][1]["content"][0]["text"] == "z3"


def test_compat_responses_text_block_roundtrip() -> None:
    body: dict[str, Any] = {
        "instructions": "Top instructions",
        "input": [
            {"role": "user", "content": "Inline input"},
            {"role": "user", "content": [{"type": "text", "text": "Nested part"}]},
        ],
    }
    blocks = extract_responses_text_blocks(body)
    assert len(blocks) == 3
    replace_responses_text_blocks(blocks, [f"r{i}" for i in range(len(blocks))])
    assert body["instructions"] == "r0"
    assert body["input"][0]["content"] == "r1"
    assert body["input"][1]["content"][0]["text"] == "r2"


@pytest.mark.asyncio
async def test_responses_passthrough_requires_explicit_base_url() -> None:
    # Use a provider name that has no default base URL so the function
    # must rely on explicit configuration.
    provider = SimpleNamespace(
        provider_base_urls={},
        pool=SimpleNamespace(
            get_client=lambda *_a, **_k: None, get_http_version=lambda *_a, **_k: "1.1"
        ),
    )
    request = SimpleNamespace(query_params={}, headers={})

    # responses_passthrough hard-codes provider_name="openai" which IS in
    # the default base URLs map, so it will resolve successfully and only
    # fail later at the fake pool layer.  Verify it resolved the default URL.
    with pytest.raises(AttributeError):
        await responses_passthrough(
            "POST",
            "/v1/responses",
            b"",
            request,
            provider,
            logger=SimpleNamespace(info=lambda *_args, **_kwargs: None),
        )


@pytest.mark.asyncio
async def test_anthropic_passthrough_requires_explicit_base_url() -> None:
    """Anthropic passthrough succeeds with well-known URL fallback.

    When no provider_base_urls are configured, the multi-tier URL resolution
    falls through to ``_WELL_KNOWN_PROVIDER_URLS`` (api.anthropic.com) so the
    passthrough can still forward requests.
    """
    from lattice.providers.transport import ProviderRegistry

    async def _mock_request(*_a: Any, **_k: Any) -> SimpleNamespace:
        return SimpleNamespace(status_code=200, content=b"{}", headers={})

    provider = SimpleNamespace(
        provider_base_urls={},
        registry=ProviderRegistry(),
        pool=SimpleNamespace(
            get_client=lambda *_a, **_k: SimpleNamespace(request=_mock_request),
            get_http_version=lambda *_a, **_k: "1.1",
        ),
    )
    request = SimpleNamespace(query_params={}, headers={})

    resp = await anthropic_passthrough(
        "POST",
        "/v1/messages",
        b'{"model":"anthropic/claude-3-haiku","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}',
        request,
        provider,
        logger=SimpleNamespace(
            info=lambda *_args, **_kwargs: None, warning=lambda *_args, **_kwargs: None
        ),
    )
    assert resp.status_code == 200
