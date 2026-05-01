"""Tests for the simplified LatticeClient SDK (lattice.client).

The old full-featured SDK surface was removed in Phase 0.  This module tests
the minimal, stable API:

* ``LatticeClient.compress()`` — local compression only
* ``LatticeClient.health()`` — config summary
"""

from __future__ import annotations

import json

import pytest

from lattice.client import LatticeClient
from lattice.core.config import LatticeConfig
from lattice.protocol.framing import BinaryFramer, FrameFlags
from lattice.sdk import LatticeProxyClient


@pytest.fixture
def client() -> LatticeClient:
    return LatticeClient()


class TestCompress:
    def test_compress_dict_messages(self, client: LatticeClient) -> None:
        result = client.compress(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ],
            model="gpt-4",
        )
        assert result.compressed_messages
        assert result.original_tokens > 0
        assert result.compressed_tokens >= 0
        assert result.elapsed_ms >= 0
        assert isinstance(result.transforms_applied, list)
        assert "runtime_contract" in result.transforms_applied
        assert result.runtime["tier"] == "SIMPLE"
        assert result.runtime_budget["budget_ms"] > 0

    def test_compress_empty_messages(self, client: LatticeClient) -> None:
        result = client.compress(messages=[], model="gpt-4")
        assert result.compressed_messages == []
        assert result.original_tokens == 0

    def test_compress_mode_safe(self) -> None:
        cfg = LatticeConfig(compression_mode="safe")
        client = LatticeClient(config=cfg)
        result = client.compress(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
            mode="safe",
        )
        assert "rate_distortion" not in result.transforms_applied

    def test_compress_mode_aggressive(self) -> None:
        cfg = LatticeConfig(compression_mode="aggressive")
        client = LatticeClient(config=cfg)
        result = client.compress(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4",
            mode="aggressive",
        )
        # Aggressive mode should include lossy transforms
        assert any(t in result.transforms_applied for t in ("rate_distortion", "hierarchical_summary"))

    def test_compress_request_includes_runtime_metadata(self, client: LatticeClient) -> None:
        request = client.compress_request(
            messages=[{"role": "user", "content": "Hello"}],
            model="openai/gpt-4",
        )
        assert request.metadata["_lattice_runtime"]["tier"] == "SIMPLE"
        assert request.metadata["_lattice_runtime_contract"]["mode"] == "minimal"


class TestHealth:
    def test_health(self, client: LatticeClient) -> None:
        health = client.health()
        assert health["status"] == "healthy"
        assert health["compression_mode"] == "balanced"
        assert isinstance(health["transforms"], list)
        assert health["pipeline"]["runtime_contract_enabled"] is True
        assert health["pipeline"]["execution_transforms"] == []


class TestProxyClient:
    @pytest.mark.asyncio
    async def test_native_gateway_dictionary_wire(self, monkeypatch) -> None:
        proxy_client = LatticeProxyClient(base_url="http://localhost:8787")
        session_id = "sess-native"
        body = {
            "model": "openai/gpt-4o",
            "session_id": session_id,
            "messages": [{"role": "user", "content": "repeat_this_custom_key repeat_this_custom_key"}],
        }

        async def _fake_post(url: str, content: bytes | None = None, headers: dict[str, str] | None = None, **_kwargs):
            assert url.endswith("/lattice/gateway")
            assert content is not None
            assert headers and headers["x-lattice-session-id"] == session_id
            request_frame = BinaryFramer().decode_frame(content)
            assert request_frame.flags & FrameFlags.DICT_COMPRESSED
            codec = proxy_client._dictionary_codecs[session_id]
            request_payload = codec.decompress(request_frame.payload)
            assert json.loads(request_payload.decode("utf-8"))["model"] == body["model"]
            response_payload = json.dumps(
                {
                    "choices": [
                        {"message": {"content": "pong"}, "index": 0, "finish_reason": "stop"}
                    ],
                    "object": "chat.completion",
                },
                separators=(",", ":"),
            ).encode("utf-8")
            response_frame = BinaryFramer().encode_response(
                codec.compress(response_payload),
                flags=FrameFlags.DICT_COMPRESSED,
            )[0]

            class _Resp:
                content = response_frame.to_bytes()

                def raise_for_status(self) -> None:
                    return None

            return _Resp()

        monkeypatch.setattr(proxy_client._client, "post", _fake_post)
        result = await proxy_client.chat.create(
            model=body["model"],
            messages=body["messages"],
            session_id=session_id,
            native_wire=True,
            dictionary_wire=True,
        )
        assert result["choices"][0]["message"]["content"] == "pong"
