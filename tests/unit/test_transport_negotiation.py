"""Tests for Phase 6 transport negotiation and observability."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from lattice.client import LatticeClient
from lattice.core.config import LatticeConfig
from lattice.core.delta_wire import DeltaWireEncoder
from lattice.protocol.framing import BinaryFramer, FrameType
from lattice.protocol.resume import StreamManager
from lattice.proxy.server import create_app


class TestDeltaWireReportsMetadata:
    def test_delta_wire_reports_metadata(self) -> None:
        encoder = DeltaWireEncoder()
        assert encoder.get_transport_metadata() == {
            "delta_mode": True,
            "anchor_version": None,
            "base_sequence": 0,
        }

        encoder.encode(
            full_messages=[{"role": "user", "content": "hi"}],
            session_id="sess_abc",
            base_sequence=3,
            anchor_version=7,
        )
        meta = encoder.get_transport_metadata()
        assert meta["anchor_version"] == 7
        assert meta["base_sequence"] == 3
        assert meta["delta_mode"] is True


class TestFramingNegotiationOutcome:
    def test_framing_negotiation_outcome_encoding(self) -> None:
        framer = BinaryFramer()

        frame = framer.encode_negotiation_outcome(True, fallback_reason="")
        assert frame.frame_type == FrameType.DICTIONARY_NEGOTIATE
        accepted, reason = framer.decode_negotiation_outcome(frame)
        assert accepted is True
        assert reason == ""

        frame = framer.encode_negotiation_outcome(False, fallback_reason="version_mismatch")
        accepted, reason = framer.decode_negotiation_outcome(frame)
        assert accepted is False
        assert reason == "version_mismatch"


class TestResumeMetadataAvailable:
    def test_resume_metadata_available(self) -> None:
        manager = StreamManager()
        stream_id = manager.create_stream()
        manager.append_chunk(stream_id, 0, "data: hello\n\n")
        manager.append_chunk(stream_id, 1, "data: world\n\n")

        meta = manager.get_resume_metadata(stream_id)
        assert meta["resumed"] is True
        assert meta["replay_chunks"] == 2
        assert meta["stream_sequence"] == 1

    def test_resume_metadata_not_found(self) -> None:
        manager = StreamManager()
        meta = manager.get_resume_metadata("missing")
        assert meta["resumed"] is False
        assert meta["reason"] == "stream_not_found"


class TestSDKClientReportsTransportMetadata:
    def test_sdk_client_reports_transport_metadata(self) -> None:
        client = LatticeClient()
        result = client.compress(
            messages=[{"role": "user", "content": "Hello!"}],
            model="gpt-4",
        )
        assert "transport" in result.runtime
        assert result.runtime["transport"]["framing"] == "json"
        assert result.runtime["transport"]["delta"] == "bypassed"


class TestProxyHeadersIncludeTransportInfo:
    def test_proxy_headers_include_transport_info(self) -> None:
        config = LatticeConfig(
            provider_base_url="http://127.0.0.1:11434",
            provider_base_urls={
                "openai": "https://api.openai.com",
            },
        )

        with patch(
            "lattice.providers.transport.DirectHTTPProvider.completion",
            new_callable=AsyncMock,
        ) as mock_completion:
            from lattice.core.transport import Response

            mock_completion.return_value = Response(
                content="hello",
                model="gpt-4",
                finish_reason="stop",
            )
            app = create_app(config=config)
            client = TestClient(app)

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "openai/gpt-4",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert response.status_code == 200
            assert "x-lattice-delta" in response.headers
            assert "x-lattice-http-version" in response.headers


class TestDeltaWireNegotiationOutcome:
    def test_delta_negotiation_outcome_roundtrip(self) -> None:
        encoder = DeltaWireEncoder()
        encoder.encode(
            full_messages=[{"role": "user", "content": "hi"}],
            session_id="sess_abc",
            base_sequence=3,
            anchor_version=7,
        )
        outcome = encoder.encode_negotiation_outcome(accepted=True, fallback_reason="")
        assert outcome["delta_accepted"] is True
        assert outcome["delta_fallback_reason"] == ""
        assert outcome["anchor_version"] == 7
        assert outcome["base_sequence"] == 3

    def test_delta_negotiation_outcome_fallback(self) -> None:
        encoder = DeltaWireEncoder()
        outcome = encoder.encode_negotiation_outcome(
            accepted=False, fallback_reason="version_mismatch"
        )
        accepted, reason = DeltaWireEncoder.decode_negotiation_outcome(outcome)
        assert accepted is False
        assert reason == "version_mismatch"


class TestProxyDeltaHeaderReflectsUsage:
    def test_proxy_delta_header_bypassed_when_not_delta(self) -> None:
        config = LatticeConfig(
            provider_base_url="http://127.0.0.1:11434",
            provider_base_urls={"openai": "https://api.openai.com"},
        )
        with patch(
            "lattice.providers.transport.DirectHTTPProvider.completion",
            new_callable=AsyncMock,
        ) as mock_completion:
            from lattice.core.transport import Response

            mock_completion.return_value = Response(
                content="hello", model="gpt-4", finish_reason="stop"
            )
            app = create_app(config=config)
            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={"model": "openai/gpt-4", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert response.status_code == 200
            assert response.headers.get("x-lattice-delta") == "bypassed"


class TestFallbackVisibleInStats:
    def test_fallback_visible_in_stats(self) -> None:
        config = LatticeConfig(
            provider_base_url="http://127.0.0.1:11434",
            provider_base_urls={
                "openai": "https://api.openai.com",
            },
        )
        app = create_app(config=config)
        client = TestClient(app)

        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "fallbacks" in data
        assert "http2_to_http11_count" in data["fallbacks"]
        assert "delta_to_full_prompt_count" in data["fallbacks"]
        assert "transport" in data
        assert "pools" in data["transport"]
