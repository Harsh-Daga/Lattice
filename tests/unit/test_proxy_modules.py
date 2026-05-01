"""Tests for proxy health endpoints and provider adapter compatibility."""

from __future__ import annotations

from fastapi.testclient import TestClient

from lattice.core.config import LatticeConfig
from lattice.core.transport import Message, Request
from lattice.providers.anthropic import AnthropicAdapter
from lattice.providers.openai import OpenAIAdapter
from lattice.proxy.health import HealthManager
from lattice.proxy.server import create_app


class TestOpenAIAdapter:
    """Compatibility shim: ensure our new adapter covers the same capabilities
    as the old proxy/router.py OpenAIAdapter."""

    def test_serialize_request(self) -> None:
        adapter = OpenAIAdapter()
        req = Request(
            model="gpt-4",
            messages=[Message(role="user", content="Hello")],
            temperature=0.5,
        )
        body = adapter.serialize_request(req)
        assert body["model"] == "gpt-4"
        assert body["messages"][0]["content"] == "Hello"
        assert body["temperature"] == 0.5

    def test_deserialize_response(self) -> None:
        adapter = OpenAIAdapter()
        raw: dict[str, object] = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Result"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 5},
        }
        resp = adapter.deserialize_response(raw)
        assert resp.content == "Result"
        assert resp.role == "assistant"
        assert resp.model == "gpt-4"


class TestAnthropicAdapter:
    """Ensure our new AnthropicAdapter handles content blocks."""

    def test_deserialize_response_content_blocks(self) -> None:
        adapter = AnthropicAdapter()
        raw: dict[str, object] = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3",
            "content": [
                {"type": "text", "text": "Block 1"},
                {"type": "text", "text": "Block 2"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        resp = adapter.deserialize_response(raw)
        assert resp.content == "Block 1Block 2"
        assert resp.finish_reason == "stop"  # mapped from end_turn


class TestHealthManager:
    def test_healthz(self) -> None:
        config = LatticeConfig()
        hm = HealthManager(config)
        result = hm.healthz()
        assert result["status"] == "healthy"
        assert "version" in result

    def test_readyz_with_transforms(self) -> None:
        config = LatticeConfig(provider_base_url="http://test")
        hm = HealthManager(config, pipeline_transform_count=4)
        result = hm.readyz()
        assert result["status"] == "ready"
        assert result["checks"]["pipeline"] is True

    def test_readyz_no_transforms(self) -> None:
        config = LatticeConfig()
        hm = HealthManager(config, pipeline_transform_count=0)
        result = hm.readyz()
        assert result["status"] == "not_ready"
        assert result["checks"]["pipeline"] is False

    def test_startupz(self) -> None:
        config = LatticeConfig()
        hm = HealthManager(config)
        result = hm.startupz()
        assert result["status"] == "started"

    def test_stats(self) -> None:
        config = LatticeConfig()
        hm = HealthManager(config)
        result = hm.stats(transform_names=["ref_sub"], session_count=0)
        assert result["transforms"] == ["ref_sub"]
        assert result["sessions"] == 0


class TestCapabilitySurface:
    def test_provider_capabilities_endpoint(self) -> None:
        config = LatticeConfig(
            provider_base_url="http://127.0.0.1:11434",
            provider_base_urls={
                "openai": "https://api.openai.com",
                "anthropic": "https://api.anthropic.com",
            },
        )
        app = create_app(config=config)
        client = TestClient(app)

        response = client.get("/providers/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data
        assert "openai" in data["providers"]
        assert data["providers"]["openai"]["cache_mode"] == "auto_prefix"
        assert "cache_modes" in data

    def test_stats_exposes_capability_summary(self) -> None:
        config = LatticeConfig(
            provider_base_url="http://127.0.0.1:11434",
            provider_base_urls={
                "openai": "https://api.openai.com",
                "anthropic": "https://api.anthropic.com",
            },
        )
        app = create_app(config=config)
        client = TestClient(app)

        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "capabilities" in data
        assert data["capabilities"]["openai"]["supports_prompt_caching"] is True
        assert "batching" in data
        assert "queue_sizes" in data["batching"]
