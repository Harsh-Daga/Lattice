"""Tests for Phase 0 fallback visibility: explicit compatibility and fallback telemetry."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from lattice.core.config import LatticeConfig
from lattice.core.delta_wire import DeltaWireDecoder
from lattice.core.session import MemorySessionStore, Session
from lattice.core.transport import Message, Request
from lattice.gateway.compat import build_routing_headers
from lattice.providers.transport import ConnectionPoolManager
from lattice.proxy.server import create_app


class TestBuildRoutingHeadersFallbackFields:
    def test_build_routing_headers_with_fallback_fields(self) -> None:
        headers = build_routing_headers(
            "gpt-4",
            framing="native",
            delta_mode="delta",
            http_version="http/2",
            semantic_cache_status="exact-hit",
            batching_status="batched",
            speculative_status="hit",
            fallback_reason="h2_unavailable",
        )
        assert headers["x-lattice-framing"] == "native"
        assert headers["x-lattice-delta"] == "delta"
        assert headers["x-lattice-http-version"] == "http/2"
        assert headers["x-lattice-semantic-cache"] == "exact-hit"
        assert headers["x-lattice-batching"] == "batched"
        assert headers["x-lattice-speculative-status"] == "hit"
        assert headers["x-lattice-fallback-reason"] == "h2_unavailable"

    def test_build_routing_headers_omits_empty_fallback_fields(self) -> None:
        headers = build_routing_headers("gpt-4")
        assert "x-lattice-framing" not in headers
        assert "x-lattice-delta" not in headers
        assert "x-lattice-http-version" not in headers
        assert "x-lattice-semantic-cache" not in headers
        assert "x-lattice-batching" not in headers
        assert "x-lattice-speculative-status" not in headers
        assert "x-lattice-fallback-reason" not in headers

    def test_build_routing_headers_uses_transport_outcome(self) -> None:
        from lattice.core.telemetry import TransportOutcome

        outcome = TransportOutcome(
            framing="native",
            delta_mode="delta",
            http_version="http/2",
            semantic_cache_status="exact-hit",
            batching_status="batched",
            speculative_status="hit",
        )
        headers = build_routing_headers("gpt-4", transport_outcome=outcome)
        assert headers["x-lattice-framing"] == "native"
        assert headers["x-lattice-delta"] == "delta"
        assert headers["x-lattice-http-version"] == "http/2"
        assert headers["x-lattice-semantic-cache"] == "exact-hit"
        assert headers["x-lattice-batching"] == "batched"
        assert headers["x-lattice-speculative-status"] == "hit"

    def test_build_routing_headers_legacy_params_override_when_no_outcome(self) -> None:
        headers = build_routing_headers(
            "gpt-4",
            framing="json",
            delta_mode="bypassed",
            http_version="http/1.1",
        )
        assert headers["x-lattice-framing"] == "json"
        assert headers["x-lattice-delta"] == "bypassed"
        assert headers["x-lattice-http-version"] == "http/1.1"

    def test_build_routing_headers_legacy_params_override_canonical_object(self) -> None:
        """Explicit legacy args must win over TransportOutcome defaults."""
        from lattice.core.telemetry import TransportOutcome

        outcome = TransportOutcome(
            framing="native",
            delta_mode="delta",
            http_version="http/2",
        )
        headers = build_routing_headers(
            "gpt-4",
            transport_outcome=outcome,
            framing="json",  # explicit override
            delta_mode="bypassed",  # explicit override
            http_version="http/1.1",  # explicit override
        )
        assert headers["x-lattice-framing"] == "json"
        assert headers["x-lattice-delta"] == "bypassed"
        assert headers["x-lattice-http-version"] == "http/1.1"

    def test_stream_resume_fallback_reason_header(self) -> None:
        from lattice.core.telemetry import TransportOutcome

        outcome = TransportOutcome(
            stream_resumed=True,
            stream_resume_fallback_reason="token_expired",
        )
        headers = build_routing_headers("gpt-4", transport_outcome=outcome)
        assert headers["x-lattice-stream-resumed"] == "true"
        assert headers["x-lattice-stream-resume-fallback-reason"] == "token_expired"

    def test_speculative_alias_strategy(self) -> None:
        """Legacy x-lattice-speculative and canonical x-lattice-speculative-status
        coexist by design; the legacy alias is added by build_routing_headers
        when used_speculative=True, while the canonical header comes from
        TransportOutcome.to_headers().
        """
        from lattice.core.telemetry import TransportOutcome

        # Legacy path
        headers = build_routing_headers("gpt-4", used_speculative=True, prediction_hit=True)
        assert headers["x-lattice-speculative"] == "hit"
        assert "x-lattice-speculative-status" not in headers

        # Canonical path
        outcome = TransportOutcome(speculative_status="miss")
        headers = build_routing_headers("gpt-4", transport_outcome=outcome)
        assert headers["x-lattice-speculative-status"] == "miss"
        assert "x-lattice-speculative" not in headers

        # Both together: canonical speculative_status + legacy used_speculative
        outcome = TransportOutcome(speculative_status="bypassed")
        headers = build_routing_headers(
            "gpt-4",
            transport_outcome=outcome,
            used_speculative=True,
            prediction_hit=True,
        )
        # Legacy alias is independent and still present
        assert headers["x-lattice-speculative"] == "hit"
        # Canonical header from outcome
        assert headers["x-lattice-speculative-status"] == "bypassed"


class TestConnectionPoolTracksHttp2Fallback:
    def test_connection_pool_tracks_http2_fallback(self) -> None:
        pool = ConnectionPoolManager(http2=True)
        # Simulate ImportError on first client creation to trigger fallback
        call_count = 0

        def _fake_async_client(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ImportError("h2 unavailable")
            return MagicMock()

        with patch("lattice.providers.transport.httpx.AsyncClient", side_effect=_fake_async_client):
            client = pool.get_client("openai", "https://api.openai.com")
            assert client is not None

        assert pool.get_http_version("openai", "https://api.openai.com") == "http/1.1"
        assert pool.get_fallback_reason("openai", "https://api.openai.com") == "h2_unavailable"
        assert ("openai", "https://api.openai.com") in pool._http2_fallback_reason

    def test_connection_pool_no_fallback_when_http2_works(self) -> None:
        pool = ConnectionPoolManager(http2=True)
        with patch("lattice.providers.transport.httpx.AsyncClient", return_value=MagicMock()):
            pool.get_client("openai", "https://api.openai.com")
        assert pool.get_http_version("openai", "https://api.openai.com") == "http/2"
        assert pool.get_fallback_reason("openai", "https://api.openai.com") is None


class TestDeltaWireFallbackStats:
    @pytest.mark.asyncio
    async def test_delta_wire_fallback_stats(self) -> None:
        store = MemorySessionStore()
        decoder = DeltaWireDecoder(store)

        # Reset class-level stats before test
        DeltaWireDecoder._delta_success_count = 0
        DeltaWireDecoder._delta_fallback_count = 0
        DeltaWireDecoder._delta_fallback_reasons = {}

        # 1. Not a delta request -> no stats change
        req = Request(messages=[Message(role="user", content="hello")])
        await decoder.decode(req)
        stats = DeltaWireDecoder.get_fallback_stats()
        assert stats["success_count"] == 0
        assert stats["fallback_count"] == 0

        # 2. Delta request with missing session -> fallback
        req2 = Request(
            messages=[Message(role="user", content="delta")],
            metadata={
                "_delta_wire": True,
                "_delta_session_id": "sess_missing",
                "_delta_base_seq": 0,
                "_delta_messages": [{"role": "user", "content": "hi"}],
            },
        )
        await decoder.decode(req2)
        stats = DeltaWireDecoder.get_fallback_stats()
        assert stats["fallback_count"] == 1
        assert stats["fallback_reasons"]["session_not_found"] == 1

        # 3. Delta request with version mismatch -> fallback
        import time as _time

        now = _time.time()
        session = Session(
            session_id="sess_v1",
            created_at=now,
            last_accessed_at=now,
            provider="openai",
            model="gpt-4",
            messages=[Message(role="user", content="hello")],
        )
        session.version = 2
        await store.set(session)

        req3 = Request(
            messages=[Message(role="user", content="delta")],
            metadata={
                "_delta_wire": True,
                "_delta_session_id": "sess_v1",
                "_delta_base_seq": 0,
                "_delta_anchor_version": 1,
                "_delta_messages": [{"role": "user", "content": "hi"}],
            },
        )
        await decoder.decode(req3)
        stats = DeltaWireDecoder.get_fallback_stats()
        assert stats["fallback_count"] == 2
        assert stats["fallback_reasons"]["version_mismatch"] == 1

        # 4. Delta request with sequence mismatch -> fallback
        req4 = Request(
            messages=[Message(role="user", content="delta")],
            metadata={
                "_delta_wire": True,
                "_delta_session_id": "sess_v1",
                "_delta_base_seq": 10,
                "_delta_messages": [{"role": "user", "content": "hi"}],
            },
        )
        await decoder.decode(req4)
        stats = DeltaWireDecoder.get_fallback_stats()
        assert stats["fallback_count"] == 3
        assert stats["fallback_reasons"]["sequence_mismatch"] == 1

        # 5. Successful delta decode
        req5 = Request(
            messages=[Message(role="user", content="delta")],
            metadata={
                "_delta_wire": True,
                "_delta_session_id": "sess_v1",
                "_delta_base_seq": 1,
                "_delta_messages": [{"role": "user", "content": "follow-up"}],
            },
        )
        await decoder.decode(req5)
        stats = DeltaWireDecoder.get_fallback_stats()
        assert stats["success_count"] == 1
        assert stats["fallback_count"] == 3


class TestStatsEndpointIncludesFallbacks:
    def test_stats_endpoint_includes_fallbacks(self) -> None:
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
        assert "fallbacks" in data
        assert "transport" in data

        fallbacks = data["fallbacks"]
        assert "http2_to_http11_count" in fallbacks
        assert "delta_to_full_prompt_count" in fallbacks
        assert "native_framing_to_json_count" in fallbacks
        assert "semantic_cache_approximate_hits" in fallbacks
        assert "semantic_cache_misses" in fallbacks

        transport = data["transport"]
        assert "pools" in transport


class TestTriStateOverrideSemantics:
    """Phase 0: Explicit tri-state overrides for fallback visibility."""

    def test_tri_state_none_preserves_canonical(self) -> None:
        from lattice.core.telemetry import TransportOutcome

        outcome = TransportOutcome(
            speculative_status="bypassed",
            batching_status="bypassed",
        )
        headers = build_routing_headers(
            "gpt-4",
            transport_outcome=outcome,
            used_speculative=None,
            batched=None,
        )
        assert headers["x-lattice-speculative-status"] == "bypassed"
        assert headers["x-lattice-batching"] == "bypassed"

    def test_tri_state_false_suppresses_speculative(self) -> None:
        from lattice.core.telemetry import TransportOutcome

        outcome = TransportOutcome(speculative_status="hit")
        headers = build_routing_headers(
            "gpt-4",
            transport_outcome=outcome,
            used_speculative=False,
        )
        assert "x-lattice-speculative" not in headers
        assert "x-lattice-speculative-status" not in headers

    def test_tri_state_cache_hit_false_suppresses(self) -> None:
        from lattice.core.telemetry import TransportOutcome

        outcome = TransportOutcome(semantic_cache_status="exact-hit")
        headers = build_routing_headers(
            "gpt-4",
            transport_outcome=outcome,
            cache_hit=False,
        )
        assert "x-lattice-cache-hit" not in headers
        assert headers["x-lattice-semantic-cache"] == "exact-hit"

    def test_tri_state_stream_resumed_false_suppresses(self) -> None:
        from lattice.core.telemetry import TransportOutcome

        outcome = TransportOutcome(
            stream_resumed=True,
            stream_resume_fallback_reason="token_expired",
        )
        headers = build_routing_headers(
            "gpt-4",
            transport_outcome=outcome,
            stream_resumed=False,
        )
        assert "x-lattice-stream-resumed" not in headers
        assert headers["x-lattice-stream-resume-fallback-reason"] == "token_expired"


class TestStatsExposesOperatorSurface:
    """Phase 5: /stats exposes maintenance, ignored chunks, and transport rollup."""

    def test_stats_exposes_maintenance_when_configured(self) -> None:
        from lattice.core.config import LatticeConfig
        from lattice.proxy.server import create_app

        config = LatticeConfig(
            provider_base_url="http://127.0.0.1:11434",
            provider_base_urls={"openai": "https://api.openai.com"},
        )
        app = create_app(config=config)
        client = TestClient(app)

        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "maintenance" in data
        maint = data["maintenance"]
        assert "last_attempt" in maint
        assert "last_success" in maint
        assert "interval_seconds" in maint
        assert "throttled_tick_count" in maint
        assert "callback_failures" in maint
        assert "last_result_summary" in maint

    def test_stats_exposes_ignored_chunks(self) -> None:
        from lattice.core.config import LatticeConfig
        from lattice.proxy.server import create_app

        config = LatticeConfig(
            provider_base_url="http://127.0.0.1:11434",
            provider_base_urls={"openai": "https://api.openai.com"},
        )
        app = create_app(config=config)
        client = TestClient(app)

        response = client.get("/stats")
        data = response.json()
        assert "ignored_chunks" in data
        assert data["ignored_chunks"]["total"] == 0
        assert isinstance(data["ignored_chunks"]["by_provider"], dict)

    def test_stats_stable_when_maintenance_not_run(self) -> None:
        from lattice.core.config import LatticeConfig
        from lattice.proxy.server import create_app

        config = LatticeConfig(
            provider_base_url="http://127.0.0.1:11434",
            provider_base_urls={"openai": "https://api.openai.com"},
        )
        app = create_app(config=config)
        client = TestClient(app)

        response = client.get("/stats")
        data = response.json()
        # Maintenance exists but has not yet run (last_attempt == 0)
        maint = data["maintenance"]
        assert maint["last_attempt"] == 0.0
        assert maint["last_success"] == 0.0
        assert maint["throttled_tick_count"] == 0

    def test_stats_fallback_includes_stream_resume_count(self) -> None:
        from lattice.core.config import LatticeConfig
        from lattice.proxy.server import create_app

        config = LatticeConfig(
            provider_base_url="http://127.0.0.1:11434",
            provider_base_urls={"openai": "https://api.openai.com"},
        )
        app = create_app(config=config)
        client = TestClient(app)

        response = client.get("/stats")
        data = response.json()
        assert "fallbacks" in data
        assert "stream_resume_fallback_reason_count" in data["fallbacks"]

    def test_stats_transport_outcome_rollup_present(self) -> None:
        from lattice.core.config import LatticeConfig
        from lattice.proxy.server import create_app

        config = LatticeConfig(
            provider_base_url="http://127.0.0.1:11434",
            provider_base_urls={"openai": "https://api.openai.com"},
        )
        app = create_app(config=config)
        client = TestClient(app)

        # Trigger a request to populate some telemetry
        _ = client.post(
            "/v1/chat/completions",
            json={
                "model": "openai/gpt-4",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        response = client.get("/stats")
        data = response.json()
        assert "transport_outcome_rollup" in data

    def test_fallback_rollups_consistent_with_downgrades(self) -> None:
        from lattice.core.config import LatticeConfig
        from lattice.proxy.server import create_app

        config = LatticeConfig(
            provider_base_url="http://127.0.0.1:11434",
            provider_base_urls={"openai": "https://api.openai.com"},
        )
        app = create_app(config=config)
        client = TestClient(app)

        response = client.get("/stats")
        data = response.json()

        fallbacks = data.get("fallbacks", {})
        downgrades = data.get("downgrades", {}).get("counts", {})

        # Fallback counts should be non-negative
        assert fallbacks["http2_to_http11_count"] >= 0
        assert fallbacks["delta_to_full_prompt_count"] >= 0
        assert fallbacks["stream_resume_fallback_reason_count"] >= 0
        # Downgrade counts should be consistent with telemetry
        assert isinstance(downgrades, dict)
