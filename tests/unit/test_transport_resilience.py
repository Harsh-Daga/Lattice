"""Tests for lattice.providers.transport resilience and stall detection.

LATTICE does NOT perform model fallback / routing. These tests verify:
- configure_resilience sets stall timeout
- completion() retries the SAME model on transient errors
- completion_stream_with_stall_detect yields chunks and detects stalls
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from lattice.core.errors import ProviderError, ProviderTimeoutError
from lattice.providers.transport import (
    ConnectionPoolManager,
    DirectHTTPProvider,
    ProviderRegistry,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def provider() -> DirectHTTPProvider:
    """Fresh provider with no real network."""
    return DirectHTTPProvider(
        registry=ProviderRegistry(),
        pool=ConnectionPoolManager(http2=False),
        default_api_key="fake",
        provider_base_urls={
            "openai": "https://api.openai.com",
            "ollama": "http://127.0.0.1:11434",
        },
    )


# =============================================================================
# Resilience configuration
# =============================================================================


class TestConfigureResilience:
    def test_sets_stall_timeout(self) -> None:
        p = DirectHTTPProvider()
        p.configure_resilience(stall_timeout=15.0)
        assert p._stall_timeout == 15.0

    def test_defaults(self) -> None:
        p = DirectHTTPProvider()
        assert p._stall_timeout == 30.0

    def test_stream_optimizer_not_implemented(self) -> None:
        # Stream optimizer was removed in Phase 0; verify it is not exposed.
        p = DirectHTTPProvider()
        assert not hasattr(p, "_stream_optimizer_enabled")


# =============================================================================
# completion() retry behaviour — same model, transient errors
# =============================================================================


class TestCompletionRetry:
    async def test_retries_on_429_and_succeeds(self, provider: DirectHTTPProvider) -> None:
        """Provider retries the SAME model on 429 and succeeds on second try."""
        p = provider

        class OkResp:
            status_code = 200
            headers = {}

            @property
            def is_success(self) -> bool:
                return True

            def json(self) -> dict[str, Any]:
                return {
                    "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }

        class RateLimitResp:
            status_code = 429
            headers = {"retry-after": "0"}

            @property
            def is_success(self) -> bool:
                return False

            async def aread(self) -> bytes:
                return b"rate limited"

            def json(self) -> dict[str, Any]:
                return {"error": "rate limited"}

        call_count = 0

        class FakeClient:
            async def post(self, *_args: Any, **_kwargs: Any) -> Any:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return RateLimitResp()
                return OkResp()

        p.pool._clients = {("openai", "https://api.openai.com"): FakeClient()}

        # Patch sleep so test runs instantly
        import asyncio

        orig_sleep = asyncio.sleep
        asyncio.sleep = AsyncMock()  # type: ignore[assignment]
        try:
            resp = await p.completion(
                model="openai/gpt-4",
                messages=[{"role": "user", "content": "hi"}],
            )
            assert resp.content == "OK"
            assert call_count == 2  # retried once
        finally:
            asyncio.sleep = orig_sleep  # type: ignore[assignment]

    async def test_raises_after_exhausted_retries(self, provider: DirectHTTPProvider) -> None:
        """All retries exhausted → raises ProviderError."""
        p = provider

        class AlwaysFail:
            status_code = 503
            headers = {}

            @property
            def is_success(self) -> bool:
                return False

            async def aread(self) -> bytes:
                return b"unavailable"

            def json(self) -> dict[str, Any]:
                return {"error": "unavailable"}

        class FakeClient:
            async def post(self, *_args: Any, **_kwargs: Any) -> Any:
                return AlwaysFail()

        p.pool._clients = {("openai", "https://api.openai.com"): FakeClient()}

        import asyncio

        orig_sleep = asyncio.sleep
        asyncio.sleep = AsyncMock()  # type: ignore[assignment]
        try:
            with pytest.raises(ProviderError):
                await p.completion(
                    model="openai/gpt-4",
                    messages=[{"role": "user", "content": "hi"}],
                )
        finally:
            asyncio.sleep = orig_sleep  # type: ignore[assignment]

    async def test_no_retry_on_400(self, provider: DirectHTTPProvider) -> None:
        """4xx client errors are not retried."""
        p = provider

        class BadRequest:
            status_code = 400
            headers = {}

            @property
            def is_success(self) -> bool:
                return False

            async def aread(self) -> bytes:
                return b"bad request"

            def json(self) -> dict[str, Any]:
                return {"error": "bad request"}

        class FakeClient:
            async def post(self, *_args: Any, **_kwargs: Any) -> Any:
                return BadRequest()

        p.pool._clients = {("openai", "https://api.openai.com"): FakeClient()}

        with pytest.raises(ProviderError):
            await p.completion(
                model="openai/gpt-4",
                messages=[{"role": "user", "content": "hi"}],
            )


# =============================================================================
# completion_stream_with_stall_detect
# =============================================================================


class TestCompletionStreamWithStallDetect:
    async def test_yields_chunks_normally(self, provider: DirectHTTPProvider) -> None:
        """Normal operation: chunks pass through."""
        p = provider

        class FakeResp:
            status_code = 200
            headers = {}

            async def __aenter__(self) -> FakeResp:
                return self

            async def __aexit__(self, *args: Any) -> None:
                return None

            async def aiter_text(self) -> AsyncGenerator[str, None]:
                yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
                yield "data: [DONE]\n\n"

            def raise_for_status(self) -> None:
                pass

        fake_client = MagicMock()
        fake_client.stream = MagicMock(return_value=FakeResp())
        p.pool._clients = {("openai", "https://api.openai.com"): fake_client}

        chunks: list[dict[str, Any]] = []
        async for chunk in p.completion_stream_with_stall_detect(
            model="openai/gpt-4",
            messages=[{"role": "user", "content": "hi"}],
        ):
            chunks.append(chunk)

        assert len(chunks) >= 1
        assert any("Hello" in str(c) for c in chunks)

    async def test_no_stall_on_fast_stream(self, provider: DirectHTTPProvider) -> None:
        """Fast stream never triggers stall."""
        p = provider
        p._stall_timeout = 0.001  # 1ms stall threshold

        class FakeResp:
            status_code = 200
            headers = {}

            async def __aenter__(self) -> FakeResp:
                return self

            async def __aexit__(self, *args: Any) -> None:
                return None

            async def aiter_text(self) -> AsyncGenerator[str, None]:
                yield 'data: {"choices": [{"delta": {"content": "Fast"}}]}\n\n'

            def raise_for_status(self) -> None:
                pass

        fake_client = MagicMock()
        fake_client.stream = MagicMock(return_value=FakeResp())
        p.pool._clients = {("openai", "https://api.openai.com"): fake_client}

        chunks = [
            c
            async for c in p.completion_stream_with_stall_detect(
                model="openai/gpt-4",
                messages=[{"role": "user", "content": "hi"}],
            )
        ]
        assert any("Fast" in str(c) for c in chunks)

    async def test_stall_detector_can_trigger_timeout(self, provider: DirectHTTPProvider) -> None:
        p = provider
        p.stall_detector.is_stalled = MagicMock(return_value=True)  # type: ignore[method-assign]

        class FakeResp:
            status_code = 200
            headers = {}

            async def __aenter__(self) -> FakeResp:
                return self

            async def __aexit__(self, *args: Any) -> None:
                return None

            async def aiter_text(self) -> AsyncGenerator[str, None]:
                yield 'data: {"choices": [{"delta": {"content": "Slow"}}]}\n\n'

            def raise_for_status(self) -> None:
                pass

        fake_client = MagicMock()
        fake_client.stream = MagicMock(return_value=FakeResp())
        p.pool._clients = {("openai", "https://api.openai.com"): fake_client}

        with pytest.raises(ProviderTimeoutError):
            _ = [
                c
                async for c in p.completion_stream_with_stall_detect(
                    model="openai/gpt-4",
                    messages=[{"role": "user", "content": "hi"}],
                )
            ]

    async def test_tacc_feedback_on_stream_with_stall_detect_success(
        self, provider: DirectHTTPProvider
    ) -> None:
        p = provider
        p.tacc.acquire_request = AsyncMock(return_value=True)  # type: ignore[method-assign]
        p.tacc.after_response = AsyncMock()  # type: ignore[method-assign]

        class FakeResp:
            status_code = 200
            headers = {}

            async def __aenter__(self) -> FakeResp:
                return self

            async def __aexit__(self, *args: Any) -> None:
                return None

            async def aiter_text(self) -> AsyncGenerator[str, None]:
                yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
                yield "data: [DONE]\n\n"

            def raise_for_status(self) -> None:
                pass

        fake_client = MagicMock()
        fake_client.stream = MagicMock(return_value=FakeResp())
        p.pool._clients = {("openai", "https://api.openai.com"): fake_client}

        _ = [
            c
            async for c in p.completion_stream_with_stall_detect(
                model="openai/gpt-4",
                messages=[{"role": "user", "content": "hi"}],
            )
        ]

        p.tacc.acquire_request.assert_awaited_once()
        before_args, before_kwargs = p.tacc.acquire_request.await_args
        assert before_args == ("openai",)
        assert before_kwargs.get("estimated_tokens", 0) > 0
        p.tacc.after_response.assert_awaited()
        args, kwargs = p.tacc.after_response.await_args
        assert kwargs == {}
        assert args[0] == "openai"
        assert args[3] == 200


class TestCompletionStream:
    async def test_tacc_feedback_on_stream_success(self, provider: DirectHTTPProvider) -> None:
        p = provider
        p.tacc.acquire_request = AsyncMock(return_value=True)  # type: ignore[method-assign]
        p.tacc.after_response = AsyncMock()  # type: ignore[method-assign]

        class FakeResp:
            status_code = 200
            headers = {}

            async def __aenter__(self) -> FakeResp:
                return self

            async def __aexit__(self, *args: Any) -> None:
                return None

            async def aiter_text(self) -> AsyncGenerator[str, None]:
                yield 'data: {"choices": [{"delta": {"content": "Stream"}}]}\n\n'
                yield "data: [DONE]\n\n"

            def raise_for_status(self) -> None:
                pass

        fake_client = MagicMock()
        fake_client.stream = MagicMock(return_value=FakeResp())
        p.pool._clients = {("openai", "https://api.openai.com"): fake_client}

        chunks = [
            c
            async for c in p.completion_stream(
                model="openai/gpt-4",
                messages=[{"role": "user", "content": "hi"}],
            )
        ]
        assert any("Stream" in str(c) for c in chunks)

        p.tacc.acquire_request.assert_awaited_once()
        before_args, before_kwargs = p.tacc.acquire_request.await_args
        assert before_args == ("openai",)
        assert before_kwargs.get("estimated_tokens", 0) > 0
        p.tacc.after_response.assert_awaited()
        args, kwargs = p.tacc.after_response.await_args
        assert kwargs == {}
        assert args[0] == "openai"
        assert args[3] == 200

    async def test_cancelled_stream_releases_tacc_slot(self, provider: DirectHTTPProvider) -> None:
        import asyncio

        p = provider

        class FakeResp:
            status_code = 200
            headers = {}

            async def __aenter__(self) -> FakeResp:
                return self

            async def __aexit__(self, *args: Any) -> None:
                return None

            async def aiter_text(self) -> AsyncGenerator[str, None]:
                await asyncio.sleep(60)
                if False:
                    yield ""

            def raise_for_status(self) -> None:
                pass

        fake_client = MagicMock()
        fake_client.stream = MagicMock(return_value=FakeResp())
        p.pool._clients = {("openai", "https://api.openai.com"): fake_client}

        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.01):
                _ = [
                    c
                    async for c in p.completion_stream(
                        model="openai/gpt-4",
                        messages=[{"role": "user", "content": "hi"}],
                    )
                ]

        stats = p.tacc.stats("openai")
        assert stats.get("active_requests", 0) == 0

    async def test_break_on_done_releases_tacc_slot(self, provider: DirectHTTPProvider) -> None:
        p = provider

        class FakeResp:
            status_code = 200
            headers = {}

            async def __aenter__(self) -> FakeResp:
                return self

            async def __aexit__(self, *args: Any) -> None:
                return None

            async def aiter_text(self) -> AsyncGenerator[str, None]:
                yield 'data: {"choices": [{"delta": {"content": "Stream"}}]}\n\n'
                yield "data: [DONE]\n\n"

            def raise_for_status(self) -> None:
                pass

        fake_client = MagicMock()
        fake_client.stream = MagicMock(return_value=FakeResp())
        p.pool._clients = {("openai", "https://api.openai.com"): fake_client}

        async for chunk in p.completion_stream(
            model="openai/gpt-4",
            messages=[{"role": "user", "content": "hi"}],
        ):
            if chunk.get("done"):
                break

        stats = p.tacc.stats("openai")
        assert stats.get("active_requests", 0) == 0

    async def test_retries_on_ttft_timeout_before_first_chunk(
        self, provider: DirectHTTPProvider
    ) -> None:
        import asyncio

        p = provider
        p.tacc.before_request = AsyncMock(return_value=True)  # type: ignore[method-assign]
        p.tacc.after_response = AsyncMock()  # type: ignore[method-assign]
        p.pool.recycle_client = AsyncMock()  # type: ignore[method-assign]

        class FakeRespTimeout:
            status_code = 200
            headers = {}

            async def __aenter__(self) -> FakeRespTimeout:
                return self

            async def __aexit__(self, *args: Any) -> None:
                return None

            async def aiter_text(self) -> AsyncGenerator[str, None]:
                await asyncio.sleep(60)
                if False:
                    yield ""

            def raise_for_status(self) -> None:
                pass

        class FakeRespOK:
            status_code = 200
            headers = {}

            async def __aenter__(self) -> FakeRespOK:
                return self

            async def __aexit__(self, *args: Any) -> None:
                return None

            async def aiter_text(self) -> AsyncGenerator[str, None]:
                yield 'data: {"choices": [{"delta": {"content": "Recovered"}}]}\n\n'
                yield "data: [DONE]\n\n"

            def raise_for_status(self) -> None:
                pass

        fake_client_timeout = MagicMock()
        fake_client_timeout.stream = MagicMock(return_value=FakeRespTimeout())
        fake_client_ok = MagicMock()
        fake_client_ok.stream = MagicMock(return_value=FakeRespOK())
        calls = 0

        def _get_client(_provider: str, _base_url: str) -> Any:
            nonlocal calls
            calls += 1
            return fake_client_timeout if calls == 1 else fake_client_ok

        p.pool.get_client = _get_client  # type: ignore[method-assign]

        chunks = [
            c
            async for c in p.completion_stream(
                model="openai/gpt-4",
                messages=[{"role": "user", "content": "hi"}],
                metadata={"ttft_timeout_seconds": 0.01, "no_first_chunk_retries": 1},
            )
        ]
        assert any("Recovered" in str(c) for c in chunks)
        p.pool.recycle_client.assert_awaited_once()  # type: ignore[attr-defined]


# =============================================================================
# Integration with proxy server (header wiring)
# =============================================================================


class TestProxyHeaderWiring:
    def test_routing_headers_structure(self) -> None:
        from lattice.proxy.server import create_app

        app = create_app()
        assert app is not None

    def test_compression_header(self) -> None:
        from lattice.proxy.server import _build_routing_headers

        h = _build_routing_headers("gpt-4", compressed_tokens=100, original_tokens=200)
        assert h["x-lattice-compression"] == "50.00%"
        assert h["x-lattice-model"] == "gpt-4"

    def test_cost_and_cache_savings_headers(self) -> None:
        from lattice.proxy.server import _build_routing_headers

        h = _build_routing_headers(
            "gpt-4o",
            cache_hit=True,
            cached_tokens=800,
            cost_usd=0.0123456,
            cache_savings_usd=0.004321,
        )
        assert h["x-lattice-cache-hit"] == "true"
        assert h["x-lattice-cached-tokens"] == "800"
        assert h["x-lattice-cost-usd"] == "0.012346"
        assert h["x-lattice-cache-savings-usd"] == "0.004321"

    def test_runtime_contract_headers(self) -> None:
        from lattice.proxy.server import _build_routing_headers

        h = _build_routing_headers(
            "gpt-4o",
            runtime_tier="SIMPLE",
            runtime_mode="minimal",
            runtime_budget_ms=2.0,
            runtime_actual_ms=2.5,
            runtime_budget_exhausted=True,
            runtime_skipped_count=3,
        )
        assert h["x-lattice-runtime-tier"] == "SIMPLE"
        assert h["x-lattice-runtime-mode"] == "minimal"
        assert h["x-lattice-runtime-budget-ms"] == "2.00"
        assert h["x-lattice-runtime-actual-ms"] == "2.50"
        assert h["x-lattice-runtime-budget-exhausted"] == "true"
        assert h["x-lattice-runtime-skipped-transforms"] == "3"
