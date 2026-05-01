"""Unit tests for BatchingEngine and BatchingTransform.

Tests cover:
- BatchKey grouping and compatibility detection
- Batching window behavior
- Error handling (streaming rejection, provider failures)
- Lifecycle (shutdown, flush)
- Pipeline integration (BatchingTransform metadata)
"""

from __future__ import annotations

import asyncio

import pytest

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.pipeline import CompressorPipeline
from lattice.core.result import unwrap
from lattice.core.transport import Message, Request, Response
from lattice.transforms.batching import (
    BatchedRequest,
    BatchedResponse,
    BatchingEngine,
    BatchingTransform,
    BatchKey,
)

# =============================================================================
# BatchKey
# =============================================================================

class TestBatchKey:
    def test_from_request_no_tools(self) -> None:
        req = Request(model="gpt-4", temperature=0.7, messages=[Message(role="user", content="hi")])
        key = BatchKey.from_request(req)
        assert key.model == "gpt-4"
        assert key.temperature == 0.7
        assert key.tools_hash == ""

    def test_from_request_with_tools(self) -> None:
        req = Request(
            model="gpt-4",
            tools=[{"type": "function", "function": {"name": "search"}}],
            messages=[Message(role="user", content="hi")],
        )
        key = BatchKey.from_request(req)
        assert key.tools_hash != ""
        # Same tools → same hash
        key2 = BatchKey.from_request(req)
        assert key == key2

    def test_batch_key_hashable(self) -> None:
        k1 = BatchKey(model="gpt-4", temperature=0.7, max_tokens=100, top_p=1.0, stream=False)
        k2 = BatchKey(model="gpt-4", temperature=0.7, max_tokens=100, top_p=1.0, stream=False)
        assert k1 == k2
        assert hash(k1) == hash(k2)

    def test_different_models_not_equal(self) -> None:
        k1 = BatchKey(model="gpt-4", temperature=0.7, max_tokens=None, top_p=None, stream=False)
        k2 = BatchKey(model="gpt-3.5", temperature=0.7, max_tokens=None, top_p=None, stream=False)
        assert k1 != k2


# =============================================================================
# BatchingEngine
# =============================================================================

class TestBatchingEngine:
    """Tests for BatchingEngine core logic."""

    @pytest.fixture
    def engine(self) -> BatchingEngine:
        async def fake_caller(batched: BatchedRequest) -> BatchedResponse:
            return BatchedResponse(
                choices=[
                    {"index": i, "message": {"role": "assistant", "content": f"resp_{i}"}, "finish_reason": "stop"}
                    for i in range(len(batched.messages))
                ],
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                model="gpt-4",
            )

        return BatchingEngine(max_batch_size=4, max_wait_ms=5.0, provider_caller=fake_caller)

    @pytest.mark.asyncio
    async def test_single_request_no_batch(self, engine: BatchingEngine) -> None:
        """A single request within the window should still be dispatched."""
        req = Request(messages=[Message(role="user", content="hello")])
        ctx = TransformContext()
        resp = await engine.submit(req, ctx)
        assert resp.content.startswith("resp_")

    @pytest.mark.asyncio
    async def test_batch_fills_and_dispatches(self, engine: BatchingEngine) -> None:
        """Multiple requests fill the batch and get individual responses."""
        reqs = [
            Request(messages=[Message(role="user", content=f"msg_{i}")])
            for i in range(3)
        ]
        ctxs = [TransformContext() for _ in range(3)]
        tasks = [asyncio.create_task(engine.submit(reqs[i], ctxs[i])) for i in range(3)]
        results = await asyncio.gather(*tasks)
        for _i, resp in enumerate(results):
            assert isinstance(resp, Response)
            assert resp.content.startswith("resp_")

    @pytest.mark.asyncio
    async def test_streaming_rejected(self, engine: BatchingEngine) -> None:
        """Streaming requests are not eligible for batching."""
        req = Request(messages=[Message(role="user", content="hi")], stream=True)
        ctx = TransformContext()
        with pytest.raises(ValueError, match="Streaming"):
            await engine.submit(req, ctx)

    @pytest.mark.asyncio
    async def test_provider_error_fails_all(self) -> None:
        """If provider fails, all pending futures raise."""
        async def fail_caller(_batched: BatchedRequest) -> BatchedResponse:
            raise RuntimeError("provider down")

        engine = BatchingEngine(max_batch_size=2, max_wait_ms=5.0, provider_caller=fail_caller)
        reqs = [
            Request(messages=[Message(role="user", content=f"msg_{i}")])
            for i in range(2)
        ]
        ctxs = [TransformContext() for _ in range(2)]
        tasks = [asyncio.create_task(engine.submit(reqs[i], ctxs[i])) for i in range(2)]
        for task in tasks:
            with pytest.raises(RuntimeError, match="provider down"):
                await task

    @pytest.mark.asyncio
    async def test_shutdown_flushes_pending(self) -> None:
        """Shutdown flushes all pending requests."""
        async def fake_caller(_batched: BatchedRequest) -> BatchedResponse:
            return BatchedResponse(
                choices=[{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                usage={},
                model="gpt-4",
            )

        engine = BatchingEngine(max_batch_size=10, max_wait_ms=100.0, provider_caller=fake_caller)
        req = Request(messages=[Message(role="user", content="hello")])
        ctx = TransformContext()

        # Start submission but don't await (it will wait for window)
        task = asyncio.create_task(engine.submit(req, ctx))
        # Immediately shutdown — should flush and resolve
        await engine.shutdown()
        resp = await task
        assert resp.content == "ok"

    @pytest.mark.asyncio
    async def test_stats_reports_pending_queues(self) -> None:
        async def fake_caller(batched: BatchedRequest) -> BatchedResponse:
            return BatchedResponse(
                choices=[
                    {
                        "index": i,
                        "message": {"role": "assistant", "content": f"resp_{i}"},
                        "finish_reason": "stop",
                    }
                    for i in range(len(batched.messages))
                ],
                usage={"prompt_tokens": 1, "completion_tokens": 1},
                model="gpt-4",
            )

        engine = BatchingEngine(max_batch_size=4, max_wait_ms=25.0, provider_caller=fake_caller)
        req = Request(model="openai/gpt-4", messages=[Message(role="user", content="hello")])
        ctx = TransformContext(provider="openai")
        task = asyncio.create_task(engine.submit(req, ctx))
        await asyncio.sleep(0)

        stats = await engine.stats()
        assert stats["enabled"] is True
        assert stats["total_pending"] >= 1
        assert stats["queue_sizes"].get("openai", 0) >= 1

        await task

    @pytest.mark.asyncio
    async def test_incompatible_requests_not_batched(self) -> None:
        """Requests with different BatchKeys are not batched together."""
        responses: list[str] = []

        async def counting_caller(_batched: BatchedRequest) -> BatchedResponse:
            responses.append("call")
            return BatchedResponse(
                choices=[{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                usage={},
                model="gpt-4",
            )

        engine = BatchingEngine(max_batch_size=10, max_wait_ms=3.0, provider_caller=counting_caller)
        # Two requests with different models → different BatchKeys
        req1 = Request(model="gpt-4", messages=[Message(role="user", content="a")])
        req2 = Request(model="gpt-3.5", messages=[Message(role="user", content="b")])

        t1 = asyncio.create_task(engine.submit(req1, TransformContext()))
        t2 = asyncio.create_task(engine.submit(req2, TransformContext()))
        await asyncio.gather(t1, t2)
        # Should be 2 separate calls because BatchKeys differ
        assert len(responses) == 2

    def test_effective_batch_size_capped_by_tacc(self) -> None:
        class _FakeTacc:
            def window_size(self, provider: str) -> int:
                return 2 if provider == "openai" else 5

        engine = BatchingEngine(
            max_batch_size=8,
            max_wait_ms=5.0,
            provider_caller=lambda _batched: None,
            tacc=_FakeTacc(),
        )
        assert engine.effective_batch_size("openai") == 2
        assert engine.effective_batch_size("anthropic") == 5


# =============================================================================
# BatchingTransform (pipeline integration)
# =============================================================================

class TestBatchingTransform:
    """Tests for the pipeline-friendly BatchingTransform."""

    def test_streaming_not_eligible(self) -> None:
        bt = BatchingTransform()
        req = Request(messages=[Message(role="user", content="hi")], stream=True)
        ctx = TransformContext()
        result = bt.process(req, ctx)
        assert result.value == req  # Ok(Request)
        assert ctx.metrics["transforms"]["batching"]["eligible"] is False

    def test_non_streaming_eligible(self) -> None:
        bt = BatchingTransform()
        req = Request(model="gpt-4", messages=[Message(role="user", content="hi")])
        ctx = TransformContext()
        result = bt.process(req, ctx)
        assert result.value == req
        assert ctx.metrics["transforms"]["batching"]["eligible"] is True

    @pytest.mark.asyncio
    async def test_in_pipeline(self) -> None:
        config = LatticeConfig()
        pipeline = CompressorPipeline(config=config)
        pipeline.register(BatchingTransform())
        req = Request(model="gpt-4", messages=[Message(role="user", content="hi")])
        ctx = TransformContext()
        result = await pipeline.process(req, ctx)
        unwrap(result)
        assert "batching" in ctx.transforms_applied
