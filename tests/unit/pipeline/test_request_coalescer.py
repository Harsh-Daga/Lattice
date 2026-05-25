"""Tests for in-process request coalescer (not provider batch API)."""

from __future__ import annotations

import pytest

from lattice.pipeline.request_coalescer import RequestCoalescer


class TestRequestCoalescer:
    @pytest.fixture
    def coalescer(self):
        return RequestCoalescer(max_hold_seconds=60, max_batch_size=3, enabled=True)

    async def test_submit_queues_request(self, coalescer) -> None:
        future = await coalescer.submit(
            {"messages": [{"role": "user", "content": "hi"}]},
            provider="openai",
            model="gpt-4",
        )
        assert coalescer.stats["total_pending"] == 1
        future.cancel()

    async def test_disabled_rejects(self) -> None:
        disabled = RequestCoalescer(enabled=False)
        future = await disabled.submit(
            {"messages": [{"role": "user", "content": "hi"}]},
            provider="openai",
            model="gpt-4",
        )
        with pytest.raises(RuntimeError, match="batch API not integrated"):
            await future

    async def test_flush_signals_batch_unavailable(self, coalescer) -> None:
        futures = []
        for i in range(2):
            fut = await coalescer.submit(
                {"messages": [{"role": "user", "content": f"msg {i}"}]},
                provider="openai",
                model="gpt-4",
            )
            futures.append(fut)

        results = await coalescer.flush()
        assert len(results) == 1
        assert results[0].success is False
        for fut in futures:
            with pytest.raises(RuntimeError, match="batch API not integrated"):
                await fut

    async def test_start_stop(self, coalescer) -> None:
        await coalescer.start()
        assert coalescer._dispatch_task is not None
        await coalescer.stop()
        assert coalescer._dispatch_task is None
