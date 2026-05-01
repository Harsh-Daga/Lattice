"""Tests for provider batch accumulator.

Covers:
- Basic accumulation and flush
- Max batch size trigger
- Unsupported provider rejection
- Force flush
- Stats reporting
- Background dispatch loop
- Graceful shutdown with flush
"""

from __future__ import annotations

import asyncio

import pytest

from lattice.core.batch_accumulator import BatchAccumulator, BatchResult


class TestBatchAccumulator:
    @pytest.fixture
    def accumulator(self):
        return BatchAccumulator(max_hold_seconds=60, max_batch_size=3, enabled=True)

    async def test_submit_supported_provider(self, accumulator) -> None:
        future = await accumulator.submit(
            {"messages": [{"role": "user", "content": "hi"}]},
            provider="openai",
            model="gpt-4",
        )
        stats = accumulator.stats
        assert stats["total_pending"] == 1
        assert stats["queue_sizes"]["openai"] == 1
        # Cancel the future to avoid unhandled warning
        future.cancel()

    async def test_submit_unsupported_provider(self, accumulator) -> None:
        future = await accumulator.submit(
            {"messages": [{"role": "user", "content": "hi"}]},
            provider="anthropic",
            model="claude-3",
        )
        with pytest.raises(RuntimeError, match="not supported"):
            await future

    async def test_disabled_rejects(self) -> None:
        disabled = BatchAccumulator(enabled=False)
        future = await disabled.submit(
            {"messages": [{"role": "user", "content": "hi"}]},
            provider="openai",
            model="gpt-4",
        )
        with pytest.raises(RuntimeError):
            await future

    async def test_flush_returns_results(self, accumulator) -> None:
        futures = []
        for i in range(3):
            fut = await accumulator.submit(
                {"messages": [{"role": "user", "content": f"msg {i}"}]},
                provider="openai",
                model="gpt-4",
            )
            futures.append(fut)

        results = await accumulator.flush()
        assert len(results) == 1
        batch = results[0]
        assert batch.success is True
        assert batch.request_count == 3

        # All futures resolved
        for fut in futures:
            result = await fut
            assert result["status"] == "completed"

    async def test_max_batch_size_trigger(self, accumulator) -> None:
        # max_batch_size=3, submit 3 to trigger immediate flush
        futures = []
        for i in range(3):
            fut = await accumulator.submit(
                {"messages": [{"role": "user", "content": f"msg {i}"}]},
                provider="openai",
                model="gpt-4",
            )
            futures.append(fut)

        # Give the background flush task a moment to run
        await asyncio.sleep(0.2)

        # Futures should be resolved
        for fut in futures:
            assert fut.done()
            result = await fut
            assert result["status"] == "completed"

    async def test_start_stop(self, accumulator) -> None:
        await accumulator.start()
        assert accumulator._dispatch_task is not None
        await accumulator.stop()
        assert accumulator._dispatch_task is None

    async def test_force_flush_empty(self, accumulator) -> None:
        results = await accumulator.flush()
        assert results == []

    async def test_stats(self, accumulator) -> None:
        await accumulator.submit(
            {"messages": [{"role": "user", "content": "hi"}]},
            provider="openai",
            model="gpt-4",
        )
        stats = accumulator.stats
        assert stats["enabled"] is True
        assert stats["max_hold_seconds"] == 60
        assert stats["max_batch_size"] == 3
        assert stats["total_pending"] == 1
