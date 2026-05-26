"""Burst admission rejects with QueueFullError."""

from __future__ import annotations

import asyncio

import pytest

from lattice.transport.backpressure import Backpressure, QueueFullError


@pytest.mark.asyncio
async def test_burst_rejects_half() -> None:
    bp = Backpressure(max_in_flight=100, overflow="reject")
    admitted = 0
    rejected = 0

    async def one() -> None:
        nonlocal admitted, rejected
        try:
            async with bp.admit():
                admitted += 1
                await asyncio.sleep(0.05)
        except QueueFullError:
            rejected += 1

    await asyncio.gather(*[one() for _ in range(200)])
    assert admitted >= 90
    assert rejected >= 90
