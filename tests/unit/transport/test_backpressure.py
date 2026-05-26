"""Backpressure queue admission."""

from __future__ import annotations

import pytest

from lattice.transport.backpressure import Backpressure, QueueFullError


@pytest.mark.asyncio
async def test_reject_when_full() -> None:
    bp = Backpressure(max_in_flight=1, overflow="reject")
    async with bp.admit():
        with pytest.raises(QueueFullError):
            async with bp.admit():
                pass
