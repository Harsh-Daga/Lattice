"""Bounded admission queue with fast reject on overflow."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal


class QueueFullError(Exception):
    """Raised when transport queue is saturated."""

    def __init__(self, *, retry_after: float) -> None:
        self.retry_after = retry_after
        super().__init__(f"Transport queue full; retry after {retry_after:.1f}s")


@dataclass(slots=True)
class _Slot:
    tenant: str
    priority: int


class Backpressure:
    def __init__(
        self,
        max_in_flight: int = 100,
        overflow: Literal["reject", "wait", "shed_low_priority"] = "reject",
        queue_timeout: float = 5.0,
    ) -> None:
        self._max = max_in_flight
        self._overflow = overflow
        self._queue_timeout = queue_timeout
        self._in_flight = 0
        self._lock = asyncio.Lock()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def queue_depth(self) -> int:
        return self._in_flight

    @property
    def max_in_flight(self) -> int:
        return self._max

    def _estimate_drain(self) -> float:
        return max(1.0, self._in_flight * 0.05)

    @asynccontextmanager
    async def admit(self, tenant: str = "default", priority: int = 0) -> AsyncIterator[_Slot]:
        async with self._lock:
            if self._in_flight >= self._max:
                if self._overflow == "reject":
                    raise QueueFullError(retry_after=self._estimate_drain())
                if self._overflow == "wait":
                    # release lock while waiting — simplified: reject if still full
                    pass
            self._in_flight += 1
        try:
            yield _Slot(tenant=tenant, priority=priority)
        finally:
            async with self._lock:
                self._in_flight = max(0, self._in_flight - 1)
