"""In-process request coalescing for compatibility-keyed concurrent calls.

This is not a provider Batch API client. Real OpenAI/Anthropic batch dispatch
lands in docs/refactor/20-non-chat-surfaces.md.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import time
from typing import Any

import structlog

logger = structlog.get_logger()

_BATCH_UNAVAILABLE = RuntimeError(
    "Provider batch API not integrated; use synchronous path "
    "(see docs/refactor/20-non-chat-surfaces.md)"
)


@dataclasses.dataclass(slots=True)
class CoalescedRequest:
    request_id: str
    request_body: dict[str, Any]
    provider: str
    model: str
    enqueued_at: float
    callback_future: asyncio.Future[dict[str, Any]]


@dataclasses.dataclass(slots=True)
class CoalescedResult:
    batch_id: str
    request_count: int
    success: bool
    results: list[dict[str, Any]]
    error: str | None = None


class RequestCoalescer:
    """Coalesces low-priority concurrent requests in-process (no provider batch API)."""

    def __init__(
        self,
        *,
        max_hold_seconds: int = 300,
        max_batch_size: int = 100,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.max_hold_seconds = max_hold_seconds
        self.max_batch_size = max_batch_size
        self._queues: dict[str, list[CoalescedRequest]] = {}
        self._lock = asyncio.Lock()
        self._dispatch_task: asyncio.Task[Any] | None = None

    async def start(self) -> None:
        if not self.enabled:
            return
        if self._dispatch_task is None:
            self._dispatch_task = asyncio.create_task(self._dispatch_loop())
            logger.info("request_coalescer_started")

    async def stop(self) -> None:
        if self._dispatch_task:
            self._dispatch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._dispatch_task
            self._dispatch_task = None
            logger.info("request_coalescer_stopped")

    async def submit(
        self,
        request_body: dict[str, Any],
        *,
        provider: str,
        model: str,
    ) -> asyncio.Future[dict[str, Any]]:
        future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        if not self.enabled:
            future.set_exception(_BATCH_UNAVAILABLE)
            return future

        req = CoalescedRequest(
            request_id=f"coalesce_{int(time.time() * 1000)}_{id(request_body)}",
            request_body=request_body,
            provider=provider.lower(),
            model=model,
            enqueued_at=time.time(),
            callback_future=future,
        )

        async with self._lock:
            self._queues.setdefault(provider.lower(), []).append(req)
            if len(self._queues[provider.lower()]) >= self.max_batch_size:
                asyncio.create_task(self._release_provider(provider.lower()))

        return future

    async def flush(self) -> list[CoalescedResult]:
        results: list[CoalescedResult] = []
        async with self._lock:
            providers = list(self._queues.keys())
        for provider in providers:
            result = await self._release_provider(provider)
            if result:
                results.append(result)
        return results

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_hold_seconds": self.max_hold_seconds,
            "max_batch_size": self.max_batch_size,
            "queue_sizes": {p: len(q) for p, q in self._queues.items()},
            "total_pending": sum(len(q) for q in self._queues.values()),
        }

    async def _dispatch_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.max_hold_seconds)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("request_coalescer_loop_error", error=str(exc))

    async def _release_provider(self, provider: str) -> CoalescedResult | None:
        async with self._lock:
            queue = self._queues.get(provider, [])
            if not queue:
                return None
            batch = queue[:]
            queue.clear()

        logger.info("request_coalescer_release", provider=provider, request_count=len(batch))
        for req in batch:
            if not req.callback_future.done():
                req.callback_future.set_exception(_BATCH_UNAVAILABLE)
        return CoalescedResult(
            batch_id=f"coalesce_{provider}_{int(time.time())}",
            request_count=len(batch),
            success=False,
            results=[],
            error=str(_BATCH_UNAVAILABLE),
        )


__all__ = ["RequestCoalescer", "CoalescedRequest", "CoalescedResult"]
