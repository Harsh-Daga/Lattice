"""Shared maintenance coordinator with throttling.

Centralizes periodic cleanup tasks (stale stream eviction, cache expiry,
etc.) so they do not run on every request and are not coupled to a single
handler surface.

Design
------
- A background loop (``start()`` / ``stop()``) runs maintenance on a fixed
  interval. This is the primary path; it never depends on chat-completion
  traffic.
- Request-path ticks serve as an opportunistic backup — throttled to avoid
  double-running.
- State is observable via ``stats()`` so operators can tell whether
  cleanup actually happened.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import time
from collections.abc import Awaitable, Callable
from typing import Any


@dataclasses.dataclass
class MaintenanceResult:
    """Result of one maintenance tick."""

    stale_streams_removed: int = 0
    stale_cache_entries_removed: int = 0
    did_work: bool = False


class MaintenanceCoordinator:
    """Throttled maintenance scheduler with background loop.

    Runs registered maintenance callbacks at most once per *interval_seconds*.
    Safe to call from multiple request handlers; only the first caller in each
    interval actually triggers work.

    Primary path: background loop (call ``start()`` at app startup).
    Fallback path: request-bound ``tick()`` for low-traffic deployments.
    """

    def __init__(self, interval_seconds: float = 60.0) -> None:
        self.interval_seconds = interval_seconds
        self._last_run: float = 0.0
        self._has_run: bool = False
        self._lock = asyncio.Lock()
        self._callbacks: list[tuple[str, Callable[[], Awaitable[MaintenanceResult]]]] = []
        # Observable state
        self.last_attempt: float = 0.0
        self.last_success: float = 0.0
        self.throttled_tick_count: int = 0
        self.callback_failures: dict[str, int] = {}
        self._last_result_summary: dict[str, Any] = {}
        # Background loop control
        self._background_task: asyncio.Task[None] | None = None

    def register(
        self,
        name: str,
        callback: Callable[[], Awaitable[MaintenanceResult]],
    ) -> None:
        """Register a named maintenance callback."""
        self._callbacks.append((name, callback))

    async def start(self) -> None:
        """Begin the background maintenance loop.

        Idempotent — safe to call multiple times.
        """
        if self._background_task is not None and not self._background_task.done():
            return
        self._background_task = asyncio.create_task(self._background_loop())

    async def stop(self) -> None:
        """Stop the background maintenance loop cleanly."""
        if self._background_task is not None and not self._background_task.done():
            self._background_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._background_task
        self._background_task = None

    async def tick(self) -> dict[str, MaintenanceResult]:
        """Run maintenance if the interval has elapsed since the last run.

        Returns a dict of ``{name: MaintenanceResult}`` for all registered
        callbacks, regardless of whether they actually ran this time.
        Callable from request handlers as an opportunistic fallback.
        """
        now = time.monotonic()
        async with self._lock:
            if self._has_run and now - self._last_run < self.interval_seconds:
                self.throttled_tick_count += 1
                return {}
            self._last_run = now
            self._has_run = True

        return await self._run_all()

    async def _run_all(self) -> dict[str, MaintenanceResult]:
        """Execute all registered callbacks and update observable state."""
        now = time.monotonic()
        self.last_attempt = now

        results: dict[str, MaintenanceResult] = {}
        any_failure = False
        for name, callback in self._callbacks:
            try:
                result = await callback()
                results[name] = result
            except Exception:
                # Maintenance must never block; record failure and continue.
                results[name] = MaintenanceResult()
                self.callback_failures[name] = self.callback_failures.get(name, 0) + 1
                any_failure = True

        if not any_failure:
            self.last_success = now
        self._last_result_summary = {
            name: {
                "stale_streams_removed": r.stale_streams_removed,
                "stale_cache_entries_removed": r.stale_cache_entries_removed,
                "did_work": r.did_work,
            }
            for name, r in results.items()
        }
        return results

    async def _background_loop(self) -> None:
        """Continuous background loop that runs maintenance on every interval."""
        while True:
            await asyncio.sleep(self.interval_seconds)
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass  # Non-fatal — let the next cycle try again

    def stats(self) -> dict[str, Any]:
        """Return observable maintenance state for /stats."""
        return {
            "last_attempt": self.last_attempt,
            "last_success": self.last_success,
            "interval_seconds": self.interval_seconds,
            "throttled_tick_count": self.throttled_tick_count,
            "callback_failures": dict(self.callback_failures),
            "last_result_summary": dict(self._last_result_summary),
        }

    @property
    def seconds_since_last_run(self) -> float:
        """Time elapsed since the last successful maintenance tick."""
        return time.monotonic() - self._last_run
