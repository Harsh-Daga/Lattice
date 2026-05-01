"""Tests for MaintenanceCoordinator throttling."""

from __future__ import annotations

import asyncio

import pytest

from lattice.core.maintenance import MaintenanceCoordinator, MaintenanceResult


@pytest.mark.asyncio
async def test_tick_runs_callbacks_and_records_work() -> None:
    coordinator = MaintenanceCoordinator(interval_seconds=0.0)
    calls: list[str] = []

    async def _stall_callback() -> MaintenanceResult:
        calls.append("stall")
        return MaintenanceResult(stale_streams_removed=3, did_work=True)

    async def _cache_callback() -> MaintenanceResult:
        calls.append("cache")
        return MaintenanceResult(stale_cache_entries_removed=5, did_work=True)

    coordinator.register("stall", _stall_callback)
    coordinator.register("cache", _cache_callback)

    results = await coordinator.tick()

    assert set(calls) == {"stall", "cache"}
    assert results["stall"].stale_streams_removed == 3
    assert results["stall"].did_work is True
    assert results["cache"].stale_cache_entries_removed == 5
    assert results["cache"].did_work is True


@pytest.mark.asyncio
async def test_tick_is_throttled() -> None:
    coordinator = MaintenanceCoordinator(interval_seconds=10.0)
    call_count = 0

    async def _callback() -> MaintenanceResult:
        nonlocal call_count
        call_count += 1
        return MaintenanceResult()

    coordinator.register("test", _callback)

    # First tick should run
    await coordinator.tick()
    assert call_count == 1

    # Immediate second tick should be throttled
    results = await coordinator.tick()
    assert call_count == 1
    assert results == {}


@pytest.mark.asyncio
async def test_tick_does_not_block_on_callback_exception() -> None:
    coordinator = MaintenanceCoordinator(interval_seconds=0.0)

    async def _broken() -> MaintenanceResult:
        raise RuntimeError("boom")

    coordinator.register("broken", _broken)

    # Should not raise
    results = await coordinator.tick()
    assert "broken" in results
    assert results["broken"].did_work is False


@pytest.mark.asyncio
async def test_seconds_since_last_run_tracks_time() -> None:
    coordinator = MaintenanceCoordinator(interval_seconds=10.0)
    assert coordinator.seconds_since_last_run > 0  # never run

    await coordinator.tick()
    assert coordinator.seconds_since_last_run < 1.0

    await asyncio.sleep(0.05)
    assert coordinator.seconds_since_last_run >= 0.05


@pytest.mark.asyncio
async def test_background_loop_runs_independently() -> None:
    """Background loop runs without request traffic."""
    coordinator = MaintenanceCoordinator(interval_seconds=0.05)
    call_count = 0

    async def _callback() -> MaintenanceResult:
        nonlocal call_count
        call_count += 1
        return MaintenanceResult(did_work=True)

    coordinator.register("bg", _callback)
    await coordinator.start()
    await asyncio.sleep(0.15)
    await coordinator.stop()
    # Background loop should have fired at least once
    assert call_count >= 1


@pytest.mark.asyncio
async def test_last_attempt_updates_on_run() -> None:
    coordinator = MaintenanceCoordinator(interval_seconds=0.0)

    async def _callback() -> MaintenanceResult:
        return MaintenanceResult(did_work=True)

    coordinator.register("test", _callback)
    assert coordinator.last_attempt == 0.0
    await coordinator.tick()
    assert coordinator.last_attempt > 0.0


@pytest.mark.asyncio
async def test_last_attempt_and_success_diverge_on_failure() -> None:
    coordinator = MaintenanceCoordinator(interval_seconds=0.0)
    fail_count = 0

    async def _passing() -> MaintenanceResult:
        return MaintenanceResult(did_work=True)

    async def _failing() -> MaintenanceResult:
        nonlocal fail_count
        fail_count += 1
        if fail_count == 1:
            raise RuntimeError("boom")
        return MaintenanceResult(did_work=True)

    coordinator.register("passing", _passing)
    coordinator.register("failing", _failing)

    await coordinator.tick()
    # last_attempt is set, last_success is NOT because _failing raised
    assert coordinator.last_attempt > 0.0
    assert coordinator.last_success == 0.0
    assert coordinator.callback_failures.get("failing") == 1

    await coordinator.tick()
    # Second tick: _failing succeeds this time
    assert coordinator.last_success > 0.0
    assert coordinator.callback_failures.get("failing") == 1  # still 1


@pytest.mark.asyncio
async def test_stats_exposes_maintenance_state() -> None:
    coordinator = MaintenanceCoordinator(interval_seconds=0.0)

    async def _callback() -> MaintenanceResult:
        return MaintenanceResult(stale_streams_removed=2, did_work=True)

    coordinator.register("streams", _callback)
    await coordinator.tick()

    stats = coordinator.stats()
    assert stats["interval_seconds"] == 0.0
    assert stats["throttled_tick_count"] >= 0
    assert "last_result_summary" in stats
    assert stats["last_result_summary"]["streams"]["stale_streams_removed"] == 2
    assert stats["last_result_summary"]["streams"]["did_work"] is True


@pytest.mark.asyncio
async def test_stats_has_no_failures_when_all_pass() -> None:
    coordinator = MaintenanceCoordinator(interval_seconds=0.0)

    async def _callback() -> MaintenanceResult:
        return MaintenanceResult(did_work=True)

    coordinator.register("clean", _callback)
    await coordinator.tick()

    stats = coordinator.stats()
    assert stats["callback_failures"] == {}


@pytest.mark.asyncio
async def test_throttled_tick_count_increments() -> None:
    coordinator = MaintenanceCoordinator(interval_seconds=60.0)

    async def _callback() -> MaintenanceResult:
        return MaintenanceResult(did_work=True)

    coordinator.register("t", _callback)
    await coordinator.tick()  # first runs
    assert coordinator.throttled_tick_count == 0

    await coordinator.tick()  # throttled
    assert coordinator.throttled_tick_count == 1

    await coordinator.tick()  # throttled again
    assert coordinator.throttled_tick_count == 2
