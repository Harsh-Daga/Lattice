"""Unit tests for fallback_executor."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from lattice.planner.execution_plan import ExecutionPlan, FallbackPlan
from lattice.planner.fallback_executor import _is_retryable_error, execute_with_fallback


class FakeLogger:
    """In-memory logger for testing."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def warning(self, event: str, **kwargs: object) -> None:
        self.events.append({"level": "warning", "event": event, **kwargs})

    def error(self, event: str, **kwargs: object) -> None:
        self.events.append({"level": "error", "event": event, **kwargs})

    def info(self, event: str, **kwargs: object) -> None:
        self.events.append({"level": "info", "event": event, **kwargs})


class TestFallbackExecutor:
    @pytest.fixture
    def logger(self) -> FakeLogger:
        return FakeLogger()

    @pytest.fixture
    def metrics(self) -> object:
        """Simple metrics stub."""

        class MetricsStub:
            def __init__(self) -> None:
                self.counters: dict[str, int] = {}

            def increment(self, key: str, value: int = 1) -> None:
                self.counters[key] = self.counters.get(key, 0) + value

        return MetricsStub()

    @pytest.fixture
    def plan_with_retry(self) -> ExecutionPlan:
        return ExecutionPlan(
            fallback_plan=FallbackPlan(
                retry_count=2,
                fallback_provider="ollama",
                fallback_model="llama3.2",
            )
        )

    @pytest.fixture
    def plan_no_fallback(self) -> ExecutionPlan:
        return ExecutionPlan(
            fallback_plan=FallbackPlan(
                retry_count=1,
                fallback_provider=None,
                fallback_model=None,
            )
        )

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self, logger: FakeLogger, metrics: object) -> None:
        provider = AsyncMock(return_value="ok")
        result = await execute_with_fallback(
            provider,
            execution_plan=None,
            provider_name="openai",
            model="gpt-4",
            logger=logger,
            metrics=metrics,
            messages=[],
        )
        assert result == "ok"
        provider.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retry_then_success(
        self, logger: FakeLogger, metrics: object, plan_no_fallback: ExecutionPlan
    ) -> None:
        provider = AsyncMock(side_effect=[TimeoutError("boom"), "ok"])
        result = await execute_with_fallback(
            provider,
            execution_plan=plan_no_fallback,
            provider_name="openai",
            model="gpt-4",
            logger=logger,
            metrics=metrics,
            messages=[],
            temperature=0.7,
        )
        assert result == "ok"
        assert provider.await_count == 2
        assert any(ev["event"] == "provider_attempt_failed" for ev in logger.events)

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_no_fallback(
        self, logger: FakeLogger, metrics: object, plan_no_fallback: ExecutionPlan
    ) -> None:
        provider = AsyncMock(side_effect=TimeoutError("always fails"))
        with pytest.raises(TimeoutError):
            await execute_with_fallback(
                provider,
                execution_plan=plan_no_fallback,
                provider_name="openai",
                model="gpt-4",
                logger=logger,
                metrics=metrics,
                messages=[],
            )
        assert provider.await_count == 2  # attempt 0 + retry 1
        assert metrics.counters.get("lattice_provider_retries_exhausted", 0) == 1

    @pytest.mark.asyncio
    async def test_fallback_provider_success(
        self, logger: FakeLogger, metrics: object, plan_with_retry: ExecutionPlan
    ) -> None:
        # retry_count=2 means 3 primary attempts (0,1,2) + 1 fallback
        provider = AsyncMock(
            side_effect=[
                TimeoutError("fail"),
                TimeoutError("fail again"),
                TimeoutError("fail again 2"),
                "fallback_ok",
            ]
        )
        result = await execute_with_fallback(
            provider,
            execution_plan=plan_with_retry,
            provider_name="openai",
            model="gpt-4",
            logger=logger,
            metrics=metrics,
            messages=[],
        )
        assert result == "fallback_ok"
        # 3 primary attempts + 1 fallback = 4 total
        assert provider.await_count == 4
        # Verify fallback provider was passed
        calls = provider.await_args_list
        assert calls[-1].kwargs["provider_name"] == "ollama"
        assert metrics.counters.get("lattice_fallback_provider_success", 0) == 1

    @pytest.mark.asyncio
    async def test_non_retryable_exits_early(
        self, logger: FakeLogger, metrics: object, plan_no_fallback: ExecutionPlan
    ) -> None:
        class BadRequestError(ValueError):
            pass

        provider = AsyncMock(side_effect=BadRequestError("bad request"))
        with pytest.raises(BadRequestError):
            await execute_with_fallback(
                provider,
                execution_plan=plan_no_fallback,
                provider_name="openai",
                model="gpt-4",
                logger=logger,
                metrics=metrics,
                messages=[],
            )
        assert provider.await_count == 1

    def test_is_retryable_error(self) -> None:
        assert _is_retryable_error(TimeoutError("t"), None) is True
        assert _is_retryable_error(ValueError("bad"), None) is False
        assert _is_retryable_error(RuntimeError("r"), 502) is True
        assert _is_retryable_error(RuntimeError("r"), 400) is False
