"""OpenAI 429 + Anthropic 529 policies route through one RetryEngine."""

from __future__ import annotations

import asyncio

import pytest

from lattice.core.errors import ProviderError
from lattice.providers.adapters.retry_policies import (
    ANTHROPIC_RETRY_POLICY,
    OPENAI_RETRY_POLICY,
)
from lattice.transport.retry import RetryEngine


def test_openai_policy_has_429_retry_after() -> None:
    assert any("429" in (r.description or "") for r in OPENAI_RETRY_POLICY.rules)


def test_anthropic_policy_has_529_exponential() -> None:
    assert any("529" in (r.description or "") for r in ANTHROPIC_RETRY_POLICY.rules)


@pytest.mark.asyncio
async def test_openai_429_honors_retry_after_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []

    async def fake_sleep(sec: float) -> None:
        delays.append(sec)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    engine = RetryEngine()
    calls = 0

    async def attempt(n: int) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            err = ProviderError(provider="openai", status_code=429, message="rate limited")
            err._response_headers = {"retry-after": "3"}  # type: ignore[attr-defined]
            raise err
        return 42

    result = await engine.run(attempt, policy=OPENAI_RETRY_POLICY)
    assert result == 42
    assert calls == 2
    assert delays and delays[0] == 3.0


def test_anthropic_529_exponential_starts_at_one_second() -> None:
    engine = RetryEngine()
    err = ProviderError(provider="anthropic", status_code=529, message="overloaded")
    rule = next(r for r in ANTHROPIC_RETRY_POLICY.rules if "529" in (r.description or ""))
    delay = engine._compute_delay(rule, err, 1, rate_limit_retry_after=None)
    assert delay == 1.0
