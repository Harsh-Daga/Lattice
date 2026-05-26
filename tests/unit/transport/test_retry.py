"""RetryEngine unit tests."""

from __future__ import annotations

import pytest

from lattice.core.errors import ProviderError
from lattice.transport.retry import RetryEngine
from lattice.transport.retry_policy import Backoff, RetryPolicy, RetryRule, policy_from_retry_config


@pytest.mark.asyncio
async def test_max_attempts_respected() -> None:
    engine = RetryEngine()
    calls = 0

    async def attempt(_n: int) -> int:
        nonlocal calls
        calls += 1
        raise ProviderError(provider="openai", status_code=429, message="rate limited")

    policy = RetryPolicy(
        rules=(
            RetryRule(
                matches=lambda e: isinstance(e, ProviderError) and e.status_code == 429,
                max_attempts=2,
                backoff=Backoff.fixed(0.01),
            ),
        )
    )
    with pytest.raises(ProviderError):
        await engine.run(attempt, policy=policy)
    assert calls == 2


def test_policy_from_retry_config_has_429_rule() -> None:
    policy = policy_from_retry_config(
        {"max_retries": 3, "backoff_factor": 1.0, "retry_on": (429, 502)}
    )
    assert len(policy.rules) >= 2


def test_header_delay_from_provider_error() -> None:
    engine = RetryEngine()
    err = ProviderError(provider="openai", status_code=429, message="x")
    err._response_headers = {"retry-after": "3"}  # type: ignore[attr-defined]
    rule = RetryRule(
        matches=lambda e: isinstance(e, ProviderError),
        max_attempts=2,
        backoff=Backoff.from_header("retry-after", fallback=Backoff.exponential()),
        respect_header="retry-after",
    )
    delay = engine._compute_delay(rule, err, 1, rate_limit_retry_after=None)
    assert delay == 3.0
