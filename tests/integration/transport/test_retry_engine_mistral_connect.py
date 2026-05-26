"""Mistral connect errors use the shared retry engine (default policy)."""

from __future__ import annotations

import httpx
import pytest

from lattice.providers.adapters.retry_policies import retry_policy_for
from lattice.transport.retry import RetryEngine


@pytest.mark.asyncio
async def test_connect_error_retries_once() -> None:
    engine = RetryEngine()
    policy = retry_policy_for("mistral")
    calls = 0

    async def attempt(_n: int) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("refused")
        return 1

    result = await engine.run(attempt, policy=policy)
    assert result == 1
    assert calls == 2
