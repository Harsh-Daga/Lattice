"""Single retry implementation for all provider traffic."""

from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable, TypeVar

import httpx

from lattice.transport.retry_policy import Backoff, RetryPolicy, RetryRule

T = TypeVar("T")


class RetryEngine:
    """Runs an async attempt function according to a declarative ``RetryPolicy``."""

    def __init__(self) -> None:
        self.last_attempts: int = 0

    async def run(
        self,
        attempt_fn: Callable[[int], Awaitable[T]],
        *,
        policy: RetryPolicy,
        rate_limit_retry_after: Callable[[], float | None] | None = None,
    ) -> T:
        last_exc: Exception | None = None
        attempt_no = 0
        while True:
            attempt_no += 1
            try:
                result = await attempt_fn(attempt_no)
                self.last_attempts = attempt_no
                return result
            except Exception as exc:
                last_exc = exc
                rule = self._match_rule(exc, policy)
                if rule is None or attempt_no >= rule.max_attempts:
                    self.last_attempts = attempt_no
                    raise
                delay = self._compute_delay(
                    rule,
                    exc,
                    attempt_no,
                    rate_limit_retry_after=rate_limit_retry_after,
                )
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _match_rule(self, exc: Exception, policy: RetryPolicy) -> RetryRule | None:
        for rule in policy.rules:
            if rule.matches(exc):
                return rule
        return None

    def _compute_delay(
        self,
        rule: RetryRule,
        exc: Exception,
        attempt_no: int,
        *,
        rate_limit_retry_after: Callable[[], float | None] | None,
    ) -> float:
        header_name = rule.respect_header or (
            rule.backoff.header if rule.backoff.kind == "from_header" else None
        )
        if header_name:
            header_val = self._header_delay(exc, header_name)
            if header_val is not None:
                return header_val
        backoff = rule.backoff
        if backoff.kind == "from_header":
            backoff = Backoff.exponential(base=backoff.base, cap=backoff.cap)
        if backoff.kind == "fixed":
            return backoff.base
        if backoff.kind == "decorrelated_jitter":
            sleep = min(backoff.cap, random.uniform(backoff.base, backoff.base * 3))
            return sleep
        # exponential
        delay = min(backoff.cap, backoff.base * (2 ** (attempt_no - 1)))
        if rate_limit_retry_after is not None:
            ra = rate_limit_retry_after()
            if ra is not None:
                delay = max(delay, ra)
        return delay

    @staticmethod
    def _header_delay(exc: Exception, header: str) -> float | None:
        resp: httpx.Response | None = None
        if isinstance(exc, httpx.HTTPStatusError):
            resp = exc.response
        else:
            from lattice.core.errors import ProviderError

            if isinstance(exc, ProviderError) and hasattr(exc, "_response_headers"):
                raw = getattr(exc, "_response_headers", None)
                if isinstance(raw, dict):
                    val = raw.get(header) or raw.get(header.lower())
                    if val is not None:
                        try:
                            return float(val)
                        except ValueError:
                            return None
        if resp is not None:
            val = resp.headers.get(header)
            if val is not None:
                try:
                    return float(val)
                except ValueError:
                    return None
        return None
