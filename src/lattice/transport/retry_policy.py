"""Declarative retry rules consumed by ``RetryEngine``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class Backoff:
    """Backoff strategy descriptor."""

    kind: Literal["exponential", "decorrelated_jitter", "fixed", "from_header"]
    base: float = 1.0
    cap: float = 30.0
    header: str | None = None

    @classmethod
    def exponential(cls, *, base: float = 1.0, cap: float = 30.0) -> Backoff:
        return cls(kind="exponential", base=base, cap=cap)

    @classmethod
    def decorrelated_jitter(cls, *, base: float = 0.5, cap: float = 10.0) -> Backoff:
        return cls(kind="decorrelated_jitter", base=base, cap=cap)

    @classmethod
    def fixed(cls, seconds: float) -> Backoff:
        return cls(kind="fixed", base=seconds, cap=seconds)

    @classmethod
    def from_header(cls, header: str, *, fallback: Backoff) -> Backoff:
        return cls(kind="from_header", header=header, base=fallback.base, cap=fallback.cap)


@dataclass(frozen=True, slots=True)
class RetryRule:
    matches: Callable[[Exception], bool]
    max_attempts: int
    backoff: Backoff
    respect_header: str | None = None
    description: str = ""


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    rules: tuple[RetryRule, ...]
    default_action: Literal["raise", "exhaust"] = "raise"


def policy_from_retry_config(cfg: dict[str, Any]) -> RetryPolicy:
    """Build a ``RetryPolicy`` from legacy adapter ``retry_config()`` dicts."""
    import httpx

    from lattice.core.errors import ProviderError, ProviderTimeoutError

    max_retries = int(cfg.get("max_retries", 3))
    backoff_factor = float(cfg.get("backoff_factor", 1.0))
    retry_on = tuple(cfg.get("retry_on", (429, 502, 503, 504)))

    def _status_match(code: int) -> Callable[[Exception], bool]:
        def _match(exc: Exception) -> bool:
            if isinstance(exc, ProviderError):
                return exc.status_code == code
            if isinstance(exc, httpx.HTTPStatusError):
                return exc.response.status_code == code
            return False

        return _match

    rules: list[RetryRule] = []
    if 429 in retry_on:
        rules.append(
            RetryRule(
                matches=_status_match(429),
                max_attempts=max_retries + 1,
                backoff=Backoff.from_header(
                    "retry-after",
                    fallback=Backoff.exponential(base=backoff_factor, cap=60.0),
                ),
                respect_header="retry-after",
                description="HTTP 429 rate limit",
            )
        )
    for code in retry_on:
        if code == 429:
            continue
        rules.append(
            RetryRule(
                matches=_status_match(code),
                max_attempts=max_retries + 1,
                backoff=Backoff.exponential(base=backoff_factor, cap=60.0),
                description=f"HTTP {code}",
            )
        )

    rules.append(
        RetryRule(
            matches=lambda e: isinstance(
                e,
                (
                    httpx.ConnectError,
                    httpx.ReadError,
                    httpx.RemoteProtocolError,
                    ProviderTimeoutError,
                ),
            ),
            max_attempts=max_retries + 1,
            backoff=Backoff.decorrelated_jitter(base=backoff_factor * 0.5, cap=10.0),
            description="transient network",
        )
    )
    return RetryPolicy(rules=tuple(rules))
