"""Module-level retry policies — sole declarations consumed by ``RetryEngine``."""

from __future__ import annotations

import httpx

from lattice.core.errors import ProviderError, ProviderTimeoutError
from lattice.transport.retry_policy import Backoff, RetryPolicy, RetryRule, policy_from_retry_config

_DEFAULT_CFG = {
    "max_retries": 3,
    "backoff_factor": 1.0,
    "retry_on": (429, 502, 503, 504),
}


def _status_rule(code: int, *, max_attempts: int, backoff: Backoff, description: str) -> RetryRule:
    def _match(exc: Exception) -> bool:
        if isinstance(exc, ProviderError):
            return exc.status_code == code
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code == code
        return False

    return RetryRule(
        matches=_match,
        max_attempts=max_attempts,
        backoff=backoff,
        respect_header="retry-after" if code == 429 else None,
        description=description,
    )


OPENAI_RETRY_POLICY = RetryPolicy(
    rules=(
        _status_rule(
            429,
            max_attempts=4,
            backoff=Backoff.from_header(
                "retry-after",
                fallback=Backoff.exponential(base=1.0, cap=30.0),
            ),
            description="OpenAI 429 with retry-after header",
        ),
        _status_rule(
            502,
            max_attempts=4,
            backoff=Backoff.exponential(base=1.0, cap=60.0),
            description="OpenAI 502",
        ),
        _status_rule(
            503,
            max_attempts=4,
            backoff=Backoff.exponential(base=1.0, cap=60.0),
            description="OpenAI 503",
        ),
        _status_rule(
            504,
            max_attempts=4,
            backoff=Backoff.exponential(base=1.0, cap=60.0),
            description="OpenAI 504",
        ),
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
            max_attempts=4,
            backoff=Backoff.decorrelated_jitter(base=0.5, cap=10.0),
            description="OpenAI transient network",
        ),
    )
)

ANTHROPIC_RETRY_POLICY = RetryPolicy(
    rules=(
        _status_rule(
            429,
            max_attempts=4,
            backoff=Backoff.from_header(
                "retry-after",
                fallback=Backoff.exponential(base=1.0, cap=30.0),
            ),
            description="Anthropic 429",
        ),
        _status_rule(
            529,
            max_attempts=4,
            backoff=Backoff.exponential(base=1.0, cap=60.0),
            description="Anthropic 529 overloaded (no header → exponential from 1s)",
        ),
        _status_rule(
            502,
            max_attempts=4,
            backoff=Backoff.exponential(base=1.0, cap=60.0),
            description="Anthropic 502",
        ),
        _status_rule(
            503,
            max_attempts=4,
            backoff=Backoff.exponential(base=1.0, cap=60.0),
            description="Anthropic 503",
        ),
        _status_rule(
            504,
            max_attempts=4,
            backoff=Backoff.exponential(base=1.0, cap=60.0),
            description="Anthropic 504",
        ),
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
            max_attempts=4,
            backoff=Backoff.decorrelated_jitter(base=0.5, cap=10.0),
            description="Anthropic transient network",
        ),
    )
)

_DEFAULT_RETRY_POLICY = policy_from_retry_config(_DEFAULT_CFG)

_BY_NAME: dict[str, RetryPolicy] = {
    "openai": OPENAI_RETRY_POLICY,
    "anthropic": ANTHROPIC_RETRY_POLICY,
}


def retry_policy_for(provider_name: str) -> RetryPolicy:
    return _BY_NAME.get(provider_name, _DEFAULT_RETRY_POLICY)
