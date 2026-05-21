"""Fallback executor — retry with optional provider fallback.

Production-grade wrapper around provider calls that:
  1. Retries transient errors with configurable backoff.
  2. Falls back to an alternate provider/model if retries are exhausted.
  3. Logs every attempt with full context for debuggability.

Phase 10 — End-to-end wiring.

Usage (non-streaming)::
    from lattice.planner.fallback_executor import execute_with_fallback
    response = await execute_with_fallback(
        deps.provider.completion,
        execution_plan=plan,
        provider_name="openai",
        model="gpt-4",
        logger=deps.logger,
        metrics=deps.metrics,
        # kwargs passed through to provider.completion()
        messages=[...],
        temperature=0.7,
    )

Usage (streaming)::
    from lattice.planner.fallback_executor import execute_with_fallback_stream
    async for chunk in execute_with_fallback_stream(
        deps.provider.completion_stream_with_stall_detect,
        execution_plan=plan,
        provider_name="openai",
        model="gpt-4",
        logger=deps.logger,
        metrics=deps.metrics,
        messages=[...],
    ):
        yield chunk
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Awaitable, Callable

from lattice.planner.execution_plan import ExecutionPlan


def _is_retryable_error(exc: Exception, status_code: int | None) -> bool:
    """Return True if *exc* is worth retrying."""
    if status_code is not None and status_code in _RETRYABLE_HTTP_CODES:
        return True
    exc_name = type(exc).__name__
    return exc_name in _RETRYABLE_EXCEPTION_TYPES


# Retryable error detection — kept private, not part of public API
_RETRYABLE_HTTP_CODES: frozenset[int] = frozenset({429, 502, 503, 504})
_RETRYABLE_EXCEPTION_TYPES: tuple[str, ...] = (
    "TimeoutError",
    "ConnectError",
    "NetworkError",
    "ReadError",
    "PoolTimeout",
    "RemoteProtocolError",
    "TransportError",
)


async def execute_with_fallback(
    provider_call: Callable[..., Awaitable[Any]],
    *,
    execution_plan: ExecutionPlan | None,
    provider_name: str,
    model: str,
    logger: Any,
    metrics: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Call provider with retry + optional fallback provider/model.

    Args:
        provider_call: The provider method to call (e.g. deps.provider.completion).
        execution_plan: ExecutionPlan with fallback_plan and retry_count.
        provider_name: Primary provider name.
        model: Primary model identifier.
        logger: Structlog logger for observability.
        metrics: Metrics collector for counters.
        **kwargs: Forwarded to provider_call.

    Returns:
        Provider response object.

    Raises:
        The last exception if all retries and fallback are exhausted.
    """
    # Defaults when no execution plan
    retry_count: int = 0
    fallback_provider: str | None = None
    fallback_model: str | None = None
    if execution_plan is not None:
        retry_count = execution_plan.fallback_plan.retry_count
        fallback_provider = execution_plan.fallback_plan.fallback_provider
        fallback_model = execution_plan.fallback_plan.fallback_model

    last_exc: Exception | None = None
    max_wait: float = 16.0

    # Total attempts = primary retries (retry_count + 1) + one fallback attempt
    max_attempts = retry_count + 2 if fallback_provider else retry_count + 1

    for attempt in range(max_attempts):
        is_fallback = False
        current_provider = provider_name
        current_model = model

        # If this is beyond primary retries, try fallback provider
        if attempt > retry_count and fallback_provider:
            current_provider = fallback_provider
            current_model = fallback_model or model
            is_fallback = True
        elif attempt > retry_count:
            # No fallback configured, break
            break

        try:
            kwargs["provider_name"] = current_provider
            kwargs["model"] = current_model
            result = await provider_call(**kwargs)
            if is_fallback and metrics is not None:
                metrics.increment("lattice_fallback_provider_success")
            return result
        except Exception as exc:
            last_exc = exc
            error_type = type(exc).__name__
            status_code: int | None = None
            if hasattr(exc, "status_code"):
                status_code = getattr(exc, "status_code", None)
            is_retryable = _is_retryable_error(exc, status_code)

            if attempt < retry_count:
                wait = min(2**attempt, max_wait)
                logger.warning(
                    "provider_attempt_failed",
                    provider=current_provider,
                    model=current_model,
                    attempt=attempt + 1,
                    max_retries=retry_count,
                    is_retryable=is_retryable,
                    error_type=error_type,
                    status_code=status_code,
                    wait_ms=round(wait * 1000, 1),
                    is_fallback=is_fallback,
                )
                if is_retryable:
                    await asyncio.sleep(wait)
                else:
                    # Non-retryable: exit early
                    break
            elif is_fallback:
                logger.error(
                    "fallback_provider_failed",
                    fallback_provider=current_provider,
                    fallback_model=current_model,
                    error_type=error_type,
                    error=str(exc),
                )
                if metrics is not None:
                    metrics.increment("lattice_fallback_provider_failed")
            else:
                logger.error(
                    "provider_retries_exhausted",
                    provider=current_provider,
                    model=current_model,
                    attempt=attempt + 1,
                    max_retries=retry_count,
                    error_type=error_type,
                    status_code=status_code,
                    error=str(exc),
                )
                if metrics is not None:
                    metrics.increment("lattice_provider_retries_exhausted")

    # All attempts exhausted
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("execute_with_fallback: unreachable")


async def execute_with_fallback_stream(
    provider_caller: Any,
    *,
    execution_plan: ExecutionPlan | None,
    provider_name: str,
    model: str,
    logger: Any,
    metrics: Any | None = None,
    **kwargs: Any,
) -> AsyncIterator[Any]:
    """Yield provider stream chunks with retry on connection failure.

    Only retries the *initial connection* — once the first chunk arrives,
    the stream is passed through transparently. This prevents partial-
    content retries that would be observable to the consumer.

    Args:
        provider_caller: A callable that returns an AsyncIterator (the stream).
        execution_plan: ExecutionPlan with fallback_plan and retry_count.
        provider_name: Primary provider name.
        model: Primary model identifier.
        logger: Structlog logger.
        metrics: Metrics collector.
        **kwargs: Forwarded to provider_caller.

    Yields:
        Stream chunks from the provider.

    Raises:
        The last exception if all connection retries are exhausted.
    """
    # Defaults when no execution plan
    retry_count: int = 0
    fallback_provider: str | None = None
    fallback_model: str | None = None
    if execution_plan is not None:
        retry_count = execution_plan.fallback_plan.retry_count
        fallback_provider = execution_plan.fallback_plan.fallback_provider
        fallback_model = execution_plan.fallback_plan.fallback_model

    last_exc: Exception | None = None
    max_wait: float = 16.0
    max_attempts = retry_count + 2 if fallback_provider else retry_count + 1

    for attempt in range(max_attempts):
        is_fallback = False
        current_provider = provider_name
        current_model = model

        if attempt > retry_count and fallback_provider:
            current_provider = fallback_provider
            current_model = fallback_model or model
            is_fallback = True
        elif attempt > retry_count:
            break

        try:
            kwargs["provider_name"] = current_provider
            kwargs["model"] = current_model
            stream = provider_caller(**kwargs)
            if is_fallback and metrics is not None:
                metrics.increment("lattice_fallback_provider_success")

            async for chunk in stream:
                yield chunk
            return

        except GeneratorExit:
            raise
        except Exception as exc:
            last_exc = exc
            error_type = type(exc).__name__
            status_code: int | None = None
            if hasattr(exc, "status_code"):
                status_code = getattr(exc, "status_code", None)
            is_retryable = _is_retryable_error(exc, status_code)

            if attempt < retry_count:
                wait = min(2**attempt, max_wait)
                logger.warning(
                    "provider_stream_init_failed",
                    provider=current_provider,
                    model=current_model,
                    attempt=attempt + 1,
                    max_retries=retry_count,
                    is_retryable=is_retryable,
                    error_type=error_type,
                    status_code=status_code,
                    wait_ms=round(wait * 1000, 1),
                    is_fallback=is_fallback,
                )
                if is_retryable:
                    await asyncio.sleep(wait)
                else:
                    break
            elif is_fallback:
                logger.error(
                    "fallback_provider_stream_failed",
                    fallback_provider=current_provider,
                    fallback_model=current_model,
                    error_type=error_type,
                    error=str(exc),
                )
                if metrics is not None:
                    metrics.increment("lattice_fallback_provider_failed")
            else:
                logger.error(
                    "provider_stream_retries_exhausted",
                    provider=current_provider,
                    model=current_model,
                    attempt=attempt + 1,
                    max_retries=retry_count,
                    error_type=error_type,
                    status_code=status_code,
                    error=str(exc),
                )
                if metrics is not None:
                    metrics.increment("lattice_provider_retries_exhausted")

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("execute_with_fallback_stream: unreachable")


__all__ = [
    "execute_with_fallback",
    "execute_with_fallback_stream",
]
