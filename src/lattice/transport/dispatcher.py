"""Unified transport dispatcher — sole execution path to providers."""

from __future__ import annotations

import contextlib
import time
from typing import Any

import httpx
import structlog

from lattice.core.errors import ProviderError, ProviderTimeoutError
from lattice.transport.backpressure import Backpressure, QueueFullError
from lattice.transport.circuit_breaker import CircuitBreakerRegistry, CircuitOpenError
from lattice.transport.congestion import TACCController
from lattice.transport.helpers import (
    await_tacc_admission,
    build_request,
    next_stream_line,
    parse_sse_line,
    process_sse_line_with_state,
    resolve_api_key,
    resolve_base_url,
    stream_chunk_text,
    stream_retry_policy,
    tacc_reservation,
)
from lattice.transport.metrics import TransportMetrics, monotonic_rtt
from lattice.transport.pool import ConnectionPoolManager
from lattice.transport.rate_limit import RateLimitTracker
from lattice.transport.registry import ProviderRegistry, _resolve_provider_name
from lattice.transport.retry import RetryEngine
from lattice.transport.retry_policy import RetryPolicy, policy_from_retry_config
from lattice.transport.stall_detector import StreamStallDetector
from lattice.transport.stream_resume import StreamResumer
from lattice.transport.streaming import StreamingMixin
from lattice.transport.telemetry import TransportTelemetry
from lattice.transport.timeout import TimeoutResolver
from lattice.transport.types import Request, Response

logger = structlog.get_logger()


class TransportDispatcher(StreamingMixin):
    """Sole execution path from pipeline/proxy → provider.

    Runs backpressure → circuit breaker → timeout → pool → adapter shape
    → httpx send → unified retry → parse → metrics.
    """

    _build_request = staticmethod(build_request)
    _stream_retry_policy = staticmethod(stream_retry_policy)
    _tacc_reservation = staticmethod(tacc_reservation)
    _stream_chunk_text = staticmethod(stream_chunk_text)
    _next_stream_line = staticmethod(next_stream_line)
    _parse_sse_line = staticmethod(parse_sse_line)
    _process_sse_line_with_state = staticmethod(process_sse_line_with_state)

    def __init__(
        self,
        registry: ProviderRegistry | None = None,
        pool: ConnectionPoolManager | None = None,
        *,
        default_api_base: str | None = None,
        default_api_key: str | None = None,
        provider_base_urls: dict[str, str] | None = None,
        timeout: float = 120.0,
        credentials: Any | None = None,
        tacc_enabled: bool = True,
        downgrade_telemetry: Any = None,
    ) -> None:
        if credentials is None:
            from lattice.providers.credentials import CredentialResolver

            credentials = CredentialResolver()

        self.registry = registry or ProviderRegistry()
        self.pool = pool or ConnectionPoolManager(downgrade_telemetry=downgrade_telemetry)
        self.default_api_base = default_api_base
        self.default_api_key = default_api_key
        self.provider_base_urls = provider_base_urls or {}
        self.timeout = timeout
        self._credentials = credentials  # CredentialResolver instance
        self._log = logger.bind(module="transport_dispatcher")
        self._stall_timeout: float = 30.0
        self._rate_limits = RateLimitTracker()
        self.stall_detector = StreamStallDetector()
        self.tacc = TACCController(enabled=tacc_enabled)
        self._downgrade_telemetry = downgrade_telemetry
        self._backpressure = Backpressure(max_in_flight=100)
        self._breaker = CircuitBreakerRegistry()
        self._retry = RetryEngine()
        self._timeout = TimeoutResolver(default_seconds=timeout)
        self._metrics = TransportMetrics()
        self._stream_resume = StreamResumer()
        self._last_telemetry: TransportTelemetry | None = None

    def _adapter_retry_policy(self, adapter: Any, ctx_model: str) -> RetryPolicy:
        policy_fn = getattr(adapter, "retry_policy", None)
        if callable(policy_fn):
            return policy_fn(ctx_model)  # type: ignore[no-any-return]
        from lattice.providers.adapters.retry_policies import retry_policy_for

        if hasattr(adapter, "retry_config"):
            legacy = getattr(adapter, "retry_config", None)
            if callable(legacy):
                return policy_from_retry_config(legacy())
        return retry_policy_for(adapter.name)

    @property
    def last_transport_telemetry(self) -> TransportTelemetry | None:
        return self._last_telemetry

    def pool_utilization(self) -> float:
        return min(1.0, self._backpressure.in_flight / max(1, self._backpressure.max_in_flight))

    def transport_health(self) -> dict[str, object]:
        breaker_states = self._breaker.all_states()
        return self._metrics.snapshot_dict(
            in_flight=self._backpressure.in_flight,
            queue_depth=self._backpressure.queue_depth,
            pool_counts={name: 1 for name in self.pool.clients_by_provider()},
            breaker_states=breaker_states,
        )

    def configure_resilience(
        self,
        stall_timeout: float = 30.0,
    ) -> None:
        """Configure stall detection for streaming.

        LATTICE does NOT do model fallback/routing. We always send
        the exact model the client requested.
        """
        self._stall_timeout = stall_timeout

    def cleanup_stale_streams(self, max_age_ms: float = 300000.0) -> int:
        """Remove abandoned stream state older than *max_age_ms*.

        Returns the number of streams cleaned up.
        """
        removed = self.stall_detector.cleanup_stale_streams(max_age_ms=max_age_ms)
        if removed > 0:
            self._log.debug("stale_streams_cleaned", count=removed)
        return removed

    def _resolve_base_url(self, provider_name: str, api_base: str | None = None) -> str:
        return resolve_base_url(self, provider_name, api_base)

    def _resolve_api_key(self, provider_name: str, api_key: str | None = None) -> str:
        return resolve_api_key(self, provider_name, api_key)

    async def _await_tacc_admission(self, provider_name: str, request: Request) -> bool:
        return await await_tacc_admission(self, provider_name, request)

    def health_check(self) -> tuple[bool, str]:
        return True, f"direct_http ({len(self.registry.list_adapters())} adapters)"

    def get_transport_metadata(self, provider_name: str) -> dict[str, str]:
        """Return transport metadata for *provider_name* (HTTP version, fallback reason)."""
        base_url = resolve_base_url(self, provider_name)
        return {
            "http_version": self.pool.get_http_version(provider_name, base_url),
            "fallback_reason": self.pool.get_fallback_reason(provider_name, base_url) or "",
        }

    async def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stop: list[str] | str | None = None,
        stream: bool = False,
        api_base: str | None = None,
        api_key: str | None = None,
        provider_name: str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Response:
        """Send a chat-completion request through the unified transport stack."""
        provider_name = _resolve_provider_name(model, provider_name, self.registry)
        adapter = self.registry.get_adapter(provider_name)

        base_url = resolve_base_url(self, provider_name, api_base)
        key = resolve_api_key(self, provider_name, api_key)

        request = build_request(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            stop=stop,
            stream=stream,
            metadata=kwargs,
            extra_headers=extra_headers,
            extra_body=extra_body,
        )
        mapped_model = adapter.map_model_name(request.model)
        request.model = mapped_model

        tenant = str(kwargs.get("tenant", "default"))
        try:
            async with self._backpressure.admit(tenant=tenant):
                return await self._completion_inner(
                    provider_name=provider_name,
                    adapter=adapter,
                    request=request,
                    mapped_model=mapped_model,
                    base_url=base_url,
                    key=key,
                    messages=messages,
                )
        except QueueFullError as exc:
            raise ProviderError(
                provider=provider_name,
                status_code=503,
                message="Transport queue full",
            ) from exc

    async def _completion_inner(
        self,
        *,
        provider_name: str,
        adapter: Any,
        request: Request,
        mapped_model: str,
        base_url: str,
        key: str,
        messages: list[dict[str, Any]],
    ) -> Response:
        breaker = self._breaker.for_(provider_name, mapped_model)
        if not breaker.allow():
            raise CircuitOpenError(provider=provider_name, model=mapped_model)

        policy = self._adapter_retry_policy(adapter, mapped_model)
        tacc_estimate = tacc_reservation(request)
        payload = adapter.serialize_request(request)
        client = self.pool.get_client(provider_name, base_url)
        url = adapter.chat_endpoint(mapped_model, base_url)
        timeout_policy = self._timeout.resolve()

        async def _attempt(attempt_no: int) -> httpx.Response:
            admitted = await await_tacc_admission(self, provider_name, request)
            if not admitted:
                raise ProviderError(
                    provider=provider_name,
                    status_code=503,
                    message="TACC admission rejected",
                )

            headers: dict[str, str] = {"Content-Type": "application/json"}
            headers.update(adapter.auth_headers(key))
            headers.update(adapter.extra_headers(request))
            headers.update(request.extra_headers)

            self._log.info(
                "http_request_start",
                provider=provider_name,
                model=mapped_model,
                url=url,
                msg_count=len(messages),
                attempt=attempt_no,
            )

            start = time.perf_counter()
            try:
                http_resp = await client.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=timeout_policy.for_attempt(attempt_no),
                )
            except httpx.TimeoutException as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                await self.tacc.after_response(provider_name, elapsed_ms, tacc_estimate, 504)
                err = ProviderTimeoutError(
                    provider=provider_name,
                    timeout_seconds=self.timeout,
                )
                err.__cause__ = exc
                raise err from None
            except httpx.ConnectError as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                await self.tacc.after_response(provider_name, elapsed_ms, tacc_estimate, 502)
                raise ProviderError(
                    provider=provider_name,
                    status_code=502,
                    message=f"Connection failed: {exc}",
                ) from exc

            elapsed_ms = (time.perf_counter() - start) * 1000
            self._rate_limits.update(provider_name, http_resp.headers)
            self._log.info(
                "http_request_done",
                provider=provider_name,
                model=mapped_model,
                status=http_resp.status_code,
                elapsed_ms=round(elapsed_ms, 3),
                attempt=attempt_no,
            )
            ttft_ms = elapsed_ms
            ttft_header = http_resp.headers.get("x-ttft-ms") or http_resp.headers.get(
                "openai-processing-ms"
            )
            if ttft_header:
                with contextlib.suppress(ValueError):
                    ttft_ms = float(ttft_header)
            await self.tacc.record_ttft(provider_name, ttft_ms)

            if not http_resp.is_success:
                body = await http_resp.aread()
                ra = self._rate_limits.retry_after(provider_name)
                await self.tacc.after_response(
                    provider_name,
                    elapsed_ms,
                    tacc_estimate,
                    http_resp.status_code,
                    retry_after=ra,
                )
                http_err = ProviderError(
                    provider=provider_name,
                    status_code=http_resp.status_code,
                    message=f"HTTP {http_resp.status_code}: {body.decode(errors='replace')[:500]}",
                )
                http_err._response_headers = dict(http_resp.headers)  # type: ignore[attr-defined]
                raise http_err
            return http_resp

        dispatch_start = time.perf_counter()
        try:
            http_resp = await self._retry.run(
                _attempt,
                policy=policy,
                rate_limit_retry_after=lambda: self._rate_limits.retry_after(provider_name),
            )
            tel = monotonic_rtt(dispatch_start)
            tel.attempt = self._retry.last_attempts
            self._metrics.record_success(provider_name, tel)
            breaker.on_success()
            self._last_telemetry = TransportTelemetry(
                provider=provider_name,
                model=mapped_model,
                rtt_ms=tel.rtt * 1000.0,
                attempt=tel.attempt,
                pool_utilization=self.pool_utilization(),
                breaker_state=breaker.state(),
            )
        except Exception as exc:
            breaker.on_failure(error_class=type(exc).__name__)
            self._metrics.record_failure(provider_name, exc)
            raise

        data = http_resp.json()
        tokens = 0
        usage = data.get("usage")
        cached_tokens = 0
        if isinstance(usage, dict):
            maybe_tokens = usage.get("completion_tokens")
            if isinstance(maybe_tokens, int):
                tokens = maybe_tokens
            cached_tokens = usage.get("cached_tokens", 0)
            if not cached_tokens:
                details = usage.get("prompt_tokens_details")
                if isinstance(details, dict):
                    ct = details.get("cached_tokens")
                    if isinstance(ct, int):
                        cached_tokens = ct
        await self.tacc.after_response(provider_name, 0.0, tokens, 200)
        resp = adapter.deserialize_response(data)
        if cached_tokens:
            resp.metadata["cached_tokens"] = cached_tokens
        return resp


# Backward-compatible alias used across proxy, gateway, and tests.
DirectHTTPProvider = TransportDispatcher
