"""Non-streaming HTTP completion for provider dispatch."""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

import httpx
import structlog

from lattice.core.errors import ProviderError, ProviderTimeoutError
from lattice.providers.transport.helpers import (
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
from lattice.providers.transport.pool import ConnectionPoolManager
from lattice.providers.transport.rate_limits import RateLimitTracker
from lattice.providers.transport.registry import (
    ProviderRegistry,
    _resolve_provider_name,
    should_retry,
)
from lattice.providers.transport.stall_detector import StreamStallDetector
from lattice.providers.transport.streaming import StreamingMixin
from lattice.transport.congestion import TACCController
from lattice.transport.types import Response

logger = structlog.get_logger()


class DirectHTTPProvider(StreamingMixin):
    """Production-grade transport layer that makes **direct HTTP calls**
    to LLM providers using provider-specific adapters.

    No dependency on LiteLLM in the hot path.

    Routing
    -------
    * **Explicit provider wins:** Pass ``provider_name`` directly.
    * **Prefix requirement:** Parse ``provider/model`` from the model string.
    * **No bare-model heuristics:** We never guess a provider from a bare
      model name because models are portable across providers.
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
        self._log = logger.bind(module="direct_http_provider")
        self._stall_timeout: float = 30.0
        self._rate_limits = RateLimitTracker()
        self.stall_detector = StreamStallDetector()
        self.tacc = TACCController(enabled=tacc_enabled)
        self._downgrade_telemetry = downgrade_telemetry

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
        """Send a chat-completion request directly to the provider."""
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

        retry_cfg = adapter.retry_config()
        max_retries = retry_cfg.get("max_retries", 3)
        backoff_factor = retry_cfg.get("backoff_factor", 1.0)
        retry_on = retry_cfg.get("retry_on", (429, 502, 503, 504))
        tacc_estimate = tacc_reservation(request)

        payload = adapter.serialize_request(request)
        client = self.pool.get_client(provider_name, base_url)
        url = adapter.chat_endpoint(mapped_model, base_url)
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            await await_tacc_admission(self, provider_name, request)

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
                attempt=attempt + 1,
                max_attempts=max_retries + 1,
            )

            start = time.perf_counter()
            try:
                http_resp = await client.post(url, json=payload, headers=headers)
                elapsed_ms = (time.perf_counter() - start) * 1000
                self._rate_limits.update(provider_name, http_resp.headers)
                rate_limit_retry_after = self._rate_limits.retry_after(provider_name)
                self._log.info(
                    "http_request_done",
                    provider=provider_name,
                    model=mapped_model,
                    status=http_resp.status_code,
                    elapsed_ms=round(elapsed_ms, 3),
                    attempt=attempt + 1,
                )
                ttft_ms = elapsed_ms
                ttft_header = http_resp.headers.get("x-ttft-ms") or http_resp.headers.get(
                    "openai-processing-ms"
                )
                if ttft_header:
                    with contextlib.suppress(ValueError):
                        ttft_ms = float(ttft_header)
                await self.tacc.record_ttft(provider_name, ttft_ms)
            except httpx.TimeoutException as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                await self.tacc.after_response(
                    provider_name,
                    elapsed_ms,
                    tacc_estimate,
                    504,
                )
                last_error = ProviderTimeoutError(
                    provider=provider_name,
                    timeout_seconds=self.timeout,
                )
                last_error.__cause__ = exc
                if attempt < max_retries:
                    await asyncio.sleep(backoff_factor * (2**attempt))
                    continue
                raise last_error from None
            except httpx.ConnectError as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000
                await self.tacc.after_response(
                    provider_name,
                    elapsed_ms,
                    tacc_estimate,
                    502,
                )
                last_error = ProviderError(
                    provider=provider_name,
                    status_code=502,
                    message=f"Connection failed: {exc}",
                )
                last_error.__cause__ = exc
                if attempt < max_retries:
                    await asyncio.sleep(backoff_factor * (2**attempt))
                    continue
                raise last_error from None

            if not http_resp.is_success:
                body = await http_resp.aread()
                await self.tacc.after_response(
                    provider_name,
                    elapsed_ms,
                    tacc_estimate,
                    http_resp.status_code,
                    retry_after=rate_limit_retry_after,
                )
                last_error = ProviderError(
                    provider=provider_name,
                    status_code=http_resp.status_code,
                    message=f"HTTP {http_resp.status_code}: {body.decode(errors='replace')[:500]}",
                )
                if should_retry(http_resp.status_code, retry_on) and attempt < max_retries:
                    wait = backoff_factor * (2**attempt)
                    if rate_limit_retry_after is not None:
                        wait = max(wait, rate_limit_retry_after)
                    self._log.warning(
                        "http_request_retry",
                        provider=provider_name,
                        model=mapped_model,
                        status=http_resp.status_code,
                        wait_seconds=round(wait, 2),
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise last_error

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
            await self.tacc.after_response(
                provider_name,
                elapsed_ms,
                tokens,
                200,
            )
            resp = adapter.deserialize_response(data)
            if cached_tokens:
                resp.metadata["cached_tokens"] = cached_tokens
            return resp

        if last_error is not None:
            raise last_error
        raise ProviderError(
            provider=provider_name,
            status_code=502,
            message="All retry attempts exhausted",
        )
