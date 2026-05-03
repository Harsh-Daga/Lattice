"""Provider registry and dispatcher.

Detects the correct `ProviderAdapter` from the ``model`` field string,
manages per-provider `httpx.AsyncClient` connection pools, and performs
the actual direct HTTP call.

Usage
-----
    registry = ProviderRegistry()
    adapter = registry.resolve("ollama/llama3")
    # adapter is OllamaAdapter

    transport = DirectHTTPProvider(registry)
    response = await transport.completion(
        model="ollama/glm-5.1:cloud",
        messages=[{"role":"user","content":"hi"}],
        api_base="http://127.0.0.1:11434",
    )
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import structlog

from lattice.core.errors import ProviderError, ProviderTimeoutError
from lattice.core.transport import Request, Response
from lattice.providers.anthropic import AnthropicAdapter
from lattice.providers.azure import AzureAdapter
from lattice.providers.base import ProviderAdapter
from lattice.providers.bedrock import BedrockAdapter
from lattice.providers.gemini import GeminiAdapter, VertexAdapter
from lattice.providers.ollama import OllamaAdapter, OllamaCloudAdapter
from lattice.providers.openai import OpenAIAdapter
from lattice.providers.openai_compatible import (
    AI21Adapter,
    CohereAdapter,
    DeepSeekAdapter,
    FireworksAdapter,
    GroqAdapter,
    MistralAdapter,
    OpenRouterAdapter,
    PerplexityAdapter,
    TogetherAdapter,
)
from lattice.providers.stall_detector import StreamStallDetector
from lattice.transport.congestion import TACCController

logger = structlog.get_logger()


# =============================================================================
# ProviderRegistry
# =============================================================================


class ProviderRegistry:
    """Maps model strings to the correct ``ProviderAdapter``.

    The registry is a simple priority list. Each adapter's ``supports``
    method is called in order until one returns ``True``.
    """

    def __init__(self, adapters: list[ProviderAdapter] | None = None) -> None:
        self._adapters = list(adapters) if adapters else []
        if not self._adapters:
            # Defer ChatGPT import to avoid circular dependency:
            # lattice.integrations.codex.adapter → lattice.providers.openai
            # and lattice.integrations.__init__ → lattice.proxy → lattice.providers.transport
            from lattice.integrations.codex.adapter import ChatGPTAdapter

            self._adapters = [
                # ChatGPT / Codex (must be before OpenAIAdapter to intercept Codex JWTs)
                ChatGPTAdapter(),  # chatgpt/ prefix, Codex JWT auth
                # OpenAI-compatible providers (prefix matching)
                GroqAdapter(),  # groq/ prefix
                TogetherAdapter(),  # together/ prefix
                DeepSeekAdapter(),  # deepseek/ prefix
                PerplexityAdapter(),  # perplexity/ prefix
                MistralAdapter(),  # mistral/ prefix
                FireworksAdapter(),  # fireworks/ prefix
                OpenRouterAdapter(),  # openrouter/ prefix
                CohereAdapter(),  # cohere/ prefix
                AI21Adapter(),  # ai21/ prefix
                GeminiAdapter(),  # gemini/ or google/ prefix
                VertexAdapter(),  # vertex/ prefix
                OllamaCloudAdapter(),  # ollama-cloud/ prefix
                OllamaAdapter(),  # ollama/ prefix
                AnthropicAdapter(),  # anthropic/ or claude- prefix
                AzureAdapter(),  # azure/ prefix
                BedrockAdapter(),  # bedrock/ prefix
                OpenAIAdapter(),  # openai/ prefix
            ]

    def resolve(self, model: str) -> ProviderAdapter:
        """Find the adapter for *model*.

        Raises:
            ProviderError: if no adapter matches.
        """
        for adapter in self._adapters:
            if adapter.supports(model):
                return adapter
        raise ProviderError(
            provider="unknown",
            status_code=400,
            message=f"No provider adapter for model '{model}'",
        )

    def get_adapter(self, name: str) -> ProviderAdapter:
        """Look up an adapter by its canonical *name* (e.g. ``"ollama"``).

        Supports provider aliases so ``ollama-cloud`` resolves to the
        ``OllamaAdapter`` (same API, different base URL).

        Raises:
            ProviderError: if no adapter matches.
        """
        resolved = _PROVIDER_ALIASES.get(name, name)
        for adapter in self._adapters:
            if adapter.name == resolved:
                return adapter
        raise ProviderError(
            provider=name,
            status_code=400,
            message=f"No adapter registered for provider '{name}'",
        )

    def list_adapters(self) -> list[str]:
        return [a.name for a in self._adapters]

    def iter_adapters(self):
        """Yield every registered adapter in priority order."""
        yield from self._adapters


# =============================================================================
# Provider aliases — same API format, different endpoint.
# Used when a provider slug should reuse the adapter of another provider.
# =============================================================================
_PROVIDER_ALIASES: dict[str, str] = {}


def _resolve_provider_name(model: str, provider_name: str | None = None) -> str:
    """Resolve provider from explicit hint or model prefix.

    Priority:
    1. ``provider_name`` if provided.
    2. ``provider/model`` prefix.

    We NEVER guess a provider from a bare model name because models
    are portable across providers (e.g. ``llama-3.1-70b`` can be on
    Groq, Together, or Fireworks).

    Raises:
        ProviderError: if no provider is explicitly specified.
    """
    if provider_name:
        return provider_name.lower()
    if "/" in model:
        prefix = model.split("/", 1)[0].lower()
        # Check against the canonical list of registered provider names
        from lattice.providers.capabilities import get_capability_registry
        if prefix in get_capability_registry().list_providers():
            return prefix
    raise ProviderError(
        provider="unknown",
        status_code=400,
        message=(
            f"Provider not specified. Use either: "
            f"1) provider_name parameter, or "
            f"2) model prefix like 'groq/llama-3b' (got model='{model}')"
        ),
    )


def _should_retry(status_code: int, retry_on: tuple[int, ...]) -> bool:
    return status_code in retry_on


# =============================================================================
# Connection Pool Manager
# =============================================================================


class ConnectionPoolManager:
    """Manages persistent ``httpx.AsyncClient`` instances per provider.

    One client per ``(provider_name, base_url)`` tuple. HTTP/2 is enabled
    where available; graceful fallback to HTTP/1.1 if ``h2`` is missing.
    """

    def __init__(self, http2: bool = True, downgrade_telemetry: Any = None) -> None:
        self._http2 = http2
        self._clients: dict[tuple[str, str], httpx.AsyncClient] = {}
        self._http2_fallback_reason: dict[tuple[str, str], str] = {}
        self._downgrade_telemetry = downgrade_telemetry
        self._log = logger.bind(module="connection_pool")

    def get_client(self, provider: str, base_url: str) -> httpx.AsyncClient:
        key = (provider, base_url)
        if key not in self._clients:
            limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
            timeout = httpx.Timeout(120.0, connect=10.0)
            try:
                client = httpx.AsyncClient(
                    timeout=timeout,
                    limits=limits,
                    http1=True,
                    http2=self._http2,
                )
            except ImportError:
                client = httpx.AsyncClient(
                    timeout=timeout,
                    limits=limits,
                    http1=True,
                    http2=False,
                )
                self._http2 = False
                self._http2_fallback_reason[key] = "h2_unavailable"
                self._log.warning("http2_unavailable", provider=provider, fallback="http1.1")
                if self._downgrade_telemetry is not None:
                    from lattice.core.telemetry import DowngradeCategory

                    self._downgrade_telemetry.record(
                        DowngradeCategory.HTTP2_TO_HTTP11,
                        reason="h2_unavailable",
                    )
            self._clients[key] = client
            self._log.info(
                "pool_created",
                provider=provider,
                base_url=base_url,
                http2=self._http2,
            )
        return self._clients[key]

    def get_http_version(self, provider: str, base_url: str) -> str:
        key = (provider, base_url)
        if key in self._http2_fallback_reason:
            return "http/1.1"
        return "http/2" if self._http2 else "http/1.1"

    def get_fallback_reason(self, provider: str, base_url: str) -> str | None:
        return self._http2_fallback_reason.get((provider, base_url))

    async def close(self) -> None:
        """Close all pooled clients."""
        for key, client in list(self._clients.items()):
            await client.aclose()
            self._log.info("pool_closed", provider=key[0])
        self._clients.clear()

    async def recycle_client(self, provider: str, base_url: str) -> None:
        """Close and evict one pooled client; next use recreates it."""
        key = (provider, base_url)
        client = self._clients.pop(key, None)
        if client is not None:
            await client.aclose()
            self._log.info("pool_recycled", provider=provider, base_url=base_url)

    @property
    def pool_count(self) -> int:
        return len(self._clients)


# =============================================================================
# DirectHTTPProvider
# =============================================================================


class RateLimitTracker:
    """Light-weight per-provider rate-limit state.

    Parses common ``x-ratelimit-*`` headers and tracks whether a provider
    is currently throttled.
    """

    def __init__(self) -> None:
        self._limits: dict[str, dict[str, Any]] = {}

    def update(self, provider: str, headers: httpx.Headers) -> None:
        """Parse rate-limit headers from a response."""
        limit = headers.get("x-ratelimit-limit")
        remaining = headers.get("x-ratelimit-remaining")
        reset = headers.get("x-ratelimit-reset")
        retry_after = headers.get("retry-after")
        if limit or remaining or reset or retry_after:
            self._limits[provider] = {
                "limit": int(limit) if limit else None,
                "remaining": int(remaining) if remaining else None,
                "reset": int(reset) if reset else None,
                "retry_after": int(retry_after) if retry_after else None,
            }

    def is_throttled(self, provider: str) -> bool:
        state = self._limits.get(provider)
        if not state:
            return False
        remaining = state.get("remaining")
        return bool(remaining is not None and remaining <= 0)

    def retry_after(self, provider: str) -> float | None:
        state = self._limits.get(provider)
        if not state:
            return None
        ra = state.get("retry_after")
        if ra is not None:
            return float(ra)
        return None


class DirectHTTPProvider:
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
            from lattice.core.credentials import CredentialResolver

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

    @staticmethod
    def _stream_retry_policy(
        provider_name: str,
        metadata: dict[str, Any],
    ) -> tuple[float | None, int]:
        ttft_raw = metadata.get("ttft_timeout_seconds")
        retries_raw = metadata.get("no_first_chunk_retries")
        ttft_timeout: float | None
        if ttft_raw is None:
            ttft_timeout = 8.0 if provider_name == "ollama-cloud" else None
        else:
            ttft_timeout = float(ttft_raw)
        if retries_raw is None:
            retries = 1 if provider_name == "ollama-cloud" else 0
        else:
            retries = max(0, int(retries_raw))
        return ttft_timeout, retries

    async def _await_tacc_admission(self, provider_name: str, request: Request) -> bool:
        """Admit a request through the queued TACC controller."""
        estimate = self._tacc_reservation(request)
        priority = int(request.metadata.get("tacc_priority", 0) or 0)
        cache_hit_expected = bool(request.metadata.get("_lattice_cache_hit_expected", False))
        is_speculative = bool(request.metadata.get("_lattice_is_speculative", False))
        is_batch = bool(request.metadata.get("_lattice_is_batch", False))

        decision, _reason = self.tacc.evaluate_admission(
            provider_name,
            estimate,
            priority,
            cache_hit_expected=cache_hit_expected,
            is_batch=is_batch,
            is_speculative=is_speculative,
        )
        if decision.value == "reject":
            return False
        if decision.value == "priority_downgrade":
            priority = max(0, priority - 2)

        return await self.tacc.acquire_request(
            provider_name,
            estimated_tokens=estimate,
            priority=priority,
        )

    @staticmethod
    def _tacc_reservation(request: Any) -> int:
        """Estimate token pressure for admission control."""
        prompt = max(1, int(getattr(request, "token_estimate", 0) or 0))
        completion = max(0, int(getattr(request, "max_tokens", 0) or 0))
        return max(1, prompt + completion)

    @staticmethod
    def _stream_chunk_text(chunk: dict[str, Any]) -> str:
        """Extract text-like content from a normalized stream chunk."""
        choices = chunk.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        delta = first.get("delta")
        if isinstance(delta, dict):
            for key in ("content", "reasoning_content", "reasoning"):
                value = delta.get(key)
                if isinstance(value, str) and value:
                    return value
        message = first.get("message")
        if isinstance(message, dict):
            for key in ("content", "reasoning_content", "reasoning"):
                value = message.get(key)
                if isinstance(value, str) and value:
                    return value
        return ""

    @staticmethod
    async def _next_stream_line(
        iterator: Any,
        *,
        deadline_monotonic: float | None,
    ) -> str:
        if deadline_monotonic is None:
            return await iterator.__anext__()  # type: ignore[no-any-return]
        remaining = deadline_monotonic - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError
        return await asyncio.wait_for(iterator.__anext__(), timeout=remaining)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> tuple[bool, str]:
        return True, f"direct_http ({len(self.registry.list_adapters())} adapters)"

    def get_transport_metadata(self, provider_name: str) -> dict[str, str]:
        """Return transport metadata for *provider_name* (HTTP version, fallback reason)."""
        base_url = self._resolve_base_url(provider_name)
        return {
            "http_version": self.pool.get_http_version(provider_name, base_url),
            "fallback_reason": self.pool.get_fallback_reason(provider_name, base_url) or "",
        }

    # ------------------------------------------------------------------
    # Stream stall detection
    # ------------------------------------------------------------------

    async def completion_stream_with_stall_detect(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stop: list[str] | str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        provider_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Wraps ``completion_stream`` with stall detection."""
        import time as time_mod

        stall_timeout = self._stall_timeout
        provider_name = _resolve_provider_name(model, provider_name)
        adapter = self.registry.get_adapter(provider_name)

        # Build base URL + key
        base_url = self._resolve_base_url(provider_name, api_base)
        key = self._resolve_api_key(provider_name, api_key)

        request = self._build_request(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            stop=stop,
            stream=True,
            metadata=metadata or {},
            extra_headers=extra_headers,
            extra_body=extra_body,
        )
        mapped_model = adapter.map_model_name(request.model)
        request.model = mapped_model

        payload = adapter.serialize_request(request)
        client = self.pool.get_client(provider_name, base_url)
        url = adapter.chat_endpoint(mapped_model, base_url)
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        headers.update(adapter.auth_headers(key))
        headers.update(adapter.extra_headers(request))
        headers.update(request.extra_headers)

        self._log.info(
            "http_stream_start_with_stall",
            provider=provider_name,
            model=mapped_model,
            url=url,
            stall_timeout=stall_timeout,
        )

        # Detect state-machine streaming (Anthropic tool_use + thinking)
        stream_state: Any | None = None
        if hasattr(adapter, "normalize_sse_stream"):
            stream_state = adapter.normalize_sse_stream(mapped_model)
        stream_optimizer: Any | None = None
        stream_optimizer_state: Any | None = None
        ttft_timeout, no_first_chunk_retries = self._stream_retry_policy(
            provider_name,
            request.metadata,
        )
        force_fresh_connection = bool(
            request.metadata.get(
                "stream_force_fresh_connection",
                provider_name == "ollama-cloud",
            )
        )

        await self._await_tacc_admission(provider_name, request)
        slot_released = False
        start = time.perf_counter()
        feedback_sent = False
        first_chunk_emitted = False
        streamed_tokens = 0
        stream_id = f"{provider_name}_{time_mod.perf_counter()}_{id(object())}"
        self.stall_detector.start_stream(provider_name, stream_id)
        try:
            for attempt in range(no_first_chunk_retries + 1):
                try:
                    if force_fresh_connection:
                        await self.pool.recycle_client(provider_name, base_url)
                        client = self.pool.get_client(provider_name, base_url)
                    async with client.stream(
                        "POST",
                        url,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout,
                    ) as resp:
                        resp.raise_for_status()
                        self._rate_limits.update(provider_name, resp.headers)
                        buffer = ""
                        last_data_at = time_mod.perf_counter()
                        iterator = resp.aiter_text().__aiter__()
                        deadline = (
                            time.perf_counter() + ttft_timeout
                            if ttft_timeout is not None and ttft_timeout > 0
                            else None
                        )
                        while True:
                            try:
                                raw_line = await self._next_stream_line(
                                    iterator,
                                    deadline_monotonic=deadline
                                    if not first_chunk_emitted
                                    else None,
                                )
                            except StopAsyncIteration:
                                break
                            elapsed_since = time_mod.perf_counter() - last_data_at
                            if self.stall_detector.is_stalled(
                                provider_name,
                                since_last_chunk_ms=elapsed_since * 1000.0,
                                fallback_timeout_ms=stall_timeout * 1000.0,
                                stream_id=stream_id,
                            ):
                                self.tacc.record_stall_state(provider_name, True)
                                raise ProviderTimeoutError(
                                    provider=provider_name,
                                    timeout_seconds=stall_timeout,
                                )
                            last_data_at = time_mod.perf_counter()
                            buffer += raw_line
                            chunk_elapsed = elapsed_since * 1000.0
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                if stream_state is not None:
                                    for st_chunk in self._process_sse_line_with_state(
                                        line, stream_state
                                    ):
                                        for optimized in self._optimize_stream_chunk(
                                            st_chunk,
                                            stream_optimizer,
                                            stream_optimizer_state,
                                        ):
                                            chunk_tokens = (
                                                len(self._stream_chunk_text(optimized)) // 4
                                            )
                                            streamed_tokens += chunk_tokens
                                            kind = (
                                                "first_chunk"
                                                if not first_chunk_emitted
                                                else "chunk"
                                            )
                                            self.stall_detector.record_chunk(
                                                provider_name,
                                                kind,
                                                chunk_elapsed,
                                                tokens=chunk_tokens,
                                                stream_id=stream_id,
                                            )
                                            chunk_elapsed = 0.0
                                            if optimized.get("done"):
                                                elapsed_ms = (time.perf_counter() - start) * 1000
                                                await self.tacc.after_response(
                                                    provider_name,
                                                    elapsed_ms,
                                                    max(streamed_tokens, 1),
                                                    200,
                                                )
                                                slot_released = True
                                                feedback_sent = True
                                                yield optimized
                                                return
                                            if not first_chunk_emitted:
                                                ttft_ms = (time.perf_counter() - start) * 1000
                                                await self.tacc.record_ttft(provider_name, ttft_ms)
                                            first_chunk_emitted = True
                                            yield optimized
                                else:
                                    parsed = self._parse_sse_line(line, adapter)
                                    if parsed is not None:
                                        for optimized in self._optimize_stream_chunk(
                                            parsed,
                                            stream_optimizer,
                                            stream_optimizer_state,
                                        ):
                                            chunk_tokens = (
                                                len(self._stream_chunk_text(optimized)) // 4
                                            )
                                            streamed_tokens += chunk_tokens
                                            kind = (
                                                "first_chunk"
                                                if not first_chunk_emitted
                                                else "chunk"
                                            )
                                            self.stall_detector.record_chunk(
                                                provider_name,
                                                kind,
                                                chunk_elapsed,
                                                tokens=chunk_tokens,
                                                stream_id=stream_id,
                                            )
                                            chunk_elapsed = 0.0
                                            if optimized.get("done"):
                                                elapsed_ms = (time.perf_counter() - start) * 1000
                                                await self.tacc.after_response(
                                                    provider_name,
                                                    elapsed_ms,
                                                    max(streamed_tokens, 1),
                                                    200,
                                                )
                                                slot_released = True
                                                feedback_sent = True
                                                yield optimized
                                                return
                                            if not first_chunk_emitted:
                                                ttft_ms = (time.perf_counter() - start) * 1000
                                                await self.tacc.record_ttft(provider_name, ttft_ms)
                                            first_chunk_emitted = True
                                            yield optimized

                        if buffer.strip():
                            if stream_state is not None:
                                for st_chunk in self._process_sse_line_with_state(
                                    buffer.strip(), stream_state
                                ):
                                    for optimized in self._optimize_stream_chunk(
                                        st_chunk,
                                        stream_optimizer,
                                        stream_optimizer_state,
                                    ):
                                        chunk_tokens = len(self._stream_chunk_text(optimized)) // 4
                                        streamed_tokens += chunk_tokens
                                        kind = "first_chunk" if not first_chunk_emitted else "chunk"
                                        self.stall_detector.record_chunk(
                                            provider_name,
                                            kind,
                                            0.0,
                                            tokens=chunk_tokens,
                                            stream_id=stream_id,
                                        )
                                        if optimized.get("done"):
                                            elapsed_ms = (time.perf_counter() - start) * 1000
                                            await self.tacc.after_response(
                                                provider_name,
                                                elapsed_ms,
                                                max(streamed_tokens, 1),
                                                200,
                                            )
                                            slot_released = True
                                            feedback_sent = True
                                            yield optimized
                                            return
                                        if not first_chunk_emitted:
                                            ttft_ms = (time.perf_counter() - start) * 1000
                                            await self.tacc.record_ttft(provider_name, ttft_ms)
                                        first_chunk_emitted = True
                                        yield optimized
                            else:
                                parsed = self._parse_sse_line(buffer.strip(), adapter)
                                if parsed is not None:
                                    for optimized in self._optimize_stream_chunk(
                                        parsed,
                                        stream_optimizer,
                                        stream_optimizer_state,
                                    ):
                                        chunk_tokens = len(self._stream_chunk_text(optimized)) // 4
                                        streamed_tokens += chunk_tokens
                                        kind = "first_chunk" if not first_chunk_emitted else "chunk"
                                        self.stall_detector.record_chunk(
                                            provider_name,
                                            kind,
                                            0.0,
                                            tokens=chunk_tokens,
                                            stream_id=stream_id,
                                        )
                                        if optimized.get("done"):
                                            elapsed_ms = (time.perf_counter() - start) * 1000
                                            await self.tacc.after_response(
                                                provider_name,
                                                elapsed_ms,
                                                max(streamed_tokens, 1),
                                                200,
                                            )
                                            slot_released = True
                                            feedback_sent = True
                                            yield optimized
                                            return
                                        if not first_chunk_emitted:
                                            ttft_ms = (time.perf_counter() - start) * 1000
                                            await self.tacc.record_ttft(provider_name, ttft_ms)
                                        first_chunk_emitted = True
                                        yield optimized

                        elapsed_ms = (time.perf_counter() - start) * 1000
                        await self.tacc.after_response(
                            provider_name,
                            elapsed_ms,
                            max(streamed_tokens, 1),
                            200,
                        )
                        slot_released = True
                        feedback_sent = True
                        return
                except TimeoutError as exc:
                    if not first_chunk_emitted and attempt < no_first_chunk_retries:
                        await self.pool.recycle_client(provider_name, base_url)
                        client = self.pool.get_client(provider_name, base_url)
                        continue
                    if not feedback_sent:
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        await self.tacc.after_response(
                            provider_name,
                            elapsed_ms,
                            max(streamed_tokens, 0),
                            504,
                        )
                        slot_released = True
                        feedback_sent = True
                    raise ProviderTimeoutError(
                        provider=provider_name,
                        timeout_seconds=ttft_timeout or stall_timeout,
                    ) from exc
                except ProviderTimeoutError:
                    if not feedback_sent:
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        await self.tacc.after_response(
                            provider_name,
                            elapsed_ms,
                            max(streamed_tokens, 0),
                            504,
                        )
                        slot_released = True
                        feedback_sent = True
                    raise
                except asyncio.CancelledError:
                    if not feedback_sent:
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        await self.tacc.after_response(
                            provider_name,
                            elapsed_ms,
                            max(streamed_tokens, 0),
                            504,
                        )
                        slot_released = True
                        feedback_sent = True
                    raise
                except httpx.HTTPStatusError as exc:
                    if not feedback_sent:
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        retry_after = self._rate_limits.retry_after(provider_name)
                        await self.tacc.after_response(
                            provider_name,
                            elapsed_ms,
                            max(streamed_tokens, 0),
                            exc.response.status_code,
                            retry_after=retry_after,
                        )
                        slot_released = True
                        feedback_sent = True
                    raise
                except httpx.TimeoutException as exc:
                    if not first_chunk_emitted and attempt < no_first_chunk_retries:
                        await self.pool.recycle_client(provider_name, base_url)
                        client = self.pool.get_client(provider_name, base_url)
                        continue
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    if not feedback_sent:
                        await self.tacc.after_response(provider_name, elapsed_ms, 0, 504)
                        slot_released = True
                        feedback_sent = True
                    raise ProviderTimeoutError(
                        provider=provider_name, timeout_seconds=self.timeout
                    ) from exc
        finally:
            self.stall_detector.end_stream(stream_id)
            if not slot_released:
                await self.tacc.release_request(provider_name)
        return

    @staticmethod
    def _process_sse_line_with_state(line: str, state: Any) -> list[dict[str, Any]]:
        """Process one SSE line through a state machine."""
        line = line.strip()
        if not line or line.startswith(":"):
            return []
        if not line.startswith("data: "):
            return []
        payload = line[len("data: ") :]
        if payload.strip() == "[DONE]":
            return [{"choices": [], "done": True}]
        try:
            data: dict[str, Any] = __import__("json").loads(payload)
        except Exception:
            return []
        result = state.process(data)
        out: list[dict[str, Any]] = list(result.chunks)
        if result.done and out:
            out[-1]["done"] = True
        if result.done and result.metadata and out:
            out[-1].setdefault("_lattice_metadata", {}).update(result.metadata)
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_base_url(self, provider_name: str, api_base: str | None = None) -> str:
        """Resolve base URL for the exact provider.

        Resolution order:
        1. explicit API base passed by caller
        2. per-provider configured base URL (``provider_base_urls``)

        There is no step 3.  Base URLs MUST be configured explicitly via
        ``provider_base_urls`` or the ``LATTICE_PROVIDER_BASE_URLS`` env var.
        If none is configured, the request fails with a clear error so the
        operator knows exactly which provider needs a base URL.
        """
        base_url = api_base or self.provider_base_urls.get(provider_name)
        if not base_url:
            raise ProviderError(
                provider=provider_name,
                status_code=400,
                message=(
                    f"No base URL configured for provider '{provider_name}'. "
                    f"Set it via provider_base_urls['{provider_name}'] or "
                    f"LATTICE_PROVIDER_BASE_URLS env var."
                ),
            )
        return base_url

    def _resolve_api_key(self, provider_name: str, api_key: str | None = None) -> str:
        """Resolve API key: explicit > credential resolver > default > error.

        Raises:
            ProviderError: if no API key can be resolved and the provider
            requires authentication.
        """
        # Explicit override (even empty string = intentional no-auth)
        if api_key is not None:
            return api_key

        # Credentials resolver (config file / env vars)
        if self._credentials is not None:
            creds = self._credentials.resolve(provider_name)
            resolved_key: str | None = creds.api_key
            if resolved_key is not None:
                return resolved_key

        # Default key
        if self.default_api_key is not None:
            return self.default_api_key

        # Ollama local does not need authentication
        if provider_name == "ollama":
            return ""

        # No key found — raise clear error
        from lattice.core.credentials import _PROVIDER_ENV_VARS

        env_var = _PROVIDER_ENV_VARS.get(provider_name, {}).get(
            "api_key", f"{provider_name.upper().replace('-', '_')}_API_KEY"
        )
        raise ProviderError(
            provider=provider_name,
            status_code=401,
            message=(
                f"No API key found for provider '{provider_name}'. "
                f"Set it via: 1) api_key parameter, 2) lattice config file "
                f"(~/.config/lattice/lattice.config.toml), or 3) environment variable "
                f"({env_var})"
            ),
        )

    @staticmethod
    def _build_request(
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stop: list[str] | str | None = None,
        stream: bool = False,
        metadata: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> Request:
        from lattice.core.serialization import message_from_dict

        resolved_stop: list[str] | None = None
        if isinstance(stop, str):
            resolved_stop = [stop]
        elif stop is not None:
            resolved_stop = list(stop)

        return Request(
            messages=[message_from_dict(msg) for msg in messages],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
            stop=resolved_stop,
            metadata=metadata or {},
            extra_headers=extra_headers or {},
            extra_body=extra_body or {},
        )

    # ------------------------------------------------------------------
    # Non-streaming completion with retries
    # ------------------------------------------------------------------

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
        """Send a chat-completion request directly to the provider.

        Steps:
        1. Resolve adapter from *provider_name* or *model* string.
        2. Map model name to provider-native name.
        3. Build internal ``Request``.
        4. Serialize to provider-native JSON via adapter.
        5. Pick HTTP client from pool.
        6. ``POST`` with provider auth + extra headers.
        7. Retry on configured status codes.
        8. Deserialize response via adapter.
        """
        provider_name = _resolve_provider_name(model, provider_name)
        adapter = self.registry.get_adapter(provider_name)

        base_url = self._resolve_base_url(provider_name, api_base)
        key = self._resolve_api_key(provider_name, api_key)

        request = self._build_request(
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
        tacc_estimate = self._tacc_reservation(request)

        payload = adapter.serialize_request(request)
        client = self.pool.get_client(provider_name, base_url)
        url = adapter.chat_endpoint(mapped_model, base_url)
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            await self._await_tacc_admission(provider_name, request)

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
                # Compute TTFT from headers if available, else use total latency
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
                if _should_retry(http_resp.status_code, retry_on) and attempt < max_retries:
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

        # Should never reach here, but satisfy type checker
        if last_error is not None:
            raise last_error
        raise ProviderError(
            provider=provider_name,
            status_code=502,
            message="All retry attempts exhausted",
        )

    # ------------------------------------------------------------------
    # Streaming completion
    # ------------------------------------------------------------------

    async def completion_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stop: list[str] | str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        provider_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator of normalized SSE chunks."""
        provider_name = _resolve_provider_name(model, provider_name)
        adapter = self.registry.get_adapter(provider_name)

        base_url = self._resolve_base_url(provider_name, api_base)
        key = self._resolve_api_key(provider_name, api_key)

        request = self._build_request(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            stop=stop,
            stream=True,
            metadata=metadata or {},
            extra_headers=extra_headers,
            extra_body=extra_body,
        )
        mapped_model = adapter.map_model_name(request.model)
        request.model = mapped_model

        payload = adapter.serialize_request(request)
        client = self.pool.get_client(provider_name, base_url)
        url = adapter.chat_endpoint(mapped_model, base_url)
        headers: dict[str, str] = {"Content-Type": "application/json"}
        headers.update(adapter.auth_headers(key))
        headers.update(adapter.extra_headers(request))
        headers.update(request.extra_headers)

        self._log.info("http_stream_start", provider=provider_name, model=mapped_model, url=url)
        stream_optimizer: Any | None = None
        stream_optimizer_state: Any | None = None
        ttft_timeout, no_first_chunk_retries = self._stream_retry_policy(
            provider_name,
            request.metadata,
        )
        force_fresh_connection = bool(
            request.metadata.get(
                "stream_force_fresh_connection",
                provider_name == "ollama-cloud",
            )
        )

        await self._await_tacc_admission(provider_name, request)
        slot_released = False
        start = time.perf_counter()
        feedback_sent = False
        first_chunk_emitted = False
        streamed_tokens = 0
        last_data_at = time.perf_counter()
        stream_id = f"{provider_name}_{time.perf_counter()}_{id(object())}"
        self.stall_detector.start_stream(provider_name, stream_id)
        try:
            for attempt in range(no_first_chunk_retries + 1):
                try:
                    if force_fresh_connection:
                        await self.pool.recycle_client(provider_name, base_url)
                        client = self.pool.get_client(provider_name, base_url)
                    async with client.stream(
                        "POST",
                        url,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout,
                    ) as resp:
                        resp.raise_for_status()
                        self._rate_limits.update(provider_name, resp.headers)
                        buffer = ""
                        iterator = resp.aiter_text().__aiter__()
                        deadline = (
                            time.perf_counter() + ttft_timeout
                            if ttft_timeout is not None and ttft_timeout > 0
                            else None
                        )
                        while True:
                            try:
                                raw_line = await self._next_stream_line(
                                    iterator,
                                    deadline_monotonic=deadline
                                    if not first_chunk_emitted
                                    else None,
                                )
                            except StopAsyncIteration:
                                break
                            elapsed_since = time.perf_counter() - last_data_at
                            if self.stall_detector.is_stalled(
                                provider_name,
                                since_last_chunk_ms=elapsed_since * 1000.0,
                                fallback_timeout_ms=self._stall_timeout * 1000.0,
                                stream_id=stream_id,
                            ):
                                self.tacc.record_stall_state(provider_name, True)
                                raise ProviderTimeoutError(
                                    provider=provider_name,
                                    timeout_seconds=self._stall_timeout,
                                )
                            last_data_at = time.perf_counter()
                            buffer += raw_line
                            chunk_elapsed = elapsed_since * 1000.0
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                chunk = self._parse_sse_line(line, adapter)
                                if chunk is not None:
                                    for optimized in self._optimize_stream_chunk(
                                        chunk,
                                        stream_optimizer,
                                        stream_optimizer_state,
                                    ):
                                        chunk_tokens = len(self._stream_chunk_text(optimized)) // 4
                                        streamed_tokens += chunk_tokens
                                        kind = "first_chunk" if not first_chunk_emitted else "chunk"
                                        self.stall_detector.record_chunk(
                                            provider_name,
                                            kind,
                                            chunk_elapsed,
                                            tokens=chunk_tokens,
                                            stream_id=stream_id,
                                        )
                                        chunk_elapsed = 0.0
                                        if optimized.get("done"):
                                            elapsed_ms = (time.perf_counter() - start) * 1000
                                            await self.tacc.after_response(
                                                provider_name,
                                                elapsed_ms,
                                                max(streamed_tokens, 1),
                                                200,
                                            )
                                            slot_released = True
                                            feedback_sent = True
                                            yield optimized
                                            return
                                        first_chunk_emitted = True
                                        yield optimized
                        if buffer.strip():
                            chunk = self._parse_sse_line(buffer.strip(), adapter)
                            if chunk is not None:
                                for optimized in self._optimize_stream_chunk(
                                    chunk,
                                    stream_optimizer,
                                    stream_optimizer_state,
                                ):
                                    chunk_tokens = len(self._stream_chunk_text(optimized)) // 4
                                    streamed_tokens += chunk_tokens
                                    kind = "first_chunk" if not first_chunk_emitted else "chunk"
                                    self.stall_detector.record_chunk(
                                        provider_name,
                                        kind,
                                        0.0,
                                        tokens=chunk_tokens,
                                        stream_id=stream_id,
                                    )
                                    if optimized.get("done"):
                                        elapsed_ms = (time.perf_counter() - start) * 1000
                                        await self.tacc.after_response(
                                            provider_name,
                                            elapsed_ms,
                                            max(streamed_tokens, 1),
                                            200,
                                        )
                                        slot_released = True
                                        feedback_sent = True
                                        yield optimized
                                        return
                                    first_chunk_emitted = True
                                    yield optimized

                        elapsed_ms = (time.perf_counter() - start) * 1000
                        await self.tacc.after_response(
                            provider_name,
                            elapsed_ms,
                            max(streamed_tokens, 1),
                            200,
                        )
                        slot_released = True
                        feedback_sent = True
                        return
                except TimeoutError as exc:
                    if not first_chunk_emitted and attempt < no_first_chunk_retries:
                        await self.pool.recycle_client(provider_name, base_url)
                        client = self.pool.get_client(provider_name, base_url)
                        continue
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    if not feedback_sent:
                        await self.tacc.after_response(
                            provider_name,
                            elapsed_ms,
                            max(streamed_tokens, 0),
                            504,
                        )
                        slot_released = True
                        feedback_sent = True
                    raise ProviderTimeoutError(
                        provider=provider_name,
                        timeout_seconds=ttft_timeout or self.timeout,
                    ) from exc
                except asyncio.CancelledError:
                    if not feedback_sent:
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        await self.tacc.after_response(
                            provider_name,
                            elapsed_ms,
                            max(streamed_tokens, 0),
                            504,
                        )
                        slot_released = True
                        feedback_sent = True
                    raise
                except httpx.TimeoutException as exc:
                    if not first_chunk_emitted and attempt < no_first_chunk_retries:
                        await self.pool.recycle_client(provider_name, base_url)
                        client = self.pool.get_client(provider_name, base_url)
                        continue
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    if not feedback_sent:
                        await self.tacc.after_response(
                            provider_name,
                            elapsed_ms,
                            max(streamed_tokens, 0),
                            504,
                        )
                        slot_released = True
                        feedback_sent = True
                    raise ProviderTimeoutError(
                        provider=provider_name,
                        timeout_seconds=self.timeout,
                    ) from exc
                except httpx.HTTPStatusError as exc:
                    if not feedback_sent:
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        retry_after = self._rate_limits.retry_after(provider_name)
                        await self.tacc.after_response(
                            provider_name,
                            elapsed_ms,
                            max(streamed_tokens, 0),
                            exc.response.status_code,
                            retry_after=retry_after,
                        )
                        slot_released = True
                        feedback_sent = True
                    raise
        finally:
            self.stall_detector.end_stream(stream_id)
            if not slot_released:
                await self.tacc.release_request(provider_name)
        # unreachable due to return/raise paths
        return

    @staticmethod
    def _parse_sse_line(line: str, adapter: ProviderAdapter) -> dict[str, Any] | None:
        """Parse one SSE line and normalize via adapter."""
        line = line.strip()
        if not line or line.startswith(":"):
            return None
        if not line.startswith("data: "):
            return None
        payload = line[len("data: ") :]
        if payload.strip() == "[DONE]":
            return {"choices": [], "done": True}
        try:
            data: dict[str, Any] = __import__("json").loads(payload)
        except Exception:
            return None
        return adapter.normalize_sse_chunk(data)

    @staticmethod
    def _optimize_stream_chunk(
        chunk: dict[str, Any],
        optimizer: Any | None,
        state: Any | None,
    ) -> list[dict[str, Any]]:
        if optimizer is None or state is None:
            return [chunk]
        optimized_chunk, emit_done = optimizer.process_chunk(chunk, state)
        out: list[dict[str, Any]] = []
        if optimized_chunk is not None:
            out.append(optimized_chunk)
        if emit_done:
            out.append({"choices": [], "done": True})
        return out
