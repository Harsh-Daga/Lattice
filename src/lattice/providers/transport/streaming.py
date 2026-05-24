"""Unified streaming transport for DirectHTTPProvider."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import structlog

from lattice.core.errors import ProviderTimeoutError
from lattice.providers.transport.helpers import (
    await_tacc_admission,
    build_request,
    next_stream_line,
    optimize_stream_chunk,
    parse_sse_line,
    process_sse_line_with_state,
    resolve_api_key,
    resolve_base_url,
    stream_chunk_text,
    stream_retry_policy,
)
from lattice.providers.transport.registry import _resolve_provider_name

logger = structlog.get_logger()


class StreamingMixin:
    """Streaming completion paths (merged single ``_stream`` implementation)."""

    async def _stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        use_stream_state_machine: bool = False,
        record_ttft_on_first: bool = False,
        log_event: str = "http_stream_start",
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
        import time as time_mod

        provider_name = _resolve_provider_name(model, provider_name, self.registry)
        adapter = self.registry.get_adapter(provider_name)

        # Build base URL + key
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
            log_event,
            provider=provider_name,
            model=mapped_model,
            url=url,
            stall_timeout=self._stall_timeout,
        )

        # Detect state-machine streaming (Anthropic tool_use + thinking)
        stream_state: Any | None = None
        if use_stream_state_machine and hasattr(adapter, "normalize_sse_stream"):
            stream_state = adapter.normalize_sse_stream(mapped_model)
        stream_optimizer: Any | None = None
        stream_optimizer_state: Any | None = None
        ttft_timeout, no_first_chunk_retries = stream_retry_policy(
            provider_name,
            request.metadata,
        )
        force_fresh_connection = bool(
            request.metadata.get(
                "stream_force_fresh_connection",
                provider_name == "ollama-cloud",
            )
        )

        await await_tacc_admission(self, provider_name, request)
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
                                raw_line = await next_stream_line(
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
                                fallback_timeout_ms=self._stall_timeout * 1000.0,
                                stream_id=stream_id,
                            ):
                                self.tacc.record_stall_state(provider_name, True)
                                raise ProviderTimeoutError(
                                    provider=provider_name,
                                    timeout_seconds=self._stall_timeout,
                                )
                            last_data_at = time_mod.perf_counter()
                            buffer += raw_line
                            chunk_elapsed = elapsed_since * 1000.0
                            while "\n" in buffer:
                                line, buffer = buffer.split("\n", 1)
                                if stream_state is not None:
                                    for st_chunk in process_sse_line_with_state(line, stream_state):
                                        for optimized in optimize_stream_chunk(
                                            st_chunk,
                                            stream_optimizer,
                                            stream_optimizer_state,
                                        ):
                                            chunk_tokens = len(stream_chunk_text(optimized)) // 4
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
                                            if record_ttft_on_first and not first_chunk_emitted:
                                                ttft_ms = (time.perf_counter() - start) * 1000
                                                await self.tacc.record_ttft(provider_name, ttft_ms)
                                            first_chunk_emitted = True
                                            yield optimized
                                else:
                                    parsed = parse_sse_line(line, adapter)
                                    if parsed is not None:
                                        for optimized in optimize_stream_chunk(
                                            parsed,
                                            stream_optimizer,
                                            stream_optimizer_state,
                                        ):
                                            chunk_tokens = len(stream_chunk_text(optimized)) // 4
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
                                            if record_ttft_on_first and not first_chunk_emitted:
                                                ttft_ms = (time.perf_counter() - start) * 1000
                                                await self.tacc.record_ttft(provider_name, ttft_ms)
                                            first_chunk_emitted = True
                                            yield optimized

                        if buffer.strip():
                            if stream_state is not None:
                                for st_chunk in process_sse_line_with_state(
                                    buffer.strip(), stream_state
                                ):
                                    for optimized in optimize_stream_chunk(
                                        st_chunk,
                                        stream_optimizer,
                                        stream_optimizer_state,
                                    ):
                                        chunk_tokens = len(stream_chunk_text(optimized)) // 4
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
                                parsed = parse_sse_line(buffer.strip(), adapter)
                                if parsed is not None:
                                    for optimized in optimize_stream_chunk(
                                        parsed,
                                        stream_optimizer,
                                        stream_optimizer_state,
                                    ):
                                        chunk_tokens = len(stream_chunk_text(optimized)) // 4
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
                        timeout_seconds=ttft_timeout or self._stall_timeout,
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
        async for chunk in self._stream(
            model,
            messages,
            use_stream_state_machine=True,
            record_ttft_on_first=True,
            log_event="http_stream_start_with_stall",
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            stop=stop,
            api_base=api_base,
            api_key=api_key,
            provider_name=provider_name,
            metadata=metadata,
            extra_headers=extra_headers,
            extra_body=extra_body,
            **_kwargs,
        ):
            yield chunk

    async def completion_stream(
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
        async for chunk in self._stream(
            model,
            messages,
            use_stream_state_machine=False,
            record_ttft_on_first=False,
            log_event="http_stream_start",
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            stop=stop,
            api_base=api_base,
            api_key=api_key,
            provider_name=provider_name,
            metadata=metadata,
            extra_headers=extra_headers,
            extra_body=extra_body,
            **_kwargs,
        ):
            yield chunk
