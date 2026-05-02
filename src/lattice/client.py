"""Simplified LATTICE client — compression-only library.

This module provides a minimal, stable API for compressing LLM requests
without running the proxy server. For multi-turn conversations and
transport, users should point their OpenAI client to the LATTICE proxy.

Usage::

    from lattice.client import LatticeClient
    client = LatticeClient()
    result = client.compress(
        messages=[{"role": "user", "content": "Hello!"}],
        model="gpt-4",
    )
    print(result.compressed_messages)
    print(result.stats)
"""

from __future__ import annotations

import asyncio
import dataclasses
import secrets
import time
from typing import Any

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.pipeline import CompressorPipeline
from lattice.core.pipeline_factory import build_default_pipeline, pipeline_summary
from lattice.core.result import is_err, unwrap
from lattice.core.serialization import message_from_dict, message_to_dict
from lattice.core.transport import Request, Response


@dataclasses.dataclass(slots=True)
class CompressResult:
    """Result of a local compression operation."""

    compressed_messages: list[dict[str, Any]]
    original_tokens: int
    compressed_tokens: int
    transforms_applied: list[str]
    elapsed_ms: float
    runtime: dict[str, Any] = dataclasses.field(default_factory=dict)
    runtime_budget: dict[str, Any] = dataclasses.field(default_factory=dict)


def _build_pipeline(config: LatticeConfig) -> CompressorPipeline:
    """Build the standard local compression pipeline."""
    return build_default_pipeline(config)


class LatticeClient:
    """Minimal LATTICE client for local compression.

    Attributes:
        config: LatticeConfig instance.
    """

    def __init__(self, config: LatticeConfig | None = None) -> None:
        self.config = config or LatticeConfig.auto()
        self._pipeline = _build_pipeline(self.config)
        # Store the last compression context for reverse-decompression.
        # Without this, <ref_N>, <g_N>, <crossref_N> placeholders leak
        # into provider responses with no way to restore them.
        self._last_compress_ctx: TransformContext | None = None

    def compress(
        self,
        messages: list[dict[str, Any]],
        model: str = "openai/gpt-4",
        mode: str = "auto",
    ) -> CompressResult:
        """Compress a list of messages.

        Args:
            messages: OpenAI-shaped message list.
            model: Model identifier (used for strategy selection).
            mode: Override compression mode. "auto" uses config value.

        Returns:
            CompressResult with compressed messages and stats.
        """
        start = time.perf_counter()

        if mode != "auto":
            self.config.compression_mode = mode
            self.config.apply_compression_mode()
            self._pipeline = _build_pipeline(self.config)

        request = Request(messages=[message_from_dict(m) for m in messages], model=model)
        original_tokens = request.token_estimate
        compressed = asyncio.run(self._process_request(request))

        elapsed_ms = (time.perf_counter() - start) * 1000
        compressed_tokens = compressed.token_estimate

        return CompressResult(
            compressed_messages=[message_to_dict(m) for m in compressed.messages],
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            transforms_applied=[t.name for t in self._pipeline.transforms],
            elapsed_ms=elapsed_ms,
            runtime=dict(compressed.metadata.get("_lattice_runtime", {})),
            runtime_budget=dict(compressed.metadata.get("_lattice_runtime_budget", {})),
        )

    def decompress_response(
        self,
        response_text: str,
        model: str = "",
    ) -> str:
        """Decompress a provider response by restoring placeholders.

        After calling compress() and sending the compressed messages to a
        provider, the response may contain <ref_N>, <g_N>, <crossref_N>
        placeholders. This method restores them to the original values.

        Args:
            response_text: The provider's raw response content.
            model: Model identifier (optional).

        Returns:
            The response with placeholders restored to original values.
        """
        if self._last_compress_ctx is None:
            return response_text  # No compression was performed
        try:
            resp = Response(content=response_text, model=model)
            restored = asyncio.run(self._pipeline.reverse(resp, self._last_compress_ctx))
            return restored.content or response_text
        except Exception:
            return response_text  # Non-fatal: return raw response

    def compress_request(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str = "openai/gpt-4",
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stop: list[str] | str | None = None,
        stream: bool = False,
        provider_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Request:
        """Compress and return an internal Request object.

        This keeps compatibility with older SDK wrappers while the public
        ``compress()`` API remains message-list oriented.
        """
        request = self._build_request(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            stop=stop,
            stream=stream,
            metadata=metadata,
        )
        return asyncio.run(self._process_request(request, provider_name=provider_name))

    async def compress_request_async(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str = "openai/gpt-4",
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        stop: list[str] | str | None = None,
        stream: bool = False,
        provider_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Request:
        """Async variant of ``compress_request`` for SDK wrappers."""
        request = self._build_request(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            stop=stop,
            stream=stream,
            metadata=metadata,
        )
        return await self._process_request(request, provider_name=provider_name)

    def health(self) -> dict[str, Any]:
        """Return client health and configuration summary."""
        return {
            "status": "healthy",
            "compression_mode": self.config.compression_mode,
            "transforms": [t.name for t in self._pipeline.transforms],
            "pipeline": pipeline_summary(self._pipeline),
            "config_source": "auto",
        }

    def compression_stats(
        self,
        original_messages: list[dict[str, Any]],
        compressed_request: Request,
    ) -> dict[str, Any]:
        """Return token and ratio stats for a compressed request."""
        original_request = Request(
            messages=[message_from_dict(m) for m in original_messages],
            model=compressed_request.model,
            tools=compressed_request.tools,
        )
        tokens_before = original_request.token_estimate
        tokens_after = compressed_request.token_estimate
        if tokens_before <= 0:
            ratio = 0.0
        else:
            ratio = max(0.0, (tokens_before - tokens_after) / tokens_before)
        return {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "tokens_saved": max(0, tokens_before - tokens_after),
            "compression_ratio": ratio,
        }

    def start_session(self, provider: str, model: str) -> str:
        """Create a local SDK session identifier.

        Local compression mode is stateless today; the returned ID is a stable
        compatibility hook for callers that already thread session IDs.
        """
        if not provider:
            raise ValueError("provider is required")
        if not model:
            raise ValueError("model is required")
        return f"lc-{secrets.token_hex(8)}"

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens using the shared token-counting utility."""
        from lattice.utils.token_count import count_tokens

        return count_tokens(text, model=model)

    @staticmethod
    def _build_request(
        *,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float | None,
        max_tokens: int | None,
        top_p: float | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        stop: list[str] | str | None,
        stream: bool,
        metadata: dict[str, Any] | None,
    ) -> Request:
        if isinstance(stop, str):
            stop = [stop]
        request = Request(
            messages=[message_from_dict(m) for m in messages],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
            stop=stop,
        )
        if metadata:
            request.metadata.update(metadata)
        return request

    async def _process_request(
        self,
        request: Request,
        *,
        provider_name: str | None = None,
    ) -> Request:
        provider = provider_name or _provider_from_model(request.model)
        ctx = TransformContext(
            request_id=str(time.time()),
            provider=provider,
            model=request.model,
        )
        result = await self._pipeline.process(request, ctx)
        self._last_compress_ctx = ctx  # Save for decompress_response()
        if is_err(result):
            if self.config.graceful_degradation:
                return request
            raise RuntimeError(f"Compression failed: {result}")
        compressed = unwrap(result)
        runtime = compressed.metadata.setdefault("_lattice_runtime", {})
        transport = runtime.setdefault("transport", {})
        transport.setdefault("framing", "json")
        transport.setdefault("delta", "bypassed")
        return compressed


def _provider_from_model(model: str) -> str:
    """Best-effort provider extraction for local SDK compression context."""
    if "/" in model:
        return model.split("/", 1)[0] or "openai"
    return "openai"
