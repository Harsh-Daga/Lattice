"""Stream resumption wrapper for interrupted SSE/byte streams."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

_RESUME_PROVIDERS = frozenset({"openai", "anthropic", "azure", "groq", "mistral"})

ResumeFn = Callable[[int], AsyncIterator[bytes]]


class StreamResumer:
    """Wrap streaming bodies with byte-offset tracking + reconnect-on-drop."""

    def can_resume(self, provider: str) -> bool:
        return provider in _RESUME_PROVIDERS

    def wrap(
        self,
        byte_iter: AsyncIterator[bytes],
        *,
        provider: str,
        resume_fn: ResumeFn | None = None,
        original_request: Any = None,
        attempt_fn: Any = None,
    ) -> AsyncIterator[bytes]:
        del original_request, attempt_fn
        if not self.can_resume(provider) or resume_fn is None:
            return byte_iter
        return self._resumable_iter(byte_iter, resume_fn=resume_fn)

    async def _resumable_iter(
        self,
        source: AsyncIterator[bytes],
        *,
        resume_fn: ResumeFn,
    ) -> AsyncIterator[bytes]:
        offset = 0
        try:
            async for chunk in source:
                yield chunk
                offset += len(chunk)
        except (
            httpx.ReadError,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            ConnectionError,
            OSError,
        ):
            async for chunk in resume_fn(offset):
                yield chunk

    async def collect_with_resume(
        self,
        source: AsyncIterator[bytes],
        *,
        provider: str,
        resume_fn: ResumeFn,
        expected_len: int | None = None,
    ) -> bytes:
        """Drain a resumable iterator into a single buffer (tests and tooling)."""
        out = bytearray()
        async for chunk in self.wrap(source, provider=provider, resume_fn=resume_fn):
            out.extend(chunk)
        if expected_len is not None and len(out) != expected_len:
            msg = f"stream resume length mismatch: got {len(out)}, want {expected_len}"
            raise ValueError(msg)
        return bytes(out)
