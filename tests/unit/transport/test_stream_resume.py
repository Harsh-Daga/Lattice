"""Stream resumer passthrough and resume-on-error."""

from __future__ import annotations

import httpx
import pytest

from lattice.transport.stream_resume import StreamResumer


@pytest.mark.asyncio
async def test_non_resumable_passthrough() -> None:
    resumer = StreamResumer()

    async def src():
        yield b"abc"
        yield b"def"

    out = b""
    wrapped = resumer.wrap(src(), provider="unknown", original_request=None, attempt_fn=None)
    async for chunk in wrapped:
        out += chunk
    assert out == b"abcdef"


@pytest.mark.asyncio
async def test_resume_on_read_error() -> None:
    payload = b"a" * 100 + b"b" * 100

    async def src():
        yield payload[:100]
        raise httpx.ReadError("drop")

    async def resume(off: int):
        assert off == 100
        yield payload[off:]

    resumer = StreamResumer()
    out = await resumer.collect_with_resume(
        src(),
        provider="openai",
        resume_fn=resume,
        expected_len=200,
    )
    assert out == payload
