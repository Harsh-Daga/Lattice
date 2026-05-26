"""Stream resume delivers full byte stream after simulated TLS drop at offset 4096."""

from __future__ import annotations

import httpx
import pytest

from lattice.transport.stream_resume import StreamResumer

_PAYLOAD = b"x" * 8000


@pytest.mark.asyncio
async def test_resume_after_drop_at_4096() -> None:
    async def source() -> bytes:
        yield _PAYLOAD[:4096]
        raise httpx.ReadError("connection dropped")

    async def resume_from(offset: int) -> bytes:
        assert offset == 4096
        yield _PAYLOAD[offset:]

    resumer = StreamResumer()
    out = await resumer.collect_with_resume(
        source(),
        provider="openai",
        resume_fn=resume_from,
        expected_len=8000,
    )
    assert out == _PAYLOAD
    assert len(out) == 8000
