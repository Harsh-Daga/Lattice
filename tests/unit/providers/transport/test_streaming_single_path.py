"""Both public stream methods delegate to unified ``_stream``."""

from __future__ import annotations

from typing import Any

import pytest

from lattice.transport import DirectHTTPProvider
from lattice.transport.types import Message, Request


@pytest.mark.asyncio
async def test_completion_stream_uses_unified_path(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = DirectHTTPProvider()
    seen: list[dict[str, Any]] = []

    async def fake_stream(*args: Any, **kwargs: Any):
        seen.append(
            {
                "stall_detect": kwargs.get("use_stream_state_machine"),
                "record_ttft": kwargs.get("record_ttft_on_first"),
            }
        )
        yield {"choices": [{"delta": {"content": "x"}}]}

    monkeypatch.setattr(provider, "_stream", fake_stream)

    request = Request(model="openai/gpt-4", messages=[Message(role="user", content="hi")])
    async for _ in provider.completion_stream(
        request.model,
        [{"role": "user", "content": "hi"}],
        provider_name="openai",
    ):
        pass
    assert seen[0]["stall_detect"] is False
    assert seen[0]["record_ttft"] is False

    seen.clear()
    async for _ in provider.completion_stream_with_stall_detect(
        request.model,
        [{"role": "user", "content": "hi"}],
        provider_name="openai",
    ):
        pass
    assert seen[0]["stall_detect"] is True
    assert seen[0]["record_ttft"] is True
