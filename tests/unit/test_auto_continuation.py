"""Tests for auto-continuation.

Covers:
- No continuation when finish_reason is "stop"
- Continuation when finish_reason is "length"
- Multiple turns of continuation
- Max turns limit
- Content and tool_call accumulation
- Usage aggregation
- Failure handling during continuation
"""

from __future__ import annotations

import pytest

from lattice.core.auto_continuation import AutoContinuation
from lattice.core.transport import Message, Request, Response


class DummyMessage(Message):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self):
        return {"role": self.role, "content": self.content}


class TestAutoContinuation:
    @pytest.fixture
    def continuator(self):
        return AutoContinuation(max_turns=3)

    @pytest.fixture
    def chat_request(self):
        return Request(
            model="gpt-4",
            messages=[
                DummyMessage(role="system", content="You are helpful."),
                DummyMessage(role="user", content="Tell me a very long story."),
            ],
        )

    async def test_no_continuation_when_stop(self, continuator, chat_request) -> None:
        initial = Response(content="The end.", finish_reason="stop")

        async def mock_provider(**_kwargs):
            return Response(content="should not be called", finish_reason="stop")

        result = await continuator.continue_if_needed(
            request=chat_request,
            initial_response=initial,
            provider_caller=mock_provider,
            session_manager=None,
            message_cls=DummyMessage,
            provider_name="openai",
        )
        assert result.was_continued is False
        assert result.turns == 0
        assert result.response.content == "The end."

    async def test_continuation_when_length(self, continuator, chat_request) -> None:
        initial = Response(content="Once upon", finish_reason="length")

        async def mock_provider(**_kwargs):
            return Response(content=" a time...", finish_reason="stop")

        result = await continuator.continue_if_needed(
            request=chat_request,
            initial_response=initial,
            provider_caller=mock_provider,
            session_manager=None,
            message_cls=DummyMessage,
            provider_name="openai",
        )
        assert result.was_continued is True
        assert result.turns == 1
        assert result.response.content == "Once upon a time..."
        assert result.response.finish_reason == "stop"

    async def test_multiple_turns(self, continuator, chat_request) -> None:
        initial = Response(content="A", finish_reason="length")
        call_count = 0

        async def mock_provider(**_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return Response(content="B", finish_reason="length")
            return Response(content="C", finish_reason="stop")

        result = await continuator.continue_if_needed(
            request=chat_request,
            initial_response=initial,
            provider_caller=mock_provider,
            session_manager=None,
            message_cls=DummyMessage,
            provider_name="openai",
        )
        assert result.turns == 2
        assert result.response.content == "ABC"

    async def test_max_turns_limit(self, chat_request) -> None:
        continuator = AutoContinuation(max_turns=2)
        initial = Response(content="A", finish_reason="length")

        async def mock_provider(**_kwargs):
            return Response(content="B", finish_reason="length")

        result = await continuator.continue_if_needed(
            request=chat_request,
            initial_response=initial,
            provider_caller=mock_provider,
            session_manager=None,
            message_cls=DummyMessage,
            provider_name="openai",
        )
        assert result.turns == 2
        assert result.response.content == "ABB"
        assert result.response.finish_reason == "length"

    async def test_usage_aggregation(self, continuator, chat_request) -> None:
        initial = Response(
            content="A",
            finish_reason="length",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )

        async def mock_provider(**_kwargs):
            return Response(
                content="B",
                finish_reason="stop",
                usage={"prompt_tokens": 110, "completion_tokens": 60},
            )

        result = await continuator.continue_if_needed(
            request=chat_request,
            initial_response=initial,
            provider_caller=mock_provider,
            session_manager=None,
            message_cls=DummyMessage,
            provider_name="openai",
        )
        assert result.response.usage["prompt_tokens"] == 210
        assert result.response.usage["completion_tokens"] == 110

    async def test_tool_calls_accumulated(self, continuator, chat_request) -> None:
        initial = Response(
            content="Calling tool",
            finish_reason="length",
            tool_calls=[{"id": "tc1", "function": {"name": "get_weather"}}],
        )

        async def mock_provider(**_kwargs):
            return Response(
                content="Done",
                finish_reason="stop",
                tool_calls=[{"id": "tc2", "function": {"name": "get_time"}}],
            )

        result = await continuator.continue_if_needed(
            request=chat_request,
            initial_response=initial,
            provider_caller=mock_provider,
            session_manager=None,
            message_cls=DummyMessage,
            provider_name="openai",
        )
        assert len(result.response.tool_calls) == 2
        assert result.response.tool_calls[0]["id"] == "tc1"
        assert result.response.tool_calls[1]["id"] == "tc2"

    async def test_provider_failure_graceful(self, continuator, chat_request) -> None:
        initial = Response(content="Partial", finish_reason="length")

        async def failing_provider(**_kwargs):
            raise RuntimeError("provider exploded")

        result = await continuator.continue_if_needed(
            request=chat_request,
            initial_response=initial,
            provider_caller=failing_provider,
            session_manager=None,
            message_cls=DummyMessage,
            provider_name="openai",
        )
        assert result.was_continued is False  # No turns completed
        assert result.response.content == "Partial"
        assert result.response.finish_reason == "length"

    async def test_request_messages_appended(self, continuator, chat_request) -> None:
        initial = Response(content="Partial", finish_reason="length")
        original_len = len(chat_request.messages)

        async def mock_provider(**_kwargs):
            return Response(content="Complete", finish_reason="stop")

        await continuator.continue_if_needed(
            request=chat_request,
            initial_response=initial,
            provider_caller=mock_provider,
            session_manager=None,
            message_cls=DummyMessage,
            provider_name="openai",
        )
        assert len(chat_request.messages) == original_len + 1
        assert chat_request.messages[-1].role == "assistant"
        assert chat_request.messages[-1].content == "Partial"


class TestInit:
    def test_max_turns_validation(self) -> None:
        with pytest.raises(ValueError):
            AutoContinuation(max_turns=0)
