"""Unit tests for SpeculativeExecutor and SpeculativeTransform.

Tests cover:
- Rule-based prediction heuristics
- Confidence scoring
- Speculative request building
- Hit/miss detection
- Pipeline integration
"""

from __future__ import annotations

import pytest

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.pipeline import CompressorPipeline
from lattice.core.result import unwrap
from lattice.core.transport import Message, Request, Response
from lattice.transforms.speculative import (
    SpeculativeExecutor,
    SpeculativeTransform,
)

# =============================================================================
# Prediction
# =============================================================================


class TestPrediction:
    """Tests for SpeculativeExecutor.predict()."""

    def test_tool_call_prediction(self) -> None:
        executor = SpeculativeExecutor()
        req = Request(
            messages=[Message(role="user", content="Please search for python docs")],
            tools=[{"type": "function", "function": {"name": "search"}}],
        )
        ctx = TransformContext()
        pred = executor.predict(req, ctx)
        assert pred == "tool_call"

    def test_no_tools_no_prediction(self) -> None:
        executor = SpeculativeExecutor()
        req = Request(messages=[Message(role="user", content="Hello")])
        ctx = TransformContext()
        pred = executor.predict(req, ctx)
        assert pred is None

    def test_code_completion_prediction(self) -> None:
        executor = SpeculativeExecutor()
        req = Request(messages=[Message(role="user", content="Write a function that sorts a list")])
        ctx = TransformContext()
        pred = executor.predict(req, ctx)
        assert pred == "code_completion"

    def test_user_answer_prediction(self) -> None:
        executor = SpeculativeExecutor()
        req = Request(
            messages=[
                Message(role="assistant", content="What is your name?"),
                Message(role="user", content="Alice"),
            ]
        )
        ctx = TransformContext()
        pred = executor.predict(req, ctx)
        assert pred == "user_answer"


# =============================================================================
# Confidence
# =============================================================================


class TestConfidence:
    def test_tool_call_confidence(self) -> None:
        executor = SpeculativeExecutor()
        req = Request(
            messages=[Message(role="user", content="Search")],
            tools=[{"type": "function", "function": {"name": "search"}}],
        )
        conf = executor.confidence("tool_call", req)
        assert conf == 0.85

    def test_user_answer_confidence(self) -> None:
        executor = SpeculativeExecutor()
        conf = executor.confidence("user_answer", Request(messages=[]))
        assert conf == 0.60


# =============================================================================
# Hit / Miss
# =============================================================================


class TestHitMiss:
    def test_tool_call_hit(self) -> None:
        executor = SpeculativeExecutor()
        assert executor.is_hit("tool_call", "tool_call") is True

    def test_completion_hit(self) -> None:
        executor = SpeculativeExecutor()
        assert executor.is_hit("code_completion", "completion") is False

    def test_none_actual(self) -> None:
        executor = SpeculativeExecutor()
        assert executor.is_hit("tool_call", None) is False

    def test_extract_actual_tool_call(self) -> None:
        executor = SpeculativeExecutor()
        resp = Response(content="", tool_calls=[{"id": "1"}])
        assert executor.extract_actual_step(resp) == "tool_call"

    def test_extract_actual_completion(self) -> None:
        executor = SpeculativeExecutor()
        resp = Response(content="Hello")
        assert executor.extract_actual_step(resp) == "completion"


# =============================================================================
# Speculative Execution
# =============================================================================


class TestSpeculativeExecution:
    @pytest.mark.asyncio
    async def test_run_speculative_without_caller(self) -> None:
        executor = SpeculativeExecutor(provider_caller=None)
        req = Request(messages=[Message(role="user", content="Hello")])
        result = await executor.run_speculative(req, "completion")
        assert result is None

    @pytest.mark.asyncio
    async def test_run_speculative_success(self) -> None:
        async def fake_provider(_req: Request) -> Response:
            return Response(content="speculative result", model="gpt-4")

        executor = SpeculativeExecutor(provider_caller=fake_provider)
        req = Request(messages=[Message(role="user", content="Hello")])
        result = await executor.run_speculative(req, "completion")
        assert isinstance(result, Response)
        assert result.content == "speculative result"

    @pytest.mark.asyncio
    async def test_run_speculative_failure_graceful(self) -> None:
        async def failing_provider(_req: Request) -> Response:
            raise RuntimeError("boom")

        executor = SpeculativeExecutor(provider_caller=failing_provider)
        req = Request(messages=[Message(role="user", content="Hello")])
        result = await executor.run_speculative(req, "completion")
        assert result is None


# =============================================================================
# Stats
# =============================================================================


class TestStats:
    def test_accuracy_empty(self) -> None:
        executor = SpeculativeExecutor()
        assert executor.accuracy == 0.0
        assert executor.stats["total"] == 0

    def test_accuracy_after_hits(self) -> None:
        executor = SpeculativeExecutor()
        executor.record_result(
            hit=True, _predicted="tool_call", _actual="tool_call", latency_ms=100.0
        )
        executor.record_result(
            hit=False, _predicted="tool_call", _actual="completion", latency_ms=100.0
        )
        assert executor.accuracy == 0.5
        assert executor.stats["hits"] == 1
        assert executor.stats["total"] == 2


# =============================================================================
# Pipeline Integration
# =============================================================================


class TestSpeculativeTransform:
    def test_prediction_metadata(self) -> None:
        st = SpeculativeTransform()
        req = Request(
            messages=[Message(role="user", content="Search python")],
            tools=[{"type": "function", "function": {"name": "search"}}],
        )
        ctx = TransformContext()
        result = st.process(req, ctx)
        assert result.value == req
        # transforms_applied is only populated by the pipeline, not direct calls
        assert ctx.metrics["transforms"]["speculative"]["prediction"] == "tool_call"
        assert ctx.session_state.get("speculative_prediction") == "tool_call"

    def test_low_confidence_no_prediction(self) -> None:
        st = SpeculativeTransform()
        req = Request(messages=[Message(role="user", content="Hello")])
        ctx = TransformContext()
        result = st.process(req, ctx)
        assert result.value == req
        assert "speculative" not in ctx.metrics["transforms"]

    @pytest.mark.asyncio
    async def test_in_pipeline(self) -> None:
        config = LatticeConfig()
        pipeline = CompressorPipeline(config=config)
        pipeline.register(SpeculativeTransform())
        req = Request(
            messages=[Message(role="user", content="Search")],
            tools=[{"type": "function", "function": {"name": "search"}}],
        )
        ctx = TransformContext()
        result = await pipeline.process(req, ctx)
        unwrap(result)
        assert "speculative" in ctx.transforms_applied
