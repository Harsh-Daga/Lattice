"""Unit tests for heuristic rate-distortion compressor."""

from __future__ import annotations

from lattice.core.context import TransformContext
from lattice.core.result import unwrap
from lattice.core.transport import Message, Request, Response
from lattice.transforms.rate_distortion import RateDistortionCompressor


def test_rate_distortion_skips_structured_content() -> None:
    transform = RateDistortionCompressor(distortion_budget=0.2, max_input_tokens=1)
    request = Request(messages=[Message(role="user", content='{"a": 1, "b": 2}')])
    result = transform.process(request, TransformContext())
    modified = unwrap(result)
    assert modified.messages[0].content == '{"a": 1, "b": 2}'


def test_rate_distortion_preserves_question_under_tight_budget() -> None:
    transform = RateDistortionCompressor(distortion_budget=0.002, max_input_tokens=1)
    text = (
        "This intro sentence has low value. "
        "What is the final numeric answer for the report? "
        "Another low priority sentence for context."
    )
    request = Request(messages=[Message(role="user", content=text)])
    result = transform.process(request, TransformContext())
    modified = unwrap(result)
    assert "What is the final numeric answer for the report?" in modified.messages[0].content


def test_rate_distortion_removes_low_value_under_relaxed_budget() -> None:
    transform = RateDistortionCompressor(distortion_budget=0.03, max_input_tokens=1)
    text = (
        "Fluff one. "
        "Fluff two. "
        "Critical summary because this is important and required."
    )
    request = Request(messages=[Message(role="user", content=text)])
    result = transform.process(request, TransformContext())
    modified = unwrap(result)
    content = modified.messages[0].content
    assert "Critical summary because this is important and required." in content
    assert len(content) < len(text)


def test_rate_distortion_reverse_is_noop() -> None:
    transform = RateDistortionCompressor()
    response = Response(content="hello")
    restored = transform.reverse(response, TransformContext())
    assert restored.content == "hello"
