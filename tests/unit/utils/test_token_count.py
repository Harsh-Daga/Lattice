"""Tests for lattice.utils.token_count."""

from __future__ import annotations

from lattice.utils.token_count import (
    ApproximateCounter,
    ModelTokenCounter,
    TiktokenCounter,
    count_message_tokens,
    count_tokens,
)


def test_approximate_counter_empty_is_one() -> None:
    assert ApproximateCounter().count("") == 1


def test_approximate_counter_scales_by_ratio() -> None:
    counter = ApproximateCounter(ratio=4.0)
    assert counter.count("abcd") == 1
    assert counter.count("abcdefgh") == 2


def test_tiktoken_counter_minimum_one() -> None:
    assert TiktokenCounter().count("") == 1
    assert TiktokenCounter().count("hello") >= 1


def test_count_tokens_openai_model() -> None:
    assert count_tokens("Hello, world!", model="gpt-4") >= 1


def test_count_tokens_non_openai_uses_approximate() -> None:
    counter = ModelTokenCounter()
    assert isinstance(counter.get_counter("claude-3-5-sonnet"), ApproximateCounter)


def test_count_message_tokens_includes_overhead() -> None:
    messages = [{"role": "user", "content": "hi"}]
    single = count_tokens("hi", model="gpt-4")
    total = count_message_tokens(messages, model="gpt-4")
    assert total > single
