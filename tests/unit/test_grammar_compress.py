"""Unit tests for grammar_compress transform."""

from __future__ import annotations

import json

from lattice.core.context import TransformContext
from lattice.core.result import unwrap
from lattice.core.transport import Message, Request, Response
from lattice.transforms.grammar_compress import GrammarCompressor


def test_compress_json_like_content() -> None:
    transform = GrammarCompressor()
    content = json.dumps({"city": "New York", "country": "United States", "city_alias": "New York"})
    request = Request(messages=[Message(role="user", content=content)])
    context = TransformContext()
    result = transform.process(request, context)
    modified = unwrap(result)
    compressed = modified.messages[0].content
    # Some replacement should have occurred
    assert "<g_" in compressed
    # Reverse should restore original
    response = Response(content=compressed)
    restored = transform.reverse(response, context)
    assert json.loads(restored.content) == json.loads(content)


def test_compress_table_like_content() -> None:
    transform = GrammarCompressor()
    content = (
        "| Name | City | Country |\n"
        "|------|------|----------|\n"
        "| Alice | New York | United States |\n"
        "| Bob | New York | United States |\n"
    )
    request = Request(messages=[Message(role="user", content=content)])
    context = TransformContext()
    result = transform.process(request, context)
    modified = unwrap(result)
    compressed = modified.messages[0].content
    # Table rows have repeated values, expect aliases
    assert "<g_" in compressed
    response = Response(content=compressed)
    restored = transform.reverse(response, context)
    assert "New York" in restored.content
    assert "United States" in restored.content


def test_non_structured_content_unchanged() -> None:
    transform = GrammarCompressor()
    content = "Just a plain sentence without JSON or tables."
    request = Request(messages=[Message(role="user", content=content)])
    context = TransformContext()
    result = transform.process(request, context)
    modified = unwrap(result)
    assert modified.messages[0].content == content


def test_reverse_with_empty_state() -> None:
    transform = GrammarCompressor()
    response = Response(content="hello world")
    restored = transform.reverse(response, TransformContext())
    assert restored.content == "hello world"


def test_metrics_recorded() -> None:
    transform = GrammarCompressor()
    context = TransformContext()
    content = json.dumps({"key": "long_value_string_here", "key2": "long_value_string_here"})
    request = Request(messages=[Message(role="user", content=content)])
    result = transform.process(request, context)
    unwrap(result)
    metrics = context.metrics["transforms"].get("grammar_compress", {})
    assert metrics.get("messages_modified", 0) >= 1
    assert "compression_ratio" in metrics


def test_max_dict_size_respected() -> None:
    transform = GrammarCompressor(max_dict_size=1)
    content = json.dumps({"a": "one", "b": "two", "c": "three"})
    request = Request(messages=[Message(role="user", content=content)])
    context = TransformContext()
    result = transform.process(request, context)
    _ = unwrap(result)
    state = context.get_transform_state("grammar_compress")
    assert len(state["alias_map"]) <= 1
