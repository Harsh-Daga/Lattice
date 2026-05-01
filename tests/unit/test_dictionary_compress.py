"""Unit tests for dictionary_compress transform."""

from __future__ import annotations

from lattice.core.context import TransformContext
from lattice.core.result import unwrap
from lattice.core.transport import Message, Request, Response
from lattice.transforms.dictionary_compress import DictionaryCompressor


def test_static_phrase_replacement() -> None:
    transform = DictionaryCompressor()
    request = Request(messages=[Message(role="user", content="The quick brown fox.")])
    result = transform.process(request, TransformContext())
    modified = unwrap(result)
    assert "<d_0>" in modified.messages[0].content
    assert "The " not in modified.messages[0].content


def test_dynamic_learning_and_reverse() -> None:
    transform = DictionaryCompressor(min_occurrences=2)
    context = TransformContext()
    text = (
        "The algorithm uses a greedy heuristic to solve the problem. "
        "The algorithm uses a greedy heuristic to solve the problem. "
        "Please summarize the results."
    )
    request = Request(messages=[Message(role="user", content=text)])
    result = transform.process(request, context)
    modified = unwrap(result)
    compressed = modified.messages[0].content
    # Since there are repeated phrases, compression should differ from original
    assert compressed != text
    # Reverse
    response = Response(content=compressed)
    restored = transform.reverse(response, context)
    assert restored.content == text


def test_reverse_no_state_is_noop() -> None:
    transform = DictionaryCompressor()
    response = Response(content="hello world")
    restored = transform.reverse(response, TransformContext())
    assert restored.content == "hello world"


def test_metrics_recorded() -> None:
    transform = DictionaryCompressor()
    context = TransformContext()
    request = Request(messages=[Message(role="user", content="The function definition follows.")])
    result = transform.process(request, context)
    unwrap(result)
    metrics = context.get_transform_state("dictionary_compress")
    assert "dict" in metrics
    # static dict entries are seeded
    assert len(metrics["dict"]) > 0


def test_max_dynamic_entries_respected() -> None:
    transform = DictionaryCompressor(max_dynamic_entries=0)
    context = TransformContext()
    text = "The function foo is used. The function foo is used." * 10
    request = Request(messages=[Message(role="user", content=text)])
    result = transform.process(request, context)
    _ = unwrap(result)
    # With max_dynamic_entries=0, only static replacements happen
    state = context.get_transform_state("dictionary_compress")
    assert len(state["dict"]) == len(transform._static_dict)
