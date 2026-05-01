"""Unit tests for submodular context selector."""

from __future__ import annotations

from lattice.core.context import TransformContext
from lattice.core.result import is_ok, unwrap
from lattice.core.transport import Message, Request, Response
from lattice.transforms.context_selector import (
    InformationTheoreticSelector,
    SubmodularContextSelector,
)

# =============================================================================
# SubmodularContextSelector
# =============================================================================


def test_selector_noop_when_no_docs() -> None:
    transform = SubmodularContextSelector(token_budget=100)
    request = Request(messages=[Message(role="user", content="Hello?")])
    result = transform.process(request, TransformContext())
    assert is_ok(result)
    modified = unwrap(result)
    assert len(modified.messages) == 1
    assert modified.messages[0].content == "Hello?"


def test_selector_noop_when_under_budget() -> None:
    transform = SubmodularContextSelector(token_budget=10000)
    request = Request(
        messages=[
            Message(role="system", content="Be helpful."),
            Message(role="user", content="What is 2+2?"),
        ]
    )
    result = transform.process(request, TransformContext())
    modified = unwrap(result)
    assert len(modified.messages) == 2
    assert modified.messages[0].content == "Be helpful."


def test_selector_selects_high_relevance_docs() -> None:
    transform = SubmodularContextSelector(token_budget=15)
    request = Request(
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(
                role="assistant",
                content="Python is a programming language with dynamic typing and garbage collection.",
            ),
            Message(
                role="assistant",
                content="The capital of France is Paris, located on the Seine river.",
            ),
            Message(role="user", content="Tell me about Paris tourism and landmarks."),
        ]
    )
    context = TransformContext()
    result = transform.process(request, context)
    modified = unwrap(result)

    # The Paris-related doc should be selected and moved earlier
    contents = [m.content for m in modified.messages]
    assert "Paris" in contents[0]
    # User message should remain at the end
    assert modified.messages[-1].content == "Tell me about Paris tourism and landmarks."

    # Metrics should be recorded
    metrics = context.metrics["transforms"].get("context_selector", {})
    assert metrics["selected_docs"] >= 1
    assert metrics["total_docs"] == 3  # system + 2 assistant docs


def test_selector_respects_token_budget() -> None:
    transform = SubmodularContextSelector(token_budget=100)
    long_doc = "x" * 400  # ~100 tokens
    request = Request(
        messages=[
            Message(role="system", content="System prompt."),
            Message(role="assistant", content=long_doc),
            Message(role="assistant", content=long_doc),
            Message(role="user", content="What is x?"),
        ]
    )
    result = transform.process(request, TransformContext())
    modified = unwrap(result)
    # At most one long doc should fit in the token budget
    selected_long_docs = sum(1 for m in modified.messages if len(m.content) == 400)
    assert selected_long_docs <= 1


def test_selector_diversity_bonus() -> None:
    # Two docs both relevant to query but covering different aspects
    transform = SubmodularContextSelector(token_budget=120)
    request = Request(
        messages=[
            Message(role="system", content="Be concise."),
            Message(
                role="assistant", content="Paris museums include the Louvre and Musée d'Orsay."
            ),
            Message(role="assistant", content="Paris cuisine features croissants and escargot."),
            Message(role="user", content="What should I know about Paris?"),
        ]
    )
    result = transform.process(request, TransformContext())
    modified = unwrap(result)
    contents = [m.content for m in modified.messages]
    # Both Paris docs should likely be selected because they are diverse
    assert any("Louvre" in c for c in contents)
    assert any("croissants" in c for c in contents)


def test_selector_reverse_is_noop() -> None:
    transform = SubmodularContextSelector()
    response = Response(content="hello world")
    restored = transform.reverse(response, TransformContext())
    assert restored.content == "hello world"


# =============================================================================
# InformationTheoreticSelector
# =============================================================================


def test_information_selector_extends_base() -> None:
    transform = InformationTheoreticSelector(token_budget=50)
    request = Request(
        messages=[
            Message(role="system", content="Helpful assistant."),
            Message(role="assistant", content="Paris is beautiful in spring."),
            Message(role="user", content="Tell me about Paris."),
        ]
    )
    result = transform.process(request, TransformContext())
    modified = unwrap(result)
    contents = [m.content for m in modified.messages]
    assert any("Paris" in c for c in contents)


def test_mutual_information_proxy() -> None:
    transform = InformationTheoreticSelector(token_budget=50)
    mi = transform._mutual_information(
        "Paris is the capital of France with the Eiffel Tower.",
        "Tell me about Paris landmarks.",
    )
    assert mi > 0.0
    mi_unrelated = transform._mutual_information(
        "Quantum computing uses qubits.",
        "Bake a chocolate cake.",
    )
    assert mi > mi_unrelated


def test_information_selector_priority() -> None:
    assert InformationTheoreticSelector.priority == 19
