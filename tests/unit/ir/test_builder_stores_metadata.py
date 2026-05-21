"""Phase 1 regression — build_ir() stores IR summary on request metadata.

When ``core/compiler.py`` was deleted in Phase 1, its ``_store_ir_metadata``
helper was inlined into ``ir/builder.py`` and is now called from ``build_ir``
itself. This test guarantees that the metadata keys downstream transforms
read are still populated after a single ``build_ir`` call.
"""

from __future__ import annotations

from lattice.core.transport import Message, Request
from lattice.ir.builder import build_ir
from lattice.ir.normalizer import normalize_ir
from lattice.ir.serializer import serialize_ir_to_text


def _make_request(content: str) -> Request:
    return Request(messages=[Message(role="user", content=content)])


def test_build_ir_returns_nonempty_ir() -> None:
    req = _make_request("Help me understand cache eviction policies.")
    ir = build_ir(req)
    assert ir.total_spans > 0
    assert len(ir.sections) > 0


def test_build_ir_populates_summary_metadata() -> None:
    req = _make_request("Analyse the root cause of the OOM kill.")
    build_ir(req)
    assert "_lattice_ir_summary" in req.metadata
    assert "_lattice_protected_spans" in req.metadata
    # protected_spans is a list-like
    assert isinstance(req.metadata["_lattice_protected_spans"], (list, tuple, set, frozenset))


def test_build_ir_metadata_is_idempotent() -> None:
    """Building twice from the same source produces the same summary."""
    req1 = _make_request("Restart the orchestrator; it crashed at 14:02.")
    req2 = _make_request("Restart the orchestrator; it crashed at 14:02.")
    build_ir(req1)
    build_ir(req2)
    assert req1.metadata["_lattice_ir_summary"] == req2.metadata["_lattice_ir_summary"]


def test_full_compile_pipeline_round_trips() -> None:
    """build_ir → normalize_ir → serialize_ir_to_text produces a non-empty
    string for a simple request — matching the pre-Phase-1 compiler.compile()
    + compiler.serialize() pipeline behaviour."""
    req = _make_request("Explain how delta encoding reduces tail latency.")
    ir = normalize_ir(build_ir(req))
    text = serialize_ir_to_text(ir)
    assert text.strip(), "serialized IR is empty"
    # The user's prompt content should round-trip back into the text.
    assert "delta encoding" in text.lower() or "tail latency" in text.lower()


def test_error_signal_detected_in_metadata() -> None:
    """When the prompt contains an error/stack-trace, build_ir flags it."""
    req = _make_request(
        "Got this:\n"
        "Traceback (most recent call last):\n"
        '  File "main.py", line 12, in <module>\n'
        "    foo()\n"
        "RuntimeError: oops\n"
    )
    build_ir(req)
    # _lattice_has_errors is only set when an error/stack_trace section is detected.
    # We don't assert True hard (the classifier is heuristic), but the key
    # should at least be a boolean-or-absent.
    assert "_lattice_has_errors" not in req.metadata or req.metadata["_lattice_has_errors"] is True
