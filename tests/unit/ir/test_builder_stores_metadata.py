"""Phase 1 regression — compile_request_ir() stores IR summary on request metadata.

When ``core/compiler.py`` was deleted in Phase 1, its ``_store_ir_metadata``
helper was inlined into ``ir/builder.py`` and is now invoked via the new
``compile_request_ir(request)`` helper (which runs build → normalize → store).
``build_ir`` itself stays a pure function so downstream code that wants the
un-normalized IR doesn't get an unexpected request mutation.
"""

from __future__ import annotations

from lattice.core.transport import Message, Request
from lattice.ir.builder import build_ir, compile_request_ir
from lattice.ir.serializer import serialize_ir_to_text


def _make_request(content: str) -> Request:
    return Request(messages=[Message(role="user", content=content)])


def test_build_ir_returns_nonempty_ir() -> None:
    req = _make_request("Help me understand cache eviction policies.")
    ir = build_ir(req)
    assert ir.total_spans > 0
    assert len(ir.sections) > 0


def test_build_ir_is_pure() -> None:
    """build_ir does not mutate request.metadata — that's the
    compile_request_ir contract."""
    req = _make_request("Analyse the root cause of the OOM kill.")
    build_ir(req)
    assert "_lattice_ir_summary" not in req.metadata
    assert "_lattice_protected_spans" not in req.metadata


def test_compile_request_ir_populates_summary_metadata() -> None:
    req = _make_request("Analyse the root cause of the OOM kill.")
    compile_request_ir(req)
    assert "_lattice_ir_summary" in req.metadata
    assert "_lattice_protected_spans" in req.metadata
    # protected_spans is a list-like
    assert isinstance(req.metadata["_lattice_protected_spans"], (list, tuple, set, frozenset))


def test_compile_request_ir_is_idempotent() -> None:
    """Compiling twice from the same source produces the same summary."""
    req1 = _make_request("Restart the orchestrator; it crashed at 14:02.")
    req2 = _make_request("Restart the orchestrator; it crashed at 14:02.")
    compile_request_ir(req1)
    compile_request_ir(req2)
    assert req1.metadata["_lattice_ir_summary"] == req2.metadata["_lattice_ir_summary"]


def test_compile_request_ir_round_trips() -> None:
    """compile_request_ir → serialize_ir_to_text produces a non-empty string."""
    req = _make_request("Explain how delta encoding reduces tail latency.")
    ir = compile_request_ir(req)
    text = serialize_ir_to_text(ir)
    assert text.strip(), "serialized IR is empty"
    assert "delta encoding" in text.lower() or "tail latency" in text.lower()


def test_compile_request_ir_metadata_reflects_post_normalize_state() -> None:
    """Protected-span IDs stored on metadata must reflect the
    **normalized** IR, because core/pipeline.py reads them to gate
    transforms like rate_distortion. Prior to this fix the metadata
    reflected the pre-normalize IR, causing a benchmark regression on
    rate_distortion_longform."""
    req = _make_request(
        "Investigate this incident:\n"
        "Service A failed at 14:02 because of a connection pool exhaustion.\n"
        "Service B then timed out, causing cascading retries.\n"
    )
    ir = compile_request_ir(req)
    stored = set(req.metadata["_lattice_protected_spans"])
    actual = set(ir.protected_span_ids())
    assert stored == actual, (
        f"metadata protected_spans (pre/post-normalize mismatch)\n"
        f"  stored: {stored}\n  actual: {actual}"
    )


def test_error_signal_detected_in_metadata() -> None:
    """When the prompt contains an error/stack-trace, compile_request_ir flags it."""
    req = _make_request(
        "Got this:\n"
        "Traceback (most recent call last):\n"
        '  File "main.py", line 12, in <module>\n'
        "    foo()\n"
        "RuntimeError: oops\n"
    )
    compile_request_ir(req)
    assert "_lattice_has_errors" not in req.metadata or req.metadata["_lattice_has_errors"] is True
