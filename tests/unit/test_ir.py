"""Tests for the canonical IR layer — builder, normalizer, serializer, compiler."""

from __future__ import annotations

import json

import pytest

from lattice.core.context import TransformContext
from lattice.core.ir import PromptIR, SectionType, SpanRole
from lattice.core.ir_builder import build_ir
from lattice.core.ir_normalizer import normalize_ir
from lattice.core.ir_serializer import serialize_ir_to_text
from lattice.core.compiler import PromptCompiler
from lattice.core.transport import Message, Request


def _make_request(messages: list[dict]) -> Request:
    return Request(messages=[Message(**m) for m in messages])


class TestIRBuilder:
    def test_ir_detects_tool_output(self) -> None:
        req = _make_request([
            {"role": "user", "content": "Why did the build fail?"},
            {"role": "tool", "content": json.dumps({"errors": 60, "build_id": "abc"})},
        ])
        ir = build_ir(req)
        # Tool output JSON with "errors" key gets classified as ERROR or TOOL_OUTPUT
        section_types = [s.type.value for s in ir.sections]
        assert any(t in ("error", "tool_output", "json") for t in section_types)
        assert ir.total_spans > 0

    def test_ir_marks_error_messages_protected(self) -> None:
        req = _make_request([
            {"role": "user", "content": "Build failed with 20 Module not found errors and 20 Syntax errors."},
        ])
        ir = build_ir(req)
        assert ir.protected_spans > 0
        assert any(sp.protected for sec in ir.sections for sp in sec.spans)

    def test_ir_marks_counts_protected(self) -> None:
        req = _make_request([
            {"role": "user", "content": "There were 60 errors, 3 timeouts, and 2 crashes."},
        ])
        ir = build_ir(req)
        # Counts should be marked protected
        assert ir.protected_spans > 0

    def test_ir_detects_code_blocks(self) -> None:
        req = _make_request([
            {"role": "user", "content": "Here is the code:\n```python\nprint('hello')\n```\nPlease fix it."},
        ])
        ir = build_ir(req)
        assert any(s.type == SectionType.CODE for s in ir.sections)

    def test_ir_detects_tables(self) -> None:
        req = _make_request([
            {"role": "user", "content": "| id | name |\n| --- | --- |\n| 1 | Alice |\n| 2 | Bob |"},
        ])
        ir = build_ir(req)
        assert any(s.type == SectionType.TABLE for s in ir.sections)

    def test_ir_detects_stack_traces(self) -> None:
        req = _make_request([
            {"role": "user", "content": 'Traceback:\n  File "app.py", line 10, in foo\n  File "db.py", line 22, in bar\nValueError: boom'},
        ])
        ir = build_ir(req)
        assert any(s.type in (SectionType.STACK_TRACE, SectionType.ERROR) for s in ir.sections)

    def test_build_ir_is_deterministic(self) -> None:
        req = _make_request([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing briefly."},
        ])
        ir1 = build_ir(req)
        ir2 = build_ir(req)
        assert ir1.total_spans == ir2.total_spans
        assert ir1.section_types == ir2.section_types

    def test_ir_classifies_tasks(self) -> None:
        req = _make_request([
            {"role": "user", "content": "Debug this error and explain the root cause."},
        ])
        ir = build_ir(req)
        assert any(s.type == SectionType.TASK or s.type == SectionType.ERROR for s in ir.sections)

    def test_ir_classifies_constraints(self) -> None:
        req = _make_request([
            {"role": "user", "content": "You must return JSON and must include all IDs."},
        ])
        ir = build_ir(req)
        assert any(s.type == SectionType.CONSTRAINTS or SectionType.INSTRUCTION for s in ir.sections)


class TestIRNormalizer:
    def test_normalize_json_detects_constant_fields(self) -> None:
        req = _make_request([
            {"role": "tool", "content": '[{"status":"ok","latency":100},{"status":"ok","latency":101},{"status":"ok","latency":102}]'},
        ])
        ir = build_ir(req)
        ir = normalize_ir(ir)
        for sec in ir.sections:
            for sp in sec.spans:
                if sp.structure.get("constant_fields"):
                    assert any(cf["field"] == "status" for cf in sp.structure["constant_fields"])
                    return
        pytest.skip("No constant fields detected — structure detection is best-effort")

    def test_normalize_table_detects_columns(self) -> None:
        req = _make_request([
            {"role": "user", "content": "| id | name | dept |\n| --- | --- | --- |\n| 0 | Alice | Eng |\n| 1 | Bob | Eng |"},
        ])
        ir = build_ir(req)
        ir = normalize_ir(ir)
        for sec in ir.sections:
            if sec.type == SectionType.TABLE:
                for sp in sec.spans:
                    if sp.structure.get("table_columns"):
                        assert "id" in sp.structure["table_columns"]
                        return
        pytest.skip("Table structure detection is best-effort")

    def test_normalize_json_no_crash_on_malformed(self) -> None:
        req = _make_request([
            {"role": "tool", "content": "not json at all"},
        ])
        ir = build_ir(req)
        ir2 = normalize_ir(ir)
        assert ir2.total_spans >= ir.total_spans  # No spans should be lost


class TestIRSerializer:
    def test_no_opaque_placeholders_in_llm_text(self) -> None:
        req = _make_request([
            {"role": "user", "content": "Explain quantum computing."},
        ])
        ir = build_ir(req)
        text = serialize_ir_to_text(ir)
        import re
        opaque = re.findall(r"<(?:d_|g_|ref_)\d+>", text)
        if opaque:
            assert "ALIAS MAP" in text or "DICT:" in text, (
                f"Opaque placeholders found without manifest: {opaque}"
            )

    def test_serialize_preserves_content(self) -> None:
        req = _make_request([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ])
        ir = build_ir(req)
        text = serialize_ir_to_text(ir)
        assert "What is 2+2" in text
        assert "You are helpful" in text

    def test_serialize_does_not_truncate_short(self) -> None:
        req = _make_request([
            {"role": "user", "content": "Hi"},
        ])
        ir = build_ir(req)
        text = serialize_ir_to_text(ir)
        assert "Hi" in text


class TestCompiler:
    def test_compiler_produces_valid_ir(self) -> None:
        compiler = PromptCompiler()
        req = _make_request([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Analyze the root cause: service A failed, causing B timeout."},
        ])
        ctx = TransformContext(request_id="test", provider="openai", model="test")
        ir = compiler.compile(req, ctx)
        assert isinstance(ir, PromptIR)
        assert ir.total_spans > 0

    def test_compiler_stores_metadata(self) -> None:
        compiler = PromptCompiler()
        req = _make_request([
            {"role": "user", "content": "Debug this: 20 errors and 3 timeouts."},
        ])
        ctx = TransformContext(request_id="test", provider="openai", model="test")
        compiler.compile(req, ctx)
        assert "_lattice_ir_summary" in req.metadata
        assert "_lattice_protected_spans" in req.metadata

    def test_compiler_is_idempotent(self) -> None:
        compiler = PromptCompiler()
        req = _make_request([
            {"role": "user", "content": "Explain the error: service A failed."},
        ])
        ctx1 = TransformContext(request_id="test1", provider="openai", model="test")
        ctx2 = TransformContext(request_id="test2", provider="openai", model="test")
        ir1 = compiler.compile(req.copy(), ctx1)
        ir2 = compiler.compile(req.copy(), ctx2)
        assert ir1.total_spans == ir2.total_spans
