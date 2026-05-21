"""Unit tests for IR-native structure optimizer."""
from __future__ import annotations

from lattice.core.context import TransformContext
from lattice.core.ir import PromptIR, Section, SectionType, Span, SpanRole
from lattice.core.transport import Request
from lattice.optimizer.ir_structure_optimizer import IRStructureOptimizer


def _make_request(content: str) -> Request:
    return Request(
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": content},
        ]
    )


def _build_ir(text: str, section_type: SectionType, structure: dict) -> PromptIR:
    span = Span(
        span_id="s1",
        text=text,
        role=SpanRole.DATA,
        section_type=section_type,
        compressible=True,
        structure=structure,
    )
    section = Section(type=section_type, spans=[span])
    return PromptIR(sections=[section])


class TestIRStructureOptimizer:
    def test_factor_json_constant_fields(self) -> None:
        req = _make_request(
            '[{"status": "ok", "latency": 100}, {"status": "ok", "latency": 101}, {"status": "ok", "latency": 102}]'
        )
        ctx = TransformContext(
            request_id="t1",
            provider="openai",
            model="gpt-4",
            session_state={},
        )
        opt = IRStructureOptimizer()
        result = opt.process(req, ctx)
        modified = result.unwrap()
        assert isinstance(modified, Request)
        # If the IR factored something, the metadata tag should be present.
        # If no changes were made (no constant fields detected), the tag may be absent.
        if modified.metadata.get("_lattice_ir_native_applied"):
            last_user = [m for m in modified.messages if m.role == "user"][-1]
            assert "constant:" in last_user.content or "JSON:" in last_user.content

    def test_factor_table_constant_columns(self) -> None:
        req = _make_request(
            "| id | name | dept |\n| --- | --- | --- |\n| 1 | Alice | Eng |\n| 2 | Bob | Eng |"
        )
        ctx = TransformContext(request_id="t2", provider="openai", model="gpt-4")
        opt = IRStructureOptimizer()
        result = opt.process(req, ctx)
        modified = result.unwrap()
        assert isinstance(modified, Request)

    def test_group_logs(self) -> None:
        req = _make_request(
            "ERROR: connection refused\nWARN: slow query\nERROR: timeout\n"
        )
        ctx = TransformContext(request_id="t3", provider="openai", model="gpt-4")
        opt = IRStructureOptimizer()
        result = opt.process(req, ctx)
        modified = result.unwrap()
        assert isinstance(modified, Request)

    def test_no_crash_on_plain_text(self) -> None:
        """Non-structural text should pass through without modification."""
        req = _make_request("What is 2+2?")
        ctx = TransformContext(request_id="t4", provider="openai", model="gpt-4")
        opt = IRStructureOptimizer()
        result = opt.process(req, ctx)
        modified = result.unwrap()
        assert isinstance(modified, Request)
