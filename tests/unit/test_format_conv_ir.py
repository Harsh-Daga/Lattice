"""Unit tests for FormatConverter.optimize() (IR-native path)."""
from __future__ import annotations

from lattice.core.context import TransformContext
from lattice.core.result import Ok
from lattice.core.transport import Request
from lattice.ir.primitives import PromptIRV2, SectionV2, SpanV2
from lattice.transforms.format_conv import FormatConverter


class TestFormatConverterIR:
    """Test the IR-native optimize() method of FormatConverter."""

    def _make_request(self, sections: list[SectionV2]) -> tuple[Request, TransformContext]:
        ir = PromptIRV2(sections=tuple(sections))
        request = Request(messages=[])
        request.metadata["_lattice_ir_v2"] = ir
        ctx = TransformContext()
        ctx.session_state["_lattice_ir_v2"] = ir
        return request, ctx

    def test_noop_when_no_structured_data(self) -> None:
        sec = SectionV2(type="context", spans=(SpanV2(span_id="s1", text="hello world"),))
        ir = PromptIRV2(sections=(sec,))
        request, ctx = self._make_request([sec])
        fc = FormatConverter()
        result = fc.optimize(ir, request, ctx)
        assert isinstance(result, Ok)
        new_ir = result.unwrap()
        # optimize always returns a new IR object even when unchanged
        assert new_ir.sections[0].spans[0].text == "hello world"

    def test_json_tabular_to_csv(self) -> None:
        json_text = '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]'
        sec = SectionV2(type="json", spans=(SpanV2(span_id="s1", text=json_text),))
        ir = PromptIRV2(sections=(sec,))
        request, ctx = self._make_request([sec])
        fc = FormatConverter()
        result = fc.optimize(ir, request, ctx)
        assert isinstance(result, Ok)
        new_ir = result.unwrap()
        converted = new_ir.sections[0].spans[0].text
        assert "age,name" in converted or "name,age" in converted
        assert "30,Alice" in converted or "Alice\t30" in converted

    def test_json_config_to_yaml(self) -> None:
        json_text = '{"server": {"host": "localhost", "port": 8080}}'
        sec = SectionV2(type="json", spans=(SpanV2(span_id="s1", text=json_text),))
        ir = PromptIRV2(sections=(sec,))
        request, ctx = self._make_request([sec])
        fc = FormatConverter()
        result = fc.optimize(ir, request, ctx)
        assert isinstance(result, Ok)
        new_ir = result.unwrap()
        converted = new_ir.sections[0].spans[0].text
        assert "server:" in converted
        assert "host:" in converted
        assert "port:" in converted

    def test_markdown_table_to_csv(self) -> None:
        md = "| Name | Age |\n|------|-----|\n| Alice | 30 |\n| Bob | 25 |"
        sec = SectionV2(type="table", spans=(SpanV2(span_id="s1", text=md),))
        ir = PromptIRV2(sections=(sec,))
        request, ctx = self._make_request([sec])
        fc = FormatConverter()
        result = fc.optimize(ir, request, ctx)
        assert isinstance(result, Ok)
        new_ir = result.unwrap()
        converted = new_ir.sections[0].spans[0].text
        assert "Name,Age" in converted or "Name\tAge" in converted
        assert "Alice,30" in converted or "Alice\t30" in converted

    def test_logs_compressed(self) -> None:
        logs = "\n".join(
            f"2024-01-0{i} INFO  request {i}" for i in range(1, 15)
        )
        sec = SectionV2(type="log", spans=(SpanV2(span_id="s1", text=logs),))
        ir = PromptIRV2(sections=(sec,))
        request, ctx = self._make_request([sec])
        fc = FormatConverter()
        result = fc.optimize(ir, request, ctx)
        assert isinstance(result, Ok)
        new_ir = result.unwrap()
        converted = new_ir.sections[0].spans[0].text
        # Should be shorter than original
        assert len(converted) < len(logs)
        # Should contain ellipsis for compressed lines
        assert "..." in converted

    def test_metrics_recorded_on_conversion(self) -> None:
        json_text = '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]'
        sec = SectionV2(type="json", spans=(SpanV2(span_id="s1", text=json_text),))
        ir = PromptIRV2(sections=(sec,))
        request, ctx = self._make_request([sec])
        fc = FormatConverter()
        result = fc.optimize(ir, request, ctx)
        assert isinstance(result, Ok)
        metrics = ctx.metrics.get("transforms", {}).get("format_conversion", {})
        assert metrics.get("spans_converted") == 1
        assert metrics.get("tokens_saved_estimate", 0) > 0
