"""Unit tests for RateDistortionCompressor.optimize() (IR-native path)."""
from __future__ import annotations

from lattice.core.context import TransformContext
from lattice.core.result import Ok
from lattice.core.transport import Message, Request, Role
from lattice.ir.primitives import PromptIRV2, SectionV2, SpanV2
from lattice.transforms.rate_distortion import RateDistortionCompressor


class TestRateDistortionIR:
    """Test the IR-native optimize() method of RateDistortionCompressor."""

    def _make_request(self, sections: list[SectionV2]) -> tuple[Request, TransformContext]:
        ir = PromptIRV2(sections=tuple(sections))
        # Build request messages that satisfy lossy_transform_allowed (long_form=True)
        long_content = " ".join([f"Sentence {i} is about various topics that matter a lot." for i in range(60)])
        request = Request(messages=[Message(role=Role.USER, content=long_content)])
        request.metadata["_lattice_ir_v2"] = ir
        ctx = TransformContext()
        ctx.session_state["_lattice_ir_v2"] = ir
        return request, ctx

    def test_noop_when_short(self) -> None:
        sec = SectionV2(type="context", spans=(SpanV2(span_id="s1", text="hello world"),))
        ir = PromptIRV2(sections=(sec,))
        request, ctx = self._make_request([sec])
        rd = RateDistortionCompressor()
        result = rd.optimize(ir, request, ctx)
        assert isinstance(result, Ok)
        new_ir = result.unwrap()
        assert new_ir.sections[0].spans[0].text == "hello world"

    def test_compress_long_natural_language(self) -> None:
        long_text = " ".join([f"Sentence {i} is about various topics that matter a lot." for i in range(50)])
        sec = SectionV2(type="context", spans=(SpanV2(span_id="s1", text=long_text),))
        ir = PromptIRV2(sections=(sec,))
        request, ctx = self._make_request([sec])
        rd = RateDistortionCompressor()
        result = rd.optimize(ir, request, ctx)
        assert isinstance(result, Ok)
        new_ir = result.unwrap()
        compressed = new_ir.sections[0].spans[0].text
        # Should be shorter or same length
        assert len(compressed) <= len(long_text)
        # Should not be empty
        assert len(compressed) > 0

    def test_noop_on_structured_data(self) -> None:
        json_text = '{"name": "Alice", "age": 30, "items": [1, 2, 3]}'
        sec = SectionV2(type="json", spans=(SpanV2(span_id="s1", text=json_text),))
        ir = PromptIRV2(sections=(sec,))
        request, ctx = self._make_request([sec])
        rd = RateDistortionCompressor()
        result = rd.optimize(ir, request, ctx)
        assert isinstance(result, Ok)
        new_ir = result.unwrap()
        assert new_ir.sections[0].spans[0].text == json_text

    def test_noop_on_protected_span(self) -> None:
        long_text = " ".join([f"Sentence {i} is about various topics that matter a lot." for i in range(50)])
        sec = SectionV2(type="context", spans=(SpanV2(span_id="s1", text=long_text, protected=True),))
        ir = PromptIRV2(sections=(sec,))
        request, ctx = self._make_request([sec])
        rd = RateDistortionCompressor()
        result = rd.optimize(ir, request, ctx)
        assert isinstance(result, Ok)
        new_ir = result.unwrap()
        assert new_ir.sections[0].spans[0].text == long_text

    def test_metrics_recorded(self) -> None:
        long_text = " ".join([f"Sentence {i} is about various topics that matter a lot." for i in range(50)])
        sec = SectionV2(type="context", spans=(SpanV2(span_id="s1", text=long_text),))
        ir = PromptIRV2(sections=(sec,))
        request, ctx = self._make_request([sec])
        rd = RateDistortionCompressor()
        result = rd.optimize(ir, request, ctx)
        assert isinstance(result, Ok)
        metrics = ctx.metrics.get("transforms", {}).get("rate_distortion", {})
        assert metrics.get("spans_compressed") == 1
        assert metrics.get("tokens_saved_estimate", 0) >= 0
