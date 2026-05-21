"""Unit tests for MessageDeduplicator.optimize() (IR-native path)."""
from __future__ import annotations

from lattice.core.context import TransformContext
from lattice.core.primitives import PromptIRV2, SectionV2, SpanV2
from lattice.core.result import Ok
from lattice.core.transport import Request
from lattice.transforms.message_dedup import MessageDeduplicator


class TestMessageDeduplicatorIR:
    """Test the IR-native optimize() method of MessageDeduplicator."""

    def _make_request(self, sections: list[SectionV2]) -> tuple[Request, TransformContext]:
        ir = PromptIRV2(sections=tuple(sections))
        request = Request(messages=[])
        # Store IR in session state so optimize() can access it if needed
        request.metadata["_lattice_ir_v2"] = ir
        ctx = TransformContext()
        ctx.session_state["_lattice_ir_v2"] = ir
        return request, ctx

    def test_no_dedup_when_single_section(self) -> None:
        sec = SectionV2(type="context", spans=(SpanV2(span_id="s1", text="hello world"),))
        ir = PromptIRV2(sections=(sec,))
        request, ctx = self._make_request([sec])
        dedup = MessageDeduplicator()
        result = dedup.optimize(ir, request, ctx)
        assert isinstance(result, Ok)
        new_ir = result.unwrap()
        assert new_ir is ir  # identical reference since no change

    def test_exact_duplicate_sections_removed(self) -> None:
        sec1 = SectionV2(type="context", spans=(SpanV2(span_id="s1", text="the quick brown fox jumps over the lazy dog"),))
        sec2 = SectionV2(type="context", spans=(SpanV2(span_id="s2", text="the quick brown fox jumps over the lazy dog"),))
        ir = PromptIRV2(sections=(sec1, sec2))
        request, ctx = self._make_request([sec1, sec2])
        dedup = MessageDeduplicator(preserve_last_n=0)
        result = dedup.optimize(ir, request, ctx)
        assert isinstance(result, Ok)
        new_ir = result.unwrap()
        assert len(new_ir.sections) == 1
        assert new_ir.sections[0].spans[0].text == "the quick brown fox jumps over the lazy dog"

    def test_near_duplicate_sections_removed(self) -> None:
        text1 = "the quick brown fox jumps over the lazy dog"
        text2 = "the quick brown fox jumps over the lazy doggy"
        sec1 = SectionV2(type="context", spans=(SpanV2(span_id="s1", text=text1),))
        sec2 = SectionV2(type="context", spans=(SpanV2(span_id="s2", text=text2),))
        ir = PromptIRV2(sections=(sec1, sec2))
        request, ctx = self._make_request([sec1, sec2])
        dedup = MessageDeduplicator(near_duplicate_threshold=0.85, preserve_last_n=0)
        result = dedup.optimize(ir, request, ctx)
        assert isinstance(result, Ok)
        new_ir = result.unwrap()
        # Near-duplicate with 0.85 threshold should remove sec2
        assert len(new_ir.sections) == 1

    def test_protected_spans_preserved(self) -> None:
        long_text = "the quick brown fox jumps over the lazy dog repeatedly"
        sec1 = SectionV2(
            type="context",
            spans=(SpanV2(span_id="s1", text=long_text),),
        )
        sec2 = SectionV2(
            type="context",
            spans=(SpanV2(span_id="s2", text=long_text, protected=True),),
        )
        ir = PromptIRV2(sections=(sec1, sec2))
        request, ctx = self._make_request([sec1, sec2])
        dedup = MessageDeduplicator(preserve_last_n=0)
        result = dedup.optimize(ir, request, ctx)
        assert isinstance(result, Ok)
        new_ir = result.unwrap()
        # sec2 has a protected span so it should be preserved even though duplicate
        assert len(new_ir.sections) == 2

    def test_last_n_sections_preserved(self) -> None:
        sec1 = SectionV2(type="context", spans=(SpanV2(span_id="s1", text="A"),))
        sec2 = SectionV2(type="context", spans=(SpanV2(span_id="s2", text="B"),))
        sec3 = SectionV2(type="context", spans=(SpanV2(span_id="s3", text="C"),))
        sec4 = SectionV2(type="context", spans=(SpanV2(span_id="s4", text="A"),))
        ir = PromptIRV2(sections=(sec1, sec2, sec3, sec4))
        request, ctx = self._make_request([sec1, sec2, sec3, sec4])
        dedup = MessageDeduplicator(preserve_last_n=2)
        result = dedup.optimize(ir, request, ctx)
        assert isinstance(result, Ok)
        new_ir = result.unwrap()
        # last 2 sections (sec3, sec4) are always preserved even though sec4 is dup of sec1
        assert len(new_ir.sections) == 4
        texts = ["\n".join(sp.text for sp in s.spans) for s in new_ir.sections]
        assert texts == ["A", "B", "C", "A"]

    def test_short_sections_skipped(self) -> None:
        sec1 = SectionV2(type="context", spans=(SpanV2(span_id="s1", text="hi"),))
        sec2 = SectionV2(type="context", spans=(SpanV2(span_id="s2", text="hi"),))
        ir = PromptIRV2(sections=(sec1, sec2))
        request, ctx = self._make_request([sec1, sec2])
        dedup = MessageDeduplicator(min_message_length=10, preserve_last_n=0)
        result = dedup.optimize(ir, request, ctx)
        assert isinstance(result, Ok)
        new_ir = result.unwrap()
        # "hi" is only 2 chars, below min_message_length, so both kept
        assert len(new_ir.sections) == 2

    def test_metrics_recorded_on_dedup(self) -> None:
        sec1 = SectionV2(type="context", spans=(SpanV2(span_id="s1", text="the quick brown fox jumps over the lazy dog"),))
        sec2 = SectionV2(type="context", spans=(SpanV2(span_id="s2", text="the quick brown fox jumps over the lazy dog"),))
        ir = PromptIRV2(sections=(sec1, sec2))
        request, ctx = self._make_request([sec1, sec2])
        dedup = MessageDeduplicator(preserve_last_n=0)
        result = dedup.optimize(ir, request, ctx)
        assert isinstance(result, Ok)
        metrics = ctx.metrics.get("transforms", {}).get("message_dedup", {})
        assert metrics.get("removed_count") == 1
        assert metrics.get("original_count") == 2
