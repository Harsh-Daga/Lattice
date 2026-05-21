"""Tests for PromptIRV2.canonical_fingerprint() (determinism + replay hardening)."""

from __future__ import annotations

from lattice.ir.primitives import PromptIRV2, SectionV2, SpanV2


class TestCanonicalFingerprint:
    """Test deterministic canonical fingerprints."""

    def test_same_ir_same_fingerprint(self) -> None:
        sec = SectionV2(
            type="context",
            spans=(
                SpanV2(span_id="s1", text="hello world", protected=True),
                SpanV2(span_id="s2", text="foo bar"),
            ),
        )
        ir1 = PromptIRV2(sections=(sec,))
        ir2 = PromptIRV2(sections=(sec,))
        assert ir1.canonical_fingerprint() == ir2.canonical_fingerprint()

    def test_different_text_different_fingerprint(self) -> None:
        sec1 = SectionV2(type="context", spans=(SpanV2(span_id="s1", text="hello"),))
        sec2 = SectionV2(type="context", spans=(SpanV2(span_id="s1", text="world"),))
        fp1 = PromptIRV2(sections=(sec1,)).canonical_fingerprint()
        fp2 = PromptIRV2(sections=(sec2,)).canonical_fingerprint()
        assert fp1 != fp2

    def test_stable_across_rebuild(self) -> None:
        ir = PromptIRV2(
            sections=(
                SectionV2(
                    type="json", spans=(SpanV2(span_id="a", text="data", compressible=True),)
                ),
                SectionV2(type="log", spans=(SpanV2(span_id="b", text="warn"),)),
            )
        )
        fp = ir.canonical_fingerprint()
        # Rebuild from scratch
        rebuilt = PromptIRV2(
            sections=(
                SectionV2(
                    type="json", spans=(SpanV2(span_id="a", text="data", compressible=True),)
                ),
                SectionV2(type="log", spans=(SpanV2(span_id="b", text="warn"),)),
            )
        )
        assert rebuilt.canonical_fingerprint() == fp

    def test_fingerprint_is_hex_64(self) -> None:
        sec = SectionV2(type="context", spans=(SpanV2(span_id="s1", text="x"),))
        fp = PromptIRV2(sections=(sec,)).canonical_fingerprint()
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

    def test_section_order_matters(self) -> None:
        s1 = SectionV2(type="a", spans=(SpanV2(span_id="s1", text="x"),))
        s2 = SectionV2(type="b", spans=(SpanV2(span_id="s2", text="y"),))
        fp_ab = PromptIRV2(sections=(s1, s2)).canonical_fingerprint()
        fp_ba = PromptIRV2(sections=(s2, s1)).canonical_fingerprint()
        assert fp_ab != fp_ba
