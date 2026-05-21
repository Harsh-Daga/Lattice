"""Tests for core/primitives.py — Unified type system."""

from __future__ import annotations

import dataclasses

from lattice.ir.primitives import (
    Candidate,
    CandidateGraph,
    ExecutionPlan,
    PromptIRV2,
    SectionV2,
    SpanV2,
    freeze_dict,
    prompt_ir_v2_from_legacy,
    thaw_dict,
)


class TestSpanV2:
    def test_immutability(self) -> None:
        sp = SpanV2(span_id="s1", text="hello")
        try:
            sp.text = "world"
            assert False, "Should have raised FrozenInstanceError"
        except dataclasses.FrozenInstanceError as e:
            assert "cannot" in str(e).lower() or "frozen" in str(e).lower()

    def test_with_text_returns_new(self) -> None:
        sp = SpanV2(span_id="s1", text="hello")
        sp2 = sp.with_text("world")
        assert sp.text == "hello"
        assert sp2.text == "world"
        assert sp2.span_id == "s1"

    def test_with_structure(self) -> None:
        sp = SpanV2(span_id="s1", text="hello")
        sp2 = sp.with_structure(json_shape=("a", "b"))
        assert list(dict(sp2.structure)["json_shape"]) == ["a", "b"]


class TestPromptIRV2:
    def test_sections_immutable(self) -> None:
        sec = SectionV2(type="json", spans=(SpanV2(span_id="s1", text="{}"),))
        ir = PromptIRV2(sections=(sec,))
        assert ir.total_spans == 1
        assert ir.section_types() == ["json"]

    def test_add_section(self) -> None:
        ir = PromptIRV2()
        sec = SectionV2(type="task", spans=(SpanV2(span_id="s1", text="hi"),))
        ir2 = ir.add_section(sec)
        assert ir.total_spans == 0
        assert ir2.total_spans == 1
        assert ir2.section_types() == ["task"]

    def test_add_metadata(self) -> None:
        ir = PromptIRV2()
        ir2 = ir.add_metadata(task="debug")
        assert thaw_dict(ir.metadata).get("task") is None
        assert thaw_dict(ir2.metadata).get("task") == "debug"


class TestCandidate:
    def test_immutable(self) -> None:
        ir = PromptIRV2()
        c = Candidate(ir=ir)
        try:
            c.applied = ("foo",)
            assert False, "Should have raised FrozenInstanceError"
        except dataclasses.FrozenInstanceError:
            pass

    def test_apply_returns_new(self) -> None:
        ir = PromptIRV2()
        c = Candidate(ir=ir)
        c2 = c.apply("structure_optimizer", ir)
        assert c.applied == ()
        assert c2.applied == ("structure_optimizer",)

    def test_score_computation(self) -> None:
        ir = PromptIRV2()
        c = Candidate(ir=ir).with_metric("tokens_before", 100).with_metric("tokens_after", 50)
        score = c.score()
        assert hasattr(score, "expected_utility")
        assert score.cost_reduction == 0.5
        assert score.composite >= score.quality  # composite may be clamped at 1.0


class TestCandidateGraph:
    def test_expand_and_top_k(self) -> None:
        ir = PromptIRV2()
        c1 = Candidate(ir=ir).with_metric("tokens_before", 100).with_metric("tokens_after", 90)
        c2 = Candidate(ir=ir).with_metric("tokens_before", 100).with_metric("tokens_after", 50)
        graph = CandidateGraph(beam=(c1,))
        graph2 = graph.expand(new_beam=(c1, c2)).top_k(1)
        assert graph2.best is not None
        assert graph2.best.score().cost_reduction > graph.best.score().cost_reduction


class TestExecutionPlan:
    def test_immutability(self) -> None:
        plan = ExecutionPlan(transforms=("content_profiler",), quality_floor=0.9)
        assert plan.transforms == ("content_profiler",)
        assert plan.quality_floor == 0.9


class TestLegacyConversion:
    def test_from_legacy(self) -> None:
        class FakeSpan:
            span_id = "s1"
            text = "hello"
            role = "data"
            entities = []
            numbers = ["42"]
            keys = []
            structure = {"constant_fields": ["name"]}
            protected = True
            compressible = False
            compression_modes_allowed = []
            metadata = {}

        class FakeSection:
            type = "task"
            spans = [FakeSpan()]
            metadata = {}

        class FakeIR:
            sections = [FakeSection()]
            metadata = {"has_errors": True}

        v2 = prompt_ir_v2_from_legacy(FakeIR())
        assert v2.total_spans == 1
        assert v2.protected_spans == 1
        assert v2.section_types() == ["task"]
        meta = thaw_dict(v2.metadata)
        assert meta.get("has_errors") is True


class TestFreezeThaw:
    def test_roundtrip(self) -> None:
        d = {"a": 1, "b": [2, 3]}
        frozen = freeze_dict(d)
        thawed = thaw_dict(frozen)
        assert thawed["a"] == 1
        assert list(thawed["b"]) == [2, 3]
