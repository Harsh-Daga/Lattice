"""Replay fingerprint stability (Phase 12)."""

from __future__ import annotations

from lattice.ir.primitives import PromptIRV2, SectionV2, SpanV2
from lattice.planner.task_classifier import TaskClass
from lattice.planner.unified_planner import SemanticProfile, UnifiedPlanner
from lattice.transport.types import Message, Request


def test_canonical_fingerprint_stable_for_same_request() -> None:
    req = Request(messages=[Message(role="user", content="hello replay")], model="m")
    profile = SemanticProfile(task_class=TaskClass.SIMPLE, provider="generic", model="m")
    planner = UnifiedPlanner()
    plan_a = planner.plan(req, profile)
    plan_b = planner.plan(req, profile)
    sec = SectionV2(
        type="context",
        spans=(SpanV2(span_id="s1", text="hello replay"),),
    )
    ir_a = PromptIRV2(sections=(sec,))
    ir_b = PromptIRV2(sections=(sec,))
    assert ir_a.canonical_fingerprint() == ir_b.canonical_fingerprint()
    assert plan_a.transforms == plan_b.transforms
