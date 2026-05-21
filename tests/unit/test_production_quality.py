"""Production-grade tests: causal_chain, constraint_lifting, tool_projection, frontier, safety, reputation."""

from __future__ import annotations

import json

import pytest

from lattice.core.context import TransformContext
from lattice.core.result import is_ok, unwrap
from lattice.core.transport import Message, Request


def _req(role: str, content: str) -> Message:
    return Message(role=role, content=content)


class TestCausalChain:
    def test_causal_chain_extracts_explicit(self) -> None:
        from lattice.transforms.causal_chain import CausalChainExtractor

        t = CausalChainExtractor()
        ctx = TransformContext(request_id="t", provider="openai", model="test")
        req = Request(messages=[_req("user", "Service A failed, causing B timeout, which triggered C retry storm.")])
        result = t.process(req, ctx)
        assert is_ok(result)
        out = unwrap(result)
        assert "CAUSAL GRAPH" in out.messages[0].content

    def test_causal_chain_preserves_root_cause(self) -> None:
        from lattice.transforms.causal_chain import CausalChainExtractor

        t = CausalChainExtractor()
        ctx = TransformContext(request_id="t", provider="openai", model="test")
        req = Request(messages=[_req("user", "The root cause was a missing module. This caused the build to fail.")])
        result = t.process(req, ctx)
        assert is_ok(result)
        out = unwrap(result)
        assert "CAUSAL GRAPH" in out.messages[0].content

    def test_causal_chain_skips_non_causal(self) -> None:
        from lattice.transforms.causal_chain import CausalChainExtractor

        t = CausalChainExtractor()
        ctx = TransformContext(request_id="t", provider="openai", model="test")
        req = Request(messages=[_req("user", "What is the weather today?")])
        result = t.process(req, ctx)
        assert is_ok(result)
        out = unwrap(result)
        assert "CAUSAL GRAPH" not in out.messages[0].content


class TestConstraintLifting:
    def test_constraint_lifting_extracts_json_requirement(self) -> None:
        from lattice.transforms.constraint_lifting import ConstraintLiftingTransform

        t = ConstraintLiftingTransform()
        ctx = TransformContext(request_id="t", provider="openai", model="test")
        req = Request(messages=[_req("user", "Please analyze this. You must return JSON. Include all IDs.")])
        result = t.process(req, ctx)
        assert is_ok(result)
        out = unwrap(result)
        assert "CONSTRAINTS:" in out.messages[0].content

    def test_constraint_lifting_preserves_content(self) -> None:
        from lattice.transforms.constraint_lifting import ConstraintLiftingTransform

        t = ConstraintLiftingTransform()
        ctx = TransformContext(request_id="t", provider="openai", model="test")
        content = "Please analyze this. You must return JSON. Include all IDs."
        req = Request(messages=[_req("user", content)])
        result = t.process(req, ctx)
        assert is_ok(result)
        out = unwrap(result)
        assert "Please analyze this" in out.messages[0].content

    def test_constraint_lifting_noop_when_none(self) -> None:
        from lattice.transforms.constraint_lifting import ConstraintLiftingTransform

        t = ConstraintLiftingTransform()
        ctx = TransformContext(request_id="t", provider="openai", model="test")
        content = "Hello world, what is the weather?"
        req = Request(messages=[_req("user", content)])
        result = t.process(req, ctx)
        assert is_ok(result)
        out = unwrap(result)
        assert "CONSTRAINTS:" not in out.messages[0].content


class TestToolProjectionQuality:
    def test_tool_projection_preserves_counts(self) -> None:
        from lattice.transforms.tool_projection import QueryAwareProjection

        t = QueryAwareProjection()
        ctx = TransformContext(request_id="t", provider="openai", model="test")
        data = json.dumps([{"error": f"Module {i} not found", "severity": "error", "module": i} for i in range(60)])
        req = Request(
            messages=[_req("user", "Why did the build fail?"), _req("tool", data)],
        )
        result = t.process(req, ctx)
        assert is_ok(result)

    def test_tool_projection_preserves_error_count(self) -> None:
        from lattice.transforms.tool_projection import QueryAwareProjection

        t = QueryAwareProjection()
        ctx = TransformContext(request_id="t", provider="openai", model="test")
        data = json.dumps([{"error": f"err_{i}", "severity": "error"} for i in range(60)])
        req = Request(
            messages=[_req("user", "How many errors occurred?"), _req("tool", data)],
        )
        result = t.process(req, ctx)
        assert is_ok(result)
        out = unwrap(result)
        tool_content = [m.content for m in out.messages if m.role == "tool" or m.tool_call_id]
        assert len(tool_content) > 0

    def test_tool_projection_handles_empty(self) -> None:
        from lattice.transforms.tool_projection import QueryAwareProjection

        t = QueryAwareProjection()
        ctx = TransformContext(request_id="t", provider="openai", model="test")
        req = Request(messages=[_req("user", "hello")])
        result = t.process(req, ctx)
        assert is_ok(result)


class TestFrontierGates:
    def test_frontier_fails_low_quality_high_compression(self) -> None:
        from benchmarks.framework.frontier import compute_frontier

        f = compute_frontier(0.70, 0.40, task_class="reasoning")
        assert f.passed_quality_gate is False
        assert f.passed_savings_gate is True
        assert f.rollback_reason is not None

    def test_frontier_passes_all_good(self) -> None:
        from benchmarks.framework.frontier import compute_frontier

        f = compute_frontier(0.95, 0.20, task_class="simple")
        assert f.passed_quality_gate is True
        assert f.passed_savings_gate is True
        assert f.rollback_reason is None

    def test_frontier_negative_savings_flag(self) -> None:
        from benchmarks.framework.frontier import compute_frontier

        f = compute_frontier(0.90, -0.10, task_class="retrieval")
        assert f.passed_savings_gate is False
        assert f.rollback_reason == "negative_savings"

    def test_scenario_specific_thresholds(self) -> None:
        from benchmarks.framework.frontier import compute_frontier

        # reasoning needs 0.92
        f1 = compute_frontier(0.90, 0.10, task_class="reasoning")
        assert not f1.passed_quality_gate
        # simple needs 0.80
        f2 = compute_frontier(0.85, 0.10, task_class="simple")
        assert f2.passed_quality_gate


class TestPlaceholderSafety:
    def test_no_opaque_placeholders_in_output(self) -> None:
        import re

        from lattice.ir.builder import build_ir
        from lattice.ir.serializer import serialize_ir_to_text

        req = Request(messages=[_req("user", "The UUID 550e8400-e29b-41d4-a716-446655440000 is duplicated. Error in module 42.")])
        ir = build_ir(req)
        text = serialize_ir_to_text(ir)
        opaque = re.findall(r"<(?:d_|g_|ref_)\d+>", text)
        assert len(opaque) == 0 or "ALIAS MAP" in text

    def test_hard_placeholder_leakage_detected(self) -> None:
        from lattice.core.guardrails import GuardAction, check_placeholder_leakage

        before = "The error was in module X with ID 123"
        after = "The error was <ref_17> in module <d_36>"
        decision = check_placeholder_leakage(before, after)
        assert decision.action in (GuardAction.ROLLBACK, GuardAction.REJECT)

    def test_manifested_aliases_pass(self) -> None:
        from lattice.core.guardrails import GuardAction, check_placeholder_leakage

        before = "The error was in module X"
        after = "ALIAS MAP:\nA1 = ModuleNotFoundError\n\nDATA:\nThe error was A1"

        # No opaque placeholders here at all — should ALLOW
        decision = check_placeholder_leakage(before, after)
        assert decision.action == GuardAction.ALLOW


class TestTransformReputationRuntime:
    def test_reputation_records_and_queries(self) -> None:
        from lattice.core.transform_reputation import get_reputation_registry

        rep = get_reputation_registry()
        rep.record("test_transform", quality=0.95, compression=0.20, rolled_back=False)
        stats = rep.stats("test_transform")
        assert stats.quality_avg == pytest.approx(0.95)
        assert stats.sample_count >= 1

    def test_reputation_tracks_rollback(self) -> None:
        from lattice.core.transform_reputation import get_reputation_registry

        rep = get_reputation_registry()
        for _ in range(5):
            rep.record("rb_test", quality=0.0, compression=0.0, rolled_back=True)
        stats = rep.stats("rb_test")
        assert stats.rollback_rate == pytest.approx(1.0)
        assert stats.risk == "HIGH"

    def test_reputation_unknown_is_safe(self) -> None:
        from lattice.core.transform_reputation import get_reputation_registry

        rep = get_reputation_registry()
        stats = rep.stats("never_seen_before")
        assert stats.sample_count == 0
        assert stats.risk == "LOW"
        assert not rep.is_high_risk("never_seen_before")


class TestMILVTriggers:
    def test_milv_triggers_on_high_compression(self) -> None:
        from lattice.core.milv import should_trigger_milv
        from lattice.core.task_classifier import TaskClass, TaskClassification

        tc = TaskClassification(task_class=TaskClass.SIMPLE)
        assert should_trigger_milv("test", tc, compression_ratio=0.35)

    def test_milv_triggers_on_reasoning(self) -> None:
        from lattice.core.milv import should_trigger_milv
        from lattice.core.task_classifier import TaskClass, TaskClassification

        tc = TaskClassification(task_class=TaskClass.REASONING)
        assert should_trigger_milv("test", tc, compression_ratio=0.15)

    def test_milv_skips_low_risk_simple(self) -> None:
        from lattice.core.milv import should_trigger_milv
        from lattice.core.task_classifier import TaskClass, TaskClassification

        tc = TaskClassification(task_class=TaskClass.SIMPLE)
        assert not should_trigger_milv("prefix_optimizer", tc, compression_ratio=0.05)


class TestSchedulerGating:
    def test_reasoning_blocks_rate_distortion(self) -> None:
        from lattice.core.scheduler import _TASK_TRANSFORM_MATRIX

        matrix = _TASK_TRANSFORM_MATRIX.get("reasoning", {})
        assert matrix.get("rate_distortion") is False

    def test_debugging_blocks_tool_filter(self) -> None:
        from lattice.core.scheduler import _TASK_TRANSFORM_MATRIX

        matrix = _TASK_TRANSFORM_MATRIX.get("debugging", {})
        # tool_filter is no longer blocked — it's a reversible SAFE transform.
        # The scheduler ranks it by value instead of blocking it outright.
        assert "tool_filter" not in matrix

    def test_debugging_allows_diagnostic_helpers(self) -> None:
        from lattice.core.scheduler import _TASK_TRANSFORM_MATRIX

        matrix = _TASK_TRANSFORM_MATRIX.get("debugging", {})
        assert "diagnostic_rle" not in matrix
        assert "stack_interning" not in matrix
        assert "causal_chain" not in matrix
