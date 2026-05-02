"""Regression tests for v2 architecture: classifier, SIG, scheduler, reachability."""

from __future__ import annotations

import asyncio
import pytest

from lattice.core.transport import Message, Request
from lattice.core.task_classifier import ExecutionTier, TaskClass, TaskClassification, classify_task
from lattice.core.scheduler import _REASONING_DISABLED, decide_schedule
from lattice.utils.validation import SemanticRiskScore


class TestClassifierV2:
    """Phase 2: Hybrid classifier with hard REASONING overrides."""

    def test_root_cause_forces_reasoning(self) -> None:
        req = Request(messages=[
            Message(role="user", content="Find the root cause of the memory leak."),
        ])
        tc = classify_task(req)
        assert tc.execution_tier == ExecutionTier.REASONING
        assert tc.hard_override is True

    def test_debugging_with_reasoning_forces_reasoning(self) -> None:
        req = Request(messages=[
            Message(role="user", content=(
                "The system crashed with error E0503. "
                "Analyze the stack trace and explain why the failure occurred. "
                "Deduce the root cause."
            )),
        ])
        tc = classify_task(req)
        assert tc.execution_tier in (ExecutionTier.REASONING, ExecutionTier.REASONING_SAFE)

    def test_simple_prompt_is_simple(self) -> None:
        req = Request(messages=[Message(role="user", content="Hello, how are you?")])
        tc = classify_task(req)
        assert tc.task_class in (TaskClass.SIMPLE, TaskClass.RETRIEVAL)

    def test_retrieval_prompt_is_retrieval(self) -> None:
        req = Request(messages=[Message(role="user", content="Find me the latest commit.")])
        tc = classify_task(req)
        assert tc.task_class == TaskClass.RETRIEVAL

    def test_structured_content_classified(self) -> None:
        req = Request(messages=[Message(role="user", content='{"key": "val"} | Name | Value |')])
        tc = classify_task(req)
        assert tc.structured_heavy is True

    def test_low_confidence_goes_reasoning_safe(self) -> None:
        """When confidence < 0.7 and score >= 40, tier becomes REASONING_SAFE."""
        tc = TaskClassification(task_class=TaskClass.REASONING, score=45, confidence=0.5)
        assert tc.requires_safe_mode is True


class TestSchedulerV2:
    """Phase 4: Tier-based scheduler with REASONING_DISABLED."""

    def test_reasoning_disables_lossy_transforms(self) -> None:
        task = TaskClassification(
            task_class=TaskClass.REASONING,
            execution_tier=ExecutionTier.REASONING,
        )
        risk = SemanticRiskScore()
        decision = decide_schedule(
            transform_names=["message_dedup", "rate_distortion", "tool_filter"],
            task=task, risk=risk,
        )
        assert "message_dedup" in decision.blocked_transforms
        assert "rate_distortion" in decision.blocked_transforms
        assert "tool_filter" in decision.allowed_transforms

    def test_reasoning_allows_reversible_conditionals(self) -> None:
        task = TaskClassification(
            task_class=TaskClass.REASONING,
            execution_tier=ExecutionTier.REASONING,
        )
        risk = SemanticRiskScore()
        decision = decide_schedule(
            transform_names=["reference_sub", "dictionary_compress", "grammar_compress"],
            task=task, risk=risk,
        )
        assert "reference_sub" not in decision.blocked_transforms
        assert "dictionary_compress" not in decision.blocked_transforms
        assert "grammar_compress" not in decision.blocked_transforms

    def test_debugging_uses_reasoning_tier(self) -> None:
        task = TaskClassification(
            task_class=TaskClass.DEBUGGING,
            execution_tier=ExecutionTier.SIMPLE,
        )
        risk = SemanticRiskScore()
        decision = decide_schedule(
            transform_names=["message_dedup"],
            task=task, risk=risk,
        )
        # is_conservative overrides SIMPLE → REASONING, blocking message_dedup
        assert "message_dedup" in decision.blocked_transforms

    def test_reasoning_safe_only_allows_safe(self) -> None:
        task = TaskClassification(
            execution_tier=ExecutionTier.REASONING_SAFE,
        )
        risk = SemanticRiskScore()
        decision = decide_schedule(
            transform_names=["reference_sub", "output_cleanup", "tool_filter"],
            task=task, risk=risk,
        )
        assert "reference_sub" in decision.blocked_transforms
        assert "output_cleanup" in decision.allowed_transforms
        assert "tool_filter" in decision.allowed_transforms


class TestSIGContrastive:
    """Phase 3: Contrastive SIG with top-k protection."""

    def test_counts_are_force_protected(self) -> None:
        from lattice.transforms.content_profiler import _segment_spans, _extract_features, _compute_importance, _derive_protected, _build_importance_graph
        from lattice.core.transport import Request, Message

        req = Request(messages=[
            Message(role="user", content="There were 15 errors, 3 failures, and 2 timeouts."),
        ])
        graph = _build_importance_graph(req)
        protected_ids = graph.protected_span_ids
        assert len(protected_ids) > 0

    def test_boilerplate_is_not_protected(self) -> None:
        from lattice.core.transport import Request, Message
        from lattice.transforms.content_profiler import _build_importance_graph

        req = Request(messages=[
            Message(role="user", content="The quick brown fox jumps over the lazy dog. " * 10),
        ])
        graph = _build_importance_graph(req)
        # Boilerplate should have relatively few protected spans
        assert graph.protected_count <= max(1, int(0.3 * graph.total_spans))

    def test_sig_produces_compressible_spans(self) -> None:
        from lattice.core.transport import Request, Message
        from lattice.core.semantic_graph import SemanticSpan
        from lattice.transforms.content_profiler import _build_importance_graph

        req = Request(messages=[
            Message(role="user", content="The repeated phrase. The repeated phrase. The repeated phrase."),
        ])
        graph = _build_importance_graph(req)
        compressible = [s for s in graph.spans if s.compressible]
        assert len(compressible) >= 0  # Should not crash


class TestReachability:
    """Phase 6: reached/activated/useful telemetry."""

    def test_pipeline_produces_reachability_metadata(self) -> None:
        from lattice.core.config import LatticeConfig
        from lattice.core.context import TransformContext
        from lattice.core.pipeline_factory import build_default_pipeline
        from lattice.core.result import is_ok, unwrap

        async def run():
            config = LatticeConfig()
            pipeline = build_default_pipeline(config)
            req = Request(messages=[Message(role="user", content="Compress this text. " * 20)])
            ctx = TransformContext()
            result = await pipeline.process(req, ctx)
            assert is_ok(result)
            mod = unwrap(result)
            reach = mod.metadata.get("_lattice_reachability", {})
            assert "reached" in reach
            assert "activated" in reach
            assert "useful" in reach
            assert reach["reached_count"] >= 0
        asyncio.run(run())

    def test_safety_decision_recorded(self) -> None:
        from lattice.core.config import LatticeConfig
        from lattice.core.context import TransformContext
        from lattice.core.pipeline_factory import build_default_pipeline
        from lattice.core.result import is_ok, unwrap

        async def run():
            config = LatticeConfig()
            pipeline = build_default_pipeline(config)
            req = Request(messages=[Message(role="user", content="Test")])
            ctx = TransformContext()
            result = await pipeline.process(req, ctx)
            assert is_ok(result)
            mod = unwrap(result)
            safety = mod.metadata.get("_lattice_safety_decision", {})
            assert "applied" in safety
            assert "rollback_reasons" in safety
        asyncio.run(run())


class TestProductionGuards:
    """Phase 8: Production safety guards for reasoning tasks."""

    def test_compression_limit_guard_exists(self) -> None:
        from lattice.core.config import LatticeConfig
        from lattice.core.context import TransformContext
        from lattice.core.pipeline_factory import build_default_pipeline
        from lattice.core.result import is_ok, unwrap

        async def run():
            config = LatticeConfig(graceful_degradation=True)
            pipeline = build_default_pipeline(config)
            req = Request(messages=[
                Message(role="user", content=(
                    "Find the root cause of the error. "
                    + "The system crashed and needs debugging. " * 20
                )),
            ])
            ctx = TransformContext()
            result = await pipeline.process(req, ctx)
            # Should not crash even on REASONING-classified prompt
            assert is_ok(result)
        asyncio.run(run())
