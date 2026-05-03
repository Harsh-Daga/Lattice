"""Tests for SIG (Semantic Importance Graph), RATS (scheduler), and PSG (guardrails)."""

from __future__ import annotations

from lattice.core.guardrails import (
    GuardAction,
    check_blank_output,
    check_entity_preservation,
    check_expansion_guard,
    check_format_preservation,
)
from lattice.core.scheduler import decide_schedule
from lattice.core.semantic_graph import SemanticImportanceGraph, SemanticSpan
from lattice.core.task_classifier import ExecutionTier, TaskClass, TaskClassification, classify_task
from lattice.core.transport import Message, Request
from lattice.utils.validation import SemanticRiskScore


class TestSemanticSpan:
    """SemanticSpan serialization and defaults."""

    def test_empty_span_defaults(self) -> None:
        s = SemanticSpan(span_id=0, text="hello")
        assert s.span_id == 0
        assert s.text == "hello"
        assert s.importance == 0.0
        assert s.protected is False

    def test_to_dict_contains_required_fields(self) -> None:
        s = SemanticSpan(span_id=1, text="test span", importance=75.0, protected=True)
        d = s.to_dict()
        assert d["span_id"] == 1
        assert d["importance"] == 75.0
        assert d["protected"] is True
        assert "text" in d
        assert len(d["text"]) <= 200

    def test_full_span_roundtrip(self) -> None:
        s = SemanticSpan(
            span_id=5,
            text="error: connection refused",
            structure_type="log_line",
            frequency=3.0,
            entity_density=0.8,
            reasoning_signal=True,
            importance=90.0,
            protected=True,
        )
        d = s.to_dict()
        assert d["structure_type"] == "log_line"
        assert d["reasoning_signal"] is True
        assert d["protected"] is True


class TestSemanticImportanceGraph:
    """SIG graph construction and metadata."""

    def test_empty_graph(self) -> None:
        g = SemanticImportanceGraph()
        assert g.total_spans == 0
        assert g.protected_spans == []
        assert g.protected_span_ids == []

    def test_graph_with_spans(self) -> None:
        spans = [
            SemanticSpan(span_id=0, text="error", protected=True),
            SemanticSpan(span_id=1, text="ok", protected=False),
            SemanticSpan(span_id=2, text="critical", protected=True),
        ]
        g = SemanticImportanceGraph(spans=spans, total_spans=3, protected_count=2)
        assert g.protected_span_ids == [0, 2]
        assert len(g.protected_spans) == 2

    def test_summary(self) -> None:
        spans = [SemanticSpan(span_id=0, text="a", structure_type="code")]
        g = SemanticImportanceGraph(spans=spans, total_spans=1)
        s = g.summary()
        assert s["total_spans"] == 1
        assert "code" in s["structure_types"]

    def test_to_dict_serializable(self) -> None:
        g = SemanticImportanceGraph(spans=[SemanticSpan(span_id=0, text="test")], total_spans=1)
        d = g.to_dict()
        assert isinstance(d, dict)
        assert len(d["spans"]) == 1
        assert "edges" in d


class TestTaskClassification:
    """RATS task classification produces correct types."""

    def test_debugging_from_logs(self) -> None:
        req = Request(
            messages=[
                Message(
                    role="user",
                    content="Here are the error logs:\n[ERROR] NullPointer\n[ERROR] Timeout\nWhy did this fail?",
                )
            ]
        )
        tc = classify_task(req)
        assert tc.task_class in (TaskClass.DEBUGGING, TaskClass.REASONING)

    def test_reasoning_from_explanation(self) -> None:
        req = Request(
            messages=[
                Message(
                    role="user",
                    content="Explain why the system failed. Deduce the root cause and solve the issue.",
                )
            ]
        )
        tc = classify_task(req)
        assert tc.task_class in (TaskClass.REASONING, TaskClass.DEBUGGING)

    def test_summarization_long_text(self) -> None:
        long_text = "The quick brown fox jumps over the lazy dog. " * 100
        req = Request(messages=[Message(role="user", content=long_text)])
        tc = classify_task(req)
        assert tc.task_class in (TaskClass.SUMMARIZATION, TaskClass.RETRIEVAL)

    def test_retrieval_lookup(self) -> None:
        req = Request(messages=[Message(role="user", content="Find me the latest commit SHA.")])
        tc = classify_task(req)
        assert tc.task_class in (TaskClass.RETRIEVAL, TaskClass.SUMMARIZATION)

    def test_conservative_tasks(self) -> None:
        tc = TaskClassification(task_class=TaskClass.DEBUGGING)
        assert tc.is_conservative is True
        tc2 = TaskClassification(task_class=TaskClass.REASONING)
        assert tc2.is_conservative is True
        tc3 = TaskClassification(task_class=TaskClass.RETRIEVAL)
        assert tc3.is_conservative is False

    def test_to_dict(self) -> None:
        tc = TaskClassification(task_class=TaskClass.ANALYSIS, reasoning_heavy=True)
        d = tc.to_dict()
        assert d["task_class"] == "analysis"
        assert d["reasoning_heavy"] is True


class TestSchedulerDecision:
    """RATS scheduler produces correct transform permissions."""

    def test_safe_transforms_always_allowed(self) -> None:
        task = TaskClassification(task_class=TaskClass.RETRIEVAL)
        risk = SemanticRiskScore()
        decision = decide_schedule(
            transform_names=["content_profiler", "tool_filter", "output_cleanup"],
            task=task,
            risk=risk,
        )
        assert "content_profiler" in decision.allowed_transforms
        assert "tool_filter" in decision.allowed_transforms

    def test_conditional_blocked_on_reasoning(self) -> None:
        task = TaskClassification(
            task_class=TaskClass.REASONING,
            reasoning_heavy=True,
            execution_tier=ExecutionTier.REASONING_SAFE,
        )
        risk = SemanticRiskScore(strict_instructions=15, sensitive_domain=10)
        decision = decide_schedule(
            transform_names=["reference_sub", "output_cleanup"],
            task=task,
            risk=risk,
        )
        # CONDITIONAL blocked on REASONING_SAFE tier (only SAFE allowed)
        assert "reference_sub" in decision.blocked_transforms
        assert "output_cleanup" in decision.allowed_transforms

    def test_dangerous_blocked_on_debugging(self) -> None:
        task = TaskClassification(task_class=TaskClass.DEBUGGING)
        risk = SemanticRiskScore(strict_instructions=5)
        decision = decide_schedule(
            transform_names=["hierarchical_summary", "tool_filter"],
            task=task,
            risk=risk,
        )
        assert "hierarchical_summary" in decision.blocked_transforms
        assert "tool_filter" in decision.allowed_transforms

    def test_schedule_sort_order(self) -> None:
        task = TaskClassification(task_class=TaskClass.RETRIEVAL)
        risk = SemanticRiskScore()
        decision = decide_schedule(
            transform_names=["reference_sub", "output_cleanup"],
            task=task,
            risk=risk,
        )
        names = [e.transform_name for e in decision.schedule]
        # SAFE before CONDITIONAL
        assert names.index("output_cleanup") < names.index("reference_sub")

    def test_to_dict(self) -> None:
        task = TaskClassification(task_class=TaskClass.ANALYSIS)
        decision = decide_schedule(
            transform_names=["tool_filter"],
            task=task,
        )
        d = decision.to_dict()
        assert "task_class" in d
        assert "blocked" in d
        assert "schedule" in d


class TestGuardrails:
    """PSG guardrails produce correct safety decisions."""

    def test_expansion_guard_allow(self) -> None:
        d = check_expansion_guard(100, 120, max_ratio=1.5)
        assert d.action == GuardAction.ALLOW

    def test_expansion_guard_rollback(self) -> None:
        d = check_expansion_guard(100, 200, max_ratio=1.5)
        assert d.action == GuardAction.ROLLBACK

    def test_expansion_guard_small_input(self) -> None:
        d = check_expansion_guard(0, 10)
        assert d.action == GuardAction.SKIP

    def test_entity_preservation_pass(self) -> None:
        d = check_entity_preservation("hello 123", "hello 123")
        assert d.action == GuardAction.ALLOW

    def test_entity_preservation_fail(self) -> None:
        d = check_entity_preservation("value 550e8400-e29b-41d4-a716-446655440000", "value missing")
        assert d.action == GuardAction.ROLLBACK

    def test_format_preservation_json(self) -> None:
        d = check_format_preservation('{"key": "val"}', "not json")
        assert d.action == GuardAction.ROLLBACK

    def test_format_preservation_pass(self) -> None:
        d = check_format_preservation("hello world", "hello world")
        assert d.action == GuardAction.ALLOW

    def test_blank_output_fail(self) -> None:
        v = check_blank_output("", "something")
        assert v.blank_output is True
        assert v.should_rollback is True

    def test_blank_output_both_empty(self) -> None:
        v = check_blank_output("", "")
        assert v.blank_output is True

    def test_short_output_rollback(self) -> None:
        v = check_blank_output("normal output text", "hi")
        assert v.should_rollback is True

    def test_valid_output_pass(self) -> None:
        v = check_blank_output("baseline text here", "optimized text here")
        assert v.should_rollback is False


class TestSIGIntegration:
    """SIG graph integrates with content_profiler."""

    def test_metadata_keys_present_on_request(self) -> None:
        from lattice.core.context import (
            METADATA_KEY_PROTECTED_SPANS,
            METADATA_KEY_RISK_SCORE,
            METADATA_KEY_SCHEDULE,
            METADATA_KEY_SIG,
            METADATA_KEY_TASK_CLASSIFICATION,
            TransformContext,
        )
        from lattice.core.transport import Message, Request
        from lattice.transforms.content_profiler import ContentProfiler

        profiler = ContentProfiler()
        req = Request(
            messages=[
                Message(
                    role="user",
                    content="Debug this error: the system crashed at line 42. Analyze why.",
                ),
            ]
        )
        ctx = TransformContext()
        from lattice.core.result import is_ok

        result = profiler.process(req, ctx)
        assert is_ok(result)

        from lattice.core.result import unwrap

        modified = unwrap(result)
        assert METADATA_KEY_RISK_SCORE in modified.metadata
        assert METADATA_KEY_TASK_CLASSIFICATION in modified.metadata
        assert METADATA_KEY_PROTECTED_SPANS in modified.metadata
        assert METADATA_KEY_SIG in modified.metadata
        assert METADATA_KEY_SCHEDULE in modified.metadata


class TestRATSSafetyIntegration:
    """RATS + PSG work together in the scheduled pipeline order."""

    def test_debugging_prompt_blocks_dangerous(self) -> None:
        task = TaskClassification(task_class=TaskClass.DEBUGGING, debug_heavy=True)
        risk = SemanticRiskScore(strict_instructions=10)
        decision = decide_schedule(
            transform_names=[
                "rate_distortion",
                "hierarchical_summary",
                "tool_filter",
                "structural_fingerprint",
            ],
            task=task,
            risk=risk,
        )
        # rate_distortion: CONDITIONAL but in _REASONING_DISABLED → blocked
        assert "rate_distortion" in decision.blocked_transforms
        # hierarchical_summary: DANGEROUS → blocked (not in REASONING allowed buckets)
        assert "hierarchical_summary" in decision.blocked_transforms
        # structural_fingerprint: CONDITIONAL, NOT in _REASONING_DISABLED → allowed
        assert "structural_fingerprint" in decision.allowed_transforms
        assert "tool_filter" in decision.allowed_transforms

    def test_retrieval_prompt_allows_aggressive(self) -> None:
        task = TaskClassification(task_class=TaskClass.RETRIEVAL)
        risk = SemanticRiskScore()
        decision = decide_schedule(
            transform_names=["reference_sub", "format_conversion", "tool_filter"],
            task=task,
            risk=risk,
        )
        # All allowed since low risk + retrieval task
        assert len(decision.blocked_transforms) == 0
        assert len(decision.allowed_transforms) == 3
