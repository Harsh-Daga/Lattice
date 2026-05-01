"""Regression tests for meaning preservation and safety gates.

If a transform preserves token count but changes meaning, the test fails.
If a transform changes formatting that the user asked to preserve, the test fails.
If a transform expands context too much, the test fails.
"""

from __future__ import annotations

from benchmarks.framework.types import QualityMeasurement, TaskEquivalenceScore
from lattice.core.config import LatticeConfig
from lattice.core.transport import Message, Request
from lattice.utils.validation import (
    SemanticRiskScore,
    TransformSafetyBucket,
    compute_risk_score,
    get_transform_safety_bucket,
    transform_allowed_at_risk,
)


class TestTaskEquivalenceScoring:
    """Task-equivalence scoring replaces weak semantic_similarity."""

    def test_composite_perfect_equivalence(self) -> None:
        te = TaskEquivalenceScore()
        assert te.composite == 1.0
        assert te.passed is True

    def test_blank_outputs_automatic_fail(self) -> None:
        """Blank or failed outputs should zero out ALL dimensions."""
        from benchmarks.evals.runner import evaluate_task_equivalence_structural

        te = evaluate_task_equivalence_structural("", "some output", [])
        assert te.composite == 0.0
        assert te.passed is False
        assert te.correctness == 0.0
        assert te.key_fact_preservation == 0.0
        assert te.harmful_drift == 1.0
        assert "blank_output" in te.failure_reasons

    def test_both_blank_outputs_fail(self) -> None:
        from benchmarks.evals.runner import evaluate_task_equivalence_structural

        te = evaluate_task_equivalence_structural("", "", [])
        assert te.composite == 0.0
        assert te.passed is False

    def test_partial_entity_loss_fails(self) -> None:
        te = TaskEquivalenceScore(
            correctness=0.5,
            key_fact_preservation=0.5,
            completeness=0.5,
        )
        assert te.passed is False  # composite < 0.85

    def test_constraint_failure_scores_low(self) -> None:
        te = TaskEquivalenceScore(correctness=0.4)
        assert te.composite < 1.0

    def test_harmful_drift_penalizes(self) -> None:
        te = TaskEquivalenceScore(
            harmful_drift=0.5, correctness=0.7, key_fact_preservation=0.7
        )
        assert te.composite < 0.85
        assert te.passed is False

    def test_quality_measurement_uses_task_equivalence(self) -> None:
        te = TaskEquivalenceScore()
        qm = QualityMeasurement(task_equivalence=te, semantic_similarity=0.5)
        assert qm.passed is True  # uses task_equivalence, not semantic_similarity

    def test_quality_measurement_falls_back_to_semantic_similarity(self) -> None:
        qm = QualityMeasurement(semantic_similarity=0.9)
        assert qm.passed is True  # no task_equivalence, uses semantic_similarity

    def test_quality_fails_when_task_equivalence_fails(self) -> None:
        te = TaskEquivalenceScore(
            correctness=0.3,
            key_fact_preservation=0.3,
            completeness=0.3,
        )
        qm = QualityMeasurement(semantic_similarity=0.9, task_equivalence=te)
        assert qm.passed is False  # task_equivalence is source of truth


class TestSemanticRiskScoring:
    """Risk scores correctly identify dangerous input shapes."""

    def test_low_risk_simple_prompt(self) -> None:
        req = Request(messages=[Message(role="user", content="Hello, how are you?")])
        score = compute_risk_score(req)
        assert score.level == "LOW"

    def test_high_risk_strict_instructions(self) -> None:
        req = Request(
            messages=[
                Message(
                    role="user",
                    content="Do not change the format. You must preserve exactly the JSON output.",
                )
            ]
        )
        score = compute_risk_score(req)
        assert score.total >= 10.0

    def test_high_risk_code_blocks(self) -> None:
        req = Request(
            messages=[
                Message(
                    role="user",
                    content="```python\ndef foo():\n    pass\n```\nDo not change this code. Keep exactly the format.",
                )
            ]
        )
        score = compute_risk_score(req)
        assert score.strict_instructions >= 3.0
        # Code blocks alone don't trigger structured output; strict instructions do

    def test_high_risk_sensitive_domain(self) -> None:
        req = Request(
            messages=[
                Message(
                    role="user",
                    content="This medical diagnosis must be accurate. Financial account numbers are confidential.",
                )
            ]
        )
        score = compute_risk_score(req)
        assert score.sensitive_domain >= 5.0

    def test_high_risk_reasoning_heavy(self) -> None:
        req = Request(
            messages=[
                Message(
                    role="user",
                    content="Reason step by step. Think carefully about why the system failed. Explain why and deduce the root cause.",
                )
            ]
        )
        score = compute_risk_score(req)
        assert score.reasoning_heavy >= 10.0

    def test_integer_risk_score(self) -> None:
        req = Request(messages=[Message(role="user", content="12345 67890 11111 22222 33333")])
        score = compute_risk_score(req)
        assert score.high_stakes_entities > 0


class TestTransformSafetyBuckets:
    """Every transform has a safety bucket assignment."""

    def test_safe_transforms(self) -> None:
        for name in (
            "tool_filter",
            "prefix_optimizer",
            "output_cleanup",
            "content_profiler",
            "cache_arbitrage",
        ):
            assert get_transform_safety_bucket(name) == TransformSafetyBucket.SAFE, name

    def test_conditional_transforms(self) -> None:
        for name in ("reference_sub", "message_dedup", "format_conversion", "semantic_compress"):
            assert get_transform_safety_bucket(name) == TransformSafetyBucket.CONDITIONAL, name

    def test_dangerous_transforms(self) -> None:
        for name in ("structural_fingerprint", "hierarchical_summary"):
            assert get_transform_safety_bucket(name) == TransformSafetyBucket.DANGEROUS, name

    def test_unknown_transform_defaults_to_dangerous(self) -> None:
        assert (
            get_transform_safety_bucket("nonexistent_transform")
            == TransformSafetyBucket.DANGEROUS
        )

    def test_alias_prefix_opt_maps_to_safe(self) -> None:
        assert get_transform_safety_bucket("prefix_opt") == TransformSafetyBucket.SAFE

    def test_alias_tool_output_filter_maps_to_safe(self) -> None:
        assert get_transform_safety_bucket("tool_output_filter") == TransformSafetyBucket.SAFE

    def test_alias_message_deduplicator_maps_to_conditional(self) -> None:
        assert (
            get_transform_safety_bucket("message_deduplicator") == TransformSafetyBucket.CONDITIONAL
        )

    def test_alias_dictionary_compressor_maps_to_conditional(self) -> None:
        assert (
            get_transform_safety_bucket("dictionary_compressor")
            == TransformSafetyBucket.CONDITIONAL
        )

    def test_alias_grammar_compressor_maps_to_conditional(self) -> None:
        assert (
            get_transform_safety_bucket("grammar_compressor") == TransformSafetyBucket.CONDITIONAL
        )

    def test_alias_hierarchical_summarizer_maps_to_dangerous(self) -> None:
        assert (
            get_transform_safety_bucket("hierarchical_summarizer")
            == TransformSafetyBucket.DANGEROUS
        )


class TestRiskGatingBehavior:
    """Risk-gating blocks transforms appropriately."""

    def test_safe_transform_always_allowed(self) -> None:
        risk = SemanticRiskScore(strict_instructions=30)
        allowed, reason = transform_allowed_at_risk("content_profiler", risk)
        assert allowed is True
        assert reason == "safe_transform"

    def test_conditional_blocked_at_high_risk(self) -> None:
        risk = SemanticRiskScore(
            strict_instructions=30, sensitive_domain=20, high_stakes_entities=15
        )
        assert risk.total > 60
        allowed, reason = transform_allowed_at_risk("semantic_compress", risk)
        assert allowed is False
        assert "blocked" in reason

    def test_conditional_allowed_at_medium_risk(self) -> None:
        risk = SemanticRiskScore(strict_instructions=12, sensitive_domain=10)
        assert risk.level == "MEDIUM"
        allowed, reason = transform_allowed_at_risk("format_conversion", risk)
        assert allowed is True

    def test_dangerous_blocked_at_medium_risk(self) -> None:
        risk = SemanticRiskScore(
            sensitive_domain=10, high_stakes_entities=15, strict_instructions=10
        )
        assert risk.level in ("MEDIUM", "HIGH")
        allowed, reason = transform_allowed_at_risk("structural_fingerprint", risk)
        assert allowed is False
        assert "blocked" in reason

    def test_dangerous_allowed_at_low_risk(self) -> None:
        risk = SemanticRiskScore()
        assert risk.level == "LOW"
        allowed, reason = transform_allowed_at_risk("structural_fingerprint", risk)
        assert allowed is True


class TestNoIntermediateExplosion:
    """Intermediate explosion guardrails work in the pipeline."""

    def test_pipeline_has_max_expansion_ratio_config(self) -> None:
        config = LatticeConfig()
        assert config.max_transform_expansion_ratio >= 1.0


class TestScenarioSafetyExpectations:
    """Every scenario has explicit safety expectations."""

    def test_all_scenarios_have_safety_expectations(self) -> None:
        from benchmarks.scenarios.prompts import _SCENARIO_SAFETY, ALL_SCENARIOS

        for scenario in ALL_SCENARIOS:
            safety = _SCENARIO_SAFETY.get(scenario.name, {})
            # New scenarios have safety inline; fallback to _SCENARIO_SAFETY map
            if not safety:
                safety = {
                    "safe_transforms": scenario.safe_transforms,
                    "forbidden_transforms": scenario.forbidden_transforms,
                    "required_answer_properties": scenario.required_answer_properties,
                    "judge_rubric": scenario.judge_rubric,
                }
            assert "safe_transforms" in safety or scenario.safe_transforms, f"{scenario.name} missing safe_transforms"
            assert "forbidden_transforms" in safety, f"{scenario.name} missing forbidden_transforms"
            assert "required_answer_properties" in safety, (
                f"{scenario.name} missing required_answer_properties"
            )
            assert "judge_rubric" in safety, f"{scenario.name} missing judge_rubric"

    def test_forbidden_transforms_never_in_safe_list(self) -> None:
        from benchmarks.scenarios.prompts import _SCENARIO_SAFETY, ALL_SCENARIOS

        for scenario in ALL_SCENARIOS:
            safety = _SCENARIO_SAFETY.get(scenario.name, {})
            safe = set(safety.get("safe_transforms", []))
            forbidden = set(safety.get("forbidden_transforms", []))
            assert not (safe & forbidden), f"{scenario.name}: transform in both safe and forbidden"

    def test_tool_call_scenario_forbids_lossy_transforms(self) -> None:
        from benchmarks.scenarios.prompts import _SCENARIO_SAFETY

        safety = _SCENARIO_SAFETY.get("tool_call_preservation", {})
        forbidden = safety.get("forbidden_transforms", [])
        assert "semantic_compress" in forbidden
        assert "reference_sub" in forbidden

    def test_simple_baseline_forbids_most_transforms(self) -> None:
        from benchmarks.scenarios.prompts import _SCENARIO_SAFETY

        safety = _SCENARIO_SAFETY.get("simple_baseline", {})
        forbidden = safety.get("forbidden_transforms", [])
        # Simple prompts should not get lossy transforms
        assert len(forbidden) > 2

    def test_reasoning_scenario_protects_reasoning(self) -> None:
        from benchmarks.scenarios.prompts import _SCENARIO_SAFETY

        safety = _SCENARIO_SAFETY.get("runtime_contract_pressure", {})
        # Reasoning-heavy: structural_fingerprint must be forbidden
        assert "structural_fingerprint" in safety.get("forbidden_transforms", [])
        # Must preserve reasoning
        assert any(
            "reasoning" in p.lower() or "mitigation" in p.lower()
            for p in safety.get("required_answer_properties", [])
        )


class TestEntityPreservation:
    """Entities, numbers, and formats must survive compression."""

    def test_uuid_preservation_required_in_dedup_scenario(self) -> None:
        from benchmarks.scenarios.prompts import _SCENARIO_SAFETY

        safety = _SCENARIO_SAFETY.get("uuid_deduplication", {})
        props = safety.get("required_answer_properties", [])
        assert any("uuid" in p.lower() for p in props), "UUID preservation not required"

    def test_number_preservation_required_in_table_scenario(self) -> None:
        from benchmarks.scenarios.prompts import _SCENARIO_SAFETY

        safety = _SCENARIO_SAFETY.get("table_compression", {})
        props = safety.get("required_answer_properties", [])
        assert any("salary" in p.lower() or "range" in p.lower() for p in props)

    def test_json_format_preserved_in_json_scenario(self) -> None:
        from benchmarks.scenarios.prompts import _SCENARIO_SAFETY

        safety = _SCENARIO_SAFETY.get("json_response_format", {})
        props = safety.get("required_answer_properties", [])
        assert any("json" in p.lower() for p in props)


class TestRiskScoreThresholds:
    """Risk thresholds align with the specification."""

    def test_low_up_to_20(self) -> None:
        score = SemanticRiskScore(strict_instructions=10, sensitive_domain=10)
        assert score.total == 20.0
        assert score.level == "LOW"

    def test_medium_21_to_40(self) -> None:
        score = SemanticRiskScore(strict_instructions=15, sensitive_domain=10, structured_output=8)
        assert score.level == "MEDIUM"

    def test_high_41_to_60(self) -> None:
        score = SemanticRiskScore(strict_instructions=25, sensitive_domain=15, structured_output=10)
        assert score.level == "HIGH"

    def test_critical_above_60(self) -> None:
        score = SemanticRiskScore(strict_instructions=30, sensitive_domain=20, reasoning_heavy=15)
        assert score.total > 60
        assert score.level == "CRITICAL"


class TestReportAvgQualityScore:
    """avg_quality_score uses task-equivalence composite as source of truth."""

    def test_prefers_task_equivalence_over_semantic_similarity(self) -> None:
        from benchmarks.framework.types import BenchmarkReport, QualityMeasurement, ScenarioResult

        te_pass = TaskEquivalenceScore()  # composite = 1.0
        te_fail = TaskEquivalenceScore(
            correctness=0.3,
            completeness=0.3,
            key_fact_preservation=0.3,
            reasoning_equivalence=0.3,
            numeric_preservation=0.3,
            schema_validity=0.3,
            harmful_drift=0.3,
        )  # composite < 0.85

        report = BenchmarkReport(
            runner_name="test",
            provider="openai",
            model="gpt-4",
            scenarios=[
                ScenarioResult(
                    scenario_name="passing",
                    category="test",
                    qualities=[
                        QualityMeasurement(semantic_similarity=0.5, task_equivalence=te_pass),
                    ],
                    optimized_tokens=[],
                    baseline_tokens=[],
                ),
                ScenarioResult(
                    scenario_name="failing",
                    category="test",
                    qualities=[
                        QualityMeasurement(semantic_similarity=0.95, task_equivalence=te_fail),
                    ],
                    optimized_tokens=[],
                    baseline_tokens=[],
                ),
            ],
        )
        # Both use task_equivalence — semantic_similarity is ignored
        score = report.avg_quality_score
        expected_avg = (1.0 + te_fail.composite) / 2.0
        assert score == round(expected_avg, 4)

    def test_falls_back_to_semantic_similarity_when_no_task_equivalence(self) -> None:
        from benchmarks.framework.types import BenchmarkReport, QualityMeasurement, ScenarioResult

        report = BenchmarkReport(
            runner_name="test",
            provider="openai",
            model="gpt-4",
            scenarios=[
                ScenarioResult(
                    scenario_name="s1",
                    category="test",
                    qualities=[QualityMeasurement(semantic_similarity=0.8)],
                    optimized_tokens=[],
                    baseline_tokens=[],
                ),
                ScenarioResult(
                    scenario_name="s2",
                    category="test",
                    qualities=[QualityMeasurement(semantic_similarity=0.6)],
                    optimized_tokens=[],
                    baseline_tokens=[],
                ),
            ],
        )
        score = report.avg_quality_score
        assert score == round((0.8 + 0.6) / 2, 4)
