"""Tests for the benchmark framework.

Ensures benchmark infrastructure works correctly without requiring
network access.
"""

from __future__ import annotations

import json
import tempfile

from benchmarks.framework.types import (
    BenchmarkReport,
    QualityMeasurement,
    ScenarioResult,
    TokenMeasurement,
)
from benchmarks.metrics.quality import (
    compute_semantic_similarity,
    evaluate_response,
    tool_calls_equivalent,
    validate_json,
)
from benchmarks.scenarios.prompts import get_scenarios, list_scenario_names


class TestScenarioRegistry:
    """Test scenario definitions."""

    def test_all_scenarios_have_name_and_category(self) -> None:
        scenarios = get_scenarios()
        assert len(scenarios) > 0
        for s in scenarios:
            assert s.name
            assert s.category
            assert s.messages

    def test_list_scenario_names(self) -> None:
        names = list_scenario_names()
        assert "uuid_deduplication" in names
        assert "simple_baseline" in names

    def test_filter_scenarios(self) -> None:
        filtered = get_scenarios(["simple_baseline", "uuid_deduplication"])
        assert len(filtered) == 2
        names = {s.name for s in filtered}
        assert "simple_baseline" in names
        assert "uuid_deduplication" in names


class TestQualityMetrics:
    """Test quality evaluation functions."""

    def test_exact_match(self) -> None:
        assert compute_semantic_similarity("Hello world", "Hello world") == 1.0

    def test_different_texts(self) -> None:
        score = compute_semantic_similarity("Hello world", "Goodbye moon")
        assert 0.0 <= score < 1.0

    def test_empty_both(self) -> None:
        assert compute_semantic_similarity("", "") == 1.0

    def test_ref_tags_stripped(self) -> None:
        score = compute_semantic_similarity(
            "The UUID is 550e8400-e29b-41d4-a716-446655440000",
            "The UUID is <ref_1>",
        )
        # After stripping ref tags, "UUID" and "is" and "The" overlap
        assert score > 0.3  # Should still have some overlap after stripping ref tags

    def test_evaluate_response_pass(self) -> None:
        q = evaluate_response("Hello world", "Hello world")
        assert q.semantic_similarity == 1.0
        assert q.exact_match is True
        assert q.passed is True

    def test_evaluate_response_fail(self) -> None:
        q = evaluate_response("Hello world", "Completely different text here")
        assert q.semantic_similarity < 0.7
        assert q.passed is False

    def test_validate_json_valid(self) -> None:
        assert validate_json('{"key": "value"}') is True

    def test_validate_json_invalid(self) -> None:
        assert validate_json('{"key": value}') is False

    def test_validate_json_none(self) -> None:
        assert validate_json("Just text") is None

    def test_tool_calls_equivalent_both_none(self) -> None:
        assert tool_calls_equivalent(None, None) is None

    def test_tool_calls_equivalent_mismatch_count(self) -> None:
        a = [{"id": "1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}]
        b = []
        assert tool_calls_equivalent(a, b) is False

    def test_tool_calls_equivalent_same(self) -> None:
        a = [{"id": "1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}]
        b = [{"id": "2", "type": "function", "function": {"name": "foo", "arguments": "{}"}}]
        assert tool_calls_equivalent(a, b) is True


class TestBenchmarkTypes:
    """Test benchmark data types and aggregation."""

    def test_scenario_result_aggregation(self) -> None:
        result = ScenarioResult(
            scenario_name="test",
            category="test",
            provider="openai",
            model="gpt-4",
        )
        result.optimized_tokens = [
            TokenMeasurement(before=100, after=80, saved=20, ratio=0.2),
            TokenMeasurement(before=100, after=70, saved=30, ratio=0.3),
        ]
        result.qualities = [
            QualityMeasurement(semantic_similarity=0.9),
            QualityMeasurement(semantic_similarity=0.8),
        ]

        avg = result.avg_optimized_tokens
        assert avg.before == 100
        assert avg.after == 75  # (80+70)/2
        assert avg.saved == 25

        avg_q = result.avg_quality
        assert abs(avg_q.semantic_similarity - 0.85) < 0.001
        assert avg_q.passed is True

    def test_scenario_result_all_passed(self) -> None:
        result = ScenarioResult(scenario_name="test", category="test")
        result.qualities = [
            QualityMeasurement(semantic_similarity=0.9),
            QualityMeasurement(semantic_similarity=0.8),
        ]
        assert result.all_passed is True

    def test_scenario_result_some_failed(self) -> None:
        result = ScenarioResult(scenario_name="test", category="test")
        result.qualities = [
            QualityMeasurement(semantic_similarity=0.9),
            QualityMeasurement(semantic_similarity=0.5),
        ]
        assert result.all_passed is False

    def test_benchmark_report_summary(self) -> None:
        report = BenchmarkReport(
            runner_name="test",
            provider="openai",
            model="gpt-4",
        )
        r1 = ScenarioResult(scenario_name="s1", category="c1")
        r1.optimized_tokens = [TokenMeasurement(before=100, after=80, saved=20, ratio=0.2)]
        r1.qualities = [QualityMeasurement(semantic_similarity=0.9)]

        r2 = ScenarioResult(scenario_name="s2", category="c2")
        r2.optimized_tokens = [TokenMeasurement(before=200, after=100, saved=100, ratio=0.5)]
        r2.qualities = [QualityMeasurement(semantic_similarity=0.8)]

        report.scenarios = [r1, r2]

        assert report.total_scenarios == 2
        assert report.total_token_savings == 120
        assert abs(report.avg_reduction_ratio - 0.35) < 0.001
        assert abs(report.avg_quality_score - 0.85) < 0.001

    def test_benchmark_report_to_dict(self) -> None:
        report = BenchmarkReport(runner_name="test", provider="openai", model="gpt-4")
        r = ScenarioResult(scenario_name="s1", category="c1")
        r.optimized_tokens = [TokenMeasurement(before=100, after=80, saved=20, ratio=0.2)]
        r.qualities = [QualityMeasurement(semantic_similarity=0.9)]
        report.scenarios = [r]

        d = report.to_dict()
        assert d["runner"] == "test"
        assert d["summary"]["total_scenarios"] == 1
        assert d["summary"]["total_token_savings"] == 20

    def test_benchmark_report_json_roundtrip(self) -> None:
        report = BenchmarkReport(runner_name="test", provider="openai", model="gpt-4")
        r = ScenarioResult(scenario_name="s1", category="c1")
        r.optimized_tokens = [TokenMeasurement(before=100, after=80, saved=20, ratio=0.2)]
        r.qualities = [QualityMeasurement(semantic_similarity=0.9)]
        report.scenarios = [r]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report.to_dict(), f)
            path = f.name

        with open(path) as f:
            loaded = json.load(f)

        assert loaded["runner"] == "test"
        assert loaded["summary"]["total_scenarios"] == 1
