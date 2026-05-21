"""Tests for replay hardening canonical fingerprints in ScenarioResult."""
from __future__ import annotations

from benchmarks.framework.types import (
    BenchmarkReport,
    ScenarioResult,
)


class TestReplayHardeningFields:
    """Verify ScenarioResult carries canonical fingerprints for determinism."""

    def test_default_fields_are_present(self) -> None:
        s = ScenarioResult(scenario_name="test", category="cat")
        assert s.request_fingerprint == ""
        assert s.execution_plan_fingerprint == ""
        assert s.final_response_fingerprint == ""
        assert s.replay_drift == 0.0
        assert s.determinism_score == 1.0
        assert s.longitudinal_index == 0
        assert s.survivability_score == 1.0

    def test_to_dict_includes_fingerprints(self) -> None:
        s = ScenarioResult(
            scenario_name="x",
            category="c",
            request_fingerprint="abc123",
            execution_plan_fingerprint="def456",
            final_response_fingerprint="fedcba",
            determinism_score=0.95,
            survivability_score=0.88,
            replay_drift=0.03,
        )
        d = s.to_dict()
        assert "replay_hardening" in d
        rh = d["replay_hardening"]
        assert rh["request_fingerprint"] == "abc123"
        assert rh["execution_plan_fingerprint"] == "def456"
        assert rh["final_response_fingerprint"] == "fedcba"
        assert rh["determinism_score"] == 0.95
        assert rh["survivability_score"] == 0.88
        assert rh["replay_drift"] == 0.03

    def test_report_passed_with_fingerprinted_scenarios(self) -> None:
        s = ScenarioResult(
            scenario_name="x",
            category="c",
            request_fingerprint="fp1",
            execution_plan_fingerprint="fp2",
        )
        report = BenchmarkReport(
            runner_name="test", provider="openai", model="gpt-4", scenarios=[s]
        )
        assert report.total_scenarios == 1
        assert report.scenarios[0].request_fingerprint == "fp1"

    def test_non_empty_fingerprint_when_ir_present(self) -> None:
        from lattice.ir.primitives import PromptIRV2, SectionV2, SpanV2

        ir = PromptIRV2(
            sections=(
                SectionV2(type="context", spans=(SpanV2(span_id="s1", text="hello"),)),
            )
        )
        fp = ir.canonical_fingerprint()
        assert len(fp) == 64
        assert all(c in "0123456789abcdef" for c in fp)

        s = ScenarioResult(
            scenario_name="y",
            category="c",
            request_fingerprint=fp,
        )
        assert s.to_dict()["replay_hardening"]["request_fingerprint"] == fp
