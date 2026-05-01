"""Report models and renderers for production evals."""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from benchmarks.framework.types import BenchmarkReport


def _excerpt(text: str, limit: int = 180) -> str:
    text = " ".join(str(text).split()).replace("|", r"\|")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


@dataclasses.dataclass(slots=True)
class EvalSectionReport:
    """A named section within a production eval bundle."""

    name: str
    kind: str
    status: str = "ok"
    summary: dict[str, Any] = dataclasses.field(default_factory=dict)
    details: dict[str, Any] = dataclasses.field(default_factory=dict)
    benchmark: BenchmarkReport | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "summary": self.summary,
            "details": self.details,
        }
        if self.benchmark is not None:
            data["benchmark"] = self.benchmark.to_dict()
        return data


@dataclasses.dataclass(slots=True)
class ProductionEvalReport:
    """Top-level bundle for all production eval sections."""

    runner_name: str
    timestamp: float = dataclasses.field(default_factory=time.time)
    config: dict[str, Any] = dataclasses.field(default_factory=dict)
    sections: list[EvalSectionReport] = dataclasses.field(default_factory=list)

    @property
    def total_sections(self) -> int:
        return len(self.sections)

    @property
    def passed_sections(self) -> int:
        return sum(1 for section in self.sections if section.status == "ok")

    def to_dict(self) -> dict[str, Any]:
        return {
            "runner": self.runner_name,
            "timestamp": self.timestamp,
            "summary": {
                "total_sections": self.total_sections,
                "passed_sections": self.passed_sections,
            },
            "config": self.config,
            "sections": [section.to_dict() for section in self.sections],
        }


def _render_special_details(lines: list[str], details: dict[str, Any]) -> bool:
    """Render structured detail tables for benchmark-friendly sections."""
    rendered = False

    coverage_summary = details.get("coverage_summary")
    if isinstance(coverage_summary, dict) and coverage_summary:
        lines.extend(
            [
                "",
                "### Coverage Summary",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
            ]
        )
        for key in ("scenario_count", "proof_passed", "proof_failed", "feature_failed", "tier_failed", "budget_failed"):
            lines.append(f"| {key} | {coverage_summary.get(key, 0)} |")
        tiers = coverage_summary.get("tiers") or {}
        for tier in ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING"):
            lines.append(f"| tier::{tier.lower()} | {tiers.get(tier, 0)} |")
        flaw_counts = coverage_summary.get("flaw_counts") or {}
        for flaw, count in sorted(flaw_counts.items()):
            lines.append(f"| flaw::{flaw} | {count} |")
        features = coverage_summary.get("features") or {}
        for feature, count in sorted(features.items()):
            lines.append(f"| feature::{feature} | {count} |")
        rendered = True

    transform_summary = details.get("transform_summary")
    if isinstance(transform_summary, dict) and transform_summary:
        lines.extend(
            [
                "",
                "### Transform Summary",
                "",
                "| Transform | Avg Before | Avg After | Avg Saved | Count |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, values in sorted(transform_summary.items()):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(name),
                        str(values.get("avg_before", 0.0)),
                        str(values.get("avg_after", 0.0)),
                        str(values.get("avg_saved", 0.0)),
                        str(values.get("count", 0)),
                    ]
                )
                + " |"
        )
        rendered = True

    scenario_proof = details.get("scenario_proof")
    if isinstance(scenario_proof, list) and scenario_proof:
        lines.extend(
            [
                "",
                "### Scenario Proof",
                "",
                "| Scenario | Complexity | Expected Tier | Observed Tier | Features | Status | Flaws |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in scenario_proof:
            features = ", ".join(row.get("target_features", [])) or "-"
            flaws = "; ".join(row.get("flaws", [])) or "-"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("scenario", "")),
                        str(row.get("complexity", "")),
                        str(row.get("expected_tier", "")),
                        str(row.get("observed_tier", "")),
                        features,
                        str(row.get("status", "fail")),
                        flaws,
                    ]
                )
                + " |"
            )
        rendered = True

    feature_matrix = details.get("feature_matrix")
    if isinstance(feature_matrix, list) and feature_matrix:
        lines.extend(
            [
                "",
                "### Feature Matrix",
                "",
                "| Feature | Surface | Evidence | Supporting Traces | Signal Metric | Off Signal | On Signal | Delta Signal | Off Savings | On Savings | Off Ratio | On Ratio | Delta Savings | Verdict |",
                "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in feature_matrix:
            traces = ", ".join(row.get("supporting_traces", [])) or "-"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("feature", "")),
                        str(row.get("surface", "")),
                        str(row.get("evidence", "")),
                        traces,
                        str(row.get("signal_metric", "")),
                        f"{float(row.get('off_signal', 0.0)):.4f}",
                        f"{float(row.get('on_signal', 0.0)):.4f}",
                        f"{float(row.get('delta_signal', 0.0)):.4f}",
                        str(row.get("off_savings", 0)),
                        str(row.get("on_savings", 0)),
                        f"{float(row.get('off_reduction_ratio', 0.0)):.4f}",
                        f"{float(row.get('on_reduction_ratio', 0.0)):.4f}",
                        str(row.get("delta_savings", 0)),
                        str(row.get("verdict", "")),
                    ]
                )
                + " |"
            )
        rendered = True

    validation_results = details.get("results")
    if isinstance(validation_results, list) and validation_results:
        lines.extend(
            [
                "",
                "### Provider Validation Results",
                "",
                "| Scenario | Provider | Task Eq. | Token Pass | Overall | Tokens Before | Tokens After | Risk | Forbidden Applied |",
                "| --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |",
            ]
        )
        for r in validation_results:
            te = r.get("task_equivalence", {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(r.get("scenario", "")),
                        str(r.get("provider", "")),
                        f"{te.get('composite', 0.0):.3f}",
                        "pass" if r.get("token_reduction_pass") else "FAIL",
                        "pass" if r.get("overall_pass") else "FAIL",
                        str(r.get("tokens_before", 0)),
                        str(r.get("tokens_after", 0)),
                        str(r.get("risk_level", "?")),
                        ", ".join(r.get("forbidden_applied", [])) or "none",
                    ]
                )
                + " |"
            )

        # PSG explainability — rollback reasons and safety decisions
        lines.extend(
            [
                "",
                "### PSG Safety Decisions",
                "",
                "| Scenario | Judge Verdict | Safety Rollbacks | Expansion Aborts | Scheduler Blocks |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for r in validation_results:
            rollbacks = [f"{k}:{v}" for k, v in r.get("rollback_reasons", {}).items()]
            expansions = [f"{k}:{v:.2f}x" for k, v in r.get("expansion_ratios", {}).items()]
            schedule = r.get("schedule", {})
            blocks = schedule.get("blocked", []) if isinstance(schedule, dict) else []
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(r.get("scenario", "")),
                        r.get("judge_verdict", ""),
                        "; ".join(rollbacks) if rollbacks else "none",
                        "; ".join(expansions) if expansions else "none",
                        ", ".join(blocks) if blocks else "none",
                    ]
                )
                + " |"
            )

        # Task equivalence breakdowns
        lines.extend(
            [
                "",
                "### Task Equivalence Breakdown",
                "",
                "| Scenario | Constraint | Entity | Format | Reasoning | Refusal | Complete | Drift |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for r in validation_results:
            te = r.get("task_equivalence", {})
            if not te:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(r.get("scenario", "")),
                        f"{te.get('constraint_preservation', 0.0):.3f}",
                        f"{te.get('entity_preservation', 0.0):.3f}",
                        f"{te.get('format_preservation', 0.0):.3f}",
                        f"{te.get('reasoning_correctness', 0.0):.3f}",
                        f"{te.get('refusal_correctness', 0.0):.3f}",
                        f"{te.get('answer_completeness', 0.0):.3f}",
                        f"{te.get('harmful_drift', 0.0):.3f}",
                    ]
                )
                + " |"
            )

        # Safety gate decisions
        lines.extend(
            [
                "",
                "### Safety Gate Decisions",
                "",
                "| Scenario | Risk Level | Risk Score | Risky Applied | Expansion Detected | Judge Verdict |",
                "| --- | --- | ---: | --- | --- | --- |",
            ]
        )
        for r in validation_results:
            expansion = r.get("expansion_ratios") or {}
            expansion_summary = ", ".join(
                f"{k}:{v:.2f}x" for k, v in expansion.items()
            ) if expansion else "none"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(r.get("scenario", "")),
                        str(r.get("risk_level", "?")),
                        f"{float(r.get('risk_score', 0.0)):.1f}",
                        ", ".join(r.get("risky_applied", [])) or "none",
                        expansion_summary,
                        r.get("judge_verdict", ""),
                    ]
                )
                + " |"
            )
        rendered = True

    return rendered


def render_markdown(report: ProductionEvalReport) -> str:
    """Render a production eval bundle as Markdown."""
    payload = report.to_dict()
    lines = [
        "# LATTICE Production Evals",
        "",
        f"- Runner: `{payload['runner']}`",
        f"- Total sections: `{payload['summary']['total_sections']}`",
        f"- Passed sections: `{payload['summary']['passed_sections']}`",
        "",
    ]
    for section in payload["sections"]:
        lines.extend(
            [
                f"## {section['name']}",
                "",
                f"- Kind: `{section['kind']}`",
                f"- Status: `{section['status']}`",
            ]
        )
        rendered_special_details = False
        if section.get("summary"):
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("| --- | ---: |")
            for key, value in section["summary"].items():
                lines.append(f"| {key} | {value} |")
        benchmark = section.get("benchmark")
        if benchmark:
            summary = benchmark["summary"]
            proof_passed = section.get("summary", {}).get("proof_passed")
            proof_failed = section.get("summary", {}).get("proof_failed")
            usage_rows: list[dict[str, Any]] = []
            response_rows: list[dict[str, Any]] = []
            prompt_rows: list[dict[str, Any]] = []
            lines.extend(
                [
                    "",
                    f"- Benchmark provider: `{benchmark['provider']}`",
                    f"- Benchmark model: `{benchmark['model']}`",
                    f"- Scenarios: `{summary['total_scenarios']}`",
                    f"- Quality passed: `{summary['passed_scenarios']}`",
                    f"- Total token savings: `{summary['total_token_savings']}`",
                    f"- Avg reduction ratio: `{summary['avg_reduction_ratio']:.4f}`",
                    f"- Avg pipeline latency: `{summary['avg_pipeline_latency_ms']:.4f} ms`",
                    f"- Avg quality score: `{summary['avg_quality_score']:.4f}`",
                    "",
                    "| Scenario | Category | Saved | Ratio | Task Eq. |",
                    "| --- | --- | ---: | ---: | ---: |",
                ]
            )
            if proof_passed is not None or proof_failed is not None:
                lines.extend(
                    [
                        "",
                        "| Proof Metric | Value |",
                        "| --- | ---: |",
                    ]
                )
                if proof_passed is not None:
                    lines.append(f"| proof_passed | {proof_passed} |")
                if proof_failed is not None:
                    lines.append(f"| proof_failed | {proof_failed} |")
            for scenario in benchmark["scenarios"]:
                saved = scenario["tokens"]["savings"]["saved"]
                ratio = scenario["tokens"]["savings"]["ratio"]
                # Task-equivalence is the source of truth; semantic similarity is legacy
                te = scenario["quality"].get("task_equivalence", {})
                if te:
                    quality = te.get("composite", scenario["quality"]["semantic_similarity"])
                else:
                    quality = scenario["quality"]["semantic_similarity"]
                lines.append(
                    f"| {scenario['scenario']} | {scenario['category']} | {saved} | {ratio:.4f} | {quality:.4f} |"
                )
                usage = scenario.get("usage") or {}
                costs = usage.get("costs") or {}
                baseline_usage = usage.get("baseline") or {}
                optimized_usage = usage.get("optimized") or {}
                samples = scenario.get("response_samples") or {}
                prompt_meta = scenario.get("telemetry", {}).get("prompt") or {}
                if costs and (baseline_usage or optimized_usage):
                    usage_rows.append(
                        {
                            "scenario": scenario["scenario"],
                            "baseline_prompt_tokens": baseline_usage.get("prompt_tokens", 0),
                            "optimized_prompt_tokens": optimized_usage.get("prompt_tokens", 0),
                            "baseline_cost_usd": costs.get("baseline_total_usd", 0.0),
                            "optimized_cost_usd": costs.get("optimized_total_usd", 0.0),
                            "delta_usd": costs.get("delta_usd", 0.0),
                        }
                    )
                if samples:
                    response_rows.append(
                        {
                            "scenario": scenario["scenario"],
                            "baseline": _excerpt(samples.get("baseline", "")),
                            "optimized": _excerpt(samples.get("optimized", "")),
                        }
                    )
                if prompt_meta:
                    safety = prompt_meta.get("safety_profile") or {}
                    prompt_rows.append(
                        {
                            "scenario": scenario["scenario"],
                            "structured": str(bool(safety.get("has_structured_content", False))),
                            "code": str(bool(safety.get("has_code_blocks", False))),
                            "strict": str(bool(safety.get("has_strict_instructions", False))),
                            "tool_calls": str(bool(safety.get("has_tool_calls", False))),
                            "high_stakes": str(bool(safety.get("has_high_stakes_entities", False))),
                            "long_form": str(bool(safety.get("long_form", False))),
                            "lossy_ok": str(bool(prompt_meta.get("lossy_transform_allowed", False))),
                        }
                    )

            details = {
                **(section.get("details") or {}),
                **(benchmark.get("details") or {}),
            }
            rendered_special_details = _render_special_details(lines, details)

            if usage_rows:
                lines.extend(
                    [
                        "",
                        "### Usage",
                        "",
                        "| Scenario | Baseline Prompt Tokens | Optimized Prompt Tokens | Baseline Cost USD | Optimized Cost USD | Delta USD |",
                        "| --- | ---: | ---: | ---: | ---: | ---: |",
                    ]
                )
                for row in usage_rows:
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                str(row["scenario"]),
                                str(row["baseline_prompt_tokens"]),
                                str(row["optimized_prompt_tokens"]),
                                f"{float(row['baseline_cost_usd']):.6f}",
                                f"{float(row['optimized_cost_usd']):.6f}",
                                f"{float(row['delta_usd']):.6f}",
                            ]
                        )
                        + " |"
                    )
                rendered_special_details = True

            if response_rows:
                lines.extend(
                    [
                        "",
                        "### Response Samples",
                        "",
                        "| Scenario | Baseline | Optimized |",
                        "| --- | --- | --- |",
                    ]
                )
                for row in response_rows:
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                str(row["scenario"]),
                                row["baseline"] or "-",
                                row["optimized"] or "-",
                            ]
                        )
                        + " |"
                    )
                rendered_special_details = True

            if prompt_rows:
                lines.extend(
                    [
                        "",
                        "### Prompt Safety",
                        "",
                        "| Scenario | Structured | Code | Strict | Tool Calls | High Stakes | Long Form | Lossy OK |",
                        "| --- | --- | --- | --- | --- | --- | --- | --- |",
                    ]
                )
                for row in prompt_rows:
                    lines.append(
                        "| "
                        + " | ".join(
                            [
                                str(row["scenario"]),
                                row["structured"],
                                row["code"],
                                row["strict"],
                                row["tool_calls"],
                                row["high_stakes"],
                                row["long_form"],
                                row["lossy_ok"],
                            ]
                        )
                        + " |"
                    )
                rendered_special_details = True

            if details and not rendered_special_details:
                lines.append("")
                lines.append("```json")
                import json

                lines.append(json.dumps(details, indent=2, sort_keys=True))
                lines.append("```")
        elif section.get("details"):
            details = section["details"] or {}
            if not _render_special_details(lines, details):
                lines.append("")
                lines.append("```json")
                import json

                lines.append(json.dumps(details, indent=2, sort_keys=True))
                lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
