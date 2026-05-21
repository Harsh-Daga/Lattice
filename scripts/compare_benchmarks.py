"""Compare two benchmark JSON outputs and exit non-zero on regression.

Used by the LATTICE refactor CI gate (REFACTOR_PLAN.md §7, step 5).

Usage:
    python scripts/compare_benchmarks.py BASELINE.json TARGET.json --tolerance-pct N

The script walks every numeric metric in the baseline's section summaries
and compares it against the same metric in the target. For metrics where
"higher is better" (quality, reduction ratio, cache hit) a negative delta
exceeding tolerance is a regression. For metrics where "lower is better"
(latency, cost, failure counts) a positive delta exceeding tolerance is
a regression. Metrics with unknown polarity are reported but do not block.

The JSON shape produced by benchmarks/evals/runner.py is:

    {
      "runner": "...",
      "timestamp": "...",
      "summary": {...},
      "config": {...},
      "sections": [
        {"name": "...", "kind": "...", "status": "...",
         "summary": {<metric_name>: <number>, ...},
         "details": {...}, "benchmark": {...}},
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# Polarity classification: higher_is_better vs lower_is_better.
# Anything not listed here is treated as "informational" — reported but
# not gated. Substring match on the metric name (lowercase).
HIGHER_IS_BETTER = (
    "quality",
    "reduction_ratio",
    "compression_pct",
    "cache_hit_rate",
    "passed",
    "proof_passed",
    "throughput",
    "score",
    "savings",
)
LOWER_IS_BETTER = (
    "latency",
    "latency_ms",
    "latency_p50_ms",
    "latency_p99_ms",
    "cost_usd",
    "cost",
    "failed",
    "feature_failed",
    "tier_failed",
    "budget_failed",
    "proof_failed",
    "error",
    "rollback",
)


def classify(metric: str) -> str:
    name = metric.lower()
    for needle in LOWER_IS_BETTER:
        if needle in name:
            return "lower_is_better"
    for needle in HIGHER_IS_BETTER:
        if needle in name:
            return "higher_is_better"
    return "informational"


def flatten_sections(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Return {section_name: {metric: value}} from a benchmark JSON."""
    out: dict[str, dict[str, float]] = {}
    for section in report.get("sections", []) or []:
        name = section.get("name") or "<unnamed>"
        summary = section.get("summary") or {}
        flat: dict[str, float] = {}
        for k, v in summary.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if math.isfinite(float(v)):
                    flat[k] = float(v)
        if flat:
            out[name] = flat
    return out


def pct_delta(baseline: float, target: float) -> float:
    if baseline == 0:
        return 0.0 if target == 0 else math.inf
    return (target - baseline) / abs(baseline) * 100.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline", type=Path, help="Baseline benchmark JSON")
    p.add_argument("target", type=Path, help="Target benchmark JSON")
    p.add_argument(
        "--tolerance-pct",
        type=float,
        default=2.0,
        help="Allowed deviation per metric (default 2%%).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print informational deltas too (lines starting with ~).",
    )
    args = p.parse_args()

    baseline = json.loads(args.baseline.read_text())
    target = json.loads(args.target.read_text())

    bsec = flatten_sections(baseline)
    tsec = flatten_sections(target)

    regressions: list[str] = []
    improvements: list[str] = []
    informational: list[str] = []

    common = sorted(set(bsec) & set(tsec))
    only_baseline = sorted(set(bsec) - set(tsec))
    only_target = sorted(set(tsec) - set(bsec))

    for section in common:
        bm = bsec[section]
        tm = tsec[section]
        for metric in sorted(set(bm) & set(tm)):
            bv, tv = bm[metric], tm[metric]
            delta = pct_delta(bv, tv)
            polarity = classify(metric)
            line = f"  {section}.{metric}: {bv:.4f} -> {tv:.4f} ({delta:+.2f}%)"
            if polarity == "higher_is_better":
                if delta < -args.tolerance_pct:
                    regressions.append("- " + line + " [REGRESSION ↓]")
                elif delta > args.tolerance_pct:
                    improvements.append("+ " + line + " [IMPROVED ↑]")
                else:
                    informational.append("= " + line)
            elif polarity == "lower_is_better":
                if delta > args.tolerance_pct:
                    regressions.append("- " + line + " [REGRESSION ↑]")
                elif delta < -args.tolerance_pct:
                    improvements.append("+ " + line + " [IMPROVED ↓]")
                else:
                    informational.append("= " + line)
            else:
                informational.append("~ " + line)

    print(f"Comparing baseline={args.baseline} target={args.target}")
    print(f"tolerance: ±{args.tolerance_pct:.2f}%")
    print(f"sections compared: {len(common)} (baseline-only={len(only_baseline)}, target-only={len(only_target)})")
    print()

    if regressions:
        print(f"REGRESSIONS ({len(regressions)}):")
        for r in regressions:
            print(r)
        print()
    if improvements:
        print(f"IMPROVEMENTS ({len(improvements)}):")
        for i in improvements:
            print(i)
        print()
    if args.verbose and informational:
        print(f"WITHIN TOLERANCE ({len(informational)}):")
        for i in informational:
            print(i)
        print()
    if only_baseline:
        print(f"Sections only in baseline (lost in target): {only_baseline}")
    if only_target:
        print(f"Sections only in target (added since baseline): {only_target}")

    if regressions:
        print(f"\nFAIL: {len(regressions)} metric(s) regressed beyond ±{args.tolerance_pct:.2f}%")
        return 1
    print("\nPASS: no regression beyond tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
