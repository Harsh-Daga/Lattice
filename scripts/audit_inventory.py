"""Generate docs/refactor/inventory.csv — file-by-file map of src/lattice/.

For every *.py file in src/lattice/, record:

    path                  : repo-relative path
    loc                   : line count (raw `wc -l`)
    top_level_classes     : count of `^class ` lines
    top_level_funcs       : count of `^def ` lines
    imports_from_lattice  : pipe-separated unique `from lattice.X` imports
    imported_by_count     : count of *.py files in src/ + tests/ that import this module
    phase_target          : phase number (from REFACTOR_PLAN.md) where this file is touched
    final_path            : target location (from FINAL_LAYOUT.md)
    disposition           : KEEP | MOVE | SPLIT | DELETE | RENAME

This is the canonical lookup table referenced by every later refactor phase.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "lattice"
TESTS_ROOT = REPO_ROOT / "tests"
OUTPUT = REPO_ROOT / "docs" / "refactor" / "inventory.csv"

# ----------------------------------------------------------------------------
# Disposition table — sourced from docs/refactor/FINAL_LAYOUT.md and the
# per-phase docs. Key is the path relative to src/lattice/.
# Value is (phase, final_path, disposition).
# ----------------------------------------------------------------------------
DISPOSITION: dict[str, tuple[str, str, str]] = {
    # ---- Phase 1: IR primitives ----
    "core/ir.py":              ("1", "ir/types.py",              "MOVE"),
    "core/ir_builder.py":      ("1", "ir/builder.py",            "MOVE"),
    "core/ir_normalizer.py":   ("1", "ir/normalizer.py",         "MOVE"),
    "core/ir_serializer.py":   ("1", "ir/serializer.py",         "MOVE"),
    "core/ir_transform.py":    ("1", "ir/transform.py",          "MOVE"),
    "core/primitives.py":      ("1", "ir/primitives.py",         "MOVE"),
    "core/semantic_graph.py":  ("1", "ir/semantic_graph.py",     "MOVE"),
    "core/compiler.py":        ("1", "(inlined into ir/builder.py)", "DELETE"),
    "transforms/semantic_segmenter.py": ("1", "core/segmentation.py", "MOVE"),
    "optimizer/ir_native_optimizer.py": ("1", "ir/native_optimizer.py", "MOVE"),
    "optimizer/validation.py":          ("1", "ir/validation.py",        "MOVE"),
    "optimizer/quality_estimator.py":   ("1", "ir/quality.py",           "MOVE"),

    # ---- Phase 2: Pipeline runner ----
    "core/pipeline.py":              ("2", "(deleted — v1)",               "DELETE"),
    "core/pipeline_v2_wrapper.py":   ("2", "(deleted — wrapper)",          "DELETE"),
    "core/pipeline_v2.py":           ("2", "pipeline/runner.py",           "RENAME"),
    "core/pipeline_factory.py":      ("2", "pipeline/factory.py",          "MOVE"),
    "core/policy.py":                ("2", "pipeline/policy.py",           "MOVE"),
    "core/guardrails.py":            ("2", "pipeline/guardrails.py",       "MOVE"),
    "core/milv.py":                  ("2", "pipeline/milv.py",             "MOVE"),
    "core/auto_continuation.py":     ("2", "pipeline/auto_continuation.py","MOVE"),
    "core/batch_accumulator.py":     ("2", "pipeline/batch_accumulator.py","MOVE"),
    "optimizer/representation_optimizer.py": ("2", "pipeline/representation_optimizer.py", "MOVE"),
    "core/transport.py":             ("2", "transport/types.py",           "MOVE"),
    "core/serialization.py":         ("2", "transport/serialization.py",   "MOVE"),
    "core/delta_wire.py":            ("2", "transport/delta_wire.py",      "MOVE"),

    # ---- Phase 4: Planner collapse (refactor numbering) ----
    "core/scheduler.py":             ("4", "(deleted)",                    "DELETE"),
    "core/optimizer_scheduler.py":   ("4", "(deleted)",                    "DELETE"),
    "core/unified_planner.py":       ("4", "planner/unified_planner.py",   "MOVE"),
    "core/task_classifier.py":       ("4", "planner/task_classifier.py",   "MOVE"),
    "core/runtime_state.py":         ("4", "planner/runtime_state.py",     "MOVE"),
    "core/credentials.py":           ("4", "providers/credentials.py",     "MOVE"),
    "optimizer/__init__.py":         ("4", "(deleted)",                    "DELETE"),
    "optimizer/structure_optimizer.py":   ("4", "(deleted — text-based)", "DELETE"),
    "optimizer/_dispatch.py":        ("4", "transforms/optimizers/_dispatch.py", "MOVE"),
    "optimizer/ir_structure_optimizer.py":("4", "transforms/optimizers/ir_structure_optimizer.py", "MOVE"),
    "optimizer/reference_optimizer.py":   ("4", "transforms/optimizers/reference_optimizer.py",   "MOVE"),
    "optimizer/tool_optimizer.py":        ("4", "transforms/optimizers/tool_optimizer.py",        "MOVE"),
    "optimizer/diagnostic_optimizer.py":  ("4", "transforms/optimizers/diagnostic_optimizer.py",  "MOVE"),
    "optimizer/context_optimizer.py":     ("4", "transforms/optimizers/context_optimizer.py",    "MOVE"),
    "runtime/router.py":             ("4", "runtime/tier_classifier.py",   "RENAME"),

    # ---- Phase 5: Transforms (refactor numbering) ----
    "core/transform_registry.py":    ("5", "transforms/registry.py",      "MOVE"),
    "core/transform_reputation.py":  ("5", "transforms/reputation.py",    "MOVE"),
    "utils/patterns.py":             ("5", "transforms/patterns.py",      "MOVE"),
    "transforms/content_profiler.py":("5", "transforms/content_profiler/", "SPLIT"),
    "transforms/format_conv.py":     ("5", "transforms/format_converter/", "SPLIT"),
    "transforms/prefix_opt.py":      ("5", "(deleted — deprecated)",     "DELETE"),
    "transforms/constraint_lifting.py": ("5", "(deleted if no consumer)", "DELETE"),
    "transforms/strategy_selector.py":  ("5", "transforms/strategy_selector/ (gated)", "SPLIT"),
    "transforms/context_selector.py":   ("5", "transforms/context_selector.py (submodular-only)", "KEEP"),

    # ---- Phase 6: Providers & transport ----
    "providers/openai.py":          ("6", "providers/adapters/openai.py + openai_compatible.py", "SPLIT"),
    "providers/anthropic.py":       ("6", "providers/adapters/anthropic.py", "MOVE"),
    "providers/azure.py":           ("6", "providers/adapters/azure.py",     "MOVE"),
    "providers/bedrock.py":         ("6", "providers/adapters/bedrock.py",   "MOVE"),
    "providers/gemini.py":          ("6", "providers/adapters/gemini.py",    "MOVE"),
    "providers/ollama.py":          ("6", "providers/adapters/ollama.py",    "MOVE"),
    "providers/stall_detector.py":  ("6", "providers/transport/stall_detector.py", "MOVE"),
    "providers/transport.py":       ("6", "providers/transport/ (7-file pkg)",     "SPLIT"),
    "providers/credentials.py":     ("4", "providers/credentials.py",     "KEEP"),

    # ---- Phase 7: Proxy/SDK/CLI ----
    "proxy/compat_exports.py":      ("7", "(deleted — already removed)",   "DELETE"),
    "sdk/client.py":                ("7", "sdk/client.py (deprecation shim)", "MODIFY"),

    # ---- Phase 8: Integrations ----
    "core/tunnel_sidecar.py":       ("8", "integrations/tunnel.py",        "MOVE"),

    # ---- Phase 9: Observability / state ----
    "core/metrics.py":              ("9", "telemetry/metrics.py",          "MOVE"),
    "core/telemetry.py":            ("9", "telemetry/downgrade.py",        "RENAME"),
    "core/agent_stats.py":          ("9", "telemetry/agent_stats.py",      "MOVE"),
    "core/cost_estimator.py":       ("9", "telemetry/cost_estimator.py",   "MOVE"),
    "core/maintenance.py":          ("9", "telemetry/maintenance.py",      "MOVE"),
    "utils/streaming_sketches.py":  ("9", "telemetry/streaming_sketches.py", "MOVE"),
    "core/session.py":              ("9", "transport/session.py",          "MOVE"),
    "core/store.py":                ("9", "state/store.py",                "MOVE"),
    "core/semantic_cache.py":       ("9", "cache/semantic.py",             "MOVE"),
    "utils/validation.py":          ("9", "safety/risk_scoring.py",        "MOVE"),

    # ---- Phase 10: Benchmarks/evals ----
    "evals/__init__.py":            ("10", "(deleted — empty placeholder)", "DELETE"),
}


def loc(path: Path) -> int:
    try:
        return sum(1 for _ in path.open("r", errors="replace"))
    except OSError:
        return 0


def count_pattern(path: Path, pat: re.Pattern[str]) -> int:
    try:
        return sum(1 for line in path.read_text(errors="replace").splitlines() if pat.match(line))
    except OSError:
        return 0


def lattice_imports(path: Path) -> list[str]:
    out: set[str] = set()
    try:
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            m = re.match(r"^(?:from|import)\s+(lattice[\w.]*)", line)
            if m:
                out.add(m.group(1))
    except OSError:
        pass
    return sorted(out)


def imported_by_count(module: str) -> int:
    """Count how many *.py files reference this module."""
    if not module:
        return 0
    try:
        # Use grep -l for speed; fall back to 0 on failure.
        result = subprocess.run(
            ["grep", "-rl", "-E", f"(from|import) +{re.escape(module)}([. ]|$)", "src/", "tests/", "benchmarks/", "scripts/"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return len([line for line in result.stdout.splitlines() if line.strip()])
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0


def path_to_module(rel: str) -> str:
    if not rel.endswith(".py"):
        return ""
    stem = rel[:-3].replace("/", ".")
    return f"lattice.{stem}"


def disposition_for(rel: str) -> tuple[str, str, str]:
    if rel in DISPOSITION:
        return DISPOSITION[rel]
    return ("-", rel, "KEEP")


def main() -> int:
    if not SRC_ROOT.is_dir():
        print(f"ERROR: {SRC_ROOT} not found.", file=sys.stderr)
        return 1

    files = sorted(p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts)
    rows: list[dict[str, object]] = []

    class_re = re.compile(r"^class\s")
    def_re = re.compile(r"^def\s")

    for path in files:
        rel = path.relative_to(SRC_ROOT).as_posix()
        module = path_to_module(rel)
        phase, final_path, disposition = disposition_for(rel)
        rows.append({
            "path": f"src/lattice/{rel}",
            "loc": loc(path),
            "top_level_classes": count_pattern(path, class_re),
            "top_level_funcs": count_pattern(path, def_re),
            "imports_from_lattice": "|".join(lattice_imports(path)),
            "imported_by_count": imported_by_count(module),
            "phase_target": phase,
            "final_path": final_path,
            "disposition": disposition,
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "path", "loc", "top_level_classes", "top_level_funcs",
                "imports_from_lattice", "imported_by_count",
                "phase_target", "final_path", "disposition",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    moves = sum(1 for r in rows if r["disposition"] == "MOVE")
    splits = sum(1 for r in rows if r["disposition"] == "SPLIT")
    renames = sum(1 for r in rows if r["disposition"] == "RENAME")
    deletes = sum(1 for r in rows if r["disposition"] == "DELETE")
    keeps = sum(1 for r in rows if r["disposition"] == "KEEP")
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)} ({total} rows)")
    print(f"  MOVE={moves}  SPLIT={splits}  RENAME={renames}  DELETE={deletes}  KEEP={keeps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
