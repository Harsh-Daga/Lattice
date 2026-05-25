# Phase 10 — Benchmarks & Evals (STATUS Phase 10; REFACTOR_PLAN historical Phase 9)

> **STATUS:** Phase 10 on `main` per `09-benchmarks.md` acceptance (wrapper, CLAIMS, v1.0.0 artifacts).

> **Goal.** Delete the dead `src/lattice/evals/` directory; make `benchmarks/` the single canonical evaluation system; produce a `benchmarks/results/v1.0.0.json` reference run; ensure every claim made in `README.md`, `AGENTS.md`, and `docs/` is backed by a measured number in `benchmarks/results/`; add a `lattice benchmark` CLI subcommand that wraps `benchmarks/evals/cli.py` (currently the subcommand just prints a redirect message); remove the now-meaningless `--use-v2-pipeline` flag in `benchmarks/evals/cli.py` (after Phase 2 there's only one pipeline).
>
> **Outcome.** Single benchmark CLI. Single results directory. A `benchmarks/results/CLAIMS.md` table maps every public claim to the JSON it came from. `lattice benchmark` is no longer a stub.
>
> **Estimated effort.** 1 day.

---

## 1. Why this phase exists

The audit found:

1. **`src/lattice/evals/` is an empty placeholder** — only `__pycache__` lives there. The actual eval system is `benchmarks/evals/`. The dead directory is import-confusing.
2. **`benchmarks/evals/cli.py` has `--use-v2-pipeline` flag.** After Phase 2 there is no v1 pipeline. The flag is a no-op accepted-for-back-compat. Remove it.
3. **`README.md` cites figures with no traceability** — "1584 tests passed", "18 transforms", "17 providers", "30-60% per-request overhead reduction", "20-50% on structured/data-heavy workloads", "15-25% improvement". Most have no JSON-backed source. Phase 9 produces a CLAIMS.md mapping each claim → benchmark run + JSON file → date.
4. **`lattice benchmark` CLI subcommand is a 5-line stub** that prints "Benchmarking has moved to benchmarks/evals/cli.py". Make it a real wrapper.
5. **Reproducibility hole**: contributors don't all have `ollama-cloud / kimi-k2.6:cloud` available. Phase 0 baseline locks the canonical provider; Phase 10 adds `--provider-detect` on `benchmarks/evals/cli.py` to pick the first credentialed provider from a preference list.

---

## 2. Files touched

### 2.1 Deleted

```
src/lattice/evals/                   # entire dead directory
```

### 2.2 Created

```
benchmarks/results/CLAIMS.md         # claim → JSON map
benchmarks/results/v1.0.0.json       # the v1.0.0 release reference run
benchmarks/results/v1.0.0.md         # rendered version
scripts/run_canonical_benchmark.sh   # one-line invocation used by CI for refactor-gate.yml
```

### 2.3 Modified

- `src/lattice/cli.py` — make `_cmd_benchmark` invoke `benchmarks/evals/cli.py` with the given args (subprocess), passing through `--suite` and other flags.
- `benchmarks/evals/cli.py` — remove `--use-v2-pipeline` flag (deprecation warning if set; no-op otherwise — actually: just remove silently, document in CHANGELOG).
- `benchmarks/evals/runner.py` — remove all `if use_v2_pipeline:` branches; v2 is the only path now.
- `README.md` — replace every uncited figure with a citation to a row in `benchmarks/results/CLAIMS.md`. (Phase 11 owns the full README rewrite; Phase 9 just produces the data and the table.)

---

## 3. Step-by-step

### 3.1 Delete `src/lattice/evals/`

```bash
rg "from lattice.evals|import lattice.evals" src/ tests/ benchmarks/
# Should return 0 matches. Delete:
git rm -r src/lattice/evals/
```

### 3.2 Remove `--use-v2-pipeline` from benchmark CLI

In `benchmarks/evals/cli.py`, find:

```python
parser.add_argument("--use-v2-pipeline", action="store_true", ...)
```

and:

```python
if args.use_v2_pipeline:
    # special v2 setup
    ...
```

Delete both. The runner unconditionally uses the new (only) pipeline. Audit `runner.py`:

```bash
rg "use_v2_pipeline" benchmarks/
```

Each match: remove the `if` branch; keep the body that was inside the `if`.

### 3.3 Make `lattice benchmark` a real wrapper

In `src/lattice/cli.py`:

```python
def _cmd_benchmark(args: list[str]) -> None:
    """Run the LATTICE benchmark suite.

    Wraps benchmarks/evals/cli.py. All args pass through.
    Common invocations:
        lattice benchmark
        lattice benchmark --suite feature
        lattice benchmark --suite all --providers ollama-cloud --provider-model ollama-cloud=kimi-k2.6:cloud
    """
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    # Find the benchmark CLI relative to the installed package
    repo_root = Path(__file__).resolve().parent.parent.parent
    bench_cli = repo_root / "benchmarks" / "evals" / "cli.py"
    if not bench_cli.exists():
        console.print("[red]benchmarks/ not packaged with the install.[/red]")
        console.print("Run from a source checkout, or install with `pip install -e '.[dev]'`.")
        sys.exit(1)

    python = sys.executable
    result = subprocess.run([python, str(bench_cli)] + args, check=False)
    sys.exit(result.returncode)
```

### 3.4 Produce v1.0.0 reference run

```bash
uv run python benchmarks/evals/cli.py --suite all \
    --providers ollama-cloud --provider-model ollama-cloud=kimi-k2.6:cloud \
    --iterations 3 --warmup 1 --provider-warmup 1 \
    --output-json benchmarks/results/v1.0.0.json \
    --output-md   benchmarks/results/v1.0.0.md
```

Note: 3 iterations + 1 warmup for the **release** run (vs. 1+0 for phase comparison runs) for tighter confidence intervals on the headline numbers.

### 3.5 Write `benchmarks/results/CLAIMS.md`

A table mapping every public claim (README, AGENTS, docs) to its source. Template:

```markdown
# LATTICE v1.0.0 — Claim Traceability

Every figure in the README, AGENTS.md, and docs/ must appear in this table.
If a claim is here, it has a JSON source under benchmarks/results/.
If a claim isn't here, it must be removed from public docs before v1.0.0 release.

| Claim | Source | JSON | Run date | Notes |
|---|---|---|---|---|
| "1623 tests passed" | `tests/contract/`, `tests/unit/`, `tests/integration/`, `tests/e2e/`, `tests/security/` | n/a (test count) | derived from `pytest --collect-only` | Phase 0 captured the actual number |
| "18 transforms" | Transform registry | `benchmarks/results/v1.0.0.json#transforms.count` | release | derived; matches `from lattice.transforms.registry import list_default_pipeline_names; list_transform_names(); len(...)` |
| "17 providers" | `ProviderRegistry` | derived from `from lattice.providers import ProviderRegistry; ProviderRegistry().adapters` | n/a | structural |
| "TACC AIMD-style adaptive concurrency" | `transport/congestion.py` | `benchmarks/results/v1.0.0.json#tacc.*` | release | qualitative; see `docs/novel/tacc.md` for behavior |
| "Binary framing — 15-byte fixed header, 17 frame types" | `protocol/framing.py` | n/a (structural) | n/a | code-derived |
| "Delta encoding — after turn 1, sends only new messages" | `transport/delta_wire.py` | `benchmarks/results/v1.0.0.json#delta.bytes_saved_pct` | release | measured savings on multi-turn scenarios |
| "30-60% per-request overhead reduction (batching)" | `benchmarks/results/v1.0.0.json#batching.overhead_reduction_pct` | `v1.0.0.json` | release | Batching scenarios only; documented per-scenario range |
| "20-40% redundant content in long conversations (message_dedup)" | `benchmarks/results/v1.0.0.json#scenarios.long_conversation.redundancy_pct` | `v1.0.0.json` | release | from `--suite feature` row |
| "20-50% on structured/data-heavy workloads (reference_sub)" | `benchmarks/results/v1.0.0.json#scenarios.structured.compression_pct` | `v1.0.0.json` | release | from `--suite feature-matrix` row |
| "Compression % default suite" | `benchmarks/results/v1.0.0.md` | `v1.0.0.json` | release | headline metric |
| ... | ... | ... | ... | ... |
```

The CLAIMS.md serves as the **acceptance test** for what the README is allowed to say. The Phase 11 README rewrite cites only values in this file.

### 3.6 Write `scripts/run_canonical_benchmark.sh`

```bash
#!/usr/bin/env bash
# Canonical benchmark run for refactor-gate CI.
# Usage: ./scripts/run_canonical_benchmark.sh <output-json>
set -euo pipefail
OUT="${1:-benchmarks/results/local.json}"
exec uv run python benchmarks/evals/cli.py --suite all \
    --providers ollama-cloud --provider-model ollama-cloud=kimi-k2.6:cloud \
    --iterations 1 --warmup 0 --provider-warmup 0 \
    --output-json "$OUT"
```

```bash
chmod +x scripts/run_canonical_benchmark.sh
```

Used by `.github/workflows/refactor-gate.yml` from Phase 0.

### 3.7 Verify

```bash
rg "from lattice.evals|import lattice.evals" src/ tests/ benchmarks/   # 0 matches
rg "use_v2_pipeline" benchmarks/ src/ tests/                            # 0 matches

uv run lattice benchmark --help              # prints benchmark CLI help
uv run lattice benchmark --suite feature     # runs the feature suite

# Reference run
./scripts/run_canonical_benchmark.sh /tmp/local.json
python scripts/compare_benchmarks.py benchmarks/results/v1.0.0.json /tmp/local.json --tolerance-pct 5
# Larger tolerance OK for re-runs on different days
```

---

## 4. Per-file disposition

| File | LoC | Action |
|---|---|---|
| `src/lattice/evals/` (entire directory) | — | **DELETED** |
| `benchmarks/evals/cli.py` | 305 | MODIFY — remove `--use-v2-pipeline` |
| `benchmarks/evals/runner.py` | 1860 | MODIFY — remove `use_v2_pipeline` branches (~30 LoC across ~10 sites) |
| `benchmarks/evals/surfaces.py` | 575 | UNCHANGED |
| `benchmarks/evals/live.py` | 323 | UNCHANGED |
| `benchmarks/evals/replay.py` | 688 | UNCHANGED |
| `benchmarks/evals/report.py` | 539 | UNCHANGED |
| `benchmarks/evals/catalog.py` | 153 | UNCHANGED |
| `benchmarks/framework/types.py` | 462 | UNCHANGED |
| `benchmarks/framework/frontier.py` | 77 | UNCHANGED |
| `benchmarks/metrics/quality.py` | 221 | UNCHANGED |
| `benchmarks/scenarios/prompts.py` | 825 | UNCHANGED |
| `benchmarks/datasets/replay_traces.jsonl` | — | UNCHANGED |
| `benchmarks/results/CLAIMS.md` | NEW | CREATE |
| `benchmarks/results/v1.0.0.json` | NEW | CREATE (output of canonical run) |
| `benchmarks/results/v1.0.0.md` | NEW | CREATE (rendered) |
| `scripts/run_canonical_benchmark.sh` | NEW (~10) | CREATE |
| `scripts/compare_benchmarks.py` | (from Phase 0) | UNCHANGED |
| `scripts/benchmark_compression.py` | 122 | UNCHANGED — useful for ad-hoc dev runs |
| `scripts/benchmark_e2e_through_proxy.py` | 152 | UNCHANGED |
| `scripts/profile_format_conv.py` | 82 | **AUDIT** — if not used since Phase 0, delete. If kept, add to scripts/README.md. |
| `scripts/test_e2e_real.py` | 122 | **AUDIT** — same logic. |
| `src/lattice/cli.py:_cmd_benchmark` | ~5 | MODIFY — real wrapper around benchmarks CLI |

### Audit decision for `scripts/profile_format_conv.py` and `scripts/test_e2e_real.py`

```bash
# Has either been modified in the last 90 days?
git log --since="90 days ago" --name-only -- scripts/profile_format_conv.py scripts/test_e2e_real.py | sort -u
```

If both show no commits in 90 days AND no benchmarks/results reference their output, delete:

```bash
git rm scripts/profile_format_conv.py
git rm scripts/test_e2e_real.py
```

Otherwise keep, add a one-liner header documenting their purpose.

---

## 5. Symbol migration table

| Old | New |
|---|---|
| `lattice.evals.*` | **DELETED** — use `benchmarks/evals/` |
| `benchmarks.evals.cli --use-v2-pipeline` | flag removed; no replacement (v2 is only path) |
| `lattice benchmark` (CLI) | now runs `benchmarks/evals/cli.py` with passed args |

---

## 6. Tests

### 6.1 New tests

**`tests/unit/cli/test_benchmark_wrapper.py`**:

```python
import subprocess

def test_benchmark_invokes_real_cli():
    """`lattice benchmark --help` should print the underlying benchmark CLI's help,
    not the old 'Benchmarking has moved' redirect."""
    result = subprocess.run(["lattice", "benchmark", "--help"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    # Old behavior emitted: "Benchmarking has moved to benchmarks/evals/cli.py"
    assert "has moved" not in result.stdout
    # New behavior runs the real CLI:
    assert "--suite" in result.stdout

def test_benchmark_no_v2_flag():
    """`--use-v2-pipeline` flag must not exist."""
    result = subprocess.run(["lattice", "benchmark", "--use-v2-pipeline"],
                            capture_output=True, text=True, timeout=10)
    # argparse exits with code 2 when an unknown arg is given
    assert result.returncode != 0
    assert "unrecognized" in result.stderr.lower() or "unknown" in result.stderr.lower()
```

**`tests/unit/test_no_lattice_evals.py`**:

```python
import pytest

def test_lattice_evals_deleted():
    with pytest.raises(ImportError):
        import lattice.evals   # noqa: F401
```

### 6.2 Contract additions

`tests/contract/test_cli_contract.py` (Phase 0) — extend to:

```python
def test_lattice_benchmark_runs():
    """`lattice benchmark --suite feature` must complete (or print clear failure if no providers)."""
    result = subprocess.run(
        ["lattice", "benchmark", "--suite", "feature",
         "--providers", "ollama-cloud",
         "--provider-model", "ollama-cloud=kimi-k2.6:cloud",
         "--iterations", "1", "--warmup", "0"],
        capture_output=True, text=True, timeout=120,
    )
    # Even if provider is unavailable, the CLI itself must exit cleanly
    # (the benchmark may report failures but the CLI returns 0 unless arg parse fails)
    assert result.returncode in (0, 1)   # 0 = success, 1 = benchmark scenario failed (acceptable in CI without keys)
    assert "Traceback" not in result.stderr
```

This is the gate that ensures the `lattice benchmark` wrapper is wired correctly.

---

## 7. Acceptance criteria

- [x] `src/lattice/evals/` directory does not exist.
- [x] `rg "from lattice.evals|import lattice.evals" src/ tests/ benchmarks/` returns 0 matches.
- [x] `--use-v2-pipeline` flag does not exist in `benchmarks/evals/cli.py`.
- [x] `rg "use_v2_pipeline" src/ tests/ benchmarks/` returns 0 matches.
- [x] `lattice benchmark --help` runs the real benchmarks CLI.
- [x] `lattice benchmark --suite feature` runs to completion (locally; CI may skip if no provider).
- [x] `benchmarks/results/v1.0.0.json` and `v1.0.0.md` exist with the canonical release run.
- [x] `benchmarks/results/CLAIMS.md` exists and includes one row per public-doc claim.
- [x] `scripts/run_canonical_benchmark.sh` exists, is executable, and is referenced by `.github/workflows/refactor-gate.yml`.
- [x] `scripts/profile_format_conv.py` and `scripts/test_e2e_real.py` — audit decision applied (kept with header).
- [x] `uv run ruff check src/ tests/ benchmarks/` clean.
- [x] `uv run pytest tests/ -q` passes.
- [x] `uv run pytest tests/contract/ -q` passes.
- [x] `python scripts/compare_benchmarks.py benchmarks/results/phase-0-baseline.json benchmarks/results/v1.0.0.json --tolerance-pct 5` — operator-run vs release JSON.

---

## 8. Risks specific to this phase

| Risk | Mitigation |
|---|---|
| `lattice benchmark` wrapper fails when LATTICE is installed via pip (benchmarks/ not packaged) | The `[tool.hatch.build.targets.wheel]` in `pyproject.toml` packages `src/lattice/` only; `benchmarks/` is intentionally excluded. The wrapper checks `bench_cli.exists()` and prints a clear "run from source" error. |
| Some users have v1 pipeline references in their own benchmark scripts | The `--use-v2-pipeline` flag removal is in CHANGELOG. Their scripts will error on argparse — clear failure mode. |
| `benchmarks/results/v1.0.0.json` requires a specific provider to produce | Document the canonical provider in `benchmarks/results/CLAIMS.md` preamble. Contributors with different providers run against their own provider and compare in PR description. |
| `CLAIMS.md` becomes stale as transforms / providers change | Phase 11 release script regenerates the count rows (transforms count, providers count, tests count). Compression-% claims are pinned to the v1.0.0 reference run JSON. |
| `scripts/test_e2e_real.py` may turn out to be a CI dependency | `git log` shows last commit and CI grep shows referencing workflows. If both are stale, safe to delete. Else keep + document. |

---

## 9. Rollback plan

```bash
git revert <phase-9-merge-commit>
```

Restores `src/lattice/evals/` (empty), the `--use-v2-pipeline` flag (no-op), and the stub `lattice benchmark` CLI. No data loss.

---

## 10. PR shape

```
chore(benchmarks): delete src/lattice/evals/; remove use_v2_pipeline; lattice benchmark wrapper [Phase 9]

- DELETE src/lattice/evals/ (dead placeholder; benchmarks/evals/ is canonical)
- REMOVE --use-v2-pipeline flag from benchmarks/evals/cli.py and runner.py (v2 is the only path)
- MAKE `lattice benchmark` a real wrapper invoking benchmarks/evals/cli.py
- ADD benchmarks/results/v1.0.0.{json,md} — canonical release reference run
- ADD benchmarks/results/CLAIMS.md — claim → JSON traceability
- ADD scripts/run_canonical_benchmark.sh — single-line wrapper used by CI

Net: -1 directory (src/lattice/evals/), +3 result/script files, ~30 LoC removed in runner.py.
All 1600+ tests green. Contract tests green. `lattice benchmark` end-to-end verified locally.
```
