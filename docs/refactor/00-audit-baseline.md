# Phase 0 — Audit & Baseline

> **Goal.** Lock the public surface, capture an objective baseline of what works today, and produce machine-readable inventories the rest of the phases will reference. **No code in `src/lattice/` is touched in this phase.**
>
> **Outcome.** A `benchmarks/results/phase-0-baseline.json`, a complete file inventory at `docs/refactor/inventory.csv`, a passing `tests/contract/` suite, and a CI gate script. Phase 1 starts from a known-green baseline.
>
> **Estimated effort.** 0.5 day.

---

## 1. Why this phase exists

You cannot refactor what you have not measured. Three concrete problems today:

1. **README/AGENTS/code disagree on counts.** README says 1584 tests, AGENTS says 1839, actual `pytest --collect-only` count is somewhere in between. README says 18 transforms; the registry exposes 25; some are no-ops.
2. **No "is this still green?" gate.** There is no recorded baseline benchmark; every phase needs to be able to say "this didn't regress vs. baseline".
3. **No machine-readable inventory.** Every later phase has to re-grep the codebase. One CSV solves that once.

This phase produces the artefacts every later phase imports.

---

## 2. Files touched

**Created** (none under `src/lattice/`):

```
docs/refactor/inventory.csv
docs/refactor/FEATURE_PARITY.md
docs/refactor/api-surface.json
scripts/compare_benchmarks.py
scripts/audit_inventory.py
scripts/audit_contract.py
tests/contract/__init__.py
tests/contract/test_cli_contract.py
tests/contract/test_http_contract.py
tests/contract/test_headers_contract.py
tests/contract/test_python_api_contract.py
benchmarks/results/phase-0-baseline.json
.github/workflows/refactor-gate.yml   # OPTIONAL — adds CI gate if not already present
```

**Deleted**:

```
repomix-output.txt                    # 3.3 MB generated artefact at repo root; not source
.uv-cache/                            # already in .gitignore; clean working copy
src/lattice/proxy/compat_exports.py   # already deleted in git status; commit the deletion
```

**Modified**:

```
README.md                             # ONLY the "tests passed" badge count — corrected to actual `pytest --collect-only` number
AGENTS.md                             # same correction; remove "1839+" estimate, replace with actual count
.gitignore                            # add repomix-output.txt and benchmarks/results/*.json (keep .gitkeep for tracked subset)
```

---

## 3. Step-by-step

### 3.1 Capture the current pytest count

```bash
uv run pytest tests/ --collect-only -q 2>&1 | tail -5
```

Write the number into a variable; update `README.md`'s badge URL and `AGENTS.md`'s "Tests passed" row to that exact number. Both files currently disagree; they must match the collector output exactly. Commit this as a stand-alone fix.

### 3.2 Generate the file inventory CSV

Create `scripts/audit_inventory.py`:

```
For every *.py file under src/lattice/:
    path
    loc            (wc -l)
    top_level_classes      (grep -E '^class ' | count)
    top_level_funcs        (grep -E '^def '   | count)
    imports_from_lattice   (grep -E '^from lattice' | unique)
    imported_by            (rg "from lattice.<this_module>" src/ tests/ | files)
    phase_target           (column derived from Phase 1-11 plans)
    final_path             (column derived from FINAL_LAYOUT.md)
    disposition            (KEEP | MOVE | SPLIT | DELETE | RENAME)
Output: docs/refactor/inventory.csv
```

The script reads `FINAL_LAYOUT.md` to populate `final_path` and `disposition`. The output is the canonical lookup table for every later phase. Approx 166 rows.

### 3.3 Capture the public API surface

Create `scripts/audit_contract.py`. Produces `docs/refactor/api-surface.json` with three top-level keys:

```json
{
  "cli": {
    "commands": [
      {"name": "proxy", "subcommands": ["run", "start", "stop", "restart", "status"], "flags": [...]},
      {"name": "init", ...},
      ...
    ]
  },
  "http": {
    "endpoints": [
      {"method": "POST", "path": "/v1/chat/completions", "handler": "compat.make_chat_completion_handler", "streaming": true},
      ...
    ],
    "headers_emitted": ["x-lattice-compression", "x-lattice-session-id", ...]
  },
  "python_api": {
    "exports_from_lattice": ["LatticeClient", "LatticeProxyClient", "CompressResult", "wrap_openai_client", "__version__"],
    "exports_from_lattice_core": ["LatticeConfig", "TransformContext", "Result", "Ok", "Err", "Request", "Response", "Message", "Role", "Transform", "SyncTransform", "ReversibleSyncTransform", "CompressorPipeline", "TransformError", ...]
  }
}
```

This file is the master plan §2 surface in machine-readable form. The Phase 6 + 11 docs reference it.

### 3.4 Write the contract test suite

`tests/contract/` holds tests that verify the public surface by **invocation, not by inspection**. They are slow (start subprocesses, HTTP requests) but they are the gate.

**`test_cli_contract.py`** — for each command in `api-surface.json#cli.commands`:

```
For each command:
    subprocess.run(["lattice", command_name, "--help"], capture)
    assert exit_code == 0
    assert "Usage:" in stdout
For lattice proxy run/start/stop/restart/status:
    spin up a real proxy on a free port
    hit lattice proxy status → assert PID/uptime/healthy
    lattice proxy stop --grace 2
    assert subsequent lattice proxy status → not running
```

**`test_http_contract.py`** — start a proxy, send canned requests, validate response shape:

```
For each endpoint in api-surface.json#http.endpoints:
    Send minimal valid request
    Assert status code
    Assert response Content-Type
    Assert response shape contains required fields
```

For `/v1/chat/completions` and `/v1/messages`: also send a streaming request, assert SSE format `data: {...}\n\n`, assert `[DONE]` final chunk.

**`test_headers_contract.py`** — after a single successful `/v1/chat/completions`, assert every header in `api-surface.json#http.headers_emitted` is present.

**`test_python_api_contract.py`** — pure import test:

```python
from lattice import (
    LatticeClient, LatticeProxyClient, CompressResult,
    wrap_openai_client, __version__,
)
# Plus every name in api-surface.json#python_api.exports_from_lattice

from lattice.core import (
    LatticeConfig, TransformContext, Result, Ok, Err,
    Request, Response, Message, Role,
    Transform, SyncTransform, ReversibleSyncTransform,
    CompressorPipeline,
    TransformError, ...
)

# Sanity check signatures
client = LatticeClient()
assert hasattr(client, "compress")
assert hasattr(client, "compress_request")
assert hasattr(client, "decompress_response")
assert hasattr(client, "health")
```

> **Important.** In Phase 2 the `CompressorPipeline` symbol moves. The contract test must keep importing `CompressorPipeline` from `lattice.core` even after the source file moves, via re-export in `src/lattice/core/__init__.py`. This is how the public surface stays stable while the internal layout changes.

### 3.5 Capture the benchmark baseline

```bash
uv run python benchmarks/evals/cli.py --suite all \
    --providers ollama-cloud --provider-model ollama-cloud=kimi-k2.6:cloud \
    --iterations 1 --warmup 0 --provider-warmup 0 \
    --output-json benchmarks/results/phase-0-baseline.json \
    --output-md   benchmarks/results/phase-0-baseline.md
```

Commit both files (they are small JSON/MD, not large traces). This is the reference every later phase compares against.

If `--providers ollama-cloud` isn't available in the contributor's environment, fall back to `--providers ollama --provider-model ollama=llama3.2`. Document the chosen provider in `phase-0-baseline.md`'s preface so later phases compare apples to apples.

### 3.6 Write the comparison script

`scripts/compare_benchmarks.py`:

```
Args: baseline_json target_json --tolerance-pct N
For each metric in baseline:
    delta = (target - baseline) / baseline * 100
    if abs(delta) > tolerance_pct:
        print regression / improvement
        nonzero exit if regression
```

Metrics it must compare: `compression_pct`, `quality_score`, `latency_p50_ms`, `latency_p99_ms`, `cost_usd_per_1k_tokens`, `cache_hit_rate`. The exact JSON keys depend on `benchmarks/evals/runner.py` output — read that file first. Output is one line per metric.

### 3.7 Clean repo artefacts

```bash
git rm --cached repomix-output.txt
echo "repomix-output.txt" >> .gitignore
git rm src/lattice/proxy/compat_exports.py    # already deleted in working tree; just commit it
```

### 3.8 Commit shape

This phase ships as one or two PRs:

```
PR1: chore(audit): inventory + contract tests + baseline (Phase 0)
  - Adds docs/refactor/inventory.csv
  - Adds docs/refactor/api-surface.json
  - Adds tests/contract/* (all green)
  - Adds scripts/audit_inventory.py, scripts/audit_contract.py, scripts/compare_benchmarks.py
  - Adds benchmarks/results/phase-0-baseline.{json,md}
  - Updates README.md + AGENTS.md test counts to match pytest collection
  - Deletes repomix-output.txt + compat_exports.py

PR2 (optional): ci: refactor gate workflow
  - Adds .github/workflows/refactor-gate.yml
  - Wires the five-step CI gate from REFACTOR_PLAN.md §7
```

---

## 4. Acceptance criteria — phase done when all are true

- [ ] `docs/refactor/inventory.csv` exists and lists 166 rows, each with `phase_target`, `final_path`, `disposition` columns populated from `FINAL_LAYOUT.md`.
- [ ] `docs/refactor/api-surface.json` exists; every CLI command, HTTP endpoint, response header, and public Python symbol that v1.0.0 must preserve is enumerated.
- [ ] `tests/contract/` exists, runs in CI, and passes on `main` today.
- [ ] `benchmarks/results/phase-0-baseline.json` exists and is referenced by `scripts/compare_benchmarks.py`.
- [ ] `repomix-output.txt` is gone from the repo (`.gitignore` updated).
- [ ] `src/lattice/proxy/compat_exports.py` is gone (commit the deletion).
- [ ] `README.md` and `AGENTS.md` test counts agree with `uv run pytest --collect-only`.
- [ ] `uv run ruff check src/`, `uv run mypy src/lattice/`, `uv run pytest tests/ -q` all pass.

---

## 5. Risks specific to this phase

| Risk | Mitigation |
|---|---|
| `tests/contract/` is too slow for CI (it spins real proxies) | Mark `@pytest.mark.contract`; run on PR merges to `main` only, not on every push. Phase 0 includes both — fast path for dev, contract suite for merge gate. |
| Benchmark baseline differs across contributor machines (different Ollama model, network) | Document the canonical baseline provider in `phase-0-baseline.md` preface. Later phases re-run against same provider. |
| `api-surface.json` misses an undocumented but used Python symbol | The Phase 6/11 review re-greps the repo for `from lattice` imports in tests/, benchmarks/, and scripts/, adding anything missed. |
| Inventory CSV gets stale during the multi-week refactor | Each phase's PR description must include a one-line "inventory diff" — files moved/deleted in that phase. Phase 11 regenerates the CSV from scratch as the final check. |

---

## 6. What this phase does **not** do

- It does not touch `src/lattice/`. Anything you find while writing the inventory that "obviously should be moved" — write it into the relevant Phase 1-11 doc and leave the code alone.
- It does not change tests outside `tests/contract/`. Existing test reshape is Phase 10's job.
- It does not introduce any deprecation warnings. The whole point of v1.0.0 is no warnings; deprecation lives in v0.x patch releases if anywhere.
- It does not modify `pyproject.toml` (version bump is Phase 11).
- It does not write a CHANGELOG (Phase 11).
