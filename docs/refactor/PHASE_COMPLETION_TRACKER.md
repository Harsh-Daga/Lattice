# Phase completion tracker (Phases 0–6)

> **Rule:** Non-benchmark acceptance items must be ✅ before a phase is **Done**.
> **Benchmarks:** `phase-*.json` compare gates are tracked separately (optional CI key).

**Last verified:** `refactor/phase-6c-providers-transport-split` — **1712 passed**, 196 skipped, contract green.

| Phase | Verdict | Notes |
|-------|---------|-------|
| **0** | ✅ | `FEATURE_PARITY.md` scaffold; inventory; contract; `refactor-gate.yml` |
| **1** | ✅ | `ir/` package; imports; `test_ir.py` under `tests/unit/ir/` |
| **2** | ✅ | Layout; `test_no_legacy_process_paths.py` (IR-native scope per Phase 3/5) |
| **3** | ✅ | V1 kill; `Pipeline.compress()` |
| **4** | ✅ | Planner collapse; `_normalize_legacy_execution_plan` documented as persisted-plan bridge only |
| **5** | ✅ | Transforms cleanup + §6.3 tests |
| **6** | ✅ code | Providers/adapters + transport package split; **benchmark** operator |

---

## Phase 0 — `00-audit-baseline.md` §4

| Criterion | Status |
|-----------|--------|
| `inventory.csv` (166 rows, columns populated) | ✅ |
| `FEATURE_PARITY.md` | ✅ scaffold |
| `api-surface.json` | ✅ |
| `tests/contract/` + CI | ✅ (`ci.yml` + `refactor-gate.yml`) |
| `phase-0-baseline.json` | ✅ |
| `repomix` gone / gitignore | ✅ |
| `compat_exports.py` gone | ✅ |
| README + AGENTS counts | ✅ (1903 collected; 1706 passed) |
| ruff / mypy / pytest | ✅ |

---

## Phase 1 — `01-ir-primitives.md` §6

All import/layout criteria ✅. Benchmark lines excluded.

---

## Phase 2 — `02-pipeline-runner.md` §6

| Criterion | Status |
|-----------|--------|
| v1 pipeline / core transport deleted | ✅ |
| `pipeline/` + `transport/` layout | ✅ |
| Forbidden imports | ✅ |
| `test_transport_types_canonical_path.py` | ✅ |
| `test_no_legacy_process_paths.py` | ✅ (IR-native; Phase 2 §4.3 intent) |
| Full-registry `process()` removal | ⚠️ deferred Phase 11 (execution-only + response-side exempt) |

---

## Phase 3 — `STATUS.md` §5

✅ CompressorPipeline deleted; `compress()` + gates; client/factory wired.

---

## Phase 4 — `03-planner-collapse.md` §8

✅ All code criteria; benchmark `[ ]` only.

---

## Phase 5 — `04-transforms.md` §9

| Criterion | Status |
|-----------|--------|
| Registry / reputation / patterns moves | ✅ |
| Deletes + splits | ✅ |
| `is_response_side`, delta flag | ✅ |
| §6.3 tests (classifier, risk, planner_bridge, table/json converter) | ✅ |
| prefix_opt hygiene (src policy/strategy/validation) | ✅ |
| IR-native `optimize()` only | ✅ |
| `phase-5-decisions.md` | ✅ |
| Benchmark compare | ⏳ operator |

---

## Phase 6 — `05-providers-transport.md` §10

| Criterion | Status |
|-----------|--------|
| `providers/transport.py` monolith deleted | ✅ |
| `providers/transport/` package (7 modules + `__init__`) | ✅ |
| `providers/adapters/` (8 files) | ✅ |
| `stall_detector.py` at providers root deleted | ✅ |
| `base.py` at providers root deleted | ✅ |
| No file in `providers/` > 850 LoC | ✅ (max `streaming.py` ~495) |
| Unified `_stream`; thin `completion_stream*` wrappers | ✅ |
| All 17 adapters at `lattice.providers` | ✅ |
| TTL `RateLimitTracker` + tests | ✅ |
| §9.2–9.3 new tests | ✅ |
| ruff / pytest / contract | ✅ |
| Benchmark `phase-6.json` ±2% | ⏳ operator (`OLLAMA_CLOUD_API_KEY`) |

---

## Remaining operator actions (not code)

1. Run canonical benchmark → `phase-6.json` when `OLLAMA_CLOUD_API_KEY` is set.
2. Merge `refactor/phase-6a` → `6b` → `6c` PRs to `main`.
