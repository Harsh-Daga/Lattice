# Phase 5 — Benchmark-gated transform decisions

## Sweep configuration

| Sweep | Config | Output file |
|-------|--------|-------------|
| A | Defaults (baseline) | `benchmarks/results/phase-5-A-baseline.json` |
| B | `transform_strategy_selector=false`, information-theoretic off | `benchmarks/results/phase-5-B-cut.json` |
| C | `transform_strategy_selector=true`, force information-theoretic on | `benchmarks/results/phase-5-C-forced.json` |

Canonical compare (post-merge gate):

```bash
bash scripts/run_canonical_benchmark.sh benchmarks/results/phase-5.json
python scripts/compare_benchmarks.py benchmarks/results/phase-0-baseline.json benchmarks/results/phase-5.json --tolerance-pct 2
```

## Benchmark gate status

**Skipped locally:** `OLLAMA_CLOUD_API_KEY` was not set in the agent environment. No `phase-5-A/B/C-*.json` artifacts were produced for this branch run.

## Verdict (default per `04-transforms.md` §4.3)

| Transform | Decision | Rationale |
|-----------|----------|-----------|
| `strategy_selector` | **DELETE** | No measured lift assumed without forced-on sweep; overlaps `content_profiler` heuristics; `legacy_only` with dual `process()`/`optimize()`. |
| `information_theoretic_selector` | **DELETE** | Submodular `context_selector` retained as sole implementation; no registry entry for info-theoretic variant. |

## Config carryover (one release)

- `transform_strategy_selector` — no-op (field kept on `LatticeConfig`)
- `transform_prefix_opt` / `transform_constraint_lifting` — no-op (removed in 5b)

Full `MIGRATION.md` updates: Phase 12.
