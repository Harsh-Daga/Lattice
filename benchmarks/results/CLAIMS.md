# LATTICE v1.0.0 — Claim Traceability

Every figure in README.md, AGENTS.md, and public docs must appear in this table before v1.0.0 release copy is finalized (full README rewrite: Phase 12 / `11-docs-release.md`). Compression and transport metrics are pinned to `benchmarks/results/v1.0.0.json` from the canonical provider run below.

**Canonical provider:** `ollama-cloud` / `kimi-k2.6:cloud` (requires `OLLAMA_CLOUD_API_KEY`).  
**Release run:** `--suite all --iterations 3 --warmup 1 --provider-warmup 1` → `v1.0.0.json`, `v1.0.0.md`.  
**CI gate run:** `./scripts/run_canonical_benchmark.sh` (1 iteration, 0 warmup).

**Compare vs `phase-0-baseline.json`:** `compare_benchmarks.py … --tolerance-pct 5` is a **release reference check**, not a phase ±2% gate. Expect drift on LLM-judge and provider-validation rows; re-run after code fixes (e.g. `provider_validation` must use `Pipeline.compress()`).

| Claim | Source | JSON / proof | Run date | Notes |
|-------|--------|--------------|----------|-------|
| "1766 tests passed" | `uv run pytest tests/ -q` | n/a (pytest) | 2026-05-25 | 196 skipped; **1962 collected** |
| "1962 tests collected" (README badge) | pytest collect | n/a | 2026-05-25 | Pinned in README + AGENTS; Phase 11 adds `test_test_count_pinned.py` |
| "18 transforms" (README) | Transform registry | structural | n/a | Registry has **20** names; README aligned in Phase 12 |
| "17 providers" | `ProviderRegistry().adapters` | structural | n/a | `len(ProviderRegistry().adapters) == 17` |
| Default pipeline transform count (6) | `list_default_pipeline_names()` | structural | n/a | Orchestrators add more at runtime |
| Headline compression % (feature suite) | `v1.0.0.json` | `sections[feature_eval].summary.avg_reduction_ratio` | release | **0.4033** (40.33%) on canonical re-run post `compress()` fix |
| Feature eval pipeline latency | `v1.0.0.json` | `feature_eval.summary.avg_pipeline_latency_ms` | release | **~35.9 ms** (3 iter run) |
| Provider validation pass rate | `v1.0.0.json` | `provider_validation.summary` | release | **18 evaluated, 3 passed** (0.167); no `process()` errors after fix |
| TACC AIMD-style adaptive concurrency | `transport/congestion.py` | `v1.0.0.json` (tacc section) | release | See `docs/novel/tacc.md` |
| Binary framing (fixed header, frame types) | `protocol/framing.py` | structural | n/a | `FrameType` enum in code |
| Delta encoding (multi-turn savings) | `transport/delta_wire.py` | `v1.0.0.json` | release | Measured where multi-turn scenarios apply |
| "30-60% per-request overhead reduction" (batching) | README / batching spec | `v1.0.0.json` | release | Batching via provider_eval / feature_matrix; many rows **n/a** offline |
| "20-40% redundant content" (message_dedup) | README | `v1.0.0.json#feature_eval` | release | Long-conversation scenarios |
| "20-50% structured workloads" (reference_sub) | README | `v1.0.0.json#feature_matrix` | release | Matrix uses replay traces; savings often 0 offline |
| Replay trace compression | `v1.0.0.json` | `replay_eval.summary.avg_reduction_ratio` | release | Synthetic replay may show **0.0** (no live provider) |
| Phase gate ±2% vs baseline | `scripts/compare_benchmarks.py` | `phase-0-baseline.json` vs `phase-*.json` | per-phase | Release compare uses ±5%, not required green |
| `lattice benchmark` runs real CLI | `src/lattice/cli.py:_cmd_benchmark` | `tests/unit/cli/test_benchmark_wrapper.py` | 2026-05-25 | No `--use-v2-pipeline` |
| `lattice benchmark --provider-detect` | `benchmarks/evals/cli.py` | `tests/unit/test_production_evals.py` | 2026-05-25 | Picks first credentialed provider |
| No `lattice.evals` package | layout | `tests/unit/test_no_lattice_evals.py` | 2026-05-25 | Canonical evals: `benchmarks/evals/` |
| Canonical benchmark script | `scripts/run_canonical_benchmark.sh` | `refactor-gate.yml` `test -x` | 2026-05-25 | Full suite needs operator API key in CI optional job |
