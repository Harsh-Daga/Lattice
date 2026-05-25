# LATTICE v1.0.0 — Claim Traceability

Every figure in README.md, AGENTS.md, and public docs must appear in this table before v1.0.0 release copy is finalized (full README rewrite: Phase 12 / `11-docs-release.md`). Compression and transport metrics are pinned to `benchmarks/results/v1.0.0.json` from the canonical provider run below.

**Canonical provider:** `ollama-cloud` / `kimi-k2.6:cloud` (requires `OLLAMA_CLOUD_API_KEY`).  
**Release run:** `--suite all --iterations 3 --warmup 1 --provider-warmup 1` → `v1.0.0.json`, `v1.0.0.md`.  
**CI gate run:** `./scripts/run_canonical_benchmark.sh` (1 iteration, 0 warmup).

| Claim | Source | JSON / proof | Run date | Notes |
|-------|--------|--------------|----------|-------|
| "1764 tests passed" | `uv run pytest tests/ -q` | n/a (pytest) | 2026-05-25 | 196 skipped; 1960 collected |
| "1903 tests collected" (README badge) | pytest collect | n/a | 2026-05-25 | README badge refresh in Phase 12; collected count drifts with skips |
| "18 transforms" (README) | Transform registry | structural | n/a | Registry has **20** names; README "18" → Phase 12 aligns copy to `list_transform_names()` |
| "17 providers" | `ProviderRegistry().adapters` | structural | n/a | `len(ProviderRegistry().adapters) == 17` |
| Default pipeline transform count (6) | `list_default_pipeline_names()` | structural | n/a | Orchestrators add more transforms at runtime |
| TACC AIMD-style adaptive concurrency | `transport/congestion.py` | `v1.0.0.json` (transport/tacc sections) | release | See `docs/novel/tacc.md` |
| Binary framing (fixed header, frame types) | `protocol/framing.py` | structural | n/a | `FrameType` enum in code |
| Delta encoding (multi-turn savings) | `transport/delta_wire.py` | `v1.0.0.json` | release | Measured in suite sections when present |
| "30-60% per-request overhead reduction" (batching) | README / batching spec | `v1.0.0.json` | release | Batching scenarios; see `sections` for batching |
| "20-40% redundant content" (message_dedup) | README | `v1.0.0.json#feature_eval` | release | Long-conversation scenarios in feature suite |
| "20-50% structured workloads" (reference_sub) | README | `v1.0.0.json#feature_matrix` | release | feature-matrix suite rows |
| Headline compression % (default suite) | `v1.0.0.md` | `v1.0.0.json#feature_eval.summary.avg_reduction_ratio` | release | Primary marketing metric |
| Phase gate ±2% vs baseline | `scripts/compare_benchmarks.py` | `phase-0-baseline.json` vs `phase-*.json` | per-phase | Release compare uses ±5% vs baseline |
| `lattice benchmark` runs real CLI | `src/lattice/cli.py:_cmd_benchmark` | `tests/unit/cli/test_benchmark_wrapper.py` | 2026-05-25 | No `--use-v2-pipeline` flag |
| No `lattice.evals` package | layout | `tests/unit/test_no_lattice_evals.py` | 2026-05-25 | Canonical evals: `benchmarks/evals/` |
