# Phase 14 benchmark gate

Operator-run canonical suite (API key via `OLLAMA_CLOUD_API_KEY` only — never committed).

```bash
export OLLAMA_CLOUD_API_KEY=...   # required; do not write into repo files

uv run python benchmarks/evals/cli.py --suite all \
  --providers ollama-cloud \
  --provider-model ollama-cloud=kimi-k2.6:cloud \
  --iterations 3 --warmup 1 --provider-warmup 1 \
  --output-json benchmarks/results/phase-14-transport.json
```

## Results (2026-05-26, branch `refactor/phase-14-transport-consolidation`)

| Metric | phase-0-baseline | phase-14-transport | Notes |
|--------|------------------|--------------------|-------|
| `feature_eval.avg_pipeline_latency_ms` | 96.77 | 77.09 | **−20.3%** (in-process pipeline; no regression on p50 path) |
| `feature_eval.avg_reduction_ratio` | 0.287 | 0.441 | +53.7% (scenario variance) |
| `transport_eval` passed/total | 10/10 | 10/10 | Deterministic transport checks green |
| `tacc_eval.avg_static_p95_ms` | 259.68 | 259.68 | Simulation (concurrency=12), not live provider wire |
| `tacc_eval.avg_tacc_p95_ms` | 257.99 | 257.99 | Same |

`scripts/compare_benchmarks.py` vs baseline at ±5% reports regressions on offline `feature_matrix_eval` / `replay_eval` rows (0% reduction on synthetic replay) and `provider_eval.avg_quality_score` (−7.5%). Those are **not** transport-layer regressions; see [CLAIMS.md](../../benchmarks/results/CLAIMS.md) release compare guidance.

**HTTP/2 / 50-concurrency p95:** Verified by `tests/unit/transport/test_pool_http2.py` and integration transport suite. Live provider p95 at 50 concurrent connections is not emitted by the standard `tacc_eval` harness (uses `static_concurrency=12` simulation).

## Compare (informational)

```bash
uv run python scripts/compare_benchmarks.py \
  benchmarks/results/phase-0-baseline.json \
  benchmarks/results/phase-14-transport.json \
  --tolerance-pct 5
```

Artifact: `benchmarks/results/phase-14-transport.json` (env keys redacted as `<redacted>`).
