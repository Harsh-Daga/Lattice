# Phase 6 benchmark gate

Canonical compare (±2% vs `phase-0-baseline.json`):

```bash
export OLLAMA_CLOUD_API_KEY=...   # required

uv run python benchmarks/evals/cli.py --suite all \
  --providers ollama-cloud \
  --provider-model ollama-cloud=kimi-k2.6:cloud \
  --iterations 1 --warmup 0 --provider-warmup 0 \
  --output-json benchmarks/results/phase-6.json

python scripts/compare_benchmarks.py \
  benchmarks/results/phase-0-baseline.json \
  benchmarks/results/phase-6.json \
  --tolerance-pct 2
```

If the key is unavailable in CI, Phase 6 code acceptance does not require `phase-6.json` in-repo; run locally before release and commit the artifact when available.
