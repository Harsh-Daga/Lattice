#!/usr/bin/env bash
# Canonical benchmark run for refactor-gate CI (see docs/refactor/09-benchmarks.md).
# Usage: ./scripts/run_canonical_benchmark.sh [output-json]
set -euo pipefail
OUT="${1:-benchmarks/results/local.json}"
exec uv run python benchmarks/evals/cli.py --suite all \
    --providers ollama-cloud --provider-model ollama-cloud=kimi-k2.6:cloud \
    --iterations 1 --warmup 0 --provider-warmup 0 \
    --output-json "$OUT"
