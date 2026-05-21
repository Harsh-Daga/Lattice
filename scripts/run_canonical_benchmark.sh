#!/usr/bin/env bash
# Canonical benchmark run for the LATTICE refactor CI gate.
#
# Usage:
#     ./scripts/run_canonical_benchmark.sh <output.json> [--iterations N] [--warmup N]
#
# Reads OLLAMA_API_KEY from the environment (never embeds it). The default
# provider is ollama-cloud with kimi-k2.6:cloud per REFACTOR_PLAN.md §7.
# Override via LATTICE_BENCHMARK_PROVIDER / LATTICE_BENCHMARK_MODEL.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <output.json> [extra args ...]" >&2
    exit 2
fi

OUTPUT_JSON="$1"
shift

PROVIDER="${LATTICE_BENCHMARK_PROVIDER:-ollama-cloud}"
MODEL="${LATTICE_BENCHMARK_MODEL:-kimi-k2.6:cloud}"
ITERATIONS="${LATTICE_BENCHMARK_ITERATIONS:-1}"
WARMUP="${LATTICE_BENCHMARK_WARMUP:-0}"
PROVIDER_WARMUP="${LATTICE_BENCHMARK_PROVIDER_WARMUP:-0}"

OUTPUT_DIR="$(dirname "$OUTPUT_JSON")"
mkdir -p "$OUTPUT_DIR"

# Derive the matching markdown report path.
OUTPUT_MD="${OUTPUT_JSON%.json}.md"

if [[ "$PROVIDER" == "ollama-cloud" ]]; then
    if [[ -z "${OLLAMA_CLOUD_API_KEY:-}" && -z "${OLLAMA_API_KEY:-}" ]]; then
        echo "ERROR: OLLAMA_CLOUD_API_KEY must be set in the environment for provider=ollama-cloud." >&2
        echo "       (OLLAMA_API_KEY is also accepted as a fallback.)" >&2
        exit 3
    fi
    # If only OLLAMA_API_KEY is set, mirror it to the canonical name the
    # runtime resolver expects.
    if [[ -z "${OLLAMA_CLOUD_API_KEY:-}" ]]; then
        export OLLAMA_CLOUD_API_KEY="${OLLAMA_API_KEY}"
    fi
fi

echo "Running canonical benchmark:"
echo "  provider:   $PROVIDER"
echo "  model:      $MODEL"
echo "  iterations: $ITERATIONS (warmup=$WARMUP, provider_warmup=$PROVIDER_WARMUP)"
echo "  output:     $OUTPUT_JSON (+ $OUTPUT_MD)"

uv run python benchmarks/evals/cli.py --suite all \
    --providers "$PROVIDER" \
    --provider-model "${PROVIDER}=${MODEL}" \
    --iterations "$ITERATIONS" \
    --warmup "$WARMUP" \
    --provider-warmup "$PROVIDER_WARMUP" \
    --output-json "$OUTPUT_JSON" \
    --output-md "$OUTPUT_MD" \
    "$@"

echo "Done: $OUTPUT_JSON"
