#!/usr/bin/env bash
# Phase 12 stub — full SDK dedup audit lands with TypeScript SDK (Phase 17).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SDK="${ROOT}/typescript-sdk/src"
if [[ ! -d "${SDK}" ]]; then
  echo "check_sdk_no_algorithm_duplication: skip (no typescript-sdk/src)"
  exit 0
fi
if rg -q 'composite_score|def score\(' "${SDK}" 2>/dev/null; then
  echo "check_sdk_no_algorithm_duplication: possible duplicated scoring in TS SDK" >&2
  exit 1
fi
echo "check_sdk_no_algorithm_duplication: ok"
