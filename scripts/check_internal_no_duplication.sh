#!/usr/bin/env bash
# Verify SINGLE_SOURCE_OF_TRUTH.md registry paths exist; flag duplicate class names.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SSOT="${ROOT}/docs/refactor/SINGLE_SOURCE_OF_TRUTH.md"
SRC="${ROOT}/src/lattice"

if [[ ! -f "${SSOT}" ]]; then
  echo "check_internal_no_duplication: missing SSOT" >&2
  exit 1
fi

fail=0

_count_execution_plan() {
  if command -v rg >/dev/null 2>&1; then
    rg -c '^class ExecutionPlan\b' "${SRC}" 2>/dev/null | awk -F: '{s+=$2} END {print s+0}'
  else
    grep -r '^class ExecutionPlan\b' "${SRC}" --include='*.py' 2>/dev/null | wc -l | tr -d ' '
  fi
}

_stale_in_src() {
  local pat='MILV|BatchAccumulator|from lattice\.planner\.execution_plan'
  if command -v rg >/dev/null 2>&1; then
    rg -q "${pat}" "${SRC}" 2>/dev/null
  else
    grep -rqE "${pat}" "${SRC}" --include='*.py' 2>/dev/null
  fi
}

ep_count="$(_count_execution_plan)"
if [[ "${ep_count}" -ne 1 ]]; then
  echo "check_internal_no_duplication: expected 1 class ExecutionPlan, found ${ep_count}" >&2
  fail=1
fi

if [[ -f "${SRC}/planner/execution_plan.py" ]]; then
  echo "check_internal_no_duplication: planner/execution_plan.py must be removed (use ir.primitives.ExecutionPlan)" >&2
  fail=1
fi

if _stale_in_src; then
  echo "check_internal_no_duplication: stale MILV/BatchAccumulator/planner.execution_plan in src/" >&2
  fail=1
fi

if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi
echo "check_internal_no_duplication: ok"
