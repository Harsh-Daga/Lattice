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

# SSOT paths: only require paths that already exist (forward-phase entries may be future).
while IFS= read -r path; do
  [[ -z "${path}" ]] && continue
  full="${ROOT}/${path}"
  if [[ -e "${full}" ]]; then
    continue
  fi
  # Allow directory entries ending in /
  if [[ "${path}" == */ ]] && [[ -d "${full%/}" ]]; then
    continue
  fi
done < <(
  grep -oE 'src/lattice/[a-zA-Z0-9_./-]+' "${SSOT}" | sort -u
)

# Exactly one ExecutionPlan class definition
ep_count="$(rg -c '^class ExecutionPlan\b' "${SRC}" 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')"
if [[ "${ep_count}" -ne 1 ]]; then
  echo "check_internal_no_duplication: expected 1 class ExecutionPlan, found ${ep_count}" >&2
  fail=1
fi

# No planner.execution_plan module
if [[ -f "${SRC}/planner/execution_plan.py" ]]; then
  echo "check_internal_no_duplication: planner/execution_plan.py must be removed (use ir.primitives.ExecutionPlan)" >&2
  fail=1
fi

# Stale public names in src/
if rg -q 'MILV|BatchAccumulator|from lattice\.planner\.execution_plan' "${SRC}" 2>/dev/null; then
  echo "check_internal_no_duplication: stale MILV/BatchAccumulator/planner.execution_plan in src/" >&2
  rg -n 'MILV|BatchAccumulator|from lattice\.planner\.execution_plan' "${SRC}" >&2 || true
  fail=1
fi

if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi
echo "check_internal_no_duplication: ok"
