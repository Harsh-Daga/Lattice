#!/usr/bin/env bash
# Enforce src/lattice/ LoC caps from docs/refactor/CODE_BUDGET.txt (Phase 13+).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUDGET_FILE="${ROOT}/docs/refactor/CODE_BUDGET.txt"
SRC="${ROOT}/src/lattice"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

if [[ ! -f "${BUDGET_FILE}" ]]; then
  echo "check_code_budget: missing ${BUDGET_FILE}" >&2
  exit 1
fi

total_loc() {
  find "${SRC}" -name '*.py' -print0 | xargs -0 wc -l 2>/dev/null | awk '/total$/ {print $1}'
}

dir_loc() {
  local rel="$1"
  local path="${SRC}/${rel}"
  if [[ ! -d "${path}" ]]; then
    echo 0
    return
  fi
  find "${path}" -name '*.py' -print0 | xargs -0 wc -l 2>/dev/null | awk '/total$/ {print $1}'
}

fail=0
total="$(total_loc)"
enforce_total="$(grep -E '^enforce_total_v2=' "${BUDGET_FILE}" 2>/dev/null | cut -d= -f2 || true)"
total_cap="$(grep -E '^total_v2=' "${BUDGET_FILE}" 2>/dev/null | cut -d= -f2 || true)"
if [[ "${enforce_total}" == "1" ]] && [[ -n "${total_cap}" ]] && [[ "${total}" -gt "${total_cap}" ]]; then
  echo "check_code_budget: total src/lattice LoC ${total} exceeds cap ${total_cap}" >&2
  fail=1
fi

enforce_dirs="$(grep -E '^enforce_dir_caps=' "${BUDGET_FILE}" 2>/dev/null | cut -d= -f2 || true)"
while IFS= read -r line; do
  [[ "${line}" =~ ^[[:space:]]*# ]] && continue
  [[ -z "${line}" ]] && continue
  [[ "${line}" == *"="* ]] || continue
  key="${line%%=*}"
  val="${line#*=}"
  case "${key}" in
    total_v2|phase_*|crates_*|bindings_*|typescript_*|cursor_*|enforce_*)
      continue
      ;;
  esac
  if [[ "${enforce_dirs}" != "1" ]]; then
    continue
  fi
  if [[ "${key}" == providers_adapters ]]; then
    cap="${val}"
    actual="$(dir_loc "providers/adapters")"
    if [[ "${actual}" -gt "${cap}" ]]; then
      echo "check_code_budget: providers/adapters LoC ${actual} > cap ${cap}" >&2
      fail=1
    fi
    continue
  fi
  cap="${val}"
  actual="$(dir_loc "${key}")"
  if [[ "${actual}" -gt "${cap}" ]]; then
    echo "check_code_budget: ${key}/ LoC ${actual} > cap ${cap}" >&2
    fail=1
  fi
done < "${BUDGET_FILE}"

while IFS= read -r big; do
  [[ -z "${big}" ]] && continue
  echo "check_code_budget: file exceeds 800 LoC: ${big}" >&2
  fail=1
done < <(find "${SRC}" -name '*.py' -print0 | xargs -0 wc -l 2>/dev/null | awk '$1 > 800 && $2 != "total" {print $2}')

enforce_delta="$(grep -E '^enforce_phase_delta=' "${BUDGET_FILE}" 2>/dev/null | cut -d= -f2 || true)"
phase_key="${LATTICE_PHASE_DELTA:-}"
if [[ -z "${phase_key}" ]]; then
  # Doc-only / multi-phase PRs: skip delta check unless env set
  phase_key=""
fi
if [[ "${enforce_delta}" == "1" ]] && [[ -n "${phase_key}" ]] && git -C "${ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  phase_decl="$(grep -E "^${phase_key}=" "${BUDGET_FILE}" | cut -d= -f2 || true)"
  if [[ -n "${phase_decl}" ]]; then
    base="$(git -C "${ROOT}" merge-base HEAD origin/main 2>/dev/null || git -C "${ROOT}" merge-base HEAD main 2>/dev/null || true)"
    if [[ -n "${base}" ]]; then
      added=0
      removed=0
      while read -r a r _; do
        added=$((added + a))
        removed=$((removed + r))
      done < <(git -C "${ROOT}" diff --numstat "${base}"...HEAD -- 'src/lattice/**/*.py' 2>/dev/null || true)
      net=$((added - removed))
      if [[ "${phase_decl}" =~ ^- ]]; then
        target="${phase_decl#-}"
        min_remove=$((target * 9 / 10))
        if [[ "${net}" -gt "-${min_remove}" ]]; then
          echo "check_code_budget: PR net LoC ${net} weaker than ${phase_key}=${phase_decl} (need ~-${min_remove} or more)" >&2
          fail=1
        fi
      elif [[ "${phase_decl}" =~ ^\+ ]]; then
        target="${phase_decl#+}"
        max_add=$((target * 11 / 10))
        if [[ "${net}" -gt "${max_add}" ]]; then
          echo "check_code_budget: PR net LoC ${net} exceeds ${phase_key}=${phase_decl} (+10% max ${max_add})" >&2
          fail=1
        fi
      fi
    fi
  fi
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "check_code_budget: dry-run total=${total} enforce_total=${enforce_total} enforce_dirs=${enforce_dirs} enforce_delta=${enforce_delta} phase_key=${phase_key:-<unset>}"
  exit 0
fi

if [[ "${fail}" -ne 0 ]]; then
  exit 1
fi
echo "check_code_budget: ok (total=${total})"
