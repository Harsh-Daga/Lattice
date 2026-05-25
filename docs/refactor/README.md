# Refactor & Forward Plan — Documentation Index

> **Start here** if you are implementing or reviewing LATTICE after Phase 12 (v1.0.0 docs release shipped). **Next:** v2.0 forward plan Phase 12 — honesty pass (`12-honesty-pass.md` in `FORWARD_PLAN.md`).

---

## Product thesis

**LATTICE is the transport / network layer for LLM traffic.** One self-hosted process sits between your app and your chosen provider. It owns connections, retries, timeouts, backpressure, framing, streaming, caching, guardrails, compression, and observability. It is **not** a compression-only library.

---

## Six constraints (every phase must obey)

| # | Constraint | Enforced by |
|---|---|---|
| 1 | Lightweight (4 GB laptop, no required models) | `tests/integration/footprint/` |
| 2 | No external LLM beyond user's provider | `tests/contract/test_default_install_no_*` |
| 3 | Open source self-hosted only (no SaaS) | Plan + review |
| 4 | One algorithm, one implementation | [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md) + dedup scripts |
| 5 | Codebase cap ≤ 35k LoC `src/lattice/` | [CODE_BUDGET.txt](CODE_BUDGET.txt) + `scripts/check_code_budget.sh` |
| 6 | Transport-first (unified transport layer) | [27-transport-layer-consolidation.md](27-transport-layer-consolidation.md) |

Full detail: [FORWARD_PLAN.md](FORWARD_PLAN.md).

---

## Document map

### Status & planning

| Doc | Purpose |
|---|---|
| [STATUS.md](STATUS.md) | What shipped (Phases 0–12) + forward-plan summary |
| [FORWARD_PLAN.md](FORWARD_PLAN.md) | Master index Phases 12–27 (+ 19.5), milestones, footprint table, execution order |
| [ARCHITECTURE_EVAL_INSIGHTS.md](ARCHITECTURE_EVAL_INSIGHTS.md) | Production evals → v2 plan (accept/reject, utility objective) |
| [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) | **Mandatory template** for every phase doc 12–27 |
| [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md) | Registry: every primitive → one canonical file |
| [CODE_BUDGET.txt](CODE_BUDGET.txt) | Per-directory LoC caps + per-phase declared deltas |
| [REFACTOR_PLAN.md](REFACTOR_PLAN.md) | Historical v1.0.0 refactor (Phases 0–11) |
| [PHASE_COMPLETION_TRACKER.md](PHASE_COMPLETION_TRACKER.md) | Line-by-line acceptance vs phase docs |
| [MIGRATION.md](MIGRATION.md) | User-facing migration notes |
| [FINAL_LAYOUT.md](FINAL_LAYOUT.md) | Target directory layout (v1 + v2 extensions) |

### v1.0 refactor phases (0–12) — shipped (historical specs)

`00-audit-baseline.md` … `11-docs-release.md` (Phase 12 docs release). Kept for acceptance audit; not required for day-to-day use.

| Supplement | Purpose |
|---|---|
| [phase-4-decisions.md](phase-4-decisions.md) | Pointer → phase-5 decisions |
| [phase-5-decisions.md](phase-5-decisions.md) | Benchmark-gated transform deletions |
| [phase-6-benchmark.md](phase-6-benchmark.md) | Operator benchmark gate for Phase 6 |

### v2.0 forward phases (12–27)

| Phase | Doc | Milestone |
|---|---|---|
| 12 | [12-honesty-pass.md](12-honesty-pass.md) | M2 |
| 13 | [13-python-sdk-quality.md](13-python-sdk-quality.md) | M2 |
| 14 | [14-hybrid-semantic-cache.md](14-hybrid-semantic-cache.md) | M2 |
| 15 | [15-native-guardrails.md](15-native-guardrails.md) | M2 |
| 16 | [16-otel-genai.md](16-otel-genai.md) | M2 |
| 17 | [17-typescript-sdk.md](17-typescript-sdk.md) | M3 |
| 18 | [18-mcp-native-gateway.md](18-mcp-native-gateway.md) | M3 |
| 19 | [19-compression-intelligence.md](19-compression-intelligence.md) | M3 |
| 19.5 | [19.5-segment-aware-planning.md](19.5-segment-aware-planning.md) | M3 — orchestration (eval-driven) |
| 20 | [20-non-chat-surfaces.md](20-non-chat-surfaces.md) | M3 |
| 21 | [21-agent-memory.md](21-agent-memory.md) | M4 |
| 22 | [22-cache-portability.md](22-cache-portability.md) | M4 |
| 23 | [23-receipts-bandit-profiles.md](23-receipts-bandit-profiles.md) | M4 |
| 24 | [24-edge-wasm-core.md](24-edge-wasm-core.md) | M3 |
| 25 | [25-cloud-multitenant.md](25-cloud-multitenant.md) | M4 — *filename historical; content is self-hosted auth only* |
| 26 | [26-agent-loop-aware.md](26-agent-loop-aware.md) | M4 |
| 27 | [27-transport-layer-consolidation.md](27-transport-layer-consolidation.md) | M3 — **transport layer** |

### Architecture (code truth)

| Doc | Purpose |
|---|---|
| [../architecture/runtime.md](../architecture/runtime.md) | Five lifecycles + module rules (updated with transport positioning) |

---

## Suggested read order for implementers

1. [FORWARD_PLAN.md](FORWARD_PLAN.md) — constraints + cut list  
2. [ARCHITECTURE_EVAL_INSIGHTS.md](ARCHITECTURE_EVAL_INSIGHTS.md) — if implementing planner/scoring/transport  
3. [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) — template  
4. Your phase doc (e.g. `27-transport-layer-consolidation.md`)  
5. [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md) — where to put code  
6. [STATUS.md](STATUS.md) — what already shipped  

---

## SDK doctrine (one paragraph)

SDKs are **thin clients**. Default: point `baseURL` at the proxy. Advanced (edge): call `@lattice/core-wasm` or `lattice-core-py`. **Never** reimplement reverse-pass, IR fingerprint, chunk buffer, or transforms in SDK source. See Phases 13, 17, 24.
