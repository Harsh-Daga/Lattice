# Refactor & Forward Plan — Documentation Index

> **Start here** after v1.0.0 (Phases 0–11 shipped). **v1 Phase 12** = docs release (`11-docs-release.md`). **v2 forward plan** = Phases **13–34** ([FORWARD_PLAN.md](FORWARD_PLAN.md)).

---

## Product thesis

**LATTICE is the transport / network layer for LLM traffic.** One self-hosted proxy owns connections, retries, timeouts, backpressure, framing, streaming, caching, guardrails, compression, and observability.

---

## Six constraints (every phase must obey)

| # | Constraint | Enforced by |
|---|---|---|
| 1 | Lightweight (4 GB laptop, no required models) | `tests/integration/footprint/` |
| 2 | No external LLM beyond user's provider | `tests/contract/test_default_install_no_*` |
| 3 | Open source self-hosted only (no SaaS) | Plan + review |
| 4 | One algorithm, one implementation | [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md) + dedup scripts |
| 5 | Codebase cap ≤ 35k LoC `src/lattice/` | [CODE_BUDGET.txt](CODE_BUDGET.txt) + `scripts/check_code_budget.sh` |
| 6 | Transport-first (unified transport layer) | [14-transport-layer-consolidation.md](14-transport-layer-consolidation.md) |

Full detail: [FORWARD_PLAN.md](FORWARD_PLAN.md).

---

## Document map

| Doc | Purpose |
|---|---|
| [STATUS.md](STATUS.md) | Phases 0–12 shipped + v2 summary |
| [FORWARD_PLAN.md](FORWARD_PLAN.md) | Master index Phases 13–34 |
| [MIGRATION-v1-to-v2.md](MIGRATION-v1-to-v2.md) | User-visible v1.x → v2 changes |
| [15-chaos-failure-modes.md](15-chaos-failure-modes.md) | Chaos contract (Phase 15) |
| [25-competitive-benchmark.md](25-competitive-benchmark.md) | Competitive benchmark (Phase 25) |
| [33-threat-model.md](33-threat-model.md) | Threat model (Phase 33) |
| [PHASE_COMPLETION_TRACKER.md](PHASE_COMPLETION_TRACKER.md) | Acceptance vs phase docs |
| [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md) | Primitive registry (CI data) |
| [CODE_BUDGET.txt](CODE_BUDGET.txt) | LoC caps + `enforce_*` flags |

### v2.0 forward phases (13–34)

| Phase | Doc | Milestone |
|---|---|---|
| 13 | [13-honesty-pass.md](13-honesty-pass.md) | M2 — shipped on branch |
| 14 | [14-transport-layer-consolidation.md](14-transport-layer-consolidation.md) | M2 — **next** |
| 15 | [15-chaos-failure-modes.md](15-chaos-failure-modes.md) | M2 |
| 16 | [16-python-sdk-quality.md](16-python-sdk-quality.md) | M2 |
| 17 | [17-hybrid-semantic-cache.md](17-hybrid-semantic-cache.md) | M2 |
| 18 | [18-native-guardrails.md](18-native-guardrails.md) | M2 |
| 19 | [19-otel-genai.md](19-otel-genai.md) | M2 |
| 20–34 | … | See [FORWARD_PLAN.md §2](FORWARD_PLAN.md) |

---

## Renumbering table (old v2 doc → new)

| Old | New | Topic |
|---|---|---|
| 12-honesty-pass | 13-honesty-pass | Honesty + CI gates |
| 27-transport | 14-transport | Transport consolidation |
| (new) | 15-chaos | Failure-mode contract |
| 13-sdk | 16-sdk | Python SDK |
| 14-cache | 17-cache | Hybrid cache |
| 15-guardrails | 18-guardrails | Guardrails |
| 16-otel | 19-otel | OTel |
| 17-ts | 20-ts | TypeScript SDK |
| 18-mcp | 21-mcp | MCP |
| 19-compression | 22-compression | Compression intel |
| old segment phase | 23-segment | Segment planning |
| 20-non-chat | 24-non-chat | Non-chat |
| (new) | 25-competitive | Competitive benchmark |
| 21-memory | 26-memory | Agent memory |
| 22-portability | 27-portability | Cache portability |
| 23-omnibus | 28 / 29 / 30 | Receipts / bandit / profiles |
| 24-wasm | 31-wasm | Rust core |
| 25-auth | 32-auth | Self-hosted auth |
| (new) | 33-threat | Threat model |
| 26-release | 34-release | Agent-loop + release |

**Unchanged:** v1 Phase 12 = `11-docs-release.md` (v1.0.0 docs release).
