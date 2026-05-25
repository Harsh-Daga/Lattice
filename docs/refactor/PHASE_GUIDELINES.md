# Phase Document Guidelines (Phases 13–34)

> **Mandatory.** Every forward-plan phase doc (`13-*.md` … `34-*.md`) must follow this template. Reviewers reject PRs whose phase doc does not comply. CI gates referenced here landed in [Phase 13](13-honesty-pass.md).

---

## 1. Product thesis (every phase must respect this)

**LATTICE is the transport / network layer for LLM traffic** — not “a compression tool with a proxy.”

| Layer | What it owns |
|---|---|
| **Transport** ([Phase 14](14-transport-layer-consolidation.md)) | Connections, HTTP/2 pool, retry, timeout, circuit breaker, backpressure, stream resumption, RTT metrics |
| **Protocol** | LATT binary framing, delta wire, manifest |
| **Policy (on the transport path)** | Compression transforms, cache layers, guardrails, agent memory, MCP tool output shaping, receipts |
| **Surfaces** | Proxy HTTP, gateway compat, MCP server, thin SDKs (Python + TypeScript) |

Phases **16+** that call upstream providers must include near the top:

> **Transport consumption.** This phase calls `transport.request()` — no per-adapter retry, no per-feature timeout, no per-endpoint backoff. Transport policy is governed solely by `transport/policy.py` (after Phase 14).

---

## 2. The six hard constraints (non-negotiable)

From [FORWARD_PLAN.md](FORWARD_PLAN.md). Every phase doc’s opening blockquote must address all six.

---

## 3–6. Required blockquote, sections, SDK rules

(Unchanged structure — see prior template in repo history.)

---

## 5. SDK phases (16, 20, 31) — extra rules

- **Proxy mode (default):** SDK sets `baseURL` to the proxy. Zero algorithm code.
- PR must pass `scripts/check_sdk_no_algorithm_duplication.sh`.

---

## 6. Registry and CI (every phase PR)

1. Update [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md).
2. Update [CODE_BUDGET.txt](CODE_BUDGET.txt) **directory caps** when a phase owns a directory (ratchet down with justification — do not add per-phase net LoC keys).
3. Ensure `scripts/check_code_budget.sh` and `scripts/check_internal_no_duplication.sh` pass in `refactor-gate.yml`.

---

## R5. Code budget schedule (Phase 13+)

| Flag | Value | Meaning |
|---|---|---|
| `enforce_dir_caps` | **1** | Per-directory caps in `CODE_BUDGET.txt` (baseline +5% at Phase 13). |
| `enforce_total_v2` | **0** until Phase 31 | **Must flip to 1** no later than Phase 31; Rust core migrates algorithms out of `src/lattice/` until total ≤ 35 000. |
| Per-phase net LoC | **removed** | Do not declare `phase_NN_*` deltas — use dir caps, 800-LoC/file, no-dup gates, and footprint tests instead. |

**Shrink targets (narrative, not CI math):** Phase 14 consolidates adapter transport; Phase 31 moves algorithms to `crates/lattice-core/`. Phase 13 structural splits may net positive LoC — that is acceptable when caps and no-dup gates pass.

---

## 7. Execution order (authoritative)

```
M2 — v1.1
  13  Honesty pass (shipped on branch)
  → 14  Transport consolidation   ← NEXT
  → 15  Chaos contract
  → 17  Cache | 18  Guardrails | 16  SDK | 19  OTel

M3 — v1.5
  → 20  TS SDK → 21  MCP | 22  Compression | 23  Segment | 24  Non-chat → 25  Competitive bench

M4 — v2.0
  26  Memory → 27  Portability → 28  Receipts | 29  Bandit | 30  Profiles
  → 31  Rust core → 32  Auth → 33  Threat model → 34  Release
```

**Why Phase 14 immediately after 13:** Every later HTTP surface should call the canonical transport layer, not per-adapter copies.

---

## 8. Filename note

| File | Note |
|---|---|
| `32-cloud-multitenant.md` | Historical filename. Self-hosted auth only — no SaaS. |

No fractional phase numbers; segment planning is Phase 23 (`23-segment-aware-planning.md`).
