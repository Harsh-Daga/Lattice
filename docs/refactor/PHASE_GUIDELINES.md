# Phase Document Guidelines (Phases 12–27)

> **Mandatory.** Every forward-plan phase doc (`12-*.md` … `27-*.md`) must follow this template. Reviewers reject PRs whose phase doc does not comply. CI gates referenced here are landed in [Phase 12](12-honesty-pass.md).

---

## 1. Product thesis (every phase must respect this)

**LATTICE is the transport / network layer for LLM traffic** — not “a compression tool with a proxy.”

| Layer | What it owns |
|---|---|
| **Transport** ([Phase 27](27-transport-layer-consolidation.md)) | Connections, HTTP/2 pool, retry, timeout, circuit breaker, backpressure, stream resumption, RTT metrics, rate-limit parsing |
| **Protocol** | LATT binary framing, delta wire, manifest |
| **Policy (on the transport path)** | Compression transforms, cache layers, guardrails, agent memory, MCP tool output shaping, receipts |
| **Surfaces** | Proxy HTTP, gateway compat, MCP server, thin SDKs (Python + TypeScript) |

Compression is a **policy** the transport layer applies. Phase docs must say where their work sits in this stack — never describe LATTICE as “only compression.”

---

## 2. The six hard constraints (non-negotiable)

From [FORWARD_PLAN.md](FORWARD_PLAN.md). Every phase doc’s opening blockquote must address all six (or explicitly state “unchanged / N/A” with justification).

| # | Constraint | Phase author must document |
|---|---|---|
| 1 | **Lightweight** | Footprint impact: disk, idle RSS, load RSS, new deps. Default install must stay ≤ 25 MB / ≤ 100 MB idle. |
| 2 | **No external LLM** | No required model download; embeddings/summarization use **user’s provider** when needed. |
| 3 | **Open source self-hosted** | No SaaS, Stripe, hosted cloud, Terraform managed service. |
| 4 | **One implementation** | Algorithm location: exact path(s). Update [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md). SDKs never reimplement. |
| 5 | **Code budget** | Declared **LoC delta**; directory cap from [FORWARD_PLAN.md §6.1](FORWARD_PLAN.md) or [CODE_BUDGET.txt](CODE_BUDGET.txt). |
| 6 | **Transport-first** | **Transport role** — how this phase relates to the unified transport layer (Phase 27). |

---

## 3. Required opening blockquote (copy this structure)

Every phase doc starts with `# Phase N — Title` then a blockquote containing **in this order**:

```markdown
> **Footprint impact.** …
> **Algorithm location.** … (canonical paths only; link to SINGLE_SOURCE_OF_TRUTH.md §)
> **External-service requirement.** None | user's provider only | optional …
> **LoC delta (declared).** +N / -M net; directory caps: `foo/` ≤ X
> **Transport role.** One sentence: where in the proxy↔provider path this phase runs
> **Guidelines.** Complies with [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) and [FORWARD_PLAN.md](FORWARD_PLAN.md) constraints 1–6.
>
> **Goal.** …
> **Outcome.** …
> **Estimated effort.** …
```

Phases may add phase-specific callouts (e.g. Phase 25: “There is no LATTICE cloud”) before **Goal**.

---

## 4. Required sections (body)

| Section | Required | Content |
|---|---|---|
| Why this phase exists | Yes | User pain + what changed vs prior draft (brutal honesty) |
| Files touched | Yes | Created / modified / deleted tables |
| Step-by-step | Yes | Implementable without reading other phase docs |
| Test plan | Yes | Commands + thresholds |
| Acceptance criteria | Yes | Numbered, verifiable |
| Out of scope | Yes | Explicit cuts with reason |

Optional: mermaid diagrams, code samples — encouraged for transport and SDK phases.

---

## 5. SDK phases (13, 17, 24) — extra rules

- **Proxy mode (default):** SDK sets `baseURL` to the proxy. Zero algorithm code.
- **In-process mode (advanced):** Delegates to runtime (`Pipeline`) or `@lattice/core-wasm` / `lattice-core-py`. Zero reimplementation.
- **Graceful degradation:** No proxy + no core → passthrough with warning; never silent reimplementation.
- PR must pass `scripts/check_sdk_no_algorithm_duplication.sh`.

---

## 6. Registry and CI (every phase PR)

Before merge:

1. Update [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md) for every new primitive.
2. Update [CODE_BUDGET.txt](CODE_BUDGET.txt) phase line with declared delta.
3. Add or extend contract tests listed in the phase doc.
4. Run footprint tests if touching default install path.
5. Canonical bench ±2% if touching pipeline hot path.

---

## 7. Execution order (authoritative)

Do not implement out of order without updating dependencies in the phase doc.

```
M2 — Credibility (v1.1)
  12 Honesty + CI gates (budget, internal dedup, SDK dedup shell)
  → 14 Cache (lightweight)
  → 15 Guardrails (lightweight)
  → 13 Python SDK (thin client)
  → 16 OTel (orthogonal)

M3 — Differentiate (v1.5)
  27 Transport consolidation FIRST  ← unified retry/pool/breaker before more surfaces
  → 24 Shared core (Rust/PyO3/WASM)
  → 17 TypeScript SDK (thin client)
  → 18 MCP | 19 Compression intel | 20 Non-chat  (parallel after 17)
  → 19.5 Segment-aware planning  ← after 19; see ARCHITECTURE_EVAL_INSIGHTS.md

M4 — Top-tier (v2.0)
  21 Agent memory
  → 22 Cache portability
  → 23 Receipts + bandit + profiles
  → 25 Self-hosted auth (optional)
  → 26 Agent-loop + release
```

**Why Phase 27 moved to start of M3:** Transport must be one layer before adding embeddings/batch/MCP/edge SDK features on top of 17 duplicated adapter implementations.

---

## 8. Filename note

| File | Note |
|---|---|
| `25-cloud-multitenant.md` | **Historical filename.** Content is *Optional Self-Hosted Auth* only — no cloud product. Do not add SaaS content to match the filename. |
| `19.5-segment-aware-planning.md` | Decimal phase id (between 19 and 20). Same template as integer phases. |

---

## 9. Compliance checklist (reviewer)

- [ ] Opening blockquote has all six constraint fields
- [ ] Transport role is explicit (not “compression only”)
- [ ] LoC delta declared and within CODE_BUDGET.txt
- [ ] SINGLE_SOURCE_OF_TRUTH.md updated
- [ ] No default local ML model (SentenceTransformers / Presidio / LLMLingua / ONNX) without opt-in extra + warning
- [ ] No SDK algorithm duplication
- [ ] Out of scope lists cloud/SaaS/distillation if relevant
- [ ] Acceptance criteria include footprint or “unchanged” evidence
