# LATTICE v2.0 Forward Plan — Master Index

> **Thesis.** LATTICE is **the transport / network layer for LLM traffic.** Compression is one capability of that layer (the most-marketed one), but the system also owns: connection pooling and HTTP/2 multiplexing to providers, unified retry + circuit breaker policy, binary framing (LATT) and delta encoding, token-aware congestion control (TACC), session continuity, stream resumption, end-to-end backpressure, layered caching, native guardrails, telemetry, MCP federation, agent-loop memory management. A single self-hosted binary sits between the user's app and their chosen LLM provider, owning every byte that crosses the boundary.
>
> Think of it the way TCP/HTTP/2/QUIC sit between an application and the network: invisible when it works, deterministic about what it does, replaces ad-hoc retry/throttle/timeout code scattered across every client.
>
> **Scope.** Everything after the v1.0.0 refactor (Phases 0-11) is captured in the per-phase docs below.
>
> **Six hard constraints, in priority order.** These override every previously-stated design decision and reshape every phase doc. Every constraint has a CI gate.
>
> 1. **Lightweight.** The base install runs on a 4 GB laptop. No required model downloads. No background processes besides the proxy itself. Idle RSS < 100 MB, under load < 200 MB. *Enforced by:* `tests/integration/footprint/test_{4gb_laptop,2gb_vps,cold_start}.py`.
> 2. **No external LLM dependency** beyond the one the user is already calling. We never load our own LLM at runtime; we never spin up our own embedding service; we never require an outside API key beyond what the user is paying for already. If a feature needs an embedding or a small model, it uses the *user's* configured provider's cheap model — no new dependency, no new bill. *Enforced by:* `tests/contract/test_default_install_no_external_services.py`, `tests/contract/test_default_install_no_models.py`.
> 3. **Open source self-hostable. No cloud product.** There is no `lattice.cloud`, no hosted offering, no Stripe, no Terraform-managed managed-service deployment. Everything ships as a Python package + Docker image + npm package. Optional self-hosted features (auth, virtual keys, quotas) exist for small teams running one shared proxy; they are never required.
> 4. **One algorithm, one implementation — across SDKs AND internally.** Reverse-pass, alias substitution, IR fingerprinting, streaming chunk buffering, transforms, retry policy, cost estimation, config loading, session state, framing, congestion control — each has exactly one home. Other surfaces (TypeScript SDK, edge runtime, optional native Python acceleration, all 17 provider adapters) consume that one implementation. *Enforced by:* `scripts/check_sdk_no_algorithm_duplication.sh` (SDK side), `scripts/check_internal_no_duplication.sh` (internal side, Phase 12 + [Phase 27](27-transport-layer-consolidation.md)), and the authoritative registry [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md).
> 5. **Codebase shall not grow unbounded.** `src/lattice/` LoC budget at v2.0 is **35 000 lines** (currently ~22 000). Each phase declares its delta in its doc; CI compares actual delta against declared budget and blocks merge on overrun. Per-directory caps prevent silent inflation. *Enforced by:* `scripts/check_code_budget.sh` (Phase 12 lands the script; every later phase respects it).
> 6. **Transport-first design.** Anything that crosses the proxy↔provider or client↔proxy boundary is the transport layer's responsibility. Retry, timeout, circuit-breaker, backpressure, framing, compression, multiplexing, observability — one configurable surface, one canonical implementation, one set of metrics. No per-adapter retry code. No per-endpoint timeout. No per-feature framing. *Established by:* [Phase 27 — Transport Layer Consolidation](27-transport-layer-consolidation.md).

---

## 1. What this plan brutally cut from the prior draft

The previous version of this index recommended things that violate the constraints above. Honest list of what's been cut and why:

| Cut | Why |
|---|---|
| Local SentenceTransformers as the default embedding backend in Phase 14 | ~80 MB model + ~1.5 GB install (torch). Default is now: embedding tier is **opt-in** and uses the **user's provider** (`openai/text-embedding-3-small`, etc.) — no new dependency, no model download. |
| Presidio as the default PII detector in Phase 15 | ~1 GB install with spaCy + model. Default is now the rule-based regex detector (zero deps, ships in base). Presidio stays as a clearly-labeled opt-in extra for users who need higher recall. |
| ONNX DeBERTa injection classifier "lazy-downloaded on first use" | 5 MB model + ~80 MB onnxruntime install. The heuristic phrase detector (zero deps) is now the default; the ONNX classifier is opt-in. |
| LLMLingua-2 implied as part of the standard compression story | ~500 MB ONNX model + onnxruntime. Now: explicit heavy opt-in extra (`pip install lattice-transport[llmlingua]`) with an explicit "this downloads 500 MB" warning at first run. Default compression story stays with our deterministic transforms — those already win the structural cases. |
| Per-tenant LoRA distillation pipeline (Phase 25) | Requires GPU + training infra + corpora object storage. This is a research lab, not a product. Bandit-based online adaptation (Phase 23) covers 80% of the value with zero ML infra. |
| Hosted `lattice.cloud`, Stripe billing, Cloudflare/Neon/Upstash Terraform | Pivots LATTICE into a SaaS product. You don't want that. Open-source self-hosted only. |
| Hosted playground at `/lattice/playground` (Phase 25) | Was bundled with cloud. A local playground (just the proxy serving a static page for debugging) survives as a small dev-only utility in Phase 26, off by default. |
| Per-SDK reimplementation of reverse-pass / hooks / streaming (Phase 13, 17) | Was the largest drift surface in the prior plan. Replaced by the single-core architecture in Phase 24. SDKs in proxy mode become URL redirectors with zero algorithm code; in in-process mode they call the shared core. |
| VSCode-marketplace publish ceremony, Helm chart with HPA/PDB/NetworkPolicy, signed-wheel sigstore ritual (Phase 26) | Vendor-relations and ops-team theatre, not user value. Trimmed to: Docker image + PyPI wheel + npm package + Cursor extension as a sideload `.vsix`. That's it. |

What survives is everything that actually moves dollars or risk on a user's spreadsheet without adding install-cost or operational complexity.

---

## 2. The "would they pay?" / "can they install it on a laptop?" matrix

Every remaining phase passes both tests:

| Phase | Doc | User benefit (the "would they pay?" claim) | Footprint added |
|---|---|---|---|
| 12 | [Honesty Pass](12-honesty-pass.md) | Codebase says what it does; nothing dead; ExecutionPlan/Scoring/registry collapsed | -2300 LoC (net shrink) |
| 13 | [SDK Thin-Client Architecture](13-python-sdk-quality.md) | `py.typed`, typed surfaces, sync wrappers, hooks, README quick-start that actually works — **with zero algorithm duplication** | + ~1 MB to Python wheel |
| 14 | [Hybrid Semantic Cache (lightweight)](14-hybrid-semantic-cache.md) | 4-tier cache; first 3 tiers (exact / IR fingerprint / Jaccard) always on with no extra deps; embedding tier opt-in via user's provider | 0 new deps default; +Redis/pgvector if user wants distributed |
| 15 | [Native Guardrails (lightweight)](15-native-guardrails.md) | Reversible PII tokenization, injection detection, JSON repair — all with rule/heuristic defaults (no model downloads); Presidio/ONNX opt-in | 0 new deps default |
| 16 | [OpenTelemetry GenAI](16-otel-genai.md) | One env var sends standard `gen_ai.*` spans to Datadog / Honeycomb / Jaeger / Phoenix | + `opentelemetry-sdk` (~3 MB) if enabled |
| 17 | [TypeScript SDK (thin client)](17-typescript-sdk.md) | `@lattice/sdk` for Node/Bun/Deno/Workers; wraps OpenAI/Anthropic/Vercel AI; **calls the proxy or the shared WASM core — no duplicated logic** | npm bundle ≤ 25 KB gzipped (edge) |
| 18 | [MCP-Native Gateway](18-mcp-native-gateway.md) | Federate upstream MCP servers, compress tool output via existing transforms, cache tool calls, scan tool output for injection | Base install (already has FastAPI + httpx) |
| 19 | [Compression Intelligence (lightweight default)](19-compression-intelligence.md) | Streaming-native compression on response chunks, tool-result diffing, JSON structural repair — all zero-dep; LLMLingua-2 separate opt-in with explicit warning | 0 new deps default; +500 MB if LLMLingua opted in |
| 19.5 | [Segment-Aware Planning](19.5-segment-aware-planning.md) | Per-section transform policies fix `features not reached by pipeline`; utility-aware beam respects segment distortion budgets | 0 new deps |
| 20 | [Non-Chat Surfaces](20-non-chat-surfaces.md) | `/v1/embeddings` (input dedup + cache reusing Phase 14's user-provider strategy), real OpenAI/Anthropic Batch APIs, audio + realtime + files | Base install |
| 21 | [Agent Memory (lightweight)](21-agent-memory.md) | Context GC + summarization + token budget + inference-aware retry — relevance scoring is rule-based by default, embedding-based optional via user's provider | 0 new deps default |
| 22 | [Cache Portability](22-cache-portability.md) | Cache survives **user-initiated** provider switches; cold-start warmer; KV-cache compatibility analyzer | Base install |
| 23 | [Receipts + Bandit + Profiles + Hot Reload](23-receipts-bandit-profiles.md) | HMAC-signed audit receipts; Thompson-sampling bandit (pure numpy — no ML deps); per-route profiles; SIGHUP reload | +`pyjwt` (~100 KB) for receipts |
| 24 | [Shared Core: Rust + PyO3 + WASM](24-edge-wasm-core.md) | **The architectural keystone.** One Rust core compiled to PyO3 wheel (3-5× Python speedup, optional) + WASM (powers Edge SDK, enforces zero SDK duplication) | Optional native wheel ~3 MB; WASM ≤200 KB |
| 25 | [Optional Self-Hosted Auth, Keys, Quotas](25-cloud-multitenant.md) | For small teams running one shared proxy: API-key auth, virtual-key indirection, per-key quotas, optional Postgres for shared state. **No cloud product. No SaaS. No Stripe.** | All optional; SQLite by default, Postgres if user wants |
| 26 | [Agent-Loop-Aware Compression, Cursor Visualizer, Minimal Release](26-agent-loop-aware.md) | Per-step compression profiles inside agent loops; Cursor extension that visualizes what LATTICE did; release as Docker + PyPI + npm + sideload Cursor `.vsix` | Cursor extension ~200 KB |
| 27 | [Transport Layer Consolidation](27-transport-layer-consolidation.md) | **The phase that makes "LATTICE is the transport layer for LLMs" true.** Unified retry + circuit breaker, HTTP/2 multiplexing, connection pooling per provider, transport-level metrics (RTT / queue depth / breaker state / in-flight), end-to-end backpressure, stream resumption on connection drop. Kills duplicated retry/timeout code across 17 adapters. | Net **-1500 LoC** (consolidation); +0 new deps (httpx already supports HTTP/2 via `h2`) |

---

## 3. The lightweight footprint budget

We commit to numbers, not adjectives.

### 3.1 Install footprints

| What you install | Size on disk | RAM at idle | RAM under load | New runtime deps |
|---|---|---|---|---|
| `pip install lattice-transport` (default) | ≤ 25 MB | ≤ 80 MB | ≤ 200 MB | FastAPI, httpx, pydantic, tiktoken, xxhash, regex, orjson |
| `+[redis]` | +1 MB | +5 MB | +15 MB | redis-py |
| `+[embeddings-provider]` | +0 MB | +0 MB | +0 MB (uses user's provider) | numpy |
| `+[embeddings-local]` (opt-in, **discouraged on small machines**) | +1.5 GB | +500 MB | +800 MB | sentence-transformers, torch |
| `+[pii]` (Presidio) | +1 GB | +250 MB | +400 MB | presidio + spaCy |
| `+[injection]` (ONNX classifier) | +90 MB | +80 MB | +120 MB | onnxruntime |
| `+[llmlingua]` (heavy compression) | +600 MB | +400 MB | +700 MB | onnxruntime + transformers + model |
| `+[otel]` | +5 MB | +10 MB | +20 MB | opentelemetry-sdk + exporters |
| `+[native]` (Rust acceleration) | +3 MB | +5 MB | +5 MB (cuts CPU usage 3-5×) | `lattice-core-py` |
| `+[auth]` (self-hosted multi-user) | +2 MB | +5 MB | +15 MB | argon2-cffi, pyjwt |
| Docker image (default) | ≤ 120 MB | same as above | same | python:3.13-slim |
| npm `@lattice/sdk` | ≤ 25 KB gzipped (edge) | n/a | n/a | zero runtime deps (peer-only) |
| `@lattice/core-wasm` (for edge in-process compression) | ≤ 200 KB gzipped | ≤ 5 MB | ≤ 8 MB | none |

### 3.2 Concrete laptop targets

LATTICE must run unmodified on:

- **4 GB RAM laptop** — default install, proxy + 1 small concurrent workload. Validated by `tests/integration/footprint/test_4gb_laptop.py` which runs the proxy with `ulimit -v 1572864` (1.5 GB virtual memory cap) and a fixture workload, asserting no OOM.
- **2 GB Raspberry Pi 4 / cheap VPS** — proxy alone (no concurrent compression). Validated by `tests/integration/footprint/test_2gb_vps.py` with `ulimit -v 786432`.
- **Cold start ≤ 1.5 s** to first served request, default install. Validated by `tests/integration/footprint/test_cold_start.py`.

CI gate: every PR runs the 4 GB and 2 GB footprint tests; over-budget changes block merge unless explicitly opt-in (`[llmlingua]` / `[embeddings-local]` / `[pii]` extras are excluded from the footprint test).

---

## 4. The single-core SDK architecture (the doctrine)

Every previous "SDK quality" plan reimplemented algorithms. We're done with that.

```
                  ┌────────────────────────────────────────────────────────┐
                  │  crates/lattice-core (Rust, no_std-compatible)        │
                  │  - PromptIR canonical builder + fingerprint            │
                  │  - reference_sub, path_prefix, format_conv, framing    │
                  │  - AliasTable + reverse-substitution                   │
                  │  - Streaming chunk buffer (sliding-window placeholder) │
                  └─────┬──────────────────────────────────────────┬──────┘
                        │                                          │
                ┌───────▼──────────┐                       ┌───────▼────────┐
                │ PyO3 binding     │                       │ WASM binding    │
                │ lattice-core-py  │                       │ @lattice/core-  │
                │   (optional      │                       │ wasm            │
                │    Python wheel) │                       │                 │
                └───────┬──────────┘                       └───────┬────────┘
                        │                                          │
         ┌──────────────▼─────────────┐                ┌───────────▼─────────┐
         │ src/lattice/* (Python)     │                │ packages/typescript-│
         │   - Pipeline runner        │                │ sdk (TypeScript)    │
         │   - Proxy server           │                │   - LatticeClient   │
         │   - Pure-Python fallbacks  │                │     (HTTP)          │
         │     for everything in core │                │   - wrapOpenAI etc. │
         │                            │                │   - (in-process: WASM)│
         │   = the canonical engine   │                │   = thin client     │
         └────────────────────────────┘                └─────────────────────┘
```

### 4.1 Two operating modes per SDK

| Mode | What the SDK does | Who handles compression |
|---|---|---|
| **Proxy mode** (default for everyone) | SDK is a thin HTTP client. Sets the upstream `baseURL` to the LATTICE proxy. That's the entire SDK responsibility. | The Python proxy runs the full pipeline server-side. Reverse-pass happens server-side and streams already-decoded chunks back. **The SDK has zero algorithm code.** |
| **In-process mode** (advanced; only when there's no proxy) | Python: calls into `src/lattice/*` directly (or `lattice-core-py` for native speed). TypeScript: lazy-loads `@lattice/core-wasm` and calls into WASM exports. | The shared core. The TS SDK does not parse SSE itself, does not do reverse-pass itself — it calls `core.streaming_decode(chunk, alias_table_handle)` and yields the result. |

If a TypeScript user has no proxy AND no WASM core installed AND no embedding-tier dependency, **the SDK skips compression entirely and forwards the request unchanged with a warning header**. Graceful degradation: never silently re-implement a Python algorithm in TypeScript "to be helpful".

### 4.2 The drift-prevention CI gate

A new CI test grep-checks every algorithm primitive name across SDK code:

```bash
# scripts/check_sdk_no_algorithm_duplication.sh
# fails if any SDK source file contains a literal implementation of:
forbidden_patterns=(
  "function reverseSubstitute"         # must call core
  "function buildCanonicalIR"
  "function applyReferenceSub"
  "class ChunkBuffer"                  # must come from WASM core
  "function xxh3"
)
```

The check runs on `packages/typescript-sdk/src/**` and `bindings/wasm/pkg/`. Implementations of these primitives are only allowed in `crates/lattice-core/` and `src/lattice/` (Python fallback). Drift = build break.

---

## 5. Milestones (revised)

### M2 — Honest & Lightweight (v1.1) — Phases 12-16

Goal: a credible v1.1 that runs on any laptop. Every name is honest, the SDK quick-start works, cache hits 3-5× more often **with zero new model downloads**, PII never reaches the provider via rule-based detection, telemetry is standards-compliant.

Footprint at end of M2: identical to v1.0 (≤ 25 MB default install).

Target: 4 weeks (faster than before because the heavy ML work is descoped to optional extras).

### M3 — Differentiate (v1.5) — Phases 17-20

Goal: TypeScript / edge users get parity via the thin-client SDK; agent users get MCP federation with compressed tool output; non-chat surfaces (embeddings, batch, audio, realtime, files) work; compression intelligence improves for the workloads that ask for it (with LLMLingua as a flagged opt-in).

Footprint at end of M3: still ≤ 25 MB default; optional LLMLingua adds 600 MB only if explicitly enabled.

Target: 6 weeks.

### M4 — Top-tier (v2.0) — Phases 21-27

Goal: agents are first-class with context GC and inference-aware retry; cache portability survives user-initiated provider switches; bandit + receipts + per-route profiles make the system self-improving and auditable; Rust/WASM core ships and powers all SDKs; optional self-hosted auth covers small-team deployments; Cursor users see what LATTICE did per request; **transport layer consolidates into one coherent surface (Phase 27) — this is what turns LATTICE from "compression tool" into "transport layer for LLMs".**

Footprint at end of M4: ≤ 28 MB default. Native Rust wheel adds optional 3 MB for 3-5× speedup. **Codebase net-smaller than M3 thanks to Phase 27's consolidation** (-1500 LoC of duplicated transport code across adapters).

Target: 10 weeks (was 8 — added Phase 27).

---

## 6. Non-negotiable design rules (now enforced)

These constraints are CI-enforced where possible. Reviewer must reject any PR that violates.

| Rule | Enforcement |
|---|---|
| **No multi-provider routing.** | Contract test `tests/contract/test_no_multi_provider_routing.py` greps for `provider != ` patterns outside Phase 22's cache re-key. |
| **No algorithm code in SDKs.** | `scripts/check_sdk_no_algorithm_duplication.sh` (above). |
| **No internal duplication of canonical primitives.** | `scripts/check_internal_no_duplication.sh` walks every entry in [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md) and verifies the symbol exists in exactly the declared file. Adds two new patterns per phase that introduces a new primitive. |
| **No required model download.** | `tests/contract/test_default_install_no_models.py` verifies the default install never writes to `assets/models/` at startup. |
| **No required external service.** | `tests/contract/test_default_install_no_external_services.py` runs the proxy with all network blocked except the configured upstream provider; default config must boot. |
| **Code budget.** | `scripts/check_code_budget.sh` (lands with Phase 12): (a) `src/lattice/` total LoC ≤ declared per-phase budget; (b) per-directory caps; (c) net delta on every PR must match the phase doc's declared `LoC delta`. Phase docs without a declared delta block merge. |
| **No file over 800 LoC.** | Existing R6 enforcement (file split required when a single file grows past 800 lines). |
| **No raw user content in receipts/headers/spans by default.** | `tests/contract/test_no_user_content_in_telemetry.py` runs a sample workload, captures all telemetry, asserts no message-text substring leaks. |
| **Footprint budget.** | `tests/integration/footprint/test_4gb_laptop.py` and `test_2gb_vps.py` block merge on regression. |
| **Reversibility property.** | Property test in `tests/unit/transforms/test_reversibility_property.py` — for any transform, `reverse(apply(x)) == x` modulo allowed lossy fields. |
| **Transport policies live in one place.** | After [Phase 27](27-transport-layer-consolidation.md): retry/timeout/circuit-breaker/backpressure logic exists exactly once in `src/lattice/transport/`. CI gate `tests/contract/test_transport_unification.py` greps for `httpx.AsyncClient(` outside the transport package — block merge on hit (adapters must go through the unified transport). |

### 6.1 Code budget (per directory)

| Directory | Today (~) | v2.0 cap | Notes |
|---|---|---|---|
| `src/lattice/core/` | 600 | 1 200 | Leaf primitives only. |
| `src/lattice/ir/` | 2 500 | 3 500 | Canonical IR + builder + validation. |
| `src/lattice/transforms/` | 4 200 | 6 500 | Includes optimizers + tool_diff + llmlingua (opt-in). |
| `src/lattice/pipeline/` | 1 800 | 2 500 | Runner + gates + streaming. |
| `src/lattice/planner/` | 1 500 | 2 800 | UnifiedPlanner + segment policy (19.5) + bandit. |
| `src/lattice/cache/` | 700 | 2 500 | Layered cache + portability + warmer. |
| `src/lattice/safety/` | 400 | 2 000 | PII + injection + output repair. |
| `src/lattice/agent/` | 0 | 2 000 | New in Phase 21 + 26. |
| `src/lattice/transport/` | 1 200 | 3 000 | **Net-smaller after Phase 27** — consolidates adapter duplication into one place. |
| `src/lattice/providers/adapters/` | 4 800 | **2 500** | **Halves** after Phase 27 strips per-adapter retry/timeout/transport code. |
| `src/lattice/proxy/` | 1 600 | 2 200 | Server + middleware + admin routes (Phase 25). |
| `src/lattice/gateway/` | 2 200 | 3 200 | Compat splits + new endpoints (Phase 20). |
| `src/lattice/integrations/` | 1 200 | 1 800 | Tunnel + doctor + mutation_store + agent_stats. |
| `src/lattice/telemetry/` | 1 100 | 1 700 | Add `otel/` (Phase 16). |
| `src/lattice/state/` | 700 | 900 | Session + store + segment_store. |
| `src/lattice/audit/` | 0 | 800 | New in Phase 23. |
| `src/lattice/auth/` + `keys/` + `quotas/` + `tenants/` | 0 | 2 800 | New in Phase 25; **optional**. |
| `src/lattice/mcp/` | 0 | 1 500 | New in Phase 18. |
| **TOTAL `src/lattice/`** | **~22 000** | **≤ 35 000** | Hard cap. |
| `bindings/python/` | 0 | 1 000 | PyO3 binding (Phase 24). |
| `bindings/wasm/` | 0 | 500 | WASM binding (Phase 24). |
| `crates/lattice-core/` (Rust) | 0 | 6 000 | Rust core (Phase 24). |
| `packages/typescript-sdk/src/` | 0 | 2 500 | TS thin client (Phase 17). |
| `tools/cursor-extension/src/` | 0 | 800 | Cursor visualizer (Phase 26). |

If a phase needs to exceed its directory cap, the doc must justify in §1 ("Why this phase exists") and the cap is bumped in this table as part of the same PR. **No silent inflation.**

---

## 7. What we deliberately don't build

| Feature | Why not |
|---|---|
| Multi-provider speculative decoding | Multi-provider routing. |
| Budget-aware automatic provider selection | Multi-provider routing. |
| Hosted `lattice.cloud` | You don't want it. |
| Stripe billing integration | No SaaS. |
| Per-tenant LoRA fine-tuning pipeline | Requires GPU + cloud infra. Bandit covers 80% of value with zero infra. |
| Local LLM-judge in the hot path | Not lightweight. Benchmark-only. |
| Bundled local embedding model as default | Not lightweight (~1.5 GB). User's provider gives embeddings at zero new install cost. |
| Bundled local Whisper / TTS | Provider passthrough only. |
| Vector DB | Out of scope — defer to specialized products (pgvector / Qdrant / Redis vector). |
| Image / video understanding compression | Out of scope. |
| VSCode marketplace listing / Helm chart with NetworkPolicy / signed-wheel sigstore ceremony | Ops-team theatre. Docker + PyPI + npm + sideload Cursor extension is enough. |

---

## 8. How to execute

Each phase doc is now structured around three constraints, prominent on every page:

> **Footprint impact:** what this phase adds to the default install / the lean install / the heavy-extras install.
> **Algorithm location:** where the canonical implementation lives. SDKs / runtimes consume from here; never duplicate.
> **External-service requirement:** none required by default; optional integrations clearly labeled.

A senior engineer can take any phase and ship it without reading the others. The drift-prevention CI gate keeps SDKs from accidentally reimplementing logic. The footprint CI gate keeps default install lightweight.

---

## 9. Suggested execution order

**Authoritative sequence** — also in [PHASE_GUIDELINES.md §7](PHASE_GUIDELINES.md). Every phase doc must comply with [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) (opening blockquote template + registry updates).

```
M2 — Credibility (v1.1)
  12  Honesty + CI gates (code budget, internal dedup, SDK dedup shell)
  → 14  Cache (lightweight)
  → 15  Guardrails (lightweight)
  → 13  Python SDK (thin client)
  → 16  OTel (orthogonal)

M3 — Differentiate (v1.5)
  27  Transport consolidation FIRST     ← one retry/pool/breaker layer before more HTTP surfaces
  → 24  Shared core (Rust / PyO3 / WASM)
  → 17  TypeScript SDK (thin client)
  → 18  MCP  |  19  Compression intel  |  20  Non-chat   (parallel after 17)
  → 19.5 Segment-aware planning        ← after 19; fixes orchestration gaps from v1.0 evals

M4 — Top-tier (v2.0)
  21  Agent memory
  → 22  Cache portability
  → 23  Receipts + bandit + profiles + hot reload
  → 25  Optional self-hosted auth (no SaaS)
  → 26  Agent-loop + Cursor visualizer + release
```

One engineer full-time: M2 ~4 weeks, M3 ~6 weeks, M4 ~8 weeks. Two engineers: roughly halve.

**Doc index:** [docs/refactor/README.md](README.md) · **LoC caps:** [CODE_BUDGET.txt](CODE_BUDGET.txt)

---

## 10. Eval-driven architecture notes (May 2026)

External architecture review + `v1.0.0.json` production evals are consolidated in [ARCHITECTURE_EVAL_INSIGHTS.md](ARCHITECTURE_EVAL_INSIGHTS.md):

- **Accept:** utility scoring, segment-aware planning (19.5), validation facade (12), transport/cache inputs to planner (22, 27)
- **Reject:** second “refound” deleting legacy pipeline (already removed), mutable-candidate rewrite, compression-% as sole KPI

Runtime law: [docs/architecture/runtime.md](../architecture/runtime.md).

---

## 11. References

- [ARCHITECTURE_EVAL_INSIGHTS.md](ARCHITECTURE_EVAL_INSIGHTS.md) — eval critique → phase mapping
- [REFACTOR_PLAN.md](REFACTOR_PLAN.md) — the v1.0.0 refactor master (Phases 0-11)
- [STATUS.md](STATUS.md) — what shipped and when (updated with forward-plan reference + lightweight + transport-layer constraints)
- [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md) — **the authoritative registry of every primitive and its one canonical file.** CI gates check against this doc on every PR.
- [FINAL_LAYOUT.md](FINAL_LAYOUT.md) — the post-v1.0 file layout (Phase 24 extends with `crates/` and `bindings/`; Phase 27 consolidates `providers/transport/`)
- [MIGRATION.md](MIGRATION.md) — user-facing migration guide; each phase updates the section relevant to its renames/dep changes
- [docs/architecture/runtime.md](../architecture/runtime.md) — the architecture all phases extend
