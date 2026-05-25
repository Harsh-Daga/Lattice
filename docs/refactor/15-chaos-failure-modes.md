# Phase 15 — Chaos & Failure-Mode Contract

> **Footprint impact.** 0 `src/lattice/` LoC; +1200 LoC `tests/integration/chaos/`; +200 LoC transport probe hooks.
> **Algorithm location.** Contract table in `tests/integration/chaos/contract.py`; probes in `src/lattice/transport/` (Phase 20 surface only).
> **External-service requirement.** None (in-process toxiproxy-style TCP injector; pure Python, no new runtime dep).
> **LoC delta (declared).** 0 src; +1400 tests.
> **Transport role.** Defines MUST behaviour for every failure on the proxy↔provider path after [Phase 20](14-transport-layer-consolidation.md). This phase calls `transport.request()` — no per-adapter retry, timeout, or backoff.
> **Guidelines.** [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) — constraints 1–6.
>
> **Goal.** Evidence-backed failure-mode contract for the "transport layer for LLMs" claim.
> **Outcome.** One row per failure mode with required behaviour, span attributes, and a deterministic chaos test.
> **Estimated effort.** 4 days (1 PR, tests-only + probe hooks).

---

**Transport consumption.** This phase calls `transport.request()` — no per-adapter retry, no per-feature timeout, no per-endpoint backoff. Transport policy is governed solely by `transport/policy.py` (Phase 20).

---

## 1. The failure surface

| Failure mode | Contract (summary) | Observability (Phase 27) |
|---|---|---|
| DNS flap | MUST retry with backoff ≤ policy max; abort after exhaustion | `gen_ai.transport.dns_error` |
| TCP half-open | MUST fail fast; count as retryable | `gen_ai.transport.connect_timeout` |
| TLS renegotiation failure | MUST abort; non-retryable unless policy says otherwise | `gen_ai.transport.tls_error` |
| Provider 5xx | MUST retry per `transport/retry.py` | `gen_ai.transport.upstream_5xx` |
| Provider 429 mid-stream | MUST respect `Retry-After` / rate-limit parser | `gen_ai.transport.rate_limited` |
| Partial SSE chunk | MUST finalize or error with stream_error event | `gen_ai.stream.chunk_partial` |
| Broken UTF-8 in stream | MUST replace or abort stream with typed error | `gen_ai.stream.encoding_error` |
| Upstream close mid-token | MUST retry stream resume if enabled else error | `gen_ai.stream.upstream_reset` |
| Idle timeout vs keepalive drift | MUST use `transport/timeout.py` idle budget | `gen_ai.transport.idle_timeout` |
| IPv6 fallback | MUST follow happy-eyeballs policy in pool | `gen_ai.transport.connect_family` |
| Proxy restart in-flight | MUST return 503 to client; no partial receipt | `gen_ai.proxy.restart` |
| Client disconnect in-flight | MUST cancel upstream; no leak | `gen_ai.stream.client_abort` |
| Slow consumer backpressure | MUST apply `transport/backpressure.py` | `gen_ai.transport.backpressure` |
| Oversized response | MUST truncate or reject per policy | `gen_ai.transport.oversized` |

Full table with latency budgets lives in `tests/integration/chaos/contract.py` (consumed by CI).

---

## 2. Test harness

`tests/integration/chaos/` wraps httpx with a pure-Python TCP injector (toxiproxy-style). Each contract row has one test. No external toxiproxy daemon.

---

## 3. Out of scope

- Chaos on user↔proxy boundary (standard load tests).
- Provider-specific quirks not reachable via unified transport.

---

## 4. Acceptance criteria

1. Every row in `contract.py` has a passing test on main.
2. Failure modes emit documented `gen_ai.*` span names (schema from Phase 27).
3. No new retry/timeout code outside `src/lattice/transport/`.
