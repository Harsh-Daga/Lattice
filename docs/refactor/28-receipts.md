# Phase 28 — Compression Receipts (HMAC-Signed Audit Log)

> **Footprint impact.** + `pyjwt[crypto]` (~100 KB) for receipt signing. No ML deps.
> **Algorithm location.** `src/lattice/audit/receipts.py`, `receipt_store.py`, `receipt_router.py`. Registry: [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md) §11.
> **External-service requirement.** None. Storage reuses memory / SQLite / Postgres / Redis backends from cache and auth phases.
> **LoC delta (declared).** +900 net (`audit/`).
> **Transport role.** Receipts record `transport.request_id`, attempt count, RTT, and breaker state from [Phase 20](14-transport-layer-consolidation.md); compression fields from the pipeline only.
> **Guidelines.** [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) — constraints 1–6.
>
> **Goal.** Every successful proxy response carries a verifiable, content-free audit record for compliance and ops.
> **Outcome.** `LatticeReceipt` JWT in `x-lattice-receipt`; `GET /lattice/receipts/{id}` for retrieval; append-only stores with replay-safe verification.
> **Estimated effort.** 2 days (1 PR).

---

## 3. When this ships without the other v2 ops phases

You can deploy receipts **without** [Phase 29](29-bandit-routing.md) or [Phase 30](30-route-profiles-reload.md). Bandit and profiles only add optional fields on the receipt (`bandit_arm_pulled`, profile name). Receipt signing and storage are independent.

---

## 1. Why this phase exists

Healthcare and finance auditors need a verifiable trail of what the gateway did to a request **without** receiving user content. Per-response receipts close that gap.

---

## 2. Files touched

### Created

```
src/lattice/audit/__init__.py
src/lattice/audit/receipts.py
src/lattice/audit/receipt_store.py
src/lattice/audit/receipt_router.py
tests/unit/audit/test_receipts.py
tests/unit/audit/test_receipt_store.py
tests/integration/test_receipt_e2e.py
docs/operations/receipts.md
```

### Modified

| File | Change |
|---|---|
| `src/lattice/pipeline/runner.py` | Emit receipt at end of compress path |
| `src/lattice/proxy/middleware.py` | Attach `x-lattice-receipt` via single header emitter |
| `src/lattice/core/config.py` | `ReceiptsConfig` |

---

## 3. Receipt shape and issuance

`LatticeReceipt` is a frozen dataclass: transforms applied, cache layer hit, guardrail kinds (not values), token counts, latency envelope, transport attempt metadata. **Never** raw prompts, tool args, or model text.

Issued as HS256 JWT (RS256 optional). Nonce bound to `transport.request_id` (Phase 20).

---

## 4. Test plan

| Check | Command |
|---|---|
| Unit | `uv run pytest tests/unit/audit -q` |
| E2E | `tests/integration/test_receipt_e2e.py` |
| Content audit | `test_no_user_content_in_receipt` |
| Replay | `tests/contract/test_receipt_replay.py` (nonce + signature) |

---

## 5. Acceptance criteria

1. Every successful response includes verifiable `x-lattice-receipt`.
2. `GET /lattice/receipts/{id}` returns structured JSON; offline verification with published key works.
3. Receipt content audit passes (no user content substrings).
4. Canonical bench ±2% on pipeline hot path.

---

## 6. Out of scope

| Topic | Phase |
|---|---|
| Per-route profile overrides on receipts | [Phase 30](30-route-profiles-reload.md) |
| Bandit arm field population | [Phase 29](29-bandit-routing.md) |
| Receipt encryption at rest | Future (payload is already content-free) |
