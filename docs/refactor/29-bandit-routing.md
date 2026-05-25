# Phase 29 — Self-Tuning Bandit Routing (Thompson Sampling)

> **Status:** Skeleton — Thompson bandit + planner wiring land in this phase's implementation PR.

> **Footprint impact.** Pure `numpy` Beta posteriors; no ML library or model download.
> **Algorithm location.** `src/lattice/planner/bandit/thompson.py`, `reward.py`, `store.py`. Registry: [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md) §4.
> **External-service requirement.** None.
> **Transport role.** Bandit selects transform allowlists only; provider calls use `transport.request()` from Phase 20.
> **Guidelines.** [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) — constraints 1–6.
>
> **Goal.** Per-tenant, per-task-class compression recipes self-tune from measured outcomes.
> **Outcome.** Thompson sampling over transform arms; reward from HTTP outcome, optional `x-lattice-feedback`, token savings, guardrail hits.
> **Estimated effort.** 2 days (1 PR).

---

## 3. When this ships without the other v2 ops phases

Bandit routing is **independent** of [Phase 28](28-receipts.md) (receipts optional) and [Phase 30](30-route-profiles-reload.md) (profiles can supply default arms). You can enable bandit with static config-only receipts disabled.

---

## 1. Why this phase exists

Static transform allowlists waste money: coding tenants need `reference_sub`; summarization tenants need `rate_distortion`. Hand-tuning per tenant does not scale.

---

## 2. Files touched

### Created

```
src/lattice/planner/bandit/__init__.py
src/lattice/planner/bandit/thompson.py
src/lattice/planner/bandit/reward.py
src/lattice/planner/bandit/store.py
tests/unit/planner/bandit/
tests/integration/test_bandit_e2e.py
docs/operations/bandit_tuning.md
```

### Modified

| File | Change |
|---|---|
| `src/lattice/planner/unified_planner.py` | Consult bandit for allowlist when enabled |
| `src/lattice/pipeline/runner.py` | Record arm on context for receipts |

---

## 3. Arms, Thompson sampling, reward

Arms are named transform sets (`safe-minimal`, `chat-default`, `code-heavy`, …). Beta(α, β) posterior per (tenant, task_class, arm). Cold-start: operator default arm for first 100 observations.

Reward ∈ [0, 1] from HTTP OK, user feedback header, token reduction, guardrail cleanliness, latency budget.

---

## 4. Test plan

| Check | Command |
|---|---|
| Unit | `uv run pytest tests/unit/planner/bandit -q` |
| Convergence | `tests/integration/test_bandit_e2e.py` — optimal arm ≥90% after 500 pulls |

---

## 5. Acceptance criteria

1. After 500 observations per arm in simulator, bandit selects highest-reward arm ≥90% of pulls.
2. Exploration cap prevents re-selecting arms with >5% measured quality regression.
3. No duplicate bandit implementation outside `planner/bandit/`.

---

## 6. Out of scope

| Topic | Phase |
|---|---|
| Cross-tenant transfer learning | Future (privacy) |
| Per-route profile defaults | [Phase 30](30-route-profiles-reload.md) |
| LoRA / distillation training | Cut from forward plan |
