# Architecture Eval Insights — Production Evals → v2 Plan

> **Purpose.** Capture the external architecture critique (May 2026) against **what is actually on `main` after Phases 0–10**, and record what we **accept**, **reject**, and **map** into Phases 16–27. This doc prevents a parallel “refound the repo” rewrite that would undo the v1.0 refactor.
>
> **Canonical runtime model:** [`../architecture/runtime.md`](../architecture/runtime.md)  
> **Implementation index:** [FORWARD_PLAN.md](FORWARD_PLAN.md)

---

## 1. Executive summary

The critique is **directionally right** about product positioning and optimization objectives:

- LATTICE’s strongest primitives are **structural factoring + transport economics**, not semantic summarization.
- Eval inconsistency (98% compression with poor quality on tables vs 0% compression with excellent reasoning) reflects a **multi-objective problem**, not three competing codebases.
- The highest-leverage v2 work is **orchestration and scoring** (segment-aware planning, utility-driven candidate search), not more transforms.

It is **factually wrong** about several “mandatory deletes” that **already shipped** in Phases 3–5 (`CompressorPipeline`, `strategy_selector`, legacy schedulers). Treat those items as **rejected** unless re-verified with `rg` on `main`.

---

## 2. What production evals actually prove (v1.0.0)

Pinned run: `benchmarks/results/v1.0.0.json` + [CLAIMS.md](../benchmarks/results/CLAIMS.md).

| Observation | Evidence | Implication |
|---|---|---|
| **Structural rewrites win** | High reduction on `table_compression`, `json_response_format`, `grammar_json_table` (~97–98%) | Invest in `reference_sub`, `format_conversion`, `ir_structure_optimizer`, `path_prefix` — not more summarizers |
| **Semantic summarization is weak** | `api_docs_summarization`, `rate_distortion_longform` quality ~0.64–0.67 | Do not market “summarize anything”; gate lossy transforms per segment |
| **Reasoning preservation is strong** | `reasoning_root_cause`, `debugging_log_analysis` quality ~0.92–0.98 | Conservative / runtime-aware paths work — protect them |
| **Orchestration gaps** | `features not reached by pipeline` (e.g. `format_conversion` on some tiers) | Fix **planner activation**, not add transforms — see [23-segment-aware-planning.md](23-segment-aware-planning.md) |
| **Provider validation** | Task-equivalence judge on live provider | Meaning preservation is the real gate; compression % alone is misleading KPI |

**Success metrics for v2 (agreed):** lower provider cost, lower latency, higher cache reuse, stable reasoning, smaller wire payloads, replay determinism — **not** headline compression ratio in isolation.

---

## 3. Three “systems” — reality check on `main`

| Paradigm in critique | Status on `main` (post Phase 10) | v2 action |
|---|---|---|
| **Legacy transform engine** (`transform.process` scheduler, `CompressorPipeline`) | **Deleted** Phases 3–5. Canonical path: `Pipeline.compress()` → `ExecutionPlan` → IR-native `optimize()` | **Reject** re-delete. Keep `tests/unit/pipeline/test_no_legacy_process_paths.py` green |
| **Representation optimizer** (beam search, `CandidateSearch`) | **Shipped.** `pipeline/representation_optimizer.py`, immutable `Candidate` in `ir/primitives.py` | **Extend** scoring utility — [13-honesty-pass.md](13-honesty-pass.md), [27-cache-portability.md](27-cache-portability.md), [14-transport-layer-consolidation.md](14-transport-layer-consolidation.md) |
| **Semantic transport runtime** (delta, manifest, cache planner, prefix) | **Partially shipped.** `transport/`, `protocol/`, `cache/`, Phase 17/27 docs | **Complete** via Phase 20 + 22; expose gains in planner utility |

There is **one** runtime with three **layers** (IR policy, search, transport), not three fighting products.

---

## 4. Accept → phase mapping

| Insight | Phase | Doc |
|---|---|---|
| Constitutional lifecycles + honest optimization objective | Runtime doc + Phase 16 | [`runtime.md`](../architecture/runtime.md), [13-honesty-pass.md](13-honesty-pass.md) |
| Centralize validation entrypoints | Phase 16 | `runtime/validation_engine.py` (facade over MILV + guardrails + `ir/validation`) |
| Segment-aware planning (per-region transform policy) | **Phase 28** (new) | [23-segment-aware-planning.md](23-segment-aware-planning.md) |
| Utility-driven `Candidate.score` (not compression-only) | Phase 16 + 19 + 22 + 27 | `ir/quality.py`, planner inputs |
| `provider_cache_probability` in search | Phase 17 | `cache/analyzer.py` → planner |
| `transport_gain` in search | Phase 20 | delta_wire + stable prefix metrics → planner |
| Replay determinism contract | Phase 16 | `tests/contract/test_replay_determinism.py` |
| Transport-first consolidation | Phase 20 | unchanged — still M3 anchor |
| KV-cache / portability | Phase 17 | unchanged |

---

## 5. Reject → do not schedule

| Proposal | Why reject |
|---|---|
| Delete `pipeline.py`, `transform_registry.py`, legacy scheduler again | Already gone or canonical (`transforms/registry.py` is SSOT for specs) |
| Delete immutable `Candidate` / “mutable candidate chaos” | `Candidate` is frozen; `apply()` forks — see `ir/primitives.py` |
| Second repo refound (“Phase 1–5 over 2–3 weeks”) | Phases 0–10 **just** refounded layout; duplicate refound = architectural debt |
| Stop reporting compression ratio | Keep as **one term** in utility; remove as sole headline in Phase 16 README/CLAIMS alignment |
| Add random transforms | Code budget + SSOT forbid; bandit (Phase 28) adapts **existing** transforms |
| LLM-judge in default hot path | Benchmark/operator only; violates lightweight constraint |

---

## 6. Optimization objective (v2 target)

Replace single-axis “maximize compression_ratio” with a documented utility function used by beam search and bandit (Phase 28):

```
utility(c) =
    + w_q * semantic_quality(c)
    + w_cache * provider_cache_probability(c)   # Phase 17 analyzer
    + w_tx * transport_gain(c)                 # Phase 20 (delta, prefix stability)
    - w_tok * normalized_token_cost(c)
    - w_lat * latency_p95_estimate(c)
    - w_inst * instability_penalty(c)           # replay / placeholder leakage
```

Weights are **profile-configurable** ([Phase 28](28-receipts.md)), not hard-coded magic numbers. Default profile favors **reasoning preservation** on agent workloads.

---

## 7. Strong components (investment priority)

Align engineering and docs with what evals reward:

| Primitive | Category | Canonical home |
|---|---|---|
| `reference_sub` | Symbolic factoring | `transforms/reference_sub.py` |
| `ir_structure_optimizer` | Structural rewrite | `transforms/optimizers/` |
| `message_dedup` | State reuse | `transforms/message_dedup.py` |
| `path_prefix` | Canonical transport | `transforms/path_prefix.py` |
| `cache_arbitrage` | Provider economics | `transforms/cache_arbitrage.py` |
| `runtime_contract` | Execution constraints | `transforms/runtime_contract.py` |
| `delta_wire` | Session transport | `transport/delta_wire.py` |
| `cache` tiers + portability | Provider KV | `cache/`, Phase 20/22 |

Summarization transforms (`rate_distortion`, `extractive_compress`) remain **opt-in per segment** with strict quality floors.

---

## 8. References

- [FORWARD_PLAN.md](FORWARD_PLAN.md) — milestones M2–M4  
- [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) — phase template  
- [benchmarks/results/CLAIMS.md](../benchmarks/results/CLAIMS.md) — claim traceability  
- [09-benchmarks.md](09-benchmarks.md) — Phase 10 benchmark harness (STATUS numbering)
