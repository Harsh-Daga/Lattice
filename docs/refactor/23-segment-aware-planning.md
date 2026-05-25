# Phase 23 — Segment-Aware Planning

> **Footprint impact.** Zero new runtime deps. Default install unchanged. Segment classification reuses `core/segmentation.py` and `transforms/content_profiler/` — no new ML models.
>
> **Algorithm location.** `src/lattice/planner/segment_policy.py` (new), wired from `planner/unified_planner.py` and `transforms/content_profiler/planner_bridge.py`. Segment types and budgets are data-only (`SegmentPolicy`, `SegmentPlan`); transforms unchanged.
>
> **External-service requirement.** None.
>
>
> **Transport role.** Per-segment `transport_strategy` feeds `TransportPlan` (delta vs full replay, prefix stability) before Phase 20 dispatch — aligns structural prompts with wire economics.
>
> **Guidelines.** Complies with [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) and [FORWARD_PLAN.md](FORWARD_PLAN.md) constraints 1–6. Informed by [ARCHITECTURE_EVAL_INSIGHTS.md](ARCHITECTURE_EVAL_INSIGHTS.md).
>
> **Goal.** Stop optimizing entire prompts with a single global tier (`SIMPLE` / `MEDIUM` / `COMPLEX`). A real agent prompt mixes reasoning, JSON, tool output, logs, instructions, and history — each region needs different transforms, distortion budgets, and transport hints.
>
> **Outcome.** `UnifiedPlanner` emits a `SegmentPlan`: ordered segments with `(segment_type, allowed_transforms, distortion_budget, transport_hint)`. Eval metric `features not reached by pipeline` drops for cases where the **wrong global tier** blocked `format_conversion`, `message_dedup`, or `reference_sub`. Beam search in `representation_optimizer` respects per-segment allowlists when expanding candidates.
>
> **Estimated effort.** 8 days (1 PR, after Phase 27 lands streaming/tool-diff; may ship in same M3 window as 19).

---

## 1. Why this phase exists

Production evals (`v1.0.0.json`) show a recurring failure mode:

```
features not reached by pipeline: format_conversion
tier mismatch: expected SIMPLE but observed SIMPLE
```

The transform **exists** and **works** on structural scenarios (98% reduction on tables). The planner **did not activate** it for mixed workloads because:

1. **Global tier classification** — one label for the whole request.
2. **Risk gating** — `format_conversion` treated as lossy globally even when only the table region should run it.
3. **Optimizer allowlist** — beam search uses plan-level transform lists, not per-IR-section policies.

This is the highest-leverage orchestration fix identified in the architecture eval review. It does **not** require new transforms.

---

## 2. Segment types (v1)

| Type | Examples in prompt | Default transform policy | Distortion budget |
|---|---|---|---|
| `instruction` | System rules, “you must…” | `runtime_contract`, light `path_prefix` | minimal |
| `reasoning` | Chain-of-thought, analysis | `reference_sub` only if high confidence; **no** `rate_distortion` | strict quality floor |
| `structured_data` | JSON, tables, YAML | `format_conversion`, `json_shape`, `reference_sub` | moderate |
| `tool_io` | Tool definitions, tool results | `tool_filter`, `tool_projection`, `message_dedup` | moderate |
| `logs` | Stack traces, debug output | `diagnostic_rle`, `extractive_compress` (bounded) | high compression OK |
| `history` | Prior turns | `message_dedup`, `cache_arbitrage` | reuse-first |
| `filler` | Boilerplate, repeated headers | `message_dedup`, `path_prefix` | aggressive |

Mapping from `content_profiler` signals → segment type lives in **`segment_policy.py`** (pure functions, unit-tested).

---

## 3. Files touched

### 3.1 Created

```
src/lattice/planner/segment_policy.py      # SegmentType, SegmentPolicy, build_segment_plan()
src/lattice/planner/segment_plan.py        # SegmentPlan frozen dataclass (or merge into execution_plan.py)
tests/unit/planner/test_segment_policy.py
tests/unit/planner/test_segment_plan_integration.py
benchmarks/evals/segment_coverage.py       # optional: extend feature_eval proof with per-segment activation
```

### 3.2 Modified

| File | Change |
|---|---|
| `planner/unified_planner.py` | After task classification, call `build_segment_plan(ir, profile)`; attach to `ExecutionPlan.metadata` |
| `pipeline/runner.py` | When gating transforms, consult segment allowlist for active IR section(s) |
| `pipeline/representation_optimizer.py` | Filter beam expansions by segment policy |
| `transforms/content_profiler/planner_bridge.py` | Pass segment hints into planner (no duplicate classification) |
| `docs/architecture/runtime.md` | § “Segment-aware planning” cross-link |
| [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md) | Register `SegmentPlan`, `build_segment_plan` |

### 3.3 Deleted

None.

---

## 4. Step-by-step

### 4.1 Define `SegmentPlan`

```python
@dataclass(frozen=True, slots=True)
class SegmentPlan:
    segments: tuple[SegmentPolicy, ...]  # aligned to PromptIR sections / spans
    global_quality_floor: float
    global_latency_budget_ms: float
```

Each `SegmentPolicy` includes: `segment_type`, `allowed_transforms: frozenset[str]`, `max_distortion: float`, `transport_hint: Literal["delta","full","prefix_stable"]`.

### 4.2 Build plan from IR + profiler output

1. Run existing `content_profiler` (already on hot path).
2. Map each IR section to `SegmentType` using rules (table detection → `structured_data`, tool spans → `tool_io`, etc.).
3. Intersect with global risk tier (forbidden transforms still forbidden).
4. Attach `SegmentPlan` to `ExecutionPlan` (extend metadata; do not fork second `ExecutionPlan` type — Phase 16 collapsed duplicates).

### 4.3 Enforce in pipeline gates

In `pipeline/gates.py` (or runner gate loop):

- Before running transform `T` on section `S`, check `T in segment_plan.policy_for(S).allowed_transforms`.
- Record skip reason in context metrics (`segment_policy_skip`) for observability.

### 4.4 Extend eval proof

Add rows to `feature_eval` scenario proof:

- `segment_type_expected` vs `segment_type_observed`
- `transforms_blocked_by_segment_policy` (should be empty when policy correct)

Target: zero `features not reached by pipeline` for scenarios where the feature is structurally present in the prompt.

### 4.5 Verify

```bash
uv run pytest tests/unit/planner/test_segment_policy.py -q
uv run pytest tests/ -q
uv run python benchmarks/evals/cli.py --suite feature --iterations 1 --warmup 0
# Compare proof_failed / feature_failed vs v1.0.0.json baseline
```

---

## 5. Relationship to other phases

| Phase | Interaction |
|---|---|
| **12** | Uses single `ExecutionPlan` + validation facade; segment metadata is part of plan |
| **19** | Streaming/tool-diff runs on response path; segment policy is request-side |
| **22** | `provider_cache_probability` may vary per segment (stable prefix segments score higher) |
| **23** | Bandit learns weights per route; segment policies are priors, not replaced by bandit |
| **27** | `transport_hint` per segment informs `TransportPlan` |

---

## 6. Acceptance criteria

- [ ] `SegmentPlan` + `build_segment_plan()` exist; registered in [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md).
- [ ] `UnifiedPlanner` attaches segment plan to every non-trivial request.
- [ ] Pipeline skips transforms with explicit `segment_policy_skip` metric (not silent).
- [ ] `tests/unit/planner/test_segment_policy.py` covers all seven segment types.
- [ ] Feature eval: `features not reached by pipeline` count ≤ baseline for `mixed_realworld`, `table_compression`, `grammar_json_table` scenarios (operator compare vs `v1.0.0.json`).
- [ ] `uv run pytest tests/ -q` green; ruff/mypy clean.
- [ ] LoC within updated `planner/` cap.

---

## 7. Out of scope

| Item | Reason |
|---|---|
| New transforms | Code budget; evals show orchestration gap |
| Per-segment LLM judges | Violates lightweight + no external LLM |
| Re-delete legacy pipeline | Already removed Phase 3 |
| Full `runtime_v3` rewrite | Covered by incremental runtime.md + this phase |
| Multi-provider routing | FORWARD_PLAN constraint |

---

## 8. PR shape

```
feat(planner): segment-aware planning policies [Phase 28]

- ADD planner/segment_policy.py + SegmentPlan
- WIRE UnifiedPlanner + pipeline gates + representation_optimizer
- EXTEND feature_eval proof with segment activation rows
- DOC runtime.md + ARCHITECTURE_EVAL_INSIGHTS cross-links
```
