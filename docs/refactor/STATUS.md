# Refactor Status & Revised Forward Plan

> Last updated after Phase 9 on `refactor/phase-9-observability-state` (2026-05-24).
> See **[PHASE_COMPLETION_TRACKER.md](PHASE_COMPLETION_TRACKER.md)** for line-by-line acceptance vs each phase doc.
>
> The original 12-phase plan (`REFACTOR_PLAN.md` + `00-audit-baseline.md` …
> `11-docs-release.md`) is preserved as the historical reference. This
> document captures **what actually shipped vs. what the original plan
> assumed**, the **carryover items**, and the **revised sequencing** for
> the remaining work.
>
> **Doc index:** [docs/refactor/README.md](README.md) · **Forward plan:** [FORWARD_PLAN.md](FORWARD_PLAN.md) · **Eval → v2 mapping:** [ARCHITECTURE_EVAL_INSIGHTS.md](ARCHITECTURE_EVAL_INSIGHTS.md) · **Phase template:** [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) · **Registry:** [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md) · **LoC caps:** [CODE_BUDGET.txt](CODE_BUDGET.txt)
>
> The v2.0 forward plan is the source of truth for Phases **13–34** ([FORWARD_PLAN.md](FORWARD_PLAN.md)). **v1 Phase 12** is the docs release only. The product thesis: **LATTICE is the transport / network layer for LLM traffic.** [Phase 14](14-transport-layer-consolidation.md) (transport consolidation, immediately after Phase 13) makes that true on the wire; before it, adapters still carry duplicated retry/timeout code.
>
> The plan lives under six hard constraints, every one CI-enforced:
>
> 1. **Lightweight.** Default install on a 4 GB laptop. No required model downloads. Idle RSS < 100 MB; under load < 200 MB.
> 2. **No external LLM dependency** beyond the user's chosen provider.
> 3. **Open source self-hosted only.** No cloud product.
> 4. **One algorithm, one implementation — across SDKs AND internally.** Enforced by `scripts/check_sdk_no_algorithm_duplication.sh` and `scripts/check_internal_no_duplication.sh` (the latter walks [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md)).
> 5. **Codebase shall not grow unbounded.** Per-directory caps + 800-LoC/file + no-dup gates (`scripts/check_code_budget.sh`, `check_internal_no_duplication.sh`). Total cap 35 000 at v2.0 (`enforce_total_v2` flips Phase 31). No per-phase net LoC gate.
> 6. **Transport-first design.** After [Phase 14](14-transport-layer-consolidation.md): retry / timeout / circuit-breaker / backpressure / pooling / multiplexing exist exactly once in `src/lattice/transport/`. Enforced by `tests/contract/test_transport_unification.py`.
>
> What's been cut from the prior forward-plan draft (full rationale in [FORWARD_PLAN.md §1](FORWARD_PLAN.md)):
>
> - Hosted `lattice.cloud`, Stripe billing, Cloudflare/Neon/Upstash Terraform, hosted playground, per-tenant LoRA distillation training pipeline.
> - Local SentenceTransformers as **default** embedding backend (Phase 20): now opt-in `[embeddings-local]` extra (~1.5 GB). Default embedding tier uses the user's provider.
> - Presidio as **default** PII detector (Phase 21): now opt-in `[pii]` extra (~1 GB). Default detector is regex-based (zero deps).
> - ONNX injection classifier as default (Phase 21): now opt-in `[injection]` extra. Default is heuristic phrase matcher (zero deps).
> - LLMLingua-2 in the default compression story (Phase 27): now opt-in `[llmlingua]` extra (~600 MB) with an explicit first-run footprint warning.
> - Per-SDK reimplementation of reverse-pass / hooks / streaming (Phase 19, 17): replaced by the single-core architecture in Phase 31.
> - Per-adapter retry / timeout / `httpx.AsyncClient` instantiation (~17 copies of nearly-identical transport code): replaced by the single transport layer in [Phase 14](14-transport-layer-consolidation.md). Shrink target ~1500 LoC (dir caps + transport unification tests, not a net-LoC gate).
> - VSCode marketplace ceremony, Helm chart, signed-wheel sigstore ritual (Phase 34): trimmed to Docker + PyPI + npm + sideload `.vsix`.

---

## 1. Phases shipped

| Phase     | Status              | Merged in            | Notes                                                                                            |
| --------- | ------------------- | -------------------- | ------------------------------------------------------------------------------------------------ |
| **0**     | ✅ Done             | `bd32c0b` (PR #2)    | Inventory, contract tests, `FEATURE_PARITY.md` scaffold, `refactor-gate.yml`.                 |
| **1**     | ✅ Done             | `e5ff6d6`            | IR primitives collapsed into `lattice.ir`; `compile_request_ir` split from `build_ir`; xfail flipped on top-level `lattice.*` imports. |
| **2a**    | ✅ Done             | `ce06872` (PR #4)    | `core/transport.py` → `transport/types.py`; `serialization.py`, `delta_wire.py` move; 138 import sites rewritten. |
| **2b-1**  | ✅ Done             | `5d9bffb` (PR #5)    | 8 files moved into `pipeline/`; `PipelineV2` → `Pipeline`, `TransformRegistryV2` → `PipelineTransformRegistry`; stub `pipeline/__init__.py` due to circular-import with v1. |
| **2b-2a** | ✅ Done             | `67e98d5` (PR #6)    | `legacy_only` flag on `TransformSpec`; `constraint_lifting` + `strategy_selector` flagged; `Pipeline.process` skips them. |
| **2b-2b-A** | ✅ Done           | `5613e5a` (PR #7)    | 4 hardening tests migrated from `transform.process()` to `transform.optimize()`. |
| **3**       | ✅ Done           | `refactor/revised-plan` | V1 Kill: deleted `CompressorPipeline` + wrapper; `Pipeline.compress()` + gates; factory/client/proxy rewired; 10 IR-native `process()` deleted. |
| **4**       | ✅ Done           | `refactor/phase-4-planner-collapse` | Planner Collapse: `UnifiedPlanner` only; `planner/` package; `transforms/optimizers/`; `TierClassifier`; deleted RATS schedulers + text `StructureOptimizer`. |
| **5**       | ✅ Done             | PR [#10](https://github.com/Harsh-Daga/Lattice/pull/10) (`05dfd2f`) | Transforms cleanup merged to `main`. See `PHASE_COMPLETION_TRACKER.md`. |
| **6**       | ✅ Done             | `4798bfb` — PR [#11](https://github.com/Harsh-Daga/Lattice/pull/11) merged | Adapters under `providers/adapters/`; `providers/transport/` package; unified `_stream`. Benchmark `phase-6.json` operator-run. |
| **7**       | ✅ Done (merge PR #12) | `refactor/phase-7-proxy-sdk-cli` | HealthManager + middleware; top-level imports; sdk deprecation shim; doc/MIGRATION slice. |
| **8**       | ✅ Done             | `refactor/phase-8-integrations` | Tunnel → `integrations/tunnel.py`; `AgentNotInstalledError`; per-agent `doctor()`; transient lace in `mutation_store`. |
| **9**       | ✅ Done             | `refactor/phase-9-observability-state` | `telemetry/`, `state/`, `cache/`, `safety/`; leaf `core/` (6 files); `utils/` → `token_count` only. |
| **10**      | ✅ Done             | PR #16 | `lattice benchmark` wrapper; `src/lattice/evals/` deleted; `CLAIMS.md`; `v1.0.0` artifacts; `run_canonical_benchmark.sh`. |
| **11**      | ✅ Done             | Phase 11 branch | `tests/unit/` mirrors `src/lattice/`; `FEATURE_PARITY.md` (61 rows); contract matrices; `pytest-xdist`; pinned count. |
| **12**      | ✅ Done             | v1.0.0 docs release | README/AGENTS rewrite, CHANGELOG, MIGRATION, `runtime.md`, version `1.0.0`, doc dedup. |

**Current totals.** **2042** tests collected (pinned on honesty branch); **Phases 0–12** ✅ on `main`. **Phase 13** (honesty pass) ✅ on branch `refactor/forward-plan-phase-12-honesty` — merge to `main` pending. Code budget: **dir caps + 800-LoC/file + no-dup** (per-phase net LoC gate removed). **Next:** [Phase 14 — Transport consolidation](14-transport-layer-consolidation.md). See [MIGRATION-v1-to-v2.md](MIGRATION-v1-to-v2.md). **Benchmark gates** remain operator-run when `OLLAMA_CLOUD_API_KEY` is set.

---

## 2. What the original Phase 2 plan assumed vs. what actually shipped

The original Phase 2 (`docs/refactor/02-pipeline-runner.md`) was **one PR estimated at 2 days**. In reality:

| Original Phase 2 goal                                                                  | Shipped? |
| -------------------------------------------------------------------------------------- | -------- |
| Move `core/transport.py` → `transport/types.py` + `serialization.py` + `delta_wire.py` | ✅       |
| Create `src/lattice/pipeline/` and move 8 files into it                                | ✅       |
| `PipelineV2 → Pipeline`, `TransformRegistryV2 → PipelineTransformRegistry`             | ✅       |
| Add `legacy_only` flag for `constraint_lifting` + `strategy_selector`                  | ✅       |
| Delete `core/pipeline.py` (v1 `CompressorPipeline`, 1089 LoC)                          | ✅ Phase 3 |
| Delete `core/pipeline_v2_wrapper.py` (113 LoC)                                         | ✅ Phase 3 |
| Delete `process()` on the 10 IR-native transforms                                      | ✅ Phase 3 |
| Drop `"pipeline_v2"` registry entry                                                    | ✅ Phase 3 |
| Rewrite `pipeline/factory.py` (drop v1 builders)                                       | ✅ Phase 3 |
| Rewire `src/lattice/client.py` to use `Pipeline` directly                              | ✅ Phase 3 |
| Simplify `ReversibleSyncTransform` Protocol (drop `process()` requirement)             | ✅ Phase 3 |
| Broaden `pipeline/__init__.py` (was stub during v1 coexistence)                        | ✅ Phase 3 |
| Update `tests/contract/test_python_api_contract.py` (remove `CompressorPipeline`)      | ✅ Phase 3 |
| Run canonical bench vs phase-0 baseline                                                | ⏳ CI gate (needs `OLLAMA_CLOUD_API_KEY`) |

### Why the original Phase 2 underestimated scope

The plan doc treated the v2 path as **already complete and the v1 path as a thin wrapper to peel off.** Inspection revealed that v1's `CompressorPipeline.process` is **800+ LoC of production safety machinery** that v2's lean `Pipeline.process` does not yet replicate:

1. **Policy engine gates** — `policy.should_run(name, request, ctx)` per transform (Allow / Skip / Reject with rollback).
2. **Runtime-budget enforcement** — per-request latency budget, transform skipping when cumulative wallclock exceeds.
3. **Semantic-risk gating** — `SemanticRiskScore` from content_profiler feeds `transform_allowed_at_risk(name, risk)`; risk-blocked transforms recorded with reason.
4. **Protected-span pre-execution veto** — `DANGEROUS` bucket transforms vetoed when `_lattice_protected_spans` non-empty.
5. **Scheduler blocking** — `_lattice_schedule` blocked/allowed lists enforced; alias resolution; optimizer-list handling.
6. **MILV invocation** — Multi-Independent Lossy Validator runs after lossy transforms; failures roll back the transform.
7. **Transform reputation** — per-transform reputation tracking; bad transforms get demoted/disabled.
8. **Rollback on failure** — `backup` snapshot before each transform; restore on `Err` if `graceful_degradation`.
9. **Expansion guards** — transform output that bloats tokens beyond a threshold is rolled back.

None of (1)–(9) originally lived in `Pipeline.process`; they all lived in v1 `CompressorPipeline.process`. **Phase 3 ported these gates into `pipeline/gates.py` and `Pipeline.compress()`.**

Additionally, Phase 3 migrated `optimizer/*.py` constituents to `optimize()`. **Phase 4 dissolved the `optimizer/` package** — orchestrators now live in `transforms/optimizers/`.

---

## 3. Current file inventory snapshot (after Phase 4)

Canonical runtime chain:

```
Request → content_profiler → UnifiedPlanner → ExecutionPlan → Pipeline.compress/process → Provider
```

**`src/lattice/core/`** (6 files) — leaf primitives only:

```
__init__.py, config, context, errors, result, segmentation
```

**`src/lattice/telemetry/`** — metrics, downgrade taxonomy, cost, agent stats, maintenance, sketches.

**`src/lattice/state/`** — `session`, `store`, `segment_store`.

**`src/lattice/cache/`** — `semantic.py` (`SemanticCache`).

**`src/lattice/safety/`** — `risk_scoring.py` (was `utils/validation.py`).

**`src/lattice/utils/`** — `token_count.py` only (+ `__init__.py`).

**`src/lattice/transforms/`** (registry + reputation + patterns moved from `core/` / `utils/` in Phase 5):

```
registry.py, reputation.py, patterns.py,
content_profiler/{__init__,classifier,risk_scorer,task_classifier_bridge,planner_bridge}.py,
format_converter/{__init__,table_converter,json_converter}.py,
… (per-transform modules; deleted: prefix_opt, constraint_lifting, strategy_selector)
```

**`src/lattice/planner/`** (10 files) — single scheduling layer:

```
execution_plan, execution_builder, unified_planner, task_classifier, runtime_state,
request_classifier, provider_strategy, transport_planner, fallback_executor, __init__
```

**`src/lattice/pipeline/`** (11 files):

```
runner, factory, policy, guardrails, gates, milv, auto_continuation, batch_accumulator,
representation_optimizer, base, __init__
```

**`src/lattice/transforms/optimizers/`** (7 files) — was `optimizer/` (deleted):

```
__init__, _dispatch, ir_structure_optimizer, reference_optimizer, tool_optimizer,
diagnostic_optimizer, context_optimizer
```

**`src/lattice/runtime/`** — `tier_classifier.py` (renamed from `router.py`; not a provider router)

**`src/lattice/providers/`** — `adapters/` (17 providers), `transport/` (`registry`, `pool`, `rate_limits`, `helpers`, `completion`, `streaming`, `stall_detector`), `credentials.py`

**Deleted in Phases 3–4:** `core/pipeline.py`, `core/pipeline_v2_wrapper.py`, `core/scheduler.py`, `core/optimizer_scheduler.py`, `core/unified_planner.py`, `core/task_classifier.py`, `core/runtime_state.py`, `core/credentials.py`, `optimizer/` (entire package), text `structure_optimizer.py`, `runtime/router.py`.

**Public planner imports:** `from lattice.planner import UnifiedPlanner, ExecutionPlan, build_execution_plan, classify_task`  
**Public runtime imports:** `from lattice.runtime import TierClassifier, Tier, TierDecision`  
**Public optimizer imports:** `from lattice.transforms.optimizers import IRStructureOptimizer, ReferenceOptimizer, ...`

---

## 4. Revised phase plan

The original `REFACTOR_PLAN.md` listed phases 0–11. We're collapsing Phase 2 (now done as scoped) and **inserting a dedicated v1-kill phase** before continuing. Subsequent phases are renumbered.

| New phase | Old phase | Name                                       | Status     | Effort (revised) |
| --------- | --------- | ------------------------------------------ | ---------- | ---------------- |
| 0         | 0         | Audit & Baseline                           | ✅ Done     | —                |
| 1         | 1         | IR Primitives                              | ✅ Done     | —                |
| 2         | 2         | Pipeline Package Structure                 | ✅ Done     | —                |
| **3 (NEW)** | (split from 2) | **V1 Kill** — port safety machinery, rewire client/factory, delete `CompressorPipeline` + wrapper | ✅ Done     | —                |
| 4         | 3         | Planner Collapse                           | ✅ Done     | —                |
| 5         | 4         | Transforms cleanup (`process()` deletion, file splits) | ✅ Done     | —                |
| 6         | 5         | Providers + Transport split                | ✅ Done     | PR #11           |
| 7         | 6         | Proxy + SDK + CLI                          | ✅ Done     | branch `refactor/phase-7-proxy-sdk-cli` |
| 8         | 7         | Integrations (tunnel, doctor, mutation store) | ✅ Done     | PR #13           |
| 9         | 8         | Observability + State                      | ✅ Done     | PR phase-9     |
| 10        | 9         | Benchmarks                                 | ⏳ Next     | 1 day            |
| 11        | 10        | Tests reorg                                | ⏳ Pending  | 1 day            |
| 12        | 11        | Docs + Release                             | ⏳ Pending  | 1 day            |

**Total remaining effort estimate: 15–24 days of focused work.** The original plan estimated ~20 days; we're roughly on track despite the Phase 2 underestimate.

---

## 5. Phase 3 (new) — V1 Kill: detailed plan

> **Status: ✅ SHIPPED** on `refactor/revised-plan`. Historical spec preserved below.

> **Goal.** Delete `core/pipeline.py` (1089 LoC) + `core/pipeline_v2_wrapper.py` (113 LoC). Rewrite `pipeline/factory.py` to return a `Pipeline` directly. Rewire `src/lattice/client.py` to call `Pipeline.compress(req, ctx) → Result[Request]` (new convenience method). Port v1's safety machinery into `Pipeline`. Delete legacy `process()` on the 10 IR-native transforms. Migrate `optimizer/*.py` constituents to `optimize()`.
>
> **Estimated effort.** 3–5 days.
>
> **Acceptance.** `rg "CompressorPipeline|PipelineV2Wrapper|pipeline_v2_wrapper" src/ tests/ benchmarks/` returns 0 matches. `from lattice.pipeline import Pipeline, build_default_pipeline, build_benchmark_pipeline` works. All 1878 unit tests + 24 contract tests pass. Canonical bench ±2% of phase-0 baseline.

### 5.1 Step 1 — Add `Pipeline.compress(req, ctx)` convenience method

In `src/lattice/pipeline/runner.py`, add a high-level entry that:

1. Runs `content_profiler` first (sets IR, task classification, risk score, prefix manifest, etc. on context).
2. Reads `_lattice_execution_plan` from context (set by content_profiler), OR builds one via `UnifiedPlanner.plan(req, profile_from_legacy(...))` if absent.
3. Runs the existing safety machinery (ported from v1 — see §5.2).
4. Calls `self.process(req, plan, ctx)`.
5. Returns `Result[Request, TransformError]`.

Signature:

```python
def compress(
    self,
    request: Request,
    context: TransformContext,
    *,
    config: LatticeConfig | None = None,
) -> Result[Request, TransformError]:
    ...
```

The existing `process(req, plan, ctx)` stays — `compress` is the high-level entry, `process` the low-level executor. Single class, two entry points; not a parallel path.

### 5.2 Step 2 — Port v1's safety machinery into `Pipeline.compress`

Verbatim-port these v1 gates (current locations in `core/pipeline.py`):

| Gate                              | v1 LoC range | Behavior to preserve                                                            |
| --------------------------------- | ------------ | ------------------------------------------------------------------------------- |
| Policy `Allow / Skip / Reject`    | 261–289      | Per-transform policy decision; Reject + graceful_degradation = restore backup   |
| Runtime-budget skip               | 291–327      | Skip budget-sensitive transforms when cumulative wallclock exceeds budget       |
| Semantic-risk gate                | 329–378      | `transform_allowed_at_risk(name, risk)`; SAFE-only fallback if no risk data     |
| Protected-span DANGEROUS veto     | 380–400      | Veto DANGEROUS transforms when `_lattice_protected_spans` is non-empty          |
| Scheduler blocking                | 402–460      | Enforce `_lattice_schedule` blocked/allowed sets, with alias resolution         |
| MILV post-transform validation    | ~580–700     | Run MILV after lossy transforms; rollback on failure                            |
| Transform reputation tracking     | ~770–850     | Update reputation per transform success/failure                                 |
| Expansion guard / rollback        | ~700–770     | Roll back transforms whose output exceeds expansion threshold                   |

Each gate is a **standalone check** that consumes context state and the current request. The clean factoring is to lift them into named helper functions inside `pipeline/runner.py` (or a new `pipeline/gates.py`):

```python
def _policy_decision(policy, name, request, context) -> Allow | Skip | Reject: ...
def _runtime_budget_exhausted(request, context, cumulative_ms) -> bool: ...
def _risk_gate_blocks(name, request, context) -> tuple[bool, str]: ...
def _protected_span_veto(name, request, context) -> bool: ...
def _scheduler_blocks(name, request, context) -> bool: ...
def _milv_validate(name, before, after, context) -> bool: ...
def _expansion_guard_rejects(before, after) -> bool: ...
```

`Pipeline.compress` calls them in order. Failures → record metric + skip transform + restore backup (if graceful).

### 5.3 Step 3 — Rewrite `pipeline/factory.py`

Replace the four v1 builders with one:

```python
def build_default_pipeline(config: LatticeConfig) -> Pipeline:
    registry = PipelineTransformRegistry()
    return Pipeline(registry=registry, config=config)


def build_benchmark_pipeline(config: LatticeConfig) -> Pipeline:
    cfg = config.model_copy() if hasattr(config, "model_copy") else config
    cfg.use_optimizer_pipeline = True
    return build_default_pipeline(cfg)


def pipeline_summary(pipeline: Pipeline) -> dict[str, Any]:
    names = pipeline.registry.get_transform_names()
    optimizers = [n for n in names if n.endswith("_optimizer")]
    core = [n for n in names if n not in optimizers]
    return {
        "count": len(names),
        "transforms": names,
        "core_transforms": core,
        "optimizers": optimizers,
        "runtime_contract_enabled": "runtime_contract" in names,
    }
```

`Pipeline.__init__` takes an optional `config` for the safety gates.

Delete `build_v2_pipeline`, `build_optimizer_pipeline`, and all references to `CompressorPipeline`.

### 5.4 Step 4 — Rewire `src/lattice/client.py`

Replace:

```python
self._pipeline = _build_pipeline(self.config)  # returns CompressorPipeline
result = await self._pipeline.process(request, ctx)
```

with:

```python
self._pipeline = build_default_pipeline(self.config)  # returns Pipeline
result = self._pipeline.compress(request, ctx)  # sync
```

Update `compress()` to drop `asyncio.run` (Pipeline.compress is sync). Update `decompress_response()` to call `pipeline.reverse(response, plan, ctx)` (needs the plan from `self._last_compress_ctx`).

Update `health()` to use `pipeline.registry.get_transform_names()` instead of `self._pipeline.transforms`.

### 5.5 Step 5 — Migrate `optimizer/*.py` constituents to `optimize()`

`structure_optimizer.py`, `context_optimizer.py`, `tool_optimizer.py`, `reference_optimizer.py` each have:

```python
self._constituents.append(("format_conversion", FormatConverter()))
```

…and somewhere later they call `constituent.process(req, ctx)`. Migrate to:

```python
result = constituent.optimize(ir, req, ctx)
```

This requires the optimizer code to have an `ir` in scope — usually it has, since it's invoked from `Pipeline.process`. If not, build one via `prompt_ir_v2_from_legacy(build_ir(req))` at call time.

### 5.6 Step 6 — Delete legacy `process()` methods

On these 10 IR-native transforms:

```
cache_arbitrage, causal_chain, format_conv, message_dedup, path_prefix,
rate_distortion, reference_sub, runtime_contract, tool_filter, tool_projection
```

After §5.5, no caller invokes `process()` on them.

### 5.7 Step 7 — Simplify `ReversibleSyncTransform` Protocol

Move it from `core/pipeline.py` into `pipeline/runner.py`. Drop the `process()` requirement:

```python
@runtime_checkable
class ReversibleSyncTransform(Protocol):
    name: str
    priority: int

    def can_process(self, request: Request, context: TransformContext) -> bool: ...
    def optimize(
        self,
        ir: PromptIRV2,
        request: Request,
        context: TransformContext,
    ) -> Result[PromptIRV2, TransformError]: ...
    def reverse(self, response: Response, context: TransformContext) -> Response: ...
```

Re-export from `core/__init__.py` for back-compat (the 4 transforms still needing `process()` were either deleted via §5.6 or are content_profiler-style scaffolding that stay in `core/pipeline.py` until Phase 4).

### 5.8 Step 8 — Delete v1 files

```bash
git rm src/lattice/core/pipeline.py
git rm src/lattice/core/pipeline_v2_wrapper.py
```

Update `transform_registry.py`: delete the `"pipeline_v2"` `TransformSpec` (factory_path pointed at the now-deleted wrapper).

Update `core/__init__.py`: drop `CompressorPipeline` re-export. `ReversibleSyncTransform` re-export switches to `lattice.pipeline.runner`.

### 5.9 Step 9 — Broaden `pipeline/__init__.py`

The circular-import constraint is gone (no more `core/pipeline.py` importing from `pipeline.policy`). Add the full public surface:

```python
from lattice.pipeline.runner import Pipeline, PipelineTransformRegistry, ReversibleSyncTransform
from lattice.pipeline.factory import build_default_pipeline, build_benchmark_pipeline, pipeline_summary
from lattice.pipeline.policy import OptimizationPolicy, Allow, Skip, Reject
from lattice.pipeline.guardrails import (
    GuardAction, SafetyDecision, ValidationOutcome,
    check_expansion_guard, check_entity_preservation, check_format_preservation,
    check_critical_signal_loss, check_placeholder_leakage, check_negative_savings,
    check_blank_output,
)
from lattice.pipeline.milv import MILVResult, should_trigger_milv, validate_transform
from lattice.pipeline.auto_continuation import AutoContinuation, ContinuationResult
from lattice.pipeline.batch_accumulator import BatchAccumulator, BatchResult, AccumulatedRequest
from lattice.pipeline.representation_optimizer import RepresentationOptimizer
```

### 5.10 Step 10 — Contract test update

`tests/contract/test_python_api_contract.py`: remove `CompressorPipeline` from the public-surface list (it was never user-facing). Add `Pipeline` to the "internal but stable" set if applicable.

### 5.11 Step 11 — CI gates + canonical bench

- `uv run ruff check src/ tests/ benchmarks/` clean
- `uv run ruff format --check src/ tests/ benchmarks/` clean
- `uv run mypy src/lattice/ --ignore-missing-imports` clean
- `uv run pytest tests/ -q` ≥ 1878 passed
- `uv run pytest tests/contract/ -q` 24 passed
- `bash scripts/run_canonical_benchmark.sh` against ollama-cloud/kimi-k2.6:cloud
- `python scripts/compare_benchmarks.py benchmarks/results/phase-0-baseline.json benchmarks/results/phase-3-v1-killed.json --tolerance-pct 2` exits 0 (LLM-judge stochasticity exception per orchestrator §6 if avg_pipeline_latency / avg_quality_score / total_passed move within noise band)

### 5.12 PR shape

Single PR titled `refactor(pipeline): kill v1 CompressorPipeline; one Pipeline runner [Phase 3]`. Diff stat will be large (1200+ deletions, ~500 additions) but mostly mechanical after the safety-machinery port is in place.

---

## 6. Phases 4–12 — high-level summary (revised numbering)

Each maps onto its original `docs/refactor/0N-*.md` doc (e.g. new Phase 4 = original Phase 3 = `03-planner-collapse.md`). The original docs remain authoritative for those phases; only Phase 3 (v1 kill) needs the new doc above. Effort estimates are revised after the Phase 2 lessons.

- **Phase 4 (Planner Collapse)** — ✅ Shipped on `refactor/phase-4-planner-collapse`. Deleted RATS schedulers; `UnifiedPlanner` only; `planner/` + `transforms/optimizers/`; `TierClassifier`; credentials → `providers/`.
- **Phase 5 (Transforms cleanup)** — ✅ Shipped (5a–5c on `refactor/phase-5-transforms-cleanup`). See `docs/refactor/phase-5-decisions.md` for benchmark-gated deletions (canonical bench skipped without API key; default DELETE applied).
- **Phase 6 (Providers + Transport)** — ✅ Shipped on PR #11. Monolith `providers/transport.py` → `providers/transport/` package + `providers/adapters/`.
- **Phase 7 (Proxy + SDK + CLI)** — ✅ Shipped on `refactor/phase-7-proxy-sdk-cli`. Health routes, `proxy/middleware.py`, top-level SDK exports, `sdk/client.py` deprecation shim.
- **Phase 8 (Integrations)** — ✅ Shipped on PR #13. Tunnel → `integrations/tunnel.py`; `AgentNotInstalledError`; per-agent `doctor()`; transient lace in `mutation_store`. (MCP module unchanged — out of scope for 07-integrations.md.)
- **Phase 9 (Observability + State)** — ✅ Shipped on `refactor/phase-9-observability-state`. `telemetry/`, `state/`, `cache/`, `safety/`; leaf `core/` + `utils/token_count`. See `08-observability-state.md`.
- **Phase 10 (Benchmarks)** — ✅ Shipped PR [#16](https://github.com/Harsh-Daga/Lattice/pull/16); completion fixes on `refactor/phase-10-completion` (provider_validation `compress()`, `--provider-detect`, doc sync).
- **Phase 11 (Tests)** — ✅ Shipped. `tests/unit/` mirrors `src/lattice/`; `FEATURE_PARITY.md` (61 rows); contract matrices; `pytest-xdist`; `test_test_count_pinned.py`.
- **Phase 16 (Docs + release)** — **Next.** `11-docs-release.md`: README rewrite, CHANGELOG, full MIGRATION import map, version 1.0.0, tag, PyPI dry-run.

---

## 7. Carryover tracker

### Phase 2 → 3 (v1 kill) — all shipped

| Item                                                                     | Status              |
| ------------------------------------------------------------------------ | ------------------- |
| Delete `core/pipeline.py` (v1 `CompressorPipeline`)                      | ✅ Phase 3          |
| Delete `core/pipeline_v2_wrapper.py`                                     | ✅ Phase 3          |
| Port v1 safety machinery (policy/guardrails/MILV/reputation/rollback)    | ✅ Phase 3          |
| Rewrite `pipeline/factory.py`                                            | ✅ Phase 3          |
| Rewire `src/lattice/client.py`                                           | ✅ Phase 3          |
| Drop `"pipeline_v2"` registry entry                                      | ✅ Phase 3          |
| Delete `process()` on 10 IR-native transforms                            | ✅ Phase 3          |
| Migrate `optimizer/*.py` constituents to `optimize()`                    | ✅ Phase 3          |
| Simplify `ReversibleSyncTransform` Protocol                              | ✅ Phase 3          |
| Broaden `pipeline/__init__.py`                                           | ✅ Phase 3          |
| Update `tests/contract/test_python_api_contract.py` (drop CompressorPipeline) | ✅ Phase 3     |
| Normalize `pipeline_v2` metric namespace → `pipeline`                    | ✅ Phase 3          |

### Phase 4 (planner collapse) — all shipped except benchmark gate

| Item                                                                     | Status              |
| ------------------------------------------------------------------------ | ------------------- |
| Delete `core/scheduler.py`, `core/optimizer_scheduler.py`                | ✅ Phase 4          |
| Move planner modules → `planner/`                                        | ✅ Phase 4          |
| Move optimizers → `transforms/optimizers/`; delete `optimizer/`          | ✅ Phase 4          |
| Rename `runtime/router.py` → `tier_classifier.py`                        | ✅ Phase 4          |
| Move `credentials.py` → `providers/`                                     | ✅ Phase 4          |
| Delete text `StructureOptimizer`                                         | ✅ Phase 4          |

### Phase 5 (transforms cleanup) — shipped

| Item                                                                     | Status              |
| ------------------------------------------------------------------------ | ------------------- |
| Move `transform_registry` / `transform_reputation` → `transforms/`       | ✅ Phase 5a         |
| Move `utils/patterns.py` → `transforms/patterns.py`                      | ✅ Phase 5a         |
| `TransformSpec.is_response_side` + `transform_delta_encode` fix          | ✅ Phase 5a         |
| Split `content_profiler/` + `optimize()` on IR path                      | ✅ Phase 5b         |
| Split `format_converter/`; delete `prefix_opt`, `constraint_lifting`     | ✅ Phase 5b         |
| Delete `strategy_selector`, `information_theoretic_selector` (gated)   | ✅ Phase 5c         |
| `transform_prefix_opt` / `transform_constraint_lifting` / `transform_strategy_selector` config no-ops | ✅ carryover until Phase 16 MIGRATION.md |
| Canonical bench A/B/C + `phase-5.json` ±2%                               | ⏳ CI / local key   |

### Phase 6 (providers + transport) — shipped

| Item                                                                     | Status              |
| ------------------------------------------------------------------------ | ------------------- |
| `providers/adapters/` + `providers/transport/` package                     | ✅ PR #11           |
| Unified `_stream()`; TTL `RateLimitTracker`                              | ✅                  |
| All 17 adapters at `lattice.providers`                                   | ✅ contract test    |
| Tests under `tests/unit/providers/transport/`                            | ✅                  |
| Docs (`STATUS`, `PHASE_COMPLETION_TRACKER`, `providers.md`, `AGENTS.md`)   | ✅                  |
| Canonical bench → `phase-6.json` ±2%                                     | ⏳ see `phase-6-benchmark.md` |

### Still pending (later phases)

| Item                                                                     | Target phase        |
| ------------------------------------------------------------------------ | ------------------- |
| Remove `--use-v2-pipeline` CLI flag                                      | ✅ Phase 10         |
| Canonical bench vs phase-0 baseline (±2%) → `phase-5.json`               | CI / local key      |
| Canonical bench vs phase-0 baseline (±2%) → `phase-6.json`               | CI / local key      |

---

## 8. Ground rules (R1–R12 — unchanged from orchestrator brief)

1. Plan is authoritative; stop and report errors, don't improvise.
2. Phases run in strict order.
3. Five CI gates after each phase (ruff check, ruff format --check, mypy, pytest, canonical bench + compare).
4. Public surface (CLI/HTTP/headers/Python symbols) is sacred — if contract test fails, revert your change, don't relax the test.
5. No bridge layers, no `_compat` files, no parallel paths.
6. No file >800 LoC in v1.0.0.
7. Imports point downhill per dependency direction; `core/` is a leaf; `pipeline/` may not import from `cli/proxy/gateway/sdk`.
8. Honest names — no `compiler.py` that doesn't compile, no `router.py` that doesn't route, no `v2` suffix once Phase 3 lands.
9. Don't invent features or expand scope; file smells as TODOs.
10. Each phase = one PR (or a small slice series if the phase splits cleanly).
11. Commit messages reference phase with `[Phase N]` suffix; no `--no-verify`, no `--amend` of pushed commits, no force-push to main.
12. **Secrets and tokens: NEVER commit. NEVER log. NEVER write into any file in the repo.** Benchmark API key (`OLLAMA_CLOUD_API_KEY`) is a runtime environment variable only.

---

## 9. v2.0 forward plan — Phases 16–26

The forward plan lives in **[FORWARD_PLAN.md](FORWARD_PLAN.md)**. Quick reference:

| Phase | Doc | Milestone | Footprint impact |
|---|---|---|---|
| 12 | [Honesty Pass](13-honesty-pass.md) | M2 v1.1 | -2300 LoC (net shrink) |
| 13 | [Python SDK (thin client)](16-python-sdk-quality.md) | M2 | +1 MB; zero new runtime deps |
| 14 | [Hybrid Semantic Cache (lightweight)](17-hybrid-semantic-cache.md) | M2 | 0 new deps default; embedding tier uses user's provider |
| 15 | [Native Guardrails (lightweight)](18-native-guardrails.md) | M2 | 0 new deps default; rule-based PII + heuristic injection + pure-Python repair |
| 16 | [OpenTelemetry GenAI](19-otel-genai.md) | M2 | + opentelemetry-sdk (~3 MB) only when enabled |
| 17 | [TypeScript SDK (thin client)](20-typescript-sdk.md) | M3 v1.5 | npm ≤ 25 KB gz (edge); zero algorithm code; consumes Phase 31 WASM core |
| 18 | [MCP-Native Gateway](21-mcp-native-gateway.md) | M3 | Base install |
| 19 | [Compression Intelligence](22-compression-intelligence.md) | M3 | 0 new deps default (streaming + tool-diff + JSON repair); LLMLingua-2 opt-in `[llmlingua]` (~600 MB) |
| 23 | [Segment-Aware Planning](23-segment-aware-planning.md) | M3 | 0 new deps; fixes eval `features not reached by pipeline` |
| 20 | [Non-Chat Surfaces](24-non-chat-surfaces.md) | M3 | Base install |
| 21 | [Agent Memory (lightweight)](26-agent-memory.md) | M4 v2.0 | 0 new deps default; rule-based relevance scoring; user's cheap model for summarization |
| 22 | [Cache Portability](27-cache-portability.md) | M4 | Base install |
| 23 | [Receipts + Bandit + Profiles + Hot Reload](28-receipts.md) | M4 | + pyjwt (~100 KB); bandit is pure numpy (no ML deps) |
| 24 | [Shared Core: Rust + PyO3 + WASM](31-edge-wasm-core.md) | M4 | Optional native wheel ~3 MB; WASM ≤ 200 KB gz; **architectural keystone — prevents SDK duplication** |
| 25 | [Optional Self-Hosted Auth, Keys, Quotas](32-cloud-multitenant.md) | M4 | All optional; SQLite default; **no cloud, no SaaS, no Stripe** |
| 26 | [Agent-Loop-Aware + Cursor Visualizer + Minimal Release](34-agent-loop-aware.md) | M4 | Cursor extension ~200 KB; Docker + PyPI + npm + sideload `.vsix` — **no marketplace ceremony, no Helm** |
| 27 | [Transport Layer Consolidation](14-transport-layer-consolidation.md) | M4 | **Net -1500 LoC.** Unified retry/timeout/CB, HTTP/2 multiplexing, connection pool per provider, transport metrics, backpressure, stream resumption. **The phase that makes "LATTICE is a transport layer for LLMs" true.** |

### Suggested execution order (matches dependencies)

See [PHASE_GUIDELINES.md §7](PHASE_GUIDELINES.md) and [docs/refactor/README.md](README.md).

```
M2: 12 → 14 → 15 → 13 → 16
M3: 27 → 24 → 17 → {18, 19, 20} → 23   ← Phase 20 FIRST in M3; 23 after compression intel
M4: 21 → 22 → 23 → 25 → 26
```

### Hard rules introduced by the forward plan (CI-enforced)

| Rule | Enforcement |
|---|---|
| No multi-provider routing | `tests/contract/test_no_multi_provider_routing.py` |
| No algorithm code in SDKs | `scripts/check_sdk_no_algorithm_duplication.sh` |
| **No internal duplication of canonical primitives** | `scripts/check_internal_no_duplication.sh` walks [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md) |
| **Code budget (total + per-directory + per-PR delta)** | `scripts/check_code_budget.sh` (Phase 16 lands it) |
| **Transport policies live in one place** | `tests/contract/test_transport_unification.py` + `test_no_per_adapter_httpx_client.py` (Phase 20) |
| No required model download in default install | `tests/contract/test_default_install_no_models.py` |
| No required external service in default install | `tests/contract/test_default_install_no_external_services.py` |
| No file > 800 LoC | Existing R6 |
| No raw user content in receipts / headers / spans | `tests/contract/test_no_user_content_in_telemetry.py` |
| Transform reversibility property | `tests/unit/transforms/test_reversibility_property.py` |
| Footprint budget (4 GB laptop, 2 GB VPS) | `tests/integration/footprint/test_{4gb_laptop,2gb_vps,cold_start}.py` |
| Cross-mode parity (proxy vs in-process; Python vs Rust vs WASM) | `tests/contract/test_python_parity.spec.ts`, `bindings/python/tests/test_parity_with_python.py` |
