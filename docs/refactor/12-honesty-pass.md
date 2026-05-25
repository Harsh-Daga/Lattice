# Phase 12 — Honesty Pass + Internal-Dedup Audit + Code-Budget Gate

> **Footprint impact.** Net **-1500 LoC** (was -1100; expanded scope catches more duplication). Zero new runtime deps. The split files de-load imports too, so cold-start time drops ~50 ms.
>
> **Algorithm location.** Pure cleanup phase — no new algorithms introduced. Collapses every primitive listed in [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md) into the declared canonical home and lands the CI gates that keep them there.
>
> **External-service requirement.** None.
>

> **LoC delta (declared).** -1500 net. Lands CI gates; shrinks codebase.
> **Transport role.** Prerequisite for Phase 27 — splits files, collapses duplicates, enables `test_transport_unification` shell.
> **Registry.** Full audit; updates [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md) process + [CODE_BUDGET.txt](CODE_BUDGET.txt).

> **Guidelines.** [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) — constraints 1–6.
> **Goal.** Every name in the codebase says what it does; every claim in the docs is true; every primitive has exactly one canonical home. After this phase a new contributor can read any file name and predict its contents, and the CI gates make it impossible to silently reintroduce duplication.
>
> **Outcome.** `MILV` → `post_transform_guard`. `BatchAccumulator` → `RequestCoalescer`. Dual `ExecutionPlan` collapses to one. `Candidate.score` / `CandidateScorer` formula has one home. Seven files over 800 LoC are split. Transform registry and runner factory are generated from the same source. **Beyond the original honesty-pass scope**, this phase now also lands:
> - `scripts/check_internal_no_duplication.sh` — walks [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md) and verifies every named primitive exists at its declared path and nowhere else.
> - `scripts/check_code_budget.sh` — enforces the per-directory LoC caps in [FORWARD_PLAN.md §6.1](FORWARD_PLAN.md). Every PR must declare its LoC delta; CI blocks merge on overrun.
> - An audit of **internal duplications** beyond the named cases — see §1.3.
>
> **Why this matters for the v2.0 forward plan.** Phase 12 sets the structural baseline that every later phase relies on. In particular, [Phase 14](14-hybrid-semantic-cache.md) needs the cache split, [Phase 15](15-native-guardrails.md) needs the guardrails split, [Phase 24](24-edge-wasm-core.md) needs the algorithms to have one canonical Python home before they can be ported to Rust, and [Phase 27](27-transport-layer-consolidation.md) needs the code-budget gate in place to enforce its -1500 LoC consolidation. Doing this work first is what keeps every subsequent phase lightweight — there's no scaffolding to drag around and no way to silently regress.
>
> **Estimated effort.** 4 days (was 3; expanded scope adds 1 day for the audit + CI gates). 1 PR, ~+1600/-3100 LoC net.

---

## 1.3 The internal-dedup audit — patterns this phase eliminates

In addition to the named honesty-pass items, this phase walks the codebase looking for the following patterns and consolidates each. These are the patterns that historically grow a codebase from "clean" to "unmaintainable":

| Pattern | Audit method | Resolution |
|---|---|---|
| **Two `ExecutionPlan` types** | `rg -nl "class ExecutionPlan"` | Already named in original scope — one home in `planner/execution_plan.py`. |
| **Two `Candidate.score` formulas** | `rg -nl "def score|class CandidateScorer"` | Already named — one home. |
| **Per-adapter retry helpers** (~ 17 copies) | `rg -nl "for attempt in range" src/lattice/providers/adapters/` | Audit only here; resolved in [Phase 27](27-transport-layer-consolidation.md). Add the contract test `test_no_per_adapter_httpx_client.py` shell now; populate when 27 lands. |
| **Multiple `httpx.AsyncClient` instantiation sites** | `rg -nl "httpx.AsyncClient\(" src/lattice/` | Same — flagged here, fixed in [Phase 27](27-transport-layer-consolidation.md). |
| **Multiple cost-estimation paths** | `rg -nl "def estimate_cost\|def compute_cost\|def calc_cost"` | One `telemetry/cost_estimator.py`. Other call sites import; never re-implement. |
| **Multiple config-loading entry points** | `rg -nl "yaml.safe_load\|tomllib.load\|TOML.load"` outside `core/config.py` | One loader: `LatticeConfig.from_env` / `from_file`. CLI / proxy / SDK / tests all use it. |
| **Multiple Session dataclasses** | `rg -nl "@dataclass.*\nclass Session"` | One in `state/session.py`. |
| **Multiple chunk-buffer / streaming-decoder implementations** | `rg -nl "def feed.*chunk\|class.*Buffer"` | One in `pipeline/streaming/chunk_buffer.py`. Rust mirror in Phase 24; parity-tested. |
| **Multiple receipt / audit-log schemas** | `rg -nl "class.*Receipt\|class.*AuditLog"` | One in `audit/receipts.py` (lands in Phase 23). Phase 12 reserves the path. |
| **Multiple rate-limit parsers** (per-adapter `retry-after` parsing) | `rg -nl "retry-after\|ratelimit-reset"` in adapters | Flagged here, consolidated in [Phase 27](27-transport-layer-consolidation.md). |
| **`x-lattice-*` header emission scattered across handlers** | `rg -nl 'headers\["x-lattice' src/lattice/` | One emitter in `proxy/middleware.py`. Handlers stash data on `request.state`; middleware reads + emits. |
| **Multiple tokenizers** | `rg -nl "tiktoken.encoding_for_model\|count_tokens"` | One wrapper in `utils/token_count.py`; all tokenization goes through it. |

For each of the above, the PR contains both the consolidation **and** a CI test that fails on regression. The tests are listed individually in `tests/contract/`:

- `test_no_dup_execution_plan.py`
- `test_no_dup_candidate_score.py`
- `test_no_dup_cost_estimator.py`
- `test_no_dup_config_loader.py`
- `test_no_dup_session_dataclass.py`
- `test_no_dup_chunk_buffer.py`
- `test_no_dup_tokenizer.py`
- `test_single_header_emitter.py`

Plus the umbrella registry-check `tests/contract/test_single_source_of_truth.py` that consumes [SINGLE_SOURCE_OF_TRUTH.md](SINGLE_SOURCE_OF_TRUTH.md) as data and enforces every row.

---

## 1.4 The code-budget gate

`scripts/check_code_budget.sh` lands in this phase and is wired into `.github/workflows/refactor-gate.yml`. It performs three checks:

1. **Total `src/lattice/` LoC ≤ declared budget per phase.** The current phase's budget lives in `docs/refactor/CODE_BUDGET.txt` (a one-liner per phase) and is read by the script.
2. **Per-directory caps** from [FORWARD_PLAN.md §6.1](FORWARD_PLAN.md), mirrored in `docs/refactor/CODE_BUDGET.txt`.
3. **Per-PR delta matches declaration.** Every PR with the label `phase:NN` must reference the phase doc and the doc's declared `LoC delta`; CI computes actual `git diff --stat` LoC delta and fails if it exceeds declared + 10%.

Block-merge behaviour: a PR over budget either (a) gets the delta declaration updated with a justification in the phase doc, or (b) is rejected.

The script is ~80 LoC of bash + awk. Lives at `scripts/check_code_budget.sh`. No runtime cost; CI-only.

---

## 1.5 Validation engine consolidation + replay contract

Production evals and the architecture review surfaced **scattered validation** (MILV, guardrails, `ir/validation`, per-optimizer checks) without a single operator-facing story. Phase 12 adds a **facade only**:

| File | Role |
|---|---|
| `src/lattice/runtime/validation_engine.py` | **New.** `validate_ir()`, `validate_post_transform()`, `validate_output()` — delegate to existing modules; no duplicate logic |
| `src/lattice/pipeline/milv.py` | Renamed from misleading `MILV`; still owns post-transform checks |
| `src/lattice/ir/validation.py` | IR structural rules unchanged |

**Replay determinism** (eval hardening, not a new feature):

- `tests/contract/test_replay_determinism.py` — fixed `(request, profile, ExecutionPlan)` → identical `PromptIRV2.canonical_fingerprint()` and transform trace hash
- Documented in [`docs/architecture/runtime.md`](../architecture/runtime.md) § Validation facade

**Scoring honesty:** collapse `Candidate.score` / `CandidateScorer` to the utility formula in [`runtime.md`](../architecture/runtime.md#the-scoring-rule). Remove duplicate `_estimate_utility` bonus tables from call sites; Phase 23 bandit learns weights — Phase 12 only ensures **one formula home**.

Cross-ref: [ARCHITECTURE_EVAL_INSIGHTS.md](ARCHITECTURE_EVAL_INSIGHTS.md).

---

## 1. Why this phase exists

A three-way audit on `refactor/phase-9-observability-state` surfaced the following items where the *name* of a thing in the repo does not match what it does, or where two implementations of the same thing exist side-by-side. None of these are user-visible bugs; all of them are credibility bugs. A user who reads the source code today and trusts the names ends up surprised.

| Item | What it claims | What it does | Source |
|---|---|---|---|
| `pipeline/milv.py` `MILVResult`, `validate_transform` | "Multi-Input Loss Validator" / "model-in-the-loop validation" | Regex Jaccard on digit tokens, UUID/URL/error-phrase preservation, placeholder leakage check. No model call. | [src/lattice/pipeline/milv.py](../../src/lattice/pipeline/milv.py) L1-9, L75-159 |
| `pipeline/batch_accumulator.py` `BatchAccumulator._flush_provider` | "Real OpenAI Batch API dispatch" (docstring claims 50% discount) | `await asyncio.sleep(0.1)` + stub results | [src/lattice/pipeline/batch_accumulator.py](../../src/lattice/pipeline/batch_accumulator.py) L228-231 |
| `planner/unified_planner._estimate_utility` | "Utility-based planner" | Fixed bonus table (+0.15 for representation_optimizer, +0.05 for tool calls); never reorders transforms; never used as an objective. | [src/lattice/planner/unified_planner.py](../../src/lattice/planner/unified_planner.py) L295-330 |
| `transforms/causal_chain.py` | Listed as a compression transform | Often *increases* tokens; correctly classified `OBSERVABILITY_ONLY` internally but the registry doesn't say so. | [src/lattice/transforms/causal_chain.py](../../src/lattice/transforms/causal_chain.py) |
| `ir/primitives.ExecutionPlan` + `planner/execution_plan.ExecutionPlan` | Same name | Two different dataclasses with different fields, bridged by `coerce_execution_plan` | [src/lattice/ir/primitives.py](../../src/lattice/ir/primitives.py) L327-339; [src/lattice/planner/execution_plan.py](../../src/lattice/planner/execution_plan.py) L118-167; [src/lattice/planner/runtime_state.py](../../src/lattice/planner/runtime_state.py) L11-37 |
| `ir/transform.CandidateScorer` + `ir/primitives.Candidate.score` | Single scoring formula | Implemented twice with identical math | [src/lattice/ir/transform.py](../../src/lattice/ir/transform.py) L116-172 vs [src/lattice/ir/primitives.py](../../src/lattice/ir/primitives.py) L234-273 |
| `transforms/registry.py` (26 specs) vs `pipeline/runner._FACTORIES` (17) | One canonical list | Two hand-maintained lists drift apart. `context_selector`, `columnar_pack`, `json_shape`, `diagnostic_rle`, `extractive_compress`, `representation_optimizer`, `causal_chain` are registry-only. | [src/lattice/transforms/registry.py](../../src/lattice/transforms/registry.py) L65-307; [src/lattice/pipeline/runner.py](../../src/lattice/pipeline/runner.py) L87-117 |
| `cli.py` `benchmark` command | "Run benchmarks" | Prints a redirect to `benchmarks/evals/cli.py` | [src/lattice/cli.py](../../src/lattice/cli.py) |
| Config flags `transform_prefix_opt`, `transform_constraint_lifting`, `transform_strategy_selector` | Configurable transforms | The transforms are deleted; flags are silent no-ops | [src/lattice/core/config.py](../../src/lattice/core/config.py) |

Seven files violate Ground-Rule R6 (no file >800 LoC in v1.0.0):

| File | LoC |
|---|---|
| [src/lattice/gateway/compat.py](../../src/lattice/gateway/compat.py) | 3176 |
| [src/lattice/integrations/agents.py](../../src/lattice/integrations/agents.py) | 1689 |
| [src/lattice/cli.py](../../src/lattice/cli.py) | 1032 |
| [src/lattice/cache/semantic.py](../../src/lattice/cache/semantic.py) | 1014 |
| [src/lattice/ir/builder.py](../../src/lattice/ir/builder.py) | 821 |
| [src/lattice/providers/adapters/anthropic.py](../../src/lattice/providers/adapters/anthropic.py) | 812 |
| [src/lattice/pipeline/runner.py](../../src/lattice/pipeline/runner.py) | 802 |

This phase fixes all of the above in one PR. None of it changes user-visible behaviour.

---

## 2. Files touched

### 2.1 Renamed / collapsed

| Current | Target | Notes |
|---|---|---|
| `src/lattice/pipeline/milv.py` | `src/lattice/pipeline/post_transform_guard.py` | Class renames (`MILVResult` → `PostTransformGuardResult`, `validate_transform` → `evaluate_post_transform`); body folded into the same module's `guardrails.py` if any check duplicates one already there |
| `src/lattice/pipeline/batch_accumulator.py` | `src/lattice/pipeline/request_coalescer.py` | `_flush_provider` stub deleted; class becomes the in-process coalescer it actually is. Real Batch API arrives in Phase 20, not here. |
| `src/lattice/planner/execution_plan.py` | merged into `src/lattice/ir/primitives.py` then deleted | All rich fields (`utility_score`, `cache_plan`, `transport_plan`) move onto the canonical `ir.primitives.ExecutionPlan` |
| `src/lattice/planner/runtime_state.py::coerce_execution_plan` | deleted | No two types to coerce |

### 2.2 Split

| File | New layout |
|---|---|
| `src/lattice/gateway/compat.py` (3176) | `src/lattice/gateway/compat/{__init__.py, openai_chat.py, openai_responses.py, anthropic_messages.py, codex.py, translation.py, headers.py}` |
| `src/lattice/integrations/agents.py` (1689) | `src/lattice/integrations/{registry.py, lifecycle.py, doctor.py, profiles.py, env_builder.py}` |
| `src/lattice/cli.py` (1032) | `src/lattice/cli/{__init__.py, _runner.py, proxy.py, init.py, lace.py, doctor.py, info.py, mcp.py}` (the package's `__init__.py` keeps the `main()` entrypoint advertised in `pyproject.toml`) |
| `src/lattice/cache/semantic.py` (1014) | `src/lattice/cache/{__init__.py, semantic.py, fingerprint.py, stores.py, eviction.py}` (semantic.py keeps only the `SemanticCache` façade; the rest move) |
| `src/lattice/ir/builder.py` (821) | `src/lattice/ir/builder/{__init__.py, messages.py, tools.py, system.py}` |
| `src/lattice/providers/adapters/anthropic.py` (812) | `src/lattice/providers/adapters/anthropic/{__init__.py, request.py, response.py, cache.py, thinking.py}` |
| `src/lattice/pipeline/runner.py` (802) | `src/lattice/pipeline/{runner.py, executor.py, reverse.py}` — `runner.py` keeps the public `Pipeline` class and the `compress()` / `process()` entry points; `executor.py` owns the per-transform inner loop ported from `_execute_transform_in_compress`; `reverse.py` owns the response side |

### 2.3 Generated / single source of truth

| Created | Removed |
|---|---|
| `scripts/generate_factories.py` — reads `transforms/registry.py`, emits `pipeline/runner._FACTORIES` body inline-comments and a generated tuple in `pipeline/runner._generated_factories.py` | Hand-maintained `_FACTORIES` dict literal in `pipeline/runner.py` |
| `tests/contract/test_registry_factory_parity.py` — fails if the generated tuple drifts | — |

### 2.4 Config cleanup

| Removed from `core/config.py` | Why |
|---|---|
| `transform_prefix_opt` | Deleted in Phase 5; flag is a no-op |
| `transform_constraint_lifting` | Deleted in Phase 5 |
| `transform_strategy_selector` | Deleted in Phase 5 |
| `transform_information_theoretic_selector` | Deleted in Phase 5 |

Replaced by a single `DeprecatedConfigKey` warning in `core/config.py::__init_subclass__` that logs once and points users at `docs/MIGRATION.md`.

### 2.5 CLI

| Removed | Replaced by |
|---|---|
| `lattice benchmark` stub in `cli.py` | `lattice benchmark` becomes a real wrapper in the new `cli/_runner.py` that calls `benchmarks.evals.cli:main` (Phase 9 already exists; this just stops being a print statement). |

### 2.6 Stale references

| Grep & replace | |
|---|---|
| `prefix_optimizer` in docs / benchmark JSON / telemetry strings | → `prefix_canonicalization` (the surviving impl in `protocol/`) |
| `MILV` in code, docs, README | → `PostTransformGuard` |
| `strategy_selector`, `constraint_lifting`, `information_theoretic_selector` in tests/docs/benchmark JSON | → delete |

---

## 3. Step-by-step

### 3.1 Step 1 — Rename MILV → PostTransformGuard

```bash
git mv src/lattice/pipeline/milv.py src/lattice/pipeline/post_transform_guard.py
```

In the renamed file:

```python
# src/lattice/pipeline/post_transform_guard.py
"""Rule-based post-transform validation.

Despite the historical name (MILV — "Multi-Input Loss Validator"), this module
contains no model calls. It performs deterministic checks on (before, after)
PromptIR pairs and returns a verdict the pipeline uses to roll back a transform
when its output fails a structural test.
"""

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class PostTransformGuardResult:
    passed: bool
    score: float
    reasons: tuple[str, ...]


def evaluate_post_transform(
    transform_name: str,
    before: PromptIRV2,
    after: PromptIRV2,
    context: TransformContext,
) -> PostTransformGuardResult:
    ...
```

Search-and-replace at the call sites:

```bash
rg -l "MILVResult|validate_transform|should_trigger_milv" src/ tests/ \
  | xargs sed -i '' 's/MILVResult/PostTransformGuardResult/g; s/validate_transform/evaluate_post_transform/g; s/should_trigger_milv/should_trigger_post_transform_guard/g'
```

In [src/lattice/pipeline/runner.py](../../src/lattice/pipeline/runner.py) L616-669, the call site keeps the same shape but reads:

```python
from lattice.pipeline.post_transform_guard import evaluate_post_transform, should_trigger_post_transform_guard

if should_trigger_post_transform_guard(transform_name, ctx):
    verdict = evaluate_post_transform(transform_name, before_ir, after_ir, ctx)
    if not verdict.passed:
        ctx.record_rollback(transform_name, reason="post_guard:" + ",".join(verdict.reasons))
        request = backup
        continue
```

### 3.2 Step 2 — Collapse duplicated checks into `guardrails.py`

Three modules currently implement the same entity/numeric/root-cause regex sweep:

- [src/lattice/pipeline/guardrails.py](../../src/lattice/pipeline/guardrails.py) — `check_entity_preservation`, `check_critical_signal_loss`
- `pipeline/post_transform_guard.py` (was milv.py)
- [src/lattice/pipeline/gates.py](../../src/lattice/pipeline/gates.py) L326-371 — `_psg_preservation_gate`

Action:

1. Move every regex / set / counter into a new module `pipeline/checks.py`:

   ```python
   # src/lattice/pipeline/checks.py
   """Atomic structural checks shared by guardrails, gates, and post_transform_guard.

   Every function takes (before_text, after_text) or (before_ir, after_ir) and
   returns a CheckResult. No I/O, no state, no model calls.
   """

   @dataclass(frozen=True, slots=True)
   class CheckResult:
       name: str
       passed: bool
       score: float
       detail: str = ""

   def numbers_preserved(before: str, after: str) -> CheckResult: ...
   def uuids_preserved(before: str, after: str) -> CheckResult: ...
   def urls_preserved(before: str, after: str) -> CheckResult: ...
   def file_paths_preserved(before: str, after: str) -> CheckResult: ...
   def error_signals_preserved(before: str, after: str, task: TaskClass) -> CheckResult: ...
   def root_cause_phrases_preserved(before: str, after: str) -> CheckResult: ...
   def placeholder_leakage(before: str, after: str, alias_table: dict) -> CheckResult: ...
   def expansion_within_ratio(before_tokens: int, after_tokens: int, ratio: float) -> CheckResult: ...
   def negative_savings(before_tokens: int, after_tokens: int) -> CheckResult: ...
   ```

2. Rewrite `guardrails.py`, `gates.py`, and `post_transform_guard.py` to compose these primitives. Each module's exported function becomes 10-30 LoC.

3. Delete the duplicated regex constants and bodies. Net reduction: ~600 LoC.

### 3.3 Step 3 — Kill the fake batch dispatcher

```bash
git mv src/lattice/pipeline/batch_accumulator.py src/lattice/pipeline/request_coalescer.py
```

In the renamed file:

- Delete `_flush_provider` (the `await asyncio.sleep(0.1)` simulator).
- Rename class `BatchAccumulator` → `RequestCoalescer`.
- Rename `BatchResult` → `CoalescedResult`.
- Update the docstring to say what it actually does (in-process coalescing of compatibility-keyed concurrent requests), and to **explicitly disclaim** any connection to provider Batch APIs.
- Drop the `provider_batch_supported` capability flag.
- Update [src/lattice/proxy/bootstrap.py](../../src/lattice/proxy/bootstrap.py) injection site.

Real OpenAI / Anthropic Batch API support is documented in [Phase 20](20-non-chat-surfaces.md) — leave a `# Real Batch API: see docs/refactor/20-non-chat-surfaces.md` comment in the file.

Tests at `tests/unit/pipeline/test_batch_accumulator.py` are renamed and the assertions that confirmed the simulator's `0.1` sleep are deleted. Replace with assertions on the coalescing keys.

### 3.4 Step 4 — Collapse dual ExecutionPlan

The canonical type is [src/lattice/ir/primitives.py](../../src/lattice/ir/primitives.py)'s `ExecutionPlan`. Move the rich fields onto it:

```python
# src/lattice/ir/primitives.py
@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    transforms: tuple[str, ...]
    quality_floor: float
    latency_budget_ms: float

    # Phase 12: fields previously on planner.execution_plan.ExecutionPlan
    tier: Tier | None = None
    utility_score: float = 0.0
    cache_plan: CachePlan | None = None
    transport_plan: TransportPlan | None = None
    metadata: frozendict[str, Any] = field(default_factory=frozendict)
```

Then:

1. `git rm src/lattice/planner/execution_plan.py`
2. In [src/lattice/planner/unified_planner.py](../../src/lattice/planner/unified_planner.py), change `from lattice.planner.execution_plan import ExecutionPlan` → `from lattice.ir.primitives import ExecutionPlan`. Builders that previously constructed the rich type now construct the canonical type.
3. In [src/lattice/planner/execution_builder.py](../../src/lattice/planner/execution_builder.py), `build_execution_plan()` returns the canonical type; delete the wrap/unwrap helpers.
4. In [src/lattice/planner/runtime_state.py](../../src/lattice/planner/runtime_state.py), delete `coerce_execution_plan`. Call sites get the canonical type directly.

Touch list:

```bash
rg -l "from lattice.planner.execution_plan|planner.execution_plan.ExecutionPlan" src/ tests/
```

Expected count: ~18 call sites.

### 3.5 Step 5 — Single scoring formula

The formula in [src/lattice/ir/transform.py](../../src/lattice/ir/transform.py) L116-172 and [src/lattice/ir/primitives.py](../../src/lattice/ir/primitives.py) L234-273 must agree by construction.

1. Move the formula into a new `ir/scoring.py`:

   ```python
   # src/lattice/ir/scoring.py
   def composite_score(metrics: Mapping[str, float], *, weights: ScoringWeights = DEFAULT_WEIGHTS) -> float:
       return (
           metrics.get("quality_estimate", 1.0)
           + weights.cost * metrics.get("cost_reduction", 0.0)
           + weights.cache * metrics.get("cache_gain", 0.0)
           + weights.transport * metrics.get("transport_gain", 0.0)
           - metrics.get("semantic_risk", 0.0)
           - weights.latency * (metrics.get("latency_ms", 0.0) / 1000.0)
           - metrics.get("instability", 0.0)
       )

   @dataclass(frozen=True, slots=True)
   class ScoringWeights:
       cost: float = 0.5
       cache: float = 0.2
       transport: float = 0.2
       latency: float = 1.0

   DEFAULT_WEIGHTS = ScoringWeights()
   ```

2. `Candidate.score` in `ir/primitives.py` calls `composite_score(self.metrics)`.
3. `CandidateScorer` in `ir/transform.py` calls the same function. The class shrinks to a thin caller that handles weight overrides per beam config.

### 3.6 Step 6 — Generate `_FACTORIES` from the registry

Today [src/lattice/pipeline/runner.py](../../src/lattice/pipeline/runner.py) L87-117 hand-maintains a dict. Drift is detectable but only manually.

1. Annotate every spec in [src/lattice/transforms/registry.py](../../src/lattice/transforms/registry.py) with a `factory: Callable[[], ReversibleSyncTransform] | None`:

   ```python
   TransformSpec(
       name="reference_sub",
       priority=20,
       safety_class=SafetyClass.CONDITIONAL,
       default_pipeline=True,
       factory=lambda: ReferenceSubstitution(),
   )
   ```

   Specs that are registry-only (experimental, not in the live pipeline) leave `factory=None`.

2. New `pipeline/_generated_factories.py`:

   ```python
   # AUTO-GENERATED by scripts/generate_factories.py. Do not edit by hand.
   from collections.abc import Mapping
   from lattice.pipeline.base import ReversibleSyncTransform
   from lattice.transforms.registry import BUILTIN_TRANSFORMS

   def build_default_factories() -> Mapping[str, Callable[[], ReversibleSyncTransform]]:
       return {spec.name: spec.factory for spec in BUILTIN_TRANSFORMS if spec.factory is not None}
   ```

3. `PipelineTransformRegistry.__init__` calls `build_default_factories()` instead of carrying a literal dict.

4. Contract test:

   ```python
   # tests/contract/test_registry_factory_parity.py
   def test_every_default_pipeline_transform_has_factory():
       for spec in BUILTIN_TRANSFORMS:
           if spec.default_pipeline:
               assert spec.factory is not None, f"{spec.name} marked default_pipeline but has no factory"
   ```

### 3.7 Step 7 — Split files >800 LoC

The discipline is identical for each file: move related private helpers to sibling modules; re-export from the original module via `__init__.py` only the previously-exported public names. **No public surface change.** This is purely structural.

Example for `gateway/compat.py`:

```bash
mkdir -p src/lattice/gateway/compat
git mv src/lattice/gateway/compat.py src/lattice/gateway/compat/_orig.py
```

Then carve `_orig.py` into:

- `compat/openai_chat.py` — `make_openai_chat_handler`, `deserialize_openai_request`, `serialize_openai_response`, SSE streaming helpers
- `compat/openai_responses.py` — Responses API + Codex aliases + WebSocket
- `compat/anthropic_messages.py` — `deserialize_anthropic_request`, `serialize_anthropic_response`, tool_use↔tool_calls mapping
- `compat/codex.py` — Codex JWT auth, WS handler
- `compat/translation.py` — request_from_dict, message normalization, image placeholder
- `compat/headers.py` — `build_routing_headers`, `LatticeHeaderMiddleware` plumbing helpers
- `compat/__init__.py` — re-export every public name that was previously importable from `gateway.compat`

Run `rg "from lattice.gateway.compat import"` first; the set of re-exports must equal that grep output. Delete `_orig.py` after all imports resolve.

Repeat for the other six files. Each split is a separate commit in the same PR for review tractability.

### 3.8 Step 8 — Config flag cleanup

In [src/lattice/core/config.py](../../src/lattice/core/config.py), delete the four dead `transform_*` Booleans. Add a one-shot deprecation warning when a user passes one in:

```python
_DEPRECATED_TRANSFORM_FLAGS = frozenset({
    "transform_prefix_opt",
    "transform_constraint_lifting",
    "transform_strategy_selector",
    "transform_information_theoretic_selector",
})

class LatticeConfig(BaseSettings):
    @model_validator(mode="before")
    @classmethod
    def _warn_deprecated(cls, values: dict) -> dict:
        for key in set(values) & _DEPRECATED_TRANSFORM_FLAGS:
            logger.warning(
                "config: %r is deprecated and ignored; transform removed in v1.0.0 "
                "(see docs/MIGRATION.md#deleted-transforms)", key,
            )
            values.pop(key)
        return values
```

### 3.9 Step 9 — Stale-reference sweep

```bash
rg -n "prefix_optimizer|MILV|strategy_selector|constraint_lifting|information_theoretic_selector" \
   docs/ benchmarks/ src/ tests/ README.md AGENTS.md
```

For each hit, either:
- Replace with the canonical name (`prefix_canonicalization`, `PostTransformGuard`), or
- Delete the reference if the feature is gone.

The `benchmarks/results/production_evals.json` file is allowed to retain historical names — but a new `benchmarks/results/SCHEMA_VERSION` file pinning `v2` notes the migration.

### 3.10 Step 10 — Real `lattice benchmark`

In the new `cli/_runner.py`:

```python
def benchmark_command(args: argparse.Namespace) -> int:
    from benchmarks.evals.cli import main as evals_main
    return evals_main(args.benchmark_args or [])
```

Wire `cli/__init__.py::main()` so `lattice benchmark --suite all` is a real one-hop wrapper.

---

## 4. Test plan

| Check | Command | Threshold |
|---|---|---|
| Lint | `uv run ruff check src/ tests/` | 0 errors |
| Format | `uv run ruff format --check src/ tests/` | 0 errors |
| Types | `uv run mypy src/lattice/` | 0 errors |
| Unit | `uv run pytest tests/unit -q` | 1760 + new tests, 0 skipped (excluding live-provider) |
| Contract | `uv run pytest tests/contract -q` | All pass; new `test_registry_factory_parity.py` and `test_no_dead_config_flags.py` included |
| File-size R6 | `find src/lattice -name '*.py' -size +30k` | 0 files |
| Dead names | `rg "MILV\|BatchAccumulator\|strategy_selector\|constraint_lifting\|information_theoretic_selector" src/` | 0 hits |
| Single ExecutionPlan | `rg "from lattice.planner.execution_plan" src/ tests/` | 0 hits |
| Canonical bench | `bash scripts/run_canonical_benchmark.sh` | ±2% vs `phase-0-baseline.json` |

New test files:

```
tests/contract/test_registry_factory_parity.py
tests/contract/test_no_dead_config_flags.py
tests/contract/test_no_dead_names.py        # asserts the grep above stays empty
tests/unit/pipeline/test_checks.py            # tests the new shared checks module
tests/unit/ir/test_scoring.py                 # tests the single composite_score
```

---

## 5. Acceptance criteria

1. `git grep -nE "(MILV|BatchAccumulator|strategy_selector|constraint_lifting|information_theoretic_selector)" -- src/ tests/` returns zero lines.
2. `find src/lattice -name '*.py' -size +30k` returns zero lines.
3. `python -c "from lattice.ir.primitives import ExecutionPlan; from lattice.planner import ExecutionPlan as P2; assert ExecutionPlan is P2"` passes (single re-export points at the single class).
4. `python -c "import lattice.pipeline.checks as c; assert callable(c.numbers_preserved)"` passes.
5. `lattice benchmark --help` prints the real benchmarks CLI help, not a stub redirect.
6. Canonical bench vs `phase-0-baseline.json` within ±2% on every aggregate metric.
7. `docs/MIGRATION.md` updated with a "Phase 12" section listing every rename so downstream consumers know what to change.

---

## 6. Out of scope

These belong to later phases and must **not** be smuggled into this PR:

| Topic | Phase |
|---|---|
| Real OpenAI/Anthropic Batch API integration | [Phase 20](20-non-chat-surfaces.md) |
| `_estimate_utility` → real expected-utility optimization | [Phase 23](23-receipts-bandit-profiles.md) (bandit); inputs from [22](22-cache-portability.md), [27](27-transport-layer-consolidation.md) |
| Segment-aware planning | [Phase 19.5](19.5-segment-aware-planning.md) |
| LLMLingua-2 plug-in | [Phase 19](19-compression-intelligence.md) |
| TypeScript SDK | [Phase 17](17-typescript-sdk.md) |

This phase is structural and naming-only; behaviour must stay byte-identical.
