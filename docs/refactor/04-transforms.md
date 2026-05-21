# Phase 4 — Transforms Audit & Cleanup

> **Goal.** Walk every transform file under `src/lattice/transforms/`, decide its fate, split the three giants (`content_profiler.py` 985 LoC, `format_conv.py` 794 LoC, `strategy_selector.py` 728 LoC) into packages, delete `prefix_opt.py` outright, gate `strategy_selector` and the information-theoretic variant of `context_selector` on benchmark evidence (delete unless they earn their keep), move `transform_registry.py` and `transform_reputation.py` from `core/` into `transforms/`, and move `utils/patterns.py` to `transforms/patterns.py`. Apply transform-specific config_flag fixes (e.g. `delta_encode` currently references `transform_batching`).
>
> **Outcome.** `src/lattice/transforms/` contains exactly the transforms the registry advertises. Each transform file has one concern, max 800 LoC. Every transform implements `optimize(ir, request, ctx)` only. The README claim count matches the registry exactly.
>
> **Estimated effort.** 4 days.

---

## 1. Why this phase exists

The audit found four problems:

1. **Three god-files conflate concerns.**
   - `content_profiler.py` (985 LoC) is doing profile classification, risk scoring, task classification bridge, IR construction, prefix canonicalisation glue, manifest building, cache plan simulation, execution plan derivation, scheduler bridge, and segmenter glue — at least ten responsibilities.
   - `format_conv.py` (794 LoC) is doing Markdown↔CSV conversion, JSON↔YAML conversion, table detection, IR-native optimisation, and reverse logic.
   - `strategy_selector.py` (728 LoC) is doing bandit-algorithm machinery (arm state, UCB1, reward computation) and strategy routing.

2. **`prefix_opt.py` is explicitly deprecated by its own header.** It's a 161-LoC wrapper around content_profiler output. It runs at priority 10 in the default pipeline; its work has already been done at priority 1 (content_profiler).

3. **`strategy_selector.py` and the information-theoretic variant of `context_selector.py` have no benchmark evidence.** They might be doing nothing. Phase 4 runs the benchmark suite with each disabled; if no compression / quality / latency regression appears, they are deleted.

4. **`transforms/semantic_segmenter.py` is not a transform** (no `ReversibleSyncTransform` subclass; just a dataclass module). Phase 1 already moved it to `core/segmentation.py`. Verify here.

5. **`transform_registry.py` and `transform_reputation.py` live in `core/` but are owned by `transforms/`.** Move them.

6. **`utils/patterns.py` (regex patterns) is used only by transforms.** Move it.

7. **`delta_encode.py` has a `config_flag` referencing `transform_batching`** — copy-paste bug. Fix.

---

## 2. Files touched

### 2.1 Moved

| Current path | New path |
|---|---|
| `src/lattice/core/transform_registry.py` | `src/lattice/transforms/registry.py` |
| `src/lattice/core/transform_reputation.py` | `src/lattice/transforms/reputation.py` |
| `src/lattice/utils/patterns.py` | `src/lattice/transforms/patterns.py` |

### 2.2 Split

| Current file | New package |
|---|---|
| `src/lattice/transforms/content_profiler.py` (985 LoC) | `src/lattice/transforms/content_profiler/` (5 files) |
| `src/lattice/transforms/format_conv.py` (794 LoC) | `src/lattice/transforms/format_converter/` (3 files) |
| `src/lattice/transforms/strategy_selector.py` (728 LoC) | `src/lattice/transforms/strategy_selector/` (2 files) *or DELETED — see §4.2* |

### 2.3 Deleted

```
src/lattice/transforms/prefix_opt.py                 # 161 LoC — deprecated, logic in content_profiler
src/lattice/transforms/strategy_selector.py          # 728 LoC — IF benchmark gate fails; else split into package per §2.2
```

Conditional on benchmark evidence (§4.3):

```
src/lattice/transforms/context_selector.py           # IF info-theoretic variant has no evidence: simplify to submodular-only
```

### 2.4 Modified

- Every transform file's `config_flag` and `name` strings in `transforms/registry.py` (formerly `core/transform_registry.py`).
- `src/lattice/transforms/__init__.py` — kept lightweight; the registry handles discovery.
- `src/lattice/core/__init__.py` — drop the `transform_registry` import (Phase 4 finalises core/'s slimming).
- `delta_encode.py` — fix the `config_flag` typo.
- Per-transform tests — move and update imports.

---

## 3. Transform-by-transform disposition

The audit produced a 24-row matrix. Replicated here with concrete actions and acceptance criteria. Order is by registry priority.

### 3.1 content_profiler (priority 1, default, 985 LoC) — **SPLIT**

Today's responsibilities (from the audit):

1. `ContentProfiler` class — profile classification dispatch
2. `ContentProfile` enum — content classes (CODE_HEAVY, TABLE_HEAVY, NARRATIVE_LONG, ...)
3. Content analysis — regex/structural classification
4. Risk scoring — `SemanticRiskScore` computation
5. Task classification — delegates to `core/task_classifier.py` (now `planner/task_classifier.py`)
6. IR building — calls `build_ir` (now `lattice.ir.builder.build_ir`)
7. Prefix canonicalisation — calls `protocol/prefix_canonicalization`
8. Protocol manifest building — calls `protocol/manifest`
9. Cache plan simulation — provider-specific cache strategy
10. Execution plan derivation — calls `planner/unified_planner.UnifiedPlanner.plan()`
11. Schedule bridge — historically built v1's `SchedulerDecision` (deleted in Phase 3)

After Phase 3, items 5, 10, 11 are simply forwarding calls — most can be inlined. After Phase 1+2, items 6-8 are deferred imports.

**Split target:**

```
src/lattice/transforms/content_profiler/
├── __init__.py                      # ~120 LoC — ContentProfiler class + process() / optimize() (entry point)
├── classifier.py                    # ~180 LoC — ContentProfile enum + classify_by_signals() heuristics
├── risk_scorer.py                   # ~150 LoC — compute_risk_score(), 8-dim risk vector
├── task_classifier_bridge.py        # ~60 LoC — thin call into planner/task_classifier
└── planner_bridge.py                # ~100 LoC — IR build + prefix canon + manifest + cache plan + planner call
```

**`__init__.py` shape:**

```python
"""Content profiler — priority 1 transform that runs FIRST and populates IR metadata.

Splits the historic 985-LoC monolith into focused submodules. The public
class ContentProfiler still satisfies ReversibleSyncTransform.
"""

from lattice.transforms.content_profiler.classifier import (
    ContentProfile, classify_by_signals,
)
from lattice.transforms.content_profiler.risk_scorer import (
    SemanticRiskScore, compute_risk_score,
)
from lattice.transforms.content_profiler.task_classifier_bridge import (
    bridge_task_classification,
)
from lattice.transforms.content_profiler.planner_bridge import (
    build_ir_and_plan,
)

class ContentProfiler:
    name = "content_profiler"
    priority = 1

    def can_process(self, request, context) -> bool:
        return True   # always runs first

    def optimize(self, ir, request, context):
        profile = classify_by_signals(request)
        risk = compute_risk_score(request, ir, profile)
        task_classification = bridge_task_classification(request)
        plan_artefacts = build_ir_and_plan(request, ir, profile, task_classification, context)
        # populate context.metadata with each artefact
        context.set("content_profile", profile)
        context.set("semantic_risk", risk)
        context.set("task_classification", task_classification)
        context.set("ir_metadata", plan_artefacts.ir_metadata)
        context.set("execution_plan", plan_artefacts.execution_plan)
        context.set("cache_plan", plan_artefacts.cache_plan)
        return Ok(ir)   # content_profiler doesn't transform IR; only metadata

    def reverse(self, response, context):
        return response   # no-op

__all__ = [
    "ContentProfiler", "ContentProfile", "classify_by_signals",
    "SemanticRiskScore", "compute_risk_score",
]
```

**`classifier.py`** — pull all the regex/heuristic content-class scoring out of the monolith:

| Function | LoC budget |
|---|---|
| `ContentProfile` enum | 20 |
| `classify_by_signals(request)` | 100 |
| `_score_code_heavy`, `_score_table_heavy`, `_score_narrative_long`, etc. | 60 |

**`risk_scorer.py`** — pull the `SemanticRiskScore` dataclass and `compute_risk_score()`:

| Function | LoC budget |
|---|---|
| `SemanticRiskScore` dataclass | 30 |
| `compute_risk_score(request, ir, profile)` | 80 |
| 8 dimension helpers (entity, format, signal, ...) | 40 |

**`task_classifier_bridge.py`** — thin glue:

```python
from lattice.planner.task_classifier import classify_task, TaskClassification
from lattice.transport.types import Request

def bridge_task_classification(request: Request) -> TaskClassification:
    """Wrap planner classify_task with content-profiler-specific overrides if any."""
    return classify_task(request)
```

**`planner_bridge.py`** — the IR/manifest/cache plan glue:

```python
from dataclasses import dataclass
from lattice.ir import build_ir, normalize_ir, PromptIRV2
from lattice.protocol.prefix_canonicalization import canonicalize_request_prefix
from lattice.protocol.manifest import build_manifest
from lattice.planner.provider_strategy import build_cache_plan_for_provider, simulate_provider_cache
from lattice.planner.unified_planner import UnifiedPlanner, SemanticProfile

@dataclass(frozen=True)
class PlanArtefacts:
    ir_metadata: dict
    execution_plan: object
    cache_plan: object

def build_ir_and_plan(request, ir, profile, task_classification, context):
    # ir already built by Pipeline; this just normalises and derives artefacts
    canon = canonicalize_request_prefix(request)
    manifest = build_manifest(request, canon)
    provider = context.provider
    cache_sim = simulate_provider_cache(provider, request)
    cache_plan = build_cache_plan_for_provider(provider, request, cache_sim)
    profile_for_planner = SemanticProfile.from_classification(task_classification)
    plan = UnifiedPlanner().plan(request, profile_for_planner)
    return PlanArtefacts(
        ir_metadata={"canon": canon, "manifest": manifest},
        execution_plan=plan,
        cache_plan=cache_plan,
    )
```

After the split: each file has one purpose; the largest is ~180 LoC. Total LoC drops because eliminated duplication and removed v1 schedule-bridge code (gone with Phase 3).

**Tests:**
- `tests/unit/transforms/content_profiler/test_classifier.py` — covers `classify_by_signals` heuristics.
- `tests/unit/transforms/content_profiler/test_risk_scorer.py` — covers the 8-dim risk scoring.
- `tests/unit/transforms/content_profiler/test_profiler_integration.py` — full `ContentProfiler.optimize(...)` against fixture requests.

### 3.2 runtime_contract (priority 2, default, 74 LoC) — **KEEP**

| Symbol | Action |
|---|---|
| `RuntimeContract` class | KEEP |
| `process()` method | DELETE (Phase 2 already removed for IR-native transforms) — verify |
| `optimize(ir, request, ctx)` | KEEP |
| Imports from `runtime/router` | UPDATED to `runtime/tier_classifier` in Phase 3 |
| `config_flag = "transform_runtime_contract"` | Verify in registry spec |

### 3.3 cache_arbitrage (priority 9, default, 545 LoC) — **KEEP**

| Symbol | Action |
|---|---|
| `CacheArbitrage` class | KEEP |
| `CacheArbitrageOutcome` helper class | KEEP (or extract to private file if file feels too dense; 545 LoC is acceptable) |
| `process()` | DELETED in Phase 2 |
| `optimize(ir, request, ctx)` | KEEP |

### 3.4 prefix_opt (priority 10, default, 161 LoC) — **DELETE**

The file's own header says (per audit): *"DEPRECATED: This transform will be removed entirely once all consumers migrate to reading from content_profiler output directly."* Phase 4 finishes the migration.

**Action:**

```bash
git rm src/lattice/transforms/prefix_opt.py
git rm tests/unit/test_prefix_opt.py   # if exists
```

In `transforms/registry.py` (the moved registry), remove the spec entry:

```python
# DELETE this TransformSpec:
TransformSpec(
    canonical_name="prefix_optimizer",
    aliases=("prefix_opt",),
    category=TransformCategory.PIPELINE,
    module_path="lattice.transforms.prefix_opt",
    ...
),
```

In `core/config.py` (`LatticeConfig`), keep the `transform_prefix_opt: bool = True` field for **one release** so existing `lattice.yaml` configs don't error; the registry just ignores it. Document in MIGRATION.md that the flag becomes a no-op at v1.0.0 and is removed in v1.1.

Verify no remaining references:

```bash
rg "prefix_opt|PrefixOpt|prefix_optimizer" src/ tests/ benchmarks/
```

Should match only:
- `core/config.py` (deprecated no-op flag)
- `docs/refactor/MIGRATION.md` (documentation)
- `CHANGELOG.md` (release notes)

### 3.5 message_dedup (priority 15, 304 LoC) — **KEEP**

| Symbol | Action |
|---|---|
| `MessageDedup` class | KEEP |
| `process()` | DELETED in Phase 2 |
| `optimize(ir, request, ctx)` | KEEP |
| n-gram Jaccard helpers | KEEP |

### 3.6 path_prefix (priority 23, 158 LoC) — **KEEP**

| Symbol | Action |
|---|---|
| `PathPrefixCompressor` class | KEEP |
| `process()` | DELETED in Phase 2 |
| `optimize(ir, request, ctx)` | KEEP |

### 3.7 reference_sub (priority 20, default, 497 LoC) — **KEEP**

| Symbol | Action |
|---|---|
| `ReferenceSubstitution` class | KEEP |
| Helper functions (`_extract_code_blocks`, `_restore_code_blocks`, `_find_repeated_phrases`) | KEEP |
| `process()` | DELETED in Phase 2 |
| `optimize(ir, request, ctx)` | KEEP |

### 3.8 rate_distortion (priority 22, 216 LoC) — **KEEP**

`process()` deleted in Phase 2. Verify `optimize(ir, ...)` is the only entry.

### 3.9 path_prefix, extractive_compress, diagnostic_rle, columnar_pack, json_shape, causal_chain, constraint_lifting — **KEEP** (with caveats)

| Transform | LoC | Action |
|---|---|---|
| `extractive_compress` (priority 22, 136) | KEEP — complements `rate_distortion` (lossless variant) |
| `diagnostic_rle` (priority 17, 147) | KEEP — narrow, low-cost |
| `columnar_pack` (priority 19, 134) | KEEP — Markdown tables; complements `format_converter` |
| `json_shape` (priority 21, 154) | KEEP — JSON shape factoring; complements `format_converter` |
| `causal_chain` (priority 9, 192) | KEEP — observability-only; default off |
| `constraint_lifting` (priority 6, 173) | KEEP — observability-only. Phase 2 marked it `legacy_only=True`. Phase 4 either (a) ports `process()` to `optimize()` and unsets the flag, or (b) deletes it. **Decision rule: if there's any production consumer of the constraint metadata it sets, port. Otherwise delete.** Grep `rg "constraint_lifting" src/ tests/ docs/` — only the file + a test + the registry. **Action: DELETE** unless reviewer flags a missed consumer. |

### 3.10 format_conversion (priority 25, default, 794 LoC) — **SPLIT** + RENAME

Today `format_conv.py` does:

1. `DataShape` enum (JSON, TABLE, MARKDOWN, YAML, ...)
2. Table detection (Markdown table parsing + heuristics)
3. Markdown → CSV conversion (with quality validation)
4. JSON → YAML conversion (nested flattening)
5. Format validation & repair
6. IR-native `optimize(ir, ...)` dispatcher
7. Reverse logic (re-expand CSV → Markdown table on response)

**Split target:**

```
src/lattice/transforms/format_converter/
├── __init__.py                      # ~180 LoC — FormatConverter class (optimize/reverse dispatch) + DataShape enum
├── table_converter.py               # ~300 LoC — Markdown ↔ CSV (detection, conversion, validation, reverse)
└── json_converter.py                # ~250 LoC — JSON ↔ YAML + nested flattening
```

**Rename:** `format_conv.py` → `format_converter/__init__.py`. The transform's registered name (`format_conversion`) stays unchanged.

`format_converter/__init__.py`:

```python
"""Format conversion transform — converts structured data to token-efficient formats.

Dispatches to:
- table_converter.py for Markdown ↔ CSV
- json_converter.py for JSON ↔ YAML and nested flattening
"""

from enum import Enum
from lattice.transforms.format_converter.table_converter import (
    detect_markdown_table, convert_markdown_table_to_csv, restore_csv_to_markdown_table,
)
from lattice.transforms.format_converter.json_converter import (
    detect_json_block, convert_json_to_yaml, restore_yaml_to_json,
)

class DataShape(Enum):
    JSON = "json"
    TABLE = "table"
    MARKDOWN = "markdown"
    YAML = "yaml"
    NONE = "none"

class FormatConverter:
    name = "format_conversion"
    priority = 25

    def can_process(self, request, context) -> bool:
        # Detect whether any section has table or large JSON
        return any(detect_markdown_table(s) or detect_json_block(s)
                   for s in (request.messages_text() if hasattr(request, "messages_text") else [request.system_message or ""]))

    def optimize(self, ir, request, context):
        # iterate ir.sections, dispatch per section type
        ...

    def reverse(self, response, context):
        # restore CSV → Markdown / YAML → JSON
        ...

__all__ = ["FormatConverter", "DataShape"]
```

**`table_converter.py`** — Markdown table parsing, column-type inference, CSV emission, reverse logic.

**`json_converter.py`** — JSON detection, YAML emission with nested-flattening rules, reverse.

After the split, each file has a single concern. Largest is `table_converter.py` at ~300 LoC.

### 3.11 tool_projection (priority 29, 224 LoC) — **KEEP**

`process()` deleted in Phase 2. `optimize(ir, ...)` is canonical.

### 3.12 tool_filter (priority 30, default, 232 LoC) — **KEEP**

`process()` deleted in Phase 2. `optimize(ir, ...)` is canonical.

### 3.13 output_cleanup (priority 40, default, 160 LoC) — **KEEP**

Response-side only; no IR-native variant needed. Keep `process()` because it operates on `Response`, not `Request`/`ir`. Update the Protocol comment to clarify this is a response-side transform.

Add a new attribute on `TransformSpec`:

```python
class TransformSpec:
    ...
    is_response_side: bool = False
```

Mark `output_cleanup` with `is_response_side=True`. The `Pipeline` runner uses this to call `process(response, ctx)` post-response instead of `optimize(ir, ...)` pre-request.

### 3.14 context_selector (priority 18/19, 373 LoC) — **GATE on benchmark**

The file today has TWO classes:

- `SubmodularContextSelector` (BM25 + submodular optimisation) — registered as `context_selector`
- `InformationTheoreticSelector` (pointwise mutual information) — registered as `information_theoretic_selector`

**Action:**

1. Run benchmarks with both enabled (baseline from Phase 0).
2. Run benchmarks with only `submodular` enabled, `information_theoretic` disabled.
3. Run benchmarks with both disabled.
4. Compare compression %, quality score, latency p99 across all three.

**Decision rule:**

| Outcome | Action |
|---|---|
| Both enabled wins by >2% on a metric and the gap requires both | Keep both. Split into `context_selector/{__init__,submodular,information_theoretic}.py`. |
| Submodular-only matches both-enabled within ±1% | Delete `InformationTheoreticSelector`, simplify to a single submodular implementation in flat `context_selector.py`. |
| Both-disabled matches both-enabled within ±1% | Delete entire file. Document the cut in MIGRATION.md. |

**Default outcome assumed:** information-theoretic provides no measurable lift (none in existing benchmarks/results/production_evals.json mentions it). Phase 4 simplifies to submodular-only unless the benchmark surprises.

If retained, file shape:

```
src/lattice/transforms/context_selector.py    # ~250 LoC, submodular only
```

If split (both retained):

```
src/lattice/transforms/context_selector/
├── __init__.py                      # SubmodularContextSelector (default) + dispatch
├── submodular.py
└── information_theoretic.py
```

The registry's `information_theoretic_selector` entry is deleted unless the third option is chosen.

### 3.15 batching (priority 3, execution_only, 476 LoC) — **KEEP**

Execution-only transforms run outside `Pipeline.run()`. They live in `pipeline/batch_accumulator.py` (Phase 2 move) for `batching`. The `transforms/batching.py` is the registry-visible spec + adapter.

Decision: keep `transforms/batching.py` as the transform-side entry; it delegates to `pipeline/batch_accumulator.py` for the heavy lifting.

### 3.16 speculative (priority 2, execution_only, 291 LoC) — **KEEP**

Same pattern: execution-only, transform-side entry stays.

### 3.17 delta_encode (priority 5, execution_only, 332 LoC) — **KEEP + FIX**

Audit-discovered bug: `config_flag` in the registry spec says `transform_batching` (copy-paste). Fix to `transform_delta_encode`.

In `transforms/registry.py`:

```python
TransformSpec(
    canonical_name="delta_encoder",
    aliases=("delta_encode",),
    category=TransformCategory.EXECUTION,
    module_path="lattice.transforms.delta_encode",
    class_name="DeltaEncoder",
    config_flag="transform_delta_encode",     # was "transform_batching" — fixed
    safety_bucket=SafetyBucket.SAFE,
    default_pipeline=False,
    execution_only=True,
),
```

In `core/config.py`, ensure `transform_delta_encode: bool = True` exists; if the field is currently named differently, add an alias.

### 3.18 strategy_selector (priority 19, 728 LoC) — **GATE on benchmark; default DELETE**

Bandit-based adaptive strategy selection. Big file, no benchmark evidence, and content_profiler heuristics overlap heavily.

**Action sequence:**

1. **Run benchmarks suite with `transform_strategy_selector=False`** (already the default in registry per audit). Capture `phase-4-no-strategy.json`.
2. Compare to `phase-0-baseline.json` (which had defaults — likely also off).
3. Run with `transform_strategy_selector=True` (force-enabled). Capture `phase-4-with-strategy.json`.
4. Compare strategy-on vs strategy-off.

**Decision rule:**

| Outcome | Action |
|---|---|
| Strategy-on wins by >2% on any of (compression %, quality, latency p99) | Keep. Split `strategy_selector.py` into `strategy_selector/{__init__,bandit}.py`. |
| Strategy-on is ±1% of strategy-off | **DELETE.** The bandit machinery is dead weight; content_profiler's heuristics cover it. |

**Assumed outcome:** strategy_selector adds no measured lift; DELETE.

```bash
git rm src/lattice/transforms/strategy_selector.py
git rm tests/unit/test_strategy_selector.py   # if exists
```

Remove the registry spec:

```python
# DELETE from registry's PIPELINE_SPECS or OPTIMIZER_SPECS:
TransformSpec(canonical_name="strategy_selector", ...)
```

Keep `transform_strategy_selector: bool = False` in `LatticeConfig` for one release as a no-op flag. Document in MIGRATION.md.

If kept, split target:

```
src/lattice/transforms/strategy_selector/
├── __init__.py        # StrategySelector class — registered transform
└── bandit.py          # _ArmState, UCB1 algorithm, reward computation
```

### 3.19 Summary table

| Transform | Today | After Phase 4 |
|---|---|---|
| content_profiler | `content_profiler.py` (985) | `content_profiler/` (5 files, ~600 total) |
| runtime_contract | `runtime_contract.py` (74) | unchanged |
| cache_arbitrage | `cache_arbitrage.py` (545) | unchanged |
| prefix_opt | `prefix_opt.py` (161) | **DELETED** |
| message_dedup | `message_dedup.py` (304) | unchanged (process() removed in P2) |
| reference_sub | `reference_sub.py` (497) | unchanged |
| extractive_compress | `extractive_compress.py` (136) | unchanged |
| rate_distortion | `rate_distortion.py` (216) | unchanged |
| path_prefix | `path_prefix.py` (158) | unchanged |
| diagnostic_rle | `diagnostic_rle.py` (147) | unchanged |
| columnar_pack | `columnar_pack.py` (134) | unchanged |
| json_shape | `json_shape.py` (154) | unchanged |
| format_conv → format_conversion | `format_conv.py` (794) | `format_converter/` (3 files, ~700 total) |
| tool_filter | `tool_filter.py` (232) | unchanged |
| tool_projection | `tool_projection.py` (224) | unchanged |
| output_cleanup | `output_cleanup.py` (160) | unchanged (is_response_side=True) |
| context_selector | `context_selector.py` (373) | **simplified to ~250 LoC** (submodular-only) — OR split if benchmarks demand |
| causal_chain | `causal_chain.py` (192) | unchanged |
| constraint_lifting | `constraint_lifting.py` (173) | **DELETED** (no production consumer) |
| strategy_selector | `strategy_selector.py` (728) | **DELETED** unless benchmark gate keeps it |
| batching | `batching.py` (476) | unchanged |
| speculative | `speculative.py` (291) | unchanged |
| delta_encode | `delta_encode.py` (332) | unchanged + config_flag bug fixed |
| semantic_segmenter | `semantic_segmenter.py` (267) | **MOVED to core/segmentation.py** (Phase 1) |

Net file count: 24 transforms (+ semantic_segmenter not a transform) → 21 transforms + 2 packages (`content_profiler/`, `format_converter/`). LoC drops from ~7,810 to ~5,800 (~25% reduction) primarily from deleting strategy_selector and prefix_opt and slimming context_selector.

---

## 4. Decision gates

### 4.1 Mandatory deletions (no gate)

- `prefix_opt.py` — explicitly deprecated by its own header.
- `constraint_lifting.py` — observability-only, no production consumer.
- Every `process()` method on a transform that also has `optimize()` — Phase 2 already cut these.

### 4.2 Mandatory splits (no gate)

- `content_profiler.py` → `content_profiler/` package.
- `format_conv.py` → `format_converter/` package.

### 4.3 Benchmark-gated deletions

Run **three** benchmark sweeps. Each is a single `benchmarks/evals/cli.py --suite all` invocation against `ollama-cloud / kimi-k2.6:cloud`, `--iterations 1 --warmup 0`. Output JSONs:

| Sweep | Config | Output |
|---|---|---|
| A | All transforms default (matches Phase 0 baseline) | `phase-4-A-baseline.json` |
| B | `transform_strategy_selector=false`, `transform_information_theoretic_selector=false` | `phase-4-B-cut.json` |
| C | `transform_strategy_selector=true`, `transform_information_theoretic_selector=true` (force-enable) | `phase-4-C-forced.json` |

**Decision logic:**

```python
# compression_pct, quality_score, latency_p99_ms compared:
diff_BA = compare(A, B)   # cutting them — does it hurt?
diff_CA = compare(A, C)   # forcing them — does it help?

if max(abs(diff_BA.compression), abs(diff_BA.quality), abs(diff_BA.latency)) < 1.0:
    # Cutting doesn't hurt → DELETE.
    delete(strategy_selector)
    delete(information_theoretic_selector_variant)
elif diff_CA.compression > 2.0 or diff_CA.quality > 2.0:
    # Forcing them helps → KEEP.
    keep_split(strategy_selector)
    keep_split(information_theoretic_variant)
else:
    # Ambiguous → ask reviewer; default cut.
```

Record the decision in `docs/refactor/phase-4-decisions.md`. Commit the benchmark JSONs.

---

## 5. Step-by-step

### 5.1 Move registry and reputation

```bash
git mv src/lattice/core/transform_registry.py   src/lattice/transforms/registry.py
git mv src/lattice/core/transform_reputation.py src/lattice/transforms/reputation.py
```

Update imports:

```bash
sd 'from lattice\.core\.transform_registry import' 'from lattice.transforms.registry import' $(rg -l "from lattice.core.transform_registry import")
sd 'from lattice\.core\.transform_reputation import' 'from lattice.transforms.reputation import' $(rg -l "from lattice.core.transform_reputation import")
```

### 5.2 Move patterns

```bash
git mv src/lattice/utils/patterns.py src/lattice/transforms/patterns.py
sd 'from lattice\.utils\.patterns import' 'from lattice.transforms.patterns import' $(rg -l "from lattice.utils.patterns import")
```

### 5.3 Delete prefix_opt

```bash
git rm src/lattice/transforms/prefix_opt.py
# Remove from registry; remove tests
```

Verify no remaining import:

```bash
rg "prefix_opt|PrefixOpt|prefix_optimizer" src/ tests/ benchmarks/
```

### 5.4 Delete constraint_lifting (after verifying no consumer)

```bash
rg "constraint_lifting|ConstraintLifting" src/ tests/ benchmarks/ docs/
# If matches show only the file + registry + test + docs: delete.
git rm src/lattice/transforms/constraint_lifting.py
git rm tests/unit/test_constraint_lifting.py
# Remove from registry
```

### 5.5 Split content_profiler

```bash
mkdir -p src/lattice/transforms/content_profiler
# move original aside as a working reference (do not commit):
mv src/lattice/transforms/content_profiler.py /tmp/content_profiler.py.bak
```

Create the 5 new files per §3.1, copying logic from the backup. Delete the original:

```bash
# (no git rm needed — the file is no longer in the working tree after the move-aside)
git add src/lattice/transforms/content_profiler/
git status   # confirm transforms/content_profiler.py is staged as deleted
```

Update consumers if any imported from inside the old file. Common targets:

- `from lattice.transforms.content_profiler import ContentProfiler` → unchanged (the new `__init__.py` exports it).
- `from lattice.transforms.content_profiler import ContentProfile` → unchanged (re-exported).
- `from lattice.transforms.content_profiler import compute_risk_score` → unchanged (re-exported).

Verify:

```bash
rg "from lattice.transforms.content_profiler" src/ tests/ benchmarks/
```

All matches should still resolve via the new `__init__.py`.

### 5.6 Split format_conv → format_converter

```bash
mkdir -p src/lattice/transforms/format_converter
mv src/lattice/transforms/format_conv.py /tmp/format_conv.py.bak
```

Create the 3 new files per §3.10. Update consumers:

```bash
sd 'from lattice\.transforms\.format_conv import' 'from lattice.transforms.format_converter import' $(rg -l "from lattice.transforms.format_conv import")
```

Update registry's spec for `format_conversion`:

```python
TransformSpec(
    canonical_name="format_conversion",
    aliases=(),
    category=TransformCategory.PIPELINE,
    module_path="lattice.transforms.format_converter",         # was "lattice.transforms.format_conv"
    class_name="FormatConverter",
    config_flag="transform_format_conversion",
    safety_bucket=SafetyBucket.CONDITIONAL,
    default_pipeline=True,
    execution_only=False,
),
```

### 5.7 Run benchmark gate for strategy_selector + info-theoretic

```bash
# Sweep A (baseline)
uv run python benchmarks/evals/cli.py --suite all \
    --providers ollama-cloud --provider-model ollama-cloud=kimi-k2.6:cloud \
    --iterations 1 --warmup 0 --provider-warmup 0 \
    --output-json benchmarks/results/phase-4-A-baseline.json

# Sweep B (both cut)
LATTICE_TRANSFORM_STRATEGY_SELECTOR=false LATTICE_TRANSFORM_INFORMATION_THEORETIC_SELECTOR=false \
    uv run python benchmarks/evals/cli.py --suite all \
    --providers ollama-cloud --provider-model ollama-cloud=kimi-k2.6:cloud \
    --iterations 1 --warmup 0 --provider-warmup 0 \
    --output-json benchmarks/results/phase-4-B-cut.json

# Sweep C (both forced on)
LATTICE_TRANSFORM_STRATEGY_SELECTOR=true LATTICE_TRANSFORM_INFORMATION_THEORETIC_SELECTOR=true \
    uv run python benchmarks/evals/cli.py --suite all \
    --providers ollama-cloud --provider-model ollama-cloud=kimi-k2.6:cloud \
    --iterations 1 --warmup 0 --provider-warmup 0 \
    --output-json benchmarks/results/phase-4-C-forced.json

# Compare
python scripts/compare_benchmarks.py benchmarks/results/phase-4-A-baseline.json benchmarks/results/phase-4-B-cut.json --tolerance-pct 1
python scripts/compare_benchmarks.py benchmarks/results/phase-4-A-baseline.json benchmarks/results/phase-4-C-forced.json --tolerance-pct 1
```

Apply the decision rule from §4.3. Commit the JSONs and write `docs/refactor/phase-4-decisions.md` with the verdict.

### 5.8 Fix delta_encode config_flag

In `transforms/registry.py` (moved in §5.1):

```bash
sd '"transform_batching"' '"transform_delta_encode"' src/lattice/transforms/registry.py
# Only the delta_encoder spec should change — verify via diff
```

Add `transform_delta_encode` to `LatticeConfig` if missing.

### 5.9 Update is_response_side flag for output_cleanup

In `transforms/registry.py`, add `is_response_side: bool = False` field to `TransformSpec` dataclass; mark `output_cleanup` as `True`. In `pipeline/runner.py`'s response path, dispatch response-side transforms by reading this flag.

### 5.10 Final verification

```bash
uv run ruff check src/ tests/
uv run mypy src/lattice/
uv run pytest tests/ -q
uv run pytest tests/contract/ -q

# Full benchmark vs Phase 0
uv run python benchmarks/evals/cli.py --suite all \
    --providers ollama-cloud --provider-model ollama-cloud=kimi-k2.6:cloud \
    --iterations 1 --warmup 0 --provider-warmup 0 \
    --output-json benchmarks/results/phase-4.json
python scripts/compare_benchmarks.py benchmarks/results/phase-0-baseline.json benchmarks/results/phase-4.json --tolerance-pct 2
```

---

## 6. Tests

### 6.1 Reshape

```bash
mkdir -p tests/unit/transforms/content_profiler tests/unit/transforms/format_converter
git mv tests/unit/test_content_profiler.py tests/unit/transforms/content_profiler/test_profiler_integration.py
git mv tests/unit/test_format_conv.py      tests/unit/transforms/format_converter/test_format_converter.py
# Per-transform tests
git mv tests/unit/test_cache_arbitrage.py  tests/unit/transforms/test_cache_arbitrage.py
git mv tests/unit/test_message_dedup.py    tests/unit/transforms/test_message_dedup.py
git mv tests/unit/test_reference_sub.py    tests/unit/transforms/test_reference_sub.py
# ...etc for each transform that has a dedicated test file
```

### 6.2 Deletions

```bash
git rm tests/unit/test_prefix_opt.py
git rm tests/unit/test_constraint_lifting.py
git rm tests/unit/test_strategy_selector.py   # IF gated cut
git rm tests/unit/test_information_theoretic_selector.py   # IF gated cut
```

### 6.3 New tests

**`tests/unit/transforms/content_profiler/test_classifier.py`** — table-driven test of `classify_by_signals` against fixture requests (code-heavy, narrative, table-heavy, etc.).

**`tests/unit/transforms/content_profiler/test_risk_scorer.py`** — table-driven test of `compute_risk_score` against fixtures.

**`tests/unit/transforms/content_profiler/test_planner_bridge.py`** — given a request and an IR, `build_ir_and_plan` returns a `PlanArtefacts` with non-None plan and cache_plan.

**`tests/unit/transforms/format_converter/test_table_converter.py`** — Markdown table → CSV → Markdown round trip preserves data.

**`tests/unit/transforms/format_converter/test_json_converter.py`** — JSON → YAML → JSON round trip preserves keys and values.

**`tests/unit/transforms/test_registry_complete.py`** — every registry spec's `module_path` and `class_name` resolves at import time:

```python
def test_registry_imports_all():
    from lattice.transforms.registry import list_transform_names, get_transform_spec
    import importlib
    for name in list_transform_names():
        spec = get_transform_spec(name)
        mod = importlib.import_module(spec.module_path)
        assert hasattr(mod, spec.class_name), f"{name}: {spec.module_path}.{spec.class_name} missing"
```

This single test catches every registry / file path / class name drift.

**`tests/unit/transforms/test_no_prefix_opt.py`** — assertion that the deprecated transform is gone:

```python
def test_prefix_opt_deleted():
    import pytest
    with pytest.raises(ImportError):
        from lattice.transforms.prefix_opt import PrefixOptimizer   # noqa: F401
    from lattice.transforms.registry import list_transform_names
    assert "prefix_optimizer" not in list_transform_names()
    assert "prefix_opt" not in list_transform_names()
```

**`tests/unit/transforms/test_delta_encode_config_flag.py`**:

```python
def test_delta_encode_uses_correct_flag():
    from lattice.transforms.registry import get_transform_spec
    spec = get_transform_spec("delta_encoder")
    assert spec.config_flag == "transform_delta_encode"   # not "transform_batching"
```

### 6.4 Contract tests

Update `tests/contract/test_python_api_contract.py`:

- Add `from lattice.transforms.content_profiler import ContentProfiler, ContentProfile, compute_risk_score` — verify package-level re-exports.
- Add `from lattice.transforms.format_converter import FormatConverter, DataShape` — verify package-level re-exports.
- Add `from lattice.transforms.registry import TransformSpec, list_transform_names, get_transform_spec`.
- Add `from lattice.transforms.reputation import TransformReputation, ReputationRegistry`.
- Remove any reference to `prefix_opt`, `constraint_lifting`, `strategy_selector`, `information_theoretic_selector` (gated).

---

## 7. Symbol migration table

| Old fully-qualified | New fully-qualified |
|---|---|
| `lattice.core.transform_registry.TransformSpec` | `lattice.transforms.registry.TransformSpec` |
| `lattice.core.transform_registry.get_transform_spec` | `lattice.transforms.registry.get_transform_spec` |
| `lattice.core.transform_registry.list_transform_names` | `lattice.transforms.registry.list_transform_names` |
| `lattice.core.transform_registry.build_transform_instance` | `lattice.transforms.registry.build_transform_instance` |
| `lattice.core.transform_reputation.TransformReputation` | `lattice.transforms.reputation.TransformReputation` |
| `lattice.core.transform_reputation.ReputationRegistry` | `lattice.transforms.reputation.ReputationRegistry` |
| `lattice.core.transform_reputation.get_reputation_registry` | `lattice.transforms.reputation.get_reputation_registry` |
| `lattice.utils.patterns.*` | `lattice.transforms.patterns.*` |
| `lattice.transforms.content_profiler.ContentProfiler` | `lattice.transforms.content_profiler.ContentProfiler` (package now, same import path) |
| `lattice.transforms.format_conv.FormatConverter` | `lattice.transforms.format_converter.FormatConverter` |
| `lattice.transforms.prefix_opt.PrefixOptimizer` | **DELETED** |
| `lattice.transforms.constraint_lifting.*` | **DELETED** |
| `lattice.transforms.strategy_selector.*` | **DELETED** (if gated cut) |
| `lattice.transforms.context_selector.InformationTheoreticSelector` | **DELETED** (if gated cut) |
| `lattice.transforms.semantic_segmenter.*` | `lattice.core.segmentation.*` (Phase 1) |

---

## 8. Import-rewrite cheatsheet

```bash
# Registry/reputation moves
sd 'from lattice\.core\.transform_registry import' 'from lattice.transforms.registry import' $(rg -l "from lattice.core.transform_registry import")
sd 'from lattice\.core\.transform_reputation import' 'from lattice.transforms.reputation import' $(rg -l "from lattice.core.transform_reputation import")

# Patterns move
sd 'from lattice\.utils\.patterns import' 'from lattice.transforms.patterns import' $(rg -l "from lattice.utils.patterns import")

# format_conv rename
sd 'from lattice\.transforms\.format_conv import' 'from lattice.transforms.format_converter import' $(rg -l "from lattice.transforms.format_conv import")
sd '"lattice\.transforms\.format_conv"' '"lattice.transforms.format_converter"' $(rg -l '"lattice.transforms.format_conv"')

# Deletions — report only
rg "prefix_opt|PrefixOpt|prefix_optimizer" src/ tests/ benchmarks/
rg "constraint_lifting|ConstraintLifting" src/ tests/ benchmarks/
rg "strategy_selector|StrategySelector" src/ tests/ benchmarks/
rg "InformationTheoreticSelector|information_theoretic_selector" src/ tests/ benchmarks/

# delta_encode config_flag
sd '"transform_batching"' '"transform_delta_encode"' src/lattice/transforms/registry.py
```

---

## 9. Acceptance criteria

- [ ] `src/lattice/transforms/registry.py` exists; `core/transform_registry.py` does not.
- [ ] `src/lattice/transforms/reputation.py` exists; `core/transform_reputation.py` does not.
- [ ] `src/lattice/transforms/patterns.py` exists; `utils/patterns.py` does not.
- [ ] `src/lattice/transforms/prefix_opt.py` does not exist.
- [ ] `src/lattice/transforms/constraint_lifting.py` does not exist.
- [ ] `src/lattice/transforms/content_profiler.py` (single file) does not exist; the directory `content_profiler/` does.
- [ ] `src/lattice/transforms/format_conv.py` does not exist; `format_converter/` does.
- [ ] `src/lattice/transforms/strategy_selector.py` is either gone (file removed) or a directory (split). Decision documented in `phase-4-decisions.md`.
- [ ] `src/lattice/transforms/context_selector*` matches the gate decision.
- [ ] `rg "process\(self, request" src/lattice/transforms/` returns matches ONLY in `output_cleanup.py` (response-side) — all other transforms expose only `optimize(...)`.
- [ ] `rg "transform_batching" src/lattice/transforms/registry.py` → 0 matches (only `transform_delta_encode` references the delta_encoder).
- [ ] `from lattice.transforms.content_profiler import ContentProfiler, ContentProfile, compute_risk_score` works.
- [ ] `from lattice.transforms.format_converter import FormatConverter, DataShape` works.
- [ ] `from lattice.transforms.registry import TransformSpec, get_transform_spec, list_transform_names` works.
- [ ] `from lattice.transforms.reputation import ReputationRegistry, get_reputation_registry` works.
- [ ] `tests/unit/transforms/test_registry_complete.py` passes (every spec resolves).
- [ ] `tests/unit/transforms/test_no_prefix_opt.py` passes.
- [ ] `tests/unit/transforms/test_delta_encode_config_flag.py` passes.
- [ ] `uv run ruff check src/ tests/` clean.
- [ ] `uv run mypy src/lattice/` clean.
- [ ] `uv run pytest tests/ -q` passes.
- [ ] `uv run pytest tests/contract/ -q` passes.
- [ ] `python scripts/compare_benchmarks.py benchmarks/results/phase-0-baseline.json benchmarks/results/phase-4.json --tolerance-pct 2` exits 0.
- [ ] `docs/refactor/phase-4-decisions.md` exists and documents the strategy_selector / information_theoretic_selector verdict.

---

## 10. Risks specific to this phase

| Risk | Mitigation |
|---|---|
| Splitting `content_profiler.py` accidentally loses a side effect (it currently writes to `context.metadata` in many places) | The new `__init__.py` `optimize()` explicitly sets each metadata key in the order the original did. `tests/unit/transforms/content_profiler/test_profiler_integration.py` asserts each expected key is present post-run, using snapshots from Phase 0 baseline. |
| Splitting `format_conv.py` breaks the IR-native dispatch — the file mixes IR optimisation with raw text manipulation | Move the IR-native `optimize()` body into `format_converter/__init__.py`; the helpers in `table_converter.py` and `json_converter.py` operate on strings, not IR. |
| The benchmark gate for strategy_selector is run on a non-representative dataset and "passes" a worthless transform | Use `--suite all` (every scenario in the catalog). If any one scenario favours strategy_selector, keep it. The decision rule in §4.3 errs on the side of cutting. |
| Removing `constraint_lifting` breaks a downstream consumer that reads the constraint metadata it sets on context | The audit's "Cross-cutting issues" said it's observability-only. Verify with `rg "constraint_lifting|lifted_constraints" src/ tests/ benchmarks/ docs/` — if any pipeline path consumes it, port the logic into `content_profiler/planner_bridge.py` instead of deleting. |
| `delta_encode` config_flag fix breaks an existing `lattice.yaml` setting `transform_delta_encode: false` that was silently being read as `transform_batching` | This is the bug. Fixing it is the goal. Document in CHANGELOG: "delta_encode now honours its own `transform_delta_encode` flag (was incorrectly using `transform_batching`)". |
| `output_cleanup` runs server-side on `Response` only; the `is_response_side=True` flag must be honoured by `Pipeline.run()` or it silently runs against the request and corrupts data | `pipeline/runner.py` change: in the request path, skip specs with `is_response_side=True`; in the response path, only invoke specs with `is_response_side=True`. New test in `tests/unit/pipeline/test_response_side_dispatch.py`. |
| Moving `transform_registry.py` while still inside `core/__init__.py`'s re-export pattern triggers an import cycle | `core/__init__.py` already does NOT re-export from `transform_registry`; verified in Phase 1 doc. After move, `core/__init__.py` is silent on transforms. |
| `tests/unit/test_cache_arbitrage.py` has `F821` noqa per `pyproject.toml` ruff config | After Phase 4, audit whether the noqa is still needed. If the underlying issue (undefined name) is fixed in the cleanup, remove the noqa. |
| Splitting `content_profiler.py` regresses cold-start (import-time) latency because the new package eagerly imports 4 submodules | The `__init__.py` imports are intentional and small. Cold-start latency is dominated by tiktoken + httpx, not transforms. Verify with `python -X importtime -c 'import lattice'` before/after; difference should be <5 ms. |

---

## 11. Rollback plan

If the phase fails at the benchmark gate (regression >2%):

1. Identify which split caused the regression — most likely `content_profiler/` (most invasive).
2. Revert the split via `git revert <split-commit>`; keep the rest of Phase 4 (deletes, moves, registry fixes).
3. Re-run benchmarks. The split must regress < 2% **on its own**; if it doesn't, the split is wrong (something was lost in the move).
4. Profile to find the missing side effect. Fix. Retry.

If the gated deletions (strategy_selector, information_theoretic) turn out to matter post-merge:

1. Resurrect from git history: `git show <pre-phase-4>:src/lattice/transforms/strategy_selector.py > src/lattice/transforms/strategy_selector.py`.
2. Re-register in `transforms/registry.py`.
3. Document the resurrection in CHANGELOG.

---

## 12. PR shape

This phase is best as **3 sequential PRs** because each is independently reviewable:

```
refactor(transforms): move registry, reputation, patterns; fix delta_encode flag [Phase 4a]
- core/transform_registry.py → transforms/registry.py
- core/transform_reputation.py → transforms/reputation.py
- utils/patterns.py → transforms/patterns.py
- Fix delta_encode config_flag bug
- Add is_response_side flag to TransformSpec; mark output_cleanup

refactor(transforms): split content_profiler and format_conv; delete prefix_opt and constraint_lifting [Phase 4b]
- content_profiler.py (985 LoC) → content_profiler/ (5 files)
- format_conv.py (794 LoC) → format_converter/ (3 files)
- DELETE prefix_opt.py (deprecated)
- DELETE constraint_lifting.py (no production consumer)

refactor(transforms): benchmark-gate strategy_selector and information_theoretic_selector [Phase 4c]
- Run benchmarks A/B/C sweep
- Apply decision rule from §4.3
- DELETE or SPLIT per outcome
- Commit phase-4-decisions.md with verdict
```

Each can be reviewed and merged in turn.
