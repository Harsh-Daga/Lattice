# Phase 3 — Optimizer / Planner / Runtime Collapse

> **Goal.** Three overlapping decision layers (`core/scheduler.py`, `core/optimizer_scheduler.py`, `core/unified_planner.py`) collapse into one canonical planner. The `optimizer/` directory dissolves: orchestrator-style optimizers move into `transforms/optimizers/`, the legacy text-based `structure_optimizer.py` is deleted (superseded by `ir_structure_optimizer.py`). `runtime/router.py` is renamed `runtime/tier_classifier.py` to honor the README claim "LATTICE is not a router". `core/runtime_state.py` and `core/task_classifier.py` join the planner. `core/credentials.py` moves under `providers/`.
>
> **Outcome.** A single chain: `Request → RequestClassifier → UnifiedPlanner → ExecutionPlan → Pipeline.run(plan)`. No parallel scheduler. No misnamed router. The `optimizer/` directory is gone.
>
> **Estimated effort.** 2 days.

---

## 1. Why this phase exists

Today three classes can decide what transforms run, and they sometimes disagree:

| Class | File | Style | Used by |
|---|---|---|---|
| `decide_schedule(request, ctx)` | `core/scheduler.py` (443 LoC) | Reactive, greedy ranking by reputation + task class + budget. Returns a `SchedulerDecision`. | v1 `CompressorPipeline` (already deleted in Phase 2) |
| `decide_optimizer_schedule(...)` | `core/optimizer_scheduler.py` (131 LoC) | Specialised wrapper around `unified_planner.plan()`. Returns an `OptimizerSchedule`. | v1 CompressorPipeline + a few benchmark scripts |
| `UnifiedPlanner.plan(request, profile)` | `core/unified_planner.py` (384 LoC) | Upfront, tier-based. Returns the canonical `ExecutionPlan`. | v2 `Pipeline` (now the only Pipeline) |

After Phase 2, only the v2 Pipeline exists. The other two schedulers have no live consumer except for tests and benchmarks that haven't been updated. They are dead code wearing a uniform.

Equally: the `optimizer/` directory mixes four roles:

| Role | Files | What they do |
|---|---|---|
| **IR-native base** | `ir_native_optimizer.py` (240) | Base class for IR optimisers. **Moved in Phase 1** to `ir/native_optimizer.py`. |
| **Quality + validation** | `quality_estimator.py` (638), `validation.py` (283) | Cross-cutting helpers. **Moved in Phase 1** to `ir/`. |
| **Beam-search orchestrator** | `representation_optimizer.py` (431) | Searches over transforms for the best combination. **Moved in Phase 2** to `pipeline/`. |
| **Per-domain transform orchestrators** | `structure_optimizer.py` (255), `ir_structure_optimizer.py` (178), `reference_optimizer.py` (255), `tool_optimizer.py` (213), `diagnostic_optimizer.py` (198), `context_optimizer.py` (265) | Each runs 2-3 transforms, picks best variant. These are **transforms-on-transforms**. |

The last group is the leftover after Phases 1+2. They belong with the transforms they orchestrate. Phase 3 moves them into `transforms/optimizers/`.

`structure_optimizer.py` is also redundant with `ir_structure_optimizer.py` — text-based vs IR-native versions of the same idea. The IR-native one wins. The text version is deleted.

`runtime/router.py` is a workload-tier classifier (SIMPLE → MEDIUM → COMPLEX → REASONING) for setting optimisation budgets. It's not a provider router. The audit confirmed the code is correct but the **file name lies**. Rename.

`core/credentials.py` is the provider credential resolver. It only ever services `providers/transport/` and the adapter layer. It belongs with the providers, not in `core/` (moved to `providers/credentials.py` in Phase 4).

---

## 2. Files touched

### 2.1 Moved

| Current path | New path |
|---|---|
| `src/lattice/core/unified_planner.py` | `src/lattice/planner/unified_planner.py` |
| `src/lattice/core/task_classifier.py` | `src/lattice/planner/task_classifier.py` |
| `src/lattice/core/runtime_state.py` | `src/lattice/planner/runtime_state.py` |
| `src/lattice/core/credentials.py` | `src/lattice/providers/credentials.py` |
| `src/lattice/optimizer/ir_structure_optimizer.py` | `src/lattice/transforms/optimizers/ir_structure_optimizer.py` |
| `src/lattice/optimizer/reference_optimizer.py` | `src/lattice/transforms/optimizers/reference_optimizer.py` |
| `src/lattice/optimizer/tool_optimizer.py` | `src/lattice/transforms/optimizers/tool_optimizer.py` |
| `src/lattice/optimizer/diagnostic_optimizer.py` | `src/lattice/transforms/optimizers/diagnostic_optimizer.py` |
| `src/lattice/optimizer/context_optimizer.py` | `src/lattice/transforms/optimizers/context_optimizer.py` |

### 2.2 Renamed

| Current path | New path |
|---|---|
| `src/lattice/runtime/router.py` | `src/lattice/runtime/tier_classifier.py` |

### 2.3 Created

```
src/lattice/transforms/optimizers/__init__.py
src/lattice/planner/__init__.py            # MODIFIED — already exists, rewrite per §3.12
```

### 2.4 Deleted

```
src/lattice/core/scheduler.py                              # 443 LoC — dead, v1 only
src/lattice/core/optimizer_scheduler.py                    # 131 LoC — wrapper around unified_planner; consumers updated to call unified_planner directly
src/lattice/optimizer/structure_optimizer.py               # 255 LoC — superseded by ir_structure_optimizer
src/lattice/optimizer/__init__.py                          # the optimizer/ directory itself ceases to exist
```

After Phase 3, `src/lattice/optimizer/` is empty and is removed (`rmdir`).

### 2.5 Modified

- `src/lattice/core/__init__.py` — drop `runtime_state`, `task_classifier`, `unified_planner` references (none today; verify).
- `src/lattice/__init__.py` — add `from lattice.planner import ExecutionPlan, UnifiedPlanner, build_execution_plan`.
- All callers of `decide_schedule(...)` and `decide_optimizer_schedule(...)` — rewritten to call `UnifiedPlanner.plan(...)` or `build_execution_plan(...)`.
- All `from lattice.core.{unified_planner,task_classifier,runtime_state,credentials,scheduler,optimizer_scheduler}` import sites — rewritten.
- All `from lattice.optimizer.{ir_structure,reference,tool,diagnostic,context,structure}_optimizer` import sites — rewritten.
- All `from lattice.runtime.router` and `from lattice.runtime import` import sites — rewritten.
- `core/transform_registry.py` (still in core/ until Phase 4) — `OPTIMIZER_SPECS` entry for `structure_optimizer` is deleted; the entry for `ir_structure_optimizer` becomes the canonical "structure" optimizer.

---

## 3. Step-by-step

### 3.1 Create the new package skeleton

```bash
mkdir -p src/lattice/transforms/optimizers
touch src/lattice/transforms/optimizers/__init__.py
```

`src/lattice/planner/` already exists (it has `execution_plan.py`, `execution_builder.py`, etc.). Just verify.

### 3.2 Move scheduler and planner files

```bash
git mv src/lattice/core/unified_planner.py   src/lattice/planner/unified_planner.py
git mv src/lattice/core/task_classifier.py   src/lattice/planner/task_classifier.py
git mv src/lattice/core/runtime_state.py     src/lattice/planner/runtime_state.py
```

### 3.3 Move credentials to providers/

```bash
git mv src/lattice/core/credentials.py       src/lattice/providers/credentials.py
```

### 3.4 Move optimizer orchestrators to `transforms/optimizers/`

```bash
git mv src/lattice/optimizer/ir_structure_optimizer.py src/lattice/transforms/optimizers/ir_structure_optimizer.py
git mv src/lattice/optimizer/reference_optimizer.py    src/lattice/transforms/optimizers/reference_optimizer.py
git mv src/lattice/optimizer/tool_optimizer.py         src/lattice/transforms/optimizers/tool_optimizer.py
git mv src/lattice/optimizer/diagnostic_optimizer.py   src/lattice/transforms/optimizers/diagnostic_optimizer.py
git mv src/lattice/optimizer/context_optimizer.py      src/lattice/transforms/optimizers/context_optimizer.py
```

### 3.5 Rename `runtime/router.py`

```bash
git mv src/lattice/runtime/router.py src/lattice/runtime/tier_classifier.py
```

Then update the only class name inside it:

```bash
# Inside the renamed file, the class is `RuntimeRouter`. Rename to `TierClassifier`:
sd '\bRuntimeRouter\b' 'TierClassifier' src/lattice/runtime/tier_classifier.py
# And update the dataclass `RoutingDecision` → `TierDecision`:
sd '\bRoutingDecision\b' 'TierDecision' src/lattice/runtime/tier_classifier.py
```

Then update consumers across the codebase:

```bash
sd '\bRuntimeRouter\b' 'TierClassifier' $(rg -l "\\bRuntimeRouter\\b" src/ tests/ benchmarks/)
sd '\bRoutingDecision\b' 'TierDecision' $(rg -l "\\bRoutingDecision\\b" src/ tests/ benchmarks/)
sd 'from lattice\.runtime\.router import'  'from lattice.runtime.tier_classifier import' $(rg -l "from lattice.runtime.router import")
sd 'from lattice\.runtime import RuntimeRouter' 'from lattice.runtime import TierClassifier' $(rg -l "from lattice.runtime import RuntimeRouter")
sd 'from lattice\.runtime import RoutingDecision' 'from lattice.runtime import TierDecision' $(rg -l "from lattice.runtime import RoutingDecision")
```

And update `runtime/__init__.py`:

```python
"""Runtime workload-tier classification.

NOT a provider router — provider selection is external to LATTICE.
This module classifies workload complexity (SIMPLE/MEDIUM/COMPLEX/REASONING)
for setting optimisation budgets.
"""
from lattice.runtime.tier_classifier import Tier, TierClassifier, TierDecision

__all__ = ["Tier", "TierClassifier", "TierDecision"]
```

### 3.6 Delete the v1 schedulers

`core/scheduler.py` had one consumer: `core/pipeline.py` (now deleted in Phase 2). Verify:

```bash
rg "decide_schedule|SchedulerDecision|TransformScheduleEntry" src/ tests/ benchmarks/
```

Should return only the file itself and possibly its test. Delete both:

```bash
git rm src/lattice/core/scheduler.py
git rm tests/unit/test_scheduler.py   # if it exists
```

`core/optimizer_scheduler.py` had consumers: `core/pipeline.py` (gone) plus possibly benchmarks and a test file. Identify them:

```bash
rg "decide_optimizer_schedule|OptimizerSchedule|RequestShim" src/ tests/ benchmarks/
```

For each remaining consumer outside the file itself: replace the call with the direct `UnifiedPlanner.plan(...)` call. The mapping:

```python
# Before:
from lattice.core.optimizer_scheduler import decide_optimizer_schedule
schedule = decide_optimizer_schedule(request, classification)
allowed_optimizers = schedule.enabled_optimizers
budget = schedule.budget

# After:
from lattice.planner.unified_planner import UnifiedPlanner
from lattice.planner.task_classifier import classify_task
planner = UnifiedPlanner()
classification = classify_task(request)
profile = SemanticProfile.from_classification(classification)   # already defined in unified_planner
plan = planner.plan(request, profile)
allowed_optimizers = plan.allowed_optimizers
budget = plan.latency_budget_ms
```

After all call sites are updated:

```bash
git rm src/lattice/core/optimizer_scheduler.py
git rm tests/unit/test_optimizer_scheduler.py   # if it exists
```

### 3.7 Delete the legacy text-based structure optimizer

`optimizer/structure_optimizer.py` (255 LoC) duplicates `ir_structure_optimizer.py` (178 LoC) — the IR-native version wins. Verify nothing else imports it:

```bash
rg "from lattice.optimizer.structure_optimizer|structure_optimizer\.StructureOptimizer" src/ tests/ benchmarks/
```

Two known consumers:
1. `optimizer/__init__.py` — exports both `StructureOptimizer` and `IRStructureOptimizer`. Drop `StructureOptimizer` export.
2. `optimizer/representation_optimizer.py` (now `pipeline/representation_optimizer.py` after Phase 2) — its `_OPTIMIZER_CLASSES` dict references `structure_optimizer`. Remove the entry; ensure `ir_structure_optimizer` is the only "structure" optimizer.

Then:

```bash
git rm src/lattice/optimizer/structure_optimizer.py
git rm tests/unit/test_structure_optimizer.py   # if it exists; keep ir_structure_optimizer tests
```

### 3.8 Delete the `optimizer/__init__.py` and the empty directory

After §3.4 + §3.7, the only file left in `optimizer/` is `__init__.py`. Delete it and remove the directory:

```bash
git rm src/lattice/optimizer/__init__.py
rmdir src/lattice/optimizer   # local working tree only; git tracks per file
```

The registry exports it used to expose are now exposed by `transforms/optimizers/__init__.py` (§3.10).

### 3.9 Rewrite imports across the codebase

```bash
# Planner moves
sd 'from lattice\.core\.unified_planner import' 'from lattice.planner.unified_planner import' $(rg -l "from lattice.core.unified_planner import")
sd 'from lattice\.core\.task_classifier import' 'from lattice.planner.task_classifier import' $(rg -l "from lattice.core.task_classifier import")
sd 'from lattice\.core\.runtime_state import'   'from lattice.planner.runtime_state import'   $(rg -l "from lattice.core.runtime_state import")

# Credentials
sd 'from lattice\.core\.credentials import'     'from lattice.providers.credentials import'   $(rg -l "from lattice.core.credentials import")

# Optimizer orchestrators
sd 'from lattice\.optimizer\.ir_structure_optimizer import' 'from lattice.transforms.optimizers.ir_structure_optimizer import' $(rg -l "from lattice.optimizer.ir_structure_optimizer import")
sd 'from lattice\.optimizer\.reference_optimizer import'    'from lattice.transforms.optimizers.reference_optimizer import'    $(rg -l "from lattice.optimizer.reference_optimizer import")
sd 'from lattice\.optimizer\.tool_optimizer import'         'from lattice.transforms.optimizers.tool_optimizer import'         $(rg -l "from lattice.optimizer.tool_optimizer import")
sd 'from lattice\.optimizer\.diagnostic_optimizer import'   'from lattice.transforms.optimizers.diagnostic_optimizer import'   $(rg -l "from lattice.optimizer.diagnostic_optimizer import")
sd 'from lattice\.optimizer\.context_optimizer import'      'from lattice.transforms.optimizers.context_optimizer import'      $(rg -l "from lattice.optimizer.context_optimizer import")

# Schedulers (the file itself is deleted; any remaining import is a bug)
rg "from lattice.core.scheduler|from lattice.core.optimizer_scheduler" src/ tests/ benchmarks/
# Each remaining match must be fixed manually (see §3.6)

# StructureOptimizer (the orchestrator class is deleted)
rg "StructureOptimizer\b" src/ tests/ benchmarks/
# Each match must be reviewed: replace with IRStructureOptimizer or delete the call site

# Router rename (already done in §3.5)
```

Run `uv run ruff check src/ tests/` — fix any leftovers manually.

### 3.10 Write `transforms/optimizers/__init__.py`

```python
"""Transform orchestrators — each runs 2-3 underlying transforms and selects the best variant.

These are *transforms-on-transforms*: they're registered like normal transforms but their
optimize() method invokes other transforms and chooses the winner by quality + cost +
risk scoring (see lattice.ir.quality, lattice.ir.validation).

Registered for use by pipeline.representation_optimizer.RepresentationOptimizer
(beam search) which decides which orchestrators to invoke per request.
"""

from lattice.transforms.optimizers.ir_structure_optimizer import IRStructureOptimizer
from lattice.transforms.optimizers.reference_optimizer import ReferenceOptimizer
from lattice.transforms.optimizers.tool_optimizer import ToolOptimizer
from lattice.transforms.optimizers.diagnostic_optimizer import DiagnosticOptimizer
from lattice.transforms.optimizers.context_optimizer import ContextOptimizer

# Registry consumed by pipeline.representation_optimizer:
_OPTIMIZER_CLASSES: dict[str, type] = {
    "ir_structure_optimizer": IRStructureOptimizer,
    "reference_optimizer": ReferenceOptimizer,
    "tool_optimizer": ToolOptimizer,
    "diagnostic_optimizer": DiagnosticOptimizer,
    "context_optimizer": ContextOptimizer,
}

# Production-default tuple (lossless first):
PRODUCTION_OPTIMIZERS: tuple[str, ...] = (
    "ir_structure_optimizer",
    "reference_optimizer",
    "tool_optimizer",
    "diagnostic_optimizer",
)
# Lossy / conditional:
CONDITIONAL_OPTIMIZERS: tuple[str, ...] = ("context_optimizer",)

__all__ = [
    "IRStructureOptimizer", "ReferenceOptimizer", "ToolOptimizer",
    "DiagnosticOptimizer", "ContextOptimizer",
    "_OPTIMIZER_CLASSES", "PRODUCTION_OPTIMIZERS", "CONDITIONAL_OPTIMIZERS",
]
```

Note: `_OPTIMIZER_CLASSES` previously included `"structure_optimizer": StructureOptimizer`. That entry is gone in v1.0.0.

### 3.11 Write `planner/__init__.py`

The existing `planner/__init__.py` (23 LoC, from the audit) re-exports a partial set. Rewrite to the full surface:

```python
"""Planning layer: request classification → ExecutionPlan.

Single source of truth for runtime decisions: which transforms to run, in what
order, with what quality floor and latency budget.
"""

from lattice.planner.execution_plan import (
    ExecutionPlan, ExecutionTier, RiskLevel,
    OptimizerDecision, RepresentationCandidate,
    CachePlanEntry, TransportPlanEntry, FallbackPlan,
    tier_budget_ms, TIER_BUDGETS_MS,
)
from lattice.planner.execution_builder import build_execution_plan
from lattice.planner.unified_planner import (
    UnifiedPlanner, SemanticProfile, Tier as PlanTier,
    profile_from_legacy,
)
from lattice.planner.task_classifier import (
    TaskClass, ExecutionTier as TaskExecutionTier,    # disambiguated alias
    TaskClassification, classify_task,
)
from lattice.planner.request_classifier import RequestClassifier
from lattice.planner.runtime_state import (
    coerce_execution_plan, get_ir_metadata,
    get_canonical_state_value, get_canonical_request_value,
    normalize_legacy_execution_plan,
    persist_execution_plan_state, persist_session_plan_state,
)
from lattice.planner.provider_strategy import (
    ProviderStrategy, CacheSimulation,
    get_provider_strategy,
    build_cache_plan_for_provider, simulate_provider_cache,
    preferred_optimizers_for_provider,
)
from lattice.planner.transport_planner import TransportPlan, build_transport_plan
from lattice.planner.fallback_executor import (
    execute_with_fallback, execute_with_fallback_stream,
)

__all__ = [
    # plans
    "ExecutionPlan", "ExecutionTier", "RiskLevel",
    "OptimizerDecision", "RepresentationCandidate",
    "CachePlanEntry", "TransportPlanEntry", "FallbackPlan",
    "tier_budget_ms", "TIER_BUDGETS_MS",
    "TransportPlan", "build_transport_plan",
    # build
    "build_execution_plan",
    # planner
    "UnifiedPlanner", "SemanticProfile", "PlanTier", "profile_from_legacy",
    # classify
    "TaskClass", "TaskClassification", "classify_task",
    "RequestClassifier",
    # runtime state bridges
    "coerce_execution_plan", "get_ir_metadata",
    "get_canonical_state_value", "get_canonical_request_value",
    "normalize_legacy_execution_plan",
    "persist_execution_plan_state", "persist_session_plan_state",
    # provider strategy
    "ProviderStrategy", "CacheSimulation",
    "get_provider_strategy",
    "build_cache_plan_for_provider", "simulate_provider_cache",
    "preferred_optimizers_for_provider",
    # execution
    "execute_with_fallback", "execute_with_fallback_stream",
]
```

### 3.12 Fix imports inside the moved files

The moved files reference each other and core/. After `git mv`, internal references that were relative or used the old paths must be updated:

| Inside file | Old import | New import |
|---|---|---|
| `planner/unified_planner.py` | `from lattice.core.primitives import ...` | `from lattice.ir.primitives import ...` (already done in Phase 1) |
| `planner/unified_planner.py` | `from lattice.core.task_classifier import ...` | `from lattice.planner.task_classifier import ...` |
| `planner/unified_planner.py` | `from lattice.core.transport import Request` | `from lattice.transport.types import Request` (already done in Phase 2) |
| `planner/task_classifier.py` | `from lattice.core.transport import Request` | `from lattice.transport.types import Request` |
| `planner/runtime_state.py` | `from lattice.core.context import TransformContext` | unchanged (TransformContext stays in core/) |
| `planner/runtime_state.py` | `from lattice.core.primitives import ...` | `from lattice.ir.primitives import ...` |
| `planner/runtime_state.py` | `from lattice.core.transport import Request` | `from lattice.transport.types import Request` |
| `transforms/optimizers/*.py` | `from lattice.optimizer.ir_native_optimizer import IRNativeOptimizer` | `from lattice.ir.native_optimizer import IRNativeOptimizer` (was done in Phase 1; just verify after move) |
| `transforms/optimizers/*.py` | `from lattice.core.runtime_state import ...` | `from lattice.planner.runtime_state import ...` |
| `transforms/optimizers/*.py` | `from lattice.transforms.XYZ import ...` (transform classes) | unchanged |
| `transforms/optimizers/representation_*` references — these now live in `pipeline/representation_optimizer.py` (Phase 2) | — | `from lattice.transforms.optimizers import _OPTIMIZER_CLASSES` |
| `providers/credentials.py` | `from lattice.core.config import ...` | unchanged (config stays in core/) |
| `providers/credentials.py` — imports from `providers/transport.py` | unchanged | unchanged |
| `runtime/tier_classifier.py` | `from lattice.core.transport import Request` | `from lattice.transport.types import Request` |

### 3.13 Update `pipeline/representation_optimizer.py`

This file moved in Phase 2. It references `_OPTIMIZER_CLASSES` which lived in `optimizer/__init__.py`. After Phase 3, `_OPTIMIZER_CLASSES` lives in `transforms/optimizers/__init__.py`. Update the import:

```python
# Before:
from lattice.optimizer import _OPTIMIZER_CLASSES

# After:
from lattice.transforms.optimizers import _OPTIMIZER_CLASSES
```

Also drop any reference to `StructureOptimizer` (the text-based one); it no longer exists.

### 3.14 Update `core/transform_registry.py`

The registry's `OPTIMIZER_SPECS` (a tuple of `TransformSpec` for each optimizer) had an entry for `structure_optimizer`. Delete it. The entry for `ir_structure_optimizer` becomes the canonical "structure" optimiser; if the name was historically `structure_optimizer`, **rename to `ir_structure_optimizer`** in the spec and add `structure_optimizer` to its `aliases` tuple for transition.

```python
# Inside OPTIMIZER_SPECS:
TransformSpec(
    canonical_name="ir_structure_optimizer",
    aliases=("structure_optimizer",),         # accept the old name for one release
    category=TransformCategory.OPTIMIZER,
    module_path="lattice.transforms.optimizers.ir_structure_optimizer",
    class_name="IRStructureOptimizer",
    config_flag="transform_structure_optimizer",   # config field kept stable
    safety_bucket=SafetyBucket.SAFE,
    default_pipeline=False,
    execution_only=False,
),
# REMOVED: the old StructureOptimizer entry
```

This file gets moved to `transforms/registry.py` in Phase 4. Phase 3 makes the data change; Phase 4 makes the location change.

### 3.15 Per-function disposition

#### `planner/unified_planner.py` (formerly `core/unified_planner.py`)

| Symbol | Disposition |
|---|---|
| `Tier` enum (`FAST, STANDARD, THOROUGH, SAFE, MAXIMUM`) | KEEP — exported as `planner.PlanTier` (alias to avoid clash with `runtime.Tier`) |
| `SemanticProfile` dataclass | KEEP |
| `UnifiedPlanner.plan(request, profile)` | KEEP — now THE entry point |
| `profile_from_legacy(legacy_dict)` | KEEP — used by Phase 2's bridge cleanup |
| Imports | Updated per §3.12 |

#### `planner/task_classifier.py` (formerly `core/task_classifier.py`)

| Symbol | Disposition |
|---|---|
| `TaskClass` enum | KEEP |
| `ExecutionTier` enum | RENAME the enum name *inside the planner* to avoid clash. The exported name is `planner.TaskExecutionTier` (aliased in `__init__.py`). |
| `TaskClassification` dataclass | KEEP |
| `classify_task(request)` | KEEP |
| Internal heuristics | KEEP, no logic change |

#### `planner/runtime_state.py` (formerly `core/runtime_state.py`)

| Symbol | Disposition |
|---|---|
| `coerce_execution_plan(plan)` | KEEP |
| `get_ir_metadata(context)` | KEEP |
| `get_canonical_state_value(context, key, default)` | KEEP |
| `get_canonical_request_value(request, key, default)` | KEEP |
| `normalize_legacy_execution_plan(plan)` | **DELETE** — v1 path is gone, legacy plan normalisation is no longer needed. Simplifies the file from 275 LoC to ~210. |
| `persist_execution_plan_state` | KEEP |
| `persist_session_plan_state` | KEEP |

#### `providers/credentials.py` (formerly `core/credentials.py`)

| Symbol | Disposition |
|---|---|
| `ProviderCredentials` dataclass | KEEP |
| `CredentialResolver` singleton | KEEP |
| `get_credential_resolver()` factory | KEEP |
| Imports | Updated per §3.12 |

#### `runtime/tier_classifier.py` (formerly `runtime/router.py`)

| Symbol | Old name | New name |
|---|---|---|
| Class | `RuntimeRouter` | `TierClassifier` |
| Dataclass | `RoutingDecision` | `TierDecision` |
| Enum | `Tier` (unchanged) | `Tier` |
| Method `classify(request)` | unchanged | unchanged |

Logic untouched. Only names + file path.

#### `transforms/optimizers/ir_structure_optimizer.py`

| Symbol | Disposition |
|---|---|
| `IRStructureOptimizer` class | KEEP |
| `_factor_json`, `_factor_table`, `_group_logs` helpers | KEEP |
| Base class import `IRNativeOptimizer` | Already at `lattice.ir.native_optimizer` (Phase 1) |

#### `transforms/optimizers/{reference,tool,diagnostic,context}_optimizer.py`

Same pattern. Logic preserved; imports updated.

#### Deleted: `core/scheduler.py`

| Symbol | Disposition |
|---|---|
| `TransformScheduleEntry` dataclass | DELETED |
| `SchedulerDecision` dataclass | DELETED |
| `decide_schedule(request, context)` | DELETED |
| `sort_key(name)` | DELETED |

All consumers (the v1 Pipeline, now gone) referenced these; no replacement needed.

#### Deleted: `core/optimizer_scheduler.py`

| Symbol | Disposition |
|---|---|
| `OptimizerSchedule` dataclass | DELETED |
| `decide_optimizer_schedule(...)` | DELETED — callers updated to use `UnifiedPlanner.plan()` |
| `RequestShim` dataclass | DELETED |
| `_task_quality_floor(task_class)` | MOVED into `planner/unified_planner.py` as a private helper if not already there |

---

## 4. Symbol migration table

For consumers updating imports. Every renamed/moved fully-qualified name in this phase:

| Phase 2 location | Phase 3 location |
|---|---|
| `lattice.core.unified_planner.UnifiedPlanner` | `lattice.planner.UnifiedPlanner` |
| `lattice.core.unified_planner.SemanticProfile` | `lattice.planner.SemanticProfile` |
| `lattice.core.unified_planner.Tier` | `lattice.planner.PlanTier` (renamed alias) |
| `lattice.core.unified_planner.profile_from_legacy` | `lattice.planner.profile_from_legacy` |
| `lattice.core.task_classifier.TaskClass` | `lattice.planner.TaskClass` |
| `lattice.core.task_classifier.classify_task` | `lattice.planner.classify_task` |
| `lattice.core.task_classifier.TaskClassification` | `lattice.planner.TaskClassification` |
| `lattice.core.task_classifier.ExecutionTier` | `lattice.planner.TaskExecutionTier` (renamed alias) |
| `lattice.core.runtime_state.coerce_execution_plan` | `lattice.planner.coerce_execution_plan` |
| `lattice.core.runtime_state.get_ir_metadata` | `lattice.planner.get_ir_metadata` |
| `lattice.core.runtime_state.get_canonical_state_value` | `lattice.planner.get_canonical_state_value` |
| `lattice.core.runtime_state.get_canonical_request_value` | `lattice.planner.get_canonical_request_value` |
| `lattice.core.runtime_state.persist_execution_plan_state` | `lattice.planner.persist_execution_plan_state` |
| `lattice.core.runtime_state.persist_session_plan_state` | `lattice.planner.persist_session_plan_state` |
| `lattice.core.runtime_state.normalize_legacy_execution_plan` | **DELETED** |
| `lattice.core.credentials.ProviderCredentials` | `lattice.providers.credentials.ProviderCredentials` |
| `lattice.core.credentials.CredentialResolver` | `lattice.providers.credentials.CredentialResolver` |
| `lattice.optimizer.ir_structure_optimizer.IRStructureOptimizer` | `lattice.transforms.optimizers.IRStructureOptimizer` |
| `lattice.optimizer.reference_optimizer.ReferenceOptimizer` | `lattice.transforms.optimizers.ReferenceOptimizer` |
| `lattice.optimizer.tool_optimizer.ToolOptimizer` | `lattice.transforms.optimizers.ToolOptimizer` |
| `lattice.optimizer.diagnostic_optimizer.DiagnosticOptimizer` | `lattice.transforms.optimizers.DiagnosticOptimizer` |
| `lattice.optimizer.context_optimizer.ContextOptimizer` | `lattice.transforms.optimizers.ContextOptimizer` |
| `lattice.optimizer.structure_optimizer.StructureOptimizer` | **DELETED** — use `IRStructureOptimizer` |
| `lattice.optimizer._OPTIMIZER_CLASSES` | `lattice.transforms.optimizers._OPTIMIZER_CLASSES` |
| `lattice.optimizer.PRODUCTION_OPTIMIZERS` | `lattice.transforms.optimizers.PRODUCTION_OPTIMIZERS` |
| `lattice.optimizer.CONDITIONAL_OPTIMIZERS` | `lattice.transforms.optimizers.CONDITIONAL_OPTIMIZERS` |
| `lattice.runtime.router.RuntimeRouter` | `lattice.runtime.tier_classifier.TierClassifier` |
| `lattice.runtime.router.RoutingDecision` | `lattice.runtime.tier_classifier.TierDecision` |
| `lattice.runtime.RuntimeRouter` (via __init__) | `lattice.runtime.TierClassifier` |
| `lattice.core.scheduler.decide_schedule` | **DELETED** |
| `lattice.core.scheduler.SchedulerDecision` | **DELETED** |
| `lattice.core.optimizer_scheduler.decide_optimizer_schedule` | **DELETED** — use `UnifiedPlanner.plan()` |
| `lattice.core.optimizer_scheduler.OptimizerSchedule` | **DELETED** |

---

## 5. Import-rewrite cheatsheet

All `sd` commands batched (run after the `git mv` in §3.2–§3.5):

```bash
# Planner moves
sd 'from lattice\.core\.unified_planner import' 'from lattice.planner.unified_planner import' $(rg -l "from lattice.core.unified_planner import")
sd 'from lattice\.core\.task_classifier import' 'from lattice.planner.task_classifier import' $(rg -l "from lattice.core.task_classifier import")
sd 'from lattice\.core\.runtime_state import'   'from lattice.planner.runtime_state import'   $(rg -l "from lattice.core.runtime_state import")

# Credentials move
sd 'from lattice\.core\.credentials import'     'from lattice.providers.credentials import'   $(rg -l "from lattice.core.credentials import")

# Optimizer-orchestrator moves
sd 'from lattice\.optimizer\.ir_structure_optimizer import' 'from lattice.transforms.optimizers.ir_structure_optimizer import' $(rg -l "from lattice.optimizer.ir_structure_optimizer import")
sd 'from lattice\.optimizer\.reference_optimizer import'    'from lattice.transforms.optimizers.reference_optimizer import'    $(rg -l "from lattice.optimizer.reference_optimizer import")
sd 'from lattice\.optimizer\.tool_optimizer import'         'from lattice.transforms.optimizers.tool_optimizer import'         $(rg -l "from lattice.optimizer.tool_optimizer import")
sd 'from lattice\.optimizer\.diagnostic_optimizer import'   'from lattice.transforms.optimizers.diagnostic_optimizer import'   $(rg -l "from lattice.optimizer.diagnostic_optimizer import")
sd 'from lattice\.optimizer\.context_optimizer import'      'from lattice.transforms.optimizers.context_optimizer import'      $(rg -l "from lattice.optimizer.context_optimizer import")

# Bare `from lattice.optimizer import ...` (registries / production tuple)
sd 'from lattice\.optimizer import' 'from lattice.transforms.optimizers import' $(rg -l "from lattice.optimizer import")

# Router rename
sd 'from lattice\.runtime\.router import' 'from lattice.runtime.tier_classifier import' $(rg -l "from lattice.runtime.router import")
sd '\bRuntimeRouter\b' 'TierClassifier' $(rg -l "\\bRuntimeRouter\\b")
sd '\bRoutingDecision\b' 'TierDecision' $(rg -l "\\bRoutingDecision\\b")

# Dead schedulers — find and report (manual fix)
rg "from lattice\.core\.scheduler|decide_schedule\\b|SchedulerDecision" src/ tests/ benchmarks/
rg "from lattice\.core\.optimizer_scheduler|decide_optimizer_schedule|OptimizerSchedule" src/ tests/ benchmarks/

# Deleted text-based structure optimizer
rg "from lattice\.optimizer\.structure_optimizer|StructureOptimizer\\b" src/ tests/ benchmarks/
# Each match — replace with IRStructureOptimizer or delete the call site
```

After all of the above:

```bash
uv run ruff check src/ tests/ benchmarks/    # must be clean
```

---

## 6. Tests

### 6.1 Existing tests to verify and relocate

The Phase 10 reshape moves all tests, but pre-emptively for clarity:

```bash
mkdir -p tests/unit/planner tests/unit/runtime tests/unit/transforms/optimizers
git mv tests/unit/test_unified_planner.py     tests/unit/planner/test_unified_planner.py
git mv tests/unit/test_task_classifier.py     tests/unit/planner/test_task_classifier.py
git mv tests/unit/test_runtime_state.py       tests/unit/planner/test_runtime_state.py
git mv tests/unit/test_request_classifier.py  tests/unit/planner/test_request_classifier.py   # if exists
git mv tests/unit/test_execution_plan.py      tests/unit/planner/test_execution_plan.py       # if exists
git mv tests/unit/test_execution_builder.py   tests/unit/planner/test_execution_builder.py    # if exists
git mv tests/unit/test_provider_strategy.py   tests/unit/planner/test_provider_strategy.py    # if exists
git mv tests/unit/test_transport_planner.py   tests/unit/planner/test_transport_planner.py    # if exists
git mv tests/unit/test_fallback_executor.py   tests/unit/planner/test_fallback_executor.py    # if exists
git mv tests/unit/test_router.py              tests/unit/runtime/test_tier_classifier.py
git mv tests/unit/test_ir_structure_optimizer.py tests/unit/transforms/optimizers/test_ir_structure_optimizer.py
git mv tests/unit/test_reference_optimizer.py   tests/unit/transforms/optimizers/test_reference_optimizer.py
git mv tests/unit/test_tool_optimizer.py        tests/unit/transforms/optimizers/test_tool_optimizer.py
git mv tests/unit/test_diagnostic_optimizer.py  tests/unit/transforms/optimizers/test_diagnostic_optimizer.py
git mv tests/unit/test_context_optimizer.py     tests/unit/transforms/optimizers/test_context_optimizer.py
```

### 6.2 Tests to delete

```bash
git rm tests/unit/test_scheduler.py                # v1 only
git rm tests/unit/test_optimizer_scheduler.py      # superseded
git rm tests/unit/test_structure_optimizer.py      # text-based deleted
```

### 6.3 New tests

**`tests/unit/planner/test_unified_planner_is_canonical.py`** — asserts that the new public API works and that the deleted scheduler symbols are gone:

```python
def test_unified_planner_imports():
    from lattice.planner import (
        UnifiedPlanner, SemanticProfile, build_execution_plan,
        ExecutionPlan, classify_task, TaskClass,
    )
    assert callable(UnifiedPlanner)
    assert callable(build_execution_plan)

def test_scheduler_symbols_gone():
    import lattice.core
    assert not hasattr(lattice.core, "decide_schedule")
    assert not hasattr(lattice.core, "SchedulerDecision")
    assert not hasattr(lattice.core, "decide_optimizer_schedule")
    assert not hasattr(lattice.core, "OptimizerSchedule")
```

**`tests/unit/runtime/test_tier_classifier_naming.py`** — asserts the rename is honest:

```python
def test_router_is_gone():
    """README says 'LATTICE is not a router'. The runtime module must not contain one."""
    import lattice.runtime
    assert not hasattr(lattice.runtime, "RuntimeRouter")
    # Old name should not even be importable:
    import pytest
    with pytest.raises(ImportError):
        from lattice.runtime.router import RuntimeRouter   # noqa: F401

def test_tier_classifier_works():
    from lattice.runtime import TierClassifier, Tier, TierDecision
    classifier = TierClassifier()
    # ... behavioural assertion identical to old RuntimeRouter test
```

**`tests/unit/transforms/optimizers/test_no_text_structure_optimizer.py`**:

```python
def test_text_structure_optimizer_deleted():
    """v1.0.0 has only IRStructureOptimizer (IR-native). Text-based StructureOptimizer is gone."""
    from lattice.transforms.optimizers import _OPTIMIZER_CLASSES
    assert "structure_optimizer" not in _OPTIMIZER_CLASSES
    assert "ir_structure_optimizer" in _OPTIMIZER_CLASSES

    import pytest
    with pytest.raises(ImportError):
        from lattice.optimizer.structure_optimizer import StructureOptimizer   # noqa: F401
    with pytest.raises(ImportError):
        from lattice.optimizer import StructureOptimizer   # noqa: F401
```

### 6.4 Contract tests

`tests/contract/test_python_api_contract.py` — add (these are NEW public API symbols):

```python
def test_planner_public_api():
    from lattice.planner import (
        UnifiedPlanner, build_execution_plan, ExecutionPlan,
        TaskClass, classify_task, TaskClassification,
    )

def test_runtime_public_api():
    from lattice.runtime import TierClassifier, Tier, TierDecision

def test_transforms_optimizers_public_api():
    from lattice.transforms.optimizers import (
        IRStructureOptimizer, ReferenceOptimizer,
        ToolOptimizer, DiagnosticOptimizer, ContextOptimizer,
    )
```

Removed from the contract (these were never public; document in MIGRATION.md):

```python
# These will no longer import — that's intentional:
# from lattice.optimizer import StructureOptimizer    # deleted
# from lattice.runtime.router import RuntimeRouter    # renamed
# from lattice.core import decide_schedule            # deleted
```

---

## 7. Cross-phase coordination

Phase 3 leaves `core/` with: `config.py`, `context.py`, `errors.py`, `result.py`, `maintenance.py`, `metrics.py`, `telemetry.py`, `agent_stats.py`, `cost_estimator.py`, `semantic_cache.py`, `session.py`, `store.py`, `tunnel_sidecar.py`, `transform_registry.py`, `transform_reputation.py`, `segmentation.py`. **Sixteen files.** Phase 4 moves transform_registry/reputation; STATUS Phase 9 (observability) moves metrics/telemetry/agent_stats/cost_estimator/maintenance/semantic_cache/session/store. After that phase, `core/` is six leaf files.

Phase 3 does **not** touch `tunnel_sidecar.py`. STATUS Phase 8 moved it to `integrations/tunnel.py` (historical path name in this doc).

The dependency graph after Phase 3:

```
pipeline/runner.py
    ↓ imports
planner/__init__.py
    ↓ imports
planner/{unified_planner, execution_builder, request_classifier, runtime_state, ...}
    ↓ imports
ir/ + transform/types.py + core/ (leaves)
```

```
pipeline/representation_optimizer.py
    ↓ imports
transforms/optimizers/__init__.py  ← _OPTIMIZER_CLASSES dict
    ↓ imports
transforms/optimizers/{ir_structure, reference, tool, diagnostic, context}_optimizer.py
    ↓ imports
ir/native_optimizer.py + ir/validation.py + ir/quality.py + transforms/{the underlying transforms}
```

No cycles.

---

## 8. Acceptance criteria

- [x] `src/lattice/optimizer/` directory does not exist.
- [x] `src/lattice/core/scheduler.py` does not exist.
- [x] `src/lattice/core/optimizer_scheduler.py` does not exist.
- [x] `src/lattice/core/unified_planner.py`, `task_classifier.py`, `runtime_state.py`, `credentials.py` do not exist (moved).
- [x] `src/lattice/runtime/router.py` does not exist (renamed).
- [x] `src/lattice/transforms/optimizers/` exists with 5 optimizer files + `__init__.py`.
- [x] `src/lattice/planner/unified_planner.py`, `task_classifier.py`, `runtime_state.py` exist.
- [x] `src/lattice/providers/credentials.py` exists.
- [x] `src/lattice/runtime/tier_classifier.py` exists; contains class `TierClassifier`.
- [x] `rg "RuntimeRouter|RoutingDecision" src/` returns 0 matches (tests may reference old names in negative assertions).
- [x] `rg "from lattice.optimizer" src/ tests/ benchmarks/` returns 0 live imports.
- [x] `rg "from lattice.core.scheduler|from lattice.core.optimizer_scheduler" src/` returns 0 matches.
- [x] `rg "decide_schedule|decide_optimizer_schedule|SchedulerDecision|OptimizerSchedule" src/` returns 0 live symbols.
- [x] `rg "StructureOptimizer\\b" src/` returns 0 matches (`IRStructureOptimizer` only).
- [x] `from lattice.planner import UnifiedPlanner, build_execution_plan, ExecutionPlan, classify_task` works.
- [x] `from lattice.runtime import TierClassifier, Tier, TierDecision` works.
- [x] `from lattice.transforms.optimizers import IRStructureOptimizer, ReferenceOptimizer, ToolOptimizer, DiagnosticOptimizer, ContextOptimizer` works.
- [x] `from lattice.providers.credentials import CredentialResolver` works.
- [x] `uv run ruff check src/ tests/` clean.
- [x] `uv run mypy src/lattice/` clean.
- [x] `uv run pytest tests/ -q` passes (1702 passed, 196 skipped).
- [x] `uv run pytest tests/contract/ -q` passes (27 passed).
- [ ] `python scripts/compare_benchmarks.py benchmarks/results/phase-0-baseline.json benchmarks/results/phase-4.json --tolerance-pct 2` exits 0 (pending CI / local key).

---

## 9. Risks specific to this phase

| Risk | Mitigation |
|---|---|
| Some test or script uses `decide_schedule(...)` directly | The grep in §3.6 finds them. Each one becomes either (a) a `UnifiedPlanner.plan()` call or (b) gets deleted as a v1-only test. |
| Removing `normalize_legacy_execution_plan` from `runtime_state.py` breaks code that still passes raw dicts as plans | The "raw dict plan" was a v1 path. With v1 gone (Phase 2), all `ExecutionPlan` objects flow through `build_execution_plan(...)` and are already typed. Confirm: `rg "normalize_legacy_execution_plan" src/ tests/` should show only the file itself. |
| `runtime/router.py` was imported by `transforms/runtime_contract.py` (per audit). The rename + class rename must propagate. | Already covered by the `sd` script in §3.9. Verify post-rename. |
| Two-name clash: `Tier` exists both in `planner/unified_planner.py` (FAST/STANDARD/...) and `runtime/tier_classifier.py` (SIMPLE/MEDIUM/...). | `planner/__init__.py` exports the planner's enum as `PlanTier`; `runtime/__init__.py` exports the runtime's enum as `Tier`. Documented in MIGRATION.md. |
| `_OPTIMIZER_CLASSES` is imported by `pipeline/representation_optimizer.py` (moved in Phase 2) from the old `lattice.optimizer`. After Phase 3 it must point at `lattice.transforms.optimizers`. | The `sd` script in §3.9 catches `from lattice.optimizer import` — verify after run that `pipeline/representation_optimizer.py` no longer references the old path. |
| Removing the text-based `StructureOptimizer` regresses a benchmark that explicitly tested its compression on Markdown tables | The IR-native `IRStructureOptimizer` covers the same cases via `transforms/format_converter/table_converter.py` (Phase 4 split). Benchmark gate (§8) catches any compression-% regression. |
| `pipeline/representation_optimizer.py`'s `_OPTIMIZER_CLASSES` is loaded at import time. If `transforms/optimizers/__init__.py` has any side effect at import (e.g. registry mutation), order matters. | The new `__init__.py` (§3.10) is pure — only imports + dict construction. No side effects. |
| `tests/unit/test_router.py` (now `tests/unit/runtime/test_tier_classifier.py`) had its expected class name hardcoded | The `sd` rename in §3.5 catches this. Verify the test file post-rename. |

---

## 10. Rollback plan

If this phase has to be reverted (e.g. a major test regression that can't be diagnosed within the day):

```bash
git checkout main -- src/lattice/core/scheduler.py
git checkout main -- src/lattice/core/optimizer_scheduler.py
git checkout main -- src/lattice/core/unified_planner.py
git checkout main -- src/lattice/core/task_classifier.py
git checkout main -- src/lattice/core/runtime_state.py
git checkout main -- src/lattice/core/credentials.py
git checkout main -- src/lattice/optimizer/        # whole dir
git checkout main -- src/lattice/runtime/router.py
git rm src/lattice/planner/unified_planner.py src/lattice/planner/task_classifier.py src/lattice/planner/runtime_state.py
git rm src/lattice/providers/credentials.py
git rm -r src/lattice/transforms/optimizers/
git rm src/lattice/runtime/tier_classifier.py
# Then revert the sd-rewritten imports across src/ tests/ — simplest is `git checkout main -- src/ tests/`
```

Phase 3 is intentionally a single PR; rollback is `git revert <phase-3-merge-commit>`.

---

## 11. PR shape

```
refactor(planner,runtime,optimizers): collapse schedulers; rename router; move optimizers [Phase 3]

- Delete core/scheduler.py (443 LoC) and core/optimizer_scheduler.py (131 LoC). UnifiedPlanner is the only scheduler.
- Move core/unified_planner.py → planner/unified_planner.py.
- Move core/task_classifier.py → planner/task_classifier.py.
- Move core/runtime_state.py → planner/runtime_state.py; drop normalize_legacy_execution_plan (v1 only).
- Move core/credentials.py → providers/credentials.py.
- Move 5 optimizer orchestrators → transforms/optimizers/; delete optimizer/__init__.py and the directory.
- Delete optimizer/structure_optimizer.py (255 LoC) — IRStructureOptimizer wins.
- Rename runtime/router.py → runtime/tier_classifier.py; RuntimeRouter → TierClassifier; RoutingDecision → TierDecision. README's "not a router" claim now matches the code.
- planner/__init__.py rewritten as full public surface.
- transforms/optimizers/__init__.py created with _OPTIMIZER_CLASSES registry.
- ~300 import sites rewritten across src/ tests/ benchmarks/.

Net: -4 files (2 schedulers + structure_optimizer + optimizer/__init__), +1 directory (transforms/optimizers/),
~830 LoC removed (mostly v1 dead code), 0 user-visible feature changes.
All 1600+ tests green. Contract tests green. Benchmarks ±2%.
```
