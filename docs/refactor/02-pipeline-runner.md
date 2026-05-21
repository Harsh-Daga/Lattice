# Phase 2 — Single Pipeline Runner

> **Goal.** End the v1/v2 pipeline duality. Delete `core/pipeline.py` (1092 LoC v1) and `core/pipeline_v2_wrapper.py` (118 LoC bridge). Rename `core/pipeline_v2.py` → `pipeline/runner.py`. Move every execution-time concern (`core/pipeline_factory.py`, `core/policy.py`, `core/guardrails.py`, `core/milv.py`, `core/auto_continuation.py`, `core/batch_accumulator.py`, `optimizer/representation_optimizer.py`) into the new `pipeline/` package. Move the transport protocol types (`core/transport.py`) into `transport/types.py`.
>
> **Outcome.** There is exactly one `Pipeline` class in the codebase. `from lattice.pipeline import Pipeline, build_pipeline` is the only entry. Every transform exposes only `optimize(ir, request, ctx) → Result[PromptIRV2]` — the legacy `process(request, ctx) → Result[Request]` method is removed wherever an IR-native path exists.
>
> **Estimated effort.** 2 days.

---

## 1. Why this phase exists

Today the runtime has **two pipelines**:

1. **`CompressorPipeline` (v1)** in `core/pipeline.py` — 1092 LoC, stateful, async, reactive scheduling, embeds policy + guardrails + reputation + budget tracking + MILV in the orchestration body. Every transform must implement `process(request, ctx) → Result[Request]`.
2. **`PipelineV2` (v2)** in `core/pipeline_v2.py` — 335 LoC, immutable, sync, plan-driven (`ExecutionPlan` pre-decides what to run), no policy/guardrails in-band. Transforms implement `optimize(ir, request, ctx) → Result[PromptIRV2]`.

They are bridged by `core/pipeline_v2_wrapper.py` (118 LoC), which makes `PipelineV2` look like a `ReversibleSyncTransform` so it can run inside `CompressorPipeline`. This produces the worst of both worlds: every transform must implement both methods, and there are two scheduling decisions (legacy reactive + planner-driven) that may disagree.

**v1.0.0 cuts v1 entirely.** The user-approved decision: PipelineV2 is the only path. Every `process()` method that has a sibling `optimize()` is deleted. Transforms that *only* have `process()` either get an `optimize()` (Phase 4) or are deleted.

This phase also resolves the **three-way name collision** on the word "transport":

- `core/transport.py` (259 LoC) — protocol *types*: `Role`, `Message`, `Request`, `Response`, `Transform`, `SyncTransform` Protocols.
- `transport/` (top-level dir) — TACC (`congestion.py`), simulation.
- `providers/transport.py` (1539 LoC) — HTTP dispatcher.

Phase 2 makes `core/transport.py` → `transport/types.py`. Phase 5 splits `providers/transport.py` into `providers/transport/{...}`. The three meanings become three orthogonal locations.

---

## 2. Files touched

### 2.1 Moved

| Current path | New path |
|---|---|
| `src/lattice/core/pipeline_v2.py` | `src/lattice/pipeline/runner.py` |
| `src/lattice/core/pipeline_factory.py` | `src/lattice/pipeline/factory.py` |
| `src/lattice/core/policy.py` | `src/lattice/pipeline/policy.py` |
| `src/lattice/core/guardrails.py` | `src/lattice/pipeline/guardrails.py` |
| `src/lattice/core/milv.py` | `src/lattice/pipeline/milv.py` |
| `src/lattice/core/auto_continuation.py` | `src/lattice/pipeline/auto_continuation.py` |
| `src/lattice/core/batch_accumulator.py` | `src/lattice/pipeline/batch_accumulator.py` |
| `src/lattice/optimizer/representation_optimizer.py` | `src/lattice/pipeline/representation_optimizer.py` |
| `src/lattice/core/transport.py` | `src/lattice/transport/types.py` |
| `src/lattice/core/serialization.py` | `src/lattice/transport/serialization.py` |
| `src/lattice/core/delta_wire.py` | `src/lattice/transport/delta_wire.py` |

### 2.2 Created

```
src/lattice/pipeline/__init__.py
src/lattice/transport/__init__.py        # MODIFIED if exists; CREATED if not
```

### 2.3 Deleted

```
src/lattice/core/pipeline.py             # v1 — 1092 LoC, gone
src/lattice/core/pipeline_v2_wrapper.py  # bridge — 118 LoC, gone
```

Plus every legacy `process()` method in every transform that also has `optimize()`. See §3.5.

### 2.4 Modified

- `src/lattice/core/__init__.py` — drop pipeline + transport re-exports; only leaf primitives now.
- `src/lattice/__init__.py` — already re-exports through `lattice.ir`; add `from lattice.pipeline import Pipeline` and `from lattice.transport.types import Request, Response, Message, Role`.
- All `from lattice.core.pipeline...`, `from lattice.core.transport...`, `from lattice.core.policy...`, `from lattice.core.guardrails...`, `from lattice.core.milv...`, `from lattice.core.auto_continuation...`, `from lattice.core.batch_accumulator...`, `from lattice.core.serialization...`, `from lattice.core.delta_wire...`, `from lattice.optimizer.representation_optimizer...` import sites — rewritten.
- `src/lattice/core/__init__.py` from Phase 1 re-exports `Request/Response/Message/Role/Transform/SyncTransform` from `lattice.core.transport`; update those to point at `lattice.transport.types`.

---

## 3. Step-by-step

### 3.1 Create the new packages

```bash
mkdir -p src/lattice/pipeline
# transport/ already exists (has congestion.py); just ensure __init__.py
```

### 3.2 Move files

```bash
git mv src/lattice/core/pipeline_v2.py            src/lattice/pipeline/runner.py
git mv src/lattice/core/pipeline_factory.py       src/lattice/pipeline/factory.py
git mv src/lattice/core/policy.py                 src/lattice/pipeline/policy.py
git mv src/lattice/core/guardrails.py             src/lattice/pipeline/guardrails.py
git mv src/lattice/core/milv.py                   src/lattice/pipeline/milv.py
git mv src/lattice/core/auto_continuation.py      src/lattice/pipeline/auto_continuation.py
git mv src/lattice/core/batch_accumulator.py      src/lattice/pipeline/batch_accumulator.py
git mv src/lattice/optimizer/representation_optimizer.py src/lattice/pipeline/representation_optimizer.py

git mv src/lattice/core/transport.py              src/lattice/transport/types.py
git mv src/lattice/core/serialization.py          src/lattice/transport/serialization.py
git mv src/lattice/core/delta_wire.py             src/lattice/transport/delta_wire.py
```

### 3.3 Delete v1 pipeline and the v2 bridge

```bash
git rm src/lattice/core/pipeline.py
git rm src/lattice/core/pipeline_v2_wrapper.py
```

After this, every consumer that imports `CompressorPipeline` or `PipelineV2Wrapper` is broken. The next steps fix them.

### 3.4 Rewrite imports across the codebase

Run these from repo root (using `sd` or `sed -i`):

```bash
# Pipeline-related renames
sd 'from lattice\.core\.pipeline_v2 import'                'from lattice.pipeline.runner import'           $(rg -l "from lattice.core.pipeline_v2 import")
sd 'from lattice\.core\.pipeline_factory import'           'from lattice.pipeline.factory import'           $(rg -l "from lattice.core.pipeline_factory import")
sd 'from lattice\.core\.policy import'                     'from lattice.pipeline.policy import'            $(rg -l "from lattice.core.policy import")
sd 'from lattice\.core\.guardrails import'                 'from lattice.pipeline.guardrails import'        $(rg -l "from lattice.core.guardrails import")
sd 'from lattice\.core\.milv import'                       'from lattice.pipeline.milv import'              $(rg -l "from lattice.core.milv import")
sd 'from lattice\.core\.auto_continuation import'          'from lattice.pipeline.auto_continuation import' $(rg -l "from lattice.core.auto_continuation import")
sd 'from lattice\.core\.batch_accumulator import'          'from lattice.pipeline.batch_accumulator import' $(rg -l "from lattice.core.batch_accumulator import")
sd 'from lattice\.optimizer\.representation_optimizer import' 'from lattice.pipeline.representation_optimizer import' $(rg -l "from lattice.optimizer.representation_optimizer import")

# Transport-types renames
sd 'from lattice\.core\.transport import'                  'from lattice.transport.types import'            $(rg -l "from lattice.core.transport import")
sd 'from lattice\.core\.serialization import'              'from lattice.transport.serialization import'    $(rg -l "from lattice.core.serialization import")
sd 'from lattice\.core\.delta_wire import'                 'from lattice.transport.delta_wire import'       $(rg -l "from lattice.core.delta_wire import")
```

This rewrites maybe 200+ import lines across the codebase. Verify with `uv run ruff check src/ tests/ benchmarks/`.

### 3.5 Delete every legacy `process()` method that has a sibling `optimize()`

Per the Phase 4 audit, **11 transforms** currently define both `process(request, ctx) → Result[Request]` *and* `optimize(ir, request, ctx) → Result[PromptIRV2]`:

```
cache_arbitrage.py
causal_chain.py
format_conv.py            (will be split in Phase 4 into format_converter/ — handle then; for Phase 2, delete here)
message_dedup.py
path_prefix.py
prefix_opt.py             (entire file is deleted in Phase 4 — leave the dual for now; or delete the file in this phase)
rate_distortion.py
reference_sub.py
runtime_contract.py
tool_filter.py
tool_projection.py
```

Plus the constraint_lifting and strategy_selector files where `optimize()` exists but is NOT in `PipelineV2._IR_NATIVE_TRANSFORMS` — for those, *delete the `optimize()` method instead*; keep `process()` until Phase 4 resolves the file's fate.

The per-file action:

| File | Action |
|---|---|
| `transforms/cache_arbitrage.py` | Delete the `process()` method (~50 LoC); keep `optimize()` |
| `transforms/causal_chain.py` | Same |
| `transforms/format_conv.py` | Same (will be re-split in Phase 4) |
| `transforms/message_dedup.py` | Same |
| `transforms/path_prefix.py` | Same |
| `transforms/rate_distortion.py` | Same |
| `transforms/reference_sub.py` | Same |
| `transforms/runtime_contract.py` | Same |
| `transforms/tool_filter.py` | Same |
| `transforms/tool_projection.py` | Same |
| `transforms/prefix_opt.py` | **NO CHANGE** in Phase 2; entire file deleted in Phase 4 |
| `transforms/constraint_lifting.py` | Delete its `optimize()` (not IR-native) |
| `transforms/strategy_selector.py` | Delete its `optimize()` (not IR-native) — entire file may be cut in Phase 4; this is harmless if deleted |

Each removal is mechanical: find `def process(`, find its matching closing `def` or end-of-class, delete the block, run tests. After deletion the transform class's `ReversibleSyncTransform` Protocol implementation must still satisfy the protocol — verify with mypy.

### 3.6 Rewrite `pipeline/runner.py` (was `pipeline_v2.py`)

The renamed file should drop `V2` from its class name. `PipelineV2` → `Pipeline`. Everywhere it's used:

```bash
sd '\bPipelineV2\b' 'Pipeline' $(rg -l "PipelineV2")
sd '_IR_NATIVE_TRANSFORMS' '_IR_NATIVE_TRANSFORMS' $(rg -l "_IR_NATIVE_TRANSFORMS")  # name stays
```

The `TransformRegistryV2` class also drops `V2`:

```bash
sd '\bTransformRegistryV2\b' 'TransformRegistry' $(rg -l "TransformRegistryV2")
```

But there's no name collision because `core/transform_registry.py` (which gets moved to `transforms/registry.py` in a later phase) defines spec metadata (`TransformSpec`), not the runtime registry that `PipelineV2`/now-`Pipeline` uses. To avoid confusion, rename:

- `TransformRegistryV2` in `pipeline/runner.py` → `PipelineTransformRegistry`

```bash
sd '\bTransformRegistryV2\b' 'PipelineTransformRegistry' $(rg -l "TransformRegistryV2")
```

### 3.7 Rewrite `pipeline/factory.py` to remove v1 builders

`pipeline_factory.py` today exports:

- `build_default_pipeline(config) → CompressorPipeline`
- `build_v2_pipeline(config) → PipelineV2`
- `build_optimizer_pipeline(config) → CompressorPipeline`
- `build_benchmark_pipeline(config) → CompressorPipeline`

After Phase 2, only one pipeline exists. Rewrite:

- `build_default_pipeline(config) → Pipeline` — was `build_v2_pipeline` body
- `build_benchmark_pipeline(config) → Pipeline` — same body with benchmark-specific config flags
- Delete `build_optimizer_pipeline` entirely — its v1-specific code path no longer makes sense
- Delete `build_v2_pipeline` (its body is now `build_default_pipeline`)

`pipeline_summary(pipeline)` is kept; its body adapts to the now-single Pipeline type.

### 3.8 Update `core/__init__.py`

Remove all the now-moved re-exports. Final shape (after Phase 1's segmentation addition):

```python
"""Core primitives — leaf types only. No domain logic."""

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.errors import (
    ConfigurationError, LatticeError, ProviderError, ProviderTimeoutError,
    RequestTooLargeError, SessionError, SessionExpiredError,
    SessionNotFoundError, SessionStoreError,
    TransformError, TransformNotFoundError, ValidationError,
)
from lattice.core.result import Err, Ok, Result, is_err, is_ok, unwrap, unwrap_err
from lattice.core.segmentation import (
    SegmentKind, SemanticSegment, segment_request, segment_summary,
)

# Re-export wire types from their new home for backwards-compatible imports:
from lattice.transport.types import (
    Message, Request, Response, Role,
    Transform, SyncTransform,
)
# ReversibleSyncTransform now lives in pipeline/; re-export here so old imports keep working:
from lattice.pipeline.runner import ReversibleSyncTransform  # type: ignore[attr-defined]

__all__ = [
    "LatticeConfig", "TransformContext",
    "ConfigurationError", "LatticeError", "ProviderError", "ProviderTimeoutError",
    "RequestTooLargeError", "SessionError", "SessionExpiredError",
    "SessionNotFoundError", "SessionStoreError",
    "TransformError", "TransformNotFoundError", "ValidationError",
    "Result", "Ok", "Err", "is_ok", "is_err", "unwrap", "unwrap_err",
    "Message", "Request", "Response", "Role",
    "Transform", "SyncTransform", "ReversibleSyncTransform",
    "SegmentKind", "SemanticSegment", "segment_request", "segment_summary",
]
```

Note: `CompressorPipeline` is gone — `core/__init__.py` no longer re-exports it. The `tests/contract/test_python_api_contract.py` from Phase 0 must be updated to **not** import `CompressorPipeline` (it's an internal name, never part of the public surface — verify by grepping user-facing docs; if it was documented, deprecate it in CHANGELOG). The user-facing surface is `LatticeClient`, `LatticeProxyClient`, `wrap_openai_client`, and the IR types — not the pipeline runner.

### 3.9 Write `pipeline/__init__.py`

```python
"""Execution-time pipeline: runner, factory, policy, guardrails, validation."""

from lattice.pipeline.runner import (
    Pipeline,
    PipelineTransformRegistry,
    ReversibleSyncTransform,
)
from lattice.pipeline.factory import (
    build_default_pipeline,
    build_benchmark_pipeline,
    pipeline_summary,
)
from lattice.pipeline.policy import OptimizationPolicy, Allow, Skip, Reject
from lattice.pipeline.guardrails import (
    GuardAction, SafetyDecision, ValidationOutcome,
    check_expansion_guard, check_entity_preservation, check_format_preservation,
    check_critical_signal_loss, check_placeholder_leakage,
    check_negative_savings, check_blank_output,
)
from lattice.pipeline.milv import MILVResult, should_trigger_milv, validate_transform
from lattice.pipeline.auto_continuation import AutoContinuation, ContinuationResult
from lattice.pipeline.batch_accumulator import BatchAccumulator, BatchResult, AccumulatedRequest
from lattice.pipeline.representation_optimizer import RepresentationOptimizer

__all__ = [
    "Pipeline", "PipelineTransformRegistry", "ReversibleSyncTransform",
    "build_default_pipeline", "build_benchmark_pipeline", "pipeline_summary",
    "OptimizationPolicy", "Allow", "Skip", "Reject",
    "GuardAction", "SafetyDecision", "ValidationOutcome",
    "check_expansion_guard", "check_entity_preservation", "check_format_preservation",
    "check_critical_signal_loss", "check_placeholder_leakage",
    "check_negative_savings", "check_blank_output",
    "MILVResult", "should_trigger_milv", "validate_transform",
    "AutoContinuation", "ContinuationResult",
    "BatchAccumulator", "BatchResult", "AccumulatedRequest",
    "RepresentationOptimizer",
]
```

### 3.10 Write `transport/__init__.py`

```python
"""Protocol-level transport: types, serialization, congestion control, delta wire, session.

NOTE: HTTP transport (DirectHTTPProvider, ConnectionPoolManager) lives in
providers/transport/. This package is for protocol concerns above the HTTP layer.
"""

from lattice.transport.types import (
    Role, Message, Request, Response,
    Transform, SyncTransform,
)
from lattice.transport.serialization import (
    message_to_dict, message_from_dict,
    request_to_dict, request_from_dict,
    response_to_dict,
)
from lattice.transport.congestion import TACCController, ProviderCongestionState, AdmissionDecision
from lattice.transport.delta_wire import DeltaWireDecoder, DeltaWireEncoder, delta_wire_bytes, compute_wire_savings
# session.py is moved here in Phase 8 — once it is, add:
# from lattice.transport.session import Session, SessionManager, SessionStore, MemorySessionStore

__all__ = [
    "Role", "Message", "Request", "Response",
    "Transform", "SyncTransform",
    "message_to_dict", "message_from_dict",
    "request_to_dict", "request_from_dict",
    "response_to_dict",
    "TACCController", "ProviderCongestionState", "AdmissionDecision",
    "DeltaWireDecoder", "DeltaWireEncoder",
    "delta_wire_bytes", "compute_wire_savings",
]
```

### 3.11 Per-file disposition table

| File (after move) | LoC | Status | Notes |
|---|---|---|---|
| `pipeline/runner.py` | ~330 | MODIFY | Rename `PipelineV2` → `Pipeline`, `TransformRegistryV2` → `PipelineTransformRegistry`. Lift `ReversibleSyncTransform` here (was in `core/pipeline.py`); re-export from `core/__init__.py` for compat. |
| `pipeline/factory.py` | ~250 | MODIFY | Delete `build_optimizer_pipeline`; rename `build_v2_pipeline` → `build_default_pipeline`; remove v1 references |
| `pipeline/policy.py` | 337 | KEEP | Path-only move |
| `pipeline/guardrails.py` | 409 | KEEP | Path-only move |
| `pipeline/milv.py` | 159 | KEEP | Path-only move |
| `pipeline/auto_continuation.py` | 215 | KEEP | Path-only move |
| `pipeline/batch_accumulator.py` | 268 | KEEP | Path-only move |
| `pipeline/representation_optimizer.py` | 431 | KEEP | Path-only move |
| `pipeline/__init__.py` | NEW (~40) | CREATE | Per §3.9 |
| `transport/types.py` | 259 | KEEP | Path-only move |
| `transport/serialization.py` | 325 | KEEP | Path-only move |
| `transport/delta_wire.py` | 383 | KEEP | Path-only move |
| `transport/__init__.py` | MODIFY (~30) | UPDATE | Per §3.10 |
| `transport/congestion.py` | 559 | UNCHANGED | Already in transport/ |
| `transport/simulation.py` | 203 | UNCHANGED | Already in transport/ |
| `core/__init__.py` | MODIFY | UPDATE | Per §3.8 — drop pipeline/transport exports, keep leaf primitives |
| `core/pipeline.py` | — | **DELETED** | v1 gone |
| `core/pipeline_v2_wrapper.py` | — | **DELETED** | Bridge gone |
| `core/transport.py` | — | **MOVED** to `transport/types.py` |
| `core/serialization.py` | — | **MOVED** to `transport/serialization.py` |
| `core/delta_wire.py` | — | **MOVED** to `transport/delta_wire.py` |
| `core/policy.py` | — | **MOVED** to `pipeline/policy.py` |
| `core/guardrails.py` | — | **MOVED** to `pipeline/guardrails.py` |
| `core/milv.py` | — | **MOVED** to `pipeline/milv.py` |
| `core/auto_continuation.py` | — | **MOVED** to `pipeline/auto_continuation.py` |
| `core/batch_accumulator.py` | — | **MOVED** to `pipeline/batch_accumulator.py` |
| `core/pipeline_factory.py` | — | **MOVED** to `pipeline/factory.py` |
| `optimizer/representation_optimizer.py` | — | **MOVED** to `pipeline/representation_optimizer.py` |

### 3.12 Update transform implementations to be `optimize()`-only

After §3.5, each transform's class no longer has a `process()` method. The `ReversibleSyncTransform` Protocol must allow this. Today the Protocol requires both. Update its Protocol definition in `pipeline/runner.py`:

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

The `process()` method is no longer required. Transforms that don't have an IR-native path (`constraint_lifting`, `strategy_selector` based on §3.5) must either gain one or be deleted in Phase 4 — Phase 2 does not yet delete them, but it removes their stale `optimize()` so they don't pretend to be IR-native.

For those v1-only transforms in Phase 2: keep their `process(request, ctx)` method, but the **`Pipeline` runner will not call it**. They become orphaned until Phase 4 either adds an IR path or deletes them. To prevent them from being silently scheduled, register them in `pipeline/runner.py` with a `legacy_only=True` flag that excludes them from `Pipeline.process()`:

```python
# In Pipeline._iter_active_transforms:
specs = list_transform_names()
for spec in specs:
    if spec.legacy_only:
        continue   # v2 Pipeline never runs legacy_only transforms
    ...
```

`TransformSpec` in `core/transform_registry.py` (moves to `transforms/registry.py` in Phase 4) already has fields; add `legacy_only: bool = False`. Mark `constraint_lifting` and `strategy_selector` `legacy_only=True` in Phase 2; Phase 4 makes the final keep/cut decision.

### 3.13 Verify imports

```bash
uv run ruff check src/ tests/ benchmarks/    # must be clean
uv run mypy src/lattice/                      # must be clean
rg "from lattice.core.pipeline" src/ tests/ benchmarks/   # 0 matches
rg "from lattice.core.transport" src/ tests/ benchmarks/  # 0 matches except core/__init__.py? — no, even that uses transport.types now
rg "CompressorPipeline" src/ tests/ benchmarks/   # 0 matches except docs/migration
rg "PipelineV2" src/ tests/ benchmarks/           # 0 matches
rg "pipeline_v2_wrapper" src/ tests/ benchmarks/  # 0 matches
```

---

## 4. Tests

### 4.1 Existing tests to verify or move

Pipeline tests at `tests/unit/test_pipeline.py`, `test_pipeline_v2.py`, `test_pipeline_factory.py`, `test_policy.py`, `test_guardrails.py`, `test_milv.py`, etc. After §3.4, their imports are correct. Move them in Phase 10 (or pre-emptively for clarity):

```bash
mkdir -p tests/unit/pipeline
git mv tests/unit/test_pipeline*.py     tests/unit/pipeline/
git mv tests/unit/test_policy.py        tests/unit/pipeline/
git mv tests/unit/test_guardrails.py    tests/unit/pipeline/
git mv tests/unit/test_milv.py          tests/unit/pipeline/
git mv tests/unit/test_auto_continuation.py tests/unit/pipeline/
git mv tests/unit/test_batch_accumulator.py tests/unit/pipeline/
git mv tests/unit/test_representation_optimizer.py tests/unit/pipeline/
```

Likewise for transport:

```bash
mkdir -p tests/unit/transport
git mv tests/unit/test_transport*.py    tests/unit/transport/
git mv tests/unit/test_delta_wire.py    tests/unit/transport/
git mv tests/unit/test_serialization.py tests/unit/transport/
git mv tests/unit/test_tacc*.py         tests/unit/transport/    # if those exist
```

### 4.2 v1 pipeline tests deleted

`tests/unit/test_pipeline.py` (if it tests CompressorPipeline specifically) and `tests/unit/test_pipeline_v2_wrapper.py` are **deleted**. Their coverage is replaced by the v2-only pipeline tests (which already exist as `test_pipeline_v2.py` → after move, `test_runner.py`).

Rename:

```bash
git mv tests/unit/pipeline/test_pipeline_v2.py tests/unit/pipeline/test_runner.py
git rm tests/unit/pipeline/test_pipeline.py        # the v1 file
git rm tests/unit/pipeline/test_pipeline_v2_wrapper.py
```

### 4.3 New tests

```
tests/unit/pipeline/test_no_legacy_process_paths.py
```

```python
"""Verify no transform that's in PipelineTransformRegistry has both optimize() and process()."""
import inspect
import pytest
from lattice.pipeline import PipelineTransformRegistry
from lattice.pipeline.runner import ReversibleSyncTransform

def test_no_dual_methods():
    registry = PipelineTransformRegistry()
    for name, transform_cls in registry.all_classes().items():
        has_optimize = hasattr(transform_cls, "optimize")
        has_process = hasattr(transform_cls, "process")
        # In v1.0.0, IR-native transforms MUST NOT have process()
        if has_optimize:
            assert not has_process, (
                f"Transform {name} has BOTH optimize() and process(). "
                f"v1.0.0 requires IR-native only."
            )
```

```
tests/unit/pipeline/test_transport_types_canonical_path.py
```

```python
"""Verify Request/Response import from canonical path."""
import inspect
from lattice.transport.types import Request, Response, Message, Role

def test_canonical_module():
    assert Request.__module__ == "lattice.transport.types"
    assert Response.__module__ == "lattice.transport.types"
    assert Message.__module__ == "lattice.transport.types"

def test_backcompat_reexport():
    # core/__init__.py re-exports these
    from lattice.core import Request as CoreRequest
    assert CoreRequest is Request
```

### 4.4 Contract tests

`tests/contract/test_python_api_contract.py` is updated:

- **Remove** `CompressorPipeline` from the imports list — it was internal, never part of public API. Phase 11 documents this in MIGRATION.md.
- **Add** `Pipeline` (the new name) to a section of "internal but stable" imports tested but documented as internal.
- Keep `from lattice.core import Request, Response, Message, Role, Transform, SyncTransform, ReversibleSyncTransform` working — the re-export buffer in `core/__init__.py` proves it.

---

## 5. Cross-phase coordination

Phase 2 leaves `optimizer/` with 7 files (was 10) — the rep-opt moved out plus the three Phase 1 IR moves. Phase 3 will collapse the rest of `optimizer/` into `transforms/optimizers/`.

Phase 2 also leaves `core/scheduler.py`, `core/optimizer_scheduler.py`, `core/unified_planner.py`, `core/task_classifier.py`, `core/runtime_state.py`. Phase 3 owns all of these.

Phase 2 deliberately does NOT touch `core/session.py`, `core/store.py`, `core/metrics.py`, `core/telemetry.py`, `core/cost_estimator.py`, `core/agent_stats.py`, `core/maintenance.py`, `core/semantic_cache.py`, `core/credentials.py`. Phase 8 owns those moves to `state/`, `telemetry/`, `cache/`, etc.

After Phase 2, `core/` contains: `config.py`, `context.py`, `credentials.py`, `errors.py`, `maintenance.py`, `metrics.py`, `result.py`, `runtime_state.py`, `scheduler.py`, `optimizer_scheduler.py`, `unified_planner.py`, `task_classifier.py`, `semantic_cache.py`, `session.py`, `store.py`, `tunnel_sidecar.py`, `telemetry.py`, `agent_stats.py`, `cost_estimator.py`, `segmentation.py` (added in Phase 1). 20 files — still bloated, but Phase 3 and Phase 8 finish the shrink.

---

## 6. Acceptance criteria

- [ ] `src/lattice/core/pipeline.py` does not exist.
- [ ] `src/lattice/core/pipeline_v2_wrapper.py` does not exist.
- [ ] `src/lattice/core/transport.py` does not exist (moved to `transport/types.py`).
- [ ] `src/lattice/pipeline/` exists with the 9 files in §3.11.
- [ ] `src/lattice/transport/types.py`, `serialization.py`, `delta_wire.py` exist.
- [ ] `rg "PipelineV2|CompressorPipeline|pipeline_v2_wrapper" src/ tests/ benchmarks/` returns 0 matches.
- [ ] `rg "from lattice.core.pipeline|from lattice.core.transport|from lattice.core.policy|from lattice.core.guardrails|from lattice.core.milv|from lattice.core.auto_continuation|from lattice.core.batch_accumulator|from lattice.core.serialization|from lattice.core.delta_wire|from lattice.optimizer.representation_optimizer" src/ tests/ benchmarks/` returns 0 matches.
- [ ] `from lattice.pipeline import Pipeline, build_default_pipeline` works.
- [ ] `from lattice.transport import Request, Response, Message, Role, TACCController` works.
- [ ] `from lattice.core import Request, Response` still works (re-export buffer).
- [ ] Every transform class in the registry has `optimize()` and no `process()` (verified by the new test in §4.3).
- [ ] `uv run ruff check src/ tests/` clean.
- [ ] `uv run mypy src/lattice/` clean.
- [ ] `uv run pytest tests/ -q` passes.
- [ ] `uv run pytest tests/contract/ -q` passes.
- [ ] Benchmarks: `python scripts/compare_benchmarks.py benchmarks/results/phase-0-baseline.json benchmarks/results/phase-2.json --tolerance-pct 2` exits 0.

---

## 7. Risks specific to this phase

| Risk | Mitigation |
|---|---|
| Removing v1's `CompressorPipeline` breaks something that wasn't exposed in `__init__.py` but was imported by an internal user (custom transform, benchmark, script) | Run `rg "CompressorPipeline" .` *before* deletion. Every match must either point to the v1 file (about to be deleted) or be updated to use `Pipeline`. If an external benchmark imports `CompressorPipeline`, update the benchmark. |
| A transform's `process()` had behaviour not yet in `optimize()` (e.g. side effects on context) | Phase 2 deletes `process()` only on transforms in `_IR_NATIVE_TRANSFORMS`. Those have known-equivalent `optimize()` paths. For any test that fails after deletion, port the missing logic into `optimize()` rather than restoring `process()`. |
| `pipeline/runner.py` after rename has internal `# noqa: V2` or "PipelineV2" docstring references | Search after rename: `rg -i "v2" src/lattice/pipeline/runner.py`. Strip any remaining references. |
| `ReversibleSyncTransform` Protocol's `process()` method was required by mypy somewhere | Update Protocol per §3.12. mypy must accept that v1.0 transforms only need `can_process` + `optimize` + `reverse`. |
| `core/transport.py` move breaks a circular-import workaround in `tunnel_sidecar.py` (the file has a `# noqa: E402` pragma) | `tunnel_sidecar.py` imports `LatticeConfig` lazily mid-module. Verify it doesn't import `Request`/`Response`. If it does, switch to `from lattice.transport.types import ...` and ensure tunnel_sidecar.py is still last-import order. |
| `benchmarks/evals/cli.py` has `--use-v2-pipeline` flag (per audit) | After Phase 2 there is no v1 pipeline. Remove the flag entirely (the only mode is the new Pipeline). The CLI flag becomes a no-op accepted-for-back-compat warning, removed in Phase 11. |
| Speed regression because v1 had async, v2 is sync | v2 sync pipeline is already faster for non-streaming requests (no event-loop yielding overhead). If a streaming-path benchmark regresses, profile and add async support specifically for the streaming path. |

---

## 8. PR shape

Single PR:

```
refactor(pipeline,transport): collapse to one Pipeline runner; move wire types [Phase 2]

- Delete core/pipeline.py (v1, 1092 LoC) and core/pipeline_v2_wrapper.py (118 LoC).
- Rename core/pipeline_v2.py → pipeline/runner.py; class PipelineV2 → Pipeline.
- Move pipeline_factory, policy, guardrails, milv, auto_continuation, batch_accumulator,
  optimizer/representation_optimizer → pipeline/.
- Move core/transport.py → transport/types.py (and serialization.py, delta_wire.py).
- Delete legacy process() method on 11 IR-native transforms; ReversibleSyncTransform
  Protocol updated to optimize()-only.
- Mark constraint_lifting and strategy_selector as legacy_only=True until Phase 4.
- core/ shrinks: removes pipeline, transport types, policy, guardrails, milv,
  auto_continuation, batch_accumulator, serialization, delta_wire, pipeline_factory.
- core/__init__.py re-exports Request/Response/Message/Role/Transform/SyncTransform/
  ReversibleSyncTransform for back-compat with v0.x importers.

Net: -2 files (v1 pipeline + bridge), +1 package (pipeline/), 3 files into transport/.
~200 import sites rewritten by scripted sd. All 1600+ tests green. Contract tests green.
Benchmarks ±2% of Phase 0 baseline.
```
