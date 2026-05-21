# Phase 1 — IR & Core Primitives

> **Goal.** Collapse the seven IR-adjacent files in `core/` (`ir.py`, `ir_builder.py`, `ir_normalizer.py`, `ir_serializer.py`, `ir_transform.py`, `primitives.py`, `compiler.py`, plus `semantic_graph.py`) into a single cohesive `ir/` package. Delete `core/compiler.py`. Move `optimizer/ir_native_optimizer.py`, `optimizer/validation.py`, `optimizer/quality_estimator.py` into `ir/`. Shrink `core/` to leaf primitives only.
>
> **Outcome.** `from lattice.ir import PromptIRV2, build_ir, normalize_ir, serialize_ir_to_text, IRTransform, CandidateSearch` is the canonical entry to everything IR-related. `core/` contains only `config.py`, `context.py`, `errors.py`, `result.py`, `tunnel_sidecar.py`, plus the relocated `segmentation.py`.
>
> **Estimated effort.** 1.5 days.

---

## 1. Why this phase exists

The IR is the spine of the system: every transform either operates on it (`optimize(ir, ...)`) or against the legacy Request shape (`process(req, ...)`). With seven files scattered across two parent directories — `core/` and `optimizer/` — even reading the IR flow takes ten file opens. The boundary between "IR types", "IR construction", "IR transformation protocol", and "IR-native optimisation" is exactly the same boundary as "things grouped together as a package".

Equally important: `core/compiler.py` is 58 lines, of which roughly 40 are docstring + boilerplate. The actual logic is **three function calls**:

```python
def compile(self, request, context):
    ir = build_ir(request)
    ir = normalize_ir(ir)
    return ir
```

There is no compiler; there is a function. Phase 1 deletes the file and inlines.

---

## 2. Files touched

### 2.1 Moved (no logic change, only path)

| Current path | New path |
|---|---|
| `src/lattice/core/ir.py` | `src/lattice/ir/types.py` |
| `src/lattice/core/ir_builder.py` | `src/lattice/ir/builder.py` |
| `src/lattice/core/ir_normalizer.py` | `src/lattice/ir/normalizer.py` |
| `src/lattice/core/ir_serializer.py` | `src/lattice/ir/serializer.py` |
| `src/lattice/core/ir_transform.py` | `src/lattice/ir/transform.py` |
| `src/lattice/core/primitives.py` | `src/lattice/ir/primitives.py` |
| `src/lattice/core/semantic_graph.py` | `src/lattice/ir/semantic_graph.py` |
| `src/lattice/optimizer/ir_native_optimizer.py` | `src/lattice/ir/native_optimizer.py` |
| `src/lattice/optimizer/validation.py` | `src/lattice/ir/validation.py` |
| `src/lattice/optimizer/quality_estimator.py` | `src/lattice/ir/quality.py` |
| `src/lattice/transforms/semantic_segmenter.py` | `src/lattice/core/segmentation.py` |

### 2.2 Created

```
src/lattice/ir/__init__.py
```

### 2.3 Deleted

```
src/lattice/core/compiler.py            # logic inlined into ir/builder.py
```

### 2.4 Modified

- `src/lattice/core/__init__.py` — strip IR re-exports; keep only `LatticeConfig`, `TransformContext`, errors, `Result/Ok/Err`, and re-export `Request/Response/Message/Role/Transform/SyncTransform` from their **new** transport location (anticipating Phase 2 — see §5 of this doc).
- `src/lattice/__init__.py` — add `from lattice.ir import PromptIR, PromptIRV2, build_ir, normalize_ir, serialize_ir_to_text` to keep the public Python API working.
- Every `from lattice.core.ir...` and `from lattice.core.primitives...` and `from lattice.core.compiler...` and `from lattice.optimizer.ir_native_optimizer...` import in the codebase — rewritten.

---

## 3. Step-by-step

### 3.1 Create the new package

```bash
mkdir -p src/lattice/ir
```

### 3.2 Move files (use `git mv` for history preservation)

```bash
git mv src/lattice/core/ir.py                    src/lattice/ir/types.py
git mv src/lattice/core/ir_builder.py            src/lattice/ir/builder.py
git mv src/lattice/core/ir_normalizer.py         src/lattice/ir/normalizer.py
git mv src/lattice/core/ir_serializer.py         src/lattice/ir/serializer.py
git mv src/lattice/core/ir_transform.py          src/lattice/ir/transform.py
git mv src/lattice/core/primitives.py            src/lattice/ir/primitives.py
git mv src/lattice/core/semantic_graph.py        src/lattice/ir/semantic_graph.py
git mv src/lattice/optimizer/ir_native_optimizer.py src/lattice/ir/native_optimizer.py
git mv src/lattice/optimizer/validation.py       src/lattice/ir/validation.py
git mv src/lattice/optimizer/quality_estimator.py src/lattice/ir/quality.py
git mv src/lattice/transforms/semantic_segmenter.py src/lattice/core/segmentation.py
```

### 3.3 Rewrite internal imports in the moved files

Each moved file imports from its old peers. Each import in the form `from lattice.core.ir import ...` or `from lattice.core.primitives import ...` becomes `from lattice.ir.types import ...` / `from lattice.ir.primitives import ...`. Specifically:

| In file | Old import | New import |
|---|---|---|
| `ir/builder.py` | `from lattice.core.ir import ...` | `from lattice.ir.types import ...` |
| `ir/builder.py` | `from lattice.core.transport import Request` | `from lattice.transport.types import Request` *(Phase 2 will guarantee this path; the re-export shim in §5 covers it now)* |
| `ir/normalizer.py` | `from lattice.core.ir import ...` | `from lattice.ir.types import ...` |
| `ir/serializer.py` | `from lattice.core.ir import ...` | `from lattice.ir.types import ...` |
| `ir/transform.py` | `from lattice.core.primitives import ...` | `from lattice.ir.primitives import ...` |
| `ir/transform.py` | `from lattice.core.compiler import ...` | (deleted call site — see 3.5 below) |
| `ir/transform.py` | `from lattice.core.context import TransformContext` | unchanged |
| `ir/transform.py` | `from lattice.core.errors import TransformError` | unchanged |
| `ir/transform.py` | `from lattice.core.result import ...` | unchanged |
| `ir/transform.py` | `from lattice.core.transport import ...` | unchanged (Phase 2 changes this — left alone for Phase 1) |
| `ir/primitives.py` | `from lattice.core.ir import ...` | `from lattice.ir.types import ...` |
| `ir/native_optimizer.py` | `from lattice.core.compiler import get_compiler` | replace with direct calls — see 3.5 |
| `ir/native_optimizer.py` | `from lattice.core.ir import ...` | `from lattice.ir.types import ...` |
| `ir/native_optimizer.py` | `from lattice.core.primitives import ...` | `from lattice.ir.primitives import ...` |
| `ir/validation.py` | `from lattice.core.primitives import Candidate` | `from lattice.ir.primitives import Candidate` |
| `ir/validation.py` | `from lattice.core.runtime_state import ...` | unchanged (Phase 3 will move runtime_state) |
| `ir/quality.py` | `from lattice.core.runtime_state import ...` | unchanged (Phase 3) |

Do this with `ruff --fix` for the simple renames, and a single `sed`-style script for paths that ruff doesn't auto-rewrite:

```bash
# Inside each moved file:
sd 'from lattice\.core\.ir import'           'from lattice.ir.types import'           src/lattice/ir/*.py
sd 'from lattice\.core\.primitives import'   'from lattice.ir.primitives import'      src/lattice/ir/*.py
sd 'from lattice\.core\.ir_builder import'   'from lattice.ir.builder import'         src/lattice/ir/*.py
sd 'from lattice\.core\.ir_normalizer import' 'from lattice.ir.normalizer import'     src/lattice/ir/*.py
sd 'from lattice\.core\.ir_serializer import' 'from lattice.ir.serializer import'     src/lattice/ir/*.py
sd 'from lattice\.core\.semantic_graph import' 'from lattice.ir.semantic_graph import' src/lattice/ir/*.py
```

(`sd` is the Rust `sed` replacement; `sed -i` works too. Verify with `ruff check` after.)

### 3.4 Rewrite consumer imports across the rest of `src/lattice/` and `tests/`

The same renames apply to every other file. Approximate counts (from the audit):

- `from lattice.core.ir` — ~30 sites
- `from lattice.core.primitives` — ~25 sites
- `from lattice.core.compiler` — ~5 sites
- `from lattice.optimizer.ir_native_optimizer` — ~3 sites
- `from lattice.optimizer.validation` — ~4 sites
- `from lattice.optimizer.quality_estimator` — ~2 sites

```bash
# Run from repo root:
sd 'from lattice\.core\.ir import'           'from lattice.ir.types import'           $(rg -l "from lattice.core.ir import")
sd 'from lattice\.core\.primitives import'   'from lattice.ir.primitives import'      $(rg -l "from lattice.core.primitives import")
sd 'from lattice\.core\.semantic_graph import' 'from lattice.ir.semantic_graph import' $(rg -l "from lattice.core.semantic_graph import")
sd 'from lattice\.optimizer\.ir_native_optimizer import' 'from lattice.ir.native_optimizer import' $(rg -l "from lattice.optimizer.ir_native_optimizer import")
sd 'from lattice\.optimizer\.validation import' 'from lattice.ir.validation import'   $(rg -l "from lattice.optimizer.validation import")
sd 'from lattice\.optimizer\.quality_estimator import' 'from lattice.ir.quality import' $(rg -l "from lattice.optimizer.quality_estimator import")
```

For `from lattice.core.compiler` (5 sites, all hand-edited because the API changes — see 3.5).

For `from lattice.transforms.semantic_segmenter import segment_request, segment_summary`:

```bash
sd 'from lattice\.transforms\.semantic_segmenter import' 'from lattice.core.segmentation import' $(rg -l "from lattice.transforms.semantic_segmenter import")
```

After all renames: `uv run ruff check src/ tests/` — fix any remaining unresolved imports manually.

### 3.5 Delete `core/compiler.py` and inline its body

`core/compiler.py` defines a `PromptCompiler` class with a `compile(request, context)` method that does exactly:

```python
ir = build_ir(request)
ir = normalize_ir(ir)
self._store_ir_metadata(request, ir)
return ir
```

and a `serialize(ir)` method that calls `serialize_ir_to_text(ir)`.

The five call sites are (per audit):

1. `core/ir_transform.py` (now `ir/transform.py`) — calls `get_compiler()` and `compiler.compile(request, context)`. **Action**: replace with direct calls:
   ```python
   from lattice.ir.builder import build_ir
   from lattice.ir.normalizer import normalize_ir
   ...
   ir = build_ir(request)
   ir = normalize_ir(ir)
   ```
2. `optimizer/ir_native_optimizer.py` (now `ir/native_optimizer.py`) — uses `get_compiler()` lazily inside `_get_ir()`. **Action**: same direct calls.
3. Any other site using `compile()` — same treatment.
4. Any site using `serialize()` — replace with `from lattice.ir.serializer import serialize_ir_to_text`.

The `_store_ir_metadata` helper inside `compiler.py` is a 5-line context-state writer. **Action**: move it into `ir/builder.py` as a private `_store_ir_metadata(request, ir)` function called at the end of `build_ir`. Now `build_ir` itself stores its result on the request's metadata — no separate compiler step needed.

Final delete:

```bash
git rm src/lattice/core/compiler.py
```

Verify nothing references it:

```bash
rg "lattice.core.compiler|PromptCompiler|get_compiler" src/ tests/ benchmarks/ scripts/
```

Should return zero matches.

### 3.6 Write `src/lattice/ir/__init__.py`

Make it the single public entry point to all IR functionality. Re-export the symbols every consumer needs:

```
__all__ = [
    # types (v1, mutable)
    "SectionType", "SpanRole", "Span", "Section", "PromptIR",
    # primitives (v2, immutable)
    "SpanV2", "SectionV2", "PromptIRV2",
    "Candidate", "CandidateGraph", "CandidateScore",
    "CachePlan", "TransportPlan", "ExecutionPlan", "ExecutionNode",
    "prompt_ir_v2_from_legacy", "prompt_ir_from_v2",
    "freeze_dict", "thaw_dict",
    # construction & rendering
    "build_ir", "normalize_ir", "serialize_ir_to_text",
    "is_repeated_template",
    "normalize_json_sections", "normalize_table_sections", "normalize_log_sections",
    "lift_constraints", "extract_causal_chains",
    # transform protocol
    "IRTransform", "LegacyRequestTransformAdapter",
    "CandidateScorer", "CandidateSearch",
    # native optimiser base
    "IRNativeOptimizer", "PromptIrLoader",
    # validation & quality
    "ValidationResult",
    "validate_candidate", "validate_request_candidate", "validate_beam_candidate",
    "QualityEstimate",
    "estimate_quality", "estimate_cache_gain", "estimate_transport_gain", "estimate_semantic_risk",
    # semantic graph
    "SemanticSpan", "SemanticEdge", "SemanticImportanceGraph",
]
```

Each symbol is imported from its specific submodule. The pattern:

```python
from lattice.ir.types import SectionType, SpanRole, Span, Section, PromptIR
from lattice.ir.primitives import (
    SpanV2, SectionV2, PromptIRV2,
    Candidate, CandidateGraph, CandidateScore,
    CachePlan, TransportPlan, ExecutionPlan, ExecutionNode,
    prompt_ir_v2_from_legacy, prompt_ir_from_v2,
    freeze_dict, thaw_dict,
)
from lattice.ir.builder import build_ir, is_repeated_template
from lattice.ir.normalizer import (
    normalize_ir,
    normalize_json_sections, normalize_table_sections, normalize_log_sections,
    lift_constraints, extract_causal_chains,
)
from lattice.ir.serializer import serialize_ir_to_text
from lattice.ir.transform import (
    IRTransform, LegacyRequestTransformAdapter,
    CandidateScorer, CandidateSearch,
)
from lattice.ir.native_optimizer import IRNativeOptimizer, PromptIrLoader
from lattice.ir.validation import (
    ValidationResult,
    validate_candidate, validate_request_candidate, validate_beam_candidate,
)
from lattice.ir.quality import (
    QualityEstimate,
    estimate_quality, estimate_cache_gain, estimate_transport_gain, estimate_semantic_risk,
)
from lattice.ir.semantic_graph import SemanticSpan, SemanticEdge, SemanticImportanceGraph
```

After this, consumers can write `from lattice.ir import build_ir, IRTransform, validate_candidate` — one import, all IR concerns.

### 3.7 Trim `core/__init__.py`

Remove IR-related re-exports:

```python
# DELETE these lines:
from lattice.core.pipeline import CompressorPipeline, ReversibleSyncTransform   # moves in Phase 2
# (IR re-exports were never in core/__init__.py per audit; nothing to delete here, but verify)
```

Add back what's still in `core/`:

```python
from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.errors import (
    ConfigurationError, LatticeError, ProviderError, ProviderTimeoutError,
    RequestTooLargeError, SessionError, SessionExpiredError,
    SessionNotFoundError, SessionStoreError,
    TransformError, TransformNotFoundError, ValidationError,
)
from lattice.core.result import Err, Ok, Result, is_err, is_ok, unwrap, unwrap_err
from lattice.core.segmentation import SegmentKind, SemanticSegment, segment_request, segment_summary
# Re-export Request/Response/etc. from their CURRENT location for Phase 1; Phase 2 will move them.
from lattice.core.transport import Message, Request, Response, Role, SyncTransform, Transform
```

This keeps backward compatibility for `from lattice.core import Request` even though `Request` will physically move in Phase 2 — the re-export here is the buffer.

### 3.8 Update `src/lattice/__init__.py`

Add public re-exports so the v0.x → v1.0 user-facing Python API works without import path changes:

```python
from lattice._version import __version__
from lattice.client import LatticeClient, CompressResult
from lattice.sdk.proxy_client import LatticeProxyClient
from lattice.sdk.wrappers import wrap_openai_client
# IR types accessible at top level for power users:
from lattice.ir import PromptIR, PromptIRV2, build_ir, serialize_ir_to_text

__all__ = [
    "__version__",
    "LatticeClient", "LatticeProxyClient", "CompressResult", "wrap_openai_client",
    "PromptIR", "PromptIRV2", "build_ir", "serialize_ir_to_text",
]
```

### 3.9 Per-file disposition table

| File (after move) | LoC | Status | Notes |
|---|---|---|---|
| `ir/types.py` | 157 | KEEP | Pure dataclasses; no logic changes |
| `ir/builder.py` | ~790 | MODIFY | Absorb the 5-line `_store_ir_metadata` from deleted compiler; ensure `build_ir` calls it at end |
| `ir/normalizer.py` | 388 | KEEP | Pure post-processing |
| `ir/serializer.py` | 280 | KEEP | Pure serialisation |
| `ir/transform.py` | 310 | MODIFY | Drop `get_compiler()` dependency; call `build_ir`/`normalize_ir` directly |
| `ir/primitives.py` | 583 | KEEP | Immutable v2 IR types; no logic changes |
| `ir/semantic_graph.py` | 125 | KEEP | Pure dataclass module |
| `ir/native_optimizer.py` | 240 | MODIFY | Same compiler-removal change as `ir/transform.py` |
| `ir/validation.py` | 283 | KEEP | Centralised rollback checks |
| `ir/quality.py` | 638 | KEEP | 8-component quality estimator |
| `ir/__init__.py` | NEW (~50) | CREATE | Re-export everything per 3.6 |
| `core/segmentation.py` | 267 | KEEP (moved) | No code change; was `transforms/semantic_segmenter.py` |
| `core/compiler.py` | — | **DELETED** | Logic inlined |

---

## 4. Tests

### 4.1 Existing tests

Most IR unit tests live at `tests/unit/test_ir.py`, `tests/unit/test_ir_builder.py`, `tests/unit/test_ir_normalizer.py`, etc. They `from lattice.core.ir import ...`. After the `sd` step in §3.4 they import from `lattice.ir.types` instead — should pass without other changes.

Move the test files to mirror new layout (optional in Phase 1 — Phase 10 does the full reshape):

```bash
mkdir -p tests/unit/ir
git mv tests/unit/test_ir*.py tests/unit/ir/
git mv tests/unit/test_primitives.py tests/unit/ir/  # if it exists
git mv tests/unit/test_semantic_graph.py tests/unit/ir/  # if it exists
```

### 4.2 New tests

```
tests/unit/ir/test_builder_stores_metadata.py
```

Verifies that `build_ir(request)` populates `request.metadata["ir"]` correctly — i.e. the compiler-deletion didn't lose `_store_ir_metadata` semantics. Test cases:

1. `build_ir(simple_request)` → returns `PromptIR` with non-empty sections.
2. `request.metadata` after `build_ir` contains `"ir_summary"` or whichever key compiler used to set.
3. `build_ir → normalize_ir → serialize_ir_to_text` round-trips a sample request producing the same string the old `compiler.compile().serialize()` path produced. (Snapshot test against a stored fixture.)

### 4.3 Contract test stays green

`tests/contract/test_python_api_contract.py` must still pass:

```python
from lattice.ir import build_ir, PromptIR, PromptIRV2, IRTransform   # NEW path works
from lattice.core import Request, Response                            # OLD path STILL works via re-export
```

If either fails, the migration is incomplete.

---

## 5. Cross-phase coordination

This phase touches `core/transport.py` indirectly — many moved files import `Request` / `Response` from it. **Do not move `core/transport.py` in Phase 1.** Phase 2 owns that move. Phase 1 leaves those imports as-is; the `core/__init__.py` re-export buffer (§3.7) is the bridge.

Phase 1 also leaves `core/runtime_state.py` alone even though `ir/validation.py` and `ir/quality.py` import from it. **Phase 3** moves runtime_state. The imports stay temporarily pointing at `lattice.core.runtime_state`.

This is the only "leave a stale import" allowance in the plan. Phase 3 will fix the two remaining `from lattice.core.runtime_state` references in `ir/`.

---

## 6. Acceptance criteria — phase done when all are true

- [ ] `src/lattice/ir/` exists with 11 files matching the table in §3.9.
- [ ] `src/lattice/core/compiler.py` does not exist.
- [ ] `src/lattice/optimizer/{ir_native_optimizer,validation,quality_estimator}.py` do not exist (moved to `ir/`).
- [ ] `src/lattice/transforms/semantic_segmenter.py` does not exist (moved to `core/segmentation.py`).
- [ ] `rg "from lattice.core.ir" src/ tests/ benchmarks/` returns 0 matches.
- [ ] `rg "from lattice.core.primitives" src/ tests/ benchmarks/` returns 0 matches.
- [ ] `rg "from lattice.core.compiler" src/ tests/ benchmarks/` returns 0 matches.
- [ ] `rg "from lattice.optimizer.ir_native_optimizer|from lattice.optimizer.validation|from lattice.optimizer.quality_estimator" src/ tests/ benchmarks/` returns 0 matches.
- [ ] `rg "from lattice.transforms.semantic_segmenter" src/ tests/ benchmarks/` returns 0 matches.
- [ ] `from lattice.ir import build_ir, PromptIR, PromptIRV2, IRTransform, CandidateSearch, validate_candidate, QualityEstimate` works.
- [ ] `from lattice.core import Request, Response, Message, LatticeConfig, Result, TransformError` still works (re-export buffer in place).
- [ ] `uv run ruff check src/ tests/` clean.
- [ ] `uv run mypy src/lattice/` clean.
- [ ] `uv run pytest tests/ -q` passes.
- [ ] `uv run pytest tests/contract/ -q` passes.
- [ ] `uv run python benchmarks/evals/cli.py --suite all --providers ollama-cloud --provider-model ollama-cloud=kimi-k2.6:cloud --iterations 1 --warmup 0 --provider-warmup 0 --output-json benchmarks/results/phase-1.json` runs to completion.
- [ ] `python scripts/compare_benchmarks.py benchmarks/results/phase-0-baseline.json benchmarks/results/phase-1.json --tolerance-pct 2` exits 0.

---

## 7. Risks specific to this phase

| Risk | Mitigation |
|---|---|
| Hidden import cycle between `ir/builder.py` and `transport/types.py` (Phase 2 anticipation) | Phase 1 imports `Request` from `lattice.core.transport` — same path as today. Cycle does not change. |
| The 5-line `_store_ir_metadata` change in `ir/builder.py` subtly changes when metadata is written, breaking a transform that read it earlier in the pipeline | The new test in §4.2 verifies the metadata is present after `build_ir` returns. The same call site that used to call `compiler.compile()` will now call `build_ir()` and see the metadata identically. |
| `ir/__init__.py` accidentally imports something not yet moved (e.g. from `runtime_state.py`), causing import-time failures | Phase 1's `__init__.py` only re-exports symbols defined in the 10 moved files. It does not pull in anything from `core/runtime_state` directly. Submodules may still import runtime_state lazily. |
| `core/__init__.py` re-export of `Request`/`Response`/etc. (anticipating Phase 2) breaks if someone deletes those symbols later | Phase 2 keeps the same re-export but updates the underlying path. The `__init__.py` signature is unchanged from a consumer's perspective. |
| `tests/unit/test_compiler.py` exists and depends on `PromptCompiler` | Delete it. Replace with `test_builder_stores_metadata.py` from §4.2. Compiler is gone. |
| `benchmarks/` imports `PromptCompiler` somewhere | The audit confirms `compiler.py` has 5 importers, all in `src/lattice/`. Verify with `rg "PromptCompiler" benchmarks/` — should be empty. |

---

## 8. PR shape

This phase ships as **one PR**:

```
refactor(ir): collapse 8 IR files into lattice.ir package; delete compiler.py [Phase 1]

- Move core/ir*.py, primitives.py, semantic_graph.py → ir/
- Move optimizer/ir_native_optimizer.py, validation.py, quality_estimator.py → ir/
- Move transforms/semantic_segmenter.py → core/segmentation.py
- Delete core/compiler.py (3-line wrapper; inlined into ir/builder.py)
- Add ir/__init__.py as single public entry
- Rewrite imports across src/ tests/ benchmarks/
- core/__init__.py still re-exports Request/Response (Phase 2 owns those moves)

Net: -1 file (core/compiler.py), +1 package (ir/), -1 directory shrunk (optimizer/ from 11 → 6).
All 1600+ tests green. Contract tests green. Benchmarks ±2% of Phase 0 baseline.
```
