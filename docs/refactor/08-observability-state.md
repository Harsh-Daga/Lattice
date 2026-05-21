# Phase 8 — Observability, State, Cache, Safety, Utils

> **Goal.** Populate the empty `observability/` directory (renamed `telemetry/`) with the six telemetry-related modules currently scattered through `core/`. Move `core/session.py` and `core/store.py` to `state/` (alongside `state/segment_store.py`) so all stateful persistence lives in one place. Move `core/semantic_cache.py` to a new `cache/` top-level domain. Move `utils/validation.py` to a new `safety/` domain. Move `utils/streaming_sketches.py` into `telemetry/`. Shrink `utils/` to just `token_count.py`. The result: `core/` becomes a pure leaf — six files — exactly what the dependency direction in REFACTOR_PLAN.md §5 mandates.
>
> **Outcome.** `from lattice.telemetry import MetricsCollector, DowngradeTelemetry, AgentStatsCollector, CostEstimator, MaintenanceCoordinator` works. `from lattice.state import Session, SessionManager, SessionStore, MemorySessionStore, RedisSessionStore, SegmentStore` works. `from lattice.cache import SemanticCache, ContentClass` works. `from lattice.safety import compute_risk_score, SemanticRiskScore` works. `core/` has only `config.py`, `context.py`, `errors.py`, `result.py`, `segmentation.py`, and (kept) `tunnel_sidecar.py` already moved in Phase 7.
>
> **Estimated effort.** 1.5 days.

---

## 1. Why this phase exists

After Phases 1–7, `core/` still holds ten files that don't belong there:

| File | LoC | Belongs in |
|---|---|---|
| `metrics.py` | 307 | `telemetry/` |
| `telemetry.py` | 203 | `telemetry/` |
| `agent_stats.py` | 301 | `telemetry/` |
| `cost_estimator.py` | 558 | `telemetry/` |
| `maintenance.py` | 159 | `telemetry/` |
| `session.py` | 516 | `state/` (sits alongside `segment_store.py`) |
| `store.py` | 196 | `state/` |
| `semantic_cache.py` | 1014 | new `cache/` (it's a top-level concern, not "core") |
| `utils/validation.py` | 463 | new `safety/` |
| `utils/streaming_sketches.py` | 179 | `telemetry/` |
| `utils/patterns.py` | 126 | (moved to `transforms/patterns.py` in Phase 4) |

Each move is mechanical (rename + import rewrites) but the *combination* is what makes the new dependency direction enforceable: `core/` shrinks to leaf primitives only.

The `observability/` placeholder created earlier in the codebase is **renamed to `telemetry/`** in Phase 8 — `telemetry` better matches the contents (metrics, downgrade taxonomy, agent stats, cost, maintenance, sketches). If `observability/` is already a directory in the repo, `git mv` it.

The semantic cache deserves its own top-level `cache/` domain because:
1. It's 1014 LoC and growing.
2. It has both in-memory and Redis backends, plus its own protocol concerns (fingerprinting, TTL, LRU, generation).
3. It's a single-concern domain, not "core".

---

## 2. Files touched

### 2.1 Moved

| Current path | New path |
|---|---|
| `src/lattice/observability/` (empty placeholder) | `src/lattice/telemetry/` (renamed if it exists; else just `mkdir`) |
| `src/lattice/core/metrics.py` | `src/lattice/telemetry/metrics.py` |
| `src/lattice/core/telemetry.py` | `src/lattice/telemetry/downgrade.py` *(renamed to avoid the obvious circular name)* |
| `src/lattice/core/agent_stats.py` | `src/lattice/telemetry/agent_stats.py` |
| `src/lattice/core/cost_estimator.py` | `src/lattice/telemetry/cost_estimator.py` |
| `src/lattice/core/maintenance.py` | `src/lattice/telemetry/maintenance.py` |
| `src/lattice/utils/streaming_sketches.py` | `src/lattice/telemetry/streaming_sketches.py` |
| `src/lattice/core/session.py` | `src/lattice/state/session.py` |
| `src/lattice/core/store.py` | `src/lattice/state/store.py` |
| `src/lattice/core/semantic_cache.py` | `src/lattice/cache/semantic.py` |
| `src/lattice/utils/validation.py` | `src/lattice/safety/risk_scoring.py` |

### 2.2 Created

```
src/lattice/telemetry/__init__.py     # public re-exports
src/lattice/cache/__init__.py
src/lattice/safety/__init__.py
src/lattice/state/__init__.py         # already exists; rewrite for the additions
```

### 2.3 Deleted

If `src/lattice/observability/` exists as an empty placeholder directory:

```bash
rmdir src/lattice/observability    # if it exists and is empty
```

`src/lattice/utils/` shrinks to just `token_count.py` + `__init__.py` (the rest have moved).

### 2.4 Modified

- All `from lattice.core.{metrics,telemetry,agent_stats,cost_estimator,maintenance,session,store,semantic_cache}` import sites — rewritten.
- All `from lattice.utils.{validation,streaming_sketches}` import sites — rewritten.
- `core/__init__.py` — drop the now-moved re-exports; final shape is leaf-only.
- `__init__.py` (top-level) — add re-exports for top-level user-facing names from telemetry/cache/safety/state where appropriate.
- The two `telemetry.py` files (one in core/, one already in transport/`congestion.py`-adjacent contexts) — the rename to `downgrade.py` disambiguates.

---

## 3. Step-by-step

### 3.1 Rename `observability/` → `telemetry/` (or create)

```bash
if [ -d src/lattice/observability ]; then
    git mv src/lattice/observability src/lattice/telemetry
else
    mkdir -p src/lattice/telemetry
fi
```

### 3.2 Move telemetry files

```bash
git mv src/lattice/core/metrics.py        src/lattice/telemetry/metrics.py
git mv src/lattice/core/telemetry.py      src/lattice/telemetry/downgrade.py        # rename
git mv src/lattice/core/agent_stats.py    src/lattice/telemetry/agent_stats.py
git mv src/lattice/core/cost_estimator.py src/lattice/telemetry/cost_estimator.py
git mv src/lattice/core/maintenance.py    src/lattice/telemetry/maintenance.py
git mv src/lattice/utils/streaming_sketches.py src/lattice/telemetry/streaming_sketches.py
```

### 3.3 Move state files

```bash
git mv src/lattice/core/session.py src/lattice/state/session.py
git mv src/lattice/core/store.py   src/lattice/state/store.py
```

### 3.4 Move semantic cache to its own top-level domain

```bash
mkdir -p src/lattice/cache
git mv src/lattice/core/semantic_cache.py src/lattice/cache/semantic.py
```

### 3.5 Move validation to safety/

```bash
mkdir -p src/lattice/safety
git mv src/lattice/utils/validation.py src/lattice/safety/risk_scoring.py
```

### 3.6 Rewrite all imports

```bash
# Telemetry
sd 'from lattice\.core\.metrics import'        'from lattice.telemetry.metrics import'        $(rg -l "from lattice.core.metrics import")
sd 'from lattice\.core\.telemetry import'      'from lattice.telemetry.downgrade import'      $(rg -l "from lattice.core.telemetry import")
sd 'from lattice\.core\.agent_stats import'    'from lattice.telemetry.agent_stats import'    $(rg -l "from lattice.core.agent_stats import")
sd 'from lattice\.core\.cost_estimator import' 'from lattice.telemetry.cost_estimator import' $(rg -l "from lattice.core.cost_estimator import")
sd 'from lattice\.core\.maintenance import'    'from lattice.telemetry.maintenance import'    $(rg -l "from lattice.core.maintenance import")
sd 'from lattice\.utils\.streaming_sketches import' 'from lattice.telemetry.streaming_sketches import' $(rg -l "from lattice.utils.streaming_sketches import")

# State
sd 'from lattice\.core\.session import' 'from lattice.state.session import' $(rg -l "from lattice.core.session import")
sd 'from lattice\.core\.store import'   'from lattice.state.store import'   $(rg -l "from lattice.core.store import")

# Cache
sd 'from lattice\.core\.semantic_cache import' 'from lattice.cache.semantic import' $(rg -l "from lattice.core.semantic_cache import")

# Safety
sd 'from lattice\.utils\.validation import' 'from lattice.safety.risk_scoring import' $(rg -l "from lattice.utils.validation import")
```

### 3.7 Write `telemetry/__init__.py`

```python
"""LATTICE telemetry — metrics, downgrade taxonomy, cost estimation, agent stats, maintenance, sketches.

This is the single source of truth for everything observable about a running LATTICE proxy.

Use:
    from lattice.telemetry import (
        MetricsCollector, LatencyTracker,
        DowngradeTelemetry, DowngradeCategory, TransportOutcome,
        AgentStatsCollector, AgentMetrics,
        CostEstimator, CostEstimate, ModelPricing,
        MaintenanceCoordinator, MaintenanceResult,
        CountMinSketch, HyperLogLog,
    )
"""

from lattice.telemetry.metrics import (
    MetricsCollector, LatencyTracker,
)
from lattice.telemetry.downgrade import (
    DowngradeCategory, DowngradeTelemetry, TransportOutcome,
)
from lattice.telemetry.agent_stats import (
    AgentMetrics, AgentStatsCollector, identify_agent,
)
from lattice.telemetry.cost_estimator import (
    ModelPricing, CostEstimate, CostEstimator,
    normalize_usage, extract_cached_tokens, format_cost_usd,
)
from lattice.telemetry.maintenance import (
    MaintenanceCoordinator, MaintenanceResult,
)
from lattice.telemetry.streaming_sketches import (
    CountMinSketch, HyperLogLog,
)

__all__ = [
    "MetricsCollector", "LatencyTracker",
    "DowngradeCategory", "DowngradeTelemetry", "TransportOutcome",
    "AgentMetrics", "AgentStatsCollector", "identify_agent",
    "ModelPricing", "CostEstimate", "CostEstimator",
    "normalize_usage", "extract_cached_tokens", "format_cost_usd",
    "MaintenanceCoordinator", "MaintenanceResult",
    "CountMinSketch", "HyperLogLog",
]
```

### 3.8 Rewrite `state/__init__.py`

```python
"""LATTICE state persistence: sessions + cross-session segment dedup."""

from lattice.state.session import (
    Session, SessionStore, SessionManager,
    MemorySessionStore,
)
from lattice.state.store import RedisSessionStore
from lattice.state.segment_store import SegmentStore, SegmentRecord   # unchanged

__all__ = [
    "Session", "SessionStore", "SessionManager",
    "MemorySessionStore", "RedisSessionStore",
    "SegmentStore", "SegmentRecord",
]
```

### 3.9 Write `cache/__init__.py`

```python
"""LATTICE semantic cache: exact-hash + approximate fingerprint.

Backends:
    InMemoryCacheBackend (default)
    RedisCacheBackend    (optional, requires `lattice-transport[redis]`)
"""

from lattice.cache.semantic import (
    SemanticCache, ContentClass, CachedResponse,
    CacheBackend, InMemoryCacheBackend, RedisCacheBackend,
    compute_cache_key, compute_semantic_fingerprint,
    assemble_cached_response, generate_sse_chunks,
)

__all__ = [
    "SemanticCache", "ContentClass", "CachedResponse",
    "CacheBackend", "InMemoryCacheBackend", "RedisCacheBackend",
    "compute_cache_key", "compute_semantic_fingerprint",
    "assemble_cached_response", "generate_sse_chunks",
]
```

### 3.10 Write `safety/__init__.py`

```python
"""LATTICE safety: semantic risk scoring + transform gating helpers."""

from lattice.safety.risk_scoring import (
    SemanticRiskScore,
    compute_risk_score,
    RiskLevel,
    is_high_risk,
)

__all__ = [
    "SemanticRiskScore",
    "compute_risk_score",
    "RiskLevel",
    "is_high_risk",
]
```

### 3.11 Update `core/__init__.py` to leaf-only

After all the moves, `core/` contains: `config.py`, `context.py`, `errors.py`, `result.py`, `segmentation.py`. Six files (counting `__init__.py`).

```python
"""Core leaf primitives — config, context, errors, result, segmentation.

Anything higher-level (planning, IR, transforms, providers, telemetry, state,
cache, safety, pipeline, transport) lives in its own package. core/ depends
on nothing from those packages — it is a true leaf.
"""

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.errors import (
    ConfigurationError, LatticeError, ProviderError, ProviderTimeoutError,
    RequestTooLargeError,
    SessionError, SessionExpiredError, SessionNotFoundError, SessionStoreError,
    TransformError, TransformNotFoundError,
    ValidationError,
)
from lattice.core.result import (
    Err, Ok, Result,
    is_err, is_ok, unwrap, unwrap_err,
)
from lattice.core.segmentation import (
    SegmentKind, SemanticSegment, segment_request, segment_summary,
)

# Re-export wire types from their canonical location (Phase 2 made these the
# permanent home in transport/types.py; we keep the back-compat re-export here
# for one release, removed in v1.1).
from lattice.transport.types import (
    Message, Request, Response, Role,
    Transform, SyncTransform,
)
from lattice.pipeline.runner import ReversibleSyncTransform   # protocol, used everywhere

__all__ = [
    "LatticeConfig", "TransformContext",
    "ConfigurationError", "LatticeError",
    "ProviderError", "ProviderTimeoutError", "RequestTooLargeError",
    "SessionError", "SessionExpiredError", "SessionNotFoundError", "SessionStoreError",
    "TransformError", "TransformNotFoundError",
    "ValidationError",
    "Result", "Ok", "Err",
    "is_ok", "is_err", "unwrap", "unwrap_err",
    "SegmentKind", "SemanticSegment", "segment_request", "segment_summary",
    "Message", "Request", "Response", "Role",
    "Transform", "SyncTransform", "ReversibleSyncTransform",
]
```

### 3.12 Update `utils/__init__.py`

```python
"""Truly general utilities. Most former contents moved to their domain:
    patterns       → transforms/patterns
    validation     → safety/risk_scoring
    streaming_sketches → telemetry/streaming_sketches
"""

from lattice.utils.token_count import (
    count_tokens, approximate_tokens, get_encoder,
)

__all__ = ["count_tokens", "approximate_tokens", "get_encoder"]
```

If `utils/` ends up with only `__init__.py` + `token_count.py`, that's the final state.

### 3.13 Top-level `__init__.py` additions

For convenience, add a few telemetry/cache/state types to the top level:

```python
from lattice.telemetry import MetricsCollector, DowngradeCategory
from lattice.state import Session, SessionManager, SegmentStore
from lattice.cache import SemanticCache
from lattice.safety import SemanticRiskScore, compute_risk_score

__all__ = [
    # existing
    ...,
    # new top-level conveniences (NOT required for v1.0.0 contract; "advanced" API):
    "MetricsCollector", "DowngradeCategory",
    "Session", "SessionManager", "SegmentStore",
    "SemanticCache",
    "SemanticRiskScore", "compute_risk_score",
]
```

The Phase 0 contract test asserts these names are importable from `lattice`.

### 3.14 Verify

```bash
uv run ruff check src/ tests/
uv run mypy src/lattice/
rg "from lattice.core.metrics|from lattice.core.telemetry|from lattice.core.agent_stats|from lattice.core.cost_estimator|from lattice.core.maintenance" src/ tests/ benchmarks/   # 0
rg "from lattice.core.session|from lattice.core.store|from lattice.core.semantic_cache" src/ tests/ benchmarks/   # 0
rg "from lattice.utils.validation|from lattice.utils.streaming_sketches" src/ tests/ benchmarks/   # 0
uv run pytest tests/ -q
uv run pytest tests/contract/ -q

uv run python benchmarks/evals/cli.py --suite all --providers ollama-cloud --provider-model ollama-cloud=kimi-k2.6:cloud --iterations 1 --warmup 0 --provider-warmup 0 --output-json benchmarks/results/phase-8.json
python scripts/compare_benchmarks.py benchmarks/results/phase-0-baseline.json benchmarks/results/phase-8.json --tolerance-pct 2
```

---

## 4. Per-file disposition

| File | LoC | Action |
|---|---|---|
| `telemetry/__init__.py` | NEW (~30) | CREATE |
| `telemetry/metrics.py` | 307 | MOVED from `core/metrics.py` |
| `telemetry/downgrade.py` | 203 | MOVED + RENAMED from `core/telemetry.py` |
| `telemetry/agent_stats.py` | 301 | MOVED |
| `telemetry/cost_estimator.py` | 558 | MOVED |
| `telemetry/maintenance.py` | 159 | MOVED |
| `telemetry/streaming_sketches.py` | 179 | MOVED from `utils/` |
| `state/__init__.py` | NEW (~20) | CREATE |
| `state/session.py` | 516 | MOVED from `core/session.py` |
| `state/store.py` | 196 | MOVED from `core/store.py` |
| `state/segment_store.py` | 334 | UNCHANGED (already in state/) |
| `cache/__init__.py` | NEW (~20) | CREATE |
| `cache/semantic.py` | 1014 | MOVED from `core/semantic_cache.py` |
| `safety/__init__.py` | NEW (~15) | CREATE |
| `safety/risk_scoring.py` | 463 | MOVED from `utils/validation.py` |
| `core/__init__.py` | small | REWRITE (leaf-only) |
| `core/config.py` | 505 | UNCHANGED |
| `core/context.py` | 160 | UNCHANGED |
| `core/errors.py` | 264 | UNCHANGED |
| `core/result.py` | 291 | UNCHANGED |
| `core/segmentation.py` | 267 | UNCHANGED |
| `utils/__init__.py` | small | REWRITE (token_count only) |
| `utils/token_count.py` | 276 | UNCHANGED |

After Phase 8: `core/` is exactly 6 files (incl. `__init__.py`). `utils/` is exactly 2 files. The package shape matches FINAL_LAYOUT.md.

---

## 5. Symbol migration table

| Old | New |
|---|---|
| `lattice.core.metrics.MetricsCollector` | `lattice.telemetry.MetricsCollector` |
| `lattice.core.metrics.LatencyTracker` | `lattice.telemetry.LatencyTracker` |
| `lattice.core.telemetry.DowngradeCategory` | `lattice.telemetry.DowngradeCategory` |
| `lattice.core.telemetry.DowngradeTelemetry` | `lattice.telemetry.DowngradeTelemetry` |
| `lattice.core.telemetry.TransportOutcome` | `lattice.telemetry.TransportOutcome` |
| `lattice.core.agent_stats.AgentStatsCollector` | `lattice.telemetry.AgentStatsCollector` |
| `lattice.core.cost_estimator.CostEstimator` | `lattice.telemetry.CostEstimator` |
| `lattice.core.cost_estimator.CostEstimate` | `lattice.telemetry.CostEstimate` |
| `lattice.core.cost_estimator.ModelPricing` | `lattice.telemetry.ModelPricing` |
| `lattice.core.maintenance.MaintenanceCoordinator` | `lattice.telemetry.MaintenanceCoordinator` |
| `lattice.utils.streaming_sketches.CountMinSketch` | `lattice.telemetry.CountMinSketch` |
| `lattice.utils.streaming_sketches.HyperLogLog` | `lattice.telemetry.HyperLogLog` |
| `lattice.core.session.Session` | `lattice.state.Session` |
| `lattice.core.session.SessionManager` | `lattice.state.SessionManager` |
| `lattice.core.session.MemorySessionStore` | `lattice.state.MemorySessionStore` |
| `lattice.core.session.SessionStore` | `lattice.state.SessionStore` |
| `lattice.core.store.RedisSessionStore` | `lattice.state.RedisSessionStore` |
| `lattice.core.semantic_cache.SemanticCache` | `lattice.cache.SemanticCache` |
| `lattice.core.semantic_cache.ContentClass` | `lattice.cache.ContentClass` |
| `lattice.core.semantic_cache.CachedResponse` | `lattice.cache.CachedResponse` |
| `lattice.core.semantic_cache.compute_cache_key` | `lattice.cache.compute_cache_key` |
| `lattice.utils.validation.SemanticRiskScore` | `lattice.safety.SemanticRiskScore` |
| `lattice.utils.validation.compute_risk_score` | `lattice.safety.compute_risk_score` |

---

## 6. Tests

### 6.1 Move existing tests

```bash
mkdir -p tests/unit/telemetry tests/unit/state tests/unit/cache tests/unit/safety
git mv tests/unit/test_metrics.py             tests/unit/telemetry/test_metrics.py
git mv tests/unit/test_telemetry.py           tests/unit/telemetry/test_downgrade.py
git mv tests/unit/test_agent_stats.py         tests/unit/telemetry/test_agent_stats.py
git mv tests/unit/test_cost_estimator.py      tests/unit/telemetry/test_cost_estimator.py
git mv tests/unit/test_maintenance.py         tests/unit/telemetry/test_maintenance.py
git mv tests/unit/test_streaming_sketches.py  tests/unit/telemetry/test_streaming_sketches.py

git mv tests/unit/test_session*.py            tests/unit/state/
git mv tests/unit/test_store.py               tests/unit/state/
git mv tests/unit/test_segment_store.py       tests/unit/state/

git mv tests/unit/test_semantic_cache*.py     tests/unit/cache/

git mv tests/unit/test_validation.py          tests/unit/safety/test_risk_scoring.py
```

### 6.2 New tests

**`tests/unit/test_core_is_leaf.py`** — verifies that `core/` doesn't import from any higher domain:

```python
import ast, pathlib

def test_core_has_no_uphill_imports():
    """core/ must be a leaf: it may import from itself, stdlib, third-party, transport.types,
    pipeline.runner (for ReversibleSyncTransform re-export). Nothing else."""
    allowed_prefixes = ("lattice.core.", "lattice.transport.types", "lattice.pipeline.runner")
    core_dir = pathlib.Path("src/lattice/core")
    bad = []
    for py in core_dir.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("lattice."):
                if not any(node.module.startswith(p.rstrip(".")) or node.module == p.rstrip(".") for p in allowed_prefixes):
                    bad.append(f"{py.name}: imports {node.module}")
    assert not bad, "core/ imports from non-leaf paths:\n" + "\n".join(bad)
```

**`tests/unit/test_no_old_paths.py`** — sanity check that no production code still uses the old paths:

```python
import subprocess

OLD_PATHS = [
    "lattice.core.metrics", "lattice.core.telemetry", "lattice.core.agent_stats",
    "lattice.core.cost_estimator", "lattice.core.maintenance",
    "lattice.core.session", "lattice.core.store", "lattice.core.semantic_cache",
    "lattice.utils.validation", "lattice.utils.streaming_sketches", "lattice.utils.patterns",
    "lattice.core.compiler", "lattice.core.pipeline", "lattice.core.pipeline_v2",
    "lattice.core.pipeline_v2_wrapper", "lattice.core.pipeline_factory",
    "lattice.core.policy", "lattice.core.guardrails", "lattice.core.milv",
    "lattice.core.auto_continuation", "lattice.core.batch_accumulator",
    "lattice.core.scheduler", "lattice.core.optimizer_scheduler",
    "lattice.core.unified_planner", "lattice.core.task_classifier",
    "lattice.core.runtime_state", "lattice.core.credentials",
    "lattice.core.transport", "lattice.core.serialization", "lattice.core.delta_wire",
    "lattice.core.ir", "lattice.core.ir_builder", "lattice.core.ir_normalizer",
    "lattice.core.ir_serializer", "lattice.core.ir_transform",
    "lattice.core.primitives", "lattice.core.semantic_graph",
    "lattice.core.transform_registry", "lattice.core.transform_reputation",
    "lattice.core.tunnel_sidecar",
    "lattice.optimizer.ir_native_optimizer", "lattice.optimizer.validation",
    "lattice.optimizer.quality_estimator", "lattice.optimizer.representation_optimizer",
    "lattice.optimizer.structure_optimizer", "lattice.optimizer.ir_structure_optimizer",
    "lattice.optimizer.reference_optimizer", "lattice.optimizer.tool_optimizer",
    "lattice.optimizer.diagnostic_optimizer", "lattice.optimizer.context_optimizer",
    "lattice.transforms.prefix_opt",
    "lattice.transforms.constraint_lifting",
    "lattice.transforms.semantic_segmenter",
    "lattice.transforms.format_conv",
    "lattice.providers.base", "lattice.providers.openai", "lattice.providers.openai_compatible",
    "lattice.providers.anthropic", "lattice.providers.azure", "lattice.providers.bedrock",
    "lattice.providers.gemini", "lattice.providers.ollama",
    "lattice.providers.stall_detector",
    "lattice.runtime.router",
    "lattice.sdk.client",      # deprecated but allowed; warn-only
]

def test_no_old_imports_in_src():
    """Production source code must not import from the old paths.
    Test files MAY (e.g. test_no_old_paths.py). sdk/client.py is allowed (shim)."""
    for path in OLD_PATHS:
        if path == "lattice.sdk.client":
            continue   # deprecated shim still works with warning
        result = subprocess.run(
            ["rg", "-c", f"from {path}|import {path}", "src/lattice/"],
            capture_output=True, text=True,
        )
        # rg -c with 0 matches exits 1; with matches exits 0
        if result.returncode == 0:
            files = result.stdout.strip()
            raise AssertionError(f"OLD PATH STILL USED: {path}\nFiles:\n{files}")
```

This is the single test that proves the entire 12-phase refactor's "no parallel paths" invariant holds.

### 6.3 Contract tests

`tests/contract/test_python_api_contract.py` — extend with the new top-level imports:

```python
def test_new_top_level_exports():
    from lattice import (
        MetricsCollector, DowngradeCategory,
        Session, SessionManager, SegmentStore,
        SemanticCache,
        SemanticRiskScore, compute_risk_score,
    )

def test_new_domain_packages():
    import lattice.telemetry
    import lattice.state
    import lattice.cache
    import lattice.safety
    # Each must export a non-empty __all__
    for mod in (lattice.telemetry, lattice.state, lattice.cache, lattice.safety):
        assert hasattr(mod, "__all__") and len(mod.__all__) > 0
```

---

## 7. Import-rewrite cheatsheet

```bash
# Telemetry (six moves)
sd 'from lattice\.core\.metrics import'        'from lattice.telemetry.metrics import'        $(rg -l "from lattice.core.metrics import")
sd 'from lattice\.core\.telemetry import'      'from lattice.telemetry.downgrade import'      $(rg -l "from lattice.core.telemetry import")
sd 'from lattice\.core\.agent_stats import'    'from lattice.telemetry.agent_stats import'    $(rg -l "from lattice.core.agent_stats import")
sd 'from lattice\.core\.cost_estimator import' 'from lattice.telemetry.cost_estimator import' $(rg -l "from lattice.core.cost_estimator import")
sd 'from lattice\.core\.maintenance import'    'from lattice.telemetry.maintenance import'    $(rg -l "from lattice.core.maintenance import")
sd 'from lattice\.utils\.streaming_sketches import' 'from lattice.telemetry.streaming_sketches import' $(rg -l "from lattice.utils.streaming_sketches import")

# State (two moves)
sd 'from lattice\.core\.session import' 'from lattice.state.session import' $(rg -l "from lattice.core.session import")
sd 'from lattice\.core\.store import'   'from lattice.state.store import'   $(rg -l "from lattice.core.store import")

# Cache (one move)
sd 'from lattice\.core\.semantic_cache import' 'from lattice.cache.semantic import' $(rg -l "from lattice.core.semantic_cache import")

# Safety (one move)
sd 'from lattice\.utils\.validation import' 'from lattice.safety.risk_scoring import' $(rg -l "from lattice.utils.validation import")

# Verification
rg "from lattice.core.(metrics|telemetry|agent_stats|cost_estimator|maintenance|session|store|semantic_cache)" src/ tests/ benchmarks/
rg "from lattice.utils.(validation|streaming_sketches)" src/ tests/ benchmarks/
# Both should return 0 matches
```

---

## 8. Acceptance criteria

- [ ] `src/lattice/observability/` directory does not exist (renamed or deleted).
- [ ] `src/lattice/telemetry/` exists with `__init__.py` + 6 module files.
- [ ] `src/lattice/state/` contains `__init__.py`, `session.py`, `store.py`, `segment_store.py`.
- [ ] `src/lattice/cache/` exists with `__init__.py` + `semantic.py`.
- [ ] `src/lattice/safety/` exists with `__init__.py` + `risk_scoring.py`.
- [ ] `src/lattice/core/` contains exactly: `__init__.py`, `config.py`, `context.py`, `errors.py`, `result.py`, `segmentation.py`. (`tunnel_sidecar.py` already moved in Phase 7.)
- [ ] `src/lattice/utils/` contains exactly: `__init__.py`, `token_count.py`.
- [ ] `from lattice.telemetry import MetricsCollector, LatencyTracker, DowngradeCategory, DowngradeTelemetry, AgentStatsCollector, CostEstimator, MaintenanceCoordinator, CountMinSketch, HyperLogLog` works.
- [ ] `from lattice.state import Session, SessionManager, SessionStore, MemorySessionStore, RedisSessionStore, SegmentStore` works.
- [ ] `from lattice.cache import SemanticCache, ContentClass, CachedResponse, compute_cache_key` works.
- [ ] `from lattice.safety import SemanticRiskScore, compute_risk_score` works.
- [ ] `from lattice import MetricsCollector, Session, SessionManager, SegmentStore, SemanticCache, SemanticRiskScore` works (top-level re-exports).
- [ ] `tests/unit/test_core_is_leaf.py` passes — core has no uphill imports.
- [ ] `tests/unit/test_no_old_paths.py` passes — no production code uses any of the 50+ old import paths.
- [ ] `tests/contract/test_python_api_contract.py` covers all new top-level names.
- [ ] `uv run ruff check src/ tests/` clean.
- [ ] `uv run mypy src/lattice/` clean.
- [ ] `uv run pytest tests/ -q` passes.
- [ ] `uv run pytest tests/contract/ -q` passes.
- [ ] `python scripts/compare_benchmarks.py benchmarks/results/phase-0-baseline.json benchmarks/results/phase-8.json --tolerance-pct 2` exits 0.

---

## 9. Risks specific to this phase

| Risk | Mitigation |
|---|---|
| The Redis test (`tests/integration/test_redis_backend.py`) uses `RedisSessionStore` from `lattice.core.store` — must update | `sd` script in §3.6 catches this. Verify in §3.14. |
| `core/metrics.py` → `telemetry/metrics.py` rename collides with `LATTICE_METRICS_PORT` env var or Prometheus client expectations | The env var stays unchanged; only the Python import path moves. Prometheus output format is unchanged. |
| `core/telemetry.py` → `telemetry/downgrade.py` rename is confusing — both names appear in different contexts | The rename is deliberate: the file's content is about downgrade taxonomy (it has `DowngradeCategory` enum and `DowngradeTelemetry` class). Naming the file `downgrade.py` makes that clear. The *package* is named `telemetry`. Documented in MIGRATION.md: "`core.telemetry` → `telemetry.downgrade`". |
| Phase 8's `tests/unit/test_no_old_paths.py` fails because some test files still use old paths | Test files are explicitly excluded by the rg path filter (`src/lattice/` only). Test code may still use old paths if there's a back-compat re-export — verify. |
| Splitting `core/__init__.py` re-exports breaks an import that depends on the file's exact `__all__` shape | The new `core/__init__.py` (§3.11) still exports every symbol that was previously exported. The `Request`/`Response`/`Transform` re-exports use `lattice.transport.types` as the underlying source — works. |
| `core/semantic_cache.py` had `lattice-transport[redis]` optional dependency wiring; moving to `cache/semantic.py` may break the optional-import path | The Redis import inside `semantic_cache.py` was `try: import redis; except ImportError: redis = None`. Move the same pattern into `cache/semantic.py` unchanged. |
| `core/maintenance.py` registers callbacks that other modules call. After move, `from lattice.core.maintenance import get_coordinator` fails | The function is re-exported via `lattice.telemetry.maintenance.get_coordinator`. Update callers via the `sd` script. |

---

## 10. Rollback plan

Phase 8 is large but mechanical. To rollback:

```bash
git revert <phase-8-merge-commit>
```

This restores all 10 files to their original `core/` / `utils/` locations and reverts all import rewrites.

---

## 11. PR shape

Best as **two PRs** to keep diffs reviewable:

```
refactor(telemetry,utils): populate telemetry/ from core/+utils/; shrink utils/ [Phase 8a]
- Move 6 files: core/metrics, core/telemetry → telemetry/downgrade, core/agent_stats,
  core/cost_estimator, core/maintenance, utils/streaming_sketches → telemetry/
- Create telemetry/__init__.py with full public surface
- ~100 import sites rewritten

refactor(state,cache,safety): move session/store/semantic_cache/validation [Phase 8b]
- Move core/session.py, core/store.py → state/
- Move core/semantic_cache.py → cache/semantic.py
- Move utils/validation.py → safety/risk_scoring.py
- Shrink core/__init__.py to leaf-only
- Add tests/unit/test_core_is_leaf.py and tests/unit/test_no_old_paths.py
- ~60 import sites rewritten

Net: -10 files from core/, +4 new packages (telemetry/, state/ pop, cache/, safety/).
core/ shrinks to 6 files (was 42 in Phase 0).
All 1600+ tests green. Contract tests green. Benchmarks ±2%.
```
