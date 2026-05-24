# LATTICE — Development Guide for AI Agents

## Setup

```bash
git clone https://github.com/Harsh-Daga/lattice
cd lattice
uv sync

# Run proxy
uv run python -m lattice.proxy.server --port 8787

# Run tests
uv run pytest tests/ -q

# Run benchmarks (local, no keys needed)
uv run python benchmarks/evals/cli.py --suite feature
```

## Architecture

LATTICE is a **unified optimization + transport system** with one canonical runtime (Phases 0–4 complete on `refactor/phase-4-planner-collapse`):

```
Request → content_profiler → UnifiedPlanner → ExecutionPlan → Pipeline.compress/process → Provider
```

The authoritative architecture document is [`docs/architecture/runtime_v2.md`](docs/architecture/runtime_v2.md).  
Refactor progress: [`docs/refactor/STATUS.md`](docs/refactor/STATUS.md).

### Key Modules

| Directory | Responsibility |
|-----------|---------------|
| `core/` | Leaf primitives only: `config`, `context`, `errors`, `result`, `segmentation`, `transform_registry`, session/store/metrics/telemetry (Phase 9 moves) |
| `ir/` | PromptIRV2, builder, primitives, native optimizer base, quality, validation |
| `planner/` | **UnifiedPlanner** (sole scheduler), task classifier, execution plan builder, runtime state bridges |
| `pipeline/` | `Pipeline`, safety gates, policy, guardrails, MILV, representation_optimizer (beam search) |
| `transforms/` | Individual transforms — IR-native via `optimize(ir, ...)`; orchestrators in `transforms/optimizers/` |
| `transforms/optimizers/` | Per-domain optimizer orchestrators (`ir_structure`, reference, tool, diagnostic, context) |
| `runtime/` | **TierClassifier** (workload complexity — not a provider router) |
| `transport/` | Request/Response types, serialization, delta wire, congestion |
| `protocol/` | Prefix canonicalization, cache planners, binary framing, manifest |
| `providers/` | Per-provider adapters, transport, **credentials** |
| `proxy/` | FastAPI server with OpenAI-compatible endpoints |
| `gateway/` | HTTP compatibility layer, routing headers |

**Deleted / moved (do not import):** `core/scheduler.py`, `core/optimizer_scheduler.py`, `core/unified_planner.py`, `optimizer/` package, `runtime/router.py`, text `StructureOptimizer`.

## Code Conventions

- **Result[T,E] monad** for error handling: `Ok(value)` or `Err(error)`
- **ReversibleSyncTransform** base class for all transforms (`lattice.pipeline.base`)
- **Immutable PromptIRV2** — transforms return new instances via `.with_sections()` / `.with_spans()` / `.with_text()`
- **Explicit allowlists** — `Pipeline._IR_NATIVE_TRANSFORMS` defines which transforms run natively
- **Single scheduler** — `UnifiedPlanner.plan()` is the only scheduling decision maker
- **mypy strict**, **ruff** for linting; run both before commits

## Adding a Transform

1. Extend `ReversibleSyncTransform` with `name` and `priority`
2. Implement `optimize(PromptIRV2, Request, TransformContext) → Result[PromptIRV2, TransformError]` (canonical path)
3. Register in `core/transform_registry.py` and `pipeline/runner.py` `PipelineTransformRegistry._FACTORIES`

## Testing

- Unit tests: `tests/unit/` — **1706 passed**, 196 skipped, 1903 collected (Phase 11 full reorg pending)
- Integration tests: `tests/integration/` — proxy sessions, Redis, IR optimizer E2E
- E2E tests: `tests/e2e/` — agent wrappers, full pipeline
- Contract tests: `tests/contract/` — **27 passed**
- Run with `uv run pytest tests/ -q`

## Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `LATTICE_PROVIDER_BASE_URL` | Default upstream provider base URL |
| `LATTICE_PROVIDER_BASE_URLS` | JSON dict of `{provider: url}` overrides |
| `OPENAI_API_KEY` | Used for OpenAI, Azure, and OpenAI-compatible providers |
| `ANTHROPIC_API_KEY` | Anthropic provider |
| `OLLAMA_CLOUD_API_KEY` | Canonical benchmark gate (runtime only — never commit) |

## Benchmark CLI

```bash
uv run python benchmarks/evals/cli.py --suite all \
  --providers ollama \
  --provider-model ollama=llama3.2 \
  --iterations 3 --warmup 1
```

Suites: `all`, `feature`, `feature-matrix`, `provider`, `protocol`, `transport`, `integration`, `capability`, `replay`, `replay-governance`, `tacc`, `control`.

## Latest CI (Phase 4)

| Metric | Value |
|--------|-------|
| Tests passed | **1706/1706** executable (+ 196 skipped, 1903 collected) |
| Contract tests | **27/27** |
| ruff / format / mypy | **0 errors** |

Canonical benchmark vs `phase-0-baseline.json` → `phase-4.json` (±2%): run on CI with `OLLAMA_CLOUD_API_KEY`.
