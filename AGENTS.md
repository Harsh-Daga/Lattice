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

LATTICE is a **unified optimization + transport system** with one canonical runtime (Phases 0–8 complete on `refactor/phase-8-integrations`):

```
Request → content_profiler → UnifiedPlanner → ExecutionPlan → Pipeline.compress/process → Provider
```

The authoritative architecture document is [`docs/architecture/runtime_v2.md`](docs/architecture/runtime_v2.md).  
Refactor progress: [`docs/refactor/STATUS.md`](docs/refactor/STATUS.md).

### Key Modules

| Directory | Responsibility |
|-----------|---------------|
| `core/` | Leaf primitives only: `config`, `context`, `errors`, `result`, `segmentation` (6 files incl. `__init__.py`) |
| `telemetry/` | `metrics`, `downgrade`, `agent_stats`, `cost_estimator`, `maintenance`, `streaming_sketches` |
| `state/` | `session`, `store`, `segment_store` — sessions + cross-session segment dedup |
| `cache/` | `semantic.py` — `SemanticCache` (in-memory + optional Redis) |
| `safety/` | `risk_scoring.py` — semantic risk score + transform gating |
| `ir/` | PromptIRV2, builder, primitives, native optimizer base, quality, validation |
| `planner/` | **UnifiedPlanner** (sole scheduler), task classifier, execution plan builder, runtime state bridges |
| `pipeline/` | `Pipeline`, safety gates, policy, guardrails, MILV, representation_optimizer (beam search) |
| `transforms/` | Individual transforms — IR-native via `optimize(ir, ...)`; orchestrators in `transforms/optimizers/`; registry in `transforms/registry.py` |
| `transforms/optimizers/` | Per-domain optimizer orchestrators (`ir_structure`, reference, tool, diagnostic, context) |
| `runtime/` | **TierClassifier** (workload complexity — not a provider router) |
| `transport/` | Request/Response types, serialization, delta wire, congestion |
| `protocol/` | Prefix canonicalization, cache planners, binary framing, manifest |
| `providers/` | `adapters/` (17 providers), `transport/` (HTTP dispatch), **credentials** |
| `proxy/` | FastAPI server, `register_health_routes`, `LatticeHeaderMiddleware` (`proxy/middleware.py`) |
| `gateway/` | HTTP compatibility layer; routing headers stashed on `request.state` (middleware emits) |
| `integrations/` | Agent wrap/lace/init; `tunnel.py` sidecar; `mutation_store` (durable + transient); per-agent `doctor()` |

**CLI integrations:** `lattice doctor` (no args) runs health checks for all five primary agents (`claude`, `codex`, `cursor`, `opencode`, `copilot`). `lattice doctor <agent>` runs one. `lattice status` uses `mutation_store.list_all_active()` (init + in-flight lace). `JsonFileIntegration.patch()` raises `AgentNotInstalledError` when config is missing (non–dry-run).

**Public Python surface:** `from lattice import LatticeClient, LatticeProxyClient, wrap_openai_client, CompressResult` (avoid `lattice.sdk.client` — deprecated, removed in v1.1).

**Deleted / moved (do not import):** `core/scheduler.py`, `core/optimizer_scheduler.py`, `core/unified_planner.py`, `core/tunnel_sidecar.py` (→ `integrations/tunnel.py`), `optimizer/` package, `runtime/router.py`, text `StructureOptimizer`, `proxy/compat_exports.py`.

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
3. Register in `transforms/registry.py` and `pipeline/runner.py` `PipelineTransformRegistry._FACTORIES`

## Testing

- Unit tests: `tests/unit/` — **1760 passed**, 196 skipped (Phase 11 full reorg pending)
- Integrations unit tests: `tests/unit/integrations/` (tunnel, doctor, mutation_store, protocol)
- Integration tests: `tests/integration/` — proxy sessions, Redis, IR optimizer E2E
- E2E tests: `tests/e2e/` — agent wrappers, full pipeline
- Contract tests: `tests/contract/` — green (HTTP health + headers + Python API)
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

## Latest CI (Phase 9)

| Metric | Value |
|--------|-------|
| Tests passed | **1760** (+ 196 skipped) |
| Contract tests | green (`tests/contract/`) |
| ruff / format / mypy | **0 errors** |

Canonical benchmark vs `phase-0-baseline.json` → `phase-9-observability.json` (±2%): operator-run with `OLLAMA_CLOUD_API_KEY`.
