# LATTICE — Development Guide for AI Agents

## Setup

```bash
git clone https://github.com/Harsh-Daga/lattice
cd lattice
uv sync

# Run proxy
uv run python -m lattice.proxy.server --port 8787

# Run tests (parallel by default)
uv run pytest tests/ -q

# Local benchmarks (no API key for feature suite)
uv run python benchmarks/evals/cli.py --suite feature
# Or via CLI wrapper:
uv run lattice benchmark --suite feature
```

## Architecture

**LATTICE is the transport / network layer for LLM traffic** — not a compression-only library. One self-hosted proxy owns the path to the user's chosen provider: connections, retry, timeouts, circuit breaker, backpressure, framing, streaming, cache, guardrails, compression, observability.

Canonical runtime (Phases 0–12 shipped on main):

```
Request → profile → UnifiedPlanner → ExecutionPlan → Pipeline.compress → providers/transport → Provider
```

| Doc | Purpose |
|-----|---------|
| [`docs/architecture/runtime.md`](docs/architecture/runtime.md) | Five lifecycles + module rules |
| [`docs/refactor/FORWARD_PLAN.md`](docs/refactor/FORWARD_PLAN.md) | Post–v1.0.0 phases 12–27 (internal numbering) |
| [`docs/refactor/PHASE_GUIDELINES.md`](docs/refactor/PHASE_GUIDELINES.md) | Template for forward phase docs |
| [`docs/refactor/SINGLE_SOURCE_OF_TRUTH.md`](docs/refactor/SINGLE_SOURCE_OF_TRUTH.md) | One primitive → one file |
| [`docs/refactor/STATUS.md`](docs/refactor/STATUS.md) | Shipped vs pending |
| [`docs/refactor/MIGRATION.md`](docs/refactor/MIGRATION.md) | v0.x → v1.0.0 import map |

### Six constraints (forward plan — do not violate in new code)

1. **Lightweight** — default install on a 4 GB laptop; no required model downloads.
2. **No external LLM** — only the user's provider for embeddings/summarization when needed.
3. **Self-hosted OSS only** — no SaaS / Stripe / hosted cloud.
4. **One implementation** — no duplicate algorithms in SDKs or adapters; update `SINGLE_SOURCE_OF_TRUTH.md`.
5. **Code budget** — `src/lattice/` ≤ 35k LoC; declare delta per PR (`CODE_BUDGET.txt`).
6. **Transport-first** — retry/timeout/pool/breaker live in `transport/` only ([Phase 27](docs/refactor/27-transport-layer-consolidation.md)).

### Key modules

| Directory | Responsibility |
|-----------|----------------|
| `core/` | Leaf primitives: `config`, `context`, `errors`, `result`, `segmentation` (6 files incl. `__init__.py`) |
| `telemetry/` | `metrics`, `downgrade`, `agent_stats`, `cost_estimator`, `maintenance`, `streaming_sketches` |
| `state/` | `session`, `store`, `segment_store` |
| `cache/` | `semantic.py` — `SemanticCache` |
| `safety/` | `risk_scoring.py` |
| `ir/` | PromptIRV2, builder, primitives, native optimizer, quality, validation |
| `planner/` | **UnifiedPlanner**, task classifier, execution plan, runtime state bridges |
| `pipeline/` | `Pipeline`, policy, guardrails, MILV, representation_optimizer |
| `transforms/` | Per-transform modules; `registry.py`; `optimizers/` orchestrators |
| `runtime/` | **TierClassifier** (workload complexity — not a provider router) |
| `transport/` | Wire types, serialization, delta_wire, TACC congestion (unified layer grows in Phase 27) |
| `protocol/` | Binary framing, manifest, segments |
| `providers/` | `adapters/` (17 providers), `transport/` (HTTP dispatch), `credentials` |
| `proxy/` | FastAPI server, health routes, `LatticeHeaderMiddleware` |
| `gateway/` | OpenAI/Anthropic compat; routing state on `request.state` |
| `integrations/` | Agent lace/init; `tunnel.py`; `mutation_store`; per-agent `doctor()` |
| `cli/` | `lattice` entry point |

**CLI:** `lattice doctor` (no args) checks all five agents (`claude`, `codex`, `cursor`, `opencode`, `copilot`). `lattice status` uses `mutation_store.list_all_active()`.

**Public Python surface:** `from lattice import LatticeClient, LatticeProxyClient, wrap_openai_client, CompressResult` — avoid `lattice.sdk.client` (deprecated, removed in v1.1).

**Do not import:** `core/scheduler.py`, `core/unified_planner.py`, `optimizer/`, `runtime/router.py`, `CompressorPipeline`, `proxy/compat_exports.py`.

## Code conventions

- **Result[T,E]** — `Ok(value)` or `Err(error)`
- **ReversibleSyncTransform** — `optimize(ir, request, ctx) → Result[PromptIRV2, TransformError]`
- **Immutable PromptIRV2** — `.with_sections()` / `.with_spans()` / `.with_text()`
- **Single scheduler** — `UnifiedPlanner.plan()` only
- **mypy strict**, **ruff** before commits

## Adding a transform

1. Subclass `ReversibleSyncTransform` with `name` and `priority`
2. Implement `optimize(PromptIRV2, Request, TransformContext) → Result[PromptIRV2, TransformError]`
3. Register in `transforms/registry.py` and `pipeline/runner.py` `PipelineTransformRegistry._FACTORIES` if pipeline-default

## Testing

- Unit: `tests/unit/<domain>/` mirrors `src/lattice/` (root: `test_test_count_pinned.py`, `test_no_old_paths.py`, `test_core_is_leaf.py`, `test_no_lattice_evals.py`)
- Integration: `tests/integration/`
- E2E: `tests/e2e/`
- Contract: `tests/contract/` — `uv run pytest tests/contract/ -q`; full live probes: `LATTICE_CONTRACT_FULL=1 uv run pytest tests/contract/ -q --run-contract`
- **2016** tests collected (pinned in `tests/unit/test_test_count_pinned.py`); **1801** passed, **215** skipped (`uv run pytest tests/ -q`)

## Key environment variables

| Variable | Purpose |
|----------|---------|
| `LATTICE_PROVIDER_BASE_URL` | Default upstream provider base URL |
| `LATTICE_PROVIDER_BASE_URLS` | JSON `{provider: url}` overrides |
| `OPENAI_API_KEY` | OpenAI, Azure, OpenAI-compatible providers |
| `ANTHROPIC_API_KEY` | Anthropic provider |
| `OLLAMA_CLOUD_API_KEY` | Canonical benchmark gate (never commit) |

## Benchmark CLI

```bash
uv run python benchmarks/evals/cli.py --suite all \
  --providers ollama-cloud \
  --provider-model ollama-cloud=kimi-k2.6:cloud \
  --iterations 3 --warmup 1
```

Suites: `all`, `feature`, `feature-matrix`, `provider`, `protocol`, `transport`, `integration`, `capability`, `replay`, `replay-governance`, `tacc`, `control`.

Numeric README claims must match [`benchmarks/results/CLAIMS.md`](benchmarks/results/CLAIMS.md).

## Latest CI (v1.0.0)

| Metric | Value |
|--------|-------|
| Tests collected | **2016** (pinned) |
| Tests passed | **1801** (215 skipped) |
| Contract tests | green (`tests/contract/`) |
| ruff / format / mypy | **0 errors** |

Canonical benchmark vs `phase-0-baseline.json`: operator-run with `OLLAMA_CLOUD_API_KEY` → `benchmarks/results/v1.0.0.json`.
