# v0.x → v1.0.0 Migration Guide

LATTICE 1.0.0 is the first stable release. The **public CLI and HTTP API are stable**, but internal Python imports were reorganised to a flat domain layout. If you import LATTICE from Python, use this guide.

For v2.0 forward-plan breaking changes (not yet shipped), see [`FORWARD_PLAN.md`](FORWARD_PLAN.md).

---

## What does NOT change

- **CLI commands.** `lattice proxy run/start/stop/restart/status`, `lattice init`, `lattice lace`, `lattice unlace`, `lattice info`, `lattice config`, `lattice health`, `lattice status`, `lattice doctor`, `lattice benchmark` — syntax unchanged.
- **HTTP endpoints.** `/v1/chat/completions`, `/v1/messages`, `/v1/models`, `/v1/responses` (POST/GET/DELETE/WS), `/lattice/gateway`, `/lattice/session/*` — unchanged.
- **Response headers.** `x-lattice-compression`, `x-lattice-session-id`, `x-lattice-delta`, `x-lattice-cost-usd`, `x-lattice-provider`, `x-lattice-transforms-applied` — unchanged (emitted by `LatticeHeaderMiddleware` in `proxy/middleware.py`).
- **Top-level Python API.** `from lattice import LatticeClient, LatticeProxyClient, wrap_openai_client, CompressResult, __version__` — unchanged.
- **Configuration.** `LatticeConfig`, `lattice.yaml`, and env vars (`LATTICE_PROVIDER_BASE_URL`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) — unchanged except `transform_delta_encode` now correctly controls delta encoding (see below).

---

## What changes if you import internals

### Pipeline & scheduler

| v0.x | v1.0.0 |
|------|--------|
| `lattice.core.pipeline.CompressorPipeline` | **REMOVED.** Use `lattice.pipeline.Pipeline`. |
| `lattice.core.pipeline_v2.PipelineV2` | `lattice.pipeline.Pipeline` |
| `lattice.core.pipeline_v2_wrapper.*` | **REMOVED** |
| `lattice.core.pipeline_factory.build_default_pipeline` | `lattice.pipeline.build_default_pipeline` |
| `lattice.core.pipeline_factory.build_v2_pipeline` | **REMOVED** — use `build_default_pipeline` |
| `lattice.core.pipeline_factory.build_optimizer_pipeline` | **REMOVED** |
| `lattice.core.pipeline_factory.build_benchmark_pipeline` | `lattice.pipeline.build_benchmark_pipeline` |
| `lattice.core.scheduler.decide_schedule` | **REMOVED.** Use `lattice.planner.UnifiedPlanner.plan()` |
| `lattice.core.scheduler.SchedulerDecision` | **REMOVED.** Use `lattice.planner.ExecutionPlan` |
| `lattice.core.optimizer_scheduler.*` | **REMOVED.** Use `lattice.planner.build_execution_plan` |
| `lattice.core.unified_planner.UnifiedPlanner` | `lattice.planner.UnifiedPlanner` |
| `lattice.core.unified_planner.SemanticProfile` | `lattice.planner.SemanticProfile` |
| `lattice.core.task_classifier.classify_task` | `lattice.planner.classify_task` |
| `lattice.core.runtime_state.*` | `lattice.planner.runtime_state.*` |
| `lattice.core.policy.OptimizationPolicy` | `lattice.pipeline.OptimizationPolicy` |
| `lattice.core.guardrails.*` | `lattice.pipeline.guardrails.*` |
| `lattice.core.milv.*` | `lattice.pipeline.milv.*` |
| `lattice.core.auto_continuation.*` | `lattice.pipeline.auto_continuation.*` |
| `lattice.core.batch_accumulator.*` | `lattice.pipeline.batch_accumulator.*` |

### IR

| v0.x | v1.0.0 |
|------|--------|
| `lattice.core.ir.PromptIR` | `lattice.ir.PromptIR` |
| `lattice.core.ir.Span, Section, SectionType, SpanRole` | `lattice.ir.types.*` |
| `lattice.core.ir_builder.build_ir` | `lattice.ir.build_ir` |
| `lattice.core.ir_normalizer.normalize_ir` | `lattice.ir.normalize_ir` |
| `lattice.core.ir_serializer.serialize_ir_to_text` | `lattice.ir.serialize_ir_to_text` |
| `lattice.core.ir_transform.IRTransform, CandidateSearch` | `lattice.ir.*` |
| `lattice.core.primitives.PromptIRV2, Candidate, ...` | `lattice.ir.primitives.*` |
| `lattice.core.compiler.PromptCompiler` | **REMOVED** — use `build_ir` + `normalize_ir` + `serialize_ir_to_text` |
| `lattice.core.semantic_graph.*` | `lattice.ir.semantic_graph.*` |

### Transport (wire types)

| v0.x | v1.0.0 |
|------|--------|
| `lattice.core.transport.Request, Response, Message, Role` | `lattice.transport.types.*` |
| `lattice.core.transport.Transform, SyncTransform` | `lattice.transport.types.*` |
| `lattice.core.serialization.*` | `lattice.transport.serialization.*` |
| `lattice.core.delta_wire.*` | `lattice.transport.delta_wire.*` |

### State, cache, telemetry, safety

| v0.x | v1.0.0 |
|------|--------|
| `lattice.core.session.*` | `lattice.state.*` |
| `lattice.core.store.RedisSessionStore` | `lattice.state.RedisSessionStore` |
| `lattice.core.semantic_cache.*` | `lattice.cache.*` |
| `lattice.core.metrics.*` | `lattice.telemetry.*` |
| `lattice.core.telemetry.*` | `lattice.telemetry.*` (module file: `telemetry/downgrade.py`) |
| `lattice.core.agent_stats.*` | `lattice.telemetry.*` |
| `lattice.core.cost_estimator.*` | `lattice.telemetry.*` |
| `lattice.core.maintenance.*` | `lattice.telemetry.*` |
| `lattice.utils.streaming_sketches.*` | `lattice.telemetry.*` |
| `lattice.utils.validation.*` | `lattice.safety.*` |
| `lattice.utils.patterns.*` | `lattice.transforms.patterns.*` |

### Providers

| v0.x | v1.0.0 |
|------|--------|
| `lattice.providers.openai.OpenAIAdapter` | `lattice.providers.adapters.openai.OpenAIAdapter` |
| `lattice.providers.anthropic.AnthropicAdapter` | `lattice.providers.adapters.anthropic.AnthropicAdapter` |
| (all per-provider modules) | `lattice.providers.adapters.*` |
| `lattice.providers.stall_detector.*` | `lattice.providers.transport.stall_detector.*` |
| `lattice.core.credentials.*` | `lattice.providers.credentials.*` |

### Transforms

| v0.x | v1.0.0 |
|------|--------|
| `lattice.transforms.prefix_opt.*` | **REMOVED** — folded into `content_profiler`; `transform_prefix_opt` is no-op until v1.1 |
| `lattice.transforms.constraint_lifting.*` | **REMOVED** — `transform_constraint_lifting` no-op |
| `lattice.transforms.strategy_selector.*` | **REMOVED** — `transform_strategy_selector` no-op |
| `lattice.transforms.format_conv.*` | `lattice.transforms.format_converter.*` |
| `lattice.core.transform_registry.*` | `lattice.transforms.registry.*` |
| `lattice.optimizer.*` (orchestrators) | `lattice.transforms.optimizers.*` |
| `lattice.optimizer.structure_optimizer.*` | **REMOVED** — use `IRStructureOptimizer` |
| `lattice.optimizer.representation_optimizer.*` | `lattice.pipeline.representation_optimizer.*` |

### Runtime & misc

| v0.x | v1.0.0 |
|------|--------|
| `lattice.runtime.router.RuntimeRouter` | `lattice.runtime.tier_classifier.TierClassifier` |
| `lattice.core.tunnel_sidecar.*` | `lattice.integrations.tunnel.*` |
| `lattice.sdk.client.LatticeClient` | `from lattice import LatticeClient` (shim warns; removed v1.1) |
| `lattice.proxy.compat_exports.*` | **REMOVED** |
| `lattice.evals.*` | **REMOVED** — use `benchmarks/evals/` |

### Config flags (no-op in 1.0.0, removed in v1.1)

- `transform_prefix_opt`
- `transform_constraint_lifting`
- `transform_strategy_selector`

---

## Removed behaviour

- Legacy `process(request, ctx)` on IR-native transforms — use `optimize(ir, request, ctx)`. Response-side `output_cleanup` keeps `process(response, ctx)` with `is_response_side=True`.
- `benchmarks/evals/cli.py --use-v2-pipeline` — one pipeline only.
- `import lattice.sdk.client` — `DeprecationWarning`; use `from lattice import LatticeClient`.

---

## Bug fixes you might rely on

- **`transform_delta_encode`** now controls `delta_encode` (was tied to `transform_batching` in v0.x).
- **`JsonFileIntegration.patch()`** raises `AgentNotInstalledError` when agent config is missing.
- **`lattice doctor`** covers all five agents (was three in early v0.x).

---

## Quick fix script

Find stale imports in your tree:

```bash
rg "from lattice\.(core\.pipeline|core\.scheduler|core\.optimizer_scheduler|core\.compiler|core\.transform_registry|core\.transform_reputation|core\.metrics|core\.telemetry|core\.agent_stats|core\.cost_estimator|core\.maintenance|core\.session|core\.store|core\.semantic_cache|core\.credentials|core\.unified_planner|core\.task_classifier|core\.runtime_state|core\.transport|core\.serialization|core\.delta_wire|core\.ir|core\.primitives|core\.semantic_graph|core\.policy|core\.guardrails|core\.milv|core\.auto_continuation|core\.batch_accumulator|core\.tunnel_sidecar|core\.pipeline_factory|core\.pipeline_v2|core\.pipeline_v2_wrapper|optimizer|utils\.validation|utils\.streaming_sketches|utils\.patterns|providers\.base|providers\.openai|providers\.openai_compatible|providers\.anthropic|providers\.azure|providers\.bedrock|providers\.gemini|providers\.ollama|providers\.stall_detector|runtime\.router|transforms\.prefix_opt|transforms\.constraint_lifting|transforms\.semantic_segmenter|transforms\.format_conv|transforms\.strategy_selector|sdk\.client|evals)\b" .
```

Validated against `tests/unit/test_no_old_paths.py::OLD_PATHS`.

---

## Documentation rename

- `docs/architecture/runtime_v2.md` → [`docs/architecture/runtime.md`](../architecture/runtime.md) (redirect stub kept one release cycle).

---

## Per-phase notes (shipped on main)

<details>
<summary>Phases 7–10 incremental changes</summary>

### Phase 7 — Proxy / SDK / CLI

- `from lattice import LatticeClient` (not `lattice.sdk.client`).
- `/healthz`, `/readyz`, `/startupz`, `/metrics`, `/stats` registered.
- `x-lattice-*` via `LatticeHeaderMiddleware`.

### Phase 8 — Integrations

- `lattice.core.tunnel_sidecar` → `lattice.integrations.tunnel`.
- `lattice doctor` all five agents; `AgentNotInstalledError`; transient lace in `mutation_store`.

### Phase 9 — Observability / state

- `metrics`, `session`, `semantic_cache`, `validation` → `telemetry`, `state`, `cache`, `safety` (see tables above).

### Phase 10 — Benchmarks

- `lattice benchmark` → `benchmarks/evals/cli.py`; `src/lattice/evals/` removed; `CLAIMS.md` + `v1.0.0.json`.

</details>

---

## Anything missing?

Open an issue with the old import path; maintainers will add it here.
