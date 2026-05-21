# Phase 11 — Docs & v1.0.0 Release

> **Goal.** Bring documentation, README, AGENTS.md, and the package version into alignment with the v1.0.0 shape produced by Phases 0–10. Rename `docs/architecture/runtime_v2.md` → `docs/architecture/runtime.md` (there is no v1). Delete any duplicate/legacy docs. Rewrite `README.md` to reflect the new structure, the corrected counts, and citations into `benchmarks/results/CLAIMS.md`. Write `CHANGELOG.md` and the user-facing `docs/refactor/MIGRATION.md`. Bump `pyproject.toml` version to `1.0.0`. Tag `v1.0.0`. PyPI publish dry-run; on green, release.
>
> **Outcome.** A v1.0.0 release on PyPI, a GitHub tag, accurate docs, no `v2` references anywhere outside the migration guide.
>
> **Estimated effort.** 1 day.

---

## 1. Why this phase exists

After Phases 0–10 the **code** matches FINAL_LAYOUT.md but the **narrative** doesn't:

1. **README.md still describes the v0.x architecture.** The 18-transforms list, the architecture diagram with "CompressorPipeline", the "v2 runtime cutover" phrasing — all of it pre-dates the refactor.
2. **AGENTS.md says "converging on one canonical v2 runtime."** v2 is now just the runtime. Drop the "v2".
3. **`docs/architecture/runtime_v2.md` is the canonical doc** per the audit, but a stale `docs/concepts/architecture.md` may still exist with conflicting prose.
4. **No CHANGELOG.md.** Every release needs one; v1.0.0 is the breaking-change release that *especially* needs one.
5. **No MIGRATION.md for users.** Internal import paths break; users need a side-by-side mapping.
6. **`pyproject.toml` says `version = "0.1.0"`.** This is `1.0.0`.
7. **`Development Status :: 4 - Beta`** classifier in `pyproject.toml` is wrong for a 1.0.0 release.

---

## 2. Files touched

### 2.1 Renamed

| Current | New |
|---|---|
| `docs/architecture/runtime_v2.md` | `docs/architecture/runtime.md` |

### 2.2 Created

```
CHANGELOG.md                          # full v1.0.0 release notes
docs/refactor/MIGRATION.md            # v0.x → v1.0.0 mapping table
docs/refactor/FEATURE_PARITY.md       # written in Phase 10; verified here
```

### 2.3 Deleted

```
docs/concepts/architecture.md         # if it exists and duplicates runtime.md
docs/concepts/proxy.md                # if it duplicates docs/operations/integrations.md (audit decides)
```

### 2.4 Modified

- `README.md` — full rewrite.
- `AGENTS.md` — full rewrite.
- `docs/index.md` — update links to renamed files.
- `pyproject.toml` — version, classifier, Development Status.
- Every doc file — remove every "v2" mention except in MIGRATION.md and CHANGELOG.md.

---

## 3. Step-by-step

### 3.1 Rename runtime_v2.md → runtime.md

```bash
git mv docs/architecture/runtime_v2.md docs/architecture/runtime.md
# Update any link to the old path
sd 'runtime_v2\.md' 'runtime.md' $(rg -l "runtime_v2\\.md")
sd '\\bruntime v2\\b' 'runtime' $(rg -l --pcre2 "(?i)\\bruntime v2\\b" docs/)
```

### 3.2 Audit and delete duplicate docs

```bash
diff docs/architecture/runtime.md docs/concepts/architecture.md 2>/dev/null && echo "DUPLICATE"
diff docs/operations/integrations.md docs/concepts/proxy.md 2>/dev/null && echo "DUPLICATE"
```

For any flagged duplicate: delete the lesser-quality copy; if both have unique content, merge into the canonical location.

### 3.3 Rewrite `README.md`

Keep the badges block (Python versions, PyPI, CI, license, tests). Update the tests badge to the Phase 10 pinned `EXPECTED_TEST_COUNT`.

The narrative body should:

- Open with a one-paragraph summary (LATTICE = transport proxy + compression pipeline + safety gating).
- Architecture diagram updated to v1.0.0 (replace `CompressorPipeline` with `Pipeline`, `UnifiedPlanner`, `ProviderRegistry`, etc.).
- The "Compression Pipeline" table lists every transform in `transforms/registry.py` — the count matches the registry exactly (audit: this is computed from `list_default_pipeline_names()` + `list_execution_only_names()` + the off-by-default set).
- The "Supported Providers" table — same 17 providers; verify nothing has been quietly dropped.
- "Novel Tech" sections — link into `docs/novel/*.md` (no content change).
- Every numerical claim is suffixed with a `[1]`, `[2]` citation pointing at `benchmarks/results/CLAIMS.md`.

A skeleton:

```markdown
# <h1 align="center">LATTICE</h1>

<p align="center"><strong>LLM Transport & Efficiency Layer</strong><br>
<em>Make every LLM call cheaper, faster, and safe — without changing your model.</em></p>

<p align="center">
  <a href="https://pypi.org/project/lattice-transport/"><img src="https://img.shields.io/pypi/v/lattice-transport?label=v1.0.0" alt="PyPI"></a>
  <a href="..."><img src="https://img.shields.io/github/actions/workflow/status/Harsh-Daga/lattice/ci.yml?branch=main&label=CI" alt="CI"></a>
  <a href="..."><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="..."><img src="https://img.shields.io/badge/tests-{N}%20passed-brightgreen" alt="Tests"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python"></a>
</p>

---

**LATTICE** sits between your application and any LLM provider. It compresses prompts, caches responses, manages concurrency (TACC), supports a native binary protocol, and routes coding agents. Your app sends standard OpenAI-format requests; LATTICE makes them smaller, faster, and cache-friendlier.

**It is not a router.** LATTICE never changes your model, never falls back between providers, never guesses. One provider per request. LATTICE optimises the transport and execution.

## Quick start

[unchanged from v0.x]

## Architecture

```
Application
   │ OpenAI/Anthropic API format
   ▼
LATTICE Proxy :8787
   │
   ├── State (Session, Segments, Cache)
   ├── Planner (RequestClassifier → UnifiedPlanner → ExecutionPlan)
   ├── Pipeline (Pipeline.run(plan) → IR-native transforms)
   ├── Telemetry (metrics, downgrade, cost, agent stats, maintenance)
   └── Provider transport (DirectHTTPProvider, TACC, stall detection)
            │
            ▼
       LLM Provider (one of 17 adapters)
```

[updated diagram showing the v1.0.0 module names, no v2 anywhere]

## Compression Pipeline

LATTICE ships **N transforms** (number matches `list_transform_names()` from `lattice.transforms.registry`) in priority order. Every transform is safety-classified and risk-gated.

| P  | Transform       | Safety       | What it does | Default? |
|---:|-----------------|--------------|--------------|:--------:|
[populated by a script that reads transforms/registry.py at release time]

[Compression-% citations replace prose like "20-50%" with "{measured}% on suite={X}"[1]]

→ [Full Transform Reference](docs/compression/transforms.md)
→ [Claim Traceability](benchmarks/results/CLAIMS.md)

## Safety

[unchanged]

## Observability

[unchanged + add /stats /metrics docs]

## Supported Providers

[17 providers table; verified against ProviderRegistry]

## CLI Reference

[unchanged]

## Agent Integration

[unchanged + mention `lattice doctor` covers all five]

## Development

```bash
git clone https://github.com/Harsh-Daga/lattice
cd lattice
uv sync
uv run pytest tests/                                          # {N} tests
uv run ruff check src/
uv run mypy src/lattice/
uv run python benchmarks/evals/cli.py --suite all \
  --providers ollama-cloud --provider-model ollama-cloud=kimi-k2.6:cloud \
  --iterations 1 --warmup 0 --provider-warmup 0
```

## Migrating from v0.x

See [docs/refactor/MIGRATION.md](docs/refactor/MIGRATION.md) — every renamed import + every removed feature.

## License

MIT © Harsh Daga
```

### 3.4 Rewrite `AGENTS.md`

```markdown
# LATTICE — Development Guide for AI Agents

## Setup

```bash
git clone https://github.com/Harsh-Daga/lattice
cd lattice
uv sync

uv run lattice proxy run --port 8787      # foreground proxy
uv run pytest tests/ -q                   # {N} tests
uv run lattice benchmark --suite feature  # local benchmark
```

## Architecture (v1.0.0)

LATTICE is a unified compression + transport system on one canonical path:

```
Request → RequestClassifier → UnifiedPlanner → ExecutionPlan → Pipeline.run(plan) → Provider
```

The single architecture document is [`docs/architecture/runtime.md`](docs/architecture/runtime.md).

### Key modules

| Path | Owns |
|---|---|
| `core/` | Leaf primitives: config, context, errors, result, segmentation |
| `safety/` | Risk scoring |
| `ir/` | IR types + builder + normalizer + serializer + transform protocol + native optimizer + validation + quality |
| `planner/` | RequestClassifier, UnifiedPlanner, ExecutionPlan, runtime_state, provider strategy, transport planner, fallback executor |
| `pipeline/` | Pipeline.run; factory; policy; guardrails; MILV; auto_continuation; batch_accumulator; representation_optimizer |
| `transforms/` | 21 transforms + optimizers subpackage; registry; reputation; patterns |
| `providers/` | Provider adapters (17) + transport package (HTTP pool, TACC, streaming, stall detection) + tool sanitizer + schema filter + MCP |
| `transport/` | Wire types (Request, Response, Message, Role); serialization; TACC congestion; delta_wire; session |
| `protocol/` | Binary framing; reliability; resume; manifest; segments; dictionary codec |
| `state/` | Session store; segment store |
| `cache/` | Semantic cache |
| `telemetry/` | Metrics; downgrade taxonomy; agent stats; cost; maintenance; sketches |
| `runtime/` | Tier classifier (NOT a provider router) |
| `proxy/` | FastAPI app: bootstrap, server, routes, lifecycle, health, middleware |
| `gateway/` | Request/response translation: native + OpenAI/Anthropic compat |
| `sdk/` | LatticeClient (local), LatticeProxyClient (HTTP), wrap_openai_client |
| `integrations/` | Agent lacing for Claude Code, Codex, Cursor, OpenCode, Copilot |
| `cli/` | `lattice` command |

## Conventions

- **Result[T, E] monad** for error handling: `Ok(value)` or `Err(error)`.
- **`optimize(ir, request, ctx) → Result[PromptIRV2, TransformError]`** — every pipeline transform implements this. Legacy `process(request, ctx)` is gone (kept only for response-side transforms with `is_response_side=True`).
- **Immutable PromptIRV2** — transforms return new instances via `.with_sections()` / `.with_spans()` / `.with_text()`.
- **`Pipeline.run(plan)`** — one entry point. ExecutionPlan pre-decides which transforms run.
- **mypy strict**, **ruff** for linting; run both before commits.

## Adding a transform

1. Subclass `ReversibleSyncTransform` (`pipeline.runner`) with `name` and `priority`.
2. Implement `optimize(ir, request, ctx) → Result[PromptIRV2, TransformError]`.
3. (Optional, response-side only) Implement `process(response, ctx) → Result[Response, TransformError]` and set `is_response_side=True` in the spec.
4. Register in `transforms/registry.py`'s spec tuple and (if pipeline-default) `pipeline/runner.py:_IR_NATIVE_TRANSFORMS`.

## Testing

- Unit: `tests/unit/<domain>/` — mirrors `src/lattice/`.
- Integration: `tests/integration/` — proxy, sessions, Redis, streaming, TACC, binary protocol.
- E2E: `tests/e2e/` — agent wrappers, full pipeline.
- Contract: `tests/contract/` — CLI, HTTP endpoints, response headers, public Python API.
- Parallel by default: `uv run pytest tests/ -q` (`-n auto`).

## Latest benchmark

| Metric | Value |
|---|---|
| Tests passed | **{N}/{N}** |
| ruff errors | **0** |
| mypy errors | **0** |
| Headline compression % (suite=all, kimi-k2.6:cloud) | see `benchmarks/results/v1.0.0.json` |
```

### 3.5 Write `CHANGELOG.md`

```markdown
# Changelog

All notable changes to LATTICE are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — YYYY-MM-DD

The v1.0.0 release converges the codebase to a single canonical path per concern.

**Highlights:**
- Single Pipeline (v1 `CompressorPipeline` removed; v2 `Pipeline` is the only runtime).
- Single Planner (`UnifiedPlanner`); old `scheduler.py` and `optimizer_scheduler.py` deleted.
- Three-way "transport" name collision resolved: `lattice.transport.types` (Request/Response), `lattice.transport.congestion` (TACC), `lattice.providers.transport` (HTTP dispatch).
- New flat domain layout: `ir/`, `planner/`, `pipeline/`, `transforms/`, `providers/`, `transport/`, `protocol/`, `state/`, `cache/`, `telemetry/`, `safety/`, `runtime/`, `proxy/`, `gateway/`, `sdk/`, `cli/`, `integrations/`.
- All user-facing CLI commands and HTTP endpoints preserve their v0.x contracts (see `tests/contract/`).

### Added
- `lattice.ir` package: `PromptIR`, `PromptIRV2`, `build_ir`, `normalize_ir`, `serialize_ir_to_text`, `IRTransform`, `CandidateSearch`, `IRNativeOptimizer`, `validate_candidate`, `estimate_quality`.
- `lattice.planner` package: `UnifiedPlanner`, `RequestClassifier`, `ExecutionPlan`, `build_execution_plan`, `classify_task`.
- `lattice.pipeline` package: `Pipeline`, `build_default_pipeline`, `OptimizationPolicy`, `GuardAction`, `RepresentationOptimizer`, `MILVResult`.
- `lattice.cache` package: `SemanticCache`, `ContentClass`.
- `lattice.safety` package: `SemanticRiskScore`, `compute_risk_score`.
- `lattice.state` package: `Session`, `SessionManager`, `SegmentStore`, `RedisSessionStore`.
- `lattice.telemetry` package: `MetricsCollector`, `DowngradeCategory`, `AgentStatsCollector`, `CostEstimator`, `MaintenanceCoordinator`, `CountMinSketch`, `HyperLogLog`.
- `lattice.providers.transport` package (split from monolith): registry, pool, rate_limits, completion, streaming, stall_detector, helpers.
- `lattice.providers.adapters` package: all 17 adapters.
- `/healthz`, `/readyz`, `/startupz`, `/metrics`, `/stats` endpoints registered (were defined but unwired in v0.x).
- `lattice benchmark` is now a real wrapper around `benchmarks/evals/cli.py` (was a redirect stub).
- `lattice version` alias for `--version`.
- `lattice doctor` now covers all 5 supported agents (was missing Cursor + Copilot).
- `AgentNotInstalledError` raised by integrations on missing config (was silent success).
- New `@runtime_checkable AgentIntegrationProtocol`; mypy now enforces every integration class is complete.
- `MutationStore.record_transient_lace` / `clear_transient_lace`: `lattice status` reports both durable (init) and transient (lace) state.
- Tests/contract suite: exhaustive CLI matrix, HTTP matrix, headers matrix, Python API matrix.
- `benchmarks/results/CLAIMS.md`: traceability table from every README claim to its JSON source.
- `pytest-xdist`: parallel test runs by default (`-n auto`).
- `is_response_side` flag on `TransformSpec`: response-side transforms (`output_cleanup`) are dispatched after the response, not before the request.

### Changed
- **BREAKING (internal):** module paths reorganised per `docs/refactor/FINAL_LAYOUT.md`. See `docs/refactor/MIGRATION.md` for the full old → new mapping.
- Response headers (`x-lattice-compression`, etc.) emit via a single middleware (`proxy/middleware.py:LatticeHeaderMiddleware`) instead of per-handler.
- `transform_delta_encode` flag now correctly governs `delta_encode` transform (was incorrectly using `transform_batching`).
- `runtime/router.py` renamed to `runtime/tier_classifier.py`; class `RuntimeRouter` → `TierClassifier`. The file does not route between providers (README always said it didn't); the rename honours the docs.
- `lattice.sdk.client` is now a deprecation shim emitting `DeprecationWarning`. Use `from lattice import LatticeClient` (or `from lattice.sdk import LatticeClient`) — same class, cleaner path. Shim removed in v1.1.

### Removed
- `core/pipeline.py` (v1 `CompressorPipeline`). Use `lattice.pipeline.Pipeline`.
- `core/pipeline_v2_wrapper.py` (bridge between v1 and v2).
- `core/scheduler.py` (v1 reactive scheduler).
- `core/optimizer_scheduler.py` (v1 optimizer scheduler wrapper).
- `core/compiler.py` (3-line wrapper inlined into `ir/builder.py`).
- `optimizer/structure_optimizer.py` (text-based; superseded by `ir_structure_optimizer.py`).
- `transforms/prefix_opt.py` (deprecated wrapper; logic in content_profiler since 0.x).
- `transforms/constraint_lifting.py` (no production consumer found).
- `transforms/strategy_selector.py` (bandit; no benchmark evidence of lift; gated cut).
- Information-theoretic variant of `transforms/context_selector.py` (submodular is canonical; gated cut).
- `src/lattice/evals/` (empty placeholder; canonical evals at `benchmarks/evals/`).
- `proxy/compat_exports.py` (orphaned re-export shim).
- Every transform's legacy `process(request, ctx)` method (where an IR-native `optimize()` existed). Response-side `output_cleanup` keeps `process(response, ctx)`.
- `benchmarks/evals/cli.py --use-v2-pipeline` flag (v2 is the only pipeline).

### Fixed
- `delta_encode` was silently controlled by `transform_batching` flag.
- `JsonFileIntegration.patch()` silently no-op'd when the agent's config file didn't exist; now raises `AgentNotInstalledError`.
- `RateLimitTracker` grew unbounded; now evicts entries older than 1 hour.
- `lattice doctor` only knew about 3 of 5 agents; now handles all 5.
- ~300 LoC of duplicated streaming retry logic between `completion_stream()` and `completion_stream_with_stall_detect()` collapsed into a single `_stream()` method.

### Performance
- Single-pipeline path is faster than the v1+v2 dual path for non-streaming requests (no v2-wrapper indirection, no policy/guardrails double-pass).
- Parallel test runs cut CI from ~4 min to ~90 s.

### Documentation
- `docs/architecture/runtime_v2.md` → `docs/architecture/runtime.md` (renamed; this is the only architecture doc).
- README rewritten; every numerical claim links to `benchmarks/results/CLAIMS.md`.
- AGENTS.md rewritten to reflect the flat layout.
- `docs/refactor/` contains the complete 12-phase plan and migration guide.

## [0.1.0] — 2025-04-xx
Initial public release.
```

(Filled date in §3.10 at tag time.)

### 3.6 Write `docs/refactor/MIGRATION.md`

```markdown
# v0.x → v1.0.0 Migration Guide

LATTICE 1.0.0 is the first stable release. It is a **major version bump**: the
public CLI and HTTP API are stable, but internal Python imports have been
reorganised to a flat domain layout. If you import LATTICE from Python, this
guide shows you what changed.

## What does NOT change

- **CLI commands.** `lattice proxy run/start/stop/restart/status`, `lattice init`, `lattice lace`, `lattice unlace`, `lattice info`, `lattice config`, `lattice health`, `lattice status`, `lattice doctor`, `lattice benchmark` — all syntax unchanged.
- **HTTP endpoints.** `/v1/chat/completions`, `/v1/messages`, `/v1/models`, `/v1/responses` (POST/GET/DELETE/WS), `/lattice/gateway`, `/lattice/session/*` — all unchanged.
- **Response headers.** `x-lattice-compression`, `x-lattice-session-id`, `x-lattice-delta`, `x-lattice-cost-usd`, `x-lattice-provider`, `x-lattice-transforms-applied` — all unchanged.
- **Top-level Python API.** `from lattice import LatticeClient, LatticeProxyClient, wrap_openai_client, CompressResult, __version__` — unchanged. `LatticeClient.compress()`, `.compress_request()`, `.decompress_response()`, `.health()`, `.count_tokens()` — all unchanged.
- **Configuration.** `LatticeConfig` field names, `lattice.yaml` keys, and environment variables (`LATTICE_PROVIDER_BASE_URL`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) — unchanged. (One small fix: `transform_delta_encode` flag now correctly controls the delta_encode transform; it previously mis-pointed at `transform_batching`.)
- **All 17 providers, 5 agent integrations, TACC, binary framing, delta encoding, semantic cache, MCP, MILV** — all functional.

## What changes if you import internals

If your code does `from lattice.core.pipeline import CompressorPipeline` or
similar, you'll need to update. Mapping below.

### Pipeline & scheduler

| v0.x | v1.0.0 |
|---|---|
| `lattice.core.pipeline.CompressorPipeline` | **REMOVED.** Use `lattice.pipeline.Pipeline` (the v2 pipeline; v1 deleted). |
| `lattice.core.pipeline_v2.PipelineV2` | `lattice.pipeline.Pipeline` (renamed). |
| `lattice.core.pipeline_v2_wrapper.*` | **REMOVED** (bridge no longer needed). |
| `lattice.core.pipeline_factory.build_default_pipeline` | `lattice.pipeline.build_default_pipeline` |
| `lattice.core.pipeline_factory.build_v2_pipeline` | **REMOVED**, use `build_default_pipeline`. |
| `lattice.core.pipeline_factory.build_optimizer_pipeline` | **REMOVED**. |
| `lattice.core.pipeline_factory.build_benchmark_pipeline` | `lattice.pipeline.build_benchmark_pipeline` |
| `lattice.core.scheduler.decide_schedule` | **REMOVED.** Use `lattice.planner.UnifiedPlanner.plan(request, profile)`. |
| `lattice.core.scheduler.SchedulerDecision` | **REMOVED.** Use `lattice.planner.ExecutionPlan`. |
| `lattice.core.optimizer_scheduler.decide_optimizer_schedule` | **REMOVED.** Use `lattice.planner.build_execution_plan`. |
| `lattice.core.optimizer_scheduler.OptimizerSchedule` | **REMOVED.** Use `lattice.planner.ExecutionPlan`. |
| `lattice.core.unified_planner.UnifiedPlanner` | `lattice.planner.UnifiedPlanner` |
| `lattice.core.unified_planner.SemanticProfile` | `lattice.planner.SemanticProfile` |
| `lattice.core.task_classifier.classify_task` | `lattice.planner.classify_task` |
| `lattice.core.runtime_state.*` | `lattice.planner.runtime_state.*` (or `lattice.planner.*` re-export) |
| `lattice.core.policy.OptimizationPolicy` | `lattice.pipeline.OptimizationPolicy` |
| `lattice.core.guardrails.*` | `lattice.pipeline.guardrails.*` |
| `lattice.core.milv.*` | `lattice.pipeline.milv.*` |
| `lattice.core.auto_continuation.*` | `lattice.pipeline.auto_continuation.*` |
| `lattice.core.batch_accumulator.*` | `lattice.pipeline.batch_accumulator.*` |

### IR

| v0.x | v1.0.0 |
|---|---|
| `lattice.core.ir.PromptIR` | `lattice.ir.PromptIR` (also at `lattice.PromptIR`) |
| `lattice.core.ir.Span, Section, SectionType, SpanRole` | `lattice.ir.types.*` (re-exported from `lattice.ir`) |
| `lattice.core.ir_builder.build_ir` | `lattice.ir.build_ir` |
| `lattice.core.ir_normalizer.normalize_ir` | `lattice.ir.normalize_ir` |
| `lattice.core.ir_serializer.serialize_ir_to_text` | `lattice.ir.serialize_ir_to_text` |
| `lattice.core.ir_transform.IRTransform, CandidateSearch, CandidateScorer` | `lattice.ir.*` |
| `lattice.core.primitives.PromptIRV2, Candidate, ExecutionPlan, ...` | `lattice.ir.primitives.*` (re-exported from `lattice.ir`) |
| `lattice.core.compiler.PromptCompiler` | **REMOVED.** Was a 3-line wrapper; call `lattice.ir.build_ir`, `normalize_ir`, `serialize_ir_to_text` directly. |
| `lattice.core.semantic_graph.*` | `lattice.ir.semantic_graph.*` |

### Transport (wire types)

| v0.x | v1.0.0 |
|---|---|
| `lattice.core.transport.Request, Response, Message, Role` | `lattice.transport.types.*` (still `from lattice.core import Request, ...` works via re-export through v1.1). |
| `lattice.core.transport.Transform, SyncTransform` | `lattice.transport.types.*` |
| `lattice.core.serialization.*` | `lattice.transport.serialization.*` |
| `lattice.core.delta_wire.*` | `lattice.transport.delta_wire.*` |

### State, cache, telemetry, safety

| v0.x | v1.0.0 |
|---|---|
| `lattice.core.session.Session, SessionManager, SessionStore, MemorySessionStore` | `lattice.state.*` |
| `lattice.core.store.RedisSessionStore` | `lattice.state.RedisSessionStore` |
| `lattice.core.semantic_cache.SemanticCache, ContentClass` | `lattice.cache.*` |
| `lattice.core.metrics.MetricsCollector, LatencyTracker` | `lattice.telemetry.*` |
| `lattice.core.telemetry.DowngradeCategory, DowngradeTelemetry, TransportOutcome` | `lattice.telemetry.*` (file renamed to `telemetry/downgrade.py`) |
| `lattice.core.agent_stats.AgentStatsCollector` | `lattice.telemetry.AgentStatsCollector` |
| `lattice.core.cost_estimator.CostEstimator, CostEstimate` | `lattice.telemetry.*` |
| `lattice.core.maintenance.MaintenanceCoordinator` | `lattice.telemetry.MaintenanceCoordinator` |
| `lattice.utils.streaming_sketches.CountMinSketch, HyperLogLog` | `lattice.telemetry.*` |
| `lattice.utils.validation.SemanticRiskScore, compute_risk_score` | `lattice.safety.*` |
| `lattice.utils.patterns.*` | `lattice.transforms.patterns.*` |

### Providers

| v0.x | v1.0.0 |
|---|---|
| `lattice.providers.base.ProviderAdapter` | `lattice.providers.adapters.base.ProviderAdapter` (or `lattice.providers.ProviderAdapter`) |
| `lattice.providers.openai.OpenAIAdapter` | `lattice.providers.adapters.openai.OpenAIAdapter` |
| `lattice.providers.anthropic.AnthropicAdapter` | `lattice.providers.adapters.anthropic.AnthropicAdapter` |
| `lattice.providers.azure.AzureAdapter` | `lattice.providers.adapters.azure.AzureAdapter` |
| `lattice.providers.bedrock.BedrockAdapter` | `lattice.providers.adapters.bedrock.BedrockAdapter` |
| `lattice.providers.gemini.GeminiAdapter, VertexAdapter` | `lattice.providers.adapters.gemini.*` |
| `lattice.providers.ollama.OllamaAdapter, OllamaCloudAdapter` | `lattice.providers.adapters.ollama.*` |
| `lattice.providers.openai_compatible.GroqAdapter, ...` | `lattice.providers.adapters.openai_compatible.*` |
| `lattice.providers.stall_detector.StreamStallDetector` | `lattice.providers.transport.stall_detector.StreamStallDetector` |
| `lattice.providers.transport.DirectHTTPProvider, ProviderRegistry, ConnectionPoolManager, RateLimitTracker` | **Unchanged path** — `lattice.providers.transport.*` is now a package with the same public surface. |
| `lattice.core.credentials.CredentialResolver` | `lattice.providers.credentials.CredentialResolver` |

### Transforms

| v0.x | v1.0.0 |
|---|---|
| `lattice.transforms.prefix_opt.PrefixOptimizer` | **REMOVED.** Functionality folded into `content_profiler` since 0.x; v1.0.0 deletes the wrapper. Config flag `transform_prefix_opt` accepted as no-op; removed in v1.1. |
| `lattice.transforms.constraint_lifting.*` | **REMOVED.** No production consumer. Config flag `transform_constraint_lifting` is no-op. |
| `lattice.transforms.strategy_selector.*` | **REMOVED** (gated by benchmark; if your benchmarks justify keeping it, you'll find a `strategy_selector/` package). Config flag `transform_strategy_selector` is no-op. |
| `lattice.transforms.context_selector.InformationTheoreticSelector` | **REMOVED** (gated; submodular is canonical). |
| `lattice.transforms.semantic_segmenter.*` | `lattice.core.segmentation.*` (not a transform; moved to core). |
| `lattice.transforms.format_conv.FormatConverter` | `lattice.transforms.format_converter.FormatConverter` (file became a package). |
| `lattice.transforms.content_profiler.ContentProfiler` | **Same import path** — `lattice.transforms.content_profiler` is now a package with the same public class. |
| `lattice.core.transform_registry.*` | `lattice.transforms.registry.*` |
| `lattice.core.transform_reputation.*` | `lattice.transforms.reputation.*` |
| `lattice.optimizer.ir_structure_optimizer.IRStructureOptimizer` | `lattice.transforms.optimizers.IRStructureOptimizer` |
| `lattice.optimizer.reference_optimizer.ReferenceOptimizer` | `lattice.transforms.optimizers.ReferenceOptimizer` |
| `lattice.optimizer.tool_optimizer.ToolOptimizer` | `lattice.transforms.optimizers.ToolOptimizer` |
| `lattice.optimizer.diagnostic_optimizer.DiagnosticOptimizer` | `lattice.transforms.optimizers.DiagnosticOptimizer` |
| `lattice.optimizer.context_optimizer.ContextOptimizer` | `lattice.transforms.optimizers.ContextOptimizer` |
| `lattice.optimizer.structure_optimizer.StructureOptimizer` | **REMOVED** (superseded by IRStructureOptimizer). |
| `lattice.optimizer.representation_optimizer.RepresentationOptimizer` | `lattice.pipeline.representation_optimizer.RepresentationOptimizer` |
| `lattice.optimizer.ir_native_optimizer.IRNativeOptimizer` | `lattice.ir.IRNativeOptimizer` |
| `lattice.optimizer.validation.*` | `lattice.ir.validation.*` |
| `lattice.optimizer.quality_estimator.*` | `lattice.ir.quality.*` |

### Runtime

| v0.x | v1.0.0 |
|---|---|
| `lattice.runtime.router.RuntimeRouter` | `lattice.runtime.tier_classifier.TierClassifier` (file + class renamed) |
| `lattice.runtime.router.RoutingDecision` | `lattice.runtime.tier_classifier.TierDecision` |
| `lattice.runtime.Tier` | unchanged |

### Misc

| v0.x | v1.0.0 |
|---|---|
| `lattice.core.tunnel_sidecar.*` | `lattice.integrations.tunnel.*` |
| `lattice.sdk.client.LatticeClient` | Still works, emits `DeprecationWarning`. Use `from lattice import LatticeClient`. Module removed in v1.1. |
| `lattice.proxy.compat_exports.*` | **REMOVED**. |

## Removed behaviour

- The legacy `process(request, ctx) → Result[Request]` method on transforms is removed (where an IR-native `optimize()` existed). All transforms now expose only `optimize(ir, request, ctx) → Result[PromptIRV2, TransformError]`. If you have custom transforms that subclass `ReversibleSyncTransform`, implement `optimize`; `process` is only honoured when `is_response_side=True` (e.g. `output_cleanup`).
- The `--use-v2-pipeline` flag in `benchmarks/evals/cli.py` is removed. v2 is the only pipeline.
- `lattice.sdk.client` emits `DeprecationWarning` on import.

## Bug fixes you might rely on

- **`delta_encode` honours its own config flag.** v0.x had `delta_encode`'s `config_flag` mis-pointing at `transform_batching`. Now it's `transform_delta_encode`. If you used `transform_batching` to control delta encoding, switch.
- **`JsonFileIntegration.patch()` raises `AgentNotInstalledError`** when the agent's config file doesn't exist (v0.x: silent success). Wrap `lattice.integrations.init.run_init(...)` calls in try/except if you depended on silent.
- **`lattice doctor` covers all five agents.** v0.x only handled `claude`, `codex`, `opencode`.

## Quick fix script

If your codebase imports a handful of removed names, this Bash one-liner finds them:

```bash
rg "from lattice\.(core\.pipeline|core\.scheduler|core\.optimizer_scheduler|core\.compiler|core\.transform_registry|core\.transform_reputation|core\.metrics|core\.telemetry|core\.agent_stats|core\.cost_estimator|core\.maintenance|core\.session|core\.store|core\.semantic_cache|core\.credentials|core\.unified_planner|core\.task_classifier|core\.runtime_state|core\.transport|core\.serialization|core\.delta_wire|core\.ir|core\.primitives|core\.semantic_graph|core\.policy|core\.guardrails|core\.milv|core\.auto_continuation|core\.batch_accumulator|core\.tunnel_sidecar|core\.pipeline_factory|core\.pipeline_v2|core\.pipeline_v2_wrapper|optimizer|utils\.validation|utils\.streaming_sketches|utils\.patterns|providers\.base|providers\.openai|providers\.openai_compatible|providers\.anthropic|providers\.azure|providers\.bedrock|providers\.gemini|providers\.ollama|providers\.stall_detector|runtime\.router|transforms\.prefix_opt|transforms\.constraint_lifting|transforms\.semantic_segmenter|transforms\.format_conv|transforms\.strategy_selector|sdk\.client|evals)\b" .
```

Each match maps to a row in the table above.

## Anything missing?

If you find an import that needs migration and isn't covered here, please open
an issue at https://github.com/Harsh-Daga/lattice/issues with the old path; the
maintainers will add it to this guide.
```

### 3.7 Update `docs/index.md` and other doc references

```bash
# Anything that linked to runtime_v2.md should link to runtime.md
sd 'runtime_v2\.md' 'runtime.md' $(rg -l 'runtime_v2\.md' docs/)

# Anything that said "v2 pipeline" or "v2 runtime" should drop the v2 (except in MIGRATION.md and CHANGELOG.md)
rg -l 'v2 pipeline|v2 runtime|PipelineV2|CompressorPipeline' docs/ | grep -v -E '(MIGRATION|CHANGELOG)' | xargs -I {} sd 'v2 pipeline|v2 runtime' 'pipeline' {}
```

Manually review any remaining `rg "\\bv2\\b" docs/` matches.

### 3.8 Update `pyproject.toml`

```toml
[project]
name = "lattice-transport"
version = "1.0.0"                        # was 0.1.0
description = "LLM Transport & Efficiency Layer — make LLM calls cheaper, faster, and smarter"
...
classifiers = [
    "Development Status :: 5 - Production/Stable",      # was "4 - Beta"
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Internet :: Proxy Servers",
    "Typing :: Typed",
]
```

Update `src/lattice/_version.py`:

```python
__version__ = "1.0.0"
```

### 3.9 Final CI gate before tagging

```bash
uv run ruff check src/ tests/ benchmarks/         # 0 errors
uv run mypy src/lattice/                           # 0 errors
uv run pytest tests/ -q                            # all green
uv run pytest tests/contract/ -q                   # all green
uv run pytest -m contract                          # contract-only run
./scripts/run_canonical_benchmark.sh /tmp/v1.0.0-final.json
python scripts/compare_benchmarks.py benchmarks/results/v1.0.0.json /tmp/v1.0.0-final.json --tolerance-pct 2

# Verify no v0.x artefacts
rg "v1\b|v2\b|legacy|wrapper|TODO|FIXME|XXX" src/lattice/ | grep -v -E '__version__|version_info' || echo "clean"
```

### 3.10 Tag and release

```bash
# Date stamp for CHANGELOG
TODAY=$(date -u +"%Y-%m-%d")
sd 'YYYY-MM-DD' "$TODAY" CHANGELOG.md     # only the v1.0.0 header line

git add CHANGELOG.md
git commit -m "chore(release): finalise v1.0.0 changelog date"

# Tag
git tag -a v1.0.0 -m "LATTICE v1.0.0 — converged runtime + flat domain layout"
git push origin v1.0.0

# PyPI dry run
uv build
twine check dist/lattice_transport-1.0.0*

# Real publish (manual decision)
twine upload dist/lattice_transport-1.0.0*
```

### 3.11 Create GitHub Release

```bash
gh release create v1.0.0 \
    --title "LATTICE v1.0.0" \
    --notes-file CHANGELOG.md \
    --discussion-category "Announcements" \
    dist/lattice_transport-1.0.0.tar.gz \
    dist/lattice_transport-1.0.0-py3-none-any.whl
```

---

## 4. Per-file disposition

| File | Action |
|---|---|
| `README.md` | REWRITE — v1.0.0 architecture, transform table from registry, claims with citations |
| `AGENTS.md` | REWRITE — v1.0.0 architecture, flat layout, convention updates |
| `CHANGELOG.md` | CREATE — full v1.0.0 release notes |
| `docs/refactor/MIGRATION.md` | CREATE — old → new mapping for every renamed/removed symbol |
| `docs/architecture/runtime.md` | RENAMED from `runtime_v2.md` |
| `docs/concepts/architecture.md` | DELETED if duplicate of runtime.md |
| `docs/concepts/proxy.md` | DELETED if duplicate of operations/integrations.md |
| `docs/index.md` | MODIFY — update links |
| `docs/refactor/FEATURE_PARITY.md` | VERIFY (written in Phase 10) |
| `pyproject.toml` | MODIFY — version, classifier |
| `src/lattice/_version.py` | MODIFY — `"1.0.0"` |
| All other doc files | MODIFY — drop `v2` references (except MIGRATION + CHANGELOG) |

---

## 5. Acceptance criteria

- [ ] `docs/architecture/runtime.md` exists; `docs/architecture/runtime_v2.md` does not.
- [ ] No file under `docs/` (except `MIGRATION.md` and `CHANGELOG.md`) contains the literal "v2".
- [ ] No file under `docs/` references `CompressorPipeline`, `PipelineV2`, `RuntimeRouter`, `decide_schedule`, `prefix_optimizer`.
- [ ] `README.md` test-count badge matches `tests/unit/test_test_count_pinned.py::EXPECTED_TEST_COUNT`.
- [ ] `README.md`'s transform count matches `len(list_transform_names())` from `lattice.transforms.registry`.
- [ ] `README.md`'s provider count is 17 and matches `len(ProviderRegistry().adapters)`.
- [ ] Every numeric claim in `README.md` and `docs/` has a citation linking to a row in `benchmarks/results/CLAIMS.md`.
- [ ] `CHANGELOG.md` exists with a complete v1.0.0 section, dated.
- [ ] `docs/refactor/MIGRATION.md` exists; the quick-fix `rg` script catches every removed/renamed import.
- [ ] `pyproject.toml` `version = "1.0.0"`, classifier `Development Status :: 5 - Production/Stable`.
- [ ] `src/lattice/_version.py` reads `__version__ = "1.0.0"`.
- [ ] `uv build` produces `lattice_transport-1.0.0.tar.gz` and `.whl`.
- [ ] `twine check dist/lattice_transport-1.0.0*` passes.
- [ ] `git tag v1.0.0` exists.
- [ ] GitHub release page created.
- [ ] All earlier-phase acceptance criteria still hold (CI green, contract tests green, benchmarks within 2% of phase-0-baseline).
- [ ] `FEATURE_PARITY.md`'s 61 rows all link to passing tests.

---

## 6. Risks specific to this phase

| Risk | Mitigation |
|---|---|
| `README.md` rewrite accidentally drops a feature claim that's still true | Use `FEATURE_PARITY.md` as the checklist; every row must appear in the README's relevant section. |
| `MIGRATION.md` is incomplete; users open issues post-release | The quick-fix `rg` regex in MIGRATION.md §"Quick fix script" is grown from this phase's own grep results — it's exhaustive by construction. Add to the doc immediately when a missing path is reported. |
| `twine upload` succeeds but the wheel is missing files | `[tool.hatch.build.targets.wheel]` in pyproject pins `packages = ["src/lattice"]`. Verify via `tar -tzf dist/*.tar.gz \| grep -c "lattice/"` against `find src/lattice -name "*.py" \| wc -l`. |
| Git tag pushed but release notes don't render correctly on GitHub | Use `gh release create --notes-file CHANGELOG.md` which renders Markdown. Verify the release page renders. |
| Someone updates `EXPECTED_TEST_COUNT` between merge and release, leaving README badge stale | The badge is the count *as of v1.0.0 tag*. Phase 11's final step pins the number; future PRs that add tests update both the test pin AND the README badge in the same commit. |
| `docs/architecture/runtime.md` rename breaks a deep link from another repo / blog post | The MIGRATION.md mentions the rename; for the first month post-release, add an `architecture/runtime_v2.md` redirect file containing just `Moved to [runtime.md](runtime.md)`. Delete after a release cycle. |

---

## 7. Rollback plan

If something goes wrong between tag and PyPI publish:

```bash
git tag -d v1.0.0                   # local
git push origin --delete v1.0.0     # remote
# Fix the issue, recommit, re-tag
```

If PyPI publish goes through but a critical bug is found:

```bash
# PyPI does not allow re-publishing the same version.
# Yank the broken version:
twine yank lattice-transport==1.0.0 --reason "Critical bug X; use 1.0.1"
# Ship 1.0.1 with the fix.
```

---

## 8. PR shape

One PR:

```
release: v1.0.0 — README + AGENTS rewrite, CHANGELOG, MIGRATION, version bump [Phase 11]

- pyproject.toml: version 0.1.0 → 1.0.0; classifier Beta → Production/Stable
- src/lattice/_version.py: 1.0.0
- README.md: full rewrite reflecting flat layout, accurate counts, claim citations
- AGENTS.md: full rewrite (key modules table, conventions, transform-authoring guide)
- CHANGELOG.md: complete v1.0.0 release notes
- docs/refactor/MIGRATION.md: every old → new import path
- docs/architecture/runtime_v2.md → runtime.md (rename)
- docs/concepts/architecture.md, docs/concepts/proxy.md: deleted if duplicates
- Update all docs/ links to renamed file
- Remove every "v2" reference from docs/ (except MIGRATION + CHANGELOG)

After merge:
- git tag -a v1.0.0
- uv build && twine check && twine upload
- gh release create v1.0.0
```
