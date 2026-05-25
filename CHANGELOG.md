# Changelog

All notable changes to LATTICE are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-05-25

The v1.0.0 release converges the codebase to a single canonical path per concern.

**Highlights:**

- Single `Pipeline` (`CompressorPipeline` and `PipelineV2` bridge removed).
- Single `UnifiedPlanner`; `scheduler.py` and `optimizer_scheduler.py` deleted.
- Three-way transport naming resolved: `lattice.transport` (wire + TACC), `lattice.providers.transport` (HTTP dispatch).
- Flat domain layout: `ir/`, `planner/`, `pipeline/`, `transforms/`, `providers/`, `transport/`, `protocol/`, `state/`, `cache/`, `telemetry/`, `safety/`, `runtime/`, `proxy/`, `gateway/`, `integrations/`.
- User-facing CLI and HTTP contracts unchanged (`tests/contract/`).

### Added

- `lattice.ir` — `PromptIR`, `PromptIRV2`, `build_ir`, `normalize_ir`, `IRNativeOptimizer`, validation, quality.
- `lattice.planner` — `UnifiedPlanner`, `RequestClassifier`, `ExecutionPlan`, `classify_task`.
- `lattice.pipeline` — `Pipeline`, `OptimizationPolicy`, guardrails, MILV, `RepresentationOptimizer`.
- `lattice.cache` — `SemanticCache`.
- `lattice.safety` — `SemanticRiskScore`, `compute_risk_score`.
- `lattice.state` — `Session`, `SessionManager`, `SegmentStore`, `RedisSessionStore`.
- `lattice.telemetry` — metrics, downgrade taxonomy, agent stats, cost, maintenance, sketches.
- `lattice.providers.transport` — registry, pool, streaming, stall detection (package split).
- `lattice.providers.adapters` — 17 provider adapters.
- Health routes `/healthz`, `/readyz`, `/startupz`, `/metrics`, `/stats` wired on the proxy app.
- `lattice benchmark` wraps `benchmarks/evals/cli.py` (no redirect stub).
- `lattice doctor` covers all five agents (Claude, Codex, Cursor, OpenCode, Copilot).
- `MutationStore` transient lace; `lattice status` reports durable + in-flight lace.
- `benchmarks/results/CLAIMS.md` — README numeric claim traceability.
- `pytest-xdist` parallel test runs (`-n auto`).
- `is_response_side` on `TransformSpec` for `output_cleanup`.

### Changed

- **BREAKING (internal imports):** module paths per `docs/refactor/FINAL_LAYOUT.md`. See `docs/refactor/MIGRATION.md`.
- Response `x-lattice-*` headers via `LatticeHeaderMiddleware` (`proxy/middleware.py`).
- `transform_delta_encode` now controls `delta_encode` (was miswired to `transform_batching`).
- `RuntimeRouter` → `TierClassifier` (`runtime/tier_classifier.py`).
- `lattice.sdk.client` deprecation shim; use `from lattice import LatticeClient` (removed in v1.1).

### Removed

- `core/pipeline.py` (`CompressorPipeline`), `pipeline_v2_wrapper`, v1 schedulers, `core/compiler.py`.
- `optimizer/` package (orchestrators → `transforms/optimizers/`).
- `transforms/prefix_opt.py`, `constraint_lifting.py`, `strategy_selector.py`.
- Information-theoretic `context_selector` variant (submodular canonical).
- `src/lattice/evals/` (canonical evals: `benchmarks/evals/`).
- `proxy/compat_exports.py`.
- Legacy `process(request, ctx)` on IR-native transforms (response-side `output_cleanup` keeps `process(response, ctx)`).
- `benchmarks/evals/cli.py --use-v2-pipeline`.

### Fixed

- `JsonFileIntegration.patch()` raises `AgentNotInstalledError` when agent config is missing.
- `RateLimitTracker` evicts entries older than one hour.
- Duplicated streaming retry logic collapsed into `_stream()`.

### Performance

- Single-pipeline path removes v1+v2 double-pass overhead.
- Parallel CI test runs (~90s vs ~4min sequential on typical hardware).

### Documentation

- `docs/architecture/runtime.md` (renamed from `runtime_v2.md`).
- README, AGENTS.md, CHANGELOG, full `MIGRATION.md` import map.

## [0.1.0] — 2025-04-01

Initial public release.
