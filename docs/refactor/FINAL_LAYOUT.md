# LATTICE v1.0.0 — Final Directory Layout

> **Purpose.** The exact post-refactor shape of `src/lattice/`, `tests/`, `benchmarks/`, `docs/`, `scripts/`. Every file in this tree has a single, named owner. Every name is honest.
>
> Read this alongside [REFACTOR_PLAN.md](REFACTOR_PLAN.md) when in doubt about where a moved file is supposed to land.

---

## 1. `src/lattice/` — the package

```
src/lattice/
├── __init__.py                         # public re-exports: LatticeClient, LatticeProxyClient, CompressResult, wrap_openai_client, __version__
├── _version.py                         # version string
├── _compat.py                          # py-version shims (tomllib, ParamSpec)
├── client.py                           # canonical local LatticeClient (no proxy needed)
├── ui.py                               # rich-based live display for proxy mode
│
├── cli/                                # CLI — split from monolithic cli.py (1011 LoC) if it grows; today: one module
│   ├── __init__.py
│   └── main.py                         # entry point referenced by [project.scripts] in pyproject.toml
│
├── core/                               # leaf primitives only — no domain logic
│   ├── __init__.py                     # re-exports the leaf types
│   ├── config.py                       # LatticeConfig (pydantic-settings)
│   ├── context.py                      # TransformContext
│   ├── errors.py                       # LatticeError + 15 subclasses
│   ├── result.py                       # Result[T,E], Ok, Err
│   ├── segmentation.py                 # was transforms/semantic_segmenter.py (data structure module)
│   └── tunnel_sidecar.py               # OPTIONAL — may move to integrations/ in v1.0.0
│
├── safety/                             # NEW domain
│   ├── __init__.py
│   └── risk_scoring.py                 # was utils/validation.py — semantic risk score 0-100
│
├── ir/                                 # IR construction, normalisation, serialisation, optimisation
│   ├── __init__.py                     # exports PromptIR, PromptIRV2, build_ir, normalize_ir, serialize_ir, IRTransform
│   ├── types.py                        # was core/ir.py — SectionType, SpanRole, Span, Section, PromptIR
│   ├── primitives.py                   # was core/primitives.py — SpanV2, SectionV2, PromptIRV2, Candidate, CandidateGraph, ExecutionPlan-adjacent v2 types, converters
│   ├── builder.py                      # was core/ir_builder.py + the 3-line core/compiler.py inlined
│   ├── normalizer.py                   # was core/ir_normalizer.py
│   ├── serializer.py                   # was core/ir_serializer.py
│   ├── transform.py                    # was core/ir_transform.py — IRTransform protocol + CandidateSearch
│   ├── semantic_graph.py               # was core/semantic_graph.py
│   ├── native_optimizer.py             # was optimizer/ir_native_optimizer.py — base class
│   ├── validation.py                   # was optimizer/validation.py — centralized rollback checks
│   └── quality.py                      # was optimizer/quality_estimator.py — 8-component quality estimate
│
├── planner/                            # request → ExecutionPlan
│   ├── __init__.py                     # exports ExecutionPlan, RequestClassifier, UnifiedPlanner, build_execution_plan
│   ├── request_classifier.py           # was planner/request_classifier.py (no change)
│   ├── task_classifier.py              # was core/task_classifier.py — TaskClass, ExecutionTier, classify_task
│   ├── execution_plan.py               # was planner/execution_plan.py — immutable ExecutionPlan dataclass
│   ├── execution_builder.py            # was planner/execution_builder.py — build_execution_plan()
│   ├── unified_planner.py              # was core/unified_planner.py — THE planner (canonical)
│   ├── provider_strategy.py            # was planner/provider_strategy.py
│   ├── transport_planner.py            # was planner/transport_planner.py
│   ├── fallback_executor.py            # was planner/fallback_executor.py
│   └── runtime_state.py                # was core/runtime_state.py — bridges (request, context, execution_plan) state
│
├── pipeline/                           # execution-time concerns
│   ├── __init__.py                     # exports Pipeline (the runner) and build_pipeline factory
│   ├── runner.py                       # was core/pipeline_v2.py — THE Pipeline class
│   ├── factory.py                      # was core/pipeline_factory.py — build_default_pipeline, build_benchmark_pipeline
│   ├── policy.py                       # was core/policy.py — OptimizationPolicy (budget, model rules)
│   ├── guardrails.py                   # was core/guardrails.py — entity/format/signal preservation checks
│   ├── milv.py                         # was core/milv.py — multi-input loss validation
│   ├── auto_continuation.py            # was core/auto_continuation.py
│   ├── batch_accumulator.py            # was core/batch_accumulator.py
│   └── representation_optimizer.py     # was optimizer/representation_optimizer.py — beam search over transforms
│
├── transforms/                         # individual transforms — IR-native optimize(ir,...) only; no legacy process() duals
│   ├── __init__.py                     # registry-driven; no explicit imports
│   ├── registry.py                     # was core/transform_registry.py — TransformSpec, build_transform_instance
│   ├── reputation.py                   # was core/transform_reputation.py — TransformReputation
│   ├── patterns.py                     # was utils/patterns.py — pre-compiled regexes
│   │
│   ├── runtime_contract.py
│   ├── cache_arbitrage.py
│   ├── message_dedup.py
│   ├── reference_sub.py
│   ├── tool_filter.py
│   ├── tool_projection.py
│   ├── output_cleanup.py
│   ├── path_prefix.py
│   ├── json_shape.py
│   ├── columnar_pack.py
│   ├── extractive_compress.py
│   ├── rate_distortion.py
│   ├── diagnostic_rle.py
│   ├── causal_chain.py
│   ├── constraint_lifting.py
│   ├── context_selector.py             # submodular only; information_theoretic merged as a strategy flag OR removed (see Phase 4)
│   │
│   ├── batching.py                     # execution-only
│   ├── speculative.py                  # execution-only
│   ├── delta_encode.py                 # execution-only
│   │
│   ├── content_profiler/               # SPLIT from 985-LoC content_profiler.py
│   │   ├── __init__.py                 # ContentProfiler class — entry point
│   │   ├── classifier.py               # ContentProfile enum + classify_by_signals()
│   │   ├── risk_scorer.py              # compute_risk_score() (8-dim)
│   │   ├── task_classifier_bridge.py   # thin call into planner/task_classifier.py
│   │   └── planner_bridge.py           # thin call into planner/unified_planner.py
│   │
│   ├── format_converter/               # SPLIT from 794-LoC format_conv.py
│   │   ├── __init__.py                 # FormatConverter class — dispatch
│   │   ├── table_converter.py          # Markdown ↔ CSV
│   │   └── json_converter.py           # JSON ↔ YAML + nested flattening
│   │
│   ├── strategy_selector/              # OPTIONAL — DELETED unless Phase 4 benchmarks justify
│   │   ├── __init__.py
│   │   └── bandit.py                   # _ArmState + UCB1 selection
│   │
│   └── optimizers/                     # transform orchestrators (was optimizer/)
│       ├── __init__.py                 # registry of orchestrators
│       ├── ir_structure_optimizer.py   # was optimizer/ir_structure_optimizer.py
│       ├── reference_optimizer.py      # was optimizer/reference_optimizer.py
│       ├── tool_optimizer.py           # was optimizer/tool_optimizer.py
│       ├── diagnostic_optimizer.py     # was optimizer/diagnostic_optimizer.py
│       └── context_optimizer.py        # was optimizer/context_optimizer.py
│       # NOTE: optimizer/structure_optimizer.py DELETED — superseded by ir_structure_optimizer.py
│
├── providers/                          # provider adapters + HTTP transport split
│   ├── __init__.py                     # exports DirectHTTPProvider, ProviderRegistry, ConnectionPoolManager, all adapters
│   ├── capabilities.py                 # capability matrix
│   ├── stream_state.py                 # Anthropic streaming state machine
│   ├── tool_sanitizer.py               # base ToolSanitizer + Anthropic + Bedrock subclasses (consolidated from duplicates)
│   ├── schema_filter.py                # JSON schema cleanup
│   ├── mcp_to_anthropic.py             # MCP tool format converter
│   ├── credentials.py                  # was core/credentials.py — provider-scoped
│   │
│   ├── transport/                      # SPLIT from 1539-LoC providers/transport.py
│   │   ├── __init__.py                 # re-exports DirectHTTPProvider, ProviderRegistry, ConnectionPoolManager, _resolve_provider_name
│   │   ├── registry.py                 # ProviderRegistry only
│   │   ├── pool.py                     # ConnectionPoolManager (HTTP/2 fallback)
│   │   ├── rate_limits.py              # RateLimitTracker (with TTL eviction)
│   │   ├── completion.py               # DirectHTTPProvider.completion() — non-streaming
│   │   ├── streaming.py                # DirectHTTPProvider._stream() — merged from two near-duplicate methods
│   │   ├── stall_detector.py           # StreamStallDetector (was providers/stall_detector.py)
│   │   └── helpers.py                  # _resolve_base_url, _resolve_api_key, _parse_sse_line, _optimize_stream_chunk, ...
│   │
│   └── adapters/                       # one file per provider family
│       ├── __init__.py                 # re-exports all 17 adapters
│       ├── base.py                     # ProviderAdapter Protocol + _pop_system, _remap_tool_choice, _strip_provider_prefix
│       ├── openai.py                   # OpenAIAdapter (canonical OpenAI)
│       ├── openai_compatible.py        # 9 OpenAI-compatible (Groq, Together, DeepSeek, Perplexity, Mistral, Fireworks, OpenRouter, Cohere, AI21)
│       ├── anthropic.py                # AnthropicAdapter (Claude + OAuth + tool sanitization + thinking)
│       ├── azure.py                    # AzureAdapter (deployment URL, api-key header)
│       ├── bedrock.py                  # BedrockAdapter (AWS SigV4, Converse API)
│       ├── gemini.py                   # GeminiAdapter + VertexAdapter
│       └── ollama.py                   # OllamaAdapter + OllamaCloudAdapter
│
├── transport/                          # protocol-level transport (NOT HTTP — see providers/transport/)
│   ├── __init__.py                     # exports types + TACC + delta wire + session manager
│   ├── types.py                        # was core/transport.py — Role, Message, Request, Response, Transform, SyncTransform
│   ├── serialization.py                # was core/serialization.py — Message/Request/Response ↔ OpenAI dict
│   ├── congestion.py                   # TACC (was transport/congestion.py)
│   ├── simulation.py                   # TACC simulator
│   ├── delta_wire.py                   # was core/delta_wire.py
│   └── session.py                      # was core/session.py — SessionManager + Session
│
├── protocol/                           # binary framing & resume — UNCHANGED structure
│   ├── __init__.py
│   ├── framing.py, reliability.py, resume.py
│   ├── manifest.py, segments.py
│   ├── content.py, boundaries.py, multiplex.py
│   ├── dictionary_codec.py, dictionary_static.py
│   ├── prefix_canonicalization.py
│   └── cache_planner.py
│
├── state/                              # all stateful persistence
│   ├── __init__.py
│   ├── store.py                        # was core/store.py — RedisSessionStore, MemorySessionStore
│   └── segment_store.py                # already here — cross-session segment dedup
│
├── cache/                              # NEW top-level domain
│   ├── __init__.py
│   └── semantic.py                     # was core/semantic_cache.py — SemanticCache (in-memory + Redis backends)
│
├── telemetry/                          # was empty observability/ — now THE telemetry surface
│   ├── __init__.py                     # exports all telemetry primitives
│   ├── metrics.py                      # was core/metrics.py — LatencyTracker, MetricsCollector
│   ├── telemetry.py                    # was core/telemetry.py — DowngradeCategory, DowngradeTelemetry, TransportOutcome
│   ├── agent_stats.py                  # was core/agent_stats.py
│   ├── cost_estimator.py               # was core/cost_estimator.py
│   ├── streaming_sketches.py           # was utils/streaming_sketches.py — CountMinSketch, HyperLogLog
│   └── maintenance.py                  # was core/maintenance.py — MaintenanceCoordinator
│
├── runtime/                            # workload tier classification (NOT a provider router)
│   ├── __init__.py
│   └── tier_classifier.py              # was runtime/router.py — RENAMED for honesty
│
├── proxy/                              # FastAPI server
│   ├── __init__.py                     # exports create_app, HealthManager
│   ├── bootstrap.py                    # build_proxy_runtime(config) — DI container
│   ├── server.py                       # create_app(config); installs LatticeHeaderMiddleware
│   ├── middleware.py                   # LatticeHeaderMiddleware — sole x-lattice-* response writer
│   ├── routes.py                       # register_health_routes, compat route wiring
│   ├── lifecycle.py                    # PIDManager, start_background_server
│   └── health.py                       # HealthManager — /healthz … /stats handlers delegate here
│
├── gateway/                            # request/response translation — UNCHANGED, already clean
│   ├── __init__.py
│   ├── server.py                       # LLMTPGateway (native LATTICE protocol)
│   ├── compat.py                       # HTTP compatibility for OpenAI/Anthropic/Responses APIs
│   ├── routing.py                      # request signal detection
│   └── detect_helpers.py
│
├── sdk/                                # external client SDKs
│   ├── __init__.py                     # exports LatticeClient, LatticeProxyClient, wrap_openai_client, CompressResult
│   ├── proxy_client.py                 # HTTP client for running proxy
│   └── wrappers.py                     # OpenAI SDK monkey-patch wrapper
│
├── integrations/                       # agent integrations — UNCHANGED structure
│   ├── __init__.py
│   ├── agents.py                       # AgentIntegration base + 6 subclasses (claude/codex/cursor/opencode/copilot/generic)
│   ├── init.py                         # durable setup (detect → patch → store mutation)
│   ├── lace.py                         # transient routing
│   ├── unlace.py                       # restore
│   ├── registry.py                     # list_supported_agents
│   ├── mutation_store.py               # track what was patched
│   ├── claude/    {install.py, runtime.py}
│   ├── codex/     {install.py, runtime.py, auth.py, ws_handler.py}
│   ├── cursor/    {install.py, runtime.py}
│   ├── opencode/  {install.py, runtime.py}
│   └── copilot/   {install.py, runtime.py}
│
└── utils/                              # GENERAL utilities only
    ├── __init__.py
    └── token_count.py                  # tiktoken-based exact + approximate, memoised
    # NOTE: patterns.py moved to transforms/; streaming_sketches.py moved to telemetry/; validation.py moved to safety/
```

**File count summary** (target):

| Domain | Files | Notes |
|---|---|---|
| `core/` | 6 | Down from 42 |
| `safety/` | 2 | New |
| `ir/` | 11 | Up from 0 (consolidated from `core/ir*` + `optimizer/`) |
| `planner/` | 11 | Up from 6 (absorbs `core/` schedulers + `core/task_classifier`) |
| `pipeline/` | 9 | Up from 0 (consolidated from `core/pipeline*` + `core/policy`/`guardrails`/`milv`/`auto_continuation`/`batch_accumulator` + `optimizer/representation_optimizer`) |
| `transforms/` | ~26 | content_profiler / format_converter become packages; prefix_opt deleted; optimizers/ subdir added |
| `providers/` | ~18 | adapters/ subdir + transport/ subdir |
| `transport/` | 6 | Absorbs `core/transport`, `core/serialization`, `core/delta_wire`, `core/session` |
| `protocol/` | 13 | Unchanged |
| `state/` | 3 | Absorbs `core/store` |
| `cache/` | 2 | New (was `core/semantic_cache`) |
| `telemetry/` | 7 | Was empty (`observability/`) |
| `runtime/` | 2 | Renamed file |
| `proxy/` | 6 | Unchanged |
| `gateway/` | 5 | Unchanged |
| `sdk/` | 3 | Unchanged |
| `integrations/` | ~20 | Unchanged |
| `cli/` | 2 | Reorganized |
| `utils/` | 2 | Shrunk to just `token_count.py` |
| **Total** | **~155** | Down from 166 |

---

## 2. `tests/` — mirrors `src/lattice/` exactly

```
tests/
├── unit/
│   ├── core/                {test_config, test_context, test_errors, test_result, test_segmentation}
│   ├── safety/              {test_risk_scoring}
│   ├── ir/                  {test_types, test_primitives, test_builder, test_normalizer, test_serializer, test_transform, test_native_optimizer, test_validation, test_quality}
│   ├── planner/             {test_request_classifier, test_task_classifier, test_execution_plan, test_execution_builder, test_unified_planner, test_provider_strategy, test_transport_planner, test_fallback_executor}
│   ├── pipeline/            {test_runner, test_factory, test_policy, test_guardrails, test_milv, test_auto_continuation, test_batch_accumulator, test_representation_optimizer}
│   ├── transforms/          {test_registry, test_reputation, test_<each_transform>.py}
│   ├── providers/           {test_<each_adapter>.py, test_capabilities, test_tool_sanitizer, test_schema_filter, test_mcp_to_anthropic}
│   ├── providers/transport/ {test_registry, test_pool, test_rate_limits, test_completion, test_streaming, test_stall_detector, test_helpers}
│   ├── transport/           {test_types, test_serialization, test_congestion, test_delta_wire, test_session}
│   ├── protocol/            {test_framing, test_reliability, test_resume, test_manifest, test_segments, ...}
│   ├── state/               {test_store, test_segment_store}
│   ├── cache/               {test_semantic}
│   ├── telemetry/           {test_metrics, test_telemetry, test_agent_stats, test_cost_estimator, test_streaming_sketches, test_maintenance}
│   ├── runtime/             {test_tier_classifier}
│   ├── proxy/               {test_bootstrap, test_server, test_routes, test_lifecycle, test_health}
│   ├── gateway/             {test_server, test_compat, test_routing}
│   ├── sdk/                 {test_client, test_proxy_client, test_wrappers}
│   ├── cli/                 {test_main, test_proxy_commands, test_init, test_lace}
│   └── integrations/        {test_agents, test_init, test_lace, test_<each_agent>.py}
│
├── integration/
│   ├── test_proxy_end_to_end.py
│   ├── test_session_flow.py
│   ├── test_redis_backend.py
│   ├── test_streaming_flow.py
│   ├── test_tacc_admission.py
│   ├── test_delta_wire.py
│   ├── test_binary_protocol.py
│   └── ...
│
├── e2e/
│   ├── test_agent_lace_claude.py
│   ├── test_agent_lace_codex.py
│   └── test_full_pipeline.py
│
├── contract/                            # NEW — protects the public surface (§2 of master plan)
│   ├── test_cli_contract.py             # every subcommand parses + exits with documented code
│   ├── test_http_contract.py            # every /v1/* endpoint accepts + returns documented shape
│   ├── test_headers_contract.py         # every x-lattice-* header still emitted
│   └── test_python_api_contract.py      # from lattice import LatticeClient, ... still works
│
├── security/
│   └── test_security.py
│
└── conftest.py
```

---

## 3. `benchmarks/` — UNCHANGED

```
benchmarks/
├── __init__.py
├── evals/
│   ├── cli.py
│   ├── runner.py
│   ├── surfaces.py
│   ├── live.py
│   ├── replay.py
│   ├── report.py
│   └── catalog.py
├── framework/  {types.py, frontier.py}
├── metrics/    {quality.py}
├── scenarios/  {prompts.py}
├── datasets/   {replay_traces.jsonl, ...}
└── results/    {production_evals.json, production_evals.md, v1.0.0-baseline.json, ...}
```

---

## 4. `docs/` — deduplicated and dressed for v1.0.0

```
docs/
├── index.md                           # landing
├── getting-started/                   # quickstart, install, CLI reference (unchanged)
├── architecture/
│   └── runtime.md                     # was runtime_v2.md — RENAMED, only architecture doc
├── concepts/
│   ├── observability.md
│   ├── safety.md
│   └── sdk.md
│                                      # NOTE: concepts/architecture.md DELETED (duplicate)
│                                      # NOTE: concepts/proxy.md verified-against runtime.md; deleted if duplicate
│
├── novel/                             # deep dives (unchanged)
│   ├── batching-speculation.md
│   ├── binary-framing.md
│   ├── delta-encoding.md
│   ├── streaming.md
│   └── tacc.md
│
├── compression/
│   ├── caching.md
│   ├── protocol.md
│   └── transforms.md
│
├── evaluation/
│   └── benchmarks.md
│
├── providers/
│   └── providers.md
│
├── operations/
│   └── integrations.md
│
└── refactor/                          # THIS directory — kept post-release for historical reference
    ├── REFACTOR_PLAN.md
    ├── FINAL_LAYOUT.md
    ├── MIGRATION.md
    ├── FEATURE_PARITY.md
    └── 00-audit-baseline.md ... 11-docs-release.md
```

---

## 5. `scripts/` — clarified

```
scripts/
├── README.md                          # NEW — describes each script
├── benchmark_compression.py           # dev — manual local compression test
├── benchmark_e2e_through_proxy.py     # dev — manual local proxy benchmark
├── compare_benchmarks.py              # NEW — used by CI gate (§7 of master plan)
└── (profile_format_conv.py + test_e2e_real.py kept ONLY if shown to be still useful in Phase 9)
```

---

## 6. Top-level project files — UNCHANGED structure

```
.
├── AGENTS.md                          # agent-facing dev guide — rewritten to match v1.0 structure (Phase 11)
├── README.md                          # rewritten (Phase 11)
├── CHANGELOG.md                       # NEW for v1.0.0
├── CONTRIBUTING.md
├── LICENSE
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── .python-version
├── .pre-commit-config.yaml
├── pyproject.toml                     # version bumped to 1.0.0; project.scripts unchanged
├── uv.lock
└── repomix-output.txt                 # DELETED — generated artefact, not source
```

---

## 7. What this layout enforces

1. **One concern, one file.** No file >800 LoC. The four files that exceeded this are now packages.
2. **Honest names.** No file named `compiler.py` if it's a 3-line wrapper. No file named `router.py` if it isn't a router. No file named `v2` if there's no `v1`.
3. **Domain locality.** Everything about IR is under `ir/`. Everything about transforms is under `transforms/`. Everything about telemetry is under `telemetry/`. No "core grab-bag".
4. **Pointed dependency direction.** `core/` is a leaf — the rest of the package imports from it. `pipeline/` is the top of the runtime stack. CLI/proxy/SDK sit above pipeline.
5. **Test parity.** Every src module has a tests/ sibling at the same path. New `tests/contract/` proves the user contract.
