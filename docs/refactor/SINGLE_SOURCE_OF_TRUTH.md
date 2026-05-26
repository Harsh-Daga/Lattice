# LATTICE Single Source of Truth Registry

> **Purpose.** Every primitive, type, algorithm, policy, or configurable behaviour in LATTICE lives in **exactly one file**. This document is that registry.
>
> **CI gate.** `scripts/check_internal_no_duplication.sh` walks every entry below and verifies:
> 1. The named symbol exists at the declared path.
> 2. The symbol does not exist (as a class / function / module-level constant) anywhere else in the repo except inside `tests/`, `benchmarks/`, and explicitly-allowlisted thin re-exports.
> 3. The declared path is under the declared module's directory cap ([FORWARD_PLAN.md §6.1](FORWARD_PLAN.md)).
>
> Every PR that introduces a new primitive **must update this registry in the same PR**. The CI gate blocks merge otherwise.
>
> **Reading guide.** Each row reads: *"There is exactly one X. It lives in Y. Every other surface that needs X imports it from Y or consumes it across the FFI boundary in [Phase 31](31-edge-wasm-core.md). If you find yourself writing a second implementation of X, stop."*

---

## 1. Core domain types

| Primitive | Canonical home | Notes / phase |
|---|---|---|
| `Result[T, E]` | `src/lattice/core/result.py` | Pre-existing. |
| `LatticeError` hierarchy | `src/lattice/core/errors.py` | Pre-existing. |
| `LatticeConfig` (root) | `src/lattice/core/config.py` | Pre-existing. Sub-configs declared here too. |
| `RequestContext` / `TransformContext` | `src/lattice/core/context.py` | Pre-existing. |
| `SegmentationStrategy` enum + types | `src/lattice/core/segmentation.py` | Pre-existing. |
| `Request`, `Response`, `Message`, `ChunkPayload` types | `src/lattice/transport/types.py` | Pre-existing (Phase 2a). |
| `PromptIRV2` immutable type | `src/lattice/ir/v2.py` | Pre-existing. Built via `src/lattice/ir/builder.py`. |
| `ExecutionPlan` | `src/lattice/ir/primitives.py` | **Phase 13** — sole type; `planner` re-exports. Session metadata uses `planner/session_plan.py::SessionExecutionPlan`. |
| `composite_score` / candidate scoring | `src/lattice/ir/scoring.py` | **Phase 13** — `Candidate.score` and `CandidateScorer` delegate here. |
| `AliasTable` (PII + reference_sub merged) | `src/lattice/transforms/alias_table.py` | **Phase 21** merges PII alias entries into the existing alias table; never two parallel tables. |

## 2. Fingerprints, hashing, canonical serialization

| Primitive | Canonical home | Notes / phase |
|---|---|---|
| `xxh3_128_hexdigest` wrapper | `src/lattice/util/hashing.py` | Pre-existing-or-new in Phase 16. Wraps `xxhash` (Python) and routes to native via Phase 31's `lattice_core_py.xxh3` when present. |
| Canonical IR serialization | `src/lattice/ir/canonical_serialize.py` | One implementation. Rust port in `crates/lattice-core/src/ir/serialize.rs` is byte-identical, parity-tested. |
| IR fingerprint | `src/lattice/cache/fingerprint.py::ir_canonical_fingerprint` | **Phase 14**. Rust mirror in `crates/lattice-core/src/ir/fingerprint.rs`. |
| Provider-invariant fingerprint | same file, `provider_invariant_fingerprint` | **Phase 17**. |
| Request SHA-256 (Tier 1 cache) | `src/lattice/cache/layers/exact.py::exact_request_hash` | **Phase 14**. |
| Jaccard signature | `src/lattice/cache/layers/jaccard.py::jaccard_signature` | **Phase 20** (lifted from current `cache/semantic.py`). |
| Tenant namespace resolution | `src/lattice/cache/namespace.py::resolve_tenant` | **Phase 14**. |
| Receipt JTI / signature | `src/lattice/audit/receipts.py` | **Phase 29**. Reuses `util/hashing.py`. |
| Bearer-key hash | `src/lattice/keys/manager.py::_hash_bearer` | **Phase 32**. SHA-256; never collides. |

## 3. Transforms

| Transform | Canonical home | Notes |
|---|---|---|
| `reference_sub` | `src/lattice/transforms/reference_sub.py` | Rust acceleration via Phase 31. |
| `path_prefix` | `src/lattice/transforms/path_prefix.py` | Same. |
| `format_conversion` | `src/lattice/transforms/format_converter/` | Phase 5b split; single optimizer entry point. |
| `tool_filter` | `src/lattice/transforms/tool_filter.py` | |
| `output_cleanup` | `src/lattice/transforms/output_cleanup.py` | Rust acceleration via Phase 31. |
| `rate_distortion` | `src/lattice/transforms/rate_distortion.py` | |
| `delta_encode` | `src/lattice/transforms/delta_encode.py` | |
| `tool_diff` | `src/lattice/transforms/tool_diff/` | **Phase 27**. |
| `llmlingua` | `src/lattice/transforms/llmlingua/` | **Phase 27**. Opt-in `[llmlingua]` extra. |
| `embedding_dedup` | `src/lattice/transforms/embedding_dedup.py` | **Phase 31**. |
| Transform registry | `src/lattice/transforms/registry.py` | One source. `pipeline/runner.py` factory generated from it (Phase 13). |
| Reverse-pass entry | `src/lattice/pipeline/reverse.py` | Calls each transform's `reverse(response, ctx)`. **Never** re-implemented in any SDK. |

## 4. Pipeline + planning + safety

| Primitive | Canonical home | Notes / phase |
|---|---|---|
| `Pipeline` (sole runner) | `src/lattice/pipeline/runner.py` | Pre-existing (Phase 3). |
| Gates (policy / risk / budget / guardrails / post-transform) | `src/lattice/pipeline/gates.py` | One module; sub-gate files compose. |
| Streaming chunk buffer | `src/lattice/pipeline/streaming/chunk_buffer.py` | **Phase 27**. Rust mirror in `crates/lattice-core/src/streaming/`. TS edge consumes via WASM only. |
| `UnifiedPlanner` (sole scheduler) | `src/lattice/planner/unified_planner.py` | Pre-existing (Phase 4). |
| Task classifier | `src/lattice/planner/task_classifier.py` | |
| Beam-search representation optimizer | `src/lattice/pipeline/representation_optimizer.py` | One implementation; Phase 27 extends with LLMLingua branch. |
| Thompson bandit | `src/lattice/planner/bandit/thompson.py` | **Phase 29**. Pure numpy. |
| Profile registry | `src/lattice/policy/profiles.py` | **Phase 28**. |
| Step classifier (agent loops) | `src/lattice/agent/step_classifier.py` | **Phase 34**. |
| Step profile overrides | `src/lattice/agent/step_profiles.py` | **Phase 34**. Layers on top of `policy/profiles.py`. |
| Risk score | `src/lattice/safety/risk_scoring.py` | Pre-existing. |
| Post-transform guard (was MILV) | `src/lattice/pipeline/post_transform_guard.py` | **Phase 13** rename; shared checks in `pipeline/checks.py`. |
| PII detector (rule, default) | `src/lattice/safety/pii/rule_detector.py` | **Phase 21**. |
| PII tokenizer (reversible via Alias Table) | `src/lattice/safety/pii/tokenizer.py` | **Phase 21**. Uses §1 `AliasTable`. |
| Injection detector (heuristic, default) | `src/lattice/safety/injection/heuristic_detector.py` | **Phase 21**. |
| Output validator + repair | `src/lattice/safety/output/validator.py` + `repair_v2.py` | **Phase 21 + Phase 27**. |

## 5. Cache layers + storage

| Primitive | Canonical home | Notes / phase |
|---|---|---|
| `SemanticCache` orchestrator | `src/lattice/cache/semantic.py` | Post-Phase-12 split. Composes layer objects only. |
| Layer interface `CacheLayer` | `src/lattice/cache/layers/__init__.py` | One protocol. |
| Exact layer | `src/lattice/cache/layers/exact.py` | |
| IR layer | `src/lattice/cache/layers/ir_fingerprint.py` | **Phase 14**. |
| Jaccard layer | `src/lattice/cache/layers/jaccard.py` | |
| Embedding layer | `src/lattice/cache/layers/embedding.py` | **Phase 14**. Default backend uses §6 `EmbeddingBackend` → user's provider. |
| Vector store interface | `src/lattice/cache/vector_store/base.py` | **Phase 14**. |
| Vector store: memory / redis / postgres | `src/lattice/cache/vector_store/{memory,redis,postgres}.py` | |
| Content-addressable blob store | `src/lattice/cache/cas_store.py` | **Phase 14**. |
| Portability manifest | `src/lattice/cache/portability/manifest.py` | **Phase 17**. |
| Cache warmer | `src/lattice/cache/warmer.py` | **Phase 17**. |
| Cache analyzer | `src/lattice/cache/analyzer.py` | **Phase 17**. |
| Embeddings cache (per-text) | `src/lattice/cache/embeddings_cache.py` | **Phase 31**. Reused by Phase 14 embedding tier. |

## 6. Embedding backends + agent memory

| Primitive | Canonical home | Notes / phase |
|---|---|---|
| `EmbeddingBackend` protocol | `src/lattice/cache/embeddings/base.py` | **Phase 14**. |
| User-provider backend (default) | `src/lattice/cache/embeddings/user_provider.py` | **Phase 14**. Uses §7 dispatcher. |
| Local SentenceTransformer (opt-in) | `src/lattice/cache/embeddings/local.py` | **Phase 14**. `[embeddings-local]` extra. |
| Ollama backend | `src/lattice/cache/embeddings/ollama.py` | **Phase 14**. |
| Relevance scorer (rule, default) | `src/lattice/agent/relevance.py::RuleRelevanceScorer` | **Phase 34**. |
| Relevance scorer (embedding, opt-in) | same file, `EmbeddingRelevanceScorer` | **Phase 34**. Reuses §6 backend. |
| Context GC | `src/lattice/agent/context_gc.py` | **Phase 34**. |
| Summarizer | `src/lattice/agent/summarizer.py` | **Phase 34**. Uses §7 dispatcher. |
| Token budget allocator | `src/lattice/agent/budget.py` | **Phase 34**. |
| Inference-aware retry strategy | `src/lattice/agent/retry.py` | **Phase 34**. **Disjoint from §7 transport retry** — agent-retry decides *whether* to retry (refusal / truncation / repair); transport retry decides *how* (backoff, breaker). |
| Loop state | `src/lattice/agent/loop_state.py` | **Phase 34**. |

## 7. Transport layer (Phase 14 establishes the canonical set)

| Primitive | Canonical home | Notes |
|---|---|---|
| Unified provider dispatcher | `src/lattice/transport/dispatcher.py` | **Phase 14**. Replaces per-adapter dispatch. |
| Connection pool (per provider) | `src/lattice/transport/pool.py` | **Phase 14**. HTTP/2 multiplexed when supported. |
| Retry policy | `src/lattice/transport/retry.py` | **Phase 14**. **Single implementation**. Adapters declare per-error-class policy; the runner is one file. |
| Circuit breaker | `src/lattice/transport/circuit_breaker.py` | **Phase 14**. |
| Timeout policy | `src/lattice/transport/timeout.py` | **Phase 14**. Per (operation, model-family) defaults; per-request override via header. |
| Backpressure / queue | `src/lattice/transport/backpressure.py` | **Phase 14**. Bounded queue with explicit overflow strategy. |
| Stream resumption | `src/lattice/transport/stream_resume.py` | **Phase 14**. |
| Transport metrics | `src/lattice/transport/metrics.py` | **Phase 14**. RTT / queue depth / breaker state / in-flight. Feeds §10 OTel + §11 receipts. |
| TACC congestion control | `src/lattice/transport/tacc.py` | Re-export of `transport/congestion.py`; canonical under transport. |
| Rate-limit header parsing | `src/lattice/transport/rate_limit.py` | **Phase 14**. |
| Provider registry / routing | `src/lattice/transport/registry.py` | **Phase 14**. |
| Stall detector (streaming) | `src/lattice/transport/stall_detector.py` | **Phase 14**. |
| Binary framing (LATT) | `src/lattice/protocol/framing.py` (Python fallback) + `crates/lattice-core/src/framing/` (canonical Rust) | Phase 31 Rust port. |
| Delta wire | `src/lattice/transport/delta_wire.py` | Pre-existing (Phase 2a). |
| Provider adapter base | `src/lattice/providers/adapters/base.py` | **Phase 14**: adapters are pure protocol shaping — **no transport code**, no httpx clients. |
| Adapter retry policy declarations | `src/lattice/providers/adapters/retry_policies.py` | **Phase 14**. Per-provider ``RetryPolicy`` constants; ``RetryEngine`` is the sole runner. |
| Transport per-request telemetry | `src/lattice/transport/telemetry.py` | **Phase 14**. RTT / attempt / pool utilization for response headers. |

## 8. Providers + cost estimation

| Primitive | Canonical home | Notes |
|---|---|---|
| Provider profiles (limits / pricing) | `src/lattice/providers/profiles.py` | One source. |
| Cost estimator | `src/lattice/telemetry/cost_estimator.py` | **One implementation.** Phase 28 receipts + Phase 32 quotas both consume from here; no second estimator anywhere. |
| Credentials | `src/lattice/providers/credentials.py` | Pre-existing (Phase 4 move). |
| Provider registry | `src/lattice/providers/__init__.py` | Adapters registered once. |

## 9. State

| Primitive | Canonical home | Notes |
|---|---|---|
| `Session` | `src/lattice/state/session.py` | Pre-existing. Phase 34 adds `memory_policy`; Phase 34 adds `loop_state`. |
| `SessionStore` | `src/lattice/state/store.py` | One store. |
| `SegmentStore` | `src/lattice/state/segment_store.py` | One store. |
| Mutation store (integrations) | `src/lattice/integrations/mutation_store/` | Phase 8 move. |

## 10. Telemetry + observability

| Primitive | Canonical home | Notes |
|---|---|---|
| Metrics collector | `src/lattice/telemetry/metrics.py` | One source. |
| Downgrade taxonomy | `src/lattice/telemetry/downgrade.py` | |
| Agent stats | `src/lattice/telemetry/agent_stats.py` | |
| OTel exporter | `src/lattice/telemetry/otel/` | **Phase 22**. Opt-in. |
| Streaming sketches | `src/lattice/telemetry/streaming_sketches.py` | |
| Health manager | `src/lattice/proxy/health.py` (or current path) | Phase 7. |
| Response headers (`x-lattice-*`) | `src/lattice/proxy/middleware.py` | **One emitter.** Adding a header = touching this file. |

## 11. Audit + auth + multi-user

| Primitive | Canonical home | Notes |
|---|---|---|
| Receipts | `src/lattice/audit/receipts.py` + `receipt_store.py` + `receipt_router.py` | **Phase 28**. HMAC-signed; never contain user content. |
| AuthContext | `src/lattice/auth/context.py` | **Phase 32**. Opt-in. |
| Auth backends | `src/lattice/auth/backends/{sqlite,postgres,jwt_external}.py` | **Phase 32**. Default: SQLite. |
| Virtual-key manager | `src/lattice/keys/manager.py` | **Phase 32**. |
| Key encryption (Fernet) | `src/lattice/keys/encryption.py` | **Phase 32**. |
| Quota limiter | `src/lattice/quotas/limiter.py` | **Phase 32**. |
| Tenant policy | `src/lattice/tenants/policy.py` | **Phase 32**. |

## 12. Proxy + gateway + SDKs

| Primitive | Canonical home | Notes |
|---|---|---|
| FastAPI app factory | `src/lattice/proxy/server.py` | One server. |
| Middleware (auth / tenant / headers / OTel start) | `src/lattice/proxy/middleware.py` | One emitter. |
| OpenAI-compat handler | `src/lattice/gateway/compat/openai_chat.py` | Phase 13 split. |
| Anthropic-compat handler | `src/lattice/gateway/compat/anthropic_messages.py` | Phase 13 split. |
| Embeddings endpoint | `src/lattice/gateway/embeddings.py` | **Phase 31**. |
| Batch endpoint | `src/lattice/gateway/batch.py` | **Phase 31**. |
| Audio endpoints | `src/lattice/gateway/audio.py` | **Phase 31**. |
| Files endpoint | `src/lattice/gateway/files.py` | **Phase 31**. |
| Realtime endpoint | `src/lattice/gateway/realtime.py` | **Phase 31**. |
| MCP server | `src/lattice/mcp/server.py` | **Phase 26**. |
| MCP dispatcher | `src/lattice/mcp/dispatcher.py` | **Phase 26**. |
| `LatticeProxyClient` (Python) | `src/lattice/sdk/proxy_client.py` | One async client. |
| `SyncLatticeProxyClient` (Python) | `src/lattice/sdk/sync_proxy_client.py` | One sync client. Both call into one body-builder (`sdk/_proxy_protocol.py`). |
| `wrap_openai` / `wrap_anthropic` (Python) | `src/lattice/sdk/wrappers.py` | One file. Proxy mode = URL redirect. In-process mode delegates to `sdk/_in_process.py` which delegates to runtime. **No algorithm code.** |
| `wrap_openai_in_process` | `src/lattice/sdk/_in_process.py` | Calls `Pipeline.compress()` / `Pipeline.reverse_response()` — never re-implements. |
| `LatticeClient` (TypeScript) | `packages/typescript-sdk/src/client.ts` | One typed HTTP client. |
| `wrapOpenAI` (TypeScript, proxy mode) | `packages/typescript-sdk/src/wrappers/openai.ts` | URL redirect only. |
| `wrapOpenAIInProcess` (TypeScript, edge mode) | `packages/typescript-sdk/src/in_process.ts` | Delegates to `@lattice/core-wasm` for every algorithm call. |
| OpenAPI source of truth | `openapi/lattice-proxy.yaml` | **Phase 24**. Both Python TypedDicts and TS types are generated from this. |

## 13. CLI + integrations

| Primitive | Canonical home | Notes |
|---|---|---|
| `lattice` CLI root | `src/lattice/cli/__init__.py` | One Typer/Click root. Subcommands registered, not re-declared. |
| `lattice doctor` | `src/lattice/cli/doctor.py` | Pre-existing. |
| `lattice cache {warm,analyze}` | `src/lattice/cli/cache.py` | **Phase 17**. |
| `lattice {user,key,quota,usage}` | `src/lattice/auth/admin_cli.py` | **Phase 32**. |
| `lattice mcp {serve,add,list,remove}` | `src/lattice/mcp/cli.py` | **Phase 26**. |
| Per-agent integration | `src/lattice/integrations/agents/{claude,codex,cursor,opencode,copilot}.py` | One per agent. Common base in `integrations/agents/base.py`. |
| Tunnel sidecar | `src/lattice/integrations/tunnel.py` | Phase 8. |

## 14. Configuration

| Primitive | Canonical home | Notes |
|---|---|---|
| Root config loader | `src/lattice/core/config.py::LatticeConfig.from_env` + `from_file` | **One loader.** CLI / proxy / SDK / tests all use this. |
| Hot reload | `src/lattice/config/reload.py` | **Phase 28**. Atomic. |
| Env-var schema | docstrings on `LatticeConfig` fields | Single source for documentation generation. |

---

## 15. Allowlisted thin re-exports (NOT duplications)

The CI gate ignores these specific re-export points:

| Re-export location | Re-exports from | Why allowed |
|---|---|---|
| `src/lattice/__init__.py` | All of the above | Public Python surface. Pure `from X import Y; __all__.append("Y")`. |
| `packages/typescript-sdk/src/index.ts` | `client.ts`, `wrappers/*.ts` | Public TS surface. No logic. |
| `src/lattice/sdk/__init__.py` | `sdk/proxy_client.py`, `sdk/wrappers.py`, etc. | Convenience. |
| `bindings/python/lattice_core_py/__init__.py` | PyO3 generated symbols | Native binding surface only. |

Any file that does anything beyond `from X import Y` for a primitive listed above is treated by CI as a duplication candidate and must be reviewed.

---

## 16. How to update this registry

1. Adding a new primitive in your phase doc? Add the row here in the same PR.
2. Moving a primitive in a phase? Update the canonical-home column here in the same PR.
3. Renaming a primitive? Update here + add a redirect note for one release in [MIGRATION.md](MIGRATION.md).
4. Splitting a file? Update the canonical-home column to the new file path; if the split creates multiple primitives, list each.
5. CI gate fails on:
   - A symbol named in this registry that doesn't exist at its declared path.
   - A symbol named in this registry that exists at a *second* path (outside `tests/` / `benchmarks/` / allowlisted re-exports).
   - A new function / class in `src/lattice/` that mirrors a registry symbol's signature but lives elsewhere (heuristic match by symbol name + arity).

---

## 17. Glossary of "would-be duplications" the registry exists to prevent

These are the patterns that historically grow LATTICE codebases until they become unmaintainable. Each is named here so a reviewer can call it out:

| Pattern | Where it would go wrong | How the registry prevents it |
|---|---|---|
| Per-adapter retry logic | 17 adapters × 50 LoC each = 850 LoC of nearly-identical code | §7 puts retry in `transport/retry.py`; CI rejects `httpx.AsyncClient(` outside `transport/` |
| Cost estimation in N places | Receipts have one number, quotas have another, dashboard has a third | §8 names exactly one cost estimator; CI rejects new `def estimate_cost` outside it |
| Multiple config loaders | CLI reads YAML, proxy reads env, SDK reads `~/.lattice.toml` independently | §14 names exactly one loader |
| Multiple session shapes | `state/session.py` has one; SDK has another; MCP server has another | §9 names one Session; CI rejects new `@dataclass class Session` |
| Multiple fingerprint formulas | request hash vs IR fingerprint vs Jaccard signature drift | §2 names each by purpose and home; reviewers can spot a fourth |
| Multiple chunk buffers | Python proxy has one, TS edge has another, both subtly different | §4 names one canonical home (Rust) with Python fallback only; SDK CI rejects implementations |
| Multiple receipt schemas | OTel span attributes diverge from receipt JSON diverge from headers | §11 says: receipts here, OTel here, headers here — three views of one data model in `audit/receipts.py` |
| Multiple bandit reward functions | Phase 28 bandit and Phase 34 step-classifier compete | §4 names one bandit; Phase 34's step classifier feeds the SAME bandit, never a parallel one |
| Multiple "is_streaming" detectors | scattered checks across pipeline/gateway/transport | one helper in `transport/types.py::Request.is_streaming` |

Every entry here is a **named anti-pattern**. When in doubt, a contributor checks this list and confirms their PR doesn't reintroduce one of them.
