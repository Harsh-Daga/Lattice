# Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        LATTICE SYSTEM                            │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐               │
│  │  Agents   │    │   Apps   │    │   SDK Users   │               │
│  │ (Claude,  │    │ (OpenAI  │    │ (LatticeClient│               │
│  │  Cursor,  │    │  SDK)   │    │   in Python)  │               │
│  │  Codex)   │    │         │    │               │               │
│  └────┬──────┘    └────┬─────┘    └───────┬───────┘               │
│       │                │                  │                       │
│       │    lace/unlace │   OPENAI_BASE_   │   LatticeClient       │
│       │    init        │   URL=localhost  │   .chat.completions   │
│       │                │                  │                       │
│       └────────────────┼──────────────────┘                       │
│                        │                                          │
│                ┌───────▼────────┐                                  │
│                │  FASTAPI PROXY  │  :8787                         │
│                │                 │                                │
│                │  /v1/chat/      │                                │
│                │  /v1/messages   │                                │
│                │  /v1/responses  │                                │
│                │  /healthz       │                                │
│                │  /stats         │                                │
│                │  /metrics       │                                │
│                └───────┬────────┘                                  │
│                        │                                          │
│           ┌────────────┼────────────┐                             │
│           ▼            ▼            ▼                             │
│    ┌──────────┐ ┌──────────┐ ┌──────────┐                         │
│    │ Session  │ │ Transform │ │ Semantic │                         │
│    │ Manager  │ │ Pipeline  │ │  Cache   │                         │
│    │          │ │           │ │          │                         │
│    │Memory or │ │18 transforms│ │Exact +  │                         │
│    │  Redis   │ │in priority │ │ Approx   │                         │
│    │          │ │   order    │ │  match   │                         │
│    └────┬─────┘ └─────┬────┘ └────┬─────┘                          │
│         │              │           │                               │
│         └──────────────┼───────────┘                               │
│                        │                                          │
│              ┌─────────▼──────────┐                               │
│              │ DirectHTTPProvider  │                               │
│              │                    │                               │
│              │ ProviderRegistry   │                               │
│              │ ConnectionPools    │                               │
│              │ StreamStallDetect  │                               │
│              │ TACC Controller    │                               │
│              └─────────┬──────────┘                               │
│                        │                                          │
└────────────────────────┼──────────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   LLM PROVIDERS     │
              │  OpenAI  Anthropic  │
              │  Groq    DeepSeek   │
              │  ...17 providers    │
              └─────────────────────┘
```

## Core Modules (v1.0.0 refactor — through Phase 4)

Canonical compression path:

```
content_profiler → UnifiedPlanner → ExecutionPlan → Pipeline.compress → Provider
```

### `lattice.core`

Leaf primitives and observability (session/store/metrics move in Phase 9):

| Module | Purpose |
|--------|---------|
| `config.py` | LatticeConfig with env var binding |
| `context.py` | TransformContext (mutable per-request scratchpad) |
| `result.py` | Result[T,E] monad (Ok/Err) |
| `errors.py` | Typed error hierarchy |
| `segmentation.py` | Semantic segmenter |
| `transform_registry.py` | TransformSpec metadata (→ `transforms/registry.py` in Phase 5) |
| `session.py`, `store.py`, `metrics.py`, … | Session, cache, telemetry (Phase 9) |

Transport types: `lattice.transport`. Pipeline: `lattice.pipeline`.

### `lattice.planner`

| Module | Purpose |
|--------|---------|
| `unified_planner.py` | Sole scheduler — `UnifiedPlanner.plan()` → `ExecutionPlan` |
| `task_classifier.py` | Task class + execution tier heuristics |
| `execution_builder.py` | `build_execution_plan()` for gateway/proxy |
| `runtime_state.py` | Canonical plan/IR metadata bridges |

### `lattice.pipeline`

| Module | Purpose |
|--------|---------|
| `runner.py` | `Pipeline.compress()` / `process()` |
| `gates.py` | Safety gates (policy, budget, PSG, compression limits) |
| `representation_optimizer.py` | Beam search over `transforms/optimizers/` |
| `factory.py` | `build_default_pipeline()` |

### `lattice.transforms` + `transforms/optimizers`

Registered transforms; orchestrators in `transforms/optimizers/` (Phase 4). See [Transforms](transforms.md).

### `lattice.runtime`

`tier_classifier.py` — workload complexity tiers. **Not a provider router.**

### `lattice.providers`

Adapters, transport, `credentials.py`. See [Providers](providers.md).

### `lattice.protocol`

Binary framing, cache planners, stream resume. See [Protocol](protocol.md).

### `lattice.gateway` / `lattice.proxy` / `lattice.integrations`

HTTP compatibility, FastAPI app, agent lace/unlace.

## Data Flow

### Non-streaming request

```
1. Client POST /v1/chat/completions → Proxy
2. Proxy deserializes OpenAI JSON → internal Request
3. SessionManager looks up or creates session
4. `content_profiler` + `UnifiedPlanner` build `ExecutionPlan`; `Pipeline.compress()` runs transforms
5. Semantic cache check (exact → approximate → miss)
6. [cache miss] DirectHTTPProvider dispatches to provider adapter
7. Provider adapter serializes Request → provider-native format
8. HTTP/2 connection pool sends to upstream
9. Provider adapter deserializes response → internal Response
10. Cache stores response
11. Pipeline reverse-pass expands references and reverses transforms
12. Proxy serializes Response → OpenAI JSON
13. Session updated with new messages
14. Response returned with routing headers
```

### Streaming request

```
Same as above, but after step 6:
→ Provider adapter opens SSE stream
→ Each chunk is normalized to OpenAI delta format
→ SSE relay pushes chunks to client
→ On stream completion: session updated, response cached
```

## Thread Safety

- `Pipeline.compress()` is sync; each request gets its own `TransformContext`.
- `SessionManager` uses `asyncio.Lock` for safe concurrent access.
- `SemanticCache` uses `asyncio.Lock` for fingerprint and index operations.
- `StreamStallDetector` uses `threading.Lock` (thread-safe, not async).
- `ConnectionPoolManager` uses rate-limited per-provider client creation.

## Error Handling

All pipeline transforms return `Result[Request, TransformError]`. On failure:
- If `graceful_degradation=true`: rollback to pre-transform state, log warning, continue
- If `graceful_degradation=false`: return error immediately

Provider errors are mapped to HTTP status codes:
- `httpx.TimeoutException` → 504
- `ProviderError` → uses embedded `status_code`
- Generic exception → 502
