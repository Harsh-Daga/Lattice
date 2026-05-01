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

## Core Modules

### `lattice.core`

| Module | Purpose |
|--------|---------|
| `transport.py` | Request, Response, Message data models |
| `pipeline.py` | CompressorPipeline orchestrator, ReversibleSyncTransform base class |
| `pipeline_factory.py` | Default pipeline construction with all transforms |
| `session.py` | SessionManager, MemorySessionStore |
| `store.py` | RedisSessionStore |
| `config.py` | LatticeConfig with env var binding |
| `telemetry.py` | TransportOutcome, DowngradeTelemetry |
| `maintenance.py` | MaintenanceCoordinator with background loop |
| `semantic_cache.py` | SemanticCache with hybrid exact/approximate matching |
| `metrics.py` | MetricsCollector (Prometheus) |
| `result.py` | Result[T,E] monad (Ok/Err) |
| `serialization.py` | request_to_dict, message_to_dict, response_to_dict |
| `delta_wire.py` | Delta encoding/decoding for session reuse |
| `cost_estimator.py` | Per-provider cost estimation |

### `lattice.transforms`

18 transforms running in priority order. See [Transforms](transforms.md).

### `lattice.providers`

Per-provider adapters. See [Providers](providers.md).

### `lattice.protocol`

Binary framing, cache planners, stream resume, dictionary codec. See [Protocol](protocol.md).

### `lattice.gateway`

HTTP compatibility handlers: OpenAI, Anthropic, Responses API passthrough, routing headers.

### `lattice.proxy`

FastAPI app factory, lifecycle management, operational routes, agent routing.

### `lattice.integrations`

Agent config injection and routing for Claude Code, Cursor, Codex, OpenCode, Copilot.

## Data Flow

### Non-streaming request

```
1. Client POST /v1/chat/completions → Proxy
2. Proxy deserializes OpenAI JSON → internal Request
3. SessionManager looks up or creates session
4. Transform pipeline processes Request (in priority order)
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

- `CompressorPipeline.process()` is async-safe. Each request gets its own `TransformContext`.
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
