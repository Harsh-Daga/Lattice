# Proxy

The LATTICE proxy is a FastAPI server that accepts standard OpenAI API requests and routes them through the compression pipeline to your configured LLM provider.

## Starting the Proxy

```bash
# Foreground (blocks terminal)
lattice proxy run --port 8787

# Background daemon
lattice proxy start --port 8787

# Status
lattice proxy status

# Stop
lattice proxy stop
```

## Endpoints

### Chat Completions

```
POST /v1/chat/completions
```

Fully OpenAI-compatible. Accepts `model`, `messages`, `temperature`, `max_tokens`, `top_p`, `stream`, `tools`, `tool_choice`, `stop`, and all standard parameters.

### Additional Headers

| Header | Purpose |
|--------|---------|
| `x-lattice-session-id` | Reuse a session across requests |
| `x-lattice-provider` | Explicitly specify provider |
| `x-lattice-disable-transforms` | Bypass all transforms |
| `x-lattice-client-profile` | Agent profile name |

### Response Headers

| Header | Description |
|--------|-------------|
| `x-lattice-model` | Model used |
| `x-lattice-compression` | Compression ratio achieved |
| `x-lattice-session-id` | Session identifier |
| `x-lattice-framing` | `native` or `json` |
| `x-lattice-delta` | `delta` or `bypassed` |
| `x-lattice-http-version` | HTTP protocol version |
| `x-lattice-semantic-cache` | Cache status |
| `x-lattice-speculative-status` | Speculation result |
| `x-lattice-fallback-reason` | Why optimization was bypassed |

### Health & Diagnostics

```
GET  /healthz        # {"status": "healthy", "version": "0.1.0"}
GET  /readyz         # Provider, pipeline, pool readiness
GET  /stats          # Full runtime state (transforms, sessions, cache)
GET  /metrics        # Prometheus exposition format
GET  /cache/stats    # Semantic cache statistics
POST /cache/clear    # Clear semantic cache

GET  /providers/capabilities  # All provider capabilities
```

### Anthropic Passthrough

```
POST /v1/messages    # Anthropic Messages API passthrough
```

### OpenAI Responses Passthrough

```
POST /v1/responses   # OpenAI Responses API passthrough
POST /v1/responses/{id}
GET  /v1/models      # Model list passthrough
```

## Configuration

Configuration via environment variables (prefix `LATTICE_`) or config file:

| Variable | Default | Description |
|----------|---------|-------------|
| `LATTICE_PROXY_PORT` | 8787 | Listen port |
| `LATTICE_PROXY_HOST` | 0.0.0.0 | Bind address |
| `LATTICE_PROXY_WORKERS` | auto | Uvicorn workers |
| `LATTICE_PROXY_RELOAD` | false | Auto-reload (dev only) |
| `LATTICE_PROVIDER_BASE_URL` | — | Upstream provider URL |
| `LATTICE_PROVIDER_BASE_URLS` | — | JSON `{"provider":"url"}` map |
| `LATTICE_SESSION_STORE` | memory | `memory` or `redis` |
| `LATTICE_REDIS_URL` | — | Redis URL |
| `LATTICE_SESSION_TTL_SECONDS` | 3600 | Session expiry |
| `LATTICE_GRACEFUL_DEGRADATION` | true | Continue on transform failure |
| `LATTICE_COMPRESSION_MODE` | balanced | `safe` / `balanced` / `aggressive` |

## Compression Modes

| Mode | Behavior |
|------|----------|
| **safe** | Only SAFE transforms. No risk of meaning change. |
| **balanced** | SAFE + CONDITIONAL at LOW/MEDIUM risk. Default. |
| **aggressive** | All transforms where risk permits. Best savings. |

## Lifecycle

```
proxy start → double-fork daemon → PID file → background uvicorn
proxy stop  → SIGTERM (grace period) → SIGKILL (force)
proxy restart → stop → start
proxy status → PID, uptime, health check
```

## Security

- **Local-only mode**: Set `LATTICE_PROXY_HOST=127.0.0.1` to reject non-local requests.
- **API keys**: Forwarded from client requests. Never stored.
- **CORS**: Restricted to localhost in daemon mode.

## Stats Output

The `/stats` endpoint returns a comprehensive JSON document:

```json
{
  "version": "0.1.0",
  "transforms": ["content_profiler", "runtime_contract", ...],
  "pipeline": { "tokens_in": 5000, "tokens_out": 2000, ... },
  "sessions": 12,
  "pools": { "openai:https://api.openai.com": { "http_version": "http/2" } },
  "batching": { "queue_size": 0, "pending": 0 },
  "speculation": { "hits": 5, "misses": 2 },
  "tacc": { "openai": { "window_size": 100, "rtt_estimate_ms": 42 } },
  "manifest": { "sessions_with_manifest": 3 },
  "agents": { "claude_code": { "requests": 100 } },
  "fallbacks": { "http2_to_http11_count": 0, "delta_to_full_prompt_count": 2 },
  "downgrades": { "counts": { "http2_to_http11": 1 }, "recent_reasons": {} },
  "transport": { "pools": {} },
  "ignored_chunks": { "total": 0, "by_provider": {} },
  "maintenance": { "last_attempt": 1234567.0, "last_success": 1234567.0 },
  "transport_outcome_rollup": {}
}
```
