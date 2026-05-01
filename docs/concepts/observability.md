# Observability

LATTICE exposes runtime state through three surfaces: `/stats`, `/metrics`, and response headers.

## /stats Endpoint

`GET /stats` returns a comprehensive JSON document with the full runtime state.

```bash
curl http://localhost:8787/stats | jq
```

### Response Structure

| Section | Contents |
|---------|----------|
| `version` | LATTICE version |
| `transforms` | All registered transforms |
| `pipeline` | Token in/out, latency, compression ratio |
| `sessions` | Active session count |
| `provider` | Provider mode |
| `adapters` | Registered provider adapters |
| `capabilities` | Per-provider capabilities (cache mode, features) |
| `pools` | HTTP connection pool count |
| `batching` | Queue size, pending, dispatched counts |
| `speculation` | Hit/miss counts, hit rate |
| `tacc` | Per-provider congestion control state |
| `manifest` | Session manifest statistics |
| `agents` | Per-agent request statistics |
| `fallbacks` | Downgrade event counts (HTTP/2→1.1, delta→full, etc.) |
| `downgrades` | Formal downgrade telemetry with recent reasons |
| `transport` | Per-pool HTTP version and fallback reasons |
| `ignored_chunks` | Stream chunk drop counts per provider |
| `maintenance` | Coordinator state (last attempt, success, failures) |
| `transport_outcome_rollup` | Aggregated transport outcome counts |

## /metrics Endpoint (Prometheus)

`GET /metrics` returns Prometheus exposition format.

```bash
curl http://localhost:8787/metrics
```

### Key Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `lattice_requests_total` | counter | Total proxied requests |
| `lattice_request_latency_ms` | histogram | Request latency distribution |
| `lattice_llm_latency_ms` | histogram | Provider latency distribution |
| `lattice_semantic_cache_hit` | counter | Cache hit count |
| `lattice_semantic_cache_miss` | counter | Cache miss count |
| `lattice_batching_dispatched` | counter | Batched requests |
| `lattice_speculative_hit` | counter | Speculation hits |
| `lattice_speculative_miss` | counter | Speculation misses |
| `lattice_tacc_window` | gauge | Per-provider concurrency window |
| `lattice_tacc_rtt_ms` | gauge | Per-provider RTT estimate |
| `lattice_cost_estimator_request_cost_usd` | gauge | Per-request cost |

## Response Headers

Every proxied response includes routing headers:

| Header | Example | Description |
|--------|---------|-------------|
| `x-lattice-model` | `openai/gpt-4o` | Model used |
| `x-lattice-compression` | `43.21%` | Token reduction |
| `x-lattice-session-id` | `sess_a1b2c3` | Session ID |
| `x-lattice-framing` | `json` | Transport framing |
| `x-lattice-delta` | `delta` | Delta mode |
| `x-lattice-http-version` | `http/2` | HTTP protocol |
| `x-lattice-semantic-cache` | `hit` | Cache result |
| `x-lattice-batching` | `batched` | Batch participation |
| `x-lattice-speculative-status` | `hit` | Speculation result |
| `x-lattice-fallback-reason` | `h2_unavailable` | Optimization bypass reason |
| `x-lattice-stream-resumed` | `true` | Stream resumed |
| `x-lattice-cost-usd` | `0.000234` | Estimated cost |
| `x-lattice-cache-savings-usd` | `0.000110` | Cache cost savings |
| `x-lattice-cached-tokens` | `3400` | Tokens served from cache |

## Downgrade Telemetry

Every non-ideal path is classified into a `DowngradeCategory`:

| Category | Trigger |
|----------|---------|
| `binary_to_json` | Binary framing fell back to JSON |
| `delta_to_full_prompt` | Delta encoding bypassed |
| `http2_to_http11` | HTTP/2 unavailable |
| `stream_resume_to_full` | Stream resume fell back to full request |
| `batching_bypassed` | Batching not applied |
| `speculation_bypassed` | Speculation not applied |
| `semantic_cache_miss` | Cache miss |
| `semantic_cache_disabled` | Cache not enabled |
| `provider_routing_failure` | Provider not found |

These are exposed in `/stats` under `downgrades.counts` and `downgrades.recent_reasons`.

## Maintenance Coordinator

Background maintenance runs every 60 seconds, cleaning up:
- Stale stream state (>5 minutes no activity)
- Expired cache entries

State is visible in `/stats` under `maintenance`:

```json
{
  "maintenance": {
    "last_attempt": 1712345678.0,
    "last_success": 1712345678.0,
    "interval_seconds": 60.0,
    "throttled_tick_count": 0,
    "callback_failures": {},
    "last_result_summary": {
      "stall_detector": { "stale_streams_removed": 3 },
      "semantic_cache": { "stale_cache_entries_removed": 12 }
    }
  }
}
```

## Logging

Structured logging via `structlog`. Key events:

- `proxy_request` — Every HTTP request
- `transform_applied` — Successful transform execution
- `transform_skipped_config` — Disabled in config
- `transform_skipped_runtime_budget` — Budget exhausted
- `transform_blocked_by_risk_gate` — Risk gate blocked
- `transform_expansion_aborted` — Expansion guardrail triggered
- `semantic_cache_hit` / `semantic_cache_miss` — Cache events
- `maintenance_tick_did_work` — Maintenance ran
