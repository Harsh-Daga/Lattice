# TACC — Token-Aware Congestion Control

TACC is LATTICE's adaptive concurrency controller for LLM providers. Inspired by TCP congestion control (AIMD), it manages per-provider request admission using **token pressure** rather than simple request counts — because a 100K-token request is not the same as a 10-token request.

## Why TACC Exists

Naive retry-on-429 creates cascading failures. Static rate limits waste capacity. TACC adapts in real-time to per-provider conditions using four signals:

| Signal | How TACC Reacts |
|--------|-----------------|
| **429 Rate Limit** | Exponential backoff: `ssthresh /= 2`, `window_size = 1`, `blocked_until = now + backoff` |
| **503 / 5xx** | Backpressure: `window_size -= 2.0` to drain the queue |
| **Increased latency** | If `latency > 3.0 × rtt_estimate`: treat as timeout — multiplicative decrease |
| **Stream stall** | Collapse: `ssthresh = window_size / 4.0`, block provider for 2s |
| **Success** | Slow-start: `window_size += 1.0`; congestion avoidance: `window_size += 1.0 / window_size` |

## Architecture

```
incoming request
      │
      ▼
┌─────────────┐     ┌──────────────────┐
│  evaluate   │────▶│  ADMIT           │ → dispatch to provider
│  admission  │     │  DELAY           │ → enqueue with priority
│             │     │  REJECT          │ → 429 response
│             │     │  PRIORITY_DOWNGRADE│ → lower priority, retry
└─────────────┘     └──────────────────┘
      │
      ▼
┌─────────────────────────────┐
│ ProviderCongestionState     │
│  window_size: float         │  Current concurrency window
│  ssthresh: float            │  Slow-start threshold
│  rtt_estimate: float (EWMA) │  Smoothed round-trip time
│  token_rate_estimate: float │  Tokens/second throughput
│  active_token_pressure: float│  Sum of in-flight token estimates
│  blocked_until: float       │  Backoff timer
│  pending_waiters: heapq     │  Priority-ordered request queue
└─────────────────────────────┘
```

## Token Pressure

Unlike request-based concurrency (N concurrent requests), TACC uses **token pressure** — the estimated token count of all in-flight requests. A provider might handle 50 concurrent 100-token requests easily, but struggle with 5 concurrent 100K-token requests. TACC builds a model of per-provider capacity in token/second:

```
token_window_limit = window_size × token_budget_per_request
admission_allowed = active_token_pressure + estimated_tokens ≤ token_window_limit
```

## Priority-Aware Admission

Higher-priority requests get a **token window boost**:

```
boost_factor = 1.0 + priority × 0.05
boosted_window = token_window_limit × boost_factor
```

Waiters are sorted in a **min-heap** by `(-priority, sequence_number)` — highest priority + earliest arrival first.

## Decision Tree

`evaluate_admission()` checks 7 conditions in order:

1. `token_pressure_ok` — tokens fit in window → possible ADMIT
2. `priority_gt_5` — urgent request → ADMIT even if over window
3. `window_size_lt_1` → REJECT
4. `blocked_until_gt_now` → REJECT
5. `token_window_available_some` → DELAY (wait for drain)
6. `priority_lt_3` → REJECT (low priority during congestion)
7. `retry_count_low` → DELAY; else REJECT

## Metrics

```bash
curl http://localhost:8787/stats | jq '.tacc'
```

```json
{
  "openai": {
    "window_size": 100,
    "ssthresh": 60,
    "rtt_estimate_ms": 243.5,
    "token_rate_estimate": 1250.0,
    "active_requests": 5,
    "pending_requests": 2,
    "active_token_pressure": 45000,
    "in_slow_start": false
  }
}
```

## Congestion Control Algorithm

```
initial state:
  window_size = 1.0 (slow-start)
  ssthresh = 16.0

on success:
  if in_slow_start: window_size += 1.0
  else:             window_size += 1.0 / window_size
  if window_size > ssthresh: exit slow-start

on 429:
  ssthresh = window_size / 2.0
  window_size = 1.0
  blocked_until = now + backoff

on stall detected:
  ssthresh = window_size / 4.0  # more aggressive than TCP
  window_size = 1.0
  blocked_until = now + 2.0

on high latency (> 3.0 × rtt_estimate):
  ssthresh = window_size / 2.0
  window_size = max(1.0, window_size / 2.0)

on 503 / 5xx:
  window_size -= 2.0
```

## Cache-Aware Latency

On cache hits, the effective latency is reduced to `max(latency × 0.5, rtt_estimate × 0.8)` — this prevents cache hits from artificially lowering RTT estimates.
