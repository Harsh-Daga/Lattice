# Streaming Architecture

LATTICE's streaming subsystem provides stall detection, resumable streams, and adaptive tolerance across 17 providers — each with unique latency profiles.

## Stream Stall Detection

Proactive detection of hung streams using three orthogonal signals: **time-based tolerance**, **token velocity**, and **fallback timeout**.

### Per-Provider Tolerance

Each provider has a custom silence tolerance derived from empirical measurement:

| Provider | Baseline (ms) | Reasoning |
|----------|--------------|-----------|
| `groq` | 15,000 | LPU inference, very fast |
| `ollama` | 20,000 | Local, predictable |
| `openai` | 30,000 | Standard API latency |
| `azure` | 30,000 | Same as OpenAI |
| `gemini` | 30,000 | Google AI Studio |
| `anthropic` | 45,000 | Extended thinking can take time |
| `bedrock` | 45,000 | AWS managed, variable |
| default | 30,000 | Conservative |

### Phase-Aware Multipliers

Tolerance adjusts based on the current stream phase:

| Phase | Multiplier | Rationale |
|-------|-----------|-----------|
| `first_chunk` | 1.5× | Time-to-first-token has higher variance |
| `streaming` | 1.0× | Baseline during active streaming |
| `think` | 2.0× | Reasoning/thinking phases are inherently slow |
| `tool_call` | 1.2× | Tool execution adds moderate latency |

### Velocity-Based Stall Detection

Beyond time thresholds, LATTICE tracks **token velocity** — the rate of new tokens per second. After 3+ chunks establish a baseline:

```
if velocity_current < 0.2 × velocity_baseline → velocity stall
```

This catches cases where the provider is still emitting data but at a trickle (e.g., a model stuck in a repetition loop).

### Per-Stream Isolation

Each concurrent stream gets independent state: own chunk count, token count, velocity, and phase. A stall on one stream doesn't affect others from the same provider. Unknown or mis-threaded stream IDs are silently ignored (counter exposed in `/stats`).

### Strict Mode

For tests and debug builds:

```python
detector = StreamStallDetector(strict_mode=True)
# Raises RuntimeError on unknown stream_id instead of silently ignoring
```

## Stream Resume

HMAC-signed tokens enable interrupted streams to resume without replaying the entire conversation.

### How It Works

```
1. Client subscribes to SSE stream at /v1/chat/completions?stream=true
2. Chunks 0-47 arrive, then connection drops
3. Client sends RESUME frame with HMAC-signed token
4. Server validates HMAC(secret, stream_id || last_seq)
5. Server replays chunks from 48 onward via ReplayWindow
6. If chunks 48+ have fallen out of the window → full replay needed
```

### Replay Window

A **circular buffer** (capacity: 1000 chunks) retains recent chunks:
- Each chunk gets a monotonically increasing sequence number
- `replay_from(seq)` returns all chunks ≥ seq from the buffer
- Returns empty if seq < oldest available (window expired)

### Security

- Resume tokens are HMAC-SHA256 signed with a server secret
- Tokens are single-use (replay window invalidates after consumption)
- Expiry is configurable (default: 5 minutes)

## Stream Multiplex

QUIC-inspired multiple logical streams over a single transport connection:

```
┌─────────────────────────────────────┐
│         Transport Connection        │
│                                     │
│  Stream[0]: PRIMARY — main LLM call │
│  Stream[1]: SPECULATIVE — prediction│
│  Stream[2]: REASONING — chain-of-t  │
│  Stream[3]: TOOL — tool call result │
│  Stream[4]: DRAFT — draft response  │
│                                     │
│  Max streams: 32                    │
└─────────────────────────────────────┘
```

Each stream has independent:
- **Type**: PRIMARY, SPECULATIVE, REASONING, TOOL, DRAFT
- **Priority**: integer, higher = more bandwidth
- **Reliability**: RELIABLE (guaranteed), PARTIAL, BEST_EFFORT
- **State**: IDLE → ACTIVE → CLOSED or MIGRATED
- **ReplayWindow**: per-stream circular buffer for resume

## Observability

```bash
# Per-provider stream health
curl http://localhost:8787/stats | jq '.ignored_chunks'
```

```json
{
  "total": 3,
  "by_provider": {
    "openai": 2,
    "anthropic": 1
  }
}
```

Ignored chunks indicate mis-threaded stream IDs — chunks arriving for streams that weren't properly initialized. These are early warnings of streaming lifecycle bugs.
