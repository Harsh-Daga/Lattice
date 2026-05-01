# Batching & Speculative Execution

LATTICE's execution layer optimizes latency through two complementary techniques: request batching (reducing per-request overhead) and speculative execution (pre-running predicted work).

## Request Batching

Groups independent LLM requests arriving within a short window into a single multi-message provider call, sharing system prompts and tool definitions.

### How It Works

```
Time ──────────────────────────────────────────────▶

Request A ──┐
Request B ──┤  wait window (max_wait_ms)
Request C ──┘       │
                    ▼
         ┌─────────────────────┐
         │  Single Batch Call   │
         │  model: gpt-4o       │
         │  messages: [shared]  │
         │   + [req A expand]   │
         │   + [req B expand]   │
         │   + [req C expand]   │
         └─────────┬───────────┘
                   ▼
         ┌─────────────────────┐
         │  Provider Response   │
         └─────────┬───────────┘
                   ▼
         Fan out by correlation_id
         req A ← response[0]
         req B ← response[1]
         req C ← response[2]
```

### Batch Compatibility

Only requests with identical characteristics can be batched:

```python
@dataclass(frozen=True)
class BatchKey:
    model: str           # Must match exactly
    temperature: float   # Must match
    max_tokens: int      # Must match
    top_p: float         # Must match
    stream: bool         # Streaming requests are NOT batched
    tools_hash: str      # MD5 of sorted tool function names
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_wait_ms` | 50 | Maximum time to wait for batch to fill |
| `max_batch_size` | 8 | Maximum requests per batch |
| `batch_overhead_savings` | 30-60% | Fractional token reduction from shared prompts |

### Savings Model

A batch of N requests shares:
- System prompt (1× instead of N×)
- Tool definitions (1× instead of N×)
- Common prefix material

This reduces request-side token overhead by 30-60% depending on prompt similarity.

---

## Speculative Execution

Predicts likely next steps and pre-runs them in parallel with the real request. If the prediction is correct → near-instant response. If wrong → discard, no penalty.

### How It Works

```
Main request ────────────▶ provider.completion() ───▶ main_response
                              │
Speculative task ──▶ run_speculative(prediction) ──▶ speculative_response
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
                Hit? Wait a bit      Miss? Discard
                    │
                    ▼
              Use speculative_response
              (saved ~1 round trip)
```

### Prediction Strategy

Rule-based (not LLM-based) for deterministic, zero-cost predictions:

| Pattern | Prediction |
|---------|-----------|
| Last message requests a known tool | Predict tool call with likely args |
| Conversation follows known sequence | Predict next turn completion |
| Cache hit pattern detected | Predict cache will satisfy |

### Safety Properties

- **Never blocks**: Speculation runs in a sidecar asyncio task. The real request proceeds independently.
- **Discard is always safe**: Wrong predictions are thrown away. Latency is no worse than baseline.
- **Confidence threshold**: Only launch speculation when `confidence > 0.7`.

### Metrics

```bash
curl http://localhost:8787/stats | jq '.speculation'
```

```json
{
  "hits": 47,
  "misses": 12,
  "hit_rate": 0.797,
  "total_tokens_saved": 340000,
  "avg_savings_ms": 120
}
```

---

## Auto-Continuation

When a provider truncates a response due to `max_tokens`, LATTICE automatically sends follow-up requests, appending the partial assistant message to the conversation.

```
Turn 1: "Explain quantum computing..."
    │
    ▼
Response: "Quantum computing leverages..." [finish_reason="length"]
    │
    ▼ auto-continuation triggered
    │
Turn 2: [conversation + partial from Turn 1] → "superposition and entanglement principles..."
    │
    ▼
Response: "superposition and entanglement principles..." [finish_reason="stop"]
    │
    ▼ content stitched: Turn 1 content + Turn 2 content
```

Configuration: `max_turns` (default: 3) limits continuation depth. Usage (tokens) is aggregated across all turns. On failure, returns whatever was accumulated so far.
