# Delta Encoding & Session Reuse

LATTICE's delta wire protocol reduces wire traffic by sending only new messages per turn, while the proxy reconstructs the full conversation from its session store. After turn 1, this achieves **~95% wire byte savings**.

## How Delta Encoding Works

```
Turn 1:
  Client → Full prompt (all messages)
  Proxy → Stores session, assigns session_id

Turn 2:
  Client → Delta: { session_id, base_seq=3, new_messages=[m4] }
  Proxy → Lookup session, reconstruct [m0,m1,m2,m3] + [m4] → [m0,...,m4]
  Proxy → Full context sent to provider

Turn 3:
  Client → Delta: { session_id, base_seq=4, new_messages=[m5] }
  Proxy → Reconstruct [m0,...,m4] + [m5] → [m0,...,m5]
```

## Wire Format

The delta payload is a JSON object:

```json
{
  "messages": [
    {"role": "user", "content": "Follow-up question"}
  ],
  "_lattice_metadata": {
    "_delta_wire": true,
    "_delta_session_id": "sess_a1b2c3d4e5f6",
    "_delta_base_seq": 5,
    "_delta_messages": [
      {"role": "user", "content": "Follow-up question"}
    ],
    "_delta_anchor_version": 7
  }
}
```

## Optimistic Concurrency (CAS-style)

Delta encoding uses anchor versions to prevent lost updates:

```
Client sends:  anchor_version=7
Server has:    session.version=7
Result:        ✅ Match — delta applies cleanly

Client sends:  anchor_version=7
Server has:    session.version=9 (another client updated)
Result:        ❌ Mismatch — delta rejected, falls back to full messages
```

The fallback is transparent: the server treats the delta messages as the full message list, and the session state is recovered.

## Fallback Scenarios

| Reason | Counter | What Happens |
|--------|---------|-------------|
| No session_id | — | Treat as full request |
| Session not found | `session_not_found` | Fallback to full |
| Version mismatch | `version_mismatch` | Fallback to full + record telemetry |
| Sequence mismatch | `sequence_mismatch` | Fallback to full |
| All ok | `delta_success` | Delta applies + telemetry recorded |

## Savings Computation

```python
from lattice.core.delta_wire import delta_wire_bytes

full_bytes, delta_bytes = delta_wire_bytes(
    full_messages=[m0, m1, m2, m3, m4],
    new_messages=[m4],
    session_id="sess_abc",
    base_sequence=3
)
savings_pct = (full_bytes - delta_bytes) / full_bytes * 100
# Typically 85-98% after turn 2
```

## Session Store

Sessions persist in memory or Redis with configurable TTL:

```python
# In config
LATTICE_SESSION_STORE=memory    # Single process
LATTICE_SESSION_STORE=redis     # Multi-process
LATTICE_SESSION_TTL_SECONDS=3600
LATTICE_REDIS_URL=redis://localhost:6379/0
```

Each session maintains:
- `session_id` — unique per conversation
- `version` — monotonic counter for CAS
- `messages` — full message history
- `manifest` — content segment DAG for cache alignment
- `provider`, `model` — for adapter routing
- `tools` — for tool-aware transforms
- `metadata` — cache hits, cost tracking

## Integration with Proxy

Session IDs flow through HTTP headers:

```
Request:  x-lattice-session-id: sess_abc123
Response: x-lattice-session-id: sess_abc123
          x-lattice-delta-savings-bytes: 3400
```

If no session ID is provided, the proxy auto-generates one and returns it in the response.
