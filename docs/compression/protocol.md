# Protocol

LATTICE implements a layered protocol stack for efficient LLM transport: session manifests, delta encoding, binary framing, and stream resume.

## Session Manifests

A Merkle-like DAG of content segments that enables deterministic cache alignment. The manifest is shared between client and server to identify which content segments are already cached.

```
Session → Manifest → Segments → Content
                        │
              breakpoints for cache alignment
              anchor hashes for versioning
              token estimates for budgeting
```

### Segment Types

| Segment | Purpose |
|---------|---------|
| SYSTEM | System prompts and instructions |
| TOOLS | Tool definitions and schemas |
| MESSAGES | Conversation messages |
| CACHE | Metadata for cache alignment |

## Delta Encoding

Reduces wire traffic by sending only new messages since the last turn. The server reconstructs the full context from the stored session.

```
Client: POST /v1/chat/completions
  Headers: x-lattice-session-id: sess_abc123
  Body: {
    "model": "openai/gpt-4o",
    "messages": [{"role": "user", "content": "Follow-up question"}]
  }

Server:
  1. Lookup session sess_abc123 → existing messages [m0, m1, m2]
  2. Detect new messages → [m3: "Follow-up question"]
  3. Reconstruct full context → [m0, m1, m2, m3]
  4. Apply transforms → compressed request
  5. Route to provider
  6. Cache response
  7. Update session with m3 + assistant response

Response:
  Headers: x-lattice-delta-savings-bytes: 3400
```

## Binary Framing

A length-prefixed binary wire protocol for O(1) frame parsing. Uses delimiter-based framing with CRC integrity checks.

```
Frame format:
┌──────────────┬─────────────┬──────────────┬──────────────┐
│  Length (4B) │  Type (1B)  │  Payload     │  CRC (4B)    │
└──────────────┴─────────────┴──────────────┴──────────────┘

Frame types:
  DATA     — Content segment
  CONTROL  — Flow control
  PING     — Keepalive
  PONG     — Keepalive response
  RESUME   — Stream resume request
  CLOSE    — Connection close
```

### Frame Flags

| Flag | Purpose |
|------|---------|
| COMPRESSED | Payload is compressed |
| FINAL | Last frame in stream |
| RESUME_OK | Resume token accepted |

## Stream Resume

HMAC-signed resume tokens allow interrupted streaming connections to resume without replaying the entire conversation.

```
1. Client connects, streams N chunks
2. Connection drops
3. Client sends RESUME frame with token
4. Server verifies HMAC(token, session_secret)
5. Server replays from chunk N+1
```

Resume tokens are time-limited and single-use.

## Dictionary Codec

A learned phrase dictionary for efficient content encoding. Repeated phrases are replaced with short integer references.

```
Encoder: "The session manifest should remain stable" → "<d_0> <d_1>"
Decoder: "<d_0> <d_1>" → "The session manifest should remain stable"
```

Dictionary entries are:
- Learned per-session from repeated content
- Lossless (roundtrip exact)
- Prioritized by frequency × length

## Multiplex

Multiple logical streams over a single transport connection:

```
Connection → Stream[0] → Conversation A
           → Stream[1] → Conversation B
           → Stream[2] → Health pings
```

Each stream has independent: framing, stall detection, resume tokens, and cleanup.
