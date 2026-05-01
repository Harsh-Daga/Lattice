# Binary Framing Protocol

LATTICE implements a length-prefixed binary wire protocol for efficient LLM transport — inspired by HTTP/2 framing (RFC 7540) and QUIC stream frames (RFC 9000) but optimized for LLM-specific workloads.

## Why Binary Framing

JSON over HTTP adds significant overhead:
- Every chunk is serialized/deserialized as JSON
- No native multiplexing
- No built-in CRC integrity
- No semantic frame types

Binary framing eliminates all of this: fixed-width headers for O(1) parsing, typed frames for semantic routing, and integrated CRC32 for tamper detection.

## Frame Format

Each frame is a 15-byte header followed by a variable-length payload:

```
┌─────────────────────────────────────────────
│  Magic (4B)  │  Type (1B)  │  Flags (2B)   │        
│  "LATT"      │             │               │      
├──────────────┴─────────────┴────────────── ┤ 
│  Length (4B) — payload size (uint32 LE)    │       
├────────────────────────────────────────────┤         
│  Checksum (4B) — CRC32 over payload        │         
├────────────────────────────────────────────┤         
│  Payload (0-65535 bytes)                   │         
└────────────────────────────────────────────┘ 
```

```python
# Header struct: <4s B H I I
#   magic:   4 bytes ("LATT")
#   type:    1 byte  (frame type enum)
#   flags:   2 bytes (bitmask)
#   length:  4 bytes (uint32 LE)
#   crc32:   4 bytes (uint32 LE)
```

## Frame Types (17)

| Type | Value | Purpose |
|------|-------|---------|
| `PING` | 0x00 | Keepalive heartbeat |
| `PONG` | 0x01 | Heartbeat response |
| `REQUEST` | 0x10 | Full LLM request |
| `RESPONSE` | 0x11 | Full LLM response |
| `STREAM_CHUNK` | 0x20 | Single streaming chunk |
| `STREAM_DONE` | 0x21 | End of stream |
| `SESSION_START` | 0x30 | New session |
| `SESSION_DELTA` | 0x31 | Delta update to session |
| `SESSION_CLOSE` | 0x32 | Close session |
| `RESUME_TOKEN` | 0x40 | Stream resume token |
| `RESUME_REPLAY` | 0x41 | Replayed chunks |
| `RESUME_REQUEST` | 0x42 | Request to resume |
| `CONNECTION_MIGRATE` | 0x50 | Move stream to new connection |
| `DICTIONARY_NEGOTIATE` | 0x60 | Dictionary exchange |
| `DICTIONARY_UPDATE` | 0x61 | Dictionary delta |
| `ERROR` | 0xF0 | Protocol error |
| `RESET` | 0xFF | Connection reset |

## Flag Bits (16-bit bitmask)

| Flag | Bit | Purpose |
|------|-----|---------|
| `COMPRESSED` | 0x0001 | Payload is zstd-compressed |
| `ENCRYPTED` | 0x0002 | Payload is encrypted (reserved) |
| `CONTINUATION` | 0x0004 | Frame continues in next |
| `ACK_REQUIRED` | 0x0008 | Receiver must acknowledge |
| `DICTIONARY_COMPRESSED` | 0x0010 | Payload uses dictionary compression |

### Semantic Boundary Flags

| Flag | Bit | Purpose |
|------|-----|---------|
| `BOUNDARY_SENTENCE` | 0x0020 | Frame ends at sentence boundary |
| `BOUNDARY_TOOL_START` | 0x0040 | Tool call start |
| `BOUNDARY_TOOL_END` | 0x0080 | Tool call end |
| `BOUNDARY_REASONING` | 0x0100 | Reasoning/thinking boundary |

### Reliability Levels

| Flag | Bit | Guarantee |
|------|-----|-----------|
| `CRITICALITY_LOW` | 0x0000 | Best-effort delivery |
| `CRITICALITY_MEDIUM` | 0x0200 | At-least-once delivery |
| `CRITICALITY_HIGH` | 0x0400 | Exactly-once delivery |

## Dictionary Compression

The binary protocol supports a shared dictionary codec for repeated phrases:

```
Frame:  type=DICTIONARY_NEGOTIATE, dictionary_compressed=true
Payload: "phrase_id=42" → replaces with dictionary entry
```

See [Transforms](transforms.md#dictionary-compress-priority-25) for the dictionary compression algorithm.

## CRC32 Integrity

Every frame includes a CRC32 checksum over the payload. Tampered frames are discarded. Unlike HMAC, CRC32 is fast (hardware-accelerated on modern CPUs) and sufficient for transport integrity (not security).

## Comparison: JSON vs Binary

| Aspect | JSON over HTTP | Binary Framing |
|--------|---------------|----------------|
| Per-chunk overhead | 50-200 bytes (JSON envelope) | 15 bytes (fixed header) |
| Parsing | Full JSON deserialization | Fixed-width struct unpack |
| Multiplexing | Needs separate HTTP connections or WebSocket | Native multi-stream |
| Integrity | Relies on TCP/TLS | Built-in CRC32 |
| Frame semantics | None (all data streams) | 17 typed frame types |
| Content boundaries | Lost in stream merge | Semantic boundary flags |
