"""Binary framing protocol for LATTICE native SDK / daemon path.

Provides a compact, low-overhead wire format for client↔Lattice hop.
This is OPTIONAL — JSON/SSE compatibility is always maintained as fallback.

Frame Format (little-endian):
    ┌─────────┬─────────┬──────────┬────────┬──────────┐
    │  magic  │  type   │  flags   │ length │ checksum │
    │ 4 bytes │ 1 byte  │ 2 bytes  │4 bytes │ 4 bytes  │
    ├─────────┴─────────┴──────────┴────────┴──────────┤
    │              payload (length bytes)              │
    └──────────────────────────────────────────────────┘

Magic: 0x4C 0x41 0x54 0x54 ("LATT")
Type:  enum FrameType
Flags: compression, encryption, continuation, reliability, boundaries, priority
Length: uint32 payload length (max 2^32-1 ≈ 4 GB)
Checksum: CRC32 of payload (0x00000000 if not computed)

Design Decisions
----------------
1. Fixed-width 15-byte header enables zero-copy parsing.
2. Length-prefixed payloads allow fast skipping of unknown frames.
3. Payload checksum enables tamper detection without full digest.
4. Chunked frames support streaming via continuation bit.

Reference:
- HTTP/2 framing (RFC 7540) — fixed header, length-prefixed
- QUIC stream frames (RFC 9000) — lightweight, multiplexed
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import struct
from typing import Any

# =============================================================================
# Constants
# =============================================================================

_MAGIC = b"LATT"
_HEADER_SIZE = 15
_MAX_PAYLOAD_SIZE = 0xFFFFFFFF


def _crc32(data: bytes) -> int:
    """Compute CRC32 checksum of data."""
    import binascii
    return binascii.crc32(data) & 0xFFFFFFFF

# =============================================================================
# Frame types
# =============================================================================

class FrameType(enum.IntEnum):
    """LATTICE binary frame types."""

    PING = 0x01          # Keep-alive / RTT measurement
    PONG = 0x02          # Ping response
    REQUEST = 0x10       # Compressed request payload
    RESPONSE = 0x11      # Compressed response payload
    STREAM_CHUNK = 0x12  # Streaming SSE chunk
    STREAM_DONE = 0x13   # Streaming end marker
    SESSION_START = 0x20 # New session with manifest
    SESSION_DELTA = 0x21 # Delta update to session
    SESSION_CLOSE = 0x22 # Close session
    RESUME_TOKEN = 0x30  # Resumable stream token
    RESUME_REPLAY = 0x31 # Replay missed chunks
    CONNECTION_MIGRATE = 0x40  # Client connection changed; keep session
    RESUME_REQUEST = 0x41      # Request replay from sequence N
    DICTIONARY_NEGOTIATE = 0x50  # Dictionary version negotiation (ADR-002)
    ERROR = 0xFF         # Protocol error


# =============================================================================
# Flags
# =============================================================================

class FrameFlags(enum.IntFlag):
    """Per-frame flag bits."""

    NONE = 0x00
    COMPRESSED = 0x01    # Payload is zstd-compressed
    ENCRYPTED = 0x02     # Payload is encrypted (future)
    CONTINUATION = 0x04  # More frames follow for same logical message
    ACK_REQUIRED = 0x08  # Receiver must send ACK
    # Semantic reliability flags
    CRITICALITY_LOW = 0x10
    CRITICALITY_MEDIUM = 0x20
    CRITICALITY_HIGH = 0x40
    # Semantic boundary flags
    BOUNDARY_SENTENCE = 0x80
    BOUNDARY_TOOL_START = 0x100
    BOUNDARY_TOOL_END = 0x200
    BOUNDARY_REASONING = 0x400
    # Wire-level dictionary compression (ADR-002)
    DICT_COMPRESSED = 0x800
    # Priority (0-15) packed in top bits
    PRIORITY_MASK = 0xF000


# =============================================================================
# Frame
# =============================================================================

@dataclasses.dataclass(frozen=True, slots=True)
class Frame:
    """A single LATTICE binary frame.

    Attributes:
        frame_type: Type of frame.
        flags: Frame flag bits.
        payload: Raw payload bytes.
        stream_id: Optional stream identifier for multiplexing.
        metadata: Optional frame metadata for higher-level protocol use.
        sequence: Optional stream sequence number.
        checksum: CRC32 of payload (0 = not computed).
    """

    frame_type: FrameType
    flags: FrameFlags
    payload: bytes
    stream_id: int = 0
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    sequence: int = 0
    checksum: int = 0

    def to_bytes(self) -> bytes:
        """Serialize frame to binary."""
        ck = self.checksum if self.checksum else _crc32(self.payload)
        header = struct.pack(
            "<4sBHI I", _MAGIC, int(self.frame_type), int(self.flags), len(self.payload), ck
        )
        return header + self.payload

    @classmethod
    def from_bytes(cls, data: bytes) -> Frame:
        """Deserialize frame from binary.

        Raises:
            ValueError: If data is too short or magic mismatch.
        """
        if len(data) < _HEADER_SIZE:
            raise ValueError(f"Frame too short: {len(data)} < {_HEADER_SIZE}")
        magic, ftype, flags, length, checksum = struct.unpack("<4sBHI I", data[:_HEADER_SIZE])
        if magic != _MAGIC:
            raise ValueError(f"Magic mismatch: expected {_MAGIC!r}, got {magic!r}")
        if len(data) < _HEADER_SIZE + length:
            raise ValueError(
                f"Incomplete frame: need {_HEADER_SIZE + length}, have {len(data)}"
            )
        return cls(
            frame_type=FrameType(ftype),
            flags=FrameFlags(flags),
            payload=data[_HEADER_SIZE:_HEADER_SIZE + length],
            checksum=checksum,
        )

    def verify_checksum(self) -> bool:
        """Verify payload integrity against stored checksum.

        Returns True if checksum is 0 (not computed) or matches payload.
        """
        if self.checksum == 0:
            return True
        return _crc32(self.payload) == self.checksum

    @property
    def header_size(self) -> int:
        return _HEADER_SIZE

    @property
    def total_size(self) -> int:
        return _HEADER_SIZE + len(self.payload)

    def is_continuation(self) -> bool:
        return bool(self.flags & FrameFlags.CONTINUATION)

    @property
    def priority(self) -> int:
        """Extract priority value (0-15) from flags."""
        return (int(self.flags) & int(FrameFlags.PRIORITY_MASK)) >> 12


# =============================================================================
# Message assembly (handles chunked frames)
# =============================================================================

class MessageAssembler:
    """Assembles multi-frame messages from continuation frames.

    Usage:
        assembler = MessageAssembler()
        assembler.append(frame)
        if assembler.is_complete:
            full_payload = assembler.payload
    """

    def __init__(self) -> None:
        self._frames: list[Frame] = []
        self._complete = False

    def append(self, frame: Frame) -> bool:
        """Append a frame. Returns True if message is complete."""
        if self._complete:
            raise ValueError("Assembler already complete — reset required")
        self._frames.append(frame)
        if not frame.is_continuation():
            self._complete = True
        return self._complete

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def payload(self) -> bytes:
        """Concatenated payload from all frames."""
        return b"".join(f.payload for f in self._frames)

    @property
    def frame_type(self) -> FrameType:
        """Type of the first frame."""
        if not self._frames:
            raise ValueError("No frames")
        return self._frames[0].frame_type

    def reset(self) -> None:
        """Clear assembler for next message."""
        self._frames.clear()
        self._complete = False


# =============================================================================
# Framer (encoder / decoder)
# =============================================================================

class BinaryFramer:
    """Encode/decode LATTICE binary frames.

    Usage:
        framer = BinaryFramer()
        frames = framer.encode_request(payload_bytes)
        for frame in frames:
            wire_bytes = frame.to_bytes()

        # Decoding:
        frame = framer.decode_frame(wire_bytes)
    """

    def __init__(self, max_frame_payload: int = 65536) -> None:
        """Initialize framer.

        Args:
            max_frame_payload: Max bytes per frame payload. Larger payloads
                               are automatically chunked.
        """
        self.max_frame_payload = max_frame_payload

    # ------------------------------------------------------------------
    # Encode helpers
    # ------------------------------------------------------------------

    def encode_request(
        self,
        payload: bytes,
        stream_id: int = 0,
        *,
        flags: FrameFlags = FrameFlags.NONE,
    ) -> list[Frame]:
        """Encode a request payload into one or more frames."""
        return self._chunk(FrameType.REQUEST, flags, payload, stream_id)

    def encode_response(
        self,
        payload: bytes,
        stream_id: int = 0,
        *,
        flags: FrameFlags = FrameFlags.NONE,
    ) -> list[Frame]:
        """Encode a response payload into one or more frames."""
        return self._chunk(FrameType.RESPONSE, flags, payload, stream_id)

    def encode_stream_chunk(self, payload: bytes, stream_id: int = 0) -> Frame:
        """Encode a streaming chunk."""
        return Frame(
            frame_type=FrameType.STREAM_CHUNK,
            flags=FrameFlags.NONE,
            payload=payload,
            stream_id=stream_id,
        )

    def encode_stream_done(self, stream_id: int = 0) -> Frame:
        """Encode stream end marker."""
        return Frame(
            frame_type=FrameType.STREAM_DONE,
            flags=FrameFlags.NONE,
            payload=b"",
            stream_id=stream_id,
        )

    def encode_ping(self) -> Frame:
        """Encode a ping frame."""
        return Frame(frame_type=FrameType.PING, flags=FrameFlags.NONE, payload=b"")

    def encode_pong(self) -> Frame:
        """Encode a pong frame."""
        return Frame(frame_type=FrameType.PONG, flags=FrameFlags.NONE, payload=b"")

    def encode_error(self, code: int, message: str) -> Frame:
        """Encode an error frame."""
        payload = struct.pack("<I", code) + message.encode("utf-8")
        return Frame(frame_type=FrameType.ERROR, flags=FrameFlags.NONE, payload=payload)

    def encode_session_start(self, session_id: str, manifest_hash: str) -> Frame:
        """Encode session start frame."""
        payload = (
            struct.pack("<I", len(session_id))
            + session_id.encode("utf-8")
            + struct.pack("<I", len(manifest_hash))
            + manifest_hash.encode("utf-8")
        )
        return Frame(
            frame_type=FrameType.SESSION_START,
            flags=FrameFlags.NONE,
            payload=payload,
        )

    def encode_session_delta(
        self, session_id: str, anchor_version: int, delta_payload: bytes
    ) -> Frame:
        """Encode session delta frame."""
        payload = (
            struct.pack("<I", len(session_id))
            + session_id.encode("utf-8")
            + struct.pack("<I", anchor_version)
            + delta_payload
        )
        return Frame(
            frame_type=FrameType.SESSION_DELTA,
            flags=FrameFlags.NONE,
            payload=payload,
        )

    def encode_connection_migrate(self, session_id: str, new_client_info: dict[str, Any]) -> Frame:
        """Encode a connection migration frame."""
        payload = json.dumps(
            {"session_id": session_id, "client_info": new_client_info},
            separators=(",", ":"),
        ).encode("utf-8")
        return Frame(
            frame_type=FrameType.CONNECTION_MIGRATE,
            flags=FrameFlags.NONE,
            payload=payload,
            metadata={"session_id": session_id, "client_info": new_client_info},
        )

    def encode_dictionary_negotiate(
        self, static_version: int, dynamic_snapshot: dict[str, Any] | None = None
    ) -> Frame:
        """Encode dictionary version negotiation frame.

        Args:
            static_version: Version of the static dictionary table.
            dynamic_snapshot: Optional serialized dynamic table state.
        """
        payload = json.dumps(
            {"static_version": static_version, "dynamic": dynamic_snapshot},
            separators=(",", ":"),
        ).encode("utf-8")
        return Frame(
            frame_type=FrameType.DICTIONARY_NEGOTIATE,
            flags=FrameFlags.NONE,
            payload=payload,
        )

    def encode_negotiation_outcome(self, accepted: bool, fallback_reason: str = "") -> Frame:
        payload = json.dumps({"accepted": accepted, "reason": fallback_reason}).encode()
        return Frame(frame_type=FrameType.DICTIONARY_NEGOTIATE, flags=FrameFlags.NONE, payload=payload)

    def decode_negotiation_outcome(self, frame: Frame) -> tuple[bool, str]:
        payload = json.loads(frame.payload.decode("utf-8"))
        return bool(payload.get("accepted", False)), str(payload.get("reason", ""))

    def encode_frame_with_boundary(
        self, data: bytes, boundary_type: str, reliability: str, priority: int = 0
    ) -> Frame:
        """Encode stream frame with semantic boundary, reliability, and priority."""
        if priority < 0 or priority > 15:
            raise ValueError("priority must be between 0 and 15")

        flags = FrameFlags.NONE

        if boundary_type == "sentence":
            flags |= FrameFlags.BOUNDARY_SENTENCE
        elif boundary_type == "tool_start":
            flags |= FrameFlags.BOUNDARY_TOOL_START
        elif boundary_type == "tool_end":
            flags |= FrameFlags.BOUNDARY_TOOL_END
        elif boundary_type == "reasoning":
            flags |= FrameFlags.BOUNDARY_REASONING

        if reliability == "low":
            flags |= FrameFlags.CRITICALITY_LOW
        elif reliability == "medium":
            flags |= FrameFlags.CRITICALITY_MEDIUM
        elif reliability == "high":
            flags |= FrameFlags.CRITICALITY_HIGH

        flags |= FrameFlags(priority << 12)
        return Frame(frame_type=FrameType.STREAM_CHUNK, flags=flags, payload=data)

    # ------------------------------------------------------------------
    # Decode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def decode_frame(data: bytes) -> Frame:
        """Decode a single frame from bytes."""
        return Frame.from_bytes(data)

    @staticmethod
    def decode_error(frame: Frame) -> tuple[int, str]:
        """Decode an error frame payload."""
        if len(frame.payload) < 4:
            return 0, "unknown error"
        code = struct.unpack("<I", frame.payload[:4])[0]
        message = frame.payload[4:].decode("utf-8", errors="replace")
        return code, message

    @staticmethod
    def decode_session_start(frame: Frame) -> tuple[str, str]:
        """Decode session start payload."""
        payload = frame.payload
        sid_len = struct.unpack("<I", payload[:4])[0]
        session_id = payload[4:4 + sid_len].decode("utf-8")
        offset = 4 + sid_len
        hash_len = struct.unpack("<I", payload[offset:offset + 4])[0]
        manifest_hash = payload[offset + 4:offset + 4 + hash_len].decode("utf-8")
        return session_id, manifest_hash

    @staticmethod
    def decode_session_delta(frame: Frame) -> tuple[str, int, bytes]:
        """Decode session delta payload."""
        payload = frame.payload
        sid_len = struct.unpack("<I", payload[:4])[0]
        session_id = payload[4:4 + sid_len].decode("utf-8")
        offset = 4 + sid_len
        anchor_version = struct.unpack("<I", payload[offset:offset + 4])[0]
        delta = payload[offset + 4:]
        return session_id, anchor_version, delta

    @staticmethod
    def decode_connection_migrate(frame: Frame) -> tuple[str, dict[str, Any]]:
        """Decode connection migration payload."""
        payload = json.loads(frame.payload.decode("utf-8"))
        return str(payload.get("session_id", "")), dict(payload.get("client_info", {}))

    @staticmethod
    def decode_dictionary_negotiate(frame: Frame) -> tuple[int, dict[str, Any] | None]:
        """Decode dictionary negotiation payload.

        Returns:
            (static_version, dynamic_snapshot or None)
        """
        payload = json.loads(frame.payload.decode("utf-8"))
        static_version = int(payload.get("static_version", 0))
        dynamic = payload.get("dynamic")
        return static_version, dynamic if dynamic is not None else None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _chunk(
        self,
        frame_type: FrameType,
        flags: FrameFlags,
        payload: bytes,
        stream_id: int,
    ) -> list[Frame]:
        """Split payload into chunked frames if needed."""
        if len(payload) <= self.max_frame_payload:
            return [Frame(frame_type=frame_type, flags=flags, payload=payload, stream_id=stream_id)]

        frames: list[Frame] = []
        offset = 0
        while offset < len(payload):
            chunk = payload[offset:offset + self.max_frame_payload]
            is_last = offset + self.max_frame_payload >= len(payload)
            chunk_flags = flags if is_last else flags | FrameFlags.CONTINUATION
            frames.append(
                Frame(
                    frame_type=frame_type,
                    flags=chunk_flags,
                    payload=chunk,
                    stream_id=stream_id,
                )
            )
            offset += self.max_frame_payload
        return frames


# =============================================================================
# Wire integrity
# =============================================================================

def compute_frame_digest(frames: list[Frame]) -> str:
    """Compute SHA-256 digest of a sequence of frames for tamper detection."""
    hasher = hashlib.sha256()
    for frame in frames:
        hasher.update(frame.to_bytes())
    return hasher.hexdigest()[:16]


__all__ = [
    "Frame",
    "FrameType",
    "FrameFlags",
    "MessageAssembler",
    "BinaryFramer",
    "compute_frame_digest",
    "_MAGIC",
    "_HEADER_SIZE",
]
