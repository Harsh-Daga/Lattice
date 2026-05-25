"""Tests for Phase A framing extensions."""

from __future__ import annotations

from lattice.protocol.framing import BinaryFramer, Frame, FrameFlags, FrameType


def test_new_flags_encode_decode() -> None:
    frame = Frame(
        frame_type=FrameType.STREAM_CHUNK,
        flags=FrameFlags.CRITICALITY_HIGH | FrameFlags.BOUNDARY_SENTENCE,
        payload=b"hello",
    )
    decoded = Frame.from_bytes(frame.to_bytes())
    assert decoded.flags & FrameFlags.CRITICALITY_HIGH
    assert decoded.flags & FrameFlags.BOUNDARY_SENTENCE


def test_priority_mask_extract() -> None:
    frame = Frame(
        frame_type=FrameType.STREAM_CHUNK,
        flags=FrameFlags(7 << 12),
        payload=b"x",
    )
    assert frame.priority == 7
    assert int(frame.flags) & int(FrameFlags.PRIORITY_MASK) == 7 << 12


def test_connection_migrate_frame() -> None:
    framer = BinaryFramer()
    frame = framer.encode_connection_migrate("sess-1", {"ip": "10.0.0.2", "network": "cellular"})
    assert frame.frame_type == FrameType.CONNECTION_MIGRATE
    session_id, client_info = framer.decode_connection_migrate(frame)
    assert session_id == "sess-1"
    assert client_info["network"] == "cellular"


def test_boundary_sentence_frame() -> None:
    framer = BinaryFramer()
    frame = framer.encode_frame_with_boundary(
        b"sentence",
        boundary_type="sentence",
        reliability="medium",
        priority=3,
    )
    assert frame.flags & FrameFlags.BOUNDARY_SENTENCE
    assert frame.priority == 3


def test_combined_flags() -> None:
    framer = BinaryFramer()
    frame = framer.encode_frame_with_boundary(
        b"tool-start",
        boundary_type="tool_start",
        reliability="high",
        priority=1,
    )
    assert frame.flags & FrameFlags.CRITICALITY_HIGH
    assert frame.flags & FrameFlags.BOUNDARY_TOOL_START
