"""Tests for binary framing protocol."""

import pytest

from lattice.protocol.framing import (
    BinaryFramer,
    Frame,
    FrameFlags,
    FrameType,
    MessageAssembler,
    compute_frame_digest,
)


class TestFrame:
    def test_encode_decode_roundtrip(self):
        frame = Frame(
            frame_type=FrameType.REQUEST,
            flags=FrameFlags.NONE,
            payload=b'{"model":"gpt-4"}',
        )
        raw = frame.to_bytes()
        decoded = Frame.from_bytes(raw)
        assert decoded.frame_type == FrameType.REQUEST
        assert decoded.flags == FrameFlags.NONE
        assert decoded.payload == b'{"model":"gpt-4"}'

    def test_frame_with_stream_id(self):
        frame = Frame(
            frame_type=FrameType.STREAM_CHUNK,
            flags=FrameFlags.NONE,
            payload=b"hello",
            stream_id=42,
        )
        assert frame.stream_id == 42

    def test_too_short_data(self):
        with pytest.raises(ValueError, match="too short"):
            Frame.from_bytes(b"short")

    def test_magic_mismatch(self):
        with pytest.raises(ValueError, match="Magic mismatch"):
            Frame.from_bytes(b"XXXX" + b"\x00" * 11)

    def test_incomplete_frame(self):
        header = b"LATT" + b"\x10\x00\x00\x00\x00\x10\x00\x00\x00\x00"  # claims 16 bytes payload
        with pytest.raises(ValueError, match="Incomplete frame"):
            Frame.from_bytes(header + b"only_5")

    def test_continuation_flag(self):
        frame = Frame(
            frame_type=FrameType.REQUEST,
            flags=FrameFlags.CONTINUATION,
            payload=b"chunk1",
        )
        assert frame.is_continuation()

    def test_no_continuation(self):
        frame = Frame(
            frame_type=FrameType.REQUEST,
            flags=FrameFlags.NONE,
            payload=b"whole",
        )
        assert not frame.is_continuation()


class TestMessageAssembler:
    def test_single_frame_complete(self):
        assembler = MessageAssembler()
        frame = Frame(FrameType.REQUEST, FrameFlags.NONE, b"data")
        assert assembler.append(frame) is True
        assert assembler.is_complete
        assert assembler.payload == b"data"
        assert assembler.frame_type == FrameType.REQUEST

    def test_multi_frame_assembly(self):
        assembler = MessageAssembler()
        f1 = Frame(FrameType.REQUEST, FrameFlags.CONTINUATION, b"part1")
        f2 = Frame(FrameType.REQUEST, FrameFlags.NONE, b"part2")
        assert assembler.append(f1) is False
        assert not assembler.is_complete
        assert assembler.append(f2) is True
        assert assembler.is_complete
        assert assembler.payload == b"part1part2"

    def test_reset(self):
        assembler = MessageAssembler()
        assembler.append(Frame(FrameType.REQUEST, FrameFlags.NONE, b"x"))
        assembler.reset()
        assert not assembler.is_complete
        with pytest.raises(ValueError):
            assembler.frame_type

    def test_append_after_complete(self):
        assembler = MessageAssembler()
        assembler.append(Frame(FrameType.REQUEST, FrameFlags.NONE, b"x"))
        with pytest.raises(ValueError, match="already complete"):
            assembler.append(Frame(FrameType.REQUEST, FrameFlags.NONE, b"y"))


class TestBinaryFramer:
    def test_encode_request_small(self):
        framer = BinaryFramer(max_frame_payload=1024)
        frames = framer.encode_request(b'{"hello":"world"}')
        assert len(frames) == 1
        assert frames[0].frame_type == FrameType.REQUEST

    def test_encode_request_chunked(self):
        framer = BinaryFramer(max_frame_payload=10)
        payload = b"a" * 25
        frames = framer.encode_request(payload)
        assert len(frames) == 3
        assert frames[0].is_continuation()
        assert frames[1].is_continuation()
        assert not frames[2].is_continuation()
        assert b"".join(f.payload for f in frames) == payload

    def test_encode_stream_chunk(self):
        framer = BinaryFramer()
        frame = framer.encode_stream_chunk(b'{"delta":"hi"}', stream_id=7)
        assert frame.frame_type == FrameType.STREAM_CHUNK
        assert frame.stream_id == 7

    def test_encode_stream_done(self):
        framer = BinaryFramer()
        frame = framer.encode_stream_done(stream_id=7)
        assert frame.frame_type == FrameType.STREAM_DONE
        assert frame.payload == b""

    def test_encode_ping_pong(self):
        framer = BinaryFramer()
        ping = framer.encode_ping()
        pong = framer.encode_pong()
        assert ping.frame_type == FrameType.PING
        assert pong.frame_type == FrameType.PONG

    def test_encode_decode_error(self):
        framer = BinaryFramer()
        frame = framer.encode_error(404, "not found")
        code, msg = framer.decode_error(frame)
        assert code == 404
        assert msg == "not found"

    def test_encode_decode_session_start(self):
        framer = BinaryFramer()
        frame = framer.encode_session_start("sess_abc", "hash123")
        sid, h = framer.decode_session_start(frame)
        assert sid == "sess_abc"
        assert h == "hash123"

    def test_encode_decode_session_delta(self):
        framer = BinaryFramer()
        frame = framer.encode_session_delta("sess_abc", 5, b'{"new":"msg"}')
        sid, ver, delta = framer.decode_session_delta(frame)
        assert sid == "sess_abc"
        assert ver == 5
        assert delta == b'{"new":"msg"}'

    def test_decode_frame_static(self):
        framer = BinaryFramer()
        frames = framer.encode_request(b"test")
        decoded = BinaryFramer.decode_frame(frames[0].to_bytes())
        assert decoded.frame_type == FrameType.REQUEST
        assert decoded.payload == b"test"

    def test_encode_decode_dictionary_negotiate(self):
        framer = BinaryFramer()
        snapshot = {"max_entries": 1024, "next_index": 205}
        frame = framer.encode_dictionary_negotiate(static_version=2, dynamic_snapshot=snapshot)
        assert frame.frame_type == FrameType.DICTIONARY_NEGOTIATE
        static_ver, dyn = framer.decode_dictionary_negotiate(frame)
        assert static_ver == 2
        assert dyn == snapshot

    def test_dictionary_negotiate_no_dynamic(self):
        framer = BinaryFramer()
        frame = framer.encode_dictionary_negotiate(static_version=1)
        static_ver, dyn = framer.decode_dictionary_negotiate(frame)
        assert static_ver == 1
        assert dyn is None


class TestFrameDigest:
    def test_digest_stable(self):
        f1 = Frame(FrameType.REQUEST, FrameFlags.NONE, b"a")
        f2 = Frame(FrameType.REQUEST, FrameFlags.NONE, b"a")
        assert compute_frame_digest([f1]) == compute_frame_digest([f2])

    def test_digest_different(self):
        f1 = Frame(FrameType.REQUEST, FrameFlags.NONE, b"a")
        f2 = Frame(FrameType.REQUEST, FrameFlags.NONE, b"b")
        assert compute_frame_digest([f1]) != compute_frame_digest([f2])

    def test_digest_multiple_frames(self):
        frames = [
            Frame(FrameType.REQUEST, FrameFlags.NONE, b"a"),
            Frame(FrameType.RESPONSE, FrameFlags.NONE, b"b"),
        ]
        digest = compute_frame_digest(frames)
        assert len(digest) == 16


class TestFramingEdgeCases:
    def test_empty_payload(self):
        frame = Frame(FrameType.PING, FrameFlags.NONE, b"")
        raw = frame.to_bytes()
        assert len(raw) == 15  # header only
        decoded = Frame.from_bytes(raw)
        assert decoded.payload == b""

    def test_checksum_verification(self):
        frame = Frame(FrameType.REQUEST, FrameFlags.NONE, b"hello")
        raw = frame.to_bytes()
        decoded = Frame.from_bytes(raw)
        assert decoded.verify_checksum() is True

    def test_checksum_mismatch_detected(self):
        frame = Frame(FrameType.REQUEST, FrameFlags.NONE, b"hello", checksum=0xDEADBEEF)
        assert frame.verify_checksum() is False

    def test_checksum_zero_always_passes(self):
        frame = Frame(FrameType.REQUEST, FrameFlags.NONE, b"hello", checksum=0)
        assert frame.verify_checksum() is True

    def test_large_payload(self):
        payload = b"x" * 1_000_000
        frame = Frame(FrameType.REQUEST, FrameFlags.NONE, payload)
        raw = frame.to_bytes()
        decoded = Frame.from_bytes(raw)
        assert decoded.payload == payload

    def test_all_frame_types(self):
        for ft in FrameType:
            frame = Frame(frame_type=ft, flags=FrameFlags.NONE, payload=b"test")
            raw = frame.to_bytes()
            decoded = Frame.from_bytes(raw)
            assert decoded.frame_type == ft

    def test_all_flag_combinations(self):
        for flag_val in range(16):
            flags = FrameFlags(flag_val)
            frame = Frame(FrameType.REQUEST, flags, b"test")
            raw = frame.to_bytes()
            decoded = Frame.from_bytes(raw)
            assert decoded.flags == flags
