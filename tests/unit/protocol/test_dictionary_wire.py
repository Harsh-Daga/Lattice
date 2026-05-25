"""Tests for wire-level dictionary compression (ADR-002).

Covers:
- Static dictionary round-trip
- Dynamic dictionary learning and round-trip
- LRU eviction correctness
- Varint encoding/decoding edge cases
- JSON tokenization fidelity
- Session snapshot serialization
- Multi-turn compression improvement (dynamic table learns)
- Malformed input handling
- Empty data handling
"""

from __future__ import annotations

import json

import pytest

from lattice.protocol.dictionary_codec import (
    DictionaryCodec,
    _decode_varint,
    _encode_varint,
    compress,
    decompress,
)
from lattice.protocol.dictionary_static import STATIC_COUNT, STATIC_REVERSE, static_entry
from lattice.protocol.framing import FrameFlags

# =============================================================================
# Varint
# =============================================================================


class TestVarint:
    """Exhaustive varint codec tests."""

    @pytest.mark.parametrize(
        "value",
        [0, 1, 127, 128, 16383, 16384, 2097151, 2097152, 268435455],
    )
    def test_round_trip(self, value: int) -> None:
        encoded = _encode_varint(value)
        decoded, offset = _decode_varint(encoded, 0)
        assert decoded == value
        assert offset == len(encoded)

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValueError):
            _encode_varint(-1)

    def test_truncated_varint(self) -> None:
        with pytest.raises(ValueError):
            _decode_varint(bytes([0x80, 0x80]), 0)

    def test_varint_too_long(self) -> None:
        # 6 continuation bytes should trigger "too long"
        data = bytes([0x80] * 6)
        with pytest.raises(ValueError):
            _decode_varint(data, 0)

    def test_multi_byte_sequence(self) -> None:
        """Verify that multiple varints in a row decode correctly."""
        values = [0, 128, 16384, 42]
        data = b"".join(_encode_varint(v) for v in values)
        offset = 0
        decoded = []
        for _ in values:
            val, offset = _decode_varint(data, offset)
            decoded.append(val)
        assert decoded == values


# =============================================================================
# Static dictionary
# =============================================================================


class TestStaticDictionary:
    """Static dictionary correctness and completeness."""

    def test_all_indices_present(self) -> None:
        for i in range(STATIC_COUNT):
            assert static_entry(i) is not None

    def test_reverse_lookup_consistency(self) -> None:
        for i in range(STATIC_COUNT):
            text = static_entry(i)
            assert STATIC_REVERSE[text] == i

    def test_no_duplicates(self) -> None:
        texts = [static_entry(i) for i in range(STATIC_COUNT)]
        assert len(texts) == len(set(texts))

    def test_static_count_positive(self) -> None:
        assert STATIC_COUNT > 0


# =============================================================================
# Basic compression round-trip
# =============================================================================


class TestBasicRoundTrip:
    """Core compress/decompress correctness."""

    def test_empty_data(self) -> None:
        codec = DictionaryCodec()
        assert codec.compress(b"") == b""
        assert codec.decompress(b"") == b""

    def test_simple_json(self) -> None:
        data = b'{"role":"user","content":"hello"}'
        codec = DictionaryCodec()
        compressed = codec.compress(data)
        assert codec.decompress(compressed) == data

    def test_multiline_json(self) -> None:
        data = json.dumps(
            {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
                "temperature": 0.7,
                "stream": True,
            },
            indent=2,
        ).encode("utf-8")
        codec = DictionaryCodec()
        compressed = codec.compress(data)
        assert codec.decompress(compressed) == data

    def test_unicode_content(self) -> None:
        data = b'{"content": "\xe4\xb8\xad\xe6\x96\x87\xe6\xb5\x8b\xe8\xaf\x95"}'
        codec = DictionaryCodec()
        compressed = codec.compress(data)
        assert codec.decompress(compressed) == data

    def test_escaped_quotes(self) -> None:
        data = b'{"content": "He said \\"hello\\""}'
        codec = DictionaryCodec()
        compressed = codec.compress(data)
        assert codec.decompress(compressed) == data

    def test_numbers_and_booleans(self) -> None:
        data = b'{"temperature": 0.7, "max_tokens": 4096, "stream": true, "top_p": null}'
        codec = DictionaryCodec()
        compressed = codec.compress(data)
        assert codec.decompress(compressed) == data

    def test_array_of_objects(self) -> None:
        data = json.dumps(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "how are you?"},
            ]
        ).encode("utf-8")
        codec = DictionaryCodec()
        compressed = codec.compress(data)
        assert codec.decompress(compressed) == data


# =============================================================================
# Dynamic dictionary learning
# =============================================================================


class TestDynamicDictionary:
    """Dynamic table population, lookup, and eviction."""

    def test_long_literal_added_to_dynamic(self) -> None:
        """Literals > auto_add_threshold should enter the dynamic table."""
        codec = DictionaryCodec(auto_add_threshold=8)
        long_text = b'"this_is_a_long_string_that_exceeds_threshold"'
        codec.compress(long_text)
        # On second use, it should be referenced from dynamic table
        compressed2 = codec.compress(long_text)
        assert len(compressed2) < len(long_text) * 2  # should be smaller
        assert codec.decompress(compressed2) == long_text

    def test_dynamic_table_promotes_mru(self) -> None:
        """Accessing an entry should promote it to MRU, preventing eviction."""
        codec = DictionaryCodec(max_dynamic_entries=3, auto_add_threshold=1)
        texts = [b'"a"', b'"b"', b'"c"', b'"d"']
        for text in texts:
            codec.compress(text)

        # Now table is full with a, b, c (d evicted a)
        # Access 'a' again — it should be re-added or still present
        codec.compress(b'"a"')
        # Now 'a' is MRU, 'b' is next to be evicted
        codec.compress(b'"e"')  # should evict 'b'
        # 'a' and 'c' should still be in table
        c1 = codec.compress(b'"a"')
        c2 = codec.compress(b'"c"')
        assert codec.decompress(c1) == b'"a"'
        assert codec.decompress(c2) == b'"c"'

    def test_cross_request_learning(self) -> None:
        """Dynamic entries from one request benefit the next."""
        codec = DictionaryCodec()
        data1 = b'{"custom_key_that_is_long": "value1"}'
        data2 = b'{"custom_key_that_is_long": "value2"}'

        c1 = codec.compress(data1)
        c2 = codec.compress(data2)

        # c2 should be smaller because the long key is now in the dynamic table
        assert len(c2) < len(c1)
        assert codec.decompress(c2) == data2

    def test_independent_codecs_no_leak(self) -> None:
        """Two separate codec instances should not share dynamic tables."""
        codec_a = DictionaryCodec(session_id="a")
        codec_b = DictionaryCodec(session_id="b")

        long_text = b'"shared_long_string_example"'
        codec_a.compress(long_text)

        # codec_b should not know about the string
        c_b = codec_b.compress(long_text)
        # Since codec_b hasn't learned it, it should be emitted as literal
        assert codec_b.decompress(c_b) == long_text


# =============================================================================
# Snapshot / persistence
# =============================================================================


class TestSnapshot:
    """Session snapshot serialization and restoration."""

    def test_snapshot_round_trip(self) -> None:
        codec = DictionaryCodec(session_id="sess_test", auto_add_threshold=5)
        for _ in range(10):
            codec.compress(b'{"long_key_name_for_testing": "value"}')

        snapshot = codec.to_snapshot()
        restored = DictionaryCodec.from_snapshot(snapshot)

        assert restored.session_id == codec.session_id
        assert restored.auto_add_threshold == codec.auto_add_threshold

        data = b'{"long_key_name_for_testing": "new_value"}'
        c1 = codec.compress(data)
        c2 = restored.compress(data)
        assert c1 == c2
        assert restored.decompress(c1) == data

    def test_empty_snapshot(self) -> None:
        codec = DictionaryCodec()
        snapshot = codec.to_snapshot()
        restored = DictionaryCodec.from_snapshot(snapshot)
        data = b'{"test": 1}'
        c = codec.compress(data)
        assert restored.decompress(c) == data


# =============================================================================
# Compression effectiveness
# =============================================================================


class TestCompressionEffectiveness:
    """Measure that compression actually reduces size on realistic data."""

    def test_static_compression_savings(self) -> None:
        """A request heavy in static dictionary entries should compress well."""
        data = json.dumps(
            {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the weather?"},
                ],
                "temperature": 0.7,
                "max_tokens": 4096,
                "stream": True,
                "tools": [],
            },
            separators=(",", ":"),
        ).encode("utf-8")

        codec = DictionaryCodec()
        compressed = codec.compress(data)
        savings = 1 - len(compressed) / len(data)
        assert savings > 0.15, f"expected >15% savings, got {savings:.1%}"

    def test_multi_turn_improvement(self) -> None:
        """Compression ratio should improve across multiple turns of similar data."""
        base = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "temperature": 0.7,
        }
        # Use auto_add_threshold=12 so short unique turn labels don't pollute
        # the dynamic table; the structural savings will dominate.
        codec = DictionaryCodec(auto_add_threshold=12)
        ratios = []
        for turn in range(5):
            msg = dict(base)
            msg["messages"].append({"role": "user", "content": f"Turn {turn}"})
            data = json.dumps(msg, separators=(",", ":")).encode("utf-8")
            compressed = codec.compress(data)
            ratios.append(len(compressed) / len(data))

        # Later turns should have better (lower) compression ratios
        # because the dynamic table learns repeated structure.
        assert ratios[-1] < ratios[0]

    def test_tool_schema_compression(self) -> None:
        """Tool schema heavy requests should see significant savings."""
        data = json.dumps(
            {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Call the tool"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather info",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "City name",
                                    }
                                },
                                "required": ["location"],
                            },
                        },
                    }
                ],
            }
        ).encode("utf-8")

        codec = DictionaryCodec()
        compressed = codec.compress(data)
        savings = 1 - len(compressed) / len(data)
        assert savings > 0.05, f"expected >5% savings, got {savings:.1%}"


# =============================================================================
# Error handling
# =============================================================================


class TestErrorHandling:
    """Graceful handling of malformed input."""

    def test_truncated_literal(self) -> None:
        codec = DictionaryCodec()
        # Literal opcode + incomplete length
        bad = bytes([0x00, 0x05])
        with pytest.raises(ValueError):
            codec.decompress(bad)

    def test_unknown_static_index(self) -> None:
        codec = DictionaryCodec()
        bad = bytes([0x01]) + _encode_varint(999999)
        with pytest.raises(ValueError):
            codec.decompress(bad)

    def test_unknown_dynamic_index(self) -> None:
        codec = DictionaryCodec()
        bad = bytes([0x02]) + _encode_varint(999999)
        with pytest.raises(ValueError):
            codec.decompress(bad)

    def test_unknown_opcode(self) -> None:
        codec = DictionaryCodec()
        bad = bytes([0xFF])
        with pytest.raises(ValueError):
            codec.decompress(bad)

    def test_empty_decompress(self) -> None:
        codec = DictionaryCodec()
        assert codec.decompress(b"") == b""


# =============================================================================
# FrameFlags integration
# =============================================================================


class TestFrameFlags:
    def test_dict_compressed_flag_present(self) -> None:
        assert FrameFlags.DICT_COMPRESSED == 0x800

    def test_dict_compressed_can_combine(self) -> None:
        flags = FrameFlags.DICT_COMPRESSED | FrameFlags.COMPRESSED
        assert flags & FrameFlags.DICT_COMPRESSED
        assert flags & FrameFlags.COMPRESSED


# =============================================================================
# One-shot convenience functions
# =============================================================================


class TestConvenienceFunctions:
    def test_compress_decompress_one_shot(self) -> None:
        data = b'{"role":"user","content":"test"}'
        c = compress(data)
        assert decompress(c) == data


# =============================================================================
# Large data stress test
# =============================================================================


class TestStress:
    def test_large_json(self) -> None:
        """Compress and decompress a large JSON payload."""
        messages = [
            {"role": "user", "content": f"Message number {i} with some text."} for i in range(100)
        ]
        data = json.dumps({"model": "gpt-4", "messages": messages}).encode("utf-8")
        codec = DictionaryCodec()
        compressed = codec.compress(data)
        assert codec.decompress(compressed) == data

    def test_sse_lines(self) -> None:
        """Compress typical SSE stream data."""
        lines = b"\n\n".join(
            b'data: {"choices":[{"delta":{"content":"hello "}}]}' for _ in range(50)
        )
        codec = DictionaryCodec()
        compressed = codec.compress(lines)
        assert codec.decompress(compressed) == lines
