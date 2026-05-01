"""HPACK/QPACK-inspired shared dictionary codec for the client↔proxy hop.

This module implements the wire-level compression protocol described in
ADR-002. It operates on raw bytes (typically JSON), tokenizing them into
a sequence of structural tokens, keys, and string literals. Each token is
replaced by a dictionary reference or emitted as a length-prefixed literal.

Protocol
--------
The compressed byte stream is a sequence of operations:

    0x00 <uint16 length> <bytes>       — literal, do NOT add to dynamic table
    0x80 <uint16 length> <bytes>       — literal, ADD to dynamic table
    0x01 <varint index>                — static dictionary reference
    0x02 <varint index>                — dynamic dictionary reference

The static dictionary is hard-coded (see ``dictionary_static.py``).
The dynamic dictionary is per-session, LRU-evicted, max 1024 entries by
default. Both client and proxy maintain the same dynamic table state by
applying identical ADD operations in identical order.

Because the ADD opcode (0x80) is explicit, both sides stay perfectly in
sync without any side-channel communication. Eviction is deterministic:
when the table is full and a new entry arrives, the least-recently-used
entry is evicted. Since both sides see the same sequence of ADDs, they
evict the same entries.

Usage
-----
    from lattice.protocol.dictionary_codec import DictionaryCodec

    codec = DictionaryCodec()
    compressed = codec.compress(b'{"role":"user","content":"hello"}')
    decompressed = codec.decompress(compressed)
    assert decompressed == b'{"role":"user","content":"hello"}'

    # Per-session usage:
    session_codec = DictionaryCodec(session_id="sess_abc")
    c1 = session_codec.compress(req1)
    c2 = session_codec.compress(req2)  # benefits from dynamic table learned in req1

Integration
-----------
- ``BinaryFramer`` sets ``FrameFlags.DICT_COMPRESSED`` on frames whose
  payload has been processed through this codec.
- ``Session.metadata["dict_compression_state"]`` stores the serialized
  dynamic table for session resumption.
- The proxy decompresses incoming frames before JSON parsing and
  compresses outgoing responses before framing.
"""

from __future__ import annotations

import struct
from collections import OrderedDict
from typing import Any

from lattice.protocol.dictionary_static import (
    STATIC_COUNT,
    STATIC_REVERSE,
    STATIC_TABLE,
)

# =============================================================================
# Opcodes
# =============================================================================

_OP_LITERAL = 0x00
_OP_LITERAL_ADD = 0x80
_OP_STATIC_REF = 0x01
_OP_DYNAMIC_REF = 0x02

# Mask to extract the base opcode (ignore add flag)
_OP_MASK = 0x7F

# =============================================================================
# Varint helpers (Protocol Buffers style, 7 bits per byte)
# =============================================================================


def _encode_varint(value: int) -> bytes:
    """Encode a non-negative integer as a varint.

    Args:
        value: Must be >= 0.

    Returns:
        1-5 bytes for values up to 2^31-1.
    """
    if value < 0:
        raise ValueError("varint must be non-negative")
    result = bytearray()
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def _decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Decode a varint from *data* starting at *offset*.

    Returns:
        (decoded_value, new_offset)

    Raises:
        ValueError: If the varint is malformed or exceeds bounds.
    """
    result = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, offset
        shift += 7
        if shift >= 35:  # Prevent excessive shift (max 5 bytes for 32-bit)
            raise ValueError("varint too long")
    raise ValueError("truncated varint")


# =============================================================================
# JSON tokenizer (flat token stream)
# =============================================================================


def _tokenize_json(data: bytes) -> list[bytes]:
    """Tokenize JSON bytes into a flat sequence of tokens.

    Tokens are:
    - Single-byte structural tokens: { } [ ] : , "
    - String literals (including the surrounding quotes)
    - Numbers (integers, floats, scientific notation)
    - Keywords: true, false, null
    - Whitespace sequences (preserved for round-trip fidelity)

    This is a hand-rolled lexer optimized for speed over correctness
    on edge cases. It handles all standard LLM API JSON shapes.

    Args:
        data: Raw JSON bytes.

    Returns:
        List of token byte strings.
    """
    tokens: list[bytes] = []
    i = 0
    n = len(data)

    while i < n:
        b = data[i]

        # String literal — MUST check before structural bytes because
        # the quote character (0x22) is also a structural token.
        if b == 0x22:  # '"'
            j = _find_string_end(data, i)
            tokens.append(data[i:j])
            i = j
            continue

        # Fast path: single-byte structural tokens (excluding 0x22 handled above)
        if b in _STRUCTURAL_BYTES:
            tokens.append(bytes([b]))
            i += 1
            continue

        # Number
        if b in _NUMBER_START_BYTES:
            j = i + 1
            while j < n and data[j] in _NUMBER_BODY_BYTES:
                j += 1
            tokens.append(data[i:j])
            i = j
            continue

        # Keyword: true, false, null
        if b in _KEYWORD_START_BYTES:
            if i + 4 <= n and data[i : i + 4] == b"true":
                tokens.append(b"true")
                i += 4
                continue
            if i + 5 <= n and data[i : i + 5] == b"false":
                tokens.append(b"false")
                i += 5
                continue
            if i + 4 <= n and data[i : i + 4] == b"null":
                tokens.append(b"null")
                i += 4
                continue

        # Whitespace
        if b in _WHITESPACE_BYTES:
            j = i + 1
            while j < n and data[j] in _WHITESPACE_BYTES:
                j += 1
            tokens.append(data[i:j])
            i = j
            continue

        # Unknown byte — emit as single-byte literal to avoid data loss
        tokens.append(bytes([b]))
        i += 1

    return tokens


# Pre-computed byte sets for fast membership testing
_STRUCTURAL_BYTES = frozenset({0x7B, 0x7D, 0x5B, 0x5D, 0x3A, 0x2C, 0x22})  # { } [ ] : , "
_NUMBER_START_BYTES = frozenset(
    {0x2D, 0x2B, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39}
)  # - + 0-9
_NUMBER_BODY_BYTES = frozenset(
    {
        0x2D,
        0x2B,
        0x2E,
        0x30,
        0x31,
        0x32,
        0x33,
        0x34,
        0x35,
        0x36,
        0x37,
        0x38,
        0x39,
        0x65,
        0x45,
    }
)  # - + . 0-9 e E
_KEYWORD_START_BYTES = frozenset({0x74, 0x66, 0x6E})  # t f n
_WHITESPACE_BYTES = frozenset({0x20, 0x09, 0x0A, 0x0D})  # space \t \n \r


def _find_string_end(data: bytes, start: int) -> int:
    """Find the end of a JSON string starting at *start* (which must be '"').

    Handles escaped quotes and escaped backslashes.
    Returns the index AFTER the closing quote.
    """
    i = start + 1
    n = len(data)
    while i < n:
        b = data[i]
        if b == 0x22:  # unescaped quote
            return i + 1
        if b == 0x5C and i + 1 < n:  # backslash escape
            i += 2
        else:
            i += 1
    return n  # unterminated string — return end of data


# =============================================================================
# Dynamic table
# =============================================================================


class _DynamicTable:
    """Per-session dynamic dictionary with LRU eviction.

    Both the encoder and decoder maintain an identical instance.
    The table is kept in sync by processing the same sequence of
    ADD operations in the same order.

    Attributes:
        max_entries: Maximum number of dynamic entries.
        entries: OrderedDict mapping index -> text (LRU order: oldest first).
        reverse: dict mapping text -> index.
        next_index: The next index to assign to a new entry.
    """

    def __init__(self, max_entries: int = 1024) -> None:
        self.max_entries = max(max_entries, 1)
        # OrderedDict with oldest entries at the front
        self.entries: OrderedDict[int, str] = OrderedDict()
        self.reverse: dict[str, int] = {}
        # Dynamic indices start after the static table
        self.next_index: int = STATIC_COUNT

    def add(self, text: str) -> int:
        """Add *text* to the dynamic table and return its index.

        If *text* is already present, it is promoted to most-recently-used
        and its existing index is returned.

        If the table is full, the least-recently-used entry is evicted.
        """
        if text in self.reverse:
            idx = self.reverse[text]
            # Promote to MRU: remove and re-insert at end
            self.entries.move_to_end(idx)
            return idx

        # Evict LRU entries if needed
        while len(self.entries) >= self.max_entries:
            self._evict_lru()

        idx = self.next_index
        self.next_index += 1
        self.entries[idx] = text
        self.reverse[text] = idx
        return idx

    def lookup(self, index: int) -> str | None:
        """Look up an entry by index.

        If found, promotes it to MRU and returns the text.
        Returns None if the index is not in this table.
        """
        text = self.entries.get(index)
        if text is not None:
            self.entries.move_to_end(index)
        return text

    def _evict_lru(self) -> None:
        """Remove the least-recently-used entry."""
        if not self.entries:
            return
        idx, text = self.entries.popitem(last=False)
        self.reverse.pop(text, None)

    def to_snapshot(self) -> dict[str, Any]:
        """Serialize table state for session persistence.

        Returns a compact dict suitable for JSON storage.
        """
        return {
            "max_entries": self.max_entries,
            "next_index": self.next_index,
            # Store in LRU order (oldest first) so both sides rebuild identically
            "entries": list(self.entries.items()),  # [(idx, text), ...]
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> _DynamicTable:
        """Restore a dynamic table from a snapshot."""
        inst = cls(max_entries=data.get("max_entries", 1024))
        inst.next_index = data.get("next_index", STATIC_COUNT)
        for idx, text in data.get("entries", []):
            inst.entries[idx] = text
            inst.reverse[text] = idx
        return inst


# =============================================================================
# DictionaryCodec
# =============================================================================


class DictionaryCodec:
    """Compress and decompress byte streams using shared static + dynamic dictionaries.

    Each codec instance holds its own dynamic table. For per-session
    compression, create one codec per session and reuse it across requests.

    Args:
        session_id: Optional session identifier (used for logging/metrics only).
        max_dynamic_entries: Maximum dynamic table size. Default 1024.
        auto_add_threshold: Literals longer than this many bytes are
            automatically emitted with the ADD flag (0x80) so they enter
            the dynamic table. Default 8. Set to 0 to disable auto-add.
    """

    def __init__(
        self,
        session_id: str | None = None,
        *,
        max_dynamic_entries: int = 1024,
        auto_add_threshold: int = 8,
    ) -> None:
        self.session_id = session_id
        self.dynamic = _DynamicTable(max_entries=max_dynamic_entries)
        self.auto_add_threshold = max(0, auto_add_threshold)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, data: bytes) -> bytes:
        """Compress *data* using the shared dictionaries.

        The output is a sequence of opcodes that can be fed to
        :meth:`decompress` to recover the original bytes exactly.

        Args:
            data: Raw bytes (typically UTF-8 JSON).

        Returns:
            Compressed byte stream.
        """
        tokens = _tokenize_json(data)
        out = bytearray()

        for token in tokens:
            text = token.decode("utf-8", errors="replace")

            # Try static dictionary first
            static_idx = STATIC_REVERSE.get(text)
            if static_idx is not None:
                out.append(_OP_STATIC_REF)
                out.extend(_encode_varint(static_idx))
                continue

            # Try dynamic dictionary
            dyn_idx = self.dynamic.reverse.get(text)
            if dyn_idx is not None:
                out.append(_OP_DYNAMIC_REF)
                out.extend(_encode_varint(dyn_idx))
                # Promote to MRU on encode side
                self.dynamic.lookup(dyn_idx)
                continue

            # Literal — decide whether to auto-add
            token_bytes = text.encode("utf-8")
            add_to_dict = len(token_bytes) >= self.auto_add_threshold

            if add_to_dict:
                out.append(_OP_LITERAL_ADD)
                self.dynamic.add(text)
            else:
                out.append(_OP_LITERAL)

            out.extend(struct.pack("<H", len(token_bytes)))
            out.extend(token_bytes)

        return bytes(out)

    def decompress(self, data: bytes) -> bytes:
        """Decompress a byte stream produced by :meth:`compress`.

        Args:
            data: Compressed byte stream.

        Returns:
            Original bytes.

        Raises:
            ValueError: If the stream is malformed.
        """
        out = bytearray()
        i = 0
        n = len(data)

        while i < n:
            opcode = data[i]
            i += 1
            base = opcode & _OP_MASK

            if base == _OP_LITERAL:
                # Literal (possibly with ADD flag)
                if i + 2 > n:
                    raise ValueError("truncated literal length")
                length = struct.unpack("<H", data[i : i + 2])[0]
                i += 2
                if i + length > n:
                    raise ValueError("truncated literal payload")
                token_bytes = data[i : i + length]
                i += length
                out.extend(token_bytes)

                if opcode & 0x80:  # ADD flag set
                    text = token_bytes.decode("utf-8", errors="replace")
                    self.dynamic.add(text)
                continue

            if base == _OP_STATIC_REF:
                idx, i = _decode_varint(data, i)
                text = STATIC_TABLE.get(idx)  # type: ignore[assignment]
                if text is None:
                    raise ValueError(f"static dictionary index {idx} not found")
                out.extend(text.encode("utf-8"))
                continue

            if base == _OP_DYNAMIC_REF:
                idx, i = _decode_varint(data, i)
                text = self.dynamic.lookup(idx)  # type: ignore[assignment]
                if text is None:
                    raise ValueError(f"dynamic dictionary index {idx} not found")
                out.extend(text.encode("utf-8"))
                continue

            raise ValueError(f"unknown opcode: 0x{opcode:02x}")

        return bytes(out)

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def to_snapshot(self) -> dict[str, Any]:
        """Serialize codec state for session storage.

        Returns:
            A JSON-serializable dict.
        """
        return {
            "session_id": self.session_id,
            "auto_add_threshold": self.auto_add_threshold,
            "dynamic": self.dynamic.to_snapshot(),
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> DictionaryCodec:
        """Restore a codec from a snapshot.

        Args:
            data: Dict produced by :meth:`to_snapshot`.

        Returns:
            A new DictionaryCodec with the restored dynamic table.
        """
        inst = cls(
            session_id=data.get("session_id"),
            auto_add_threshold=data.get("auto_add_threshold", 8),
            max_dynamic_entries=data.get("dynamic", {}).get("max_entries", 1024),
        )
        inst.dynamic = _DynamicTable.from_snapshot(data.get("dynamic", {}))
        return inst

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        """Return current compression statistics."""
        return {
            "static_entries": STATIC_COUNT,
            "dynamic_entries": len(self.dynamic.entries),
            "dynamic_max": self.dynamic.max_entries,
            "auto_add_threshold": self.auto_add_threshold,
            "session_id": self.session_id,
        }


# =============================================================================
# Convenience functions for one-shot compression
# =============================================================================


def compress(data: bytes, codec: DictionaryCodec | None = None) -> bytes:
    """One-shot compression. Creates a fresh codec if none provided."""
    c = codec or DictionaryCodec()
    return c.compress(data)


def decompress(data: bytes, codec: DictionaryCodec | None = None) -> bytes:
    """One-shot decompression. Creates a fresh codec if none provided."""
    c = codec or DictionaryCodec()
    return c.decompress(data)


__all__ = [
    "DictionaryCodec",
    "compress",
    "decompress",
    "_DynamicTable",
    "_encode_varint",
    "_decode_varint",
    "_tokenize_json",
]
