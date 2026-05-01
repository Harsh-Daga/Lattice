"""Streaming data sketches for approximate frequency, cardinality, and membership queries.

Pure-Python implementations suitable for tracking tokens, chunks, and phrases
with sub-linear memory.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


class CountMinSketch:
    """Approximate frequency counting for token/phrase tracking.

    Parameters
    ----------
    w: int
        Width of each hash table (default 1024).
    d: int
        Number of hash tables (default 4).
    """

    def __init__(self, w: int = 1024, d: int = 4) -> None:
        if w <= 0:
            raise ValueError("w must be positive")
        if d <= 0:
            raise ValueError("d must be positive")
        self.w = w
        self.d = d
        self._table: list[list[int]] = [[0] * w for _ in range(d)]

    def _hash(self, item: str, seed: int) -> int:
        return (hash((item, seed)) & 0xFFFFFFFF) % self.w

    def update(self, item: str, delta: int = 1) -> None:
        """Increment the estimated count for *item* by *delta*."""
        for seed in range(self.d):
            idx = self._hash(item, seed)
            self._table[seed][idx] += delta

    def query(self, item: str) -> int:
        """Return the minimum estimated count across all tables."""
        return min(self._table[seed][self._hash(item, seed)] for seed in range(self.d))

    def merge(self, other: CountMinSketch) -> CountMinSketch:
        """Return a new sketch that is the point-wise sum of this and *other*."""
        if self.w != other.w or self.d != other.d:
            raise ValueError("Cannot merge sketches with different dimensions")
        merged = CountMinSketch(self.w, self.d)
        for i in range(self.d):
            for j in range(self.w):
                merged._table[i][j] = self._table[i][j] + other._table[i][j]
        return merged

    def __repr__(self) -> str:  # pragma: no cover
        return f"CountMinSketch(w={self.w}, d={self.d})"


class HyperLogLog:
    """Approximate distinct count. 1.5KB with 2% error.

    Parameters
    ----------
    p: int
        Precision (default 14).  Register count = 2**p.
    """

    def __init__(self, p: int = 14) -> None:
        if not (4 <= p <= 18):
            raise ValueError("p must be between 4 and 18")
        self.p = p
        self.m = 1 << p  # 2**p
        self._registers: list[int] = [0] * self.m
        self._alpha_value = self._alpha(p)

    @classmethod
    def _alpha(cls, p: int) -> float:
        if p == 4:
            return 0.673
        if p == 5:
            return 0.697
        if p == 6:
            return 0.709
        return 0.7213 / (1 + 1.079 / (1 << p))

    def _hash(self, item: str) -> int:
        # 32-bit hash, top p bits select register, remaining bits used for rank
        return hash(item) & 0xFFFFFFFF

    def add(self, item: str) -> None:
        """Add *item* to the HLL set."""
        h = self._hash(item)
        idx = h >> (32 - self.p)
        # Count leading zeros + 1 in the remaining 32-p bits
        bits = h & ((1 << (32 - self.p)) - 1)
        rank = self._leading_zero_count(bits, 32 - self.p) + 1
        if rank > self._registers[idx]:
            self._registers[idx] = rank

    @staticmethod
    def _leading_zero_count(value: int, width: int) -> int:
        """Count leading zeros in *value* treated as *width* bits."""
        if value == 0:
            return width
        return width - value.bit_length()

    def count(self) -> int:
        """Return the estimated cardinality."""
        raw_estimate = self._alpha_value * (self.m ** 2) / sum(2 ** -r for r in self._registers)

        # Small range correction
        if raw_estimate <= 2.5 * self.m:
            zeros = self._registers.count(0)
            if zeros != 0:
                return int(round(self.m * math.log(self.m / zeros)))

        # Large range correction (for 32-bit hash)
        if raw_estimate > (1 << 32) / 30.0:
            return int(round(-(1 << 32) * math.log(1 - raw_estimate / (1 << 32))))

        return int(round(raw_estimate))

    def merge(self, other: HyperLogLog) -> HyperLogLog:
        """Return a new HLL that is the union of this and *other*."""
        if self.p != other.p:
            raise ValueError("Cannot merge HLLs with different precision")
        merged = HyperLogLog(self.p)
        for i in range(self.m):
            merged._registers[i] = max(self._registers[i], other._registers[i])
        return merged

    def __repr__(self) -> str:  # pragma: no cover
        return f"HyperLogLog(p={self.p}, m={self.m})"


class BloomFilter:
    """O(1) "have we seen this chunk before?" check.

    Parameters
    ----------
    size: int
        Bit array length (default 10_000).
    num_hashes: int
        Number of hash functions (default 4).
    """

    def __init__(self, size: int = 10_000, num_hashes: int = 4) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if num_hashes <= 0:
            raise ValueError("num_hashes must be positive")
        self.size = size
        self.num_hashes = num_hashes
        self._bits: list[bool] = [False] * size

    def _hashes(self, item: str) -> Iterable[int]:
        for seed in range(self.num_hashes):
            yield (hash((item, seed)) & 0xFFFFFFFF) % self.size

    def add(self, item: str) -> None:
        """Insert *item* into the filter."""
        for idx in self._hashes(item):
            self._bits[idx] = True

    def contains(self, item: str) -> bool:
        """Return True if *item* is probably in the set, False if definitely not."""
        return all(self._bits[idx] for idx in self._hashes(item))

    def merge(self, other: BloomFilter) -> BloomFilter:
        """Return a new Bloom filter that is the union of this and *other*."""
        if self.size != other.size or self.num_hashes != other.num_hashes:
            raise ValueError(
                "Cannot merge Bloom filters with different parameters"
            )
        merged = BloomFilter(self.size, self.num_hashes)
        merged._bits = [a or b for a, b in zip(self._bits, other._bits)]
        return merged

    def __repr__(self) -> str:  # pragma: no cover
        return f"BloomFilter(size={self.size}, num_hashes={self.num_hashes})"
