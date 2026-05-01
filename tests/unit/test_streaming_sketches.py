"""Tests for lattice.utils.streaming_sketches.

Covers:
- CountMinSketch construction, update, query, merge
- HyperLogLog construction, add, count, merge
- BloomFilter construction, add, contains, merge
- Validation of parameter constraints
"""

from __future__ import annotations

import pytest

from lattice.utils.streaming_sketches import BloomFilter, CountMinSketch, HyperLogLog


class TestCountMinSketch:
    def test_default_dimensions(self) -> None:
        cms = CountMinSketch()
        assert cms.w == 1024
        assert cms.d == 4

    def test_custom_dimensions(self) -> None:
        cms = CountMinSketch(w=512, d=3)
        assert cms.w == 512
        assert cms.d == 3

    def test_invalid_width(self) -> None:
        with pytest.raises(ValueError, match="w must be positive"):
            CountMinSketch(w=0)

    def test_invalid_depth(self) -> None:
        with pytest.raises(ValueError, match="d must be positive"):
            CountMinSketch(d=0)

    def test_update_and_query(self) -> None:
        cms = CountMinSketch()
        cms.update("foo", 5)
        cms.update("bar", 3)
        cms.update("foo", 2)
        assert cms.query("foo") == 7
        assert cms.query("bar") == 3
        assert cms.query("baz") == 0

    def test_query_never_undercounts(self) -> None:
        cms = CountMinSketch()
        for i in range(100):
            cms.update(str(i), 1)
        for i in range(100):
            assert cms.query(str(i)) >= 1

    def test_merge_sums_counts(self) -> None:
        cms1 = CountMinSketch()
        cms2 = CountMinSketch()
        cms1.update("foo", 5)
        cms2.update("foo", 3)
        merged = cms1.merge(cms2)
        assert merged.query("foo") == 8

    def test_merge_different_dimensions_fails(self) -> None:
        cms1 = CountMinSketch(w=1024, d=4)
        cms2 = CountMinSketch(w=512, d=4)
        with pytest.raises(ValueError, match="Cannot merge sketches with different dimensions"):
            cms1.merge(cms2)


class TestHyperLogLog:
    def test_default_precision(self) -> None:
        hll = HyperLogLog()
        assert hll.p == 14
        assert hll.m == 2**14

    def test_custom_precision(self) -> None:
        hll = HyperLogLog(p=10)
        assert hll.p == 10
        assert hll.m == 2**10

    def test_invalid_precision_low(self) -> None:
        with pytest.raises(ValueError, match="p must be between 4 and 18"):
            HyperLogLog(p=3)

    def test_invalid_precision_high(self) -> None:
        with pytest.raises(ValueError, match="p must be between 4 and 18"):
            HyperLogLog(p=19)

    def test_empty_count(self) -> None:
        hll = HyperLogLog()
        assert hll.count() == 0

    def test_single_item_count(self) -> None:
        hll = HyperLogLog()
        hll.add("foo")
        assert hll.count() >= 1

    def test_multiple_items_reasonable_estimate(self) -> None:
        hll = HyperLogLog()
        for i in range(1000):
            hll.add(f"item_{i}")
        estimate = hll.count()
        # Allow 10% relative error for 1000 items with p=14
        assert 900 <= estimate <= 1100

    def test_merge_union(self) -> None:
        hll1 = HyperLogLog()
        hll2 = HyperLogLog()
        for i in range(500):
            hll1.add(f"a_{i}")
            hll2.add(f"b_{i}")
        merged = hll1.merge(hll2)
        estimate = merged.count()
        # 1000 unique items, allow 10% error
        assert 900 <= estimate <= 1100

    def test_merge_different_precision_fails(self) -> None:
        hll1 = HyperLogLog(p=14)
        hll2 = HyperLogLog(p=10)
        with pytest.raises(ValueError, match="Cannot merge HLLs with different precision"):
            hll1.merge(hll2)


class TestBloomFilter:
    def test_default_parameters(self) -> None:
        bf = BloomFilter()
        assert bf.size == 10_000
        assert bf.num_hashes == 4

    def test_custom_parameters(self) -> None:
        bf = BloomFilter(size=5_000, num_hashes=2)
        assert bf.size == 5_000
        assert bf.num_hashes == 2

    def test_invalid_size(self) -> None:
        with pytest.raises(ValueError, match="size must be positive"):
            BloomFilter(size=0)

    def test_invalid_num_hashes(self) -> None:
        with pytest.raises(ValueError, match="num_hashes must be positive"):
            BloomFilter(num_hashes=0)

    def test_contains_empty(self) -> None:
        bf = BloomFilter()
        assert not bf.contains("foo")

    def test_add_and_contains(self) -> None:
        bf = BloomFilter()
        bf.add("foo")
        assert bf.contains("foo")
        assert not bf.contains("bar")

    def test_add_multiple_items(self) -> None:
        bf = BloomFilter()
        items = [f"item_{i}" for i in range(100)]
        for item in items:
            bf.add(item)
        for item in items:
            assert bf.contains(item)

    def test_merge_union(self) -> None:
        bf1 = BloomFilter()
        bf2 = BloomFilter()
        bf1.add("foo")
        bf2.add("bar")
        merged = bf1.merge(bf2)
        assert merged.contains("foo")
        assert merged.contains("bar")
        assert not merged.contains("baz")

    def test_merge_different_size_fails(self) -> None:
        bf1 = BloomFilter(size=10_000)
        bf2 = BloomFilter(size=5_000)
        with pytest.raises(
            ValueError, match="Cannot merge Bloom filters with different parameters"
        ):
            bf1.merge(bf2)

    def test_merge_different_num_hashes_fails(self) -> None:
        bf1 = BloomFilter(num_hashes=4)
        bf2 = BloomFilter(num_hashes=3)
        with pytest.raises(
            ValueError, match="Cannot merge Bloom filters with different parameters"
        ):
            bf1.merge(bf2)
