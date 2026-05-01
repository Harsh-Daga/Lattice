"""Tests for segment store."""

import asyncio

import pytest

from lattice.protocol.content import TextPart
from lattice.protocol.segments import SegmentType, build_segment
from lattice.state.segment_store import MemorySegmentStore


class TestMemorySegmentStore:
    @pytest.fixture
    def store(self):
        return MemorySegmentStore(ttl_seconds=3600)

    @pytest.mark.asyncio
    async def test_put_and_get(self, store):
        seg = build_segment(SegmentType.SYSTEM, [TextPart(text="hello")])
        is_new = await store.put(seg)
        assert is_new is True

        fetched = await store.get(seg.hash)
        assert fetched is not None
        assert fetched.hash == seg.hash

    @pytest.mark.asyncio
    async def test_put_duplicate(self, store):
        seg = build_segment(SegmentType.SYSTEM, [TextPart(text="hello")])
        await store.put(seg)
        is_new = await store.put(seg)
        assert is_new is False

    @pytest.mark.asyncio
    async def test_refcount(self, store):
        seg = build_segment(SegmentType.SYSTEM, [TextPart(text="hello")])
        await store.put(seg)
        await store.put(seg)  # refcount = 2

        removed = await store.dec_ref(seg.hash)
        assert removed is False
        assert await store.get(seg.hash) is not None

        removed = await store.dec_ref(seg.hash)
        assert removed is True
        assert await store.get(seg.hash) is None

    @pytest.mark.asyncio
    async def test_delete(self, store):
        seg = build_segment(SegmentType.SYSTEM, [TextPart(text="hello")])
        await store.put(seg)
        assert await store.delete(seg.hash) is True
        assert await store.get(seg.hash) is None
        assert await store.delete(seg.hash) is False

    @pytest.mark.asyncio
    async def test_keys(self, store):
        seg1 = build_segment(SegmentType.SYSTEM, [TextPart(text="a")])
        seg2 = build_segment(SegmentType.TOOLS, [TextPart(text="b")])
        await store.put(seg1)
        await store.put(seg2)
        keys = await store.keys()
        assert seg1.hash in keys
        assert seg2.hash in keys

    @pytest.mark.asyncio
    async def test_ttl_eviction(self, store):
        store = MemorySegmentStore(ttl_seconds=0)
        seg = build_segment(SegmentType.SYSTEM, [TextPart(text="hello")])
        await store.put(seg)
        await asyncio.sleep(0.01)
        assert await store.get(seg.hash) is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self, store):
        store = MemorySegmentStore(max_segments=2)
        seg1 = build_segment(SegmentType.SYSTEM, [TextPart(text="a")])
        seg2 = build_segment(SegmentType.SYSTEM, [TextPart(text="b")])
        seg3 = build_segment(SegmentType.SYSTEM, [TextPart(text="c")])
        await store.put(seg1)
        await store.put(seg2)
        await asyncio.sleep(0.01)
        await store.put(seg3)
        # seg1 should be evicted (oldest, refcount=1)
        assert await store.get(seg1.hash) is None
        assert await store.get(seg2.hash) is not None
        assert await store.get(seg3.hash) is not None

    @pytest.mark.asyncio
    async def test_stats(self, store):
        seg = build_segment(SegmentType.SYSTEM, [TextPart(text="hello")])
        await store.put(seg)
        stats = store.stats
        assert stats["segments"] == 1
        assert stats["total_refcount"] >= 1

    @pytest.mark.asyncio
    async def test_concurrent_access(self, store):
        seg = build_segment(SegmentType.SYSTEM, [TextPart(text="hello")])

        async def worker():
            await store.put(seg)
            await store.get(seg.hash)
            await store.inc_ref(seg.hash)
            await store.dec_ref(seg.hash)

        await asyncio.gather(*[worker() for _ in range(10)])
        assert await store.get(seg.hash) is not None
