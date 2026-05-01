"""Tests for semantic response cache.

Covers:
- Cache key stability (same request = same key)
- Cache key uniqueness (different request = different key)
- TTL expiration
- LRU eviction
- Max entry size rejection
- Streaming response assembly and SSE chunk generation
- Cache hit/miss rate tracking
- Invalidation by key and pattern
"""

from __future__ import annotations

import asyncio

import pytest

from lattice.core.semantic_cache import (
    CachedResponse,
    ContentClass,
    InMemoryCacheBackend,
    SemanticCache,
    _detect_content_class,
    assemble_cached_response,
    compute_cache_key,
    generate_sse_chunks,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache():
    return SemanticCache(ttl_seconds=60, max_entries=100, enabled=True)


@pytest.fixture
def fast_expiring_cache():
    return SemanticCache(ttl_seconds=0.1, max_entries=100, enabled=True)


# ---------------------------------------------------------------------------
# Cache key computation
# ---------------------------------------------------------------------------


class DummyRequest:
    """Minimal request stand-in."""

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def to_dict(self):
        return dict(self.__dict__)


class TestCacheKey:
    def test_same_request_same_key(self) -> None:
        r1 = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
        )
        r2 = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
        )
        assert compute_cache_key(r1) == compute_cache_key(r2)

    def test_different_model_different_key(self) -> None:
        r1 = DummyRequest(model="gpt-4", messages=[{"role": "user", "content": "hi"}])
        r2 = DummyRequest(model="gpt-3.5", messages=[{"role": "user", "content": "hi"}])
        assert compute_cache_key(r1) != compute_cache_key(r2)

    def test_different_messages_different_key(self) -> None:
        r1 = DummyRequest(model="gpt-4", messages=[{"role": "user", "content": "hi"}])
        r2 = DummyRequest(model="gpt-4", messages=[{"role": "user", "content": "hello"}])
        assert compute_cache_key(r1) != compute_cache_key(r2)

    def test_stream_excluded_from_key(self) -> None:
        """Streaming flag should NOT affect cache key."""
        r1 = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        r2 = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
        )
        assert compute_cache_key(r1) == compute_cache_key(r2)

    def test_metadata_excluded_from_key(self) -> None:
        """Session metadata should NOT affect cache key."""
        r1 = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"session_id": "abc"},
        )
        r2 = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            metadata={"session_id": "xyz"},
        )
        assert compute_cache_key(r1) == compute_cache_key(r2)

    def test_tools_included_in_key(self) -> None:
        r1 = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "get_weather"}],
        )
        r2 = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert compute_cache_key(r1) != compute_cache_key(r2)


# ---------------------------------------------------------------------------
# Basic get / set
# ---------------------------------------------------------------------------


class TestBasicOperations:
    async def test_get_missing_returns_none(self, cache) -> None:
        assert await cache.get("nonexistent") is None

    async def test_set_and_get(self, cache) -> None:
        resp = CachedResponse(content="Hello, world!")
        await cache.set("key1", resp)
        hit = await cache.get("key1")
        assert hit is not None
        assert hit.content == "Hello, world!"

    async def test_disabled_cache_always_misses(self) -> None:
        disabled = SemanticCache(enabled=False)
        await disabled.set("key", CachedResponse(content="x"))
        assert await disabled.get("key") is None

    async def test_set_returns_false_when_disabled(self) -> None:
        disabled = SemanticCache(enabled=False)
        assert await disabled.set("key", CachedResponse(content="x")) is False

    async def test_semantic_lookup_hits_on_near_duplicate_request(self) -> None:
        cache = SemanticCache(ttl_seconds=60, max_entries=100, enabled=True)
        exact = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Explain how to sort a list in Python."}],
            temperature=0.2,
        )
        near = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Explain how to sort a list in Python!"}],
            temperature=0.2,
        )

        key = compute_cache_key(exact)
        await cache.set(key, CachedResponse(content="Use sorted(list)."), exact)

        assert compute_cache_key(exact) != compute_cache_key(near)
        hit = await cache.get(compute_cache_key(near), near)
        assert hit is not None
        assert hit.content == "Use sorted(list)."

        stats = await cache.stats
        assert stats["semantic_hits"] == 1
        assert stats["semantic_misses"] == 0


# ---------------------------------------------------------------------------
# TTL expiration
# ---------------------------------------------------------------------------


class TestTTL:
    async def test_entry_expires(self, fast_expiring_cache) -> None:
        await fast_expiring_cache.set("key", CachedResponse(content="x"))
        assert await fast_expiring_cache.get("key") is not None
        await asyncio.sleep(0.15)
        assert await fast_expiring_cache.get("key") is None

    async def test_expire_stale_removes_expired(self, fast_expiring_cache) -> None:
        await fast_expiring_cache.set("a", CachedResponse(content="x"))
        await fast_expiring_cache.set("b", CachedResponse(content="y"))
        await asyncio.sleep(0.15)
        removed = await fast_expiring_cache.expire_stale()
        assert removed == 2
        assert await fast_expiring_cache.get("a") is None
        assert await fast_expiring_cache.get("b") is None


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestLRUEviction:
    async def test_eviction_when_full(self) -> None:
        small = SemanticCache(ttl_seconds=3600, max_entries=3)
        for i in range(4):
            await small.set(f"key{i}", CachedResponse(content=f"v{i}"))

        # Oldest entry (key0) should be evicted
        assert await small.get("key0") is None
        for i in range(1, 4):
            assert (await small.get(f"key{i}")) is not None

    async def test_access_promotes_mru(self) -> None:
        small = SemanticCache(ttl_seconds=3600, max_entries=3)
        for i in range(3):
            await small.set(f"key{i}", CachedResponse(content=f"v{i}"))

        # Access key0 to promote it to MRU
        await small.get("key0")
        # Add new entry — should evict key1 (now LRU)
        await small.set("key3", CachedResponse(content="v3"))
        assert await small.get("key0") is not None
        assert await small.get("key1") is None
        assert await small.get("key2") is not None
        assert await small.get("key3") is not None


# ---------------------------------------------------------------------------
# Max entry size
# ---------------------------------------------------------------------------


class TestMaxEntrySize:
    async def test_large_entry_rejected(self) -> None:
        tiny = SemanticCache(ttl_seconds=3600, max_entries=100, max_entry_size_kb=1)
        big = CachedResponse(content="x" * 2048)  # 2 KB
        assert await tiny.set("key", big) is False
        assert await tiny.get("key") is None

    async def test_small_entry_accepted(self) -> None:
        tiny = SemanticCache(ttl_seconds=3600, max_entries=100, max_entry_size_kb=1)
        small = CachedResponse(content="hello")
        assert await tiny.set("key", small) is True
        assert await tiny.get("key") is not None


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------


class TestInvalidation:
    async def test_invalidate_existing(self, cache) -> None:
        await cache.set("k", CachedResponse(content="x"))
        assert await cache.invalidate("k") is True
        assert await cache.get("k") is None

    async def test_invalidate_missing(self, cache) -> None:
        assert await cache.invalidate("nonexistent") is False

    async def test_invalidate_by_pattern(self, cache) -> None:
        await cache.set("a", CachedResponse(content="x"))
        await cache.set("b", CachedResponse(content="x"))
        removed = await cache.invalidate_by_pattern(lambda r: r.content == "x")
        assert removed == 2

    async def test_clear(self, cache) -> None:
        await cache.set("a", CachedResponse(content="x"))
        await cache.set("b", CachedResponse(content="y"))
        assert await cache.clear() == 2
        assert await cache.get("a") is None
        assert await cache.get("b") is None


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    async def test_hit_rate(self, cache) -> None:
        await cache.set("k", CachedResponse(content="x"))
        await cache.get("k")  # hit
        await cache.get("k")  # hit
        await cache.get("missing")  # miss
        stats = await cache.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == round(2 / 3, 4)

    async def test_entry_count(self, cache) -> None:
        await cache.set("a", CachedResponse(content="x"))
        await cache.set("b", CachedResponse(content="y"))
        assert cache.entry_count == 2


# ---------------------------------------------------------------------------
# SSE chunk generation
# ---------------------------------------------------------------------------


class TestSSEChunks:
    def test_generate_chunks_from_content(self) -> None:
        cached = CachedResponse(content="Hello world", model="gpt-4")
        chunks = generate_sse_chunks(cached, chunk_size=5)
        assert len(chunks) >= 3  # first + content chunks + final
        # First chunk should have role delta
        assert "assistant" in chunks[0]
        # Content chunks
        content_parts = []
        for chunk in chunks[1:-1]:
            if '"content":' in chunk:
                data = chunk.removeprefix("data: ").strip()
                import json
                payload = json.loads(data)
                content_parts.append(payload["choices"][0]["delta"]["content"])
        assert "".join(content_parts) == "Hello world"
        # Final chunk has finish_reason
        assert "stop" in chunks[-1]

    def test_tool_calls_included(self) -> None:
        cached = CachedResponse(
            content="",
            model="gpt-4",
            tool_calls=[{"id": "tc_1", "type": "function"}],
        )
        chunks = generate_sse_chunks(cached)
        # Should have tool call chunk before final
        tool_found = any('"tool_calls"' in c for c in chunks)
        assert tool_found

    def test_reuses_existing_sse_chunks(self) -> None:
        cached = CachedResponse(
            content="Hello",
            sse_chunks=["data: foo\n\n", "data: bar\n\n"],
        )
        chunks = generate_sse_chunks(cached)
        assert chunks == ["data: foo\n\n", "data: bar\n\n"]


# ---------------------------------------------------------------------------
# Assemble cached response
# ---------------------------------------------------------------------------


class TestAssemble:
    def test_assemble_basic(self) -> None:
        resp = assemble_cached_response(
            model="gpt-4",
            content="Hi",
            tool_calls=None,
            usage={"total_tokens": 10},
        )
        assert resp.model == "gpt-4"
        assert resp.content == "Hi"
        assert resp.usage["total_tokens"] == 10
        assert resp.expires_at == 0  # set later by SemanticCache.set


# ---------------------------------------------------------------------------
# Concurrency safety
# ---------------------------------------------------------------------------


class TestConcurrency:
    async def test_concurrent_reads(self, cache) -> None:
        await cache.set("k", CachedResponse(content="shared"))

        async def read() -> CachedResponse | None:
            return await cache.get("k")

        results = await asyncio.gather(*(read() for _ in range(100)))
        assert all(r is not None and r.content == "shared" for r in results)

    async def test_concurrent_writes(self) -> None:
        c = SemanticCache(ttl_seconds=3600, max_entries=1000)

        async def write(i: int) -> bool:
            return await c.set(f"key{i}", CachedResponse(content=f"v{i}"))

        await asyncio.gather(*(write(i) for i in range(100)))
        assert c.entry_count == 100


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    async def test_empty_content(self, cache) -> None:
        await cache.set("k", CachedResponse(content=""))
        hit = await cache.get("k")
        assert hit is not None
        assert hit.content == ""

    async def test_unicode_content(self, cache) -> None:
        await cache.set("k", CachedResponse(content="中文测试"))
        hit = await cache.get("k")
        assert hit is not None
        assert hit.content == "中文测试"

    async def test_none_tool_calls(self, cache) -> None:
        await cache.set("k", CachedResponse(content="x", tool_calls=None))
        hit = await cache.get("k")
        assert hit.tool_calls is None


# ---------------------------------------------------------------------------
# Structured approximate cache (Phase 2)
# ---------------------------------------------------------------------------


class TestStructuredApproximateCache:
    async def test_exact_hit_still_works(self, cache) -> None:
        resp = CachedResponse(content="Hello, world!")
        await cache.set("key1", resp)
        hit = await cache.get("key1")
        assert hit is not None
        assert hit.content == "Hello, world!"
        stats = await cache.stats
        assert stats["exact_hits"] == 1

    async def test_near_duplicate_hit_same_class(self) -> None:
        cache = SemanticCache(ttl_seconds=60, max_entries=100, enabled=True)
        exact = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Explain how to sort a list in Python."}],
            temperature=0.2,
        )
        near = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Explain how to sort a list in Python!"}],
            temperature=0.2,
        )
        key = compute_cache_key(exact)
        await cache.set(key, CachedResponse(content="Use sorted(list)."), exact)
        hit = await cache.get(compute_cache_key(near), near)
        assert hit is not None
        assert hit.content == "Use sorted(list)."
        stats = await cache.stats
        assert stats["approximate_hits"] >= 1

    async def test_different_content_classes_no_false_hit(self) -> None:
        cache = SemanticCache(ttl_seconds=60, max_entries=100, enabled=True)
        plain = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Explain how to sort a list in Python."}],
        )
        code = DummyRequest(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": "Here is some code:\n```python\nprint('hello')\n```\nWhat does this do?",
            }],
        )
        key = compute_cache_key(plain)
        await cache.set(key, CachedResponse(content="Use sorted(list)."), plain)
        hit = await cache.get(compute_cache_key(code), code)
        assert hit is None

    async def test_tool_prompt_requires_same_tools(self) -> None:
        cache = SemanticCache(ttl_seconds=60, max_entries=100, enabled=True)
        req_a = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Get weather"}],
            tools=[{"name": "get_weather", "parameters": {}}],
        )
        req_b = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Get weather"}],
            tools=[{"name": "get_stock", "parameters": {}}],
        )
        key = compute_cache_key(req_a)
        await cache.set(key, CachedResponse(content="sunny"), req_a)
        hit = await cache.get(compute_cache_key(req_b), req_b)
        assert hit is None

    async def test_expired_entries_never_returned(self, fast_expiring_cache) -> None:
        await fast_expiring_cache.set("key", CachedResponse(content="x"))
        assert await fast_expiring_cache.get("key") is not None
        await asyncio.sleep(0.15)
        assert await fast_expiring_cache.get("key") is None

    async def test_stats_report_exact_vs_approximate_separately(self) -> None:
        cache = SemanticCache(ttl_seconds=60, max_entries=100, enabled=True)
        exact = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello world"}],
        )
        near = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello world!"}],
        )
        key = compute_cache_key(exact)
        await cache.set(key, CachedResponse(content="hi"), exact)
        await cache.get(key)  # exact hit
        await cache.get(compute_cache_key(near), near)  # approximate hit
        stats = await cache.stats
        assert stats["exact_hits"] == 1
        assert stats["approximate_hits"] == 1
        assert stats["hits"] == 2
        assert stats["content_class_hits"].get("plain_text") == 2

    async def test_json_content_class_detected(self) -> None:
        req = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": '{"key": "value"}'}],
        )
        assert _detect_content_class(req) == ContentClass.JSON

    async def test_code_content_class_detected(self) -> None:
        req = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "```python\nprint('hi')\n```"}],
        )
        assert _detect_content_class(req) == ContentClass.CODE

    async def test_mixed_content_class_detected(self) -> None:
        req = DummyRequest(
            model="gpt-4",
            messages=[
                {"role": "user", "content": '{"key": "value"}', "tool_call_id": "tc_1"},
            ],
        )
        assert _detect_content_class(req) == ContentClass.MIXED

    async def test_backend_abstraction_works(self) -> None:
        backend = InMemoryCacheBackend(max_entries=10)
        cache = SemanticCache(backend=backend, ttl_seconds=60, enabled=True)
        resp = CachedResponse(content="hello")
        await cache.set("k", resp)
        hit = await cache.get("k")
        assert hit is not None
        assert hit.content == "hello"


class TestApproximateIndex:
    """Verify the fingerprint index narrows candidates correctly."""

    async def test_index_buckets_isolate_models(self) -> None:
        cache = SemanticCache(ttl_seconds=300, enabled=True)
        req1 = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hello world"}],
        )
        req2 = DummyRequest(
            model="gpt-3.5",
            messages=[{"role": "user", "content": "hello world"}],
        )
        resp = CachedResponse(content="hi")
        await cache.set(compute_cache_key(req1), resp, req1)
        # Same content, different model → approximate miss
        hit = await cache.get(compute_cache_key(req2), req2)
        assert hit is None
        stats = await cache.stats
        assert stats["approximate_misses"] == 1

    async def test_index_buckets_isolate_content_classes(self) -> None:
        cache = SemanticCache(ttl_seconds=300, enabled=True)
        req1 = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "```python\nprint('hello world')\n```"}],
        )
        req2 = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "```python\nprint('hello world!')\n```"}],
        )
        resp = CachedResponse(content="ok")
        await cache.set(compute_cache_key(req1), resp, req1)
        # Both are CODE class and very similar, so approximate hit is possible
        hit = await cache.get(compute_cache_key(req2), req2)
        assert hit is not None
        assert hit.content == "ok"

    async def test_invalidate_removes_from_index(self) -> None:
        cache = SemanticCache(ttl_seconds=300, enabled=True)
        req = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hello world"}],
        )
        key = compute_cache_key(req)
        await cache.set(key, CachedResponse(content="hi"), req)
        await cache.invalidate(key)
        hit = await cache.get(key, req)
        assert hit is None

    async def test_clear_removes_all_from_index(self) -> None:
        cache = SemanticCache(ttl_seconds=300, enabled=True)
        req1 = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hello world"}],
        )
        req2 = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "foo bar"}],
        )
        await cache.set(compute_cache_key(req1), CachedResponse(content="a"), req1)
        await cache.set(compute_cache_key(req2), CachedResponse(content="b"), req2)
        await cache.clear()
        assert await cache.get(compute_cache_key(req1), req1) is None
        assert await cache.get(compute_cache_key(req2), req2) is None
        assert cache.entry_count == 0

    async def test_expire_stale_cleans_index(self) -> None:
        cache = SemanticCache(ttl_seconds=0, enabled=True)
        req = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hello world"}],
        )
        key = compute_cache_key(req)
        await cache.set(key, CachedResponse(content="hi"), req)
        # Wait for TTL to expire
        await asyncio.sleep(0.1)
        await cache.expire_stale()
        hit = await cache.get(key, req)
        assert hit is None


class TestBackendClearSemantics:
    """Phase 4: Backend clear() returns count, aligns with cache-level semantics."""

    async def test_inmemory_backend_clear_returns_count(self) -> None:
        backend = InMemoryCacheBackend(max_entries=10)
        await backend.set("k1", CachedResponse(content="a"), ttl=60)
        await backend.set("k2", CachedResponse(content="b"), ttl=60)
        removed = await backend.clear()
        assert removed == 2
        assert await backend.get("k1") is None

    async def test_inmemory_backend_clear_empty_returns_zero(self) -> None:
        backend = InMemoryCacheBackend(max_entries=10)
        removed = await backend.clear()
        assert removed == 0

    async def test_cache_clear_returns_backend_count(self) -> None:
        cache = SemanticCache(ttl_seconds=60, enabled=True)
        req = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hello"}],
        )
        key = compute_cache_key(req)
        await cache.set(key, CachedResponse(content="hi"), req)
        removed = await cache.clear()
        assert removed == 1
        assert cache.entry_count == 0

    async def test_maintenance_stats_accurate_after_expire(self) -> None:
        cache = SemanticCache(ttl_seconds=0, enabled=True)
        req = DummyRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hello world"}],
        )
        key = compute_cache_key(req)
        await cache.set(key, CachedResponse(content="hi"), req)
        await asyncio.sleep(0.1)
        removed = await cache.expire_stale()
        assert removed == 1

        stats = await cache.stats
        assert stats["maintenance_runs"] == 1
        assert stats["stale_removed"] == 1

    async def test_multiple_expire_runs_accumulate_stats(self) -> None:
        cache = SemanticCache(ttl_seconds=0, enabled=True)
        for i in range(3):
            req = DummyRequest(
                model="gpt-4",
                messages=[{"role": "user", "content": f"msg {i}"}],
            )
            await cache.set(compute_cache_key(req), CachedResponse(content=f"r{i}"), req)

        await asyncio.sleep(0.1)
        await cache.expire_stale()

        stats = await cache.stats
        assert stats["stale_removed"] == 3
        assert stats["maintenance_runs"] == 1