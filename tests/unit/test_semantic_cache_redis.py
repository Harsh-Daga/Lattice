"""Tests for RedisCacheBackend and backend configuration."""

from __future__ import annotations

import pytest

from lattice.core.config import LatticeConfig
from lattice.core.semantic_cache import (
    CachedResponse,
    ContentClass,
    RedisCacheBackend,
    SemanticCache,
    compute_cache_key,
)
from lattice.core.transport import Message, Request


class TestRedisCacheBackend:
    """Unit tests for RedisCacheBackend (requires redis, skipped if unavailable)."""

    @pytest.fixture
    async def backend(self):
        try:
            be = RedisCacheBackend(url="redis://localhost:6379/15")
        except ImportError:
            pytest.skip("redis not installed")
        try:
            await be.start()
        except Exception as exc:
            pytest.skip(f"redis not reachable: {exc}")
        yield be
        await be.stop()

    @pytest.mark.asyncio
    async def test_redis_backend_roundtrip(self, backend: RedisCacheBackend) -> None:
        resp = CachedResponse(
            content="hello",
            model="gpt-4",
            metadata={"_lattice_content_class": ContentClass.PLAIN_TEXT.value},
        )
        assert await backend.set("key1", resp, ttl=60)
        got = await backend.get("key1")
        assert got is not None
        assert got.content == "hello"
        assert got.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_redis_backend_delete(self, backend: RedisCacheBackend) -> None:
        resp = CachedResponse(content="bye", model="gpt-4")
        await backend.set("key2", resp, ttl=60)
        assert await backend.delete("key2")
        assert await backend.get("key2") is None

    @pytest.mark.asyncio
    async def test_redis_backend_ttl_expiry(self, backend: RedisCacheBackend) -> None:
        resp = CachedResponse(content="short", model="gpt-4")
        await backend.set("key3", resp, ttl=1)
        assert await backend.get("key3") is not None
        import asyncio

        await asyncio.sleep(1.5)
        assert await backend.get("key3") is None

    @pytest.mark.asyncio
    async def test_redis_backend_keys(self, backend: RedisCacheBackend) -> None:
        resp = CachedResponse(content="k4", model="gpt-4")
        await backend.set("key4", resp, ttl=60)
        keys = await backend.keys()
        assert "key4" in keys


class TestBackendConfiguration:
    """Tests for backend selection via LatticeConfig."""

    def test_default_backend_is_memory(self) -> None:
        config = LatticeConfig()
        assert config.semantic_cache_backend == "memory"
        assert config.semantic_cache_backend_url is None

    def test_redis_backend_requires_url(self) -> None:
        with pytest.raises(ValueError):
            LatticeConfig(semantic_cache_backend="redis")

    def test_redis_backend_accepts_explicit_url(self) -> None:
        config = LatticeConfig(
            semantic_cache_backend="redis",
            semantic_cache_backend_url="redis://localhost:6379/0",
        )
        assert config.semantic_cache_backend == "redis"
        assert config.semantic_cache_backend_url == "redis://localhost:6379/0"

    def test_redis_backend_falls_back_to_redis_url(self) -> None:
        config = LatticeConfig(
            semantic_cache_backend="redis",
            redis_url="redis://localhost:6379/0",
        )
        assert config.semantic_cache_backend == "redis"
        assert config.redis_url == "redis://localhost:6379/0"


class TestSemanticCacheWithRedisBackend:
    """Integration tests for SemanticCache using RedisCacheBackend."""

    @pytest.fixture
    async def redis_cache(self):
        try:
            backend = RedisCacheBackend(url="redis://localhost:6379/15")
        except ImportError:
            pytest.skip("redis not installed")
        try:
            await backend.start()
        except Exception as exc:
            pytest.skip(f"redis not reachable: {exc}")
        cache = SemanticCache(backend=backend, ttl_seconds=60)
        yield cache
        await backend.stop()

    @pytest.mark.asyncio
    async def test_exact_hit_with_redis(self, redis_cache: SemanticCache) -> None:
        req = Request(messages=[Message(role="user", content="hello")], model="gpt-4")
        key = compute_cache_key(req)
        resp = CachedResponse(content="world", model="gpt-4")
        assert await redis_cache.set(key, resp, request=req)
        hit = await redis_cache.get(key)
        assert hit is not None
        assert hit.content == "world"

    @pytest.mark.asyncio
    async def test_approximate_hit_with_redis(self, redis_cache: SemanticCache) -> None:
        req1 = Request(messages=[Message(role="user", content="hello world")], model="gpt-4")
        key1 = compute_cache_key(req1)
        resp = CachedResponse(content="result", model="gpt-4")
        assert await redis_cache.set(key1, resp, request=req1)

        req2 = Request(messages=[Message(role="user", content="hello world!")], model="gpt-4")
        key2 = compute_cache_key(req2)
        hit = await redis_cache.get(key2, request=req2)
        assert hit is not None
        assert hit.content == "result"

    @pytest.mark.asyncio
    async def test_stats_reflect_backend(self, redis_cache: SemanticCache) -> None:
        req = Request(messages=[Message(role="user", content="stats test")], model="gpt-4")
        key = compute_cache_key(req)
        await redis_cache.set(key, CachedResponse(content="ok", model="gpt-4"), request=req)
        await redis_cache.get(key)
        stats = await redis_cache.stats
        assert stats["exact_hits"] >= 1
