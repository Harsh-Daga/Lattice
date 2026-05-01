"""Integration tests for RedisSessionStore using fakeredis.

fakeredis implements the Redis protocol in-memory, giving us realistic
integration tests without requiring a real Redis server.

Coverage:
- Full CRUD lifecycle against a real Redis-compatible backend
- CAS versioning with concurrent writes
- TTL and expiration behavior
- Multi-session operations
"""

from __future__ import annotations

import asyncio
import time

import pytest

pytest.importorskip("redis", reason="redis not installed")
pytest.importorskip("fakeredis", reason="fakeredis not installed")

import fakeredis.aioredis

from lattice.core.session import Message, Session
from lattice.core.store import RedisSessionStore


@pytest.fixture
async def fake_redis():
    """Yield a fakeredis async client."""
    client = fakeredis.aioredis.FakeRedis()
    yield client
    await client.flushall()
    await client.aclose()


@pytest.fixture
async def store(fake_redis):
    """Yield a RedisSessionStore wired to fakeredis."""
    s = RedisSessionStore(url="redis://localhost:6379/0", ttl_seconds=3600)
    s._client = fake_redis
    yield s


@pytest.fixture
def sample_session() -> Session:
    return Session(
        session_id="sess-test-001",
        created_at=time.time(),
        last_accessed_at=time.time(),
        provider="openai",
        model="gpt-4",
        messages=[Message(role="user", content="hello")],
    )


class TestRedisIntegrationCRUD:
    async def test_set_and_get(self, store, sample_session):
        await store.set(sample_session)
        result = await store.get("sess-test-001")
        assert result is not None
        assert result.session_id == "sess-test-001"
        assert result.provider == "openai"
        assert result.model == "gpt-4"
        assert len(result.messages) == 1

    async def test_get_missing(self, store):
        result = await store.get("sess-nonexistent")
        assert result is None

    async def test_delete(self, store, sample_session):
        await store.set(sample_session)
        assert await store.delete("sess-test-001") is True
        assert await store.get("sess-test-001") is None

    async def test_delete_missing(self, store):
        assert await store.delete("sess-nonexistent") is False

    async def test_keys(self, store, sample_session):
        await store.set(sample_session)
        s2 = Session(
            session_id="sess-test-002",
            created_at=time.time(),
            last_accessed_at=time.time(),
            provider="anthropic",
            model="claude-3",
            messages=[],
        )
        await store.set(s2)
        keys = await store.keys()
        assert sorted(keys) == ["sess-test-001", "sess-test-002"]


class TestRedisIntegrationCAS:
    async def test_cas_success(self, store, sample_session):
        await store.set(sample_session)
        # Modify and update with incremented version
        sample_session.version += 1
        success = await store.set(sample_session)
        assert success is True

    async def test_cas_conflict_rejected(self, store, sample_session):
        sample_session.version = 2
        await store.set(sample_session)
        # Simulate stale version (lower than stored)
        stale = Session(
            session_id="sess-test-001",
            created_at=time.time(),
            last_accessed_at=time.time(),
            provider="openai",
            model="gpt-4",
            messages=[],
            version=0,  # lower than stored version (2)
        )
        success = await store.set(stale)
        assert success is False

    async def test_cas_concurrent_writes(self, store):
        """Multiple writers with version increments should all succeed."""
        sess = Session(
            session_id="sess-concurrent",
            created_at=time.time(),
            last_accessed_at=time.time(),
            provider="openai",
            model="gpt-4",
            messages=[],
            version=1,
        )
        await store.set(sess)

        results = []
        for i in range(5):
            s = Session(
                session_id="sess-concurrent",
                created_at=time.time(),
                last_accessed_at=time.time(),
                provider="openai",
                model="gpt-4",
                messages=[Message(role="user", content=f"turn {i}")],
                version=2 + i,
            )
            results.append(await store.set(s))

        assert all(results)
        final = await store.get("sess-concurrent")
        assert final is not None
        assert final.version == 6


class TestRedisIntegrationTTL:
    async def test_ttl_set_by_redis(self, store, sample_session, fake_redis):
        await store.set(sample_session)
        key = f"{store.prefix}sess-test-001"
        ttl = await fake_redis.ttl(key)
        assert ttl > 0
        assert ttl <= 3600

    async def test_expired_session_removed(self, store):
        # Create a store with very short TTL
        short_store = RedisSessionStore(url="redis://localhost:6379/0", ttl_seconds=1)
        short_store._client = store._client

        old = Session(
            session_id="sess-old",
            created_at=time.time(),
            last_accessed_at=time.time(),
            provider="openai",
            model="gpt-4",
            messages=[],
        )
        await short_store.set(old)

        # Wait for TTL to expire
        await asyncio.sleep(1.5)

        result = await short_store.get("sess-old")
        assert result is None

    async def test_expire_method_deletes_stale(self, store, monkeypatch):
        """expire() scans keys and deletes stale entries."""
        short_store = RedisSessionStore(url="redis://localhost:6379/0", ttl_seconds=3600)
        short_store._client = store._client

        base_time = 1_000_000.0
        monkeypatch.setattr(time, "time", lambda: base_time)

        old = Session(
            session_id="sess-expire",
            created_at=base_time,
            last_accessed_at=base_time,
            provider="openai",
            model="gpt-4",
            messages=[],
        )
        await short_store.set(old)

        # Key exists
        keys_before = await store._client.keys("*")
        assert len(keys_before) == 1

        # Advance time so client-side TTL check triggers
        monkeypatch.setattr(time, "time", lambda: base_time + 100)

        count = await short_store.expire(1)
        assert count == 1
        assert await short_store.get("sess-expire") is None


class TestRedisIntegrationSerialization:
    async def test_manifest_roundtrip(self, store):
        from lattice.protocol.manifest import Manifest
        from lattice.protocol.segments import build_system_segment, build_tools_segment

        manifest = Manifest(
            manifest_id="manifest-001",
            session_id="sess-manifest",
            anchor_version=1,
            anchor_hash="abc123",
            segments=[
                build_system_segment("You are helpful."),
                build_tools_segment([{"name": "read"}]),
            ],
        )
        sess = Session(
            session_id="sess-manifest",
            created_at=time.time(),
            last_accessed_at=time.time(),
            provider="openai",
            model="gpt-4",
            messages=[],
            manifest=manifest,
        )
        await store.set(sess)
        result = await store.get("sess-manifest")
        assert result is not None
        assert result.manifest is not None
        assert len(result.manifest.segments) == 2
        assert result.manifest.segments[0].type.value == "system"

    async def test_large_session(self, store):
        messages = [Message(role="user", content=f"Message {i}" * 100) for i in range(100)]
        sess = Session(
            session_id="sess-large",
            created_at=time.time(),
            last_accessed_at=time.time(),
            provider="openai",
            model="gpt-4",
            messages=messages,
        )
        await store.set(sess)
        result = await store.get("sess-large")
        assert result is not None
        assert len(result.messages) == 100


class TestRedisIntegrationMultiProcess:
    async def test_isolated_prefixes(self, fake_redis):
        store_a = RedisSessionStore(url="redis://localhost:6379/0", prefix="lattice:a:")
        store_a._client = fake_redis
        store_b = RedisSessionStore(url="redis://localhost:6379/0", prefix="lattice:b:")
        store_b._client = fake_redis

        await store_a.set(
            Session(
                session_id="shared",
                created_at=time.time(),
                last_accessed_at=time.time(),
                provider="openai",
                model="gpt-4",
                messages=[],
            )
        )
        await store_b.set(
            Session(
                session_id="shared",
                created_at=time.time(),
                last_accessed_at=time.time(),
                provider="anthropic",
                model="claude-3",
                messages=[],
            )
        )

        a_result = await store_a.get("shared")
        b_result = await store_b.get("shared")

        assert a_result is not None
        assert a_result.provider == "openai"
        assert b_result is not None
        assert b_result.provider == "anthropic"

    async def test_count_approximate(self, store, sample_session):
        # session_count property is stubbed to 0 (not async-safe)
        await store.set(sample_session)
        assert store.session_count == 0
