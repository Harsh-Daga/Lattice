"""Unit tests for RedisSessionStore using mocked redis client."""

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

# Patch redis BEFORE importing our store
pytest.importorskip("redis", reason="redis not installed")

from lattice.core.session import Session
from lattice.core.store import RedisSessionStore


@pytest.fixture
def mock_redis_client(monkeypatch):
    """Provide a mock redis client for tests."""
    client = AsyncMock()
    monkeypatch.setattr("lattice.core.store.redis", MagicMock())
    monkeypatch.setattr("lattice.core.store._REDIS_AVAILABLE", True)
    return client


@pytest.fixture
def store(mock_redis_client) -> RedisSessionStore:
    s = RedisSessionStore(url="redis://localhost:6379/0", ttl_seconds=3600)
    s._client = mock_redis_client
    return s


@pytest.fixture
def sample_session() -> Session:
    return Session(
        session_id="sess-test-001",
        created_at=time.time(),
        last_accessed_at=time.time(),
        provider="openai",
        model="gpt-4",
        messages=[],
    )


class TestRedisSessionStore:
    async def test_get_existing_session(self, store, mock_redis_client, sample_session):
        mock_redis_client.get = AsyncMock(return_value=json.dumps(sample_session.to_dict()))
        result = await store.get("sess-test-001")
        assert result is not None
        assert result.session_id == "sess-test-001"
        assert result.provider == "openai"

    async def test_get_missing_session(self, store, mock_redis_client):
        mock_redis_client.get = AsyncMock(return_value=None)
        result = await store.get("sess-nonexistent")
        assert result is None

    async def test_get_expired_session(self, store, mock_redis_client):
        old_session = Session(
            session_id="sess-old",
            created_at=0.0,
            last_accessed_at=0.0,
            provider="openai",
            model="gpt-4",
            messages=[],
        )
        # TTL 3600, last_accessed_at=0 means expired
        mock_redis_client.get = AsyncMock(return_value=json.dumps(old_session.to_dict()))
        mock_redis_client.delete = AsyncMock(return_value=1)
        result = await store.get("sess-old")
        assert result is None
        mock_redis_client.delete.assert_awaited_once()

    async def test_set_session(self, store, mock_redis_client, sample_session):
        mock_redis_client.get = AsyncMock(return_value=None)  # no existing session
        mock_redis_client.setex = AsyncMock(return_value=True)
        await store.set(sample_session)
        mock_redis_client.setex.assert_awaited_once()
        key, ttl, raw = mock_redis_client.setex.call_args[0]
        assert ttl == 3600
        data = json.loads(raw)
        assert data["session_id"] == "sess-test-001"

    async def test_delete_session(self, store, mock_redis_client):
        mock_redis_client.delete = AsyncMock(return_value=1)
        result = await store.delete("sess-test-001")
        assert result is True
        mock_redis_client.delete.assert_awaited_once()

    async def test_delete_missing_session(self, store, mock_redis_client):
        mock_redis_client.delete = AsyncMock(return_value=0)
        result = await store.delete("sess-nonexistent")
        assert result is False

    async def test_keys(self, store, mock_redis_client):
        mock_redis_client.scan = AsyncMock(side_effect=[
            (0, ["lattice:session:sess-1", "lattice:session:sess-2"]),
        ])
        result = await store.keys()
        assert result == ["sess-1", "sess-2"]

    async def test_expire(self, store, mock_redis_client):
        old = Session(
            session_id="sess-old",
            created_at=0.0,
            last_accessed_at=0.0,
            provider="openai",
            model="gpt-4",
            messages=[],
        )
        # First scan returns key, second returns nothing
        mock_redis_client.scan = AsyncMock(side_effect=[
            (0, ["lattice:session:sess-old"]),
        ])
        mock_redis_client.get = AsyncMock(return_value=json.dumps(old.to_dict()))
        mock_redis_client.delete = AsyncMock(return_value=1)
        count = await store.expire(3600)
        assert count == 1


class TestRedisStoreLifecycle:
    async def test_start_and_stop(self, mock_redis_client):
        # Ensure start() initializes _client when None
        store = RedisSessionStore(url="redis://localhost:6379/0")
        assert store._client is None

        mock_redis_client.ping = AsyncMock(return_value=True)
        mock_redis_client.close = AsyncMock()

        # Patch redis.from_url to return our mock
        import lattice.core.store as store_mod
        orig_redis = store_mod.redis
        store_mod.redis = MagicMock()
        store_mod.redis.from_url = MagicMock(return_value=mock_redis_client)
        try:
            await store.start()
            mock_redis_client.ping.assert_awaited_once()
            await store.stop()
            mock_redis_client.close.assert_awaited_once()
        finally:
            store_mod.redis = orig_redis
