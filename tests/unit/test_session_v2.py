"""Unit tests for LATTICE session management — manifest support and CAS.

Tests cover:
- Session creation with manifest generation
- Session update with optimistic concurrency
- Session delta computation
- MemorySessionStore CAS behavior
- RedisSessionStore CAS behavior (mocked)
"""

from __future__ import annotations

import pytest

from lattice.core.session import MemorySessionStore, Session, SessionManager
from lattice.core.transport import Message
from lattice.protocol.content import TextPart

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
async def store() -> MemorySessionStore:
    s = MemorySessionStore(ttl_seconds=3600)
    await s.start()
    yield s
    await s.stop()


@pytest.fixture
async def session_manager(store: MemorySessionStore) -> SessionManager:
    return SessionManager(store, ttl_seconds=3600)


# =============================================================================
# Session creation
# =============================================================================

class TestCreateSession:
    @pytest.mark.asyncio
    async def test_creates_session_with_manifest(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(
            provider="openai",
            model="gpt-4",
            messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="hi"),
            ],
        )
        assert session.session_id.startswith("lattice-")
        assert session.manifest is not None
        assert session.manifest.metadata["model"] == "gpt-4"
        assert session.manifest.metadata["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_creates_session_with_tools(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(
            provider="openai",
            model="gpt-4",
            messages=[Message(role="user", content="hi")],
            tools=[{"name": "search"}],
        )
        assert session.tool_schemas == [{"name": "search"}]
        assert session.manifest is not None
        # Manifest should have a tools segment
        from lattice.protocol.segments import SegmentType
        tools_seg = session.manifest.get_segment(SegmentType.TOOLS)
        assert tools_seg is not None

    @pytest.mark.asyncio
    async def test_system_prompt_extracted(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(
            provider="openai",
            model="gpt-4",
            messages=[
                Message(role="system", content="line1"),
                Message(role="system", content="line2"),
                Message(role="user", content="hi"),
            ],
        )
        assert session.system_prompt == "line1\nline2"


# =============================================================================
# Session get_or_create
# =============================================================================

class TestGetOrCreateSession:
    @pytest.mark.asyncio
    async def test_creates_new_when_no_session_id(self, session_manager: SessionManager) -> None:
        session, was_created = await session_manager.get_or_create_session(
            session_id=None,
            provider="openai",
            model="gpt-4",
            messages=[Message(role="user", content="hi")],
        )
        assert was_created is True
        assert session.message_count == 1

    @pytest.mark.asyncio
    async def test_returns_existing_when_found(self, session_manager: SessionManager) -> None:
        created = await session_manager.create_session(
            provider="openai",
            model="gpt-4",
            messages=[Message(role="user", content="hi")],
        )
        session, was_created = await session_manager.get_or_create_session(
            session_id=created.session_id,
            provider="openai",
            model="gpt-4",
            messages=[Message(role="user", content="hi")],
        )
        assert was_created is False
        assert session.session_id == created.session_id

    @pytest.mark.asyncio
    async def test_creates_new_when_not_found(self, session_manager: SessionManager) -> None:
        session, was_created = await session_manager.get_or_create_session(
            session_id="nonexistent",
            provider="openai",
            model="gpt-4",
            messages=[Message(role="user", content="hi")],
        )
        assert was_created is True


# =============================================================================
# Session update
# =============================================================================

class TestUpdateSession:
    @pytest.mark.asyncio
    async def test_updates_messages(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(
            provider="openai",
            model="gpt-4",
            messages=[Message(role="user", content="hi")],
        )
        updated = await session_manager.update_session(
            session.session_id,
            [Message(role="user", content="hi"), Message(role="assistant", content="hello")],
        )
        assert updated is not None
        assert updated.message_count == 2
        assert updated.version == 1  # bumped

    @pytest.mark.asyncio
    async def test_updates_manifest(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(
            provider="openai",
            model="gpt-4",
            messages=[Message(role="user", content="hi")],
        )
        from lattice.protocol.manifest import build_manifest
        from lattice.protocol.segments import build_messages_segment

        new_manifest = build_manifest(
            session.session_id,
            [build_messages_segment([TextPart(text="updated")])],
            anchor_version=1,
        )
        updated = await session_manager.update_session(
            session.session_id, session.messages, manifest=new_manifest
        )
        assert updated is not None
        assert updated.manifest is not None
        assert updated.manifest.anchor_version == 1

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, session_manager: SessionManager) -> None:
        result = await session_manager.update_session(
            "nonexistent", [Message(role="user", content="hi")]
        )
        assert result is None


# =============================================================================
# CAS / optimistic concurrency
# =============================================================================

class TestOptimisticConcurrency:
    @pytest.mark.asyncio
    async def test_set_fails_on_version_conflict(self, store: MemorySessionStore) -> None:
        session = Session(
            session_id="sess_1",
            created_at=0.0,
            last_accessed_at=0.0,
            messages=[],
            version=1,
        )
        await store.set(session)

        # Simulate stale version (lower than stored)
        stale = Session(
            session_id="sess_1",
            created_at=0.0,
            last_accessed_at=0.0,
            messages=[],
            version=0,
        )
        success = await store.set(stale)
        assert success is False

    @pytest.mark.asyncio
    async def test_set_succeeds_on_newer_version(self, store: MemorySessionStore) -> None:
        session = Session(
            session_id="sess_2",
            created_at=0.0,
            last_accessed_at=0.0,
            messages=[],
            version=5,
        )
        await store.set(session)

        newer = Session(
            session_id="sess_2",
            created_at=0.0,
            last_accessed_at=0.0,
            messages=[],
            version=6,
        )
        success = await store.set(newer)
        assert success is True


# =============================================================================
# Session delta computation
# =============================================================================

class TestComputeDelta:
    @pytest.mark.asyncio
    async def test_all_new_when_no_session(self, session_manager: SessionManager) -> None:
        msgs = [Message(role="user", content="hi")]
        delta = await session_manager.compute_delta("nonexistent", msgs)
        assert delta == msgs

    @pytest.mark.asyncio
    async def test_append_only(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(
            provider="openai",
            model="gpt-4",
            messages=[Message(role="user", content="q1"), Message(role="assistant", content="a1")],
        )
        new_msgs = [
            Message(role="user", content="q1"),
            Message(role="assistant", content="a1"),
            Message(role="user", content="q2"),
        ]
        delta = await session_manager.compute_delta(session.session_id, new_msgs)
        assert len(delta) == 1
        assert delta[0].content == "q2"

    @pytest.mark.asyncio
    async def test_empty_when_truncation(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(
            provider="openai",
            model="gpt-4",
            messages=[
                Message(role="user", content="q1"),
                Message(role="assistant", content="a1"),
            ],
        )
        new_msgs = [Message(role="user", content="q1")]
        delta = await session_manager.compute_delta(session.session_id, new_msgs)
        assert delta == []


# =============================================================================
# Session expiry
# =============================================================================

class TestSessionExpiry:
    @pytest.mark.asyncio
    async def test_expired_session_removed(self, store: MemorySessionStore) -> None:
        import time
        session = Session(
            session_id="sess_exp",
            created_at=0.0,
            last_accessed_at=time.time() - 100,  # very old
            messages=[],
        )
        await store.set(session)
        # Expire with 0 TTL should remove all sessions
        count = await store.expire(0)
        assert count >= 1
        result = await store.get("sess_exp")
        assert result is None

    @pytest.mark.asyncio
    async def test_expire_method(self, store: MemorySessionStore) -> None:
        session = Session(
            session_id="sess_exp2",
            created_at=0.0,
            last_accessed_at=0.0,
            messages=[],
        )
        await store.set(session)
        count = await store.expire(0)
        assert count >= 1


# =============================================================================
# Session serialization
# =============================================================================

class TestSessionSerialization:
    def test_roundtrip_with_manifest(self) -> None:
        from lattice.protocol.manifest import build_manifest
        from lattice.protocol.segments import build_system_segment

        manifest = build_manifest("sess_1", [build_system_segment("sys")])
        session = Session(
            session_id="sess_1",
            created_at=1.0,
            last_accessed_at=2.0,
            messages=[Message(role="user", content="hi")],
            manifest=manifest,
            version=3,
        )
        d = session.to_dict()
        restored = Session.from_dict(d)
        assert restored.session_id == "sess_1"
        assert restored.version == 3
        assert restored.manifest is not None
        assert restored.manifest.anchor_hash == manifest.anchor_hash

    def test_roundtrip_without_manifest(self) -> None:
        session = Session(
            session_id="sess_1",
            created_at=1.0,
            last_accessed_at=2.0,
            messages=[Message(role="user", content="hi")],
        )
        d = session.to_dict()
        restored = Session.from_dict(d)
        assert restored.manifest is None


class TestSessionMigration:
    @pytest.mark.asyncio
    async def test_session_migration(self, session_manager: SessionManager) -> None:
        session = await session_manager.create_session(
            provider="openai",
            model="gpt-4",
            messages=[Message(role="user", content="hi")],
        )
        ok = await session_manager.migrate_session(session.session_id, "conn-2")
        assert ok is True
        updated = await session_manager.store.get(session.session_id)
        assert updated is not None
        assert updated.connection_id == "conn-2"
