"""Unit tests for DeltaEncoder.

Tests cover:
- Delta classification (append, truncation, full_replacement, error)
- Session reconstruction (sync path via MemorySessionStore)
- Message equality (ignores metadata)
- Edge cases (empty messages, identical sets, sliding window)
"""

from __future__ import annotations

import pytest

from lattice.core.context import TransformContext
from lattice.core.result import unwrap
from lattice.core.session import MemorySessionStore, SessionManager
from lattice.core.transport import Message, Request
from lattice.transforms.delta_encode import DeltaEncoder, DeltaType

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
async def encoder() -> DeltaEncoder:
    store = MemorySessionStore()
    await store.start()
    sm = SessionManager(store)
    enc = DeltaEncoder(sm)
    yield enc
    await store.stop()


# =============================================================================
# Delta Classification
# =============================================================================

class TestDeltaClassification:
    def test_append(self) -> None:
        existing = [Message(role="user", content="a")]
        new = [Message(role="user", content="a"), Message(role="assistant", content="b")]
        dt = DeltaEncoder._classify_delta(existing, new)
        assert dt == DeltaType.APPEND

    def test_truncation(self) -> None:
        existing = [
            Message(role="user", content="a"),
            Message(role="assistant", content="b"),
        ]
        new = [Message(role="user", content="a")]
        dt = DeltaEncoder._classify_delta(existing, new)
        assert dt == DeltaType.TRUNCATION

    def test_identical(self) -> None:
        msgs = [Message(role="user", content="a")]
        dt = DeltaEncoder._classify_delta(msgs, msgs)
        assert dt == DeltaType.APPEND  # identical counts as append (no new data needed)

    def test_sliding_window(self) -> None:
        existing = [
            Message(role="user", content="A"),
            Message(role="assistant", content="B"),
            Message(role="user", content="C"),
            Message(role="assistant", content="D"),
            Message(role="user", content="E"),
        ]
        new = [
            Message(role="user", content="A"),  # common prefix
            Message(role="user", content="C"),   # skipped B
            Message(role="assistant", content="D"),
            Message(role="user", content="E"),
        ]
        dt = DeltaEncoder._classify_delta(existing, new)
        assert dt == DeltaType.TRUNCATION

    def test_full_replacement(self) -> None:
        existing = [Message(role="user", content="a")]
        new = [Message(role="user", content="b")]
        dt = DeltaEncoder._classify_delta(existing, new)
        assert dt == DeltaType.FULL_REPLACEMENT

    def test_empty_new(self) -> None:
        existing = [Message(role="user", content="a")]
        new: list[Message] = []
        dt = DeltaEncoder._classify_delta(existing, new)
        assert dt == DeltaType.ERROR

    def test_apply_append(self) -> None:
        existing = [Message(role="user", content="a")]
        new = [Message(role="user", content="a"), Message(role="assistant", content="b")]
        full = DeltaEncoder._apply_append(existing, new)
        assert len(full) == 2
        assert full[0].content == "a"
        assert full[1].content == "b"


# =============================================================================
# Messages Equal
# =============================================================================

class TestMessagesEqual:
    def test_same(self) -> None:
        a = Message(role="user", content="hello")
        b = Message(role="user", content="hello")
        assert DeltaEncoder._messages_equal(a, b) is True

    def test_different_content(self) -> None:
        a = Message(role="user", content="hello")
        b = Message(role="user", content="world")
        assert DeltaEncoder._messages_equal(a, b) is False

    def test_different_role(self) -> None:
        a = Message(role="user", content="hello")
        b = Message(role="assistant", content="hello")
        assert DeltaEncoder._messages_equal(a, b) is False

    def test_ignores_metadata(self) -> None:
        a = Message(role="user", content="hello", metadata={"x": 1})
        b = Message(role="user", content="hello", metadata={"x": 2})
        assert DeltaEncoder._messages_equal(a, b) is True


# =============================================================================
# Session Reconstruction (via encoder.process)
# =============================================================================

class TestSessionReconstruction:
    @pytest.mark.asyncio
    async def test_no_session_id_untouched(self, encoder: DeltaEncoder) -> None:
        req = Request(messages=[Message(role="user", content="hello")])
        ctx = TransformContext(session_id=None)
        result = await encoder.process(req, ctx)
        assert unwrap(result) == req
        assert ctx.metrics["transforms"]["delta_encoder"]["delta_type"] == "new_session"

    @pytest.mark.asyncio
    async def test_append_reconstructs_full(self, encoder: DeltaEncoder) -> None:
        # Turn 1: create session manually
        session = await encoder.session_manager.create_session(
            provider="openai",
            model="gpt-4",
            messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="q1"),
                Message(role="assistant", content="a1"),
            ],
        )
        # Override session_id for test consistency
        object.__setattr__(session, "session_id", "sess_1")
        await encoder.session_manager.store.set(session)
        # Turn 2: only send new messages (delta)
        new_req = Request(
            messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="q1"),
                Message(role="assistant", content="a1"),
                Message(role="user", content="q2"),
            ]
        )
        ctx = TransformContext(session_id="sess_1")
        result = await encoder.process(new_req, ctx)
        modified = unwrap(result)
        assert len(modified.messages) == 4
        assert modified.messages[-1].content == "q2"
        assert ctx.metrics["transforms"]["delta_encoder"]["delta_type"] == "append"

    @pytest.mark.asyncio
    async def test_truncation_reconstructs(self, encoder: DeltaEncoder) -> None:
        session = await encoder.session_manager.create_session(
            provider="openai",
            model="gpt-4",
            messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="q1"),
                Message(role="assistant", content="a1"),
                Message(role="user", content="q2"),
            ],
        )
        object.__setattr__(session, "session_id", "sess_2")
        await encoder.session_manager.store.set(session)
        # Client pruned old messages
        new_req = Request(
            messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="q2"),
            ]
        )
        ctx = TransformContext(session_id="sess_2")
        result = await encoder.process(new_req, ctx)
        modified = unwrap(result)
        assert len(modified.messages) == 2
        assert ctx.metrics["transforms"]["delta_encoder"]["delta_type"] == "truncation"

    @pytest.mark.asyncio
    async def test_session_not_found_fallback(self, encoder: DeltaEncoder) -> None:
        new_req = Request(messages=[Message(role="user", content="hello")])
        ctx = TransformContext(session_id="nonexistent")
        result = await encoder.process(new_req, ctx)
        modified = unwrap(result)
        assert modified == new_req  # unchanged
        assert ctx.metrics["transforms"]["delta_encoder"]["delta_type"] == "session_not_found_fallback"

    @pytest.mark.asyncio
    async def test_metadata_preserved(self, encoder: DeltaEncoder) -> None:
        session = await encoder.session_manager.create_session(
            provider="openai",
            model="gpt-4",
            messages=[Message(role="user", content="a")],
        )
        object.__setattr__(session, "session_id", "sess_3")
        await encoder.session_manager.store.set(session)
        new_req = Request(
            messages=[
                Message(role="user", content="a"),
                Message(role="assistant", content="b"),
            ]
        )
        ctx = TransformContext(session_id="sess_3")
        result = await encoder.process(new_req, ctx)
        modified = unwrap(result)
        assert modified.metadata.get("_delta_type") == "append"
        assert modified.metadata.get("_delta_messages_count") == 2
        assert modified.metadata.get("_full_messages_count") == 2
