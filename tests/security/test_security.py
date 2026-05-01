"""Security tests for LATTICE.

Validates:
- Session fixation rejection
- Origin validation for local daemon
- Manifest tamper detection
- Replay attack on resume token
"""

import time

import pytest

from lattice.core.session import MemorySessionStore, Session, SessionManager
from lattice.protocol.manifest import Manifest, build_manifest
from lattice.protocol.resume import ResumeTokenManager, StreamManager
from lattice.protocol.segments import build_system_segment

# =============================================================================
# Session fixation
# =============================================================================


class TestSessionFixation:
    @pytest.mark.asyncio
    async def test_reject_predictable_session_id(self):
        """Sessions must use cryptographically strong IDs."""
        store = MemorySessionStore()
        mgr = SessionManager(store)

        # Normal creation uses strong ID
        session = await mgr.create_session(
            provider="openai",
            model="gpt-4",
            messages=[],
        )
        assert session.session_id.startswith("lattice-")
        assert len(session.session_id) > 20

    @pytest.mark.asyncio
    async def test_session_id_not_guessable(self):
        """Session IDs should not be sequential or predictable."""
        store = MemorySessionStore()
        mgr = SessionManager(store)

        ids = [(await mgr.create_session("openai", "gpt-4", [])).session_id for _ in range(5)]
        # All unique
        assert len(set(ids)) == 5
        # No common prefix beyond lattice-
        prefixes = {sid[:20] for sid in ids}
        assert len(prefixes) == 5

    @pytest.mark.asyncio
    async def test_version_increments_prevent_replay(self):
        """CAS versioning prevents replay of old session state."""
        store = MemorySessionStore()
        mgr = SessionManager(store)

        session = await mgr.create_session("openai", "gpt-4", [])
        _ = session.version

        # Simulate concurrent update from another instance
        # Use from_dict to create an independent copy with stale version
        stale = Session.from_dict(session.to_dict())

        # Bump version on original and store
        session.bump_version()
        await store.set(session)

        # Stale session should fail CAS
        success = await store.set(stale)
        assert success is False  # CAS failure


# =============================================================================
# Origin validation
# =============================================================================


class TestOriginValidation:
    def test_local_origin_detection(self):
        """Local origin detection works for 127.0.0.1 and localhost."""
        from lattice.proxy.server import _is_local_origin

        class FakeRequest:
            def __init__(self, host, client_host, forwarded=""):
                self.headers = {"host": host, "x-forwarded-for": forwarded}
                self.client = type("Client", (), {"host": client_host})()

        assert _is_local_origin(FakeRequest("127.0.0.1:8787", "127.0.0.1")) is True
        assert _is_local_origin(FakeRequest("localhost:8787", "127.0.0.1")) is True
        assert _is_local_origin(FakeRequest("localhost:8787", "::1")) is True
        assert _is_local_origin(FakeRequest("example.com", "1.2.3.4")) is False
        assert _is_local_origin(FakeRequest("localhost:8787", "1.2.3.4", "127.0.0.1")) is True


# =============================================================================
# Manifest tamper detection
# =============================================================================


class TestManifestTamper:
    def test_anchor_hash_changes_on_modification(self):
        """Changing any segment changes the anchor hash."""
        seg1 = build_system_segment("You are helpful.")
        m1 = build_manifest("sess-1", [seg1])
        hash1 = m1.anchor_hash

        seg2 = build_system_segment("You are unhelpful.")
        m2 = build_manifest("sess-1", [seg2])
        hash2 = m2.anchor_hash

        assert hash1 != hash2

    def test_anchor_hash_stable_for_identical(self):
        """Identical manifests produce identical hashes."""
        seg = build_system_segment("You are helpful.")
        m1 = build_manifest("sess-1", [seg])
        m2 = build_manifest("sess-1", [seg])
        assert m1.anchor_hash == m2.anchor_hash

    def test_manifest_roundtrip_preserves_hash(self):
        """Serialization roundtrip preserves anchor hash."""
        seg = build_system_segment("You are helpful.")
        m1 = build_manifest("sess-1", [seg], metadata={"model": "gpt-4"})
        data = m1.to_dict()
        m2 = Manifest.from_dict(data)
        assert m2.anchor_hash == m1.anchor_hash


# =============================================================================
# Resume token security
# =============================================================================


class TestResumeTokenSecurity:
    def test_token_expiration(self):
        """Expired tokens are rejected."""
        mgr = ResumeTokenManager(ttl_seconds=0)
        token = mgr.create_token("stream-1", 0)
        time.sleep(0.01)
        assert mgr.validate_token(token) is None

    def test_tampered_token_rejected(self):
        """Tampered tokens fail signature verification."""
        mgr = ResumeTokenManager(secret="super-secret")
        token = mgr.create_token("stream-1", 0)
        # Corrupt token
        corrupted = token[:-5] + "XXXXX"
        assert mgr.validate_token(corrupted) is None

    def test_wrong_secret_rejected(self):
        """Tokens from different secrets are rejected."""
        mgr1 = ResumeTokenManager(secret="secret-a")
        mgr2 = ResumeTokenManager(secret="secret-b")
        token = mgr1.create_token("stream-1", 0)
        assert mgr2.validate_token(token) is None

    def test_replay_attack_mitigation(self):
        """Replaying the same token is allowed (idempotent), but sequence must advance."""
        mgr = ResumeTokenManager(secret="secret")
        token = mgr.create_token("stream-1", 5)
        result = mgr.validate_token(token)
        assert result == ("stream-1", 5)
        # Replaying same token is fine
        result2 = mgr.validate_token(token)
        assert result2 == ("stream-1", 5)

    def test_stream_manager_token_flow(self):
        """Full token create → validate → replay flow."""
        sm = StreamManager()
        sid = sm.create_stream()
        sm.append_chunk(sid, 0, "chunk0")
        sm.append_chunk(sid, 1, "chunk1")

        token = sm.create_resume_token(sid, 1)
        validated = sm.validate_resume_token(token)
        assert validated == (sid, 1)

        replayed = sm.replay(sid, 1)
        assert len(replayed) == 1
        assert replayed[0].sequence == 1

    def test_replay_window_limits(self):
        """Old sequences fall out of window."""
        sm = StreamManager(window_capacity=5)
        sid = sm.create_stream()
        for i in range(10):
            sm.append_chunk(sid, i, f"chunk{i}")

        # Sequence 0 is too old
        replayed = sm.replay(sid, 0)
        assert len(replayed) == 0

        # Sequence 5 should be available
        replayed = sm.replay(sid, 5)
        assert len(replayed) == 5
