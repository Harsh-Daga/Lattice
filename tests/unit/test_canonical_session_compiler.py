"""Phase 1: Canonical Session Compiler tests.

Verifies that the manifest compiler is the single source of truth for
request shape across proxy, SDK, and native paths.
"""

from __future__ import annotations

import pytest

from lattice.core.session import MemorySessionStore, SessionManager
from lattice.core.transport import Message
from lattice.protocol.content import TextPart
from lattice.protocol.manifest import (
    apply_delta,
    build_manifest,
    canonicalize_segments,
    compute_anchor_hash,
    manifest_from_messages,
    manifest_to_messages,
)
from lattice.protocol.segments import (
    SegmentType,
    build_messages_segment,
    build_segment,
    build_system_segment,
    build_tools_segment,
)

# =============================================================================
# Manifest hash stability
# =============================================================================


class TestManifestHashStability:
    def test_same_segments_same_hash(self) -> None:
        segments = [
            build_system_segment("sys"),
            build_tools_segment([{"name": "search"}]),
            build_messages_segment([TextPart(text="hi")]),
        ]
        m1 = build_manifest("sess_1", segments)
        m2 = build_manifest("sess_1", segments)
        assert m1.anchor_hash == m2.anchor_hash

    def test_different_order_different_hash(self) -> None:
        seg_a = build_system_segment("sys")
        seg_b = build_tools_segment([{"name": "search"}])
        m1 = build_manifest("sess_1", [seg_a, seg_b])
        m2 = build_manifest("sess_1", [seg_b, seg_a])
        assert m1.anchor_hash != m2.anchor_hash

    def test_metadata_included_in_hash(self) -> None:
        seg = build_system_segment("sys")
        m1 = build_manifest("sess_1", [seg], metadata={"model": "a"})
        m2 = build_manifest("sess_1", [seg], metadata={"model": "b"})
        assert m1.anchor_hash != m2.anchor_hash

    def test_compute_anchor_hash_deterministic(self) -> None:
        segments = [build_system_segment("sys")]
        h1 = compute_anchor_hash(segments, {"model": "gpt-4"})
        h2 = compute_anchor_hash(segments, {"model": "gpt-4"})
        assert h1 == h2
        assert len(h1) == 64


# =============================================================================
# Segment ordering determinism
# =============================================================================


class TestSegmentOrderingDeterminism:
    def test_canonical_order_tools_system_docs_artifacts_messages(self) -> None:
        segments = [
            build_messages_segment([TextPart(text="hi")]),
            build_system_segment("sys"),
            build_tools_segment([{"name": "search"}]),
            build_segment(SegmentType.DOCS, [TextPart(text="docs")]),
            build_segment(SegmentType.ARTIFACTS, [TextPart(text="artifact")]),
        ]
        ordered = canonicalize_segments(segments)
        types = [s.type for s in ordered]
        assert types == [
            SegmentType.TOOLS,
            SegmentType.SYSTEM,
            SegmentType.DOCS,
            SegmentType.ARTIFACTS,
            SegmentType.MESSAGES,
        ]

    def test_canonicalize_idempotent(self) -> None:
        segments = [
            build_messages_segment([TextPart(text="hi")]),
            build_system_segment("sys"),
        ]
        o1 = canonicalize_segments(segments)
        o2 = canonicalize_segments(o1)
        assert [s.type for s in o1] == [s.type for s in o2]
        assert [s.hash for s in o1] == [s.hash for s in o2]


# =============================================================================
# Session update auto-rebuilds manifest
# =============================================================================


class TestSessionUpdateRebuildsManifest:
    @pytest.mark.asyncio
    async def test_update_rebuilds_manifest(self) -> None:
        store = MemorySessionStore()
        await store.start()
        manager = SessionManager(store)

        session = await manager.create_session(
            provider="openai",
            model="gpt-4",
            messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="hi"),
            ],
        )
        original_hash = session.manifest.anchor_hash if session.manifest else ""

        updated = await manager.update_session(
            session.session_id,
            messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="hi"),
                Message(role="assistant", content="hello"),
            ],
        )
        assert updated is not None
        assert updated.manifest is not None
        assert updated.manifest.anchor_hash != original_hash
        assert updated.manifest.anchor_version > 0
        # Should have a messages segment with the assistant turn
        msg_seg = updated.manifest.get_segment(SegmentType.MESSAGES)
        assert msg_seg is not None
        assert any("assistant" in str(p).lower() for p in msg_seg.parts)

        await store.stop()

    @pytest.mark.asyncio
    async def test_update_preserves_explicit_manifest(self) -> None:
        store = MemorySessionStore()
        await store.start()
        manager = SessionManager(store)

        session = await manager.create_session(
            provider="openai",
            model="gpt-4",
            messages=[Message(role="user", content="hi")],
        )
        explicit_manifest = build_manifest(
            session.session_id,
            [build_system_segment("explicit")],
            anchor_version=99,
        )
        updated = await manager.update_session(
            session.session_id,
            messages=[Message(role="user", content="hi")],
            manifest=explicit_manifest,
        )
        assert updated is not None
        assert updated.manifest is not None
        assert updated.manifest.anchor_version == 99
        assert updated.manifest.get_segment(SegmentType.SYSTEM) is not None
        assert updated.manifest.get_segment(SegmentType.SYSTEM).parts[0].text == "explicit"

        await store.stop()

    @pytest.mark.asyncio
    async def test_update_bumps_version(self) -> None:
        store = MemorySessionStore()
        await store.start()
        manager = SessionManager(store)

        session = await manager.create_session(
            provider="openai",
            model="gpt-4",
            messages=[Message(role="user", content="hi")],
        )
        assert session.version == 0

        updated = await manager.update_session(
            session.session_id,
            messages=[Message(role="user", content="hello")],
        )
        assert updated is not None
        assert updated.version == 1

        await store.stop()


# =============================================================================
# Delta operations
# =============================================================================


class TestDeltaOperations:
    def test_delta_append_new_segment(self) -> None:
        manifest = build_manifest(
            "sess_1",
            [build_system_segment("sys")],
        )
        new_docs = build_segment(SegmentType.DOCS, [TextPart(text="docs")])
        updated = apply_delta(manifest, new_segments=[new_docs])
        assert updated.anchor_version == 1
        assert updated.get_segment(SegmentType.DOCS) is not None

    def test_delta_invalidate_removes_segment(self) -> None:
        seg = build_system_segment("sys")
        manifest = build_manifest("sess_1", [seg])
        updated = apply_delta(manifest, invalidate_hashes=[seg.hash])
        assert updated.get_segment(SegmentType.SYSTEM) is None
        assert updated.anchor_version == 1

    def test_delta_replace_messages(self) -> None:
        manifest = build_manifest(
            "sess_1",
            [
                build_system_segment("sys"),
                build_messages_segment([TextPart(text="old")]),
            ],
        )
        updated = apply_delta(
            manifest,
            replace_messages=[TextPart(text="new1"), TextPart(text="new2")],
        )
        msg_seg = updated.get_segment(SegmentType.MESSAGES)
        assert msg_seg is not None
        assert len(msg_seg.parts) == 2
        assert msg_seg.parts[0].text == "new1"

    def test_delta_does_not_mutate_original(self) -> None:
        manifest = build_manifest(
            "sess_1",
            [build_system_segment("sys"), build_messages_segment([TextPart(text="old")])],
        )
        original_hash = manifest.anchor_hash
        apply_delta(manifest, replace_messages=[TextPart(text="new")])
        assert manifest.anchor_hash == original_hash
        assert manifest.anchor_version == 0


# =============================================================================
# Multimodal round-trip
# =============================================================================


class TestMultimodalRoundTrip:
    def test_manifest_preserves_image_parts(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            },
        ]
        manifest = manifest_from_messages("sess_1", messages)
        msg_seg = manifest.get_segment(SegmentType.MESSAGES)
        assert msg_seg is not None
        # Should contain both text and image parts
        assert len(msg_seg.parts) >= 2

    def test_manifest_to_messages_reconstructs_text(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        manifest = manifest_from_messages("sess_1", messages)
        reconstructed = manifest_to_messages(manifest)
        roles = [m["role"] for m in reconstructed]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles


# =============================================================================
# Reconstruction from manifest
# =============================================================================


class TestReconstructionFromManifest:
    def test_reconstruct_exact_messages(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        manifest = manifest_from_messages("sess_1", messages)
        reconstructed = manifest_to_messages(manifest)
        assert len(reconstructed) == 3
        assert reconstructed[0]["role"] == "system"
        assert reconstructed[1]["role"] == "user"
        assert reconstructed[2]["role"] == "assistant"

    def test_reconstruct_with_tools(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        tools = [{"name": "search", "description": "Search the web"}]
        manifest = manifest_from_messages("sess_1", messages, tools=tools)
        assert manifest.get_segment(SegmentType.TOOLS) is not None
        reconstructed = manifest_to_messages(manifest)
        # Tools don't appear as messages, but system/user do
        assert any(m["role"] == "system" for m in reconstructed)
        assert any(m["role"] == "user" for m in reconstructed)

    def test_manifest_segments_have_cache_eligibility(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        manifest = manifest_from_messages("sess_1", messages)
        for seg in manifest.segments:
            assert seg.hash != ""
            assert seg.token_estimate > 0

    def test_token_accounting_across_segments(self) -> None:
        segments = [
            build_system_segment("sys" * 100),
            build_messages_segment([TextPart(text="hello world")]),
        ]
        manifest = build_manifest("sess_1", segments)
        total = manifest.token_estimate
        individual = sum(s.token_estimate for s in manifest.segments)
        assert total == individual
        assert total > 0
