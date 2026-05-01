"""Unit tests for LATTICE protocol layer.

Tests cover:
- ContentPart types (TextPart, ImagePart, ToolCallPart, ToolResultPart, ReasoningPart)
- Segment building and hashing
- Manifest construction, canonicalization, and delta operations
- Cache planners (OpenAI, Anthropic, Generic)
- Provider-aware prefix packing
"""

from __future__ import annotations

from lattice.protocol.cache_planner import (
    AnthropicCachePlanner,
    ContextCachePlanner,
    GenericCachePlanner,
    OpenAICachePlanner,
    get_cache_planner,
)
from lattice.protocol.content import (
    ImagePart,
    ImageSource,
    ImageSourceType,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    content_part_hash,
    content_parts_hash,
    content_to_parts,
    parts_from_dict_list,
    parts_to_dict_list,
    parts_to_str,
)
from lattice.protocol.manifest import (
    Manifest,
    apply_delta,
    build_manifest,
    canonicalize_segments,
    compute_anchor_hash,
    manifest_from_messages,
    manifest_summary,
    manifest_to_messages,
)
from lattice.protocol.segments import (
    Segment,
    SegmentType,
    build_messages_segment,
    build_segment,
    build_system_segment,
    build_tools_segment,
)

# =============================================================================
# ContentPart tests
# =============================================================================

class TestTextPart:
    def test_roundtrip(self) -> None:
        part = TextPart(text="hello")
        d = part.to_dict()
        assert d == {"type": "text", "text": "hello"}
        restored = TextPart.from_dict(d)
        assert restored.text == "hello"

    def test_content_to_parts_str(self) -> None:
        parts = content_to_parts("hello")
        assert len(parts) == 1
        assert isinstance(parts[0], TextPart)
        assert parts[0].text == "hello"

    def test_content_to_parts_none(self) -> None:
        parts = content_to_parts(None)
        assert parts == []

    def test_parts_to_str(self) -> None:
        parts = [TextPart(text="hello"), TextPart(text="world")]
        assert parts_to_str(parts) == "hello\nworld"


class TestImagePart:
    def test_roundtrip(self) -> None:
        src = ImageSource(type=ImageSourceType.URL, data="https://example.com/img.jpg")
        part = ImagePart(source=src, detail="high")
        d = part.to_dict()
        restored = ImagePart.from_dict(d)
        assert restored.source.data == "https://example.com/img.jpg"
        assert restored.detail == "high"


class TestToolCallPart:
    def test_roundtrip(self) -> None:
        part = ToolCallPart(id="call_1", name="search", arguments='{"q": "test"}')
        d = part.to_dict()
        restored = ToolCallPart.from_dict(d)
        assert restored.id == "call_1"
        assert restored.name == "search"


class TestToolResultPart:
    def test_roundtrip(self) -> None:
        part = ToolResultPart(tool_call_id="call_1", content="result", is_error=True)
        d = part.to_dict()
        restored = ToolResultPart.from_dict(d)
        assert restored.content == "result"
        assert restored.is_error is True


class TestReasoningPart:
    def test_roundtrip(self) -> None:
        part = ReasoningPart(text="thinking...", signature="sig123")
        d = part.to_dict()
        restored = ReasoningPart.from_dict(d)
        assert restored.text == "thinking..."
        assert restored.signature == "sig123"


class TestContentHelpers:
    def test_parts_to_dict_list(self) -> None:
        parts = [TextPart(text="a"), TextPart(text="b")]
        assert parts_to_dict_list(parts) == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]

    def test_parts_from_dict_list_unknown_type(self) -> None:
        data = [{"type": "unknown", "foo": "bar"}]
        parts = parts_from_dict_list(data)
        assert len(parts) == 1
        assert isinstance(parts[0], TextPart)

    def test_content_part_hash_deterministic(self) -> None:
        part = TextPart(text="hello")
        h1 = content_part_hash(part)
        h2 = content_part_hash(part)
        assert h1 == h2
        assert len(h1) == 16

    def test_content_parts_hash_empty(self) -> None:
        assert content_parts_hash([]) == ""


# =============================================================================
# Segment tests
# =============================================================================

class TestSegment:
    def test_build_segment_computes_hash(self) -> None:
        seg = build_segment(SegmentType.SYSTEM, [TextPart(text="hello")])
        assert seg.type == SegmentType.SYSTEM
        assert seg.hash != ""
        assert len(seg.hash) == 16

    def test_build_system_segment(self) -> None:
        seg = build_system_segment("You are helpful.")
        assert seg.type == SegmentType.SYSTEM
        assert seg.parts[0].text == "You are helpful."

    def test_build_tools_segment(self) -> None:
        tools = [{"name": "search"}]
        seg = build_tools_segment(tools)
        assert seg.type == SegmentType.TOOLS
        assert "search" in seg.parts[0].text

    def test_build_messages_segment(self) -> None:
        seg = build_messages_segment([TextPart(text="hello")], version=3)
        assert seg.type == SegmentType.MESSAGES
        assert seg.version == 3

    def test_segment_token_estimate(self) -> None:
        seg = build_system_segment("hello world")  # 11 chars ~ 3 tokens
        assert seg.token_estimate >= 2

    def test_segment_roundtrip(self) -> None:
        seg = build_system_segment("sys")
        d = seg.to_dict()
        restored = Segment.from_dict(d)
        assert restored.type == seg.type
        assert restored.hash == seg.hash


# =============================================================================
# Manifest tests
# =============================================================================

class TestManifest:
    def test_build_manifest_computes_anchor_hash(self) -> None:
        segments = [build_system_segment("sys")]
        manifest = build_manifest("sess_1", segments)
        assert manifest.anchor_hash != ""
        assert len(manifest.anchor_hash) == 64  # full SHA-256
        assert manifest.anchor_version == 0

    def test_manifest_id_auto_generated(self) -> None:
        manifest = build_manifest("sess_1", [build_system_segment("sys")])
        assert manifest.manifest_id.startswith("mf-")

    def test_manifest_token_estimate(self) -> None:
        segments = [
            build_system_segment("sys"),
            build_messages_segment([TextPart(text="hello")]),
        ]
        manifest = build_manifest("sess_1", segments)
        assert manifest.token_estimate > 0

    def test_get_segment(self) -> None:
        segments = [build_system_segment("sys")]
        manifest = build_manifest("sess_1", segments)
        assert manifest.get_segment(SegmentType.SYSTEM) is not None
        assert manifest.get_segment(SegmentType.MESSAGES) is None

    def test_manifest_roundtrip(self) -> None:
        segments = [build_system_segment("sys")]
        manifest = build_manifest("sess_1", segments, metadata={"model": "gpt-4"})
        d = manifest.to_dict()
        restored = Manifest.from_dict(d)
        assert restored.anchor_hash == manifest.anchor_hash

    def test_manifest_summary(self) -> None:
        manifest = build_manifest(
            "sess_1",
            [build_system_segment("sys"), build_messages_segment([TextPart(text="hello")])],
            metadata={"provider": "openai"},
        )
        summary = manifest_summary(manifest)
        assert summary["session_id"] == "sess_1"
        assert summary["segment_count"] == 2
        assert summary["token_estimate"] == manifest.token_estimate
        assert summary["segment_counts"]["system"] == 1
        assert summary["segment_counts"]["messages"] == 1
        assert summary["metadata"] == {"provider": "openai"}


class TestCanonicalizeSegments:
    def test_order_tools_first(self) -> None:
        segments = [
            build_messages_segment([TextPart(text="hi")]),
            build_system_segment("sys"),
            build_tools_segment([{"name": "search"}]),
        ]
        ordered = canonicalize_segments(segments)
        assert ordered[0].type == SegmentType.TOOLS
        assert ordered[1].type == SegmentType.SYSTEM
        assert ordered[2].type == SegmentType.MESSAGES


class TestManifestFromMessages:
    def test_extracts_system(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        manifest = manifest_from_messages("sess_1", messages)
        assert manifest.get_segment(SegmentType.SYSTEM) is not None
        assert manifest.get_segment(SegmentType.MESSAGES) is not None

    def test_extracts_tools(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        tools = [{"name": "search"}]
        manifest = manifest_from_messages("sess_1", messages, tools=tools)
        assert manifest.get_segment(SegmentType.TOOLS) is not None

    def test_roundtrip(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        manifest = manifest_from_messages("sess_1", messages)
        reconstructed = manifest_to_messages(manifest)
        assert any(m["role"] == "system" for m in reconstructed)


class TestApplyDelta:
    def test_append_messages(self) -> None:
        manifest = build_manifest(
            "sess_1",
            [build_system_segment("sys"), build_messages_segment([TextPart(text="hi")])],
        )
        new_msg = build_messages_segment([TextPart(text="hello")], version=1)
        updated = apply_delta(manifest, new_segments=[new_msg])
        assert updated.anchor_version == 1

    def test_invalidate_hashes(self) -> None:
        seg = build_system_segment("sys")
        manifest = build_manifest("sess_1", [seg])
        updated = apply_delta(manifest, invalidate_hashes=[seg.hash])
        assert updated.get_segment(SegmentType.SYSTEM) is None

    def test_replace_messages(self) -> None:
        manifest = build_manifest(
            "sess_1",
            [build_system_segment("sys"), build_messages_segment([TextPart(text="old")])],
        )
        updated = apply_delta(manifest, replace_messages=[TextPart(text="new")])
        msg_seg = updated.get_segment(SegmentType.MESSAGES)
        assert msg_seg is not None
        assert msg_seg.parts[0].text == "new"


class TestComputeAnchorHash:
    def test_deterministic(self) -> None:
        segments = [build_system_segment("sys")]
        h1 = compute_anchor_hash(segments, {})
        h2 = compute_anchor_hash(segments, {})
        assert h1 == h2

    def test_different_metadata_different_hash(self) -> None:
        segments = [build_system_segment("sys")]
        h1 = compute_anchor_hash(segments, {"a": 1})
        h2 = compute_anchor_hash(segments, {"a": 2})
        assert h1 != h2


# =============================================================================
# Cache planner tests
# =============================================================================

class TestOpenAICachePlanner:
    def test_stable_prefix_order(self) -> None:
        segments = [
            build_messages_segment([TextPart(text="hi")]),
            build_system_segment("sys"),
            build_tools_segment([{"name": "search"}]),
        ]
        manifest = build_manifest("sess_1", segments)
        planner = OpenAICachePlanner()
        plan = planner.plan(manifest)
        assert plan.annotations["provider"] == "openai"
        assert plan.expected_cached_tokens > 0

    def test_breakpoints(self) -> None:
        segments = [
            build_tools_segment([{"name": "search"}]),
            build_system_segment("sys"),
            build_messages_segment([TextPart(text="hi")]),
        ]
        manifest = build_manifest("sess_1", segments)
        planner = OpenAICachePlanner()
        plan = planner.plan(manifest)
        # Breakpoint should be at messages segment
        assert len(plan.breakpoints) > 0


class TestAnthropicCachePlanner:
    def test_cache_control_annotations(self) -> None:
        segments = [
            build_tools_segment([{"name": "search"}]),
            build_system_segment("sys"),
            build_messages_segment([TextPart(text="hi")]),
        ]
        manifest = build_manifest("sess_1", segments)
        planner = AnthropicCachePlanner()
        plan = planner.plan(manifest)
        assert plan.annotations["provider"] == "anthropic"
        assert plan.annotations["cache_breakpoints"] > 0

    def test_max_breakpoints(self) -> None:
        segments = [
            build_tools_segment([{"name": "search"}]),
            build_system_segment("sys"),
            build_system_segment("sys2"),
            build_system_segment("sys3"),
            build_system_segment("sys4"),
            build_system_segment("sys5"),
            build_messages_segment([TextPart(text="hi")]),
        ]
        manifest = build_manifest("sess_1", segments)
        planner = AnthropicCachePlanner()
        plan = planner.plan(manifest)
        assert plan.annotations["cache_breakpoints"] <= planner.max_breakpoints


class TestContextCachePlanner:
    def test_context_cache_annotations(self) -> None:
        segments = [
            build_system_segment("sys"),
            build_tools_segment([{"name": "search"}]),
            build_messages_segment([TextPart(text="hi")]),
        ]
        manifest = build_manifest("sess_1", segments)
        planner = ContextCachePlanner(provider="gemini")
        plan = planner.plan(manifest)
        assert plan.annotations["provider"] == "gemini"
        assert plan.annotations["explicit_context_cache"] is True
        assert plan.expected_cached_tokens > 0
        assert plan.segments[0].metadata["cache_resource"]["eligible"] is True


class TestGenericCachePlanner:
    def test_no_annotations(self) -> None:
        segments = [build_system_segment("sys")]
        manifest = build_manifest("sess_1", segments)
        planner = GenericCachePlanner()
        plan = planner.plan(manifest)
        assert plan.annotations["provider"] == "generic"
        assert plan.expected_cached_tokens == 0


class TestCachePlannerRegistry:
    def test_openai(self) -> None:
        planner = get_cache_planner("openai")
        assert isinstance(planner, OpenAICachePlanner)

    def test_anthropic(self) -> None:
        planner = get_cache_planner("anthropic")
        assert isinstance(planner, AnthropicCachePlanner)

    def test_bedrock(self) -> None:
        planner = get_cache_planner("bedrock")
        assert isinstance(planner, AnthropicCachePlanner)

    def test_gemini(self) -> None:
        planner = get_cache_planner("gemini")
        assert isinstance(planner, ContextCachePlanner)

    def test_vertex(self) -> None:
        planner = get_cache_planner("vertex")
        assert isinstance(planner, ContextCachePlanner)

    def test_unknown_fallback(self) -> None:
        planner = get_cache_planner("unknown")
        assert isinstance(planner, GenericCachePlanner)
