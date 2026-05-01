"""Unit tests for cache arbitrage optimizer."""

from __future__ import annotations

from typing import Any

from lattice.core.context import TransformContext
from lattice.core.result import is_ok, unwrap
from lattice.core.transport import Message, Request, Response
from lattice.transforms.cache_arbitrage import CacheArbitrageOptimizer


def test_cache_arbitrage_reorders_system_first() -> None:
    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="user", content="Hello."),
            Message(role="system", content="Be helpful."),
        ]
    )
    result = transform.process(request, TransformContext())
    modified = unwrap(result)
    roles = [m.role for m in modified.messages]
    assert roles[0] == "system"
    assert roles[-1] == "user"


def test_cache_arbitrage_groups_tools_before_variable() -> None:
    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="user", content="Run the tool."),
            Message(role="tool", content='{"result": 42}'),
            Message(role="system", content="System."),
        ]
    )
    result = transform.process(request, TransformContext())
    modified = unwrap(result)
    roles = [m.role for m in modified.messages]
    assert roles == ["system", "tool", "user"]


def test_cache_arbitrage_annotates_stable_metadata() -> None:
    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="system", content="System."),
            Message(role="user", content="Hello."),
        ]
    )
    result = transform.process(request, TransformContext())
    modified = unwrap(result)
    assert modified.messages[0].metadata.get("_cache_stable") is True
    assert modified.messages[1].metadata.get("_cache_stable") is False


def test_cache_arbitrage_records_metrics() -> None:
    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="system", content="System prompt here."),
            Message(role="user", content="Hello."),
        ]
    )
    context = TransformContext()
    result = transform.process(request, context)
    assert is_ok(result)
    metrics = context.metrics["transforms"].get("cache_arbitrage", {})
    assert "stability_score" in metrics
    assert "stable_tokens" in metrics
    assert metrics["stability_score"] > 0.0


def test_cache_arbitrage_tracks_hits_and_misses() -> None:
    transform = CacheArbitrageOptimizer(track_hits=True)
    request = Request(
        messages=[
            Message(role="system", content="System."),
            Message(role="user", content="Hello."),
        ]
    )
    ctx = TransformContext()
    transform.process(request.copy(), ctx)
    metrics = ctx.metrics["transforms"].get("cache_arbitrage", {})
    assert metrics.get("miss_count", 0) == 1
    assert metrics.get("hit_count", 0) == 0

    # Identical request should hit cache
    transform.process(request.copy(), ctx)
    metrics = ctx.metrics["transforms"].get("cache_arbitrage", {})
    assert metrics.get("hit_count", 0) == 1


def test_cache_arbitrage_stability_score_in_metadata() -> None:
    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="system", content="System."),
            Message(role="user", content="Hello."),
        ]
    )
    result = transform.process(request, TransformContext())
    modified = unwrap(result)
    arb = modified.metadata.get("_cache_arbitrage", {})
    assert "stability_score" in arb
    assert "stable_tokens" in arb


def test_cache_arbitrage_reverse_is_noop() -> None:
    transform = CacheArbitrageOptimizer()
    response = Response(content="world")
    restored = transform.reverse(response, TransformContext())
    assert restored.content == "world"


def test_cache_arbitrage_preserves_tools() -> None:
    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="user", content="Use the tool."),
        ],
        tools=[{"type": "function", "function": {"name": "foo"}}],
    )
    result = transform.process(request, TransformContext())
    modified = unwrap(result)
    assert modified.tools is not None
    assert len(modified.tools) == 1


def test_cache_arbitrage_uses_bedrock_cache_semantics() -> None:
    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="system", content="System prompt."),
            Message(role="user", content="Hello."),
        ]
    )
    context = TransformContext(provider="bedrock")
    result = transform.process(request, context)
    modified = unwrap(result)
    arb = modified.metadata.get("_cache_arbitrage", {})
    assert arb["annotations"]["provider"] == "bedrock"
    assert arb["annotations"]["max_breakpoints"] == 4
    assert modified.metadata.get("bedrock_prompt_caching") is True


def test_cache_arbitrage_uses_context_cache_semantics() -> None:
    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="system", content="System prompt."),
            Message(role="user", content="Hello."),
        ]
    )
    context = TransformContext(provider="gemini")
    result = transform.process(request, context)
    modified = unwrap(result)
    arb = modified.metadata.get("_cache_arbitrage", {})
    assert arb["annotations"]["provider"] == "gemini"
    assert arb["annotations"]["explicit_context_cache"] is True


def test_cache_arbitrage_sets_openai_prompt_cache_hints() -> None:
    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="system", content="System prompt."),
            Message(role="user", content="Hello."),
        ]
    )
    context = TransformContext(provider="openai")
    result = transform.process(request, context)
    modified = unwrap(result)
    assert "prompt_cache_key" in modified.metadata
    assert modified.metadata["prompt_cache_key"]
    assert modified.metadata["prompt_cache_retention"] == "10m"


def test_cache_arbitrage_sets_anthropic_cache_hints() -> None:
    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="system", content="System prompt."),
            Message(role="user", content="Hello."),
        ]
    )
    context = TransformContext(provider="anthropic")
    result = transform.process(request, context)
    modified = unwrap(result)
    assert modified.metadata.get("anthropic_cache_control") is True
    assert modified.metadata.get("anthropic_cache_ttl_seconds") == 300


def test_cache_arbitrage_noop_when_already_ordered() -> None:
    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="system", content="System."),
            Message(role="user", content="Hello."),
        ]
    )
    ctx = TransformContext()
    result = transform.process(request, ctx)
    modified = unwrap(result)
    state = ctx.get_transform_state("cache_arbitrage")
    assert state.get("reordered") is False
    assert [m.role for m in modified.messages] == ["system", "user"]


def test_planner_failure_is_reported_not_swallowed(monkeypatch: Any) -> None:
    class BadPlanner:
        def plan(self, manifest: Any) -> Any:
            raise ValueError("bad plan")

    def mock_get_cache_planner(provider: str) -> Any:
        return BadPlanner()

    monkeypatch.setattr(
        "lattice.transforms.cache_arbitrage.get_cache_planner",
        mock_get_cache_planner,
    )

    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="system", content="System."),
            Message(role="user", content="Hello."),
        ]
    )
    context = TransformContext(provider="openai")
    result = transform.process(request, context)
    modified = unwrap(result)
    arb = modified.metadata.get("_cache_arbitrage", {})
    assert "planner_failure" in arb
    assert arb["planner_failure"]["category"] == "validation_error"
    assert "skip_reason" not in arb
    assert "plan_applied" not in arb
    metrics = context.metrics["transforms"].get("cache_arbitrage", {})
    assert metrics.get("planner_failure") == "validation_error"


def test_prefix_hash_is_stable_across_equivalent_requests() -> None:
    transform = CacheArbitrageOptimizer(track_hits=True)
    request = Request(
        messages=[
            Message(role="system", content="System."),
            Message(role="user", content="Hello."),
        ]
    )
    ctx1 = TransformContext()
    transform.process(request.copy(), ctx1)
    state1 = ctx1.get_transform_state("cache_arbitrage")
    hash1 = state1.get("prefix_hash")

    ctx2 = TransformContext()
    transform.process(request.copy(), ctx2)
    state2 = ctx2.get_transform_state("cache_arbitrage")
    hash2 = state2.get("prefix_hash")

    assert hash1 == hash2
    assert hash1 is not None


def test_provider_hints_only_applied_when_valid() -> None:
    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="system", content="System."),
            Message(role="user", content="Hello."),
        ]
    )
    context = TransformContext(provider="unknown_provider")
    result = transform.process(request, context)
    modified = unwrap(result)
    arb = modified.metadata.get("_cache_arbitrage", {})
    assert arb.get("skip_reason") == "provider_not_in_registry"
    assert "prompt_cache_key" not in modified.metadata
    assert "anthropic_cache_control" not in modified.metadata


def test_telemetry_shows_applied_vs_skipped_plan_reasons() -> None:
    # Applied case
    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="system", content="System."),
            Message(role="user", content="Hello."),
        ]
    )
    ctx_applied = TransformContext(provider="openai")
    transform.process(request.copy(), ctx_applied)
    metrics_applied = ctx_applied.metrics["transforms"].get("cache_arbitrage", {})
    assert metrics_applied.get("plan_applied") is True

    # Skipped case — unknown provider
    ctx_skipped = TransformContext(provider="unknown_provider")
    transform.process(request.copy(), ctx_skipped)
    metrics_skipped = ctx_skipped.metrics["transforms"].get("cache_arbitrage", {})
    assert metrics_skipped.get("plan_skipped") == "provider_not_in_registry"


def test_md5_replaced_with_sha256() -> None:
    transform = CacheArbitrageOptimizer(track_hits=True)
    request = Request(
        messages=[
            Message(role="system", content="System."),
            Message(role="user", content="Hello."),
        ]
    )
    ctx = TransformContext()
    transform.process(request, ctx)
    state = ctx.get_transform_state("cache_arbitrage")
    prefix_hash = state.get("prefix_hash", "")
    assert len(prefix_hash) == 64


def test_plan_idempotent() -> None:
    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="system", content="System."),
            Message(role="user", content="Hello."),
        ]
    )
    ctx1 = TransformContext(provider="openai")
    result1 = transform.process(request, ctx1)
    modified1 = unwrap(result1)
    arb1 = dict(modified1.metadata.get("_cache_arbitrage", {}))

    ctx2 = TransformContext(provider="openai")
    result2 = transform.process(modified1, ctx2)
    modified2 = unwrap(result2)
    arb2 = dict(modified2.metadata.get("_cache_arbitrage", {}))

    # Messages should be same order
    assert [m.role for m in modified1.messages] == [m.role for m in modified2.messages]

    # plan_applied should still be True, plan_summary should be identical
    assert arb1.get("plan_applied") is True
    assert arb2.get("plan_applied") is True
    assert arb1.get("plan_summary") == arb2.get("plan_summary")


def test_anthropic_breakpoint_limit_respected() -> None:
    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="system", content="System 1."),
            Message(role="system", content="System 2."),
            Message(role="system", content="System 3."),
            Message(role="system", content="System 4."),
            Message(role="system", content="System 5."),
            Message(role="user", content="Hello."),
        ]
    )
    context = TransformContext(provider="anthropic")
    result = transform.process(request, context)
    modified = unwrap(result)
    arb = modified.metadata.get("_cache_arbitrage", {})
    breakpoints = arb.get("breakpoints", [])
    assert len(breakpoints) <= 4
    assert arb["annotations"]["max_breakpoints"] == 4


def test_injected_manifest_is_preferred() -> None:
    from lattice.protocol.content import TextPart
    from lattice.protocol.manifest import build_manifest
    from lattice.protocol.segments import SegmentType, build_segment

    transform = CacheArbitrageOptimizer()
    injected_manifest = build_manifest(
        session_id="sess-123",
        segments=[
            build_segment(
                SegmentType.SYSTEM,
                parts=[TextPart(text="Injected system prompt.")],
            ),
        ],
    )
    request = Request(
        messages=[
            Message(role="user", content="Hello."),
        ]
    )
    request.metadata["_lattice_manifest"] = injected_manifest
    context = TransformContext(provider="openai")
    result = transform.process(request, context)
    modified = unwrap(result)
    arb = modified.metadata.get("_cache_arbitrage", {})
    assert arb.get("manifest_source") == "injected"
    assert arb.get("plan_applied") is True


def test_dict_manifest_deserializes_correctly() -> None:
    from lattice.protocol.content import TextPart
    from lattice.protocol.manifest import build_manifest
    from lattice.protocol.segments import SegmentType, build_segment

    transform = CacheArbitrageOptimizer()
    injected_manifest = build_manifest(
        session_id="sess-456",
        segments=[
            build_segment(
                SegmentType.SYSTEM,
                parts=[TextPart(text="Dict system prompt.")],
            ),
        ],
    )
    request = Request(
        messages=[
            Message(role="user", content="Hello."),
        ]
    )
    request.metadata["_lattice_manifest"] = injected_manifest.to_dict()
    context = TransformContext(provider="openai")
    result = transform.process(request, context)
    modified = unwrap(result)
    arb = modified.metadata.get("_cache_arbitrage", {})
    assert arb.get("manifest_source") == "injected"
    assert arb.get("plan_applied") is True


def test_fallback_manifest_reconstructed_when_none_injected() -> None:
    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="system", content="System."),
            Message(role="user", content="Hello."),
        ]
    )
    context = TransformContext(provider="openai")
    result = transform.process(request, context)
    modified = unwrap(result)
    arb = modified.metadata.get("_cache_arbitrage", {})
    assert arb.get("manifest_source") == "reconstructed"
    assert arb.get("plan_applied") is True


def test_normalized_outcome_payload_present() -> None:
    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="system", content="System prompt."),
            Message(role="user", content="Hello."),
        ]
    )
    context = TransformContext(provider="openai")
    result = transform.process(request, context)
    modified = unwrap(result)

    # Legacy metadata still present
    assert "_cache_arbitrage" in modified.metadata

    # Normalized outcome also present
    outcome = modified.metadata.get("_cache_arbitrage_outcome")
    assert outcome is not None
    assert outcome["manifest_source"] == "reconstructed"
    assert outcome["plan_applied"] is True
    assert "plan_summary" in outcome
    assert "stability_score" in outcome
    assert "stable_tokens" in outcome
    assert "prefix_hash" in outcome
    assert "reordered" in outcome


def test_normalized_outcome_with_skip_reason() -> None:
    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="system", content="System."),
            Message(role="user", content="Hello."),
        ]
    )
    context = TransformContext(provider="unknown_provider")
    result = transform.process(request, context)
    modified = unwrap(result)

    outcome = modified.metadata.get("_cache_arbitrage_outcome")
    assert outcome is not None
    assert outcome["skip_reason"] == "provider_not_in_registry"
    assert outcome["plan_applied"] is False
    assert outcome["planner_failure"] is None


def test_no_dead_code_path_in_process() -> None:
    """The process method has exactly two return paths: early (tool_call guard)
    and normalized outcome. No unreachable dead code follows the final return."""
    import inspect

    source = inspect.getsource(CacheArbitrageOptimizer.process)
    # Two return paths: early return for tool-call conversations + final return
    return_lines = [line for line in source.split("\n") if line.strip().startswith("return Ok")]
    assert len(return_lines) == 2
    # The second `return Ok` should be the last logical statement in the method body
    # (not followed by any unreachable code)
    last_return_idx = source.rindex("return Ok(request)")
    rest = source[last_return_idx + len("return Ok(request)") :]
    # After the final return, only the def line of reverse() follows
    assert "def reverse" in rest or rest.strip() == ""


def test_legacy_metadata_matches_normalized_outcome() -> None:
    """Legacy _cache_arbitrage fields match the normalized outcome."""
    transform = CacheArbitrageOptimizer()
    request = Request(
        messages=[
            Message(role="system", content="System prompt."),
            Message(role="user", content="Hello."),
        ]
    )
    context = TransformContext(provider="openai")
    result = transform.process(request, context)
    modified = unwrap(result)

    legacy = modified.metadata.get("_cache_arbitrage", {})
    outcome = modified.metadata.get("_cache_arbitrage_outcome", {})

    assert legacy.get("manifest_source") == outcome.get("manifest_source")
    assert legacy.get("plan_applied") == outcome.get("plan_applied")
    assert legacy.get("stability_score") == outcome.get("stability_score")
    assert legacy.get("stable_tokens") == outcome.get("stable_tokens")
    # prefix_hash and reordered are only in normalized outcome, not legacy
    assert "prefix_hash" in outcome
    # skip_reason is only in legacy when truthy (success path has empty skip_reason)
    if outcome.get("skip_reason"):
        assert legacy.get("skip_reason") == outcome.get("skip_reason")
    else:
        assert "skip_reason" not in legacy
    # planner_failure is only present in legacy when not None
    if outcome.get("planner_failure") is not None:
        assert legacy.get("planner_failure") == outcome.get("planner_failure")


def test_manifest_provenance_no_regression() -> None:
    """Manifest provenance is reported correctly for all cases."""
    from lattice.protocol.content import TextPart
    from lattice.protocol.manifest import build_manifest
    from lattice.protocol.segments import SegmentType, build_segment

    transform = CacheArbitrageOptimizer()

    # Injected manifest
    injected = build_manifest(
        session_id="sess-x",
        segments=[build_segment(SegmentType.SYSTEM, parts=[TextPart(text="System.")])],
    )
    req = Request(messages=[Message(role="user", content="Hello.")])
    req.metadata["_lattice_manifest"] = injected
    ctx = TransformContext(provider="openai")
    result = transform.process(req, ctx)
    modified = unwrap(result)
    outcome = modified.metadata.get("_cache_arbitrage_outcome", {})
    assert outcome["manifest_source"] == "injected"

    # Reconstructed
    req2 = Request(
        messages=[
            Message(role="system", content="System prompt."),
            Message(role="user", content="Hello."),
        ]
    )
    ctx2 = TransformContext(provider="openai")
    result2 = transform.process(req2, ctx2)
    modified2 = unwrap(result2)
    outcome2 = modified2.metadata.get("_cache_arbitrage_outcome", {})
    assert outcome2["manifest_source"] == "reconstructed"

    # Missing (unknown provider)
    req3 = Request(messages=[Message(role="user", content="Hello.")])
    ctx3 = TransformContext(provider="unknown_provider")
    result3 = transform.process(req3, ctx3)
    modified3 = unwrap(result3)
    outcome3 = modified3.metadata.get("_cache_arbitrage_outcome", {})
    assert outcome3["skip_reason"] == "provider_not_in_registry"


def test_skip_reason_no_regression() -> None:
    """Skip reasons are reported correctly."""
    transform = CacheArbitrageOptimizer()

    # Provider not in registry
    req = Request(messages=[Message(role="user", content="Hello.")])
    ctx = TransformContext(provider="unknown_provider")
    result = transform.process(req, ctx)
    modified = unwrap(result)
    outcome = modified.metadata.get("_cache_arbitrage_outcome", {})
    assert outcome["skip_reason"] == "provider_not_in_registry"
    assert outcome["plan_applied"] is False
