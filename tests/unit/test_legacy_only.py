"""Phase 2b-2a regression — ``legacy_only`` flag on TransformSpec."""

from __future__ import annotations

from lattice.transforms.registry import BUILTIN_TRANSFORMS, get_transform_spec, is_legacy_only


def test_no_builtin_transforms_are_legacy_only() -> None:
    legacy = [s.canonical_name for s in BUILTIN_TRANSFORMS if s.legacy_only]
    assert legacy == [], f"unexpected legacy_only transforms: {legacy}"


def test_legacy_only_helper_returns_false_for_ir_native_transforms() -> None:
    for name in (
        "content_profiler",
        "runtime_contract",
        "cache_arbitrage",
        "reference_sub",
        "tool_filter",
        "causal_chain",
        "message_dedup",
        "rate_distortion",
        "path_prefix",
        "format_conversion",
        "tool_projection",
    ):
        assert is_legacy_only(name) is False, f"{name} should NOT be legacy_only"


def test_legacy_only_helper_returns_false_for_unknown_name() -> None:
    assert is_legacy_only("not_a_real_transform") is False


def test_phase_5_removed_transforms_not_in_registry() -> None:
    for name in (
        "strategy_selector",
        "information_theoretic_selector",
        "constraint_lifting",
        "prefix_optimizer",
    ):
        assert get_transform_spec(name) is None
