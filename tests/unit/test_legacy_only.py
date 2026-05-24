"""Phase 2b-2a regression — ``legacy_only`` flag on TransformSpec.

Transforms marked ``legacy_only=True`` have no IR-native ``optimize()``
path and must be skipped by the v2 Pipeline runner.
"""

from __future__ import annotations

from lattice.transforms.registry import (
    BUILTIN_TRANSFORMS,
    get_transform_spec,
    is_legacy_only,
)


def test_legacy_only_helper_recognises_strategy_selector() -> None:
    assert is_legacy_only("strategy_selector") is True


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


def test_strategy_selector_spec_has_legacy_only_set() -> None:
    spec = get_transform_spec("strategy_selector")
    assert spec is not None
    assert spec.legacy_only is True


def test_constraint_lifting_removed_from_registry() -> None:
    assert get_transform_spec("constraint_lifting") is None
    assert get_transform_spec("prefix_optimizer") is None


def test_transform_spec_legacy_only_defaults_false() -> None:
    for spec in BUILTIN_TRANSFORMS:
        if spec.canonical_name == "strategy_selector":
            continue
        assert spec.legacy_only is False, (
            f"{spec.canonical_name} unexpectedly marked legacy_only=True"
        )
