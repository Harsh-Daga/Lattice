"""Verify deleted transforms are absent + unimportable from the canonical v2 path."""

from __future__ import annotations

import pytest

from lattice.planner.unified_planner import Tier, UnifiedPlanner

_DELETED_TRANSFORMS = {
    "alias_manifest",
    "dictionary_compress",
    "grammar_compress",
}


class TestDeletedTransforms:
    """Deleted legacy transforms must not appear anywhere in the planner."""

    @pytest.mark.parametrize("tx_name", _DELETED_TRANSFORMS)
    def test_not_in_transform_order(self, tx_name: str) -> None:
        assert tx_name not in UnifiedPlanner._TRANSFORM_ORDER, (
            f"{tx_name} must not be in _TRANSFORM_ORDER"
        )

    @pytest.mark.parametrize("tx_name", _DELETED_TRANSFORMS)
    def test_not_in_any_tier(self, tx_name: str) -> None:
        for tier in Tier:
            allowed = UnifiedPlanner._TIER_ALLOWED.get(tier, set())
            assert tx_name not in allowed, f"{tx_name} must not be in tier {tier.value}"

    @pytest.mark.parametrize("tx_name", _DELETED_TRANSFORMS)
    def test_not_in_fast_tier(self, tx_name: str) -> None:
        assert tx_name not in UnifiedPlanner._TIER_ALLOWED[Tier.FAST]

    @pytest.mark.parametrize("tx_name", _DELETED_TRANSFORMS)
    def test_not_in_standard_tier(self, tx_name: str) -> None:
        assert tx_name not in UnifiedPlanner._TIER_ALLOWED[Tier.STANDARD]

    @pytest.mark.parametrize("tx_name", _DELETED_TRANSFORMS)
    def test_not_in_safe_tier(self, tx_name: str) -> None:
        assert tx_name not in UnifiedPlanner._TIER_ALLOWED[Tier.SAFE]

    @pytest.mark.parametrize("tx_name", _DELETED_TRANSFORMS)
    def test_not_in_reasoning_tier(self, tx_name: str) -> None:
        assert tx_name not in UnifiedPlanner._TIER_ALLOWED[Tier.REASONING]

    @pytest.mark.parametrize(
        "mod",
        [
            "lattice.transforms.alias_manifest",
            "lattice.transforms.dictionary_compress",
            "lattice.transforms.grammar_compress",
        ],
    )
    def test_module_not_importable(self, mod: str) -> None:
        with pytest.raises(ModuleNotFoundError):
            __import__(mod)
