"""Deprecated transform config flags must not exist on LatticeConfig (Phase 12)."""

from __future__ import annotations

from lattice.core.config import LatticeConfig


def test_lattice_config_has_no_removed_transform_fields() -> None:
    removed = {
        "transform_prefix_opt",
        "transform_constraint_lifting",
        "transform_strategy_selector",
        "transform_information_theoretic_selector",
    }
    fields = set(LatticeConfig.model_fields)
    assert removed.isdisjoint(fields)


def test_deprecated_flags_in_input_are_ignored() -> None:
    cfg = LatticeConfig(
        transform_prefix_opt=True,
        transform_strategy_selector=True,
    )
    assert not hasattr(cfg, "transform_prefix_opt")
