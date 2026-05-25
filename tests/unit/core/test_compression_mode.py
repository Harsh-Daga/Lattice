"""Tests for LatticeConfig compression_mode mapping.

In the new architecture, the default pipeline has exactly 7 transforms.
Mode-based flag changes do NOT affect default pipeline membership.
Deleted transforms are blocked by is_transform_enabled regardless of mode.
"""

from __future__ import annotations

import pytest

from lattice.core.config import LatticeConfig


def test_default_compression_mode() -> None:
    cfg = LatticeConfig()
    assert cfg.compression_mode == "balanced"


def test_safe_mode_keeps_core_transforms() -> None:
    """All modes keep the 7 production transforms."""
    cfg = LatticeConfig(compression_mode="safe")
    cfg.apply_compression_mode()
    assert cfg.transform_content_profiler is True
    assert cfg.transform_runtime_contract is True
    assert cfg.transform_cache_arbitrage is True
    assert cfg.transform_prefix_opt is True
    assert cfg.transform_reference_sub is True
    assert cfg.transform_tool_filter is True
    assert cfg.transform_output_cleanup is True


def test_mode_does_not_affect_default_pipeline() -> None:
    """Default pipeline is always 7 transforms regardless of mode."""
    for mode in ("safe", "balanced", "aggressive"):
        cfg = LatticeConfig(compression_mode=mode)
        cfg.apply_compression_mode()
        assert cfg.transform_content_profiler is True
        assert cfg.transform_output_cleanup is True
        # Deleted transforms blocked by is_transform_enabled
        assert cfg.is_transform_enabled("hierarchical_summary") is False
        assert cfg.is_transform_enabled("structural_fingerprint") is False


def test_aggressive_mode_sets_extra_flags() -> None:
    cfg = LatticeConfig(compression_mode="aggressive")
    cfg.apply_compression_mode()
    # Extra conditional transforms enabled in aggressive mode
    assert cfg.transform_diagnostic_rle is True
    assert cfg.transform_columnar_pack is True
    assert cfg.transform_json_shape is True
    assert cfg.transform_path_prefix is True
    assert cfg.transform_extractive_compress is True
    assert cfg.transform_tool_projection is True
    assert cfg.transform_format_conversion is True
    # Deleted transforms still blocked
    assert cfg.is_transform_enabled("hierarchical_summary") is False
    assert cfg.is_transform_enabled("structural_fingerprint") is False
    assert cfg.is_transform_enabled("self_information") is False


def test_invalid_compression_mode() -> None:
    with pytest.raises(ValueError, match="compression_mode"):
        LatticeConfig(compression_mode="invalid")
