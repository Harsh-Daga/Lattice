"""Tests for LatticeConfig compression_mode mapping."""

from __future__ import annotations

import pytest

from lattice.core.config import LatticeConfig


def test_default_compression_mode() -> None:
    cfg = LatticeConfig()
    assert cfg.compression_mode == "balanced"


def test_safe_mode_disables_lossy() -> None:
    cfg = LatticeConfig(compression_mode="safe")
    cfg.apply_compression_mode()
    assert cfg.transform_semantic_compress is False
    assert cfg.transform_hierarchical_summary is False
    assert cfg.transform_context_selector is False
    assert cfg.transform_reference_sub is True
    assert cfg.transform_tool_filter is True


def test_balanced_mode_enables_selective() -> None:
    cfg = LatticeConfig(compression_mode="balanced")
    cfg.apply_compression_mode()
    assert cfg.transform_content_profiler is True
    assert cfg.transform_structural_fingerprint is True
    assert cfg.transform_self_information is True
    assert cfg.transform_hierarchical_summary is False
    assert cfg.transform_semantic_compress is False


def test_aggressive_mode_enables_all() -> None:
    cfg = LatticeConfig(compression_mode="aggressive")
    cfg.apply_compression_mode()
    assert cfg.transform_content_profiler is True
    assert cfg.transform_structural_fingerprint is True
    assert cfg.transform_self_information is True
    assert cfg.transform_hierarchical_summary is True
    assert cfg.transform_context_selector is True
    assert cfg.transform_semantic_compress is True


def test_invalid_compression_mode() -> None:
    with pytest.raises(ValueError, match="compression_mode"):
        LatticeConfig(compression_mode="invalid")
