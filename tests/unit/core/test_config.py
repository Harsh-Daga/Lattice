"""Tests for LatticeConfig.apply_compression_mode."""

from __future__ import annotations

from lattice.core.config import LatticeConfig


class TestApplyCompressionMode:
    """Coverage for ``apply_compression_mode``."""

    def test_safe_mode(self):
        cfg = LatticeConfig(compression_mode="safe")
        cfg.apply_compression_mode()
        assert cfg.transform_content_profiler is True
        assert cfg.transform_reference_sub is True

    def test_balanced_mode(self):
        cfg = LatticeConfig(compression_mode="balanced")
        cfg.apply_compression_mode()
        assert cfg.transform_content_profiler is True
        assert cfg.transform_causal_chain is True

    def test_aggressive_mode(self):
        cfg = LatticeConfig(compression_mode="aggressive")
        cfg.apply_compression_mode()
        assert cfg.transform_content_profiler is True
        # Deleted transforms are blocked by is_transform_enabled
        assert cfg.is_transform_enabled("hierarchical_summary") is False
        assert cfg.is_transform_enabled("structural_fingerprint") is False

    def test_production_transforms_always_on(self):
        for mode in ("safe", "balanced", "aggressive"):
            cfg = LatticeConfig(compression_mode=mode)
            cfg.apply_compression_mode()
            assert cfg.transform_content_profiler is True
            assert cfg.transform_runtime_contract is True
            assert cfg.transform_cache_arbitrage is True
            assert cfg.transform_reference_sub is True
            assert cfg.transform_tool_filter is True
            assert cfg.transform_output_cleanup is True
