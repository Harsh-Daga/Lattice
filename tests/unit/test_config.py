"""Unit tests for LatticeConfig fields and helpers."""

from __future__ import annotations

import pytest

from lattice.core.config import LatticeConfig


class TestConfigDefaults:
    """Verify config defaults align with codebase reality."""

    def test_basic_defaults(self):
        cfg = LatticeConfig()
        assert cfg.request_timeout_seconds == 120
        assert cfg.provider_stall_timeout_seconds == 30
        assert cfg.graceful_degradation is True
        assert cfg.proxy_port == 8787

    def test_tacc_defaults(self):
        cfg = LatticeConfig()
        assert cfg.tacc_enabled is True
        assert cfg.tacc_initial_window == 1

    def test_strategy_selection_mode(self):
        cfg = LatticeConfig()
        assert cfg.strategy_selection_mode == "bandit"

    def test_compression_mode_default(self):
        cfg = LatticeConfig()
        assert cfg.compression_mode == "balanced"

    def test_rate_distortion_budget(self):
        cfg = LatticeConfig()
        assert cfg.rate_distortion_budget == pytest.approx(0.02)

    def test_submodular_token_budget(self):
        cfg = LatticeConfig()
        assert cfg.submodular_token_budget == 4096


class TestIsTransformEnabled:
    """Coverage for the ``is_transform_enabled`` helper."""

    def test_known_transform(self):
        cfg = LatticeConfig()
        assert cfg.is_transform_enabled("reference_sub") is True
        assert cfg.is_transform_enabled("tool_filter") is True
        assert cfg.is_transform_enabled("batching") is True
        assert cfg.is_transform_enabled("runtime_contract") is True
        assert cfg.is_transform_enabled("rate_distortion") is True

    def test_unknown_transform(self):
        cfg = LatticeConfig()
        assert cfg.is_transform_enabled("nonexistent") is False

    def test_deleted_transform_returns_false(self):
        cfg = LatticeConfig()
        # These transforms were removed in Phase 0
        assert cfg.is_transform_enabled("fountain_codes") is False
        assert cfg.is_transform_enabled("optimal_stopping") is False
        assert cfg.is_transform_enabled("convex_selector") is False


class TestApplyCompressionMode:
    """Coverage for ``apply_compression_mode``."""

    def test_safe_mode(self):
        cfg = LatticeConfig(compression_mode="safe")
        cfg.apply_compression_mode()
        assert cfg.transform_semantic_compress is False
        assert cfg.transform_hierarchical_summary is False
        assert cfg.transform_reference_sub is True

    def test_balanced_mode(self):
        cfg = LatticeConfig(compression_mode="balanced")
        cfg.apply_compression_mode()
        assert cfg.transform_content_profiler is True
        assert cfg.transform_hierarchical_summary is False

    def test_aggressive_mode(self):
        cfg = LatticeConfig(compression_mode="aggressive")
        cfg.apply_compression_mode()
        assert cfg.transform_semantic_compress is True
        assert cfg.transform_hierarchical_summary is True
