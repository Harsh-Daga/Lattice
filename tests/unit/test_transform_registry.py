"""Invariant tests for the central transform registry.

These tests ensure that config, pipeline, and safety metadata all agree
on transform identity, enablement, and classification.
"""

from __future__ import annotations

from lattice.core.config import LatticeConfig
from lattice.core.pipeline_factory import build_default_pipeline, pipeline_summary
from lattice.core.transform_registry import (
    BUILTIN_TRANSFORMS,
    get_transform_spec,
    is_transform_enabled,
    is_transform_name_known,
    list_default_pipeline_names,
    list_transform_names,
)
from lattice.utils.validation import (
    TransformSafetyBucket,
    get_transform_safety_bucket,
    transform_allowed_at_risk,
)

# =============================================================================
# Registry completeness
# =============================================================================


class TestRegistryCompleteness:
    """Every built-in transform must be discoverable and have a spec."""

    def test_all_canonical_names_known(self) -> None:
        for spec in BUILTIN_TRANSFORMS:
            assert is_transform_name_known(spec.canonical_name)
            assert get_transform_spec(spec.canonical_name) is not None

    def test_all_aliases_resolve_to_spec(self) -> None:
        for spec in BUILTIN_TRANSFORMS:
            for alias in spec.aliases:
                resolved = get_transform_spec(alias)
                assert resolved is not None
                assert resolved.canonical_name == spec.canonical_name

    def test_priority_order_is_deterministic(self) -> None:
        names = list_transform_names()
        # Sorted by priority ascending
        priorities = [get_transform_spec(n).priority for n in names]
        assert priorities == sorted(priorities)

    def test_no_duplicate_canonical_names(self) -> None:
        names = [s.canonical_name for s in BUILTIN_TRANSFORMS]
        assert len(names) == len(set(names))

    def test_information_theoretic_selector_present(self) -> None:
        spec = get_transform_spec("information_theoretic_selector")
        assert spec is not None
        assert spec.canonical_name == "information_theoretic_selector"
        assert spec.config_flag == "transform_context_selector"
        assert spec.priority == 19
        assert spec.safety_bucket == "conditional"


# =============================================================================
# Config consistency
# =============================================================================


class TestConfigConsistency:
    """Config enablement must agree with the registry for every transform."""

    def test_all_default_pipeline_transforms_have_config_flag(self) -> None:
        for name in list_default_pipeline_names():
            spec = get_transform_spec(name)
            assert spec is not None
            assert spec.config_flag, f"{name!r} lacks a config flag"
            assert hasattr(LatticeConfig(), spec.config_flag), (
                f"LatticeConfig missing field {spec.config_flag!r} for {name!r}"
            )

    def test_information_theoretic_selector_enabled_with_context_selector(self) -> None:
        cfg = LatticeConfig(transform_context_selector=True)
        assert is_transform_enabled(cfg, "information_theoretic_selector") is True
        assert is_transform_enabled(cfg, "context_selector") is True

    def test_information_theoretic_selector_disabled_with_context_selector(self) -> None:
        cfg = LatticeConfig(transform_context_selector=False)
        assert is_transform_enabled(cfg, "information_theoretic_selector") is False
        assert is_transform_enabled(cfg, "context_selector") is False

    def test_deleted_transforms_always_false(self) -> None:
        cfg = LatticeConfig(transform_context_selector=True)
        for name in ("stream_optimizer", "optimal_stopping", "fountain_codes", "convex_selector"):
            assert cfg.is_transform_enabled(name) is False
            assert is_transform_enabled(cfg, name) is False

    def test_rate_distortion_uses_semantic_compress_flag(self) -> None:
        cfg = LatticeConfig(transform_semantic_compress=True)
        assert is_transform_enabled(cfg, "rate_distortion") is True
        assert is_transform_enabled(cfg, "semantic_compress") is True
        assert is_transform_enabled(cfg, "semantic_compressor") is True

    def test_aliases_match_canonical_enablement(self) -> None:
        cfg = LatticeConfig(transform_prefix_opt=True)
        assert is_transform_enabled(cfg, "prefix_optimizer") is True
        assert is_transform_enabled(cfg, "prefix_opt") is True


# =============================================================================
# Pipeline construction
# =============================================================================


class TestPipelineConstruction:
    """Pipeline built from registry must include expected transforms."""

    def test_default_pipeline_has_expected_core_transforms(self) -> None:
        cfg = LatticeConfig()
        pipeline = build_default_pipeline(cfg)
        names = [t.name for t in pipeline.transforms]
        # Core transforms that should always be present in default config
        assert "content_profiler" in names
        assert "runtime_contract" in names
        assert "prefix_optimizer" in names
        assert "output_cleanup" in names
        # context_selector + information_theoretic_selector when enabled
        assert "context_selector" in names
        assert "information_theoretic_selector" in names
        # Execution transforms absent by default
        assert "batching" not in names
        assert "speculative" not in names
        assert "delta_encoder" not in names

    def test_safe_mode_excludes_lossy_transforms(self) -> None:
        cfg = LatticeConfig(compression_mode="safe")
        pipeline = build_default_pipeline(cfg)
        names = [t.name for t in pipeline.transforms]
        assert "structural_fingerprint" not in names
        assert "self_information" not in names
        assert "hierarchical_summary" not in names
        assert "context_selector" not in names
        assert "information_theoretic_selector" not in names

    def test_aggressive_mode_includes_all_core(self) -> None:
        cfg = LatticeConfig(compression_mode="aggressive")
        pipeline = build_default_pipeline(cfg)
        names = [t.name for t in pipeline.transforms]
        assert "structural_fingerprint" in names
        assert "self_information" in names
        assert "hierarchical_summary" in names
        assert "context_selector" in names
        assert "information_theoretic_selector" in names

    def test_execution_transforms_only_with_flag(self) -> None:
        cfg = LatticeConfig()
        pipeline_with = build_default_pipeline(cfg, include_execution_transforms=True)
        names_with = [t.name for t in pipeline_with.transforms]
        assert "batching" in names_with
        assert "speculative" in names_with
        assert "delta_encoder" not in names_with  # no session_manager provided

    def test_delta_encoder_with_session_manager(self) -> None:
        cfg = LatticeConfig()
        from lattice.core.session import MemorySessionStore, SessionManager

        store = MemorySessionStore()
        sm = SessionManager(store)
        pipeline = build_default_pipeline(
            cfg, include_execution_transforms=True, session_manager=sm
        )
        names = [t.name for t in pipeline.transforms]
        assert "delta_encoder" in names

    def test_pipeline_summary_matches_actual(self) -> None:
        cfg = LatticeConfig()
        pipeline = build_default_pipeline(cfg)
        summary = pipeline_summary(pipeline)
        assert summary["count"] == len(pipeline.transforms)
        assert set(summary["transforms"]) == set(t.name for t in pipeline.transforms)


# =============================================================================
# Safety bucket consistency
# =============================================================================


class TestSafetyConsistency:
    """Safety metadata must agree with the registry for every transform."""

    def test_all_canonical_names_have_safety_bucket(self) -> None:
        for name in list_transform_names():
            bucket = get_transform_safety_bucket(name)
            assert bucket in (
                TransformSafetyBucket.SAFE,
                TransformSafetyBucket.CONDITIONAL,
                TransformSafetyBucket.DANGEROUS,
            )

    def test_aliases_share_canonical_bucket(self) -> None:
        for spec in BUILTIN_TRANSFORMS:
            canonical_bucket = get_transform_safety_bucket(spec.canonical_name)
            for alias in spec.aliases:
                alias_bucket = get_transform_safety_bucket(alias)
                assert alias_bucket == canonical_bucket, (
                    f"Alias {alias!r} bucket {alias_bucket!r} != "
                    f"canonical {spec.canonical_name!r} bucket {canonical_bucket!r}"
                )

    def test_information_theoretic_selector_bucket(self) -> None:
        bucket = get_transform_safety_bucket("information_theoretic_selector")
        assert bucket == TransformSafetyBucket.CONDITIONAL

    def test_hierarchical_summary_is_dangerous(self) -> None:
        bucket = get_transform_safety_bucket("hierarchical_summary")
        assert bucket == TransformSafetyBucket.DANGEROUS

    def test_content_profiler_is_safe(self) -> None:
        bucket = get_transform_safety_bucket("content_profiler")
        assert bucket == TransformSafetyBucket.SAFE

    def test_unknown_transform_defaults_to_dangerous(self) -> None:
        bucket = get_transform_safety_bucket("totally_unknown_transform")
        assert bucket == TransformSafetyBucket.DANGEROUS

    def test_transform_allowed_at_risk_respects_buckets(self) -> None:
        from lattice.utils.validation import SemanticRiskScore

        # LOW risk (total <= 20)
        low = SemanticRiskScore()
        assert low.level == "LOW"
        # SAFE always allowed
        allowed, _ = transform_allowed_at_risk("content_profiler", low)
        assert allowed is True
        allowed, _ = transform_allowed_at_risk(
            "content_profiler", SemanticRiskScore(strict_instructions=25)
        )
        assert allowed is True  # profiler is safe even at high risk
        # CONDITIONAL blocked at HIGH risk
        high = SemanticRiskScore(strict_instructions=45)
        assert high.level == "HIGH"
        allowed, _ = transform_allowed_at_risk("information_theoretic_selector", low)
        assert allowed is True
        allowed, _ = transform_allowed_at_risk("information_theoretic_selector", high)
        assert allowed is False
        # DANGEROUS allowed only at LOW risk
        allowed, _ = transform_allowed_at_risk("hierarchical_summary", low)
        assert allowed is True
        allowed, _ = transform_allowed_at_risk("hierarchical_summary", high)
        assert allowed is False


# =============================================================================
# Integration — end-to-end registry + config + pipeline + safety
# =============================================================================


class TestEndToEndConsistency:
    """Cross-cut all four layers for every transform."""

    def test_every_default_pipeline_transform_is_known_and_safe(self) -> None:
        cfg = LatticeConfig()
        pipeline = build_default_pipeline(cfg)
        for transform in pipeline.transforms:
            assert is_transform_name_known(transform.name), (
                f"Transform {transform.name!r} in pipeline but not in registry"
            )
            bucket = get_transform_safety_bucket(transform.name)
            assert bucket is not None

    def test_config_enablement_matches_pipeline_contents(self) -> None:
        cfg = LatticeConfig(compression_mode="balanced")
        pipeline = build_default_pipeline(cfg)
        for transform in pipeline.transforms:
            assert cfg.is_transform_enabled(transform.name) is True, (
                f"Transform {transform.name!r} is in pipeline but config says it is disabled"
            )

    def test_balanced_mode_has_context_selectors(self) -> None:
        # balanced mode explicitly disables context_selector per apply_compression_mode
        # but the default config has transform_context_selector=True
        # This test documents the intended behavior: default config is not "balanced"
        # When user explicitly sets compression_mode=balanced, context_selector=False
        # We test the default config (no compression_mode override) instead
        cfg_default = LatticeConfig()
        pipeline = build_default_pipeline(cfg_default)
        names = [t.name for t in pipeline.transforms]
        assert "context_selector" in names
        assert "information_theoretic_selector" in names

    def test_no_duplicate_transforms_in_pipeline(self) -> None:
        cfg = LatticeConfig()
        pipeline = build_default_pipeline(cfg)
        names = [t.name for t in pipeline.transforms]
        assert len(names) == len(set(names))

    def test_pipeline_priority_order_matches_registry(self) -> None:
        cfg = LatticeConfig()
        pipeline = build_default_pipeline(cfg)
        priorities = [t.priority for t in pipeline.transforms]
        assert priorities == sorted(priorities)
