"""Tests for shared pipeline construction."""

from __future__ import annotations

from lattice.core.config import LatticeConfig
from lattice.pipeline.factory import build_default_pipeline, pipeline_summary
from lattice.pipeline.runner import Pipeline


def test_default_pipeline_returns_pipeline_with_lazy_registry() -> None:
    pipeline = build_default_pipeline(LatticeConfig())
    assert isinstance(pipeline, Pipeline)
    names = pipeline.registry.get_transform_names()
    # Lazy registry exposes the canonical transform names regardless of
    # config — actual enablement is decided by Pipeline.compress() gates.
    assert "runtime_contract" in names
    assert "prefix_optimizer" in names
    assert "content_profiler" in names


def test_default_pipeline_excludes_proxy_execution_transforms() -> None:
    # Execution-only transforms are registered post-build by the proxy
    # bootstrap; they are NOT in the default registry.
    names = build_default_pipeline(LatticeConfig()).registry.get_transform_names()
    assert "batching" not in names
    assert "speculative" not in names
    assert "delta_encoder" not in names


def test_register_instance_injects_execution_transform() -> None:
    pipeline = build_default_pipeline(LatticeConfig())

    class _Stub:
        name = "stub"

    pipeline.registry.register_instance("stub", _Stub())
    assert pipeline.registry.get("stub").__class__.__name__ == "_Stub"


def test_pipeline_summary_shape() -> None:
    pipeline = build_default_pipeline(LatticeConfig())
    summary = pipeline_summary(pipeline)
    assert summary["count"] == len(summary["transforms"])
    assert summary["runtime_contract_enabled"] is True
    assert "runtime_contract" in summary["core_transforms"]
    assert "reference_optimizer" in summary["optimizers"]
