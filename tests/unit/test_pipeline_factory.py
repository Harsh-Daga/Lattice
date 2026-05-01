"""Tests for shared pipeline construction."""

from __future__ import annotations

from lattice.core.config import LatticeConfig
from lattice.core.pipeline_factory import build_default_pipeline, pipeline_summary
from lattice.core.session import MemorySessionStore, SessionManager


def _names(config: LatticeConfig, *, execution: bool = False) -> list[str]:
    session_manager = None
    if execution:
        session_manager = SessionManager(MemorySessionStore())
    pipeline = build_default_pipeline(
        config,
        include_execution_transforms=execution,
        session_manager=session_manager,
    )
    return [t.name for t in pipeline.transforms]


def test_default_pipeline_contains_runtime_contract_before_strategy_selector() -> None:
    names = _names(LatticeConfig())
    assert "runtime_contract" in names
    assert "strategy_selector" in names
    assert names.index("runtime_contract") < names.index("strategy_selector")


def test_default_pipeline_excludes_proxy_execution_transforms() -> None:
    names = _names(LatticeConfig())
    assert "batching" not in names
    assert "speculative" not in names
    assert "delta_encoder" not in names


def test_proxy_pipeline_includes_execution_transforms() -> None:
    names = _names(LatticeConfig(), execution=True)
    assert "batching" in names
    assert "speculative" in names
    assert "delta_encoder" in names


def test_config_disables_runtime_contract_in_factory() -> None:
    names = _names(LatticeConfig(transform_runtime_contract=False))
    assert "runtime_contract" not in names


def test_pipeline_summary_splits_core_and_execution_transforms() -> None:
    session_manager = SessionManager(MemorySessionStore())
    pipeline = build_default_pipeline(
        LatticeConfig(),
        include_execution_transforms=True,
        session_manager=session_manager,
    )
    summary = pipeline_summary(pipeline)
    assert summary["count"] == len(summary["transforms"])
    assert summary["runtime_contract_enabled"] is True
    assert summary["strategy_selector_enabled"] is True
    assert "runtime_contract" in summary["core_transforms"]
    assert {"batching", "speculative", "delta_encoder"}.issubset(
        set(summary["execution_transforms"])
    )
