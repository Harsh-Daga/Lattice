"""Python API contract — pure import test.

The plan locks the public Python surface in REFACTOR_PLAN.md §2.4. The full
top-level shape (``from lattice import LatticeClient, ...``) lands in Phase 6;
until then the contract is split between ``lattice``, ``lattice.client``,
``lattice.sdk``, and ``lattice.core``. This test verifies the SURVIVING set
today and the methods promised on ``LatticeClient``.
"""

from __future__ import annotations

import importlib

import pytest


def test_lattice_version_exported() -> None:
    import lattice

    assert hasattr(lattice, "__version__")
    assert isinstance(lattice.__version__, str)
    assert lattice.__version__


def test_lattice_client_module_exports() -> None:
    """LatticeClient + CompressResult live in lattice.client today."""
    mod = importlib.import_module("lattice.client")
    for name in ("LatticeClient", "CompressResult"):
        assert hasattr(mod, name), f"lattice.client.{name} missing"


def test_lattice_sdk_exports() -> None:
    """LatticeClient/LatticeProxyClient/wrap_openai/wrap_anthropic via lattice.sdk."""
    sdk = importlib.import_module("lattice.sdk")
    for name in ("LatticeClient", "LatticeProxyClient", "wrap_openai", "wrap_anthropic"):
        assert hasattr(sdk, name), f"lattice.sdk.{name} missing"


def test_lattice_core_exports(api_surface) -> None:
    """Every symbol in api-surface.json#python_api.today.from_lattice_core resolves."""
    expected = api_surface["python_api"]["today"]["from_lattice_core"]
    core = importlib.import_module("lattice.core")
    missing = [n for n in expected if not hasattr(core, n)]
    assert not missing, f"missing from lattice.core: {missing}"


def test_lattice_client_has_promised_methods(api_surface) -> None:
    """REFACTOR_PLAN §2.4 lists LatticeClient methods that must stay."""
    from lattice.client import LatticeClient

    expected_methods = api_surface["python_api"]["lattice_client_methods"]
    missing = [m for m in expected_methods if not hasattr(LatticeClient, m)]
    assert not missing, f"LatticeClient missing methods: {missing}"


def test_planner_public_api() -> None:
    from lattice.planner import (
        ExecutionPlan,
        TaskClass,
        TaskClassification,
        UnifiedPlanner,
        build_execution_plan,
        classify_task,
    )

    assert callable(UnifiedPlanner)
    assert callable(build_execution_plan)
    assert callable(classify_task)
    assert ExecutionPlan is not None
    assert TaskClass is not None
    assert TaskClassification is not None


def test_runtime_public_api() -> None:
    from lattice.runtime import Tier, TierClassifier, TierDecision

    assert TierClassifier is not None
    assert Tier is not None
    assert TierDecision is not None


def test_transforms_optimizers_public_api() -> None:
    from lattice.transforms.optimizers import (
        ContextOptimizer,
        DiagnosticOptimizer,
        IRStructureOptimizer,
        ReferenceOptimizer,
        ToolOptimizer,
    )

    assert IRStructureOptimizer is not None
    assert ReferenceOptimizer is not None
    assert ToolOptimizer is not None
    assert DiagnosticOptimizer is not None
    assert ContextOptimizer is not None


def test_transforms_registry_public_api() -> None:
    from lattice.transforms.registry import (
        TransformSpec,
        get_transform_spec,
        is_response_side,
        list_transform_names,
    )

    assert TransformSpec is not None
    assert callable(get_transform_spec)
    assert callable(list_transform_names)
    assert callable(is_response_side)
    assert get_transform_spec("output_cleanup") is not None


def test_all_adapters_importable_at_top_level() -> None:
    from lattice.providers import (
        AnthropicAdapter,
        OpenAIAdapter,
    )

    assert OpenAIAdapter is not None
    assert AnthropicAdapter is not None


# --- Phase 6 placeholder tests ---------------------------------------------
# These will become PASSING tests after Phase 6 hoists symbols to top level.
# Until then they xfail to document the target surface without blocking CI.


@pytest.mark.xfail(
    reason="Phase 6 hoists LatticeClient et al. to top-level lattice.*", strict=False
)
def test_lattice_toplevel_target_imports() -> None:
    from lattice import (  # noqa: F401
        CompressResult,
        LatticeClient,
        LatticeProxyClient,
        wrap_openai_client,
    )
