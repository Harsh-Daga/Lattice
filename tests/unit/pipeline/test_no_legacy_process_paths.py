"""Phase 2 §4.3 / Phase 3 — IR-native transforms must not define process(request).

Legacy and execution-only transforms may keep process() until Phase 11.
"""

from __future__ import annotations

import inspect

import pytest

from lattice.pipeline.runner import Pipeline, PipelineTransformRegistry


@pytest.mark.parametrize("name", sorted(Pipeline._IR_NATIVE_TRANSFORMS))
def test_ir_native_registry_transform_has_no_process(name: str) -> None:
    registry = PipelineTransformRegistry()
    inst = registry.get(name)
    assert inst is not None, f"missing registry instance for {name!r}"
    cls = type(inst)
    assert "process" not in cls.__dict__, (
        f"{name} is IR-native but defines process() on {cls.__name__}; use optimize() only."
    )
    assert hasattr(cls, "optimize")
    assert inspect.isfunction(getattr(cls, "optimize")) or inspect.ismethod(
        getattr(cls, "optimize")
    )
