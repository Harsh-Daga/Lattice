"""Registry factory_path entries must resolve to importable transform classes."""

from __future__ import annotations

import importlib

import pytest

from lattice.transforms.registry import BUILTIN_TRANSFORMS, OPTIMIZER_SPECS, TransformSpec


def _load_class(spec: TransformSpec) -> type:
    if not spec.factory_path:
        pytest.skip(f"{spec.canonical_name} has no factory_path (manual construction)")
    module_name, class_name = spec.factory_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


@pytest.mark.parametrize(
    "spec",
    [*BUILTIN_TRANSFORMS, *OPTIMIZER_SPECS],
    ids=lambda s: s.canonical_name,
)
def test_factory_path_imports(spec: TransformSpec) -> None:
    cls = _load_class(spec)
    assert cls is not None
