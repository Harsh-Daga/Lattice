"""Registry ↔ pipeline factory parity (Phase 12)."""

from __future__ import annotations

from lattice.pipeline._generated_factories import DEFAULT_TRANSFORM_FACTORIES
from lattice.transforms.registry import BUILTIN_TRANSFORMS, OPTIMIZER_SPECS


def test_every_default_pipeline_transform_has_factory_path() -> None:
    for spec in (*BUILTIN_TRANSFORMS, *OPTIMIZER_SPECS):
        if spec.default_pipeline:
            assert spec.factory_path, f"{spec.canonical_name} default_pipeline but no factory_path"


def test_generated_factories_cover_registry_factory_paths() -> None:
    for spec in (*BUILTIN_TRANSFORMS, *OPTIMIZER_SPECS):
        if not spec.factory_path or spec.execution_only:
            continue
        mod, _, cls = spec.factory_path.rpartition(".")
        assert spec.canonical_name in DEFAULT_TRANSFORM_FACTORIES, spec.canonical_name
        assert DEFAULT_TRANSFORM_FACTORIES[spec.canonical_name] == (mod, cls)
