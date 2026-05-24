"""Phase 5b — prefix_opt and constraint_lifting must not exist on disk or in registry."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from lattice.transforms.registry import get_transform_spec


def test_prefix_opt_module_removed() -> None:
    root = Path(__file__).resolve().parents[3] / "src" / "lattice" / "transforms"
    assert not (root / "prefix_opt.py").exists()


def test_constraint_lifting_module_removed() -> None:
    root = Path(__file__).resolve().parents[3] / "src" / "lattice" / "transforms"
    assert not (root / "constraint_lifting.py").exists()


def test_registry_has_no_prefix_or_constraint_specs() -> None:
    assert get_transform_spec("prefix_optimizer") is None
    assert get_transform_spec("prefix_opt") is None
    assert get_transform_spec("constraint_lifting") is None


def test_prefix_opt_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("lattice.transforms.prefix_opt")
