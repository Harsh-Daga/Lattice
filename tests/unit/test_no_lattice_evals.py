"""Phase 10 — dead lattice.evals package must not exist."""

from __future__ import annotations

import importlib

import pytest


def test_lattice_evals_deleted() -> None:
    with pytest.raises(ImportError):
        importlib.import_module("lattice.evals")
