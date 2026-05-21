"""Verify deleted transform modules are no longer importable.

After deletion, these modules raise ModuleNotFoundError as expected.
"""
from __future__ import annotations

import pytest


class TestDeletedTransformsCannotBeImported:
    """Deleted legacy transforms must raise ModuleNotFoundError."""

    @pytest.mark.parametrize("mod_path", [
        "lattice.transforms.alias_manifest",
        "lattice.transforms.dictionary_compress",
        "lattice.transforms.grammar_compress",
    ])
    def test_module_not_found(self, mod_path: str) -> None:
        with pytest.raises(ModuleNotFoundError):
            __import__(mod_path)

    def test_optimizer_modules_use_try_except_safely(self) -> None:
        from lattice.optimizer.reference_optimizer import ReferenceOptimizer
        from lattice.optimizer.structure_optimizer import StructureOptimizer

        so = StructureOptimizer()
        ro = ReferenceOptimizer()
        assert so.name == "structure_optimizer"
        assert ro.name == "reference_optimizer"
