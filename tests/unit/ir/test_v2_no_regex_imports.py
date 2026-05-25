"""Verify deleted transform modules are no longer importable.

After deletion, these modules raise ModuleNotFoundError as expected.
"""

from __future__ import annotations

import pytest


class TestDeletedTransformsCannotBeImported:
    """Deleted legacy transforms must raise ModuleNotFoundError."""

    @pytest.mark.parametrize(
        "mod_path",
        [
            "lattice.transforms.alias_manifest",
            "lattice.transforms.dictionary_compress",
            "lattice.transforms.grammar_compress",
        ],
    )
    def test_module_not_found(self, mod_path: str) -> None:
        with pytest.raises(ModuleNotFoundError):
            __import__(mod_path)

    def test_optimizer_modules_importable(self) -> None:
        from lattice.transforms.optimizers.ir_structure_optimizer import IRStructureOptimizer
        from lattice.transforms.optimizers.reference_optimizer import ReferenceOptimizer

        ir_opt = IRStructureOptimizer()
        ref_opt = ReferenceOptimizer()
        assert ir_opt.name == "ir_structure_optimizer"
        assert ref_opt.name == "reference_optimizer"
