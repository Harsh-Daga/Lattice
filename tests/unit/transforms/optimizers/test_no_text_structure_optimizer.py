"""Text-based StructureOptimizer was deleted in Phase 4; IRStructureOptimizer remains."""

from __future__ import annotations

import pytest


def test_text_structure_optimizer_deleted() -> None:
    from lattice.transforms.optimizers import _OPTIMIZER_CLASSES

    assert "structure_optimizer" not in _OPTIMIZER_CLASSES
    assert "ir_structure_optimizer" in _OPTIMIZER_CLASSES

    with pytest.raises(ImportError):
        from lattice.optimizer.structure_optimizer import StructureOptimizer  # noqa: F401

    with pytest.raises(ImportError):
        from lattice.optimizer import StructureOptimizer  # noqa: F401
