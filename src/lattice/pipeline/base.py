"""Foundational pipeline types — kept IR-free to break import cycles.

``ReversibleSyncTransform`` and ``TransformClass`` live here (not in
``pipeline.runner``) because ``pipeline.runner`` depends on
``lattice.ir`` and ``lattice.ir.native_optimizer`` in turn depends on
``ReversibleSyncTransform``. Putting the base class in this leaf module
keeps the dependency graph acyclic.
"""

from __future__ import annotations

import enum

from lattice.core.context import TransformContext
from lattice.transport.types import Request, Response


class TransformClass(enum.Enum):
    """Semantic classification of a transform's effect on content."""

    LOSSLESS_SAFE = "lossless_safe"
    LOSSLESS_CONTEXTUAL = "lossless_contextual"
    SEMANTIC_LOSSY = "semantic_lossy"
    STRUCTURAL_RISKY = "structural_risky"
    CACHE_ONLY = "cache_only"
    OBSERVABILITY_ONLY = "observability_only"


class ReversibleSyncTransform:
    """Base class for transforms in the LATTICE pipeline.

    The canonical entry point is ``optimize(ir, request, context) ->
    Result[PromptIRV2, ...]`` for IR-native transforms, with
    ``reverse(response, context)`` undoing the transform on the response
    side. Some non-IR-native transforms still implement
    ``process(request, context)``; :class:`Pipeline` dispatches to it
    when present and skips the transform otherwise.

    Phase 3 dropped the abstract ``process`` default: subclasses that
    need legacy behaviour implement it themselves, and the pipeline asks
    via ``hasattr(inst, "process")`` rather than relying on the base.
    """

    name: str = ""
    enabled: bool = True
    priority: int = 50
    transform_class: TransformClass = TransformClass.LOSSLESS_SAFE

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        """Default reverse: pass response through unchanged."""
        return response

    def can_process(self, _request: Request, _context: TransformContext) -> bool:
        """Default gate: enabled instances accept every request."""
        return self.enabled


__all__ = ["ReversibleSyncTransform", "TransformClass"]
