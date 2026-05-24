"""Transform orchestrators — each runs 2-3 underlying transforms and selects the best variant.

These are *transforms-on-transforms*: they're registered like normal transforms but their
optimize() method invokes other transforms and chooses the winner by quality + cost +
risk scoring (see lattice.ir.quality, lattice.ir.validation).

Registered for use by pipeline.representation_optimizer.RepresentationOptimizer
(beam search) which decides which orchestrators to invoke per request.
"""

from lattice.transforms.optimizers.context_optimizer import ContextOptimizer
from lattice.transforms.optimizers.diagnostic_optimizer import DiagnosticOptimizer
from lattice.transforms.optimizers.ir_structure_optimizer import IRStructureOptimizer
from lattice.transforms.optimizers.reference_optimizer import ReferenceOptimizer
from lattice.transforms.optimizers.tool_optimizer import ToolOptimizer

# Registry consumed by pipeline.representation_optimizer:
_OPTIMIZER_CLASSES: dict[str, type] = {
    "ir_structure_optimizer": IRStructureOptimizer,
    "reference_optimizer": ReferenceOptimizer,
    "tool_optimizer": ToolOptimizer,
    "diagnostic_optimizer": DiagnosticOptimizer,
    "context_optimizer": ContextOptimizer,
}

# Production-default tuple (lossless first):
PRODUCTION_OPTIMIZERS: tuple[str, ...] = (
    "ir_structure_optimizer",
    "reference_optimizer",
    "tool_optimizer",
    "diagnostic_optimizer",
)
# Lossy / conditional:
CONDITIONAL_OPTIMIZERS: tuple[str, ...] = ("context_optimizer",)

__all__ = [
    "IRStructureOptimizer",
    "ReferenceOptimizer",
    "ToolOptimizer",
    "DiagnosticOptimizer",
    "ContextOptimizer",
    "_OPTIMIZER_CLASSES",
    "PRODUCTION_OPTIMIZERS",
    "CONDITIONAL_OPTIMIZERS",
]
