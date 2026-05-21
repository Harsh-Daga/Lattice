"""LATTICE optimizer layer — unified representation optimizers."""

from lattice.optimizer.context_optimizer import ContextOptimizer
from lattice.optimizer.diagnostic_optimizer import DiagnosticOptimizer
from lattice.optimizer.ir_structure_optimizer import IRStructureOptimizer
from lattice.optimizer.reference_optimizer import ReferenceOptimizer
from lattice.optimizer.structure_optimizer import StructureOptimizer
from lattice.optimizer.tool_optimizer import ToolOptimizer

# Canonical optimizer names and their classes
_OPTIMIZER_CLASSES: dict[str, type] = {
    "structure_optimizer": StructureOptimizer,
    "reference_optimizer": ReferenceOptimizer,
    "tool_optimizer": ToolOptimizer,
    "context_optimizer": ContextOptimizer,
    "diagnostic_optimizer": DiagnosticOptimizer,
    "ir_structure_optimizer": IRStructureOptimizer,
}

# Production path: always include content_profiler, runtime_contract
# Optimizers are gated by scheduler
PRODUCTION_OPTIMIZERS: tuple[str, ...] = (
    "structure_optimizer",
    "ir_structure_optimizer",
    "reference_optimizer",
    "tool_optimizer",
    "diagnostic_optimizer",
)

# Optional optimizers (gated by context length, task class, etc.)
CONDITIONAL_OPTIMIZERS: tuple[str, ...] = ("context_optimizer",)

__all__ = [
    "StructureOptimizer",
    "ReferenceOptimizer",
    "ToolOptimizer",
    "ContextOptimizer",
    "DiagnosticOptimizer",
    "_OPTIMIZER_CLASSES",
    "PRODUCTION_OPTIMIZERS",
    "CONDITIONAL_OPTIMIZERS",
]
