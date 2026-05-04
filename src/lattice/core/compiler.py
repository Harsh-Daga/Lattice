"""Prompt Compiler — orchestrates the IR pipeline before transform execution.

The compiler converts a Request through:
1. build_ir — parse messages into structured PromptIR
2. normalize_ir — canonicalize IR (JSON, tables, logs, constraints, causal chains)
3. Optimizations (applied by individual transforms reading IR metadata)
4. serialize — produce final LLM-readable text (no opaque placeholders)

This is a quality-preserving compiler, not a lossy compressor.
"""

from __future__ import annotations

from lattice.core.context import TransformContext
from lattice.core.ir import PromptIR
from lattice.core.ir_builder import build_ir
from lattice.core.ir_normalizer import normalize_ir
from lattice.core.ir_serializer import serialize_ir_to_text
from lattice.core.transport import Request


class PromptCompiler:
    """Compiles a Request into optimized, LLM-readable output via the IR pipeline."""

    def compile(self, request: Request, context: TransformContext) -> PromptIR:
        ir = build_ir(request)
        ir = normalize_ir(ir)
        _store_ir_metadata(request, ir)
        context.record_metric("compiler", "total_sections", len(ir.sections))
        context.record_metric("compiler", "total_spans", ir.total_spans)
        context.record_metric("compiler", "protected_spans", ir.protected_spans)
        context.record_metric("compiler", "compressible_spans", ir.compressible_spans)
        return ir

    @staticmethod
    def serialize(ir: PromptIR) -> str:
        return serialize_ir_to_text(ir)


def _store_ir_metadata(request: Request, ir: PromptIR) -> None:
    """Store IR summary in request metadata for scheduler and safety guards."""
    request.metadata["_lattice_ir_summary"] = ir.summary()
    request.metadata["_lattice_protected_spans"] = ir.protected_span_ids()
    request.metadata.setdefault("_lattice_ir", ir.to_dict())

    section_types = ir.section_types
    if "error" in section_types or "stack_trace" in section_types:
        request.metadata["_lattice_has_errors"] = True

    if ir.metadata.get("has_causal_chains"):
        request.metadata["_lattice_has_causal"] = True
        request.metadata["_lattice_causal_count"] = ir.metadata.get("causal_span_count", 0)


_compiler = PromptCompiler()


def get_compiler() -> PromptCompiler:
    return _compiler
