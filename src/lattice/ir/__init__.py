"""LATTICE Intermediate Representation (IR) — the spine of the system.

Consumers import everything IR-related from this single entry point:

    from lattice.ir import (
        # types (v1, mutable dataclasses)
        SectionType, SpanRole, Span, Section, PromptIR,
        # primitives (v2, immutable / frozen)
        SpanV2, SectionV2, PromptIRV2,
        Candidate, CandidateGraph, CandidateScore,
        CachePlan, TransportPlan, ExecutionPlan, ExecutionNode,
        prompt_ir_v2_from_legacy, prompt_ir_from_v2,
        freeze_dict, thaw_dict,
        # construction & rendering
        build_ir, normalize_ir, serialize_ir_to_text,
        is_repeated_template,
        normalize_json_sections, normalize_table_sections, normalize_log_sections,
        lift_constraints, extract_causal_chains,
        # transform protocol
        IRTransform, LegacyRequestTransformAdapter,
        CandidateScorer, CandidateSearch,
        # native optimiser base
        IRNativeOptimizer, PromptIrLoader,
        # validation & quality
        ValidationResult,
        validate_candidate, validate_request_candidate, validate_beam_candidate,
        QualityEstimate,
        estimate_quality, estimate_cache_gain, estimate_transport_gain, estimate_semantic_risk,
        # semantic graph
        SemanticSpan, SemanticEdge, SemanticImportanceGraph,
    )
"""

from lattice.ir.builder import build_ir, is_repeated_template
from lattice.ir.native_optimizer import IRNativeOptimizer, PromptIrLoader
from lattice.ir.normalizer import (
    extract_causal_chains,
    lift_constraints,
    normalize_ir,
    normalize_json_sections,
    normalize_log_sections,
    normalize_table_sections,
)
from lattice.ir.primitives import (
    CachePlan,
    Candidate,
    CandidateGraph,
    CandidateScore,
    ExecutionNode,
    ExecutionPlan,
    PromptIRV2,
    SectionV2,
    SpanV2,
    TransportPlan,
    freeze_dict,
    prompt_ir_from_v2,
    prompt_ir_v2_from_legacy,
    thaw_dict,
)
from lattice.ir.quality import (
    QualityEstimate,
    estimate_cache_gain,
    estimate_quality,
    estimate_semantic_risk,
    estimate_transport_gain,
)
from lattice.ir.semantic_graph import SemanticEdge, SemanticImportanceGraph, SemanticSpan
from lattice.ir.serializer import serialize_ir_to_text
from lattice.ir.transform import (
    CandidateScorer,
    CandidateSearch,
    IRTransform,
    LegacyRequestTransformAdapter,
)
from lattice.ir.types import PromptIR, Section, SectionType, Span, SpanRole
from lattice.ir.validation import (
    ValidationResult,
    validate_beam_candidate,
    validate_candidate,
    validate_request_candidate,
)

__all__ = [
    # types (v1)
    "SectionType",
    "SpanRole",
    "Span",
    "Section",
    "PromptIR",
    # primitives (v2)
    "SpanV2",
    "SectionV2",
    "PromptIRV2",
    "Candidate",
    "CandidateGraph",
    "CandidateScore",
    "CachePlan",
    "TransportPlan",
    "ExecutionPlan",
    "ExecutionNode",
    "prompt_ir_v2_from_legacy",
    "prompt_ir_from_v2",
    "freeze_dict",
    "thaw_dict",
    # construction & rendering
    "build_ir",
    "normalize_ir",
    "serialize_ir_to_text",
    "is_repeated_template",
    "normalize_json_sections",
    "normalize_table_sections",
    "normalize_log_sections",
    "lift_constraints",
    "extract_causal_chains",
    # transform protocol
    "IRTransform",
    "LegacyRequestTransformAdapter",
    "CandidateScorer",
    "CandidateSearch",
    # native optimiser base
    "IRNativeOptimizer",
    "PromptIrLoader",
    # validation & quality
    "ValidationResult",
    "validate_candidate",
    "validate_request_candidate",
    "validate_beam_candidate",
    "QualityEstimate",
    "estimate_quality",
    "estimate_cache_gain",
    "estimate_transport_gain",
    "estimate_semantic_risk",
    # semantic graph
    "SemanticSpan",
    "SemanticEdge",
    "SemanticImportanceGraph",
]
