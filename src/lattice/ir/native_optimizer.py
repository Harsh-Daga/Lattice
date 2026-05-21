"""optimizer/ir_native_optimizer.py — Base class for IR-native optimizers.

Phase 4 — IR-native optimization.

Why this exists:
- Text-based optimizers operate on raw strings using regex — fragile and error-prone
- IR-native optimizers operate on typed PromptIR nodes (Section/Span) — structural and safe
- The IR carries: protection flags, compression flags, structured metadata from the normalizer
- The serializer renders IR back to LLM-readable text

Usage:
    class MyOptimizer(IRNativeOptimizer):
        can_process_sections = {SectionType.JSON, SectionType.TABLE}

        def optimize_ir(self, ir, request, context):
            for section in ir.sections:
                if section.type not in self.can_process_sections:
                    continue
                for span in section.spans:
                    if span.protected:
                        continue
                    span.text = self._do_something(span)
            return Ok(ir)

        def _do_something(self, span):
            return span.text.replace("old", "new")
"""
from __future__ import annotations

from typing import Any

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.runtime_state import get_canonical_request_value
from lattice.core.transport import Request, Response
from lattice.ir.primitives import PromptIRV2, prompt_ir_from_v2
from lattice.ir.types import PromptIR, SectionType


class IRNativeOptimizer(ReversibleSyncTransform):
    """Base class for IR-native optimizers.

    Provides:
      - IR extraction from request metadata or fresh compilation
      - Section-aware iteration (skip protected, respect compression flags)
      - Serialization back to Request messages

    Subclasses set `can_process_sections` and override `optimize_ir`.
    """

    name = "ir_native_optimizer"
    priority = 20
    # Which section types this optimizer acts on (empty = all)
    can_process_sections: set[SectionType] = set()

    def __init__(self) -> None:
        self._compiler: Any = None

    def can_process(self, _request: Request, context: TransformContext) -> bool:
        """Check whether an IR can be obtained for this request."""
        ir = self._get_ir(_request, context)
        if ir is None:
            return False
        return True

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        """Entrypoint: compile/get IR, call optimize_ir, serialize back."""
        from lattice.ir.serializer import serialize_ir_to_text

        ir = self._get_ir(request, context)
        if ir is None:
            return Ok(request)

        # Check if this optimizer should act on any section of this IR
        if not self._has_processable_span(ir):
            return Ok(request)

        original_text = serialize_ir_to_text(ir)

        try:
            result = self.optimize_ir(ir, request, context)
        except Exception:
            return Ok(request)

        match result:
            case Ok(modified_ir):
                modified_text = serialize_ir_to_text(modified_ir)
                if modified_text == original_text:
                    # No changes
                    return Ok(request)

                # Serialize modified IR back to a request copy
                new_request = self._serialize_to_request(modified_ir, request)
                return Ok(new_request)
            case _:
                return Ok(request)

    def optimize_ir(
        self, ir: PromptIR, request: Request, context: TransformContext
    ) -> Result[PromptIR, TransformError]:
        """Override this to mutate spans in *ir* in place.

        Rules:
          - Only modify spans with span.protected == False
          - Only modify spans with span.compressible == True (optional)
          - Mutate span.text and optionally span.structure/span.metadata
          - Return Ok(ir) even if no changes were made (caller checks identity)
        """
        raise NotImplementedError

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        """No-op — IR-native transforms are lossless."""
        return response

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_ir(self, request: Request, context: TransformContext) -> PromptIR | None:
        """Return IR from request metadata or compile fresh."""
        # 1. Stored in request metadata (canonical v2 IR or legacy fallback)
        ir_v2 = get_canonical_request_value(request, context, "_lattice_ir_v2")
        if ir_v2 is not None:
            if isinstance(ir_v2, dict):
                try:
                    ir_v2 = PromptIRV2.from_dict(ir_v2)
                except Exception:
                    ir_v2 = None
            if ir_v2 is not None:
                try:
                    return prompt_ir_from_v2(ir_v2)
                except Exception:
                    pass

        # 2. Stored in request metadata (legacy compiler output)
        ir_dict = get_canonical_request_value(request, context, "_lattice_ir")
        if ir_dict is not None:
            return self._reconstruct_ir(ir_dict)

        # 3. Stored in session state
        stored = get_canonical_request_value(request, context, "_lattice_ir")
        if stored is not None:
            return stored

        # 4. Compile fresh (lazy)
        try:
            from lattice.ir.builder import build_ir
            from lattice.ir.normalizer import normalize_ir
            ir = normalize_ir(build_ir(request))
            # Cache in session state for reuse by other IR-native optimizers
            context.session_state["_lattice_ir"] = ir
            return ir
        except Exception:
            return None

    def _has_processable_span(self, ir: PromptIR) -> bool:
        """Return True if IR contains at least one non-protected, matching span."""
        for section in ir.sections:
            if self.can_process_sections and section.type not in self.can_process_sections:
                continue
            for span in section.spans:
                if not span.protected:
                    return True
        return False

    def _serialize_to_request(self, ir: PromptIR, original: Request) -> Request:
        """Serialize modified IR into a fresh copy of the Request."""
        from lattice.ir.serializer import serialize_ir_to_text

        text = serialize_ir_to_text(ir)
        request = original.copy()

        # Replace the *last* user-visible message with the serialized text
        user_msgs = [m for m in request.messages if m.role == "user"]
        if user_msgs:
            last_user = user_msgs[-1]
            last_user.content = text
        else:
            # Create a new user message if none exists
            from lattice.core.transport import Message
            request.messages.append(
                Message(role="user", content=text)
            )

        # Record that we produced IR in transport so downstream
        # optimizers or the proxy can see it
        request.metadata["_lattice_ir_native_applied"] = self.name
        return request

    def _reconstruct_ir(self, data: dict[str, Any]) -> PromptIR:
        """Reconstruct a PromptIR from its `to_dict()` output."""
        return PromptIrLoader.from_dict(data)


# ------------------------------------------------------------------
# PromptIR reconstruction helper
# ------------------------------------------------------------------


class PromptIrLoader:
    """Static helper for reconstructing PromptIR from nested dicts."""

    @staticmethod
    def from_dict(data: dict[str, Any]) -> PromptIR:
        from lattice.ir.types import Section, SectionType, Span, SpanRole

        sections: list[Section] = []
        for sec_data in data.get("sections", []):
            spans: list[Span] = []
            for sp_data in sec_data.get("spans", []):
                spans.append(
                    Span(
                        span_id=sp_data.get("span_id", "0"),
                        text=sp_data.get("text", ""),
                        role=SpanRole(sp_data.get("role", "data")),
                        section_type=SectionType(sec_data.get("type", "context")),
                        entities=sp_data.get("entities", []),
                        numbers=sp_data.get("numbers", []),
                        keys=sp_data.get("keys", []),
                        structure=sp_data.get("structure", {}),
                        protected=sp_data.get("protected", False),
                        compressible=sp_data.get("compressible", False),
                        compression_modes_allowed=sp_data.get(
                            "compression_modes_allowed", []
                        ),
                        metadata=sp_data.get("metadata", {}),
                    )
                )
            sections.append(
                Section(
                    type=SectionType(sec_data.get("type", "context")),
                    spans=spans,
                    metadata=sec_data.get("metadata", {}),
                )
            )
        return PromptIR(sections=sections, metadata=data.get("metadata", {}))
