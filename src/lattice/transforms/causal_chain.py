"""Causal Chain Extraction — OBSERVABILITY_ONLY quality transform.

Detects cause-effect relationships in prompts and makes them explicit
by extracting causal graphs from diagnostic, debugging, and reasoning text.

This is a quality transform — it preserves inference structure rather
than compressing. The extracted chains are serialized as readable
CAUSAL GRAPH blocks that the LLM can reason over directly.

Typical use cases:
- Debugging: service A failed → B timeout → C retry storm
- Root cause analysis: module not found → build failed → CI blocked
- Incident reports: deployment error → config mismatch → rollback needed
"""

from __future__ import annotations

import re

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform, TransformClass
from lattice.core.result import Ok, Result
from lattice.core.transport import Request, Response
from lattice.ir.primitives import PromptIRV2

_CAUSAL_EXPLICIT = re.compile(
    r"(.+?)\s+(?:caused|causes|causing|triggered|triggers|triggering"
    r"|led\s+to|leads\s+to|leading\s+to|resulted\s+in|results\s+in"
    r"|resulting\s+in)\s+(.+?)(?:[.,;]|$)",
    re.IGNORECASE,
)
_CAUSAL_IMPLICIT = re.compile(
    r"(?:because|due\s+to|as\s+a\s+result\s+of|on\s+account\s+of)\s+(.+?)"
    r"[.,;]\s*(.+?)(?:[.,;]|$)",
    re.IGNORECASE,
)
_CHAIN_LINK = re.compile(
    r"(.+?)\s+(?:->|→|-->|—>|==>|=>|then|which|this|that)\s+(.+?)(?:[.,;]|$)",
    re.IGNORECASE,
)
_ERROR_CAUSE = re.compile(
    r"(\w+(?:Error|Exception|Failure|Fault|Crash|Timeout|Panic|Abort|OOM"
    r"|Segfault))\s+(?:caused|triggered|due\s+to|because)\s+(.+?)(?:[.,;]|$)",
    re.IGNORECASE,
)


class CausalChainExtractor(ReversibleSyncTransform):
    name = "causal_chain"
    priority = 9
    transform_class = TransformClass.OBSERVABILITY_ONLY

    # ------------------------------------------------------------------
    # IR-native optimize() — v2 path
    # ------------------------------------------------------------------

    def optimize(
        self,
        ir: PromptIRV2,
        _request: Request,
        context: TransformContext,
    ) -> Result[PromptIRV2, TransformError]:
        """IR-native: extract causal chains from span text and prepend CAUSAL GRAPH block."""
        chains_found = 0
        new_sections = []
        for sec in ir.sections:
            new_spans = []
            for span in sec.spans:
                if len(span.text) < 30 or span.protected:
                    new_spans.append(span)
                    continue
                has_causal = (
                    _CAUSAL_EXPLICIT.search(span.text)
                    or _CAUSAL_IMPLICIT.search(span.text)
                    or _ERROR_CAUSE.search(span.text)
                )
                if not has_causal:
                    new_spans.append(span)
                    continue
                chains = _extract_chains(span.text)
                if not chains or _has_chains_in_output(span.text, chains):
                    new_spans.append(span)
                    continue
                annotated = _format_chain_output(span.text, chains)
                if annotated != span.text:
                    new_spans.append(span.with_text(annotated))
                    chains_found += 1
                else:
                    new_spans.append(span)
            new_sections.append(sec.with_spans(tuple(new_spans)))
        context.record_metric(self.name, "spans_annotated", chains_found)
        return Ok(ir.with_sections(tuple(new_sections)))

    # ------------------------------------------------------------------
    # Legacy process()
    # ------------------------------------------------------------------

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        chains_found = 0
        for msg in request.messages:
            if msg.role not in ("user", "assistant", "system"):
                continue
            content = msg.content
            if not content or len(content) < 30:
                continue
            has_causal = (
                _CAUSAL_EXPLICIT.search(content)
                or _CAUSAL_IMPLICIT.search(content)
                or _ERROR_CAUSE.search(content)
            )
            if not has_causal:
                continue
            chains = _extract_chains(content)
            if not chains:
                continue
            if _has_chains_in_output(content, chains):
                continue
            annotated = _format_chain_output(content, chains)
            msg.content = annotated
            chains_found += 1

        context.record_metric(self.name, "chains_extracted", chains_found)
        return Ok(request)

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        return response

    def can_process(self, request: Request, _context: TransformContext) -> bool:
        if not self.enabled:
            return False
        combined = "\n".join(m.content for m in request.messages)
        return (
            bool(_CAUSAL_EXPLICIT.search(combined)) or bool(_ERROR_CAUSE.search(combined))
        ) and len(combined) > 100


def _extract_chains(text: str) -> list[dict[str, str]]:
    chains: list[dict[str, str]] = []
    seen: set[str] = set()

    for pattern in (_CAUSAL_EXPLICIT, _CAUSAL_IMPLICIT, _CHAIN_LINK, _ERROR_CAUSE):
        for m in pattern.finditer(text):
            groups = m.groups()
            if len(groups) == 2:
                cause = groups[0].strip().rstrip(".,;")
                effect = groups[1].strip().rstrip(".,;")
                if cause and effect and len(cause) > 3 and len(effect) > 3:
                    key = f"{cause[:30]}|{effect[:30]}"
                    if key not in seen:
                        seen.add(key)
                        chains.append({"cause": cause, "effect": effect})
            elif len(groups) == 1:
                cause = groups[0].strip().rstrip(".,;")
                if cause and len(cause) > 3:
                    key = cause[:30]
                    if key not in seen:
                        seen.add(key)
                        chains.append({"cause": cause, "effect": "unknown"})

    return chains


def _has_chains_in_output(text: str, chains: list[dict[str, str]]) -> bool:
    if not chains:
        return False
    return "CAUSAL GRAPH" in text and any(
        chain["cause"][:20] in text.split("CAUSAL GRAPH")[-1] for chain in chains
    )


def _format_chain_output(text: str, chains: list[dict[str, str]]) -> str:
    lines = ["CAUSAL GRAPH:"]
    for i, chain in enumerate(chains):
        lines.append(f"  {_shorten(chain['cause'])}")
        lines.append(f"    → {_shorten(chain['effect'])}")
        if i < len(chains) - 1:
            lines.append("")

    graph_block = "\n".join(lines)
    if "CAUSAL GRAPH" in text:
        return text

    return f"{graph_block}\n\n{text}"


def _shorten(text: str) -> str:
    if len(text) <= 60:
        return text
    return text[:57] + "..."
