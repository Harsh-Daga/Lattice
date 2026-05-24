"""Rate-distortion style prompt compression with heuristic distortion scoring."""

from __future__ import annotations

import re

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.result import Ok, Result
from lattice.core.runtime_state import (
    get_canonical_state_value,
    get_ir_metadata_value,
    thaw_value,
)
from lattice.ir.primitives import PromptIRV2
from lattice.pipeline.base import ReversibleSyncTransform, TransformClass
from lattice.transport.types import Request, Response
from lattice.utils.validation import lossy_transform_allowed

_QUESTION_PATTERN = re.compile(r"\?")
_NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_HIGH_VALUE_KEYWORDS = frozenset(
    {
        "error",
        "exception",
        "critical",
        "important",
        "must",
        "required",
        "because",
        "therefore",
        "summary",
        "answer",
    }
)


class RateDistortionCompressor(ReversibleSyncTransform):
    """Heuristic rate-distortion optimizer for long natural-language inputs.

    This is a practical Phase D baseline: instead of a learned distortion model,
    it uses sentence-level heuristics to estimate distortion cost and performs
    budgeted removal with a greedy utility objective.
    """

    name = "rate_distortion"
    transform_class = TransformClass.SEMANTIC_LOSSY
    priority = 22

    def __init__(
        self,
        distortion_budget: float = 0.02,
        max_input_tokens: int = 200,
        min_sentences: int = 2,
    ) -> None:
        self.distortion_budget = max(0.0, min(1.0, distortion_budget))
        self.max_input_tokens = max_input_tokens
        self.min_sentences = max(1, min_sentences)

    # ------------------------------------------------------------------
    # IR-native optimize() — v2 path
    # ------------------------------------------------------------------

    def optimize(
        self,
        ir: PromptIRV2,
        _request: Request,
        context: TransformContext,
    ) -> Result[PromptIRV2, TransformError]:
        """IR-native: compress spans that are long natural-language (not structured)."""
        if not lossy_transform_allowed(_request):
            context.record_metric(self.name, "guarded", 1)
            return Ok(ir)

        compressed_spans = 0
        total_saved = 0
        new_sections = []

        for sec in ir.sections:
            new_spans = []
            for span in sec.spans:
                text = span.text
                if not text or span.protected or len(text) < self.max_input_tokens * 4:
                    new_spans.append(span)
                    continue
                if self._is_structured(text):
                    new_spans.append(span)
                    continue
                compressed = self._compress_text(text)
                if compressed != text:
                    new_spans.append(span.with_text(compressed))
                    compressed_spans += 1
                    total_saved += len(text) - len(compressed)
                else:
                    new_spans.append(span)
            new_sections.append(sec.with_spans(tuple(new_spans)))

        if compressed_spans > 0:
            context.record_metric(self.name, "spans_compressed", compressed_spans)
            context.record_metric(self.name, "tokens_saved_estimate", total_saved // 4)
            context.record_metric(self.name, "distortion_budget", self.distortion_budget)
        return Ok(ir.with_sections(tuple(new_sections)))

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        return response

    def _compress_text(self, text: str) -> str:
        sentences = self._split_sentences(text)
        if len(sentences) <= self.min_sentences:
            return text

        candidates: list[tuple[int, float, int, float]] = []
        for idx, sentence in enumerate(sentences):
            distortion = self._estimate_distortion(sentence, idx, len(sentences))
            savings = max(1, len(sentence) // 4)
            utility = savings / max(distortion, 1e-6)
            candidates.append((idx, distortion, savings, utility))

        max_remove = max(0, len(sentences) - self.min_sentences)
        removable = sorted(candidates, key=lambda x: x[3], reverse=True)

        removed: set[int] = set()
        used_distortion = 0.0
        for idx, distortion, _savings, _utility in removable:
            if len(removed) >= max_remove:
                break
            if used_distortion + distortion > self.distortion_budget:
                continue
            removed.add(idx)
            used_distortion += distortion

        if not removed:
            return text

        kept = [s for i, s in enumerate(sentences) if i not in removed]
        return " ".join(kept)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        pieces = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in pieces if p.strip()]

    @staticmethod
    def _is_structured(text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if stripped[0] in ("{", "["):
            return True
        if stripped.startswith("<") and ">" in stripped:
            return True
        if "```" in stripped:
            return True
        return bool("|" in stripped and "---" in stripped)

    @staticmethod
    def _estimate_distortion(sentence: str, index: int, total: int) -> float:
        lowered = sentence.lower()
        cost = 0.001
        if _QUESTION_PATTERN.search(sentence):
            cost += 0.02
        if _NUMBER_PATTERN.search(sentence):
            cost += 0.01
        if any(k in lowered for k in _HIGH_VALUE_KEYWORDS):
            cost += 0.02
        if index == 0 or index == total - 1:
            cost += 0.015
        return min(0.2, cost)


def _strategy(request: Request, context: TransformContext) -> dict[str, object]:
    ir_strategy = thaw_value(get_ir_metadata_value(context, "_lattice_strategy"))
    if isinstance(ir_strategy, dict):
        return ir_strategy
    strategy = get_canonical_state_value(context, "_lattice_strategy", {})
    if isinstance(strategy, dict):
        return strategy
    return {}
