"""Rate-distortion style prompt compression with heuristic distortion scoring."""

from __future__ import annotations

import re

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.transport import Request, Response
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

    def process(
        self,
        request: Request,
        context: TransformContext,
    ) -> Result[Request, TransformError]:
        strategy = request.metadata.get("_lattice_strategy", {})
        if isinstance(strategy, dict) and strategy.get("semantic_compress") is False:
            return Ok(request)
        if not lossy_transform_allowed(request):
            context.record_metric(self.name, "guarded", 1)
            return Ok(request)

        compressed_messages = 0
        total_saved_chars = 0

        for msg in request.messages:
            original = msg.content
            if not original or len(original) < self.max_input_tokens * 4:
                continue
            if self._is_structured(original):
                continue

            compressed = self._compress_text(original)
            if compressed != original:
                compressed_messages += 1
                total_saved_chars += len(original) - len(compressed)
                msg.content = compressed

        if compressed_messages > 0:
            context.record_metric(self.name, "messages_compressed", compressed_messages)
            context.record_metric(self.name, "tokens_saved_estimate", total_saved_chars // 4)
            context.record_metric(self.name, "distortion_budget", self.distortion_budget)
        return Ok(request)

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
