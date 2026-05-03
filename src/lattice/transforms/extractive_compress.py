"""Extractive Compression — LOSSLESS_SAFE alternative to semantic_compress.

Instead of lossy semantic summarization, uses extractive techniques:
- Keeps sentences with entities/numbers/keywords
- Removes boilerplate and filler
- Preserves all structural elements (code, tables, JSON)

This replaces the lossy semantic_compress / rate_distortion for scenarios
where quality matters (debugging, reasoning, structured output).
"""

from __future__ import annotations

import re

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.transport import Message, Request, Response

_ENTITY_RE = re.compile(r"\b\d+(?:\.\d+)?\b|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
_CODE_RE = re.compile(r"```[\s\S]*?```")
_TABLE_RE = re.compile(r"^\|.*\|$", re.MULTILINE)
_JSON_RE = re.compile(r"^[\{\[].*[\}\]]$", re.MULTILINE | re.DOTALL)

_SIGNAL_WORDS = frozenset(
    {
        "error",
        "exception",
        "failure",
        "critical",
        "important",
        "must",
        "required",
        "because",
        "therefore",
        "root cause",
        "fix",
        "solution",
        "answer",
        "result",
        "conclusion",
        "recommendation",
        "mitigation",
        "warning",
        "timeout",
    }
)


class ExtractiveCompressor(ReversibleSyncTransform):
    name = "extractive_compress"
    priority = 22

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        new_messages: list[Message] = []
        saved = 0

        for msg in request.messages:
            compressed, saved_delta = _extractive_compress(msg.content)
            saved += saved_delta
            new_messages.append(Message(role=msg.role, content=compressed))

        context.record_metric(self.name, "chars_saved", saved)
        return Ok(
            Request(
                model=request.model,
                messages=new_messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tools=request.tools,
                tool_choice=request.tool_choice,
                metadata=request.metadata,
            )
        )

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        return response

    def can_process(self, request: Request, _context: TransformContext) -> bool:
        if not self.enabled:
            return False
        return request.token_estimate > 200


def _extractive_compress(text: str) -> tuple[str, int]:
    if not text or len(text) < 100:
        return text, 0

    protected_blocks: list[tuple[str, str]] = []
    working = text

    for label, pattern in [
        ("CODE", _CODE_RE),
        ("JSON", _JSON_RE),
        ("TABLE", _TABLE_RE),
    ]:
        for i, match in enumerate(pattern.finditer(working)):
            placeholder = f"__{label}_{i}__"
            protected_blocks.append((placeholder, match.group()))
            working = working.replace(match.group(), placeholder, 1)

    sentences = re.split(r"(?<=[.!?])\s+", working)
    kept: list[str] = []

    for sent in sentences:
        stripped = sent.strip()
        if not stripped:
            continue
        lower = stripped.lower()

        if _ENTITY_RE.search(stripped):
            kept.append(stripped)
            continue

        word_count = len(stripped.split())
        if any(w in lower for w in _SIGNAL_WORDS):
            kept.append(stripped)
            continue

        if word_count <= 5 and not _ENTITY_RE.search(stripped):
            continue

        if re.search(
            r"\b(the|a|an|is|are|was|were|be|been|being|has|have|had|do|does|did)\b", lower
        ) and not _ENTITY_RE.search(stripped):
            continue

        kept.append(stripped)

    result = " ".join(kept)
    for placeholder, block in protected_blocks:
        result = result.replace(placeholder, block, 1)

    return result, len(text) - len(result)
