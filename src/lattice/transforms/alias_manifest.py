"""Reversible Alias Manifest — LOSSLESS_CONTEXTUAL.

Replaces repeated long literals with human-readable aliases (A1, A2, ...)
and prepends an ALIAS MAP so the model can resolve them. Opaque placeholders
like ``<d_36>`` are forbidden — only manifest-declared aliases are emitted.

Reversible: alias map is saved in session_state and resolved during reverse().
"""

from __future__ import annotations

import re
from collections import Counter

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.transport import Message, Request, Response

_MIN_LITERAL_LEN = 30
_MIN_OCCURRENCES = 3
_MAX_ALIASES = 30
_ALIAS_PREFIX = "A"


class AliasManifestTransform(ReversibleSyncTransform):
    name = "alias_manifest"
    priority = 16

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        long_literals = _find_repeated_literals(request)
        if not long_literals:
            return Ok(request)

        alias_map: dict[str, str] = {}
        for i, literal in enumerate(long_literals, start=1):
            alias_map[f"{_ALIAS_PREFIX}{i}"] = literal

        manifest_lines = ["ALIAS MAP:"]
        for alias, literal in alias_map.items():
            manifest_lines.append(f"{alias} = {literal!r}")
        manifest_lines.append("")
        manifest_header = "\n".join(manifest_lines) + "\nDATA:\n"

        new_messages: list[Message] = []
        for msg in request.messages:
            content = msg.content
            for alias, literal in alias_map.items():
                content = content.replace(literal, alias)
            new_messages.append(Message(role=msg.role, content=content))

        state = context.get_transform_state(self.name)
        state["alias_map"] = alias_map
        state["manifest_header"] = manifest_header

        context.record_metric(self.name, "aliases_created", len(alias_map))
        context.record_metric(
            self.name,
            "chars_saved",
            sum(len(v) - len(k) for k, v in alias_map.items()),
        )

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

    def reverse(self, response: Response, context: TransformContext) -> Response:
        state = context.get_transform_state(self.name)
        alias_map: dict[str, str] = state.get("alias_map", {})
        if not alias_map:
            return response

        content = response.content
        for alias, literal in reversed(
            sorted(alias_map.items(), key=lambda x: len(x[0]), reverse=True)
        ):
            content = content.replace(alias, literal)

        return Response(
            content=content,
            tool_calls=response.tool_calls,
            usage=response.usage,
            model=response.model,
            finish_reason=response.finish_reason,
            metadata=response.metadata,
        )

    def can_process(self, request: Request, _context: TransformContext) -> bool:
        if not self.enabled:
            return False
        total_tokens = request.token_estimate
        return total_tokens > 200


def _find_repeated_literals(request: Request) -> list[str]:
    all_text = "\n".join(m.content for m in request.messages)
    words = all_text.split()
    if len(words) < 10:
        return []

    # Extract long string literals and phrases
    phrase = re.compile(r'"([^"]{' + str(_MIN_LITERAL_LEN) + r',})"')
    matches = phrase.findall(all_text)

    # Also extract repeated long words/phrases
    literal = re.compile(r"(\b[A-Za-z_][A-Za-z0-9_/.-]{' + str(_MIN_LITERAL_LEN) + r',})")
    ident_matches = literal.findall(all_text)

    candidates = matches + ident_matches
    freq: Counter = Counter(candidates)

    aliases_list = [
        text
        for text, count in freq.most_common(_MAX_ALIASES)
        if count >= _MIN_OCCURRENCES and not re.match(r"^\d+$", text)
    ]
    return aliases_list
