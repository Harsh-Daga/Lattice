"""Path Prefix Compression — LOSSLESS_SAFE.

Compresses repeated path prefixes in logs, file lists, and infra output.
Simple, lossless, and highly effective for devops/infra contexts.

Example:
  /var/log/app/service-a/error.log
  /var/log/app/service-b/error.log
→
  PREFIX P=/var/log/app
  P/service-a/error.log
  P/service-b/error.log
"""

from __future__ import annotations

import os
from collections import Counter

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.ir.primitives import PromptIRV2
from lattice.transport.types import Request, Response


class PathPrefixCompressor(ReversibleSyncTransform):
    name = "path_prefix"
    priority = 23

    # ------------------------------------------------------------------
    # IR-native optimize() — v2 path
    # ------------------------------------------------------------------

    def optimize(
        self,
        ir: PromptIRV2,
        _request: Request,
        context: TransformContext,
    ) -> Result[PromptIRV2, TransformError]:
        """IR-native: compress repeated path prefixes in span text."""
        total_saved = 0
        new_sections = []
        prefix_map: dict[str, str] = {}
        for sec in ir.sections:
            new_spans = []
            for span in sec.spans:
                compressed, prefix, saved = _compress_paths(span.text)
                if prefix and compressed != span.text:
                    prefix_map[prefix] = compressed
                    new_spans.append(span.with_text(compressed))
                    total_saved += saved
                else:
                    new_spans.append(span)
            new_sections.append(sec.with_spans(tuple(new_spans)))

        if prefix_map:
            state = context.get_transform_state(self.name)
            state["prefix_map"] = prefix_map
            context.record_metric(self.name, "chars_saved", total_saved)

        return Ok(ir.with_sections(tuple(new_sections)))

    def reverse(self, response: Response, context: TransformContext) -> Response:
        state = context.get_transform_state(self.name)
        prefix = state.get("prefix", "")
        if not prefix:
            return response
        content = response.content.replace("PREFIX P=", "")
        lines = content.splitlines()
        restored = []
        for line in lines:
            if line.startswith("P/"):
                restored.append(prefix + line[1:])
            else:
                restored.append(line)
        return Response(
            content="\n".join(restored),
            tool_calls=response.tool_calls,
            usage=response.usage,
            model=response.model,
            finish_reason=response.finish_reason,
            metadata=response.metadata,
        )

    def can_process(self, request: Request, _context: TransformContext) -> bool:
        if not self.enabled:
            return False
        return request.token_estimate > 200


def _find_common_prefix(paths: list[str]) -> str:
    if not paths:
        return ""
    prefix = os.path.commonpath(paths) if len(paths) > 1 else ""
    if prefix and prefix != "/":
        return prefix
    return ""


def _compress_paths(text: str) -> tuple[str, str, int]:
    lines = text.splitlines()
    paths: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("/") and "/" in stripped[1:]:
            paths.append(stripped)

    if len(paths) < 3:
        return text, "", 0

    prefix = _find_common_prefix(paths)
    if not prefix or len(prefix) < 8:
        return text, "", 0

    freq_check: Counter = Counter(
        p.split(prefix, 1)[1] if p.startswith(prefix) else p for p in paths
    )
    if sum(1 for v in freq_check.values() if v >= 2) < 2:
        return text, "", 0

    result_lines = [f"PREFIX P={prefix}"]
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix):
            result_lines.append(f"P{stripped[len(prefix) :]}")
        else:
            result_lines.append(line)

    result = "\n".join(result_lines)
    return result, prefix, len(text) - len(result)
