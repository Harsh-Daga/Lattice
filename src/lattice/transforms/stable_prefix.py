"""Stable Prefix Handle — CACHE_ONLY.

Identifies the stable prefix of a conversation (system prompt + tool schemas +
static documents) and computes a cache handle. If the provider supports prompt
caching, the handle can be used to skip re-sending the prefix.

Does NOT mutate the prompt if cache is unsupported — only annotates metadata.
"""

from __future__ import annotations

import hashlib
import re

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform, TransformClass
from lattice.core.result import Ok, Result
from lattice.core.transport import Request, Response


class StablePrefixHandle(ReversibleSyncTransform):
    name = "stable_prefix"
    priority = 7
    transform_class = TransformClass.CACHE_ONLY

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        prefix_msgs, delta_msgs = _split_prefix_and_delta(request)
        if not prefix_msgs or not delta_msgs:
            return Ok(request)

        prefix_text = "\n".join(m.content for m in prefix_msgs)
        prefix_hash = hashlib.sha256(prefix_text.encode()).hexdigest()[:16]

        state = context.get_transform_state(self.name)
        state["prefix_hash"] = prefix_hash
        state["prefix_length"] = len(prefix_text)
        state["prefix_message_count"] = len(prefix_msgs)

        request.metadata["_lattice_prefix_hash"] = prefix_hash
        request.metadata["_lattice_prefix_length"] = len(prefix_text)
        request.metadata["_lattice_cache_handle"] = f"CACHE_HANDLE: H({prefix_hash})"

        context.record_metric(self.name, "prefix_chars", len(prefix_text))
        context.record_metric(self.name, "prefix_messages", len(prefix_msgs))
        context.record_metric(self.name, "delta_messages", len(delta_msgs))

        return Ok(request)

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        return response

    def can_process(self, request: Request, _context: TransformContext) -> bool:
        if not self.enabled:
            return False
        return len(request.messages) >= 3


def _split_prefix_and_delta(request: Request) -> tuple[list, list]:
    """Split messages into stable prefix and dynamic delta.

    The stable prefix includes:
    - The system message (if present)
    - Any messages containing tool schemas or static docs
    - Messages before the first user query

    The delta is everything after the prefix.
    """
    msgs = request.messages
    if len(msgs) < 3:
        return [], list(msgs)

    prefix: list = []
    delta: list = []
    found_user = False

    for msg in msgs:
        if not found_user:
            is_system = msg.role == "system"
            is_tool_def = msg.role == "tool" or (
                msg.role == "user"
                and any(
                    kw in msg.content.lower()
                    for kw in ["schema", "specification", "api docs", "documentation"]
                )
            )
            is_static = len(msg.content) > 100 and not re.search(
                r"\b(error|fix|debug)\b", msg.content, re.IGNORECASE
            )

            if is_system or is_tool_def or (is_static and msg.role in ("system", "user")):
                prefix.append(msg)
            elif msg.role == "user":
                found_user = True
                delta.append(msg)
            else:
                delta.append(msg)
        else:
            delta.append(msg)

    return prefix, delta
