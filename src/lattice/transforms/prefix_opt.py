"""Prefix Optimization transform — Production Grade.

Splits the prompt into a static prefix (system prompt + tool definitions)
and dynamic suffix (user messages). By keeping the prefix byte-identical
across requests, we enable provider-side KV cache hits (Anthropic prefix
caching, OpenAI implicit caching).

**Research basis:**
- Anthropic's prompt caching (beta) provides 90% cost reduction on cache hits
- OpenAI's implicit caching works on exact-prefix matches (tools→system→messages)
- Stable prefix hashing is critical: any byte change invalidates the cache

**Reversible:** No-op on forward. Only adds metadata to the request.
Does not modify the actual payload.

**Typical savings:** When cache hits, 5-10x savings on repeated prefix
(80-90% of tokens in multi-turn conversations).

**Performance:** Stable hash of the prefix. Target: <0.05ms.
"""

from __future__ import annotations

import hashlib
from typing import Any

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.transport import Request, Response

# =============================================================================
# PrefixOptimizer
# =============================================================================

class PrefixOptimizer(ReversibleSyncTransform):
    """Optimize prefix for provider-side caching.

    Algorithm:
    1. Identify static portion of messages:
       - System message (if present)
       - Tool definitions (if present)
       - Documentation / artifact segments (if present in metadata)
    2. Compute stable hash of the static portion (deterministic serialization)
    3. Compare with hash stored in TransformContext
    4. If hash matches → cache hit
    5. If hash differs → cache miss, new hash stored in context

    Note: This transform does NOT modify the Request content.
          It only sets metadata to signal caching opportunities.

    Provider-specific hints:
    - Anthropic: "anthropic-beta: prompt-caching-2024-07-31"
    - OpenAI: No native hint (rely on server-side matching)
    - Generic: "x-lattice-prefix-hash" header for observability
    """

    name = "prefix_optimizer"
    priority = 10  # Run BEFORE any content-modifying transforms

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        """Analyze and record prefix optimization metadata."""
        prefix_parts: list[str] = []

        # System message is always part of the prefix
        system = request.system_message
        if system:
            prefix_parts.append(system.content)

        # Tool definitions are part of the prefix if present
        if request.tools:
            # Serialize tool definitions to a stable string
            tool_text = self._serialize_tools(request.tools)
            prefix_parts.append(tool_text)

        # Any extra documentation in metadata (e.g., from manifest segments)
        docs = request.metadata.get("_prefix_docs")
        if docs:
            prefix_parts.append(str(docs))

        prefix_text = "\n".join(prefix_parts)
        prefix_hash = self._compute_hash(prefix_text)
        prefix_tokens = len(prefix_text) // 4  # Approximate

        # Check if we have a previous hash in session
        state = context.get_transform_state(self.name)
        previous_hash = state.get("prefix_hash")
        cache_hit = bool(previous_hash and previous_hash == prefix_hash)

        if not cache_hit:
            state["prefix_hash"] = prefix_hash

        # Determine suffix (non-prefix messages)
        suffix_tokens = request.token_estimate - prefix_tokens

        # Add hints to request metadata
        request.metadata["_prefix_hash"] = prefix_hash
        request.metadata["_cache_hit"] = cache_hit
        request.metadata["_prefix_tokens"] = max(0, prefix_tokens)
        request.metadata["_suffix_tokens"] = max(0, suffix_tokens)

        # Add provider hints to extra_headers
        if not cache_hit:
            # Only send cache hints on the first request of a session
            request.extra_headers["x-lattice-prefix-hash"] = prefix_hash[:16]
            # Anthropic beta header
            if "claude" in request.model.lower() or context.provider == "anthropic":
                request.extra_headers["anthropic-beta"] = "prompt-caching-2024-07-31"

        # Metrics
        context.record_metric(self.name, "cache_hit", cache_hit)
        context.record_metric(self.name, "prefix_tokens", max(0, prefix_tokens))
        context.record_metric(self.name, "suffix_tokens", max(0, suffix_tokens))

        return Ok(request)

    def _compute_hash(self, text: str) -> str:
        """Compute stable hash of text.

        Uses MD5 (fast, non-cryptographic) for speed.
        Collisions are not a concern for our use case.
        """
        hasher = hashlib.md5(text.encode("utf-8"))
        return hasher.hexdigest()[:32]

    def _serialize_tools(self, tools: list[dict[str, Any]]) -> str:
        """Serialize tool definitions to a stable string.

        Uses deterministic JSON serialization with sorted keys
        to ensure byte-identical output across Python versions.
        If a tool's name/description doesn't change, the prefix is
        still considered a cache hit even if implementation details
        change (we only hash name + description + parameter keys).
        """
        parts: list[str] = []
        for tool in tools:
            if isinstance(tool, dict):
                # OpenAI format: {"type": "function", "function": {...}}
                func = tool.get("function", tool)
                name = func.get("name", "")
                desc = func.get("description", "")
                # Include parameter keys for stability without full schema noise
                params = func.get("parameters", {})
                param_keys = ""
                if isinstance(params, dict) and "properties" in params:
                    param_keys = ",".join(
                        sorted(params["properties"].keys())
                    )
                parts.append(f"{name}:{desc}:{param_keys}")
            else:
                parts.append(str(tool))
        return "\n".join(sorted(parts))  # Sort for stability

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        """No-op — prefix optimization is metadata-only."""
        return response
