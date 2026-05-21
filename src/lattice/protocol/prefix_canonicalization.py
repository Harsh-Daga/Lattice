"""protocol/prefix_canonicalization.py — Canonical prefix extraction and manifest.

Phase 3 cutover: prefix optimization is a TRANSPORT concern.

Extracts the stable prefix from a request (system prompt, tool definitions,
documentation) and creates a canonical segment manifest. This manifest is used
by:
  - cache_planner.py     (to compute KV cache hit probability)
  - transport_planner.py (to decide delta vs full send)
  - provider runtime     (to set provider-specific prefix-caching headers)

Not a transform. Called from content_profiler (first pipeline stage) or
from the gateway/execution builder before pipeline execution.

Design:
  1. Extract prefix segments from request
  2. Compute stable canonical hash (deterministic, tool-agnostic)
  3. Compare with session-stored hash
  4. Build prefix manifest with cache hit / miss metadata
  5. Store manifest in TransformContext for downstream consumers
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

from lattice.transport.types import Request


@dataclasses.dataclass(slots=True)
class PrefixManifest:
    """Canonical representation of a request's stable prefix."""

    prefix_hash: str
    prefix_tokens: int
    suffix_tokens: int
    cache_hit: bool
    segments: list[dict[str, Any]]  # typed canonical segments
    provider_hint: str  # anthropic, openai, generic
    version: int  # manifest version for delta detection

    def to_dict(self) -> dict[str, Any]:
        return {
            "prefix_hash": self.prefix_hash,
            "prefix_tokens": self.prefix_tokens,
            "suffix_tokens": self.suffix_tokens,
            "cache_hit": self.cache_hit,
            "segments": self.segments,
            "provider_hint": self.provider_hint,
            "version": self.version,
        }


@dataclasses.dataclass(slots=True)
class CanonicalPrefix:
    """Extracted canonical prefix with metadata."""

    text: str
    hash: str
    tokens: int
    components: list[str]  # what sources make up the prefix


def extract_canonical_prefix(request: Request) -> CanonicalPrefix:
    """Extract the stable prefix from a request.

    Prefix components (in order):
      1. System message content
      2. Tool definitions (name + description + parameter keys only)
      3. Documentation / artifact segments from metadata
      4. Model card / safety context (if present)

    Returns a CanonicalPrefix with deterministic hash.
    """
    prefix_parts: list[str] = []
    components: list[str] = []

    # 1. System message
    system = request.system_message
    if system:
        prefix_parts.append(system.content or "")
        components.append("system")

    # 2. Tool definitions (canonical, deterministic)
    if request.tools:
        tool_text = _canonicalize_tools(request.tools)
        prefix_text = _canonicalize_tools(request.tools)
        prefix_parts.append(tool_text)
        components.append("tools")

    # 3. Documentation artifacts from metadata
    docs = request.metadata.get("_prefix_docs")
    if docs:
        prefix_parts.append(str(docs))
        components.append("docs")

    # 4. Model card / safety context
    model_card = request.metadata.get("_model_card")
    if model_card:
        prefix_parts.append(str(model_card))
        components.append("model_card")

    prefix_text = "\n".join(prefix_parts)
    prefix_hash = _stable_hash(prefix_text)
    prefix_tokens = max(0, len(prefix_text) // 4)

    return CanonicalPrefix(
        text=prefix_text,
        hash=prefix_hash,
        tokens=prefix_tokens,
        components=components,
    )


def build_prefix_manifest(
    request: Request,
    previous_hash: str | None = None,
    provider: str = "",
) -> PrefixManifest:
    """Build a full prefix manifest with cache hit detection.

    Args:
        request: The incoming request
        previous_hash: Hash from the previous turn in this session (if any)
        provider: Provider name for provider-specific hints

    Returns:
        PrefixManifest with cache metadata
    """
    canonical = extract_canonical_prefix(request)

    cache_hit = bool(previous_hash and previous_hash == canonical.hash)

    # Determine provider-specific caching hint
    provider_hint = _provider_hint(provider, request.model)

    # Build canonical segments for transport reuse
    segments: list[dict[str, Any]] = []
    for component in canonical.components:
        segments.append(
            {
                "type": component,
                "hash": _stable_hash(canonical.text),
                "tokens": canonical.tokens,
            }
        )

    # Compute manifest version for delta tracking
    version = 0 if not previous_hash else (1 if not cache_hit else 2)

    return PrefixManifest(
        prefix_hash=canonical.hash,
        prefix_tokens=canonical.tokens,
        suffix_tokens=max(0, request.token_estimate - canonical.tokens),
        cache_hit=cache_hit,
        segments=segments,
        provider_hint=provider_hint,
        version=version,
    )


def _canonicalize_tools(tools: list[dict[str, Any]]) -> str:
    """Serialize tool definitions to a stable canonical string.

    Only hashes name + description + parameter keys (not full schema).
    This ensures that minor tool implementation changes don't invalidate
    the prefix cache.
    """
    parts: list[str] = []
    for tool in tools:
        if isinstance(tool, dict):
            func = tool.get("function", tool)
            name = func.get("name", "")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            param_keys = ""
            if isinstance(params, dict) and "properties" in params:
                param_keys = ",".join(sorted(params["properties"].keys()))
            parts.append(f"{name}:{desc}:{param_keys}")
        else:
            parts.append(str(tool))
    return "\n".join(sorted(parts))


def _stable_hash(text: str) -> str:
    """Compute a fast, stable hash of text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:32]


def _provider_hint(provider: str, model: str) -> str:
    """Determine provider-specific prefix caching strategy."""
    if not provider and model:
        # Infer from model name
        if "claude" in model.lower() or "anthropic" in model.lower():
            return "anthropic"
        if "gpt" in model.lower():
            return "openai"
    provider_lower = (provider or "").lower()
    if provider_lower in ("anthropic", "claude", "bedrock"):
        return "anthropic"
    if provider_lower in ("openai", "azure", "azure-openai"):
        return "openai"
    if provider_lower in ("ollama", "ollama-cloud"):
        return "none"  # No native prefix caching
    return "generic"


# ──────────────────────────────────────────────────────────────────
# Integration helpers
# ──────────────────────────────────────────────────────────────────


def canonicalize_request_prefix(
    request: Request,
    previous_hash: str | None,
    provider: str,
) -> PrefixManifest:
    """High-level convenience: extract canonical prefix and build manifest.

    Usage:
        from lattice.protocol.prefix_canonicalization import canonicalize_request_prefix

        manifest = canonicalize_request_prefix(
            request,
            previous_hash=context.session_state.get("_prefix_hash"),
            provider=context.provider,
        )
        # Store for next turn
        context.session_state["_prefix_hash"] = manifest.prefix_hash
        context.session_state["_prefix_manifest"] = manifest.to_dict()

        # Set provider-specific headers
        if manifest.provider_hint == "anthropic":
            request.extra_headers["anthropic-beta"] = "prompt-caching-2024-07-31"
        request.extra_headers["x-lattice-prefix-hash"] = manifest.prefix_hash[:16]
    """
    return build_prefix_manifest(request, previous_hash, provider)
