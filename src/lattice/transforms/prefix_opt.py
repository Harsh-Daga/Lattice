"""Prefix Optimizer — Phase 3 thin transport wrapper.

This is now a metadata-only wrapper around protocol/prefix_canonicalization.py.

The actual prefix extraction and manifest building happens in content_profiler
(which runs at priority 1, before this transform). This transform only:
  1. Reads the pre-built prefix manifest from session_state
  2. Adds provider-specific headers for prefix caching
  3. Records metrics

Why this architecture:
  - Prefixing is a TRANSPORT concern, not a transform concern.
  - The canonical prefix lives in protocol/ as an independent module.
  - The pipeline wrapper only emits transport signals.

DEPRECATED: This transform will be removed entirely once all consumers
migrate to reading from content_profiler output directly.
"""

from __future__ import annotations

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.result import Ok, Result
from lattice.ir.primitives import PromptIRV2
from lattice.pipeline.base import ReversibleSyncTransform
from lattice.planner.runtime_state import get_canonical_state_value
from lattice.transport.types import Request, Response


class PrefixOptimizer(ReversibleSyncTransform):
    """Transport-level prefix caching signal wrapper.

    Reads prefix manifest built by content_profiler and adds
    provider-specific headers.
    """

    name = "prefix_optimizer"
    priority = 10  # AFTER content_profiler (priority 1)

    def process(
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        """Read prefix manifest and emit transport signals.

        Phase 3: The canonical prefix is now built by content_profiler at
        priority 1, before this transform. This transform only emits
        transport signals and records metrics.

        Backward-compatibility: if content_profiler did not run, the old
        logic is available as fallback (standalone mode).
        """
        # --- Phase 3 path: read from session_state (content_profiler already ran) ---
        manifest = get_canonical_state_value(context, "_prefix_manifest", {})
        if manifest.get("prefix_hash"):
            prefix_hash = manifest.get("prefix_hash", "")
            cache_hit = manifest.get("cache_hit", False)
            prefix_tokens = manifest.get("prefix_tokens", 0)
            suffix_tokens = manifest.get("suffix_tokens", 0)
            provider_hint = manifest.get("provider_hint", "")
        else:
            # --- Fallback: compute locally (standalone mode for tests) ---
            from lattice.protocol.prefix_canonicalization import (
                build_prefix_manifest,
            )

            # Check phase-3 canonical key first, then legacy transform state
            previous_hash = get_canonical_state_value(context, "_prefix_hash")
            if previous_hash is None:
                legacy_state = context.session_state.get("prefix_optimizer", {})
                previous_hash = legacy_state.get("prefix_hash")
            provider = get_canonical_state_value(context, "_lattice_provider", "")
            manifest_obj = build_prefix_manifest(
                request,
                previous_hash=previous_hash,
                provider=provider,
            )
            prefix_hash = manifest_obj.prefix_hash
            cache_hit = manifest_obj.cache_hit
            prefix_tokens = manifest_obj.prefix_tokens
            suffix_tokens = manifest_obj.suffix_tokens
            provider_hint = manifest_obj.provider_hint

        # Backward-compat: write metadata keys that tests expect
        request.metadata["_prefix_hash"] = prefix_hash
        request.metadata["_cache_hit"] = cache_hit
        request.metadata["_prefix_tokens"] = max(0, prefix_tokens)
        request.metadata["_suffix_tokens"] = max(0, suffix_tokens)

        # Provider-specific headers
        if not cache_hit and prefix_hash:
            request.extra_headers["x-lattice-prefix-hash"] = prefix_hash[:16]
            if provider_hint == "anthropic":
                request.extra_headers["anthropic-beta"] = "prompt-caching-2024-07-31"

        # Metrics
        context.record_metric(self.name, "cache_hit", cache_hit)
        context.record_metric(self.name, "prefix_tokens", max(0, prefix_tokens))
        context.record_metric(self.name, "suffix_tokens", max(0, suffix_tokens))

        return Ok(request)

    def optimize(
        self,
        ir: PromptIRV2,
        request: Request,
        context: TransformContext,
    ) -> Result[PromptIRV2, TransformError]:
        """Attach prefix cache metadata to immutable IR."""
        manifest = get_canonical_state_value(context, "_prefix_manifest", {})
        if manifest.get("prefix_hash"):
            prefix_hash = manifest.get("prefix_hash", "")
            cache_hit = manifest.get("cache_hit", False)
            prefix_tokens = manifest.get("prefix_tokens", 0)
            suffix_tokens = manifest.get("suffix_tokens", 0)
            provider_hint = manifest.get("provider_hint", "")
        else:
            from lattice.protocol.prefix_canonicalization import build_prefix_manifest

            previous_hash = get_canonical_state_value(context, "_prefix_hash")
            if previous_hash is None:
                legacy_state = context.session_state.get("prefix_optimizer", {})
                previous_hash = legacy_state.get("prefix_hash")
            provider = get_canonical_state_value(context, "_lattice_provider", "")
            manifest_obj = build_prefix_manifest(
                request,
                previous_hash=previous_hash,
                provider=provider,
            )
            prefix_hash = manifest_obj.prefix_hash
            cache_hit = manifest_obj.cache_hit
            prefix_tokens = manifest_obj.prefix_tokens
            suffix_tokens = manifest_obj.suffix_tokens
            provider_hint = manifest_obj.provider_hint

        updated_ir = ir.add_metadata(
            _prefix_hash=prefix_hash,
            _cache_hit=cache_hit,
            _prefix_tokens=max(0, prefix_tokens),
            _suffix_tokens=max(0, suffix_tokens),
            _lattice_provider_hint=provider_hint,
        )
        request.metadata["_prefix_hash"] = prefix_hash
        request.metadata["_cache_hit"] = cache_hit
        request.metadata["_prefix_tokens"] = max(0, prefix_tokens)
        request.metadata["_suffix_tokens"] = max(0, suffix_tokens)

        if not cache_hit and prefix_hash:
            request.extra_headers["x-lattice-prefix-hash"] = prefix_hash[:16]
            if provider_hint == "anthropic":
                request.extra_headers["anthropic-beta"] = "prompt-caching-2024-07-31"

        context.session_state["_lattice_ir_v2"] = updated_ir
        context.record_metric(self.name, "cache_hit", cache_hit)
        context.record_metric(self.name, "prefix_tokens", max(0, prefix_tokens))
        context.record_metric(self.name, "suffix_tokens", max(0, suffix_tokens))
        return Ok(updated_ir)

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        """No-op — prefix optimization is metadata-only."""
        return response
