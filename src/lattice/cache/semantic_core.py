"""Semantic cache package — split modules."""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import structlog

from lattice.cache.fingerprint import (
    CachedResponse,
    ContentClass,
    _compute_semantic_fingerprint,
    _compute_similarity,
    _FingerprintEntry,
    _SemanticFingerprint,
    compute_cache_key,
)
from lattice.cache.stores import CacheBackend, InMemoryCacheBackend

_logger = structlog.get_logger()

class SemanticCache:
    """In-memory hybrid exact/semantic response cache with TTL and LRU eviction.

    Usage:
        cache = SemanticCache(ttl_seconds=300, max_entries=1000)
        key = compute_cache_key(request)
        hit = cache.get(key)
        if hit:
            return hit
        response = await provider.completion(...)
        cache.set(key, response)
    """

    def __init__(
        self,
        *,
        backend: CacheBackend | None = None,
        ttl_seconds: int = 300,
        max_entries: int = 1000,
        max_entry_size_kb: int = 512,
        semantic_threshold: float = 0.86,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.max_entry_size_bytes = max_entry_size_kb * 1024
        self.semantic_threshold = max(0.0, min(1.0, semantic_threshold))
        self._backend = backend or InMemoryCacheBackend(max_entries=max_entries)
        self._fingerprints: OrderedDict[str, _FingerprintEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._exact_hits = 0
        self._approximate_hits = 0
        self._approximate_misses = 0
        self._evictions = 0
        self._rejects = 0
        self._content_class_hits: dict[str, int] = {}
        self._maintenance_runs = 0
        self._stale_removed = 0
        # Approximate lookup index: (model, content_class, role_pattern, tool_schema_hash) -> set of keys
        self._approximate_index: dict[
            tuple[str, ContentClass, tuple[str, ...], str | None],
            set[str],
        ] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, key: str, request: Any | None = None) -> CachedResponse | None:
        """Retrieve a cached response by *key*.

        Returns None if missing or expired. Promotes to MRU on hit.
        If *request* is provided, performs a conservative semantic lookup
        on exact misses.
        """
        if not self.enabled:
            return None

        # Fast exact hit
        resp = await self._backend.get(key)
        if resp is not None:
            async with self._lock:
                self._hits += 1
                self._exact_hits += 1
                cc = resp.metadata.get("_lattice_content_class", ContentClass.PLAIN_TEXT.value)
                self._content_class_hits[cc] = self._content_class_hits.get(cc, 0) + 1
            return resp

        if request is None:
            async with self._lock:
                self._misses += 1
            return None

        query_fp = _compute_semantic_fingerprint(request)
        if not query_fp.token_set:
            async with self._lock:
                self._misses += 1
                self._approximate_misses += 1
            return None

        query_model = str(getattr(request, "model", "") or "")
        index_key = (
            query_model,
            query_fp.content_class,
            query_fp.role_pattern,
            query_fp.tool_schema_hash,
        )

        # Snapshot candidate keys under lock, then release before scoring / I/O
        async with self._lock:
            candidate_keys = list(self._approximate_index.get(index_key, set()))

        best_key: str | None = None
        best_score = 0.0
        for cand_key in candidate_keys:
            # Verify backend still holds the entry before scoring
            resp = await self._backend.get(cand_key)
            if resp is None:
                async with self._lock:
                    self._fingerprints.pop(cand_key, None)
                    self._approximate_index.get(index_key, set()).discard(cand_key)
                continue
            cand_entry = self._fingerprints.get(cand_key)
            if cand_entry is None:
                continue
            if cand_entry.model != query_model:
                continue
            score = _compute_similarity(query_fp, cand_entry.fingerprint)
            if score > best_score:
                best_score = score
                best_key = cand_key

        if best_key is None or best_score < self.semantic_threshold:
            async with self._lock:
                self._misses += 1
                self._approximate_misses += 1
            return None

        # Final backend verification and stats update under lock
        resp = await self._backend.get(best_key)
        async with self._lock:
            if resp is None:
                self._fingerprints.pop(best_key, None)
                self._approximate_index.get(index_key, set()).discard(best_key)
                self._misses += 1
                self._approximate_misses += 1
                return None

            self._fingerprints.move_to_end(best_key)
            self._hits += 1
            self._approximate_hits += 1
            cc = resp.metadata.get("_lattice_content_class", ContentClass.PLAIN_TEXT.value)
            self._content_class_hits[cc] = self._content_class_hits.get(cc, 0) + 1
        return resp

    async def set(
        self,
        key: str,
        response: CachedResponse,
        request: Any | None = None,
    ) -> bool:
        """Store *response* under *key*.

        Returns True if stored, False if rejected (too large or disabled).
        Evicts oldest entries if max size exceeded.
        """
        if not self.enabled:
            return False

        # Rough size estimate
        size = len(response.content.encode("utf-8"))
        if response.tool_calls:
            size += len(json.dumps(response.tool_calls).encode("utf-8"))
        if response.sse_chunks:
            size += sum(len(c.encode("utf-8")) for c in response.sse_chunks)
        if size > self.max_entry_size_bytes:
            self._rejects += 1
            return False

        model = ""
        if request is not None:
            model = str(getattr(request, "model", "") or response.model or "")
        else:
            model = response.model or ""

        fp: _SemanticFingerprint | None = None
        if request is not None:
            fp = _compute_semantic_fingerprint(request)
            response.metadata = dict(response.metadata) if response.metadata is not None else {}
            response.metadata["_lattice_content_class"] = fp.content_class.value
        else:
            # Minimal fingerprint so _fingerprints mirrors the backend for
            # entry counting and invalidation.
            fp = _SemanticFingerprint(
                token_set=frozenset(),
                role_pattern=(),
                tool_schema_hash=None,
                normalized_text="",
                content_class=ContentClass.PLAIN_TEXT,
                message_count=0,
                has_tools=False,
                has_tool_calls=False,
                has_images=False,
            )

        await self._backend.set(key, response, self.ttl_seconds)

        index_key = (
            model,
            fp.content_class,
            fp.role_pattern,
            fp.tool_schema_hash,
        )

        async with self._lock:
            self._fingerprints[key] = _FingerprintEntry(model=model, fingerprint=fp)
            self._fingerprints.move_to_end(key)
            self._approximate_index.setdefault(index_key, set()).add(key)
            # Incremental eviction: if over max_entries, remove oldest from index
            while len(self._fingerprints) > self.max_entries:
                oldest_key, oldest_entry = self._fingerprints.popitem(last=False)
                oldest_index_key = (
                    oldest_entry.model,
                    oldest_entry.fingerprint.content_class,
                    oldest_entry.fingerprint.role_pattern,
                    oldest_entry.fingerprint.tool_schema_hash,
                )
                self._approximate_index.get(oldest_index_key, set()).discard(oldest_key)
                self._evictions += 1

        return True

    async def invalidate(self, key: str) -> bool:
        """Remove *key* from cache. Returns True if existed."""
        async with self._lock:
            entry = self._fingerprints.pop(key, None)
            if entry is not None:
                index_key = (
                    entry.model,
                    entry.fingerprint.content_class,
                    entry.fingerprint.role_pattern,
                    entry.fingerprint.tool_schema_hash,
                )
                self._approximate_index.get(index_key, set()).discard(key)
        return await self._backend.delete(key)

    async def invalidate_by_pattern(self, predicate: Callable[..., bool]) -> int:
        """Remove all entries matching *predicate* (called with CachedResponse).

        Returns count removed.
        """
        removed = 0
        keys = await self._backend.keys()
        matched: list[str] = []
        for k in keys:
            resp = await self._backend.get(k)
            if resp is not None and predicate(resp):
                await self._backend.delete(k)
                matched.append(k)
                removed += 1
        async with self._lock:
            for k in matched:
                entry = self._fingerprints.pop(k, None)
                if entry is not None:
                    index_key = (
                        entry.model,
                        entry.fingerprint.content_class,
                        entry.fingerprint.role_pattern,
                        entry.fingerprint.tool_schema_hash,
                    )
                    self._approximate_index.get(index_key, set()).discard(k)
        return removed

    async def clear(self) -> int:
        """Drop all entries. Returns count removed from backend."""
        async with self._lock:
            self._fingerprints.clear()
            self._approximate_index.clear()
        backend_count = await self._backend.clear()
        return backend_count

    async def expire_stale(self) -> int:
        """Remove expired entries. Returns count removed.

        Locates stale keys via the backend, then performs batch deletion
        when the backend supports ``delete_many``, falling back to
        per-key deletion.
        """
        # Identify expired keys without holding the main lock
        backend_keys = await self._backend.keys()
        expired_keys: list[str] = []
        for k in backend_keys:
            resp = await self._backend.get(k)
            if resp is None:
                expired_keys.append(k)
        # Batch-delete expired entries from backend if supported
        if not expired_keys:
            return 0
        if hasattr(self._backend, "delete_many"):
            await self._backend.delete_many(expired_keys)
        else:
            for k in expired_keys:
                await self._backend.delete(k)
        # Remove from index under lock
        async with self._lock:
            for k in expired_keys:
                entry = self._fingerprints.pop(k, None)
                if entry is not None:
                    index_key = (
                        entry.model,
                        entry.fingerprint.content_class,
                        entry.fingerprint.role_pattern,
                        entry.fingerprint.tool_schema_hash,
                    )
                    self._approximate_index.get(index_key, set()).discard(k)
            self._maintenance_runs += 1
            self._stale_removed += len(expired_keys)
        return len(expired_keys)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    async def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        async with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            approx_total = self._approximate_hits + self._approximate_misses
            approx_hit_rate = self._approximate_hits / approx_total if approx_total > 0 else 0.0
            return {
                "enabled": self.enabled,
                "entries": len(self._fingerprints),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
                "exact_hits": self._exact_hits,
                "approximate_hits": self._approximate_hits,
                "approximate_misses": self._approximate_misses,
                "semantic_hits": self._approximate_hits,
                "semantic_misses": self._approximate_misses,
                "semantic_hit_rate": round(approx_hit_rate, 4),
                "semantic_threshold": round(self.semantic_threshold, 4),
                "evictions": self._evictions,
                "rejects": self._rejects,
                "content_class_hits": dict(self._content_class_hits),
                "maintenance_runs": self._maintenance_runs,
                "stale_removed": self._stale_removed,
            }

    @property
    def entry_count(self) -> int:
        """Current number of cached entries (no lock — best-effort)."""
        return len(self._fingerprints)


# ---------------------------------------------------------------------------
# Helpers for assembling streaming responses into cacheable form
# ---------------------------------------------------------------------------


def assemble_cached_response(
    model: str,
    content: str,
    tool_calls: list[dict[str, Any]] | None,
    usage: dict[str, Any],
    finish_reason: str = "stop",
    sse_chunks: list[str] | None = None,
) -> CachedResponse:
    """Build a CachedResponse from provider output."""
    return CachedResponse(
        content=content,
        tool_calls=tool_calls,
        usage=usage or {},
        model=model,
        finish_reason=finish_reason,
        sse_chunks=sse_chunks or [],
    )


def generate_sse_chunks(
    cached: CachedResponse,
    request_id: str = "",
    session_id: str = "",
    chunk_size: int = 20,
) -> list[str]:
    """Generate SSE chunk strings from a CachedResponse for re-streaming.

    If *cached.sse_chunks* is already populated, returns those.
    Otherwise synthesizes chunks by splitting content into *chunk_size*
    character segments and wrapping each in a ``data: {"choices":...}``
    envelope.
    """
    if cached.sse_chunks:
        return list(cached.sse_chunks)

    chunks: list[str] = []
    content = cached.content
    model = cached.model or ""

    # First chunk carries lattice metadata
    first_chunk: dict[str, Any] = {
        "id": request_id or "lattice-cache-hit",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "",
                    "_lattice": {"model": model, "session_id": session_id, "cache_hit": True},
                },
                "finish_reason": None,
            }
        ],
    }
    chunks.append(f"data: {json.dumps(first_chunk)}\n\n")

    # Content chunks
    for i in range(0, len(content), chunk_size):
        piece = content[i : i + chunk_size]
        chunk: dict[str, Any] = {
            "id": request_id or "lattice-cache-hit",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": piece},
                    "finish_reason": None,
                }
            ],
        }
        chunks.append(f"data: {json.dumps(chunk)}\n\n")

    # Tool calls (if any) — emit as a single chunk
    if cached.tool_calls:
        tc_chunk: dict[str, Any] = {
            "id": request_id or "lattice-cache-hit",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": cached.tool_calls},
                    "finish_reason": None,
                }
            ],
        }
        chunks.append(f"data: {json.dumps(tc_chunk)}\n\n")

    # Final chunk with finish_reason
    final_chunk: dict[str, Any] = {
        "id": request_id or "lattice-cache-hit",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": cached.finish_reason,
            }
        ],
    }
    chunks.append(f"data: {json.dumps(final_chunk)}\n\n")
    return chunks


__all__ = [
    "ExactResponseCache",
    "SemanticCache",
    "CachedResponse",
    "CacheBackend",
    "InMemoryCacheBackend",
    "ContentClass",
    "compute_cache_key",
    "assemble_cached_response",
    "generate_sse_chunks",
]


# Backward-compatible alias.
ExactResponseCache = SemanticCache
