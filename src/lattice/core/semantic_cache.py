"""Hybrid response cache for LATTICE.

Caches LLM responses keyed by a canonical hash of the request payload and
optionally by a lightweight semantic fingerprint of the request text.
On cache hit, returns the stored response directly — bypassing the provider
entirely. Supports both streaming and non-streaming responses, TTL-based
expiration, approximate semantic lookup, and LRU eviction.

The cache sits AFTER the transform pipeline (so the key reflects the
compressed/optimized request) and BEFORE the provider call.

Design decisions
----------------
1. Checksum-based keys (SHA-256 over canonical JSON) — deterministic,
   collision-resistant, fast enough for our throughput.
2. Semantic lookup is content-class-aware and uses a weighted fingerprint
   (token Jaccard, role pattern, tool schema, text similarity).
3. Pluggable backend (default: in-memory). Future work: Redis backend.
4. Streaming responses are assembled into a single CachedResponse and
   re-emitted as SSE chunks on cache hit.
5. TTL is per-entry, not global sweep, to avoid stopping-the-world cleanup.
6. Max size is enforced at insertion time (LRU eviction of oldest entries).
"""

from __future__ import annotations

import asyncio
import dataclasses
import difflib
import enum
import hashlib
import json
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import structlog

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ContentClass(enum.Enum):
    PLAIN_TEXT = "plain_text"
    CODE = "code"
    JSON = "json"
    TOOL_OUTPUT = "tool_output"
    MIXED = "mixed"


@dataclass(slots=True)
class CachedResponse:
    """A cached LLM response, storage-agnostic.

    Attributes:
        content: Full text content of the response.
        tool_calls: Optional list of tool calls.
        usage: Token usage dict (may be empty for cached hits).
        model: Model identifier used for the original response.
        finish_reason: Finish reason string.
        sse_chunks: For streaming responses, the pre-computed SSE chunk strings.
        created_at: Unix timestamp when the entry was cached.
        expires_at: Unix timestamp when the entry becomes stale.
        metadata: Extra metadata (e.g. content class for stats).
    """

    content: str
    tool_calls: list[dict[str, Any]] | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    finish_reason: str = "stop"
    sse_chunks: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _SemanticFingerprint:
    token_set: frozenset[str]
    role_pattern: tuple[str, ...]
    tool_schema_hash: str | None
    normalized_text: str
    content_class: ContentClass
    message_count: int
    has_tools: bool
    has_tool_calls: bool
    has_images: bool


@dataclass(slots=True)
class _FingerprintEntry:
    model: str
    fingerprint: _SemanticFingerprint


class CacheBackend(Protocol):
    async def get(self, key: str) -> CachedResponse | None: ...
    async def set(self, key: str, response: CachedResponse, ttl: float) -> bool: ...
    async def delete(self, key: str) -> bool: ...
    async def keys(self) -> list[str]: ...
    async def clear(self) -> int: ...
    async def delete_many(self, keys: list[str]) -> int: ...


class InMemoryCacheBackend:
    """Default in-memory backend with TTL and LRU eviction."""

    def __init__(self, max_entries: int = 1000) -> None:
        self._store: OrderedDict[str, CachedResponse] = OrderedDict()
        self._max_entries = max_entries
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> CachedResponse | None:
        async with self._lock:
            resp = self._store.get(key)
            if resp is None:
                return None
            if time.time() > resp.expires_at:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return resp

    async def set(self, key: str, response: CachedResponse, ttl: float) -> bool:
        async with self._lock:
            response.expires_at = time.time() + ttl
            self._store[key] = response
            self._store.move_to_end(key)
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)
            return True

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    async def keys(self) -> list[str]:
        async with self._lock:
            return list(self._store.keys())

    async def clear(self) -> int:
        async with self._lock:
            count = len(self._store)
            self._store.clear()
        return count


# Optional redis dependency
_REDIS_AVAILABLE = False
try:
    import redis.asyncio as _redis

    _REDIS_AVAILABLE = True
except ImportError:
    _redis = None  # type: ignore[assignment]


class RedisCacheBackend:
    """Redis-backed cache backend for multi-process deployments.

    Stores serialized CachedResponse objects with Redis TTL.
    All operations are async via redis-py.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        prefix: str = "lattice:cache:",
    ) -> None:
        if not _REDIS_AVAILABLE:
            raise ImportError(
                "Redis is not installed. "
                "Install with: pip install redis"
            )
        self.url = url
        self.prefix = prefix
        self._client: Any | None = None
        self._log = structlog.get_logger().bind(module="redis_cache_backend")

    async def start(self) -> None:
        """Initialize Redis connection."""
        if self._client is None:
            self._client = _redis.from_url(self.url, decode_responses=True)
            await self._client.ping()  # type: ignore[misc]
            self._log.info("redis_cache_connected", url=self.url)

    async def stop(self) -> None:
        """Close Redis connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._log.info("redis_cache_disconnected")

    async def get(self, key: str) -> CachedResponse | None:
        if self._client is None:
            raise RuntimeError("Redis cache not connected")
        redis_key = f"{self.prefix}{key}"
        raw = await self._client.get(redis_key)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            return CachedResponse(
                content=data["content"],
                tool_calls=data.get("tool_calls"),
                usage=data.get("usage", {}),
                model=data.get("model", ""),
                finish_reason=data.get("finish_reason", "stop"),
                sse_chunks=data.get("sse_chunks", []),
                created_at=data.get("created_at", 0.0),
                expires_at=data.get("expires_at", 0.0),
                metadata=data.get("metadata", {}),
            )
        except (json.JSONDecodeError, KeyError) as exc:
            self._log.warning("redis_cache_decode_failed", key=key, error=str(exc))
            await self._client.delete(redis_key)
            return None

    async def set(self, key: str, response: CachedResponse, ttl: float) -> bool:
        if self._client is None:
            raise RuntimeError("Redis cache not connected")
        redis_key = f"{self.prefix}{key}"
        data = {
            "content": response.content,
            "tool_calls": response.tool_calls,
            "usage": response.usage,
            "model": response.model,
            "finish_reason": response.finish_reason,
            "sse_chunks": response.sse_chunks,
            "created_at": response.created_at,
            "expires_at": response.expires_at,
            "metadata": response.metadata,
        }
        raw = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
        await self._client.setex(redis_key, int(ttl), raw)
        return True

    async def delete(self, key: str) -> bool:
        if self._client is None:
            return False
        redis_key = f"{self.prefix}{key}"
        result = await self._client.delete(redis_key)
        return bool(result > 0)

    async def keys(self) -> list[str]:
        if self._client is None:
            return []
        cursor = 0
        ids: list[str] = []
        while True:
            cursor, keys = await self._client.scan(cursor, match=f"{self.prefix}*", count=100)
            for k in keys:
                key_str = k.decode("utf-8") if isinstance(k, bytes) else k
                ids.append(key_str.replace(self.prefix, ""))
            if cursor == 0:
                break
        return ids

    async def clear(self) -> int:
        if self._client is None:
            return 0
        cursor = 0
        total_removed = 0
        while True:
            cursor, keys = await self._client.scan(cursor, match=f"{self.prefix}*", count=100)
            if keys:
                total_removed += await self._client.delete(*keys)
            if cursor == 0:
                break
        return total_removed

    async def delete_many(self, keys: list[str]) -> int:
        """Batch-delete multiple keys. Returns count removed."""
        if self._client is None or not keys:
            return 0
        prefixed = [f"{self.prefix}{k}" for k in keys]
        return await self._client.delete(*prefixed)


# ---------------------------------------------------------------------------
# Key computation
# ---------------------------------------------------------------------------


def _canonical_request(request: Any) -> dict[str, Any]:
    """Extract canonical fields from a request for stable hashing.

    Only fields that affect the LLM response are included.  Streaming
    flag is intentionally excluded — the same logical request can be
    served as either streaming or non-streaming from cache.
    """
    # Prefer the central serializer for internal slotted dataclasses.  Request
    # and Message intentionally use ``slots=True``, so ``__dict__``-based
    # duck-typing does not work on the proxy hot path.
    if _is_internal_request(request):
        from lattice.core.serialization import request_to_dict

        raw = request_to_dict(request)
    elif hasattr(request, "to_dict"):
        raw = request.to_dict()
    elif dataclasses.is_dataclass(request):
        raw = dataclasses.asdict(request)
    elif isinstance(request, dict):
        raw = request
    elif hasattr(request, "__dict__"):
        raw = vars(request)
    else:
        raw = dict(request)

    # Normalize messages
    messages: list[dict[str, Any]] = []
    raw_msgs = raw.get("messages", [])
    for m in raw_msgs:
        if hasattr(m, "to_dict"):
            m = m.to_dict()
        msg: dict[str, Any] = {}
        for k in ("role", "content", "name", "tool_calls", "tool_call_id"):
            v = m.get(k) if isinstance(m, dict) else getattr(m, k, None)
            if v is not None:
                msg[k] = v
        messages.append(msg)

    key_obj: dict[str, Any] = {
        "model": raw.get("model", ""),
        "messages": messages,
    }

    for k in (
        "temperature",
        "max_tokens",
        "top_p",
        "tools",
        "tool_choice",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "response_format",
    ):
        v = raw.get(k)
        if v is not None:
            key_obj[k] = v

    return key_obj


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _tokenize_for_semantics(text: str) -> frozenset[str]:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return frozenset(t for t in tokens if len(t) >= 2)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def _normalized_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _is_internal_request(request: Any) -> bool:
    """Return True for ``lattice.core.transport.Request`` without hard import."""
    return (
        request.__class__.__name__ == "Request"
        and request.__class__.__module__ == "lattice.core.transport"
    )


def compute_cache_key(request: Any) -> str:
    """Compute a stable SHA-256 hex digest for *request*."""
    canonical = _canonical_request(request)
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Content class detection
# ---------------------------------------------------------------------------


def _detect_content_class(request: Any) -> ContentClass:
    """Classify request content based on message content."""
    canonical = _canonical_request(request)
    messages = canonical.get("messages", [])

    classes_detected: set[ContentClass] = set()

    # TOOL_OUTPUT: any message has tool_calls or tool_call_id
    for m in messages:
        if m.get("tool_calls") or m.get("tool_call_id"):
            classes_detected.add(ContentClass.TOOL_OUTPUT)
            break

    # CODE: >30% of text content is inside ``` fences
    total_text_len = 0
    code_fence_len = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total_text_len += len(content)
            for fence in re.findall(r"```[\s\S]*?```", content):
                code_fence_len += len(fence)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    total_text_len += len(text)
                    for fence in re.findall(r"```[\s\S]*?```", text):
                        code_fence_len += len(fence)
    if total_text_len > 0 and code_fence_len / total_text_len > 0.30:
        classes_detected.add(ContentClass.CODE)

    # JSON: any message content starts with { or [ and is valid JSON
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            stripped = content.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    json.loads(stripped)
                    classes_detected.add(ContentClass.JSON)
                    break
                except json.JSONDecodeError:
                    pass

    if len(classes_detected) > 1:
        return ContentClass.MIXED
    if len(classes_detected) == 1:
        return next(iter(classes_detected))
    return ContentClass.PLAIN_TEXT


# ---------------------------------------------------------------------------
# Semantic fingerprint
# ---------------------------------------------------------------------------


def _compute_semantic_fingerprint(request: Any) -> _SemanticFingerprint:
    canonical = _canonical_request(request)
    messages = canonical.get("messages", [])

    roles = tuple(str(m.get("role", "")) for m in messages)

    tools = canonical.get("tools")
    tool_schema_hash = None
    if tools:
        tool_json = json.dumps(tools, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        tool_schema_hash = hashlib.sha256(tool_json.encode("utf-8")).hexdigest()

    parts: list[str] = []
    for m in messages:
        parts.append(str(m.get("role", "")))
        parts.append(_normalize_text(m.get("content")))
        parts.append(_normalize_text(m.get("name")))
        parts.append(_normalize_text(m.get("tool_call_id")))
        if m.get("tool_calls"):
            parts.append(_normalize_text(m.get("tool_calls")))

    for key in ("tools", "tool_choice", "response_format", "stop"):
        if canonical.get(key) is not None:
            parts.append(_normalize_text(canonical.get(key)))

    raw_text = " ".join(parts).lower()
    normalized_text = re.sub(r"[^a-z0-9\s]", "", raw_text)
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    token_set = _tokenize_for_semantics(normalized_text)

    content_class = _detect_content_class(request)

    has_tool_calls = any(m.get("tool_calls") or m.get("tool_call_id") for m in messages)
    has_images = False
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    has_images = True
                    break
        if has_images:
            break

    return _SemanticFingerprint(
        token_set=token_set,
        role_pattern=roles,
        tool_schema_hash=tool_schema_hash,
        normalized_text=normalized_text,
        content_class=content_class,
        message_count=len(messages),
        has_tools=tools is not None,
        has_tool_calls=has_tool_calls,
        has_images=has_images,
    )


def _compute_similarity(query: _SemanticFingerprint, candidate: _SemanticFingerprint) -> float:
    """Weighted similarity score between two fingerprints."""
    jaccard = _jaccard(query.token_set, candidate.token_set)
    role_score = 1.0 if query.role_pattern == candidate.role_pattern else 0.0

    if query.has_tools or candidate.has_tools:
        tool_score = 1.0 if query.tool_schema_hash == candidate.tool_schema_hash else 0.0
    else:
        tool_score = 1.0

    text_score = _normalized_similarity(query.normalized_text, candidate.normalized_text)

    return jaccard * 0.40 + role_score * 0.20 + tool_score * 0.20 + text_score * 0.20


# ---------------------------------------------------------------------------
# Exact response cache
# ---------------------------------------------------------------------------


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
            approx_hit_rate = (
                self._approximate_hits / approx_total if approx_total > 0 else 0.0
            )
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
