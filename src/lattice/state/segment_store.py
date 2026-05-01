"""Segment store for LATTICE state plane.

Stores canonical segments (immutable, hash-addressed chunks) separately
from session metadata. This enables:
- Cross-session deduplication
- Refcounted segment lifecycle
- Large-value offload (object store / KV)

Backends:
- MemorySegmentStore: dict-backed, for local dev
- RedisSegmentStore: persistent, for production
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from typing import Any, Protocol

import structlog

from lattice.protocol.segments import Segment

logger = structlog.get_logger()


# =============================================================================
# SegmentRecord
# =============================================================================


@dataclasses.dataclass(slots=True)
class SegmentRecord:
    """Stored segment with metadata for lifecycle management."""

    hash: str
    segment: Segment
    refcount: int = 1
    created_at: float = dataclasses.field(default_factory=time.time)
    last_accessed_at: float = dataclasses.field(default_factory=time.time)

    def touch(self) -> None:
        self.last_accessed_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash,
            "segment": self.segment.to_dict(),
            "refcount": self.refcount,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentRecord:
        return cls(
            hash=data["hash"],
            segment=Segment.from_dict(data["segment"]),
            refcount=data.get("refcount", 1),
            created_at=data.get("created_at", 0.0),
            last_accessed_at=data.get("last_accessed_at", 0.0),
        )


# =============================================================================
# SegmentStore Protocol
# =============================================================================


class SegmentStore(Protocol):
    """Protocol for segment storage backends."""

    async def get(self, hash: str) -> Segment | None:
        """Retrieve a segment by its hash."""
        ...

    async def put(self, segment: Segment) -> bool:
        """Store a segment. Returns True if newly stored, False if already exists."""
        ...

    async def inc_ref(self, hash: str) -> bool:
        """Increment refcount. Returns True if segment exists."""
        ...

    async def dec_ref(self, hash: str) -> bool:
        """Decrement refcount. Returns True if segment was removed (refcount=0)."""
        ...

    async def delete(self, hash: str) -> bool:
        """Force delete a segment."""
        ...

    async def keys(self) -> list[str]:
        """Return all segment hashes."""
        ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


# =============================================================================
# MemorySegmentStore
# =============================================================================


class MemorySegmentStore:
    """In-memory segment store with refcounting and TTL eviction."""

    def __init__(self, ttl_seconds: int = 3600, max_segments: int = 100_000) -> None:
        self._segments: dict[str, SegmentRecord] = {}
        self._ttl_seconds = ttl_seconds
        self._max_segments = max_segments
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def get(self, hash: str) -> Segment | None:
        async with self._lock:
            record = self._segments.get(hash)
            if record is None:
                return None
            if (time.time() - record.last_accessed_at) > self._ttl_seconds:
                del self._segments[hash]
                return None
            record.touch()
            return record.segment

    async def put(self, segment: Segment) -> bool:
        hash = segment.hash
        async with self._lock:
            if hash in self._segments:
                self._segments[hash].refcount += 1
                return False
            self._segments[hash] = SegmentRecord(hash=hash, segment=segment)
            # LRU eviction
            if len(self._segments) > self._max_segments:
                oldest = min(
                    self._segments.values(),
                    key=lambda r: r.last_accessed_at,
                )
                if oldest.refcount <= 1:
                    del self._segments[oldest.hash]
            return True

    async def inc_ref(self, hash: str) -> bool:
        async with self._lock:
            record = self._segments.get(hash)
            if record is None:
                return False
            record.refcount += 1
            return True

    async def dec_ref(self, hash: str) -> bool:
        async with self._lock:
            record = self._segments.get(hash)
            if record is None:
                return False
            record.refcount -= 1
            if record.refcount <= 0:
                del self._segments[hash]
                return True
            return False

    async def delete(self, hash: str) -> bool:
        async with self._lock:
            if hash in self._segments:
                del self._segments[hash]
                return True
            return False

    async def keys(self) -> list[str]:
        async with self._lock:
            return list(self._segments.keys())

    @property
    def segment_count(self) -> int:
        return len(self._segments)

    @property
    def stats(self) -> dict[str, Any]:
        total_refcount = sum(r.refcount for r in self._segments.values())
        return {
            "segments": len(self._segments),
            "total_refcount": total_refcount,
            "max_segments": self._max_segments,
        }


# =============================================================================
# RedisSegmentStore
# =============================================================================

_REDIS_AVAILABLE = False
try:
    import redis.asyncio as redis

    _REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore[assignment]


class RedisSegmentStore:
    """Redis-backed segment store for multi-process deployments."""

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        prefix: str = "lattice:segment:",
        ttl_seconds: int = 3600,
    ) -> None:
        if not _REDIS_AVAILABLE:
            raise ImportError("Redis is not installed. Install with: pip install redis")
        self.url = url
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        self._client: Any | None = None
        self._log = logger.bind(module="redis_segment_store")

    async def start(self) -> None:
        if self._client is None:
            self._client = redis.from_url(self.url, decode_responses=True)
            await self._client.ping()  # type: ignore[misc]
            self._log.info("redis_connected", url=self.url)

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def get(self, hash: str) -> Segment | None:
        if self._client is None:
            raise RuntimeError("Redis not connected")
        key = f"{self.prefix}{hash}"
        raw = await self._client.get(key)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            record = SegmentRecord.from_dict(data)
            record.touch()
            await self._client.setex(key, self.ttl_seconds, json.dumps(record.to_dict()))
            return record.segment
        except (json.JSONDecodeError, KeyError):
            await self._client.delete(key)
            return None

    async def put(self, segment: Segment) -> bool:
        if self._client is None:
            raise RuntimeError("Redis not connected")
        key = f"{self.prefix}{segment.hash}"
        existing = await self._client.get(key)
        if existing is not None:
            try:
                data = json.loads(existing)
                data["refcount"] = data.get("refcount", 1) + 1
                data["last_accessed_at"] = time.time()
                await self._client.setex(key, self.ttl_seconds, json.dumps(data))
                return False
            except json.JSONDecodeError:
                pass
        record = SegmentRecord(hash=segment.hash, segment=segment)
        await self._client.setex(key, self.ttl_seconds, json.dumps(record.to_dict()))
        return True

    async def inc_ref(self, hash: str) -> bool:
        if self._client is None:
            raise RuntimeError("Redis not connected")
        key = f"{self.prefix}{hash}"
        raw = await self._client.get(key)
        if raw is None:
            return False
        try:
            data = json.loads(raw)
            data["refcount"] = data.get("refcount", 1) + 1
            await self._client.setex(key, self.ttl_seconds, json.dumps(data))
            return True
        except json.JSONDecodeError:
            return False

    async def dec_ref(self, hash: str) -> bool:
        if self._client is None:
            raise RuntimeError("Redis not connected")
        key = f"{self.prefix}{hash}"
        raw = await self._client.get(key)
        if raw is None:
            return False
        try:
            data = json.loads(raw)
            data["refcount"] = max(0, data.get("refcount", 1) - 1)
            if data["refcount"] <= 0:
                await self._client.delete(key)
                return True
            await self._client.setex(key, self.ttl_seconds, json.dumps(data))
            return False
        except json.JSONDecodeError:
            await self._client.delete(key)
            return True

    async def delete(self, hash: str) -> bool:
        if self._client is None:
            return False
        key = f"{self.prefix}{hash}"
        result = await self._client.delete(key)
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


__all__ = [
    "SegmentRecord",
    "SegmentStore",
    "MemorySegmentStore",
    "RedisSegmentStore",
]
