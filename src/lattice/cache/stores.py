"""Cache storage backends."""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from typing import Any, Protocol

import structlog

from lattice.cache.fingerprint import CachedResponse


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
            raise ImportError("Redis is not installed. Install with: pip install redis")
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
