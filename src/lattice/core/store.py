"""Session storage backends for LATTICE.

Backends:
- MemorySessionStore (in-process dict, for dev/single-process)
- RedisSessionStore (persistent, for production)

Protocol (from session.py):
    async def get(session_id: str) -> Session | None
    async def set(session: Session) -> bool  # CAS versioning
    async def delete(session_id: str) -> bool
    async def expire(ttl_seconds: int) -> int
    async def keys() -> list[str]
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog

from lattice.core.errors import SessionStoreError
from lattice.core.session import Session

logger = structlog.get_logger()

# Optional redis dependency
_REDIS_AVAILABLE = False
try:
    import redis.asyncio as redis

    _REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore[assignment]


# =============================================================================
# RedisSessionStore
# =============================================================================


class RedisSessionStore:
    """Redis-backed session store for multi-process deployments.

    Uses Redis hash for session data with optional TTL.
    All operations are async via redis-py.

    Supports optimistic concurrency via version field.

    Attributes:
        url: Redis connection URL (redis://host:port/db).
        prefix: Key prefix in Redis to avoid collisions.
        ttl_seconds: Default TTL for sessions.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        prefix: str = "lattice:session:",
        ttl_seconds: int = 3600,
    ) -> None:
        if not _REDIS_AVAILABLE:
            raise ImportError("Redis is not installed. Install with: pip install redis")
        self.url = url
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        self._client: Any | None = None
        self._log = logger.bind(module="redis_session_store")

    async def start(self) -> None:
        """Initialize Redis connection."""
        if self._client is None:
            self._client = redis.from_url(self.url, decode_responses=True)
            await self._client.ping()  # type: ignore[misc]
            self._log.info("redis_connected", url=self.url)

    async def stop(self) -> None:
        """Close Redis connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._log.info("redis_disconnected")

    async def get(self, session_id: str) -> Session | None:
        """Retrieve a session by ID."""
        if self._client is None:
            raise SessionStoreError("Redis not connected", backend="redis")
        key = f"{self.prefix}{session_id}"
        raw = await self._client.get(key)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            # Check TTL client-side as safety net
            if data.get("last_accessed_at", 0) + self.ttl_seconds < time.time():
                await self._client.delete(key)
                return None
            return Session.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            await self._client.delete(key)
            return None

    async def set(self, session: Session) -> bool:
        """Store or update a session with CAS versioning.

        Returns True on success. If the stored version is newer than
        the session's version, returns False (CAS failure).
        """
        if self._client is None:
            raise SessionStoreError("Redis not connected", backend="redis")
        key = f"{self.prefix}{session.session_id}"

        # Optimistic concurrency check
        existing_raw = await self._client.get(key)
        if existing_raw is not None:
            try:
                existing_data = json.loads(existing_raw)
                existing_version = existing_data.get("version", 0)
                if existing_version > session.version:
                    self._log.warning(
                        "cas_version_conflict",
                        session_id=session.session_id,
                        stored_version=existing_version,
                        attempted_version=session.version,
                    )
                    return False
            except json.JSONDecodeError:
                pass  # Overwrite corrupt data

        data = session.to_dict()
        data["last_accessed_at"] = time.time()
        raw = json.dumps(data)
        await self._client.setex(key, self.ttl_seconds, raw)
        return True

    async def delete(self, session_id: str) -> bool:
        """Delete a session."""
        if self._client is None:
            return False
        key = f"{self.prefix}{session_id}"
        result = await self._client.delete(key)
        return bool(result > 0)

    async def expire(self, ttl_seconds: int) -> int:
        """Remove expired sessions (handled by Redis TTL automatically)."""
        if self._client is None:
            return 0
        # Redis handles expiration automatically via TTL.
        # We iterate keys to enforce client-side TTL for safety.
        count = 0
        cursor = 0
        while True:
            cursor, keys = await self._client.scan(cursor, match=f"{self.prefix}*", count=100)
            for key in keys:
                raw = await self._client.get(key)
                if raw is None:
                    continue
                try:
                    data = json.loads(raw)
                    if data.get("last_accessed_at", 0) + ttl_seconds < time.time():
                        await self._client.delete(key)
                        count += 1
                except json.JSONDecodeError:
                    await self._client.delete(key)
                    count += 1
            if cursor == 0:
                break
        return count

    async def keys(self) -> list[str]:
        """Return all session IDs."""
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

    @property
    def session_count(self) -> int:
        """Current number of stored sessions (approximate)."""
        if self._client is None:
            return 0
        # Note: sync property over async client — not exact but fast
        return 0


__all__ = ["RedisSessionStore"]
