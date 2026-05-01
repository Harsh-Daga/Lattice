"""Session management for LATTICE — production-grade with manifest support.

Tracks anchored context across multi-turn conversations using manifest-based
segment storage. Each session stores:
- Full conversation history (for backward compatibility)
- Current manifest (canonical segments)
- Provider/model metadata
- Versioned state for optimistic concurrency

Storage backends:
- MemorySessionStore: dict-backed, for local dev
- RedisSessionStore: persistent, for production

All SessionStore operations are async to support I/O backends.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import secrets
import time
from typing import Any, Protocol

from lattice.core.transport import Message, Role
from lattice.protocol.manifest import Manifest

# =============================================================================
# Session
# =============================================================================

@dataclasses.dataclass(slots=True)
class Session:
    """Tracks anchored context across turns.

    Attributes:
        session_id: Unique identifier (cryptographically strong).
        created_at: Unix timestamp when session was created.
        last_accessed_at: Unix timestamp of last access.
        messages: Full conversation history (newest last).
        manifest: Current manifest (canonical segments).
        provider: Target provider name.
        model: Model identifier used.
        system_prompt: Cached system prompt text.
        tool_schemas: Static tool definitions for this session.
        metadata: Arbitrary session metadata.
        client_info: Last known client metadata/address information.
        connection_id: Current transport connection identifier.
        version: Optimistic concurrency version.
    """

    session_id: str
    created_at: float
    last_accessed_at: float
    messages: list[Message]
    provider: str = "openai"
    model: str = ""
    system_prompt: str | None = None
    tool_schemas: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    client_info: dict[str, Any] = dataclasses.field(default_factory=dict)
    connection_id: str | None = None
    manifest: Manifest | None = None
    version: int = 0

    @property
    def message_count(self) -> int:
        """Number of messages in the conversation."""
        return len(self.messages)

    @property
    def token_estimate(self) -> int:
        """Rough token estimate of all messages."""
        if self.manifest:
            return self.manifest.token_estimate
        return sum(msg.token_estimate for msg in self.messages)

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if session has exceeded TTL."""
        return (time.time() - self.last_accessed_at) > ttl_seconds

    def touch(self) -> None:
        """Update last_accessed_at to now."""
        self.last_accessed_at = time.time()

    def bump_version(self) -> None:
        """Increment optimistic concurrency version."""
        self.version += 1

    def record_cache_hit(self, cached_tokens: int) -> None:
        """Record the number of tokens that were cache hits in the last turn."""
        self.metadata["last_cache_hit_tokens"] = cached_tokens

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for JSON storage."""
        from lattice.core.serialization import message_to_dict

        data: dict[str, Any] = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "provider": self.provider,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "tool_schemas": self.tool_schemas,
            "metadata": self.metadata,
            "client_info": self.client_info,
            "connection_id": self.connection_id,
            "version": self.version,
            "messages": [message_to_dict(msg) for msg in self.messages],
        }
        if self.manifest:
            data["manifest"] = self.manifest.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        """Deserialize from a dict."""
        from lattice.core.serialization import message_from_dict

        manifest = None
        if "manifest" in data:
            manifest = Manifest.from_dict(data["manifest"])
        return cls(
            session_id=data["session_id"],
            created_at=data["created_at"],
            last_accessed_at=data["last_accessed_at"],
            provider=data.get("provider", "openai"),
            model=data.get("model", ""),
            system_prompt=data.get("system_prompt"),
            tool_schemas=data.get("tool_schemas", []),
            metadata=data.get("metadata", {}),
            client_info=data.get("client_info", {}),
            connection_id=data.get("connection_id"),
            manifest=manifest,
            version=data.get("version", 0),
            messages=[message_from_dict(msg) for msg in data.get("messages", [])],
        )


# =============================================================================
# SessionStore Protocol
# =============================================================================

class SessionStore(Protocol):
    """Protocol for session storage backends."""

    async def get(self, session_id: str) -> Session | None:
        """Retrieve a session by ID."""
        ...

    async def set(self, session: Session) -> bool:
        """Store or update a session.

        Returns True if stored, False if version conflict (CAS failure).
        """
        ...

    async def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if deleted, False if not found."""
        ...

    async def expire(self, ttl_seconds: int) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        ...

    async def keys(self) -> list[str]:
        """Return all session IDs (for debugging/admin)."""
        ...

    async def start(self) -> None:
        """Initialize the store (connect to backend, etc.)."""
        ...

    async def stop(self) -> None:
        """Shutdown the store."""
        ...


# =============================================================================
# MemorySessionStore
# =============================================================================

class MemorySessionStore:
    """In-memory session store with TTL eviction and CAS versioning.

    Thread-safe via asyncio.Lock. Suitable for single-process deployments.
    Background expiry task runs every 60 seconds.
    """

    def __init__(self, ttl_seconds: int = 3600, max_sessions: int = 10000) -> None:
        self._sessions: dict[str, Session] = {}
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions
        self._lock = asyncio.Lock()
        self._expiry_task: asyncio.Task[Any] | None = None

    async def start(self) -> None:
        """Start background expiry task."""
        if self._expiry_task is None:
            self._expiry_task = asyncio.create_task(self._expiry_loop())

    async def stop(self) -> None:
        """Stop background expiry task."""
        if self._expiry_task:
            self._expiry_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._expiry_task
            self._expiry_task = None

    async def _expiry_loop(self) -> None:
        """Background task that removes expired sessions every 60s."""
        while True:
            try:
                await asyncio.sleep(60)
                await self.expire(self._ttl_seconds)
            except asyncio.CancelledError:
                break
            except Exception:
                # Log but don't crash the expiry task
                pass

    async def get(self, session_id: str) -> Session | None:
        """Retrieve a session by ID.

        Also updates last_accessed_at and checks expiry.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired(self._ttl_seconds):
                del self._sessions[session_id]
                return None
            session.touch()
            return session

    async def set(self, session: Session) -> bool:
        """Store or update a session with optimistic concurrency.

        Returns True on success, False if a newer version exists.
        """
        session.touch()
        async with self._lock:
            existing = self._sessions.get(session.session_id)
            if existing is not None and existing.version > session.version:
                # CAS failure — existing is newer
                return False
            self._sessions[session.session_id] = session
            # LRU eviction if exceeding max
            if len(self._sessions) > self._max_sessions:
                # Remove oldest by last_accessed_at
                oldest = min(
                    self._sessions.values(), key=lambda s: s.last_accessed_at
                )
                del self._sessions[oldest.session_id]
            return True

    async def delete(self, session_id: str) -> bool:
        """Delete a session."""
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    async def expire(self, ttl_seconds: int) -> int:
        """Remove expired sessions."""
        now = time.time()
        removed = 0
        async with self._lock:
            expired = [
                sid
                for sid, session in self._sessions.items()
                if (now - session.last_accessed_at) > ttl_seconds
            ]
            for sid in expired:
                del self._sessions[sid]
                removed += 1
        return removed

    async def keys(self) -> list[str]:
        """Return all session IDs."""
        async with self._lock:
            return list(self._sessions.keys())

    @property
    def session_count(self) -> int:
        """Current number of stored sessions."""
        return len(self._sessions)


# =============================================================================
# SessionManager
# =============================================================================

class SessionManager:
    """High-level session management API.

    SessionManager wraps a SessionStore and provides:
    - Automatic session creation with strong IDs
    - Manifest-based canonicalization
    - Delta computation for multi-turn conversations
    - Expiry handling
    - Optimistic concurrency
    """

    def __init__(self, store: SessionStore, ttl_seconds: int = 3600) -> None:
        self.store = store
        self.ttl_seconds = ttl_seconds

    async def create_session(
        self,
        provider: str,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Session:
        """Create a new session from an initial request.

        Args:
            provider: Target provider name.
            model: Model identifier.
            messages: Full conversation history.
            tools: Optional tool definitions.

        Returns:
            The newly created Session.
        """
        from lattice.protocol.manifest import manifest_from_messages

        session_id = self._generate_session_id()
        manifest = manifest_from_messages(
            session_id=session_id,
            messages=[
                {"role": m.role.value if isinstance(m.role, Role) else str(m.role), "content": m.content}
                for m in messages
            ],
            tools=tools,
            model=model,
            provider=provider,
        )

        # Extract system prompt
        system_texts: list[str] = []
        for msg in messages:
            if msg.role == Role.SYSTEM or msg.role == "system":
                system_texts.append(msg.content)

        session = Session(
            session_id=session_id,
            created_at=time.time(),
            last_accessed_at=time.time(),
            provider=provider,
            model=model,
            messages=messages,
            system_prompt="\n".join(system_texts) if system_texts else None,
            tool_schemas=tools or [],
            manifest=manifest,
            version=0,
        )
        await self.store.set(session)
        return session

    async def get_or_create_session(
        self,
        session_id: str | None,
        provider: str,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[Session, bool]:
        """Get existing session or create a new one.

        Args:
            session_id: Existing session ID, or None to always create.
            provider: Target provider.
            model: Model identifier.
            messages: Current messages.
            tools: Tool definitions.

        Returns:
            Tuple of (session, was_created).
        """
        if session_id:
            existing = await self.store.get(session_id)
            if existing is not None:
                return existing, False

        session = await self.create_session(provider, model, messages, tools)
        return session, True

    async def update_session(
        self,
        session_id: str,
        messages: list[Message],
        manifest: Manifest | None = None,
    ) -> Session | None:
        """Update an existing session with new messages and optional manifest.

        If no manifest is provided, the manifest is automatically rebuilt
        from the current messages to keep session state canonical.

        Args:
            session_id: Session to update.
            messages: Updated message list.
            manifest: Optional new manifest (auto-built if None).

        Returns:
            Updated session, or None if not found.
        """
        session = await self.store.get(session_id)
        if session is None:
            return None

        session.messages = messages
        session.bump_version()

        # Rebuild system prompt
        system_texts: list[str] = []
        for msg in messages:
            if msg.role == Role.SYSTEM or msg.role == "system":
                system_texts.append(msg.content)
        session.system_prompt = "\n".join(system_texts) if system_texts else None

        # Rebuild manifest from messages if not explicitly provided
        if manifest is None:
            manifest = self._rebuild_manifest(session)
        session.manifest = manifest

        success = await self.store.set(session)
        if not success:
            # CAS failure — session was modified concurrently
            # Re-fetch and retry once
            session = await self.store.get(session_id)
            if session is None:
                return None
            session.messages = messages
            session.bump_version()
            session.system_prompt = "\n".join(system_texts) if system_texts else None
            manifest = manifest or self._rebuild_manifest(session)
            session.manifest = manifest
            await self.store.set(session)
        return session

    def _rebuild_manifest(self, session: Session) -> Manifest:
        """Rebuild a canonical manifest from session state."""
        from lattice.protocol.manifest import build_manifest, manifest_from_messages

        base = manifest_from_messages(
            session_id=session.session_id,
            messages=[
                {"role": m.role.value if isinstance(m.role, Role) else str(m.role), "content": m.content}
                for m in session.messages
            ],
            tools=session.tool_schemas or None,
            model=session.model,
            provider=session.provider,
        )
        # Align manifest version with session version for optimistic concurrency
        return build_manifest(
            session_id=session.session_id,
            segments=base.segments,
            anchor_version=session.version,
            metadata=base.metadata,
            manifest_id=base.manifest_id,
        )

    async def compute_delta(
        self, session_id: str, new_messages: list[Message]
    ) -> list[Message]:
        """Compute which messages are new compared to the session.

        Simple append-only logic: messages after session.message_count
        are new.

        Args:
            session_id: The session to compare against.
            new_messages: The complete message list from the client.

        Returns:
            The subset of new_messages that are new.
        """
        session = await self.store.get(session_id)
        if session is None:
            # No session — everything is new
            return new_messages

        existing_count = session.message_count
        if len(new_messages) <= existing_count:
            # Client sent fewer or same messages — possible truncation
            # Return nothing (full replacement handled upstream)
            return []

        return new_messages[existing_count:]

    async def migrate_session(self, session_id: str, new_connection_id: str) -> bool:
        """Re-bind an existing session to a new connection ID."""
        session = await self.store.get(session_id)
        if session is None:
            return False
        session.connection_id = new_connection_id
        session.bump_version()
        return await self.store.set(session)

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a cryptographically strong session ID."""
        return f"lattice-{secrets.token_urlsafe(16)}"
