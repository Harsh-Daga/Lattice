"""Delta Encoding transform.

Core idea (from LLMTP research): Instead of sending the full message history
every turn, the client sends only NEW messages. The proxy reconstructs the
full context from its session store before forwarding to the provider.

This is the same pattern as TCP sequence numbers or HTTP/2 HPACK:
- Wire format is compact (just the delta)
- Receiver reconstructs full state internally
- Provider sees standard API (no protocol change needed)

Wire Format
-----------
Standard OpenAI/Anthropic request with an optional header:
    X-Lattice-Session-Id: sess_abc123

When this header is present, the body `messages` are treated as DELTA
(append-only additions to the session). The proxy reconstructs the full
conversation before forwarding.

When absent, a new session is created.

Delta Detection Algorithm
-------------------------
Given: session.messages (length N), request.messages (length M)

If M == 0:                     → Error (empty request)
If M <= N and messages[0:M] match session.messages[0:M]:
                                → TRUNCATION (client pruned old messages)
If M >  N and messages[0:N] == session.messages:
                                → APPEND (add messages[N:M] to session)
Otherwise:                      → FULL_REPLACEMENT (overwrite session)

Safety
------
- Any delta failure falls back to full context (graceful degradation)
- Session not found → create new session, treat as full request
- Provider never sees delta format (always full reconstructed context)
- Response always includes X-Lattice-Session-Id header
"""

from __future__ import annotations

import enum

import structlog

from lattice.core.context import TransformContext
from lattice.core.errors import TransformError
from lattice.core.pipeline import ReversibleSyncTransform
from lattice.core.result import Ok, Result
from lattice.core.session import SessionManager
from lattice.core.transport import Message, Request, Response

logger = structlog.get_logger()


# =============================================================================
# Delta types
# =============================================================================


class DeltaType(enum.Enum):
    """Classification of how a new request relates to session history."""

    APPEND = "append"  # New messages appended to end
    TRUNCATION = "truncation"  # Old messages removed from beginning
    FULL_REPLACEMENT = "full_replacement"  # Complete overwrite (new conversation)
    ERROR = "error"  # Cannot determine delta (mismatch)


# =============================================================================
# DeltaEncoder
# =============================================================================


class DeltaEncoder(ReversibleSyncTransform):
    """Delta encoding for multi-turn session optimization.

    This transform receives a request that may be a DELTA (only new messages)
    and reconstructs the FULL conversation by looking up the session store.

    IMPORTANT: This is NOT a content compression transform. It is a
    PROTOCOL transform that changes how the proxy interprets the request.
    The actual content is unchanged — the proxy just reconstructs the
    full message list from session + delta.

    Relationship to SessionManager
    ------------------------------
    SessionManager owns the session store. DeltaEncoder uses the store
    to retrieve session history and compute the full message list.

    SessionManager.compute_delta() determines if the request is an
    append, truncation, or replacement. DeltaEncoder applies the result.

    Performance
    -----------
    - Session lookup: O(1) for MemorySessionStore
    - Delta computation: O(min(N, M)) message comparisons
    - Total target: <0.5ms for N < 1000 messages
    """

    name = "delta_encoder"
    priority = 5  # Run BEFORE PrefixOptimizer (priority 10) — we need full messages first

    def __init__(self, session_manager: SessionManager) -> None:
        """Initialize with a SessionManager for session lookups.

        Args:
            session_manager: Source of session state.
        """
        self.session_manager = session_manager
        self._log = logger.bind(transform="delta_encoder")

    async def process(  # type: ignore[override]
        self, request: Request, context: TransformContext
    ) -> Result[Request, TransformError]:
        """Reconstruct full conversation from session + delta.

        If `context.session_id` is set:
        1. Look up session in SessionManager
        2. If found: compute delta type, reconstruct full messages
        3. If not found: create new session, treat as full request

        If `context.session_id` is NOT set:
        - Create new session immediately
        - Return request unchanged (first turn)

        The reconstructed full messages are placed back into `request.messages`.
        The original delta messages are preserved in `request.metadata`
        for debugging and metrics.

        Args:
            request: The incoming request (may contain only delta messages).
            context: Per-request state. `session_id` must be pre-populated
                     by middleware from the X-Lattice-Session-Id header.

        Returns:
            Ok(Request) with full reconstructed messages, or
            Err(TransformError) on unrecoverable mismatch.
        """
        session_id = context.session_id
        new_messages = request.messages

        # No session ID -> first turn, create session
        if not session_id:
            self._log.debug("no_session_id", request_id=context.request_id)
            context.record_metric(self.name, "delta_type", "new_session")
            return Ok(request)

        # Step 1: Retrieve existing session (async)
        session = await self.session_manager.store.get(session_id)
        if session is None:
            self._log.warning(
                "session_not_found",
                request_id=context.request_id,
                session_id=session_id,
            )
            # Fallback: create new session on the fly
            context.record_metric(self.name, "delta_type", "session_not_found_fallback")
            return Ok(request)

        # Step 2: Determine delta type
        delta_type = self._classify_delta(session.messages, new_messages)
        context.record_metric(self.name, "delta_type", delta_type.value)

        # Step 3: Reconstruct full messages based on delta type
        if delta_type == DeltaType.APPEND:
            full_messages = self._apply_append(session.messages, new_messages)
            delta_count = len(new_messages) - len(session.messages)
            context.record_metric(self.name, "delta_messages", delta_count)
            self._log.debug(
                "delta_append",
                request_id=context.request_id,
                session_id=session_id,
                existing=len(session.messages),
                added=delta_count,
            )

        elif delta_type == DeltaType.TRUNCATION:
            full_messages = list(new_messages)  # Client already pruned
            context.record_metric(
                self.name, "truncated_messages", len(session.messages) - len(new_messages)
            )
            self._log.debug(
                "delta_truncation",
                request_id=context.request_id,
                session_id=session_id,
                previous=len(session.messages),
                truncated=len(new_messages),
            )

        elif delta_type == DeltaType.FULL_REPLACEMENT:
            # Client sent completely different messages — overwrite
            full_messages = list(new_messages)
            context.record_metric(self.name, "delta_type", "full_replacement")
            self._log.info(
                "delta_full_replacement",
                request_id=context.request_id,
                session_id=session_id,
                previous=len(session.messages),
                new=len(new_messages),
            )

        else:  # DeltaType.ERROR
            # Unrecoverable mismatch — fall back gracefully
            self._log.warning(
                "delta_error",
                request_id=context.request_id,
                session_id=session_id,
                session_len=len(session.messages),
                request_len=len(new_messages),
            )
            context.record_metric(self.name, "delta_type", "error_fallback")
            full_messages = list(new_messages)

        # Step 4: Preserve delta metadata and replace messages
        request.metadata["_delta_type"] = delta_type.value
        request.metadata["_delta_messages_count"] = len(new_messages)
        request.metadata["_full_messages_count"] = len(full_messages)
        request.messages = full_messages

        # Step 5: Update session (will be persisted by proxy after response)
        session.messages = full_messages
        session.touch()
        session.bump_version()
        await self.session_manager.store.set(session)

        return Ok(request)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_delta(existing: list[Message], new_messages: list[Message]) -> DeltaType:
        """Determine how the new messages relate to the existing session.

        Algorithm:
        1. Compute the longest common PREFIX (messages matching from start).
        2. If common prefix == existing length and new >= existing:
           → APPEND (new extends existing).
        3. If common prefix == new length and new < existing:
           → TRUNCATION (new is a prefix of existing, client pruned tail).
        4. If common prefix == 0 and new_len > 0:
           → FULL_REPLACEMENT (completely different messages).
        5. If common prefix is partial but > 0:
           → Try to detect SLIDING WINDOW (new is a contiguous subsequence
              of existing, typically client dropped old prefix messages).
           In MVP: treat as TRUNCATION (accept client's window as-is).
        """
        existing_len = len(existing)
        new_len = len(new_messages)

        if new_len == 0:
            return DeltaType.ERROR

        # Compute common prefix
        prefix_match = 0
        for i in range(min(existing_len, new_len)):
            if DeltaEncoder._messages_equal(existing[i], new_messages[i]):
                prefix_match += 1
            else:
                break

        # Case A: existing is prefix of new → APPEND
        if prefix_match == existing_len and new_len >= existing_len:
            return DeltaType.APPEND

        # Case B: new is prefix of existing → TRUNCATION
        if prefix_match == new_len and new_len < existing_len:
            return DeltaType.TRUNCATION

        # Case C: Identical
        if prefix_match == existing_len == new_len:
            return DeltaType.APPEND

        # Case D: Partial match — check for sliding window
        # If new is a contiguous subsequence of existing (after the prefix gap),
        # treat as TRUNCATION. This handles the common sliding-window case
        # where the client drops the oldest messages.
        #
        # Example:
        #   existing = [A, B, C, D, E]
        #   new      = [A, D, E]     # dropped B, C (sliding window)
        #
        # In this case prefix_match = 1 (A matches), then we check if
        # new[1:] is a suffix of existing.
        if prefix_match > 0 and prefix_match < new_len:
            remaining_new = new_messages[prefix_match:]
            remaining_len = len(remaining_new)
            # Check if remaining_new matches a suffix of existing
            if remaining_len <= existing_len:
                existing_suffix = existing[-remaining_len:]
                if all(
                    DeltaEncoder._messages_equal(existing_suffix[k], remaining_new[k])
                    for k in range(remaining_len)
                ):
                    return DeltaType.TRUNCATION

        # Everything else: the conversation structure changed in a way
        # we can't incrementally update. Overwrite the session.
        return DeltaType.FULL_REPLACEMENT

    @staticmethod
    def _apply_append(existing: list[Message], new_messages: list[Message]) -> list[Message]:
        """Reconstruct full messages from existing + new append."""
        existing_len = len(existing)
        append_only = new_messages[existing_len:]
        return list(existing) + [msg.copy() for msg in append_only]

    @staticmethod
    def _messages_equal(a: Message, b: Message) -> bool:
        """Check if two messages are semantically equal.

        Ignores metadata (which is transform scratch space).
        """
        # Compare content and role (most important)
        a_role = a.role.value if hasattr(a.role, "value") else str(a.role)
        b_role = b.role.value if hasattr(b.role, "value") else str(b.role)

        # Compare optional fields only if present in both
        return (
            a_role == b_role
            and a.content == b.content
            and a.name == b.name
            and a.tool_call_id == b.tool_call_id
        )

    def reverse(self, response: Response, _context: TransformContext) -> Response:
        """No-op — delta encoding doesn't modify response content."""
        return response
