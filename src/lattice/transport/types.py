"""Transport abstractions for LATTICE.

Defines the core data types and protocols for LLM request/response
processing. These types are intentionally model-neutral — they support
OpenAI, Anthropic, and any other LLM API format.

Module Design
-------------
- Message: A single message in a conversation (role + content).
- Request: An incoming LLM request with messages, model, and config.
- Response: An outgoing LLM response with content and metadata.
- Transform: The protocol that all optimizers implement.

Performance Notes
-----------------
- Message.content remains `str` for backward compatibility.
- Message.content_parts provides typed ContentPart support for multimodal,
  tool blocks, and reasoning. The two fields are kept in sync.
- Request model uses frozen=False to allow transforms to mutate in-place.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from lattice.core.context import TransformContext
    from lattice.protocol.content import ContentPart

from lattice.core.result import Result

# =============================================================================
# Message
# =============================================================================


class Role(str, enum.Enum):
    """Standard message roles.

    Using a string enum for JSON serialization compatibility
    while maintaining type safety.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclasses.dataclass(slots=True)
class Message:
    """A single message in an LLM conversation.

    Backward Compatibility
    ----------------------
    ``content`` is still the primary string field. ``content_parts`` is
    an optional extension for multimodal/tool/reasoning content. When
    ``content_parts`` is set, ``content`` returns the concatenated text.
    When ``content`` is set directly, ``content_parts`` is cleared.
    """

    role: Role | str
    content: str = ""
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    _content_parts: list[ContentPart] | None = dataclasses.field(default=None, repr=False)

    @property
    def content_parts(self) -> list[ContentPart]:
        """Typed content parts. Falls back to [TextPart(content)] if not set."""
        if self._content_parts is None:
            from lattice.protocol.content import content_to_parts

            return content_to_parts(self.content)
        return list(self._content_parts)

    @content_parts.setter
    def content_parts(self, parts: list[ContentPart] | None) -> None:
        self._content_parts = parts
        if parts is not None:
            from lattice.protocol.content import parts_to_str

            self.content = parts_to_str(parts)

    @property
    def token_estimate(self) -> int:
        """Rough token estimate. 1 token ~ 4 chars for English text."""
        if self._content_parts:
            text_len = sum(
                len(getattr(p, "text", getattr(p, "content", ""))) for p in self._content_parts
            )
            return max(1, text_len // 4)
        return max(1, len(self.content) // 4)

    def copy(self) -> Message:
        """Return a shallow copy of this message."""
        return dataclasses.replace(
            self,
            metadata=self.metadata.copy(),
            tool_calls=(list(self.tool_calls) if self.tool_calls else None),
            _content_parts=(list(self._content_parts) if self._content_parts else None),
        )

    def __repr__(self) -> str:
        content = self.content
        if len(content) > 80:
            content = content[:60] + "... [truncated]"
        role = self.role.value if isinstance(self.role, Role) else self.role
        parts_hint = f" parts={len(self._content_parts)}" if self._content_parts else ""
        return f"Message(role={role!r}, content={content!r}{parts_hint})"


# =============================================================================
# Request
# =============================================================================


@dataclasses.dataclass(slots=True)
class Request:
    """An incoming LLM request."""

    messages: list[Message] = dataclasses.field(default_factory=list)
    model: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    stream: bool = False
    stop: list[str] | None = None
    extra_headers: dict[str, str] = dataclasses.field(default_factory=dict)
    extra_body: dict[str, Any] = dataclasses.field(default_factory=dict)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def system_message(self) -> Message | None:
        """Return the first system message, if any."""
        for msg in self.messages:
            if msg.role == Role.SYSTEM or msg.role == "system":
                return msg
        return None

    @property
    def user_messages(self) -> list[Message]:
        """Return all user messages."""
        return [msg for msg in self.messages if msg.role == Role.USER or msg.role == "user"]

    @property
    def token_estimate(self) -> int:
        """Rough estimate of total input tokens."""
        overhead = len(self.messages) * 3
        content = sum(msg.token_estimate for msg in self.messages)
        tools = 0
        if self.tools:
            tools = sum(len(str(t)) // 4 for t in self.tools)
        return overhead + content + tools

    @property
    def is_tool_conversation(self) -> bool:
        """Whether this request involves tool definitions."""
        return self.tools is not None and len(self.tools) > 0

    def copy(self) -> Request:
        """Return a shallow copy with copied mutable fields."""
        return dataclasses.replace(
            self,
            messages=[msg.copy() for msg in self.messages],
            extra_headers=self.extra_headers.copy(),
            extra_body=self.extra_body.copy(),
            metadata=self.metadata.copy(),
            tools=(list(self.tools) if self.tools else None),
            stop=(list(self.stop) if self.stop else None),
        )

    def add_message(self, message: Message) -> None:
        """Append a message in-place."""
        self.messages.append(message)

    def __repr__(self) -> str:
        msg_count = len(self.messages)
        return (
            f"Request(model={self.model!r}, "
            f"messages={msg_count}, "
            f"tools={len(self.tools) if self.tools else 0}, "
            f"stream={self.stream})"
        )


# =============================================================================
# Response
# =============================================================================


@dataclasses.dataclass(slots=True)
class Response:
    """An LLM response."""

    content: str = ""
    role: str = "assistant"
    model: str = ""
    usage: dict[str, int] = dataclasses.field(default_factory=dict)
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Rough token estimate for response content."""
        return max(1, len(self.content) // 4)


# =============================================================================
# Transform Protocol
# =============================================================================


class Transform(Protocol):
    """Protocol for all optimization transforms.

    Every transform in LATTICE implements this interface. Transforms are
    pure functions that take a Request and context, and return a modified
    Request or an error.

    Design Principles
    -----------------
    - **Deterministic:** Given the same input, produce the same output.
    - **Idempotent (preferred):** Applying twice shouldn't double effects.
    - **No side effects:** Transforms don't hit the network or filesystem.
    - **Fast:** Transforms should complete in <1ms for typical inputs.
    """

    name: str
    enabled: bool = True
    priority: int = 50

    def process(self, request: Request, context: TransformContext) -> Result[Request, Any]:
        """Process a request, returning modified request or error."""
        ...

    def can_process(self, _request: Request, _context: TransformContext) -> bool:
        """Check whether this transform can handle the request."""
        return self.enabled


class SyncTransform(Protocol):
    """Synchronous version of Transform."""

    name: str
    enabled: bool = True
    priority: int = 50

    def process(self, request: Request, context: TransformContext) -> Result[Request, Any]: ...

    def can_process(self, _request: Request, _context: TransformContext) -> bool:
        return self.enabled
