from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from lattice.transport.serialization import message_to_dict, request_from_dict, response_to_dict
from lattice.transport.types import Message, Request, Response

Handler = Callable[..., Awaitable[Any]]

def detect_new_messages(existing: list[Message], incoming: list[Message]) -> list[Message]:
    """Detect newly appended messages against existing conversation state."""
    if len(incoming) <= len(existing):
        return []
    if len(existing) == 0:
        return list(incoming)
    for _i, (a, b) in enumerate(zip(existing, incoming, strict=False)):
        if a.content != b.content or a.role != b.role:
            return list(incoming)
    return list(incoming[len(existing) :])


def deserialize_openai_request(body: dict[str, Any]) -> Request:
    """Convert OpenAI JSON request body into internal request."""
    return request_from_dict(body)


def serialize_messages(request: Request) -> list[dict[str, Any]]:
    """Convert internal request messages into OpenAI list format."""
    return [message_to_dict(m) for m in request.messages]


def serialize_openai_response(response: Response, request: Request) -> dict[str, Any]:
    """Convert internal response into OpenAI-compatible response body."""
    return response_to_dict(response, request_model=request.model)


def is_local_origin(request: Any) -> bool:
    """Check whether request appears to come from localhost."""
    host = request.headers.get("host", "")
    remote = request.client.host if request.client else ""
    forwarded = request.headers.get("x-forwarded-for", "")
    return (
        host.startswith("127.0.0.1")
        or host.startswith("localhost")
        or remote in ("127.0.0.1", "::1")
        or forwarded.startswith("127.0.0.1")
    )

