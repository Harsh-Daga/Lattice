from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable
from typing import Any

Handler = Callable[..., Awaitable[Any]]


@dataclasses.dataclass(slots=True)
class ChatCompatDeps:
    """Dependencies required by chat completion compatibility handler."""

    config: Any
    pipeline: Any
    provider: Any
    session_manager: Any
    batching_engine: Any
    speculative_executor: Any
    semantic_cache: Any
    cost_estimator: Any
    auto_continuation: Any
    agent_stats: Any
    metrics: Any
    logger: Any
    deserialize_openai_request: Callable[[dict[str, Any]], Any]
    serialize_messages: Callable[[Any], list[dict[str, Any]]]
    serialize_openai_response: Callable[[Any, Any], dict[str, Any]]
    build_routing_headers: Callable[..., dict[str, str]]
    detect_new_messages: Callable[[list[Any], list[Any]], list[Any]]
    get_cache_planner: Callable[[str], Any]
    message_cls: Any
    provider_timeout_error: type[Exception]
    provider_error: type[Exception]
    sse_done: str
    maintenance: Any = None


_ws_lib: Any = None
