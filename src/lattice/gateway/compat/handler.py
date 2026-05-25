from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from lattice.gateway.server import LLMTPGateway

Handler = Callable[..., Awaitable[Any]]


class HTTPCompatHandler:
    """Handles OpenAI/Anthropic/Responses compatibility routes.

    The proxy can register concrete handlers and keep this class as a
    thin, testable delegation layer while extraction is in progress.
    """

    def __init__(
        self,
        gateway: LLMTPGateway,
        *,
        chat_completion_handler: Handler | None = None,
        anthropic_handler: Handler | None = None,
        responses_handler: Handler | None = None,
        models_handler: Handler | None = None,
    ) -> None:
        self.gateway = gateway
        self.chat_completion_handler = chat_completion_handler
        self.anthropic_handler = anthropic_handler
        self.responses_handler = responses_handler
        self.models_handler = models_handler

    async def handle_chat_completion(self, *args: Any, **kwargs: Any) -> Any:
        if self.chat_completion_handler is None:
            raise RuntimeError("chat_completion_handler is not configured")
        return await self.chat_completion_handler(*args, **kwargs)

    async def handle_anthropic_message(self, *args: Any, **kwargs: Any) -> Any:
        if self.anthropic_handler is None:
            raise RuntimeError("anthropic_handler is not configured")
        return await self.anthropic_handler(*args, **kwargs)

    async def handle_responses_api(self, *args: Any, **kwargs: Any) -> Any:
        if self.responses_handler is None:
            raise RuntimeError("responses_handler is not configured")
        return await self.responses_handler(*args, **kwargs)

    async def handle_models(self, *args: Any, **kwargs: Any) -> Any:
        if self.models_handler is None:
            raise RuntimeError("models_handler is not configured")
        return await self.models_handler(*args, **kwargs)
