"""Provider SDK wrappers for LATTICE.

Provides drop-in replacements for OpenAI and Anthropic SDK clients that
automatically compress requests through LATTICE before sending.

Usage:
    >>> import openai
    >>> from lattice.sdk import wrap_openai
    >>> base_client = openai.AsyncOpenAI(api_key="sk-...")
    >>> client = wrap_openai(base_client, lattice_config={...})
    >>> # All requests are automatically compressed
    >>> response = await client.chat.completions.create(
    ...     model="gpt-4",
    ...     messages=[{"role": "user", "content": "Hello"}],
    ... )
"""

from __future__ import annotations

from typing import Any

from lattice.core.config import LatticeConfig
from lattice.core.serialization import message_to_dict
from lattice.client import LatticeClient

# =============================================================================
# OpenAI wrapper
# =============================================================================

def wrap_openai(
    client: Any,
    *,
    config: LatticeConfig | None = None,
    compress: bool = True,
) -> Any:
    """Wrap an OpenAI SDK client with LATTICE compression.

    Args:
        client: An ``openai.AsyncOpenAI`` or ``openai.OpenAI`` instance.
        config: Optional LatticeConfig for compression settings.
        compress: Whether to enable compression. Set to False to disable
            without changing code.

    Returns:
        A wrapper that intercepts ``chat.completions.create()`` calls,
        compresses the messages, and forwards to the original client.

    Example:
        >>> import openai
        >>> from lattice.sdk import wrap_openai
        >>> base = openai.AsyncOpenAI()
        >>> wrapped = wrap_openai(base)
        >>> resp = await wrapped.chat.completions.create(
        ...     model="gpt-4", messages=[...]
        ... )
    """
    if not compress:
        return client

    lattice = LatticeClient(config=config)

    original_create = client.chat.completions.create

    async def _create(*args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
        model = kwargs.get("model", args[0] if args else "")

        # Compress
        compressed = await lattice.compress_request_async(
            messages=messages,
            model=model,
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            top_p=kwargs.get("top_p"),
            tools=kwargs.get("tools"),
            tool_choice=kwargs.get("tool_choice"),
            stop=kwargs.get("stop"),
            stream=kwargs.get("stream", False),
        )

        # Replace messages with compressed versions
        kwargs = dict(kwargs)
        kwargs["messages"] = [message_to_dict(m) for m in compressed.messages]

        return await original_create(*args, **kwargs)

    # Monkey-patch
    client.chat.completions.create = _create
    return client


# =============================================================================
# Anthropic wrapper
# =============================================================================

def wrap_anthropic(
    client: Any,
    *,
    config: LatticeConfig | None = None,
    compress: bool = True,
) -> Any:
    """Wrap an Anthropic SDK client with LATTICE compression.

    Args:
        client: An ``anthropic.AsyncAnthropic`` or ``anthropic.Anthropic`` instance.
        config: Optional LatticeConfig for compression settings.
        compress: Whether to enable compression.

    Returns:
        A wrapper that intercepts ``messages.create()`` calls.

    Example:
        >>> import anthropic
        >>> from lattice.sdk import wrap_anthropic
        >>> base = anthropic.AsyncAnthropic()
        >>> wrapped = wrap_anthropic(base)
        >>> resp = await wrapped.messages.create(
        ...     model="claude-3-opus", max_tokens=1024, messages=[...]
        ... )
    """
    if not compress:
        return client

    lattice = LatticeClient(config=config)
    original_create = client.messages.create

    async def _create(*args: Any, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "")

        # Compress (Anthropic messages are similar to OpenAI format)
        compressed = await lattice.compress_request_async(
            messages=messages,
            model=model,
            max_tokens=kwargs.get("max_tokens"),
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            stop=kwargs.get("stop_sequences"),
            stream=kwargs.get("stream", False),
        )

        kwargs = dict(kwargs)
        kwargs["messages"] = [message_to_dict(m) for m in compressed.messages]

        return await original_create(*args, **kwargs)

    client.messages.create = _create
    return client
