from __future__ import annotations

from lattice.providers.adapters.anthropic.context import _ctx_tool_id_mapping
from lattice.providers.adapters.anthropic.core import AnthropicAdapterCore
from lattice.providers.adapters.anthropic.deserialization import AnthropicDeserializeMixin
from lattice.providers.adapters.anthropic.serialization import AnthropicSerializeMixin
from lattice.providers.adapters.anthropic.streaming import AnthropicStreamMixin


class AnthropicAdapter(
    AnthropicStreamMixin,
    AnthropicDeserializeMixin,
    AnthropicSerializeMixin,
    AnthropicAdapterCore,
):
    """Anthropic Messages API adapter with full Claude Code parity."""

    name = "anthropic"


__all__ = ["AnthropicAdapter", "_ctx_tool_id_mapping"]
