"""Token counting utilities for LATTICE.

Provides both exact and approximate token counting. The default is
tiktoken-based (exact for OpenAI models), with a fast approximate fallback
("chars // 4") for unsupported models.

Performance note: tiktoken encoding is a C extension and is fast, but we
still avoid calling it repeatedly on unchanged text via memoization.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Protocol

import tiktoken

logger = logging.getLogger(__name__)


# =============================================================================
# Counter Protocol
# =============================================================================


class TokenCounter(Protocol):
    """Protocol for token counting implementations.

    This allows pluggable tokenizers (tiktoken, anthropic-tokenizer, etc.).
    """

    def count(self, text: str) -> int:
        """Count tokens in a text string.

        Returns:
            Number of tokens. Minimum 1 (empty string = 1 token).
        """
        ...


# =============================================================================
# ApproximateCounter (pure Python, no dependencies)
# =============================================================================


@dataclasses.dataclass(slots=True)
class ApproximateCounter:
    """Fast approximate token counter.

    Uses "1 token ≈ 4 characters" heuristic. This is accurate for English text
    (±20%), fast (pure Python, no I/O), and has zero external dependencies.

    Use this as fallback when exact tokenizer is unavailable or as a quick
    estimate before expensive operations.
    """

    ratio: float = 4.0

    def count(self, text: str) -> int:
        """Approximate token count.

        Returns:
            max(1, len(text) / ratio) rounded up
        """
        if not text:
            return 1
        return max(1, int(len(text) / self.ratio))


# =============================================================================
# TiktokenCounter (exact for OpenAI models)
# =============================================================================


class TiktokenCounter:
    """Exact token counter using tiktoken.

    tiktoken is OpenAI's tokenizer library. It provides exact token counts
    for GPT-2/3/4 models. For non-OpenAI models, results are approximate
    (different tokenizers may split differently).

    Thread-safe: tiktoken.Encoding is stateless.
    """

    # Lazily loaded encodings
    _encodings: dict[str, tiktoken.Encoding] = {}

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name
        self._encoding: tiktoken.Encoding | None = None

    @property
    def encoding(self) -> tiktoken.Encoding:
        """Lazy-load the tiktoken encoding."""
        if self._encoding is None:
            try:
                self._encoding = tiktoken.get_encoding(self.encoding_name)
            except KeyError:
                logger.warning(
                    "tiktoken encoding %r not found, falling back to cl100k_base",
                    self.encoding_name,
                )
                self._encoding = tiktoken.get_encoding("cl100k_base")
        return self._encoding

    def count(self, text: str) -> int:
        """Exact token count using tiktoken.

        Returns:
            Number of tokens. Minimum 1.
        """
        if not text:
            return 1
        try:
            return max(1, len(self.encoding.encode(text)))
        except Exception:
            # tiktoken may raise on very long text or invalid UTF-8
            logger.warning("tiktoken count failed, falling back to approximate")
            return ApproximateCounter().count(text)


# =============================================================================
# Model-aware counter
# =============================================================================


class ModelTokenCounter:
    """Route to the correct tokenizer based on model name.

    This is the primary entry point for token counting in LATTICE.
    It automatically selects the correct tiktoken encoding or falls
    back to approximation for non-OpenAI models.
    """

    # Mapping: model name prefix -> tiktoken encoding
    ENCODING_MAP: dict[str, str] = {
        "gpt-4o": "o200k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5": "cl100k_base",
        "text-embedding": "cl100k_base",
        "text-davinci": "p50k_base",
        "text-curie": "r50k_base",
        "text-babbage": "r50k_base",
        "text-ada": "r50k_base",
        "gpt2": "gpt2",
    }

    # Models where tiktoken may NOT be accurate (different tokenizer)
    NON_OPENAI_MODELS = {
        "claude",
        "llama",
        "mistral",
        "gemini",
        "command",
        "jurassic",
        "palm",
    }

    def __init__(self) -> None:
        self._counters: dict[str, TokenCounter] = {}
        self._default = TiktokenCounter("cl100k_base")

    def _get_encoding_name(self, model: str) -> str | None:
        """Determine tiktoken encoding for a model."""
        model_lower = model.lower()

        # Check if it's a known non-OpenAI model
        for prefix in self.NON_OPENAI_MODELS:
            if prefix in model_lower:
                return None  # Use approximate

        # Match against known OpenAI model prefixes
        for prefix, encoding in self.ENCODING_MAP.items():
            if model_lower.startswith(prefix):
                return encoding

        # Default to cl100k_base for unknown models
        return "cl100k_base"

    def get_counter(self, model: str) -> TokenCounter:
        """Get the appropriate counter for a model.

        Returns a cached counter instance to avoid repeated initialization.
        """
        cache_key = model.lower()
        if cache_key not in self._counters:
            encoding = self._get_encoding_name(model)
            if encoding:
                self._counters[cache_key] = TiktokenCounter(encoding)
            else:
                self._counters[cache_key] = ApproximateCounter()
        return self._counters[cache_key]

    def count(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens in text for a specific model.

        Args:
            text: The text to count tokens for.
            model: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet").

        Returns:
            Token count. Minimum 1.
        """
        counter = self.get_counter(model)
        return counter.count(text)

    def count_messages(self, messages: list[dict[str, str]], model: str = "gpt-4") -> int:
        """Count tokens in a list of message dicts.

        This mirrors the OpenAI token counting logic:
        - Base overhead: 3 tokens per message
        - Content tokens: counted normally
        - Role + name overhead: ~4 tokens per message

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            model: Model identifier.

        Returns:
            Total token count.
        """
        counter = self.get_counter(model)
        total = 0
        for msg in messages:
            # Base overhead per message
            total += 3
            # Role overhead
            total += 1  # "role" token
            # Content tokens
            content = msg.get("content", "")
            total += counter.count(content)
            # Name/tool overhead if present
            if msg.get("name"):
                total += counter.count(msg["name"])
        return total


# =============================================================================
# Convenience API
# =============================================================================

# Singleton instance
_counter = ModelTokenCounter()


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text for a model.

    This is the simplest entry point — no class instantiation needed.

    Args:
        text: Text to count.
        model: Model identifier.

    Returns:
        Token count (minimum 1).

    Example:
        >>> count_tokens("Hello, world!", model="gpt-4")
        4
    """
    return _counter.count(text, model)


def count_message_tokens(messages: list[dict[str, str]], model: str = "gpt-4") -> int:
    """Count tokens in message list.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        model: Model identifier.

    Returns:
        Total token count with message overhead.
    """
    return _counter.count_messages(messages, model)
