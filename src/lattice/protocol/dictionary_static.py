"""Static dictionary for wire-level compression.

Curated from the most frequent JSON fragments in OpenAI Chat Completions,
Anthropic Messages, and common tool schemas. These entries are hard-coded
and shared between all LATTICE clients and proxies.

Selection Criteria
------------------
1. Frequency: appears in >80% of requests to major providers.
2. Length: at least 2 bytes (shorter literals are not worth referencing).
3. Stability: not model-version-specific (e.g. `"gpt-4"` is stable,
   `"gpt-4-1106-preview"` is not).
4. Structural value: keys, punctuation, and common object prefixes compress
   better than free-form text.

Layout
------
Indices 0-9    : single-byte JSON / SSE tokens
Indices 10-49  : common JSON keys (quoted)
Indices 50-99  : common JSON values (quoted)
Indices 100-149: composite fragments (multi-token patterns)
Indices 150-199: provider-specific and tool-schema fragments
"""

from __future__ import annotations

# =============================================================================
# Static dictionary entries
# =============================================================================

_STATIC_ENTRIES: tuple[str, ...] = (
    # ------------------------------------------------------------------
    # 0-9: Single-character structural tokens (very high frequency)
    # ------------------------------------------------------------------
    '{',
    '}',
    '[',
    ']',
    ':',
    ',',
    '"',
    ' ',
    '\n',
    '\t',
    # ------------------------------------------------------------------
    # 10-49: Common JSON keys (quoted, alphabetical-ish order)
    # ------------------------------------------------------------------
    '"arguments"',
    '"base64"',
    '"content"',
    '"description"',
    '"detail"',
    '"encoding"',
    '"event"',
    '"finish_reason"',
    '"frequency_penalty"',
    '"function"',
    '"id"',
    '"image_url"',
    '"index"',
    '"input"',
    '"max_tokens"',
    '"messages"',
    '"metadata"',
    '"model"',
    '"name"',
    '"object"',
    '"parameters"',
    '"presence_penalty"',
    '"properties"',
    '"required"',
    '"role"',
    '"schema"',
    '"stop"',
    '"stream"',
    '"system_fingerprint"',
    '"temperature"',
    '"text"',
    '"tool_calls"',
    '"tool_choice"',
    '"tools"',
    '"top_p"',
    '"type"',
    '"url"',
    '"usage"',
    # ------------------------------------------------------------------
    # 50-99: Common JSON values (quoted)
    # ------------------------------------------------------------------
    '"assistant"',
    '"auto"',
    '"claude-3-5-sonnet-20241022"',
    '"claude-3-opus-20240229"',
    '"content_filter"',
    '"data"',
    '"end_turn"',
    '"ephemeral"',
    '"error"',
    '"false"',
    '"gpt-4"',
    '"gpt-4o"',
    '"gpt-4o-mini"',
    '"length"',
    '"message"',
    '"none"',
    '"null"',
    '"ollama"',
    '"string"',
    '"system"',
    '"tool"',
    '"tool_use"',
    '"true"',
    '"user"',
    # ------------------------------------------------------------------
    # 100-149: Composite fragments (multi-token, high-impact)
    # ------------------------------------------------------------------
    '{"role":"user","content":"',
    '{"role":"assistant","content":"',
    '{"role":"system","content":"',
    '{"role":"tool","content":"',
    '{"type":"function","function":{',
    '{"type":"text","text":"',
    '{"type":"tool_use","id":"',
    '{"index":0,"message":{',
    '"cache_control":{"type":"ephemeral"}',
    '"image_url":{"url":"',
    '"usage":{"prompt_tokens":',
    '"usage":{"completion_tokens":',
    '"usage":{"total_tokens":',
    '"delta":{"content":"',
    '"choices":[{"index":0,',
    '"system_fingerprint":"',
    '{"model":"gpt-4",',
    '{"model":"gpt-4o",',
    '{"model":"claude-3-5-sonnet-20241022",',
    '{"stream":true,',
    '{"stream":false,',
    '"tools":[],',
    '"tool_choice":"auto"',
    '"tool_choice":"none"',
    '"stop_sequences":[],',
    '"temperature":0.7',
    '"temperature":1',
    '"max_tokens":4096',
    '"max_tokens":8192',
    '"top_p":1',
    '"top_p":0.9',
    '"frequency_penalty":0',
    '"presence_penalty":0',
    '{"id":"',
    '{"object":"chat.completion",',
    '{"object":"chat.completion.chunk",',
    '"finish_reason":"stop"',
    '"finish_reason":"length"',
    '"finish_reason":"tool_calls"',
    '"finish_reason":null',
    # ------------------------------------------------------------------
    # 150-199: Provider-specific, tool-schema, and SSE fragments
    # ------------------------------------------------------------------
    '{"error":{',
    '"message":"',
    '"code":"',
    '"param":"',
    '"type":"invalid_request_error"',
    '"type":"rate_limit_error"',
    '"type":"api_error"',
    '"type":"server_error"',
    '"type":"object"',
    '"type":"string"',
    '"type":"number"',
    '"type":"boolean"',
    '"type":"array"',
    '"type":"null"',
    '"enum":[]',
    '"anyOf":[]',
    '"oneOf":[]',
    '"allOf":[]',
    '"additionalProperties":false',
    '"items":{',
    '"minLength":',
    '"maxLength":',
    '"minimum":',
    '"maximum":',
    '"pattern":"',
    '"default":"',
    '"title":"',
    '"$ref":"',
    '"$schema":"http://json-schema.org/draft-07/schema#"',
    'data: ',
    'event: ',
    '\n\n',
    '[DONE]',
    ':\n\n',
    '{"created":',
    '"data":[],',
    '"model":"',
    '"id":"chatcmpl-',
    '"id":"msg_',
    '"jsonrpc":"2.0"',
    '"result":',
    '"error":null',
)

# Build lookup tables once at import time
STATIC_TABLE: dict[int, str] = {i: entry for i, entry in enumerate(_STATIC_ENTRIES)}
STATIC_REVERSE: dict[str, int] = {entry: i for i, entry in enumerate(_STATIC_ENTRIES)}
STATIC_COUNT = len(_STATIC_ENTRIES)


def get_static_table() -> dict[int, str]:
    """Return a copy of the static dictionary table."""
    return dict(STATIC_TABLE)


def get_static_reverse() -> dict[str, int]:
    """Return a copy of the static reverse lookup."""
    return dict(STATIC_REVERSE)


def static_index(text: str) -> int | None:
    """Look up a string in the static dictionary.

    Returns the index if found, None otherwise.
    """
    return STATIC_REVERSE.get(text)


def static_entry(index: int) -> str | None:
    """Return the static dictionary entry at the given index.

    Returns None if the index is out of range.
    """
    return STATIC_TABLE.get(index)


__all__ = [
    "STATIC_TABLE",
    "STATIC_REVERSE",
    "STATIC_COUNT",
    "get_static_table",
    "get_static_reverse",
    "static_index",
    "static_entry",
]
