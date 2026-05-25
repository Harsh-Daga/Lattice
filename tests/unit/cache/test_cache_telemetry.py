"""Tests for cache telemetry extraction in gateway/compat.py."""

from __future__ import annotations

from lattice.gateway.compat import _extract_cached_tokens


def test_extract_openai_cached_tokens() -> None:
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "prompt_tokens_details": {"cached_tokens": 30},
    }
    assert _extract_cached_tokens(usage) == 30


def test_extract_normalized_cached_tokens() -> None:
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "cached_tokens": 40,
    }
    assert _extract_cached_tokens(usage) == 40


def test_extract_anthropic_cached_tokens() -> None:
    usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 25,
    }
    assert _extract_cached_tokens(usage) == 25


def test_extract_bedrock_cached_tokens() -> None:
    usage = {
        "inputTokens": 100,
        "outputTokens": 50,
        "cacheReadInputTokens": 64,
    }
    assert _extract_cached_tokens(usage) == 64


def test_extract_bedrock_pascal_case_cached_tokens() -> None:
    usage = {
        "InputTokens": 100,
        "OutputTokens": 50,
        "CacheReadInputTokens": 32,
    }
    assert _extract_cached_tokens(usage) == 32


def test_extract_gemini_cached_content_tokens() -> None:
    usage = {
        "promptTokenCount": 100,
        "candidatesTokenCount": 50,
        "cachedContentTokenCount": 28,
    }
    assert _extract_cached_tokens(usage) == 28


def test_extract_wrapped_gemini_usage_metadata() -> None:
    usage = {
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 50,
            "cachedContentTokenCount": 28,
        }
    }
    assert _extract_cached_tokens(usage) == 28


def test_extract_generic_cached_tokens() -> None:
    usage = {"prompt_tokens": 100, "cached_tokens": 10}
    assert _extract_cached_tokens(usage) == 10


def test_extract_no_cached_tokens() -> None:
    usage = {"prompt_tokens": 100, "completion_tokens": 50}
    assert _extract_cached_tokens(usage) == 0


def test_extract_invalid_usage() -> None:
    assert _extract_cached_tokens(None) == 0
    assert _extract_cached_tokens("bad") == 0
    assert _extract_cached_tokens([]) == 0
