"""Cache fingerprinting and key computation."""

from __future__ import annotations

import dataclasses
import difflib
import enum
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any


class ContentClass(enum.Enum):
    PLAIN_TEXT = "plain_text"
    CODE = "code"
    JSON = "json"
    TOOL_OUTPUT = "tool_output"
    MIXED = "mixed"


@dataclass(slots=True)
class CachedResponse:
    """A cached LLM response, storage-agnostic.

    Attributes:
        content: Full text content of the response.
        tool_calls: Optional list of tool calls.
        usage: Token usage dict (may be empty for cached hits).
        model: Model identifier used for the original response.
        finish_reason: Finish reason string.
        sse_chunks: For streaming responses, the pre-computed SSE chunk strings.
        created_at: Unix timestamp when the entry was cached.
        expires_at: Unix timestamp when the entry becomes stale.
        metadata: Extra metadata (e.g. content class for stats).
    """

    content: str
    tool_calls: list[dict[str, Any]] | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    model: str = ""
    finish_reason: str = "stop"
    sse_chunks: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _SemanticFingerprint:
    token_set: frozenset[str]
    role_pattern: tuple[str, ...]
    tool_schema_hash: str | None
    normalized_text: str
    content_class: ContentClass
    message_count: int
    has_tools: bool
    has_tool_calls: bool
    has_images: bool


@dataclass(slots=True)
class _FingerprintEntry:
    model: str
    fingerprint: _SemanticFingerprint


def _canonical_request(request: Any) -> dict[str, Any]:
    """Extract canonical fields from a request for stable hashing.

    Only fields that affect the LLM response are included.  Streaming
    flag is intentionally excluded — the same logical request can be
    served as either streaming or non-streaming from cache.
    """
    # Prefer the central serializer for internal slotted dataclasses.  Request
    # and Message intentionally use ``slots=True``, so ``__dict__``-based
    # duck-typing does not work on the proxy hot path.
    if _is_internal_request(request):
        from lattice.transport.serialization import request_to_dict

        raw = request_to_dict(request)
    elif hasattr(request, "to_dict"):
        raw = request.to_dict()
    elif dataclasses.is_dataclass(request):
        raw = (
            dataclasses.asdict(request)
            if dataclasses.is_dataclass(request) and not isinstance(request, type)
            else {}
        )
    elif isinstance(request, dict):
        raw = request
    elif hasattr(request, "__dict__"):
        raw = vars(request)
    else:
        raw = dict(request)

    # Normalize messages
    messages: list[dict[str, Any]] = []
    raw_msgs = raw.get("messages", [])
    for m in raw_msgs:
        if hasattr(m, "to_dict"):
            m = m.to_dict()
        msg: dict[str, Any] = {}
        for k in ("role", "content", "name", "tool_calls", "tool_call_id"):
            v = m.get(k) if isinstance(m, dict) else getattr(m, k, None)
            if v is not None:
                msg[k] = v
        messages.append(msg)

    key_obj: dict[str, Any] = {
        "model": raw.get("model", ""),
        "messages": messages,
    }

    for k in (
        "temperature",
        "max_tokens",
        "top_p",
        "tools",
        "tool_choice",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "response_format",
    ):
        v = raw.get(k)
        if v is not None:
            key_obj[k] = v

    return key_obj


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _tokenize_for_semantics(text: str) -> frozenset[str]:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return frozenset(t for t in tokens if len(t) >= 2)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def _normalized_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _is_internal_request(request: Any) -> bool:
    """Return True for ``lattice.core.transport.Request`` without hard import."""
    return (
        request.__class__.__name__ == "Request"
        and request.__class__.__module__ == "lattice.core.transport"
    )


def compute_cache_key(request: Any) -> str:
    """Compute a stable SHA-256 hex digest for *request*."""
    canonical = _canonical_request(request)
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Content class detection
# ---------------------------------------------------------------------------


def _detect_content_class(request: Any) -> ContentClass:
    """Classify request content based on message content."""
    canonical = _canonical_request(request)
    messages = canonical.get("messages", [])

    classes_detected: set[ContentClass] = set()

    # TOOL_OUTPUT: any message has tool_calls or tool_call_id
    for m in messages:
        if m.get("tool_calls") or m.get("tool_call_id"):
            classes_detected.add(ContentClass.TOOL_OUTPUT)
            break

    # CODE: >30% of text content is inside ``` fences
    total_text_len = 0
    code_fence_len = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total_text_len += len(content)
            for fence in re.findall(r"```[\s\S]*?```", content):
                code_fence_len += len(fence)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    total_text_len += len(text)
                    for fence in re.findall(r"```[\s\S]*?```", text):
                        code_fence_len += len(fence)
    if total_text_len > 0 and code_fence_len / total_text_len > 0.30:
        classes_detected.add(ContentClass.CODE)

    # JSON: any message content starts with { or [ and is valid JSON
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            stripped = content.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    json.loads(stripped)
                    classes_detected.add(ContentClass.JSON)
                    break
                except json.JSONDecodeError:
                    pass

    if len(classes_detected) > 1:
        return ContentClass.MIXED
    if len(classes_detected) == 1:
        return next(iter(classes_detected))
    return ContentClass.PLAIN_TEXT


# ---------------------------------------------------------------------------
# Semantic fingerprint
# ---------------------------------------------------------------------------


def _compute_semantic_fingerprint(request: Any) -> _SemanticFingerprint:
    canonical = _canonical_request(request)
    messages = canonical.get("messages", [])

    roles = tuple(str(m.get("role", "")) for m in messages)

    tools = canonical.get("tools")
    tool_schema_hash = None
    if tools:
        tool_json = json.dumps(tools, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        tool_schema_hash = hashlib.sha256(tool_json.encode("utf-8")).hexdigest()

    parts: list[str] = []
    for m in messages:
        parts.append(str(m.get("role", "")))
        parts.append(_normalize_text(m.get("content")))
        parts.append(_normalize_text(m.get("name")))
        parts.append(_normalize_text(m.get("tool_call_id")))
        if m.get("tool_calls"):
            parts.append(_normalize_text(m.get("tool_calls")))

    for key in ("tools", "tool_choice", "response_format", "stop"):
        if canonical.get(key) is not None:
            parts.append(_normalize_text(canonical.get(key)))

    raw_text = " ".join(parts).lower()
    normalized_text = re.sub(r"[^a-z0-9\s]", "", raw_text)
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    token_set = _tokenize_for_semantics(normalized_text)

    content_class = _detect_content_class(request)

    has_tool_calls = any(m.get("tool_calls") or m.get("tool_call_id") for m in messages)
    has_images = False
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    has_images = True
                    break
        if has_images:
            break

    return _SemanticFingerprint(
        token_set=token_set,
        role_pattern=roles,
        tool_schema_hash=tool_schema_hash,
        normalized_text=normalized_text,
        content_class=content_class,
        message_count=len(messages),
        has_tools=tools is not None,
        has_tool_calls=has_tool_calls,
        has_images=has_images,
    )


def _compute_similarity(query: _SemanticFingerprint, candidate: _SemanticFingerprint) -> float:
    """Weighted similarity score between two fingerprints."""
    jaccard = _jaccard(query.token_set, candidate.token_set)
    role_score = 1.0 if query.role_pattern == candidate.role_pattern else 0.0

    if query.has_tools or candidate.has_tools:
        tool_score = 1.0 if query.tool_schema_hash == candidate.tool_schema_hash else 0.0
    else:
        tool_score = 1.0

    text_score = _normalized_similarity(query.normalized_text, candidate.normalized_text)

    return jaccard * 0.40 + role_score * 0.20 + tool_score * 0.20 + text_score * 0.20


# ---------------------------------------------------------------------------
# Exact response cache
# ---------------------------------------------------------------------------
