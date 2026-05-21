# Phase 5 — Providers & Transport Consolidation

> **Goal.** Split the 1539-LoC `providers/transport.py` monolith into a 7-file `providers/transport/` package; merge ~300 LoC of duplicated streaming retry logic into a single method; reorganise adapter files under `providers/adapters/`; consolidate the duplicated `AnthropicToolSanitizer` / `BedrockToolSanitizer` into a shared base class; fix the unbounded growth of `RateLimitTracker`. Move `providers/stall_detector.py` into the new `providers/transport/` package. All 17 provider adapters keep working byte-for-byte from the client's perspective.
>
> **Outcome.** `from lattice.providers import DirectHTTPProvider, ProviderRegistry, ConnectionPoolManager` works unchanged. `from lattice.providers.adapters import OpenAIAdapter, AnthropicAdapter, ...` works. No file in `providers/` exceeds 850 LoC. The streaming code path exists in exactly one place.
>
> **Estimated effort.** 3 days.

---

## 1. Why this phase exists

The audit surfaced six concrete issues in the provider layer:

1. **`providers/transport.py` is 1539 LoC** mixing seven concerns:
   - `ProviderRegistry` (model-string → adapter routing)
   - `ConnectionPoolManager` (httpx pool with HTTP/2 fallback)
   - `RateLimitTracker` (parse `x-ratelimit-*` headers)
   - `DirectHTTPProvider.completion()` (non-streaming with retry + TACC + rate-limit)
   - `DirectHTTPProvider.completion_stream()` (streaming without stall detection)
   - `DirectHTTPProvider.completion_stream_with_stall_detect()` (streaming with stall detection)
   - SSE parsing + buffering helpers
2. **The two streaming methods duplicate ~300 LoC.** They differ only in: (a) whether stall detection runs, (b) whether a state machine processes the SSE line (Anthropic) vs. direct adapter normalisation (others). Both reasons can become parameters.
3. **`AnthropicToolSanitizer` and `BedrockToolSanitizer` are ~95% identical** (validate `^[a-zA-Z0-9_-]+$`, maintain bidirectional ID mapping, sanitize/restore). One base class plus two subclasses.
4. **`RateLimitTracker` grows unbounded** — every distinct provider name adds an entry that's never evicted. Add a TTL-based cleanup.
5. **`providers/stall_detector.py` is conceptually part of HTTP transport.** Move into `providers/transport/`.
6. **`providers/__init__.py` exports only 10 of 17 adapters.** The OpenAI-compatible 9 are reachable only via `ProviderRegistry`. Either export all 17 or document the deliberate omission.

---

## 2. Files touched

### 2.1 Restructured

`providers/transport.py` is replaced by a package:

```
providers/transport/
├── __init__.py         # re-exports DirectHTTPProvider, ProviderRegistry, ConnectionPoolManager, RateLimitTracker, _resolve_provider_name
├── registry.py         # ProviderRegistry; ~150 LoC
├── pool.py             # ConnectionPoolManager; ~120 LoC
├── rate_limits.py      # RateLimitTracker with TTL eviction; ~80 LoC
├── completion.py       # DirectHTTPProvider.completion() (non-streaming); ~280 LoC
├── streaming.py        # DirectHTTPProvider._stream() (merged single streaming impl); ~600 LoC
├── stall_detector.py   # was providers/stall_detector.py; ~280 LoC
└── helpers.py          # _resolve_base_url, _resolve_api_key, _parse_sse_line, _optimize_stream_chunk, _next_stream_line, _stream_chunk_text; ~180 LoC
```

The 17 adapter files move under `providers/adapters/`:

```
providers/adapters/
├── __init__.py         # re-exports all 17 adapter classes + ProviderAdapter Protocol
├── base.py             # ProviderAdapter Protocol + _pop_system, _remap_tool_choice, _remap_tools, _strip_provider_prefix, _format_sse_event
├── openai.py
├── openai_compatible.py     # 9 OpenAI-compatible adapters
├── anthropic.py
├── azure.py
├── bedrock.py
├── gemini.py                # Gemini + Vertex
└── ollama.py                # Ollama + OllamaCloud
```

The cross-cutting utility files stay at `providers/` top level:

```
providers/
├── __init__.py              # public re-exports (transport + adapters)
├── capabilities.py          # Capability matrix
├── stream_state.py          # Anthropic streaming state machine
├── tool_sanitizer.py        # base ToolSanitizer + Anthropic + Bedrock subclasses (consolidated)
├── schema_filter.py         # JSON schema cleanup
├── mcp_to_anthropic.py      # MCP tool format converter
├── credentials.py           # moved here in Phase 3
├── transport/    (above)
└── adapters/     (above)
```

### 2.2 Moved

| Current path | New path |
|---|---|
| `src/lattice/providers/transport.py` | (split — see §2.1) |
| `src/lattice/providers/stall_detector.py` | `src/lattice/providers/transport/stall_detector.py` |
| `src/lattice/providers/base.py` | `src/lattice/providers/adapters/base.py` |
| `src/lattice/providers/openai.py` | `src/lattice/providers/adapters/openai.py` |
| `src/lattice/providers/openai_compatible.py` | `src/lattice/providers/adapters/openai_compatible.py` |
| `src/lattice/providers/anthropic.py` | `src/lattice/providers/adapters/anthropic.py` |
| `src/lattice/providers/azure.py` | `src/lattice/providers/adapters/azure.py` |
| `src/lattice/providers/bedrock.py` | `src/lattice/providers/adapters/bedrock.py` |
| `src/lattice/providers/gemini.py` | `src/lattice/providers/adapters/gemini.py` |
| `src/lattice/providers/ollama.py` | `src/lattice/providers/adapters/ollama.py` |

### 2.3 Created

```
providers/transport/__init__.py
providers/transport/registry.py
providers/transport/pool.py
providers/transport/rate_limits.py
providers/transport/completion.py
providers/transport/streaming.py
providers/transport/helpers.py
providers/adapters/__init__.py
```

### 2.4 Deleted

```
providers/transport.py              # replaced by the package
```

### 2.5 Modified

- `providers/__init__.py` — re-exports updated to point at new locations.
- `providers/tool_sanitizer.py` — consolidated with a base class.
- `providers/capabilities.py`, `providers/stream_state.py`, `providers/schema_filter.py`, `providers/mcp_to_anthropic.py` — only the imports change (since `base.py` moved).
- Every adapter file imports from `..base` after the move into `adapters/`.
- All `from lattice.providers.transport import` consumers — unchanged (re-exported by package `__init__.py`).
- All `from lattice.providers.stall_detector import` consumers — rewritten.
- All `from lattice.providers.{openai,anthropic,...} import` consumers — rewritten.

---

## 3. The streaming merge — concrete plan

The two methods today have this shape (simplified):

```python
# completion_stream(...) — about 600 LoC
async def completion_stream(self, request, provider_name, ...):
    adapter = self.registry.resolve(...)
    await self._await_tacc_admission(provider_name, request)
    async with self.pool.client(provider_name, base_url) as client:
        async with client.stream("POST", endpoint, ...) as response:
            buffer = b""
            async for chunk in response.aiter_bytes():
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.startswith(b"data:"): continue
                    parsed = self._parse_sse_line(line, adapter)
                    optimized = self._optimize_stream_chunk(parsed, optimizer, state=None)
                    yield optimized
            # finally: rate-limit + TACC.after_response
```

```python
# completion_stream_with_stall_detect(...) — about 700 LoC
async def completion_stream_with_stall_detect(self, request, provider_name, stream_state=None, ...):
    adapter = self.registry.resolve(...)
    await self._await_tacc_admission(provider_name, request)
    self.stall_detector.start_stream(provider_name, stream_id)
    async with self.pool.client(provider_name, base_url) as client:
        async with client.stream("POST", endpoint, ...) as response:
            buffer = b""
            async for chunk in response.aiter_bytes():
                buffer += chunk
                self.stall_detector.record_chunk(provider_name, "stream_chunk", elapsed, tokens, stream_id)
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.startswith(b"data:"): continue
                    parsed = self._process_sse_line_with_state(line, stream_state) if stream_state \
                             else self._parse_sse_line(line, adapter)
                    optimized = self._optimize_stream_chunk(parsed, optimizer, state=stream_state)
                    yield optimized
            # finally: stall_detector.end_stream + rate-limit + TACC.after_response
```

**Merge target:** single `_stream(self, request, *, stall_detect: bool = False, stream_state: Optional[AnthropicStreamState] = None, ...)` that branches at the three points where the two diverge:

```python
async def _stream(
    self,
    request: Request,
    *,
    provider_name: str,
    stall_detect: bool = False,
    stream_state: Optional["AnthropicStreamState"] = None,
    optimizer: Optional[Callable] = None,
    timeout: float = 60.0,
) -> AsyncIterator[dict]:
    adapter = self.registry.resolve(request.model, provider_name)
    stream_id = uuid4().hex

    await self._await_tacc_admission(provider_name, request)
    if stall_detect:
        self.stall_detector.start_stream(provider_name, stream_id)

    try:
        async with self.pool.client(provider_name, self._resolve_base_url(provider_name)) as client:
            async with client.stream("POST", adapter.chat_endpoint(...), ...) as response:
                buffer = b""
                first_chunk_at = None
                async for chunk in response.aiter_bytes():
                    now = time.monotonic()
                    if first_chunk_at is None:
                        first_chunk_at = now
                        self.tacc.record_ttft(provider_name, (now - request._submitted_at) * 1000)

                    if stall_detect:
                        elapsed_ms = (now - first_chunk_at) * 1000
                        approx_tokens = max(1, len(chunk) // 4)
                        self.stall_detector.record_chunk(provider_name, "stream_chunk", elapsed_ms, approx_tokens, stream_id)

                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if not line.startswith(b"data:"):
                            continue
                        if stream_state is not None:
                            parsed = self._process_sse_line_with_state(line, stream_state)
                        else:
                            parsed = self._parse_sse_line(line, adapter)
                        if parsed is None:
                            continue
                        if optimizer:
                            parsed = self._optimize_stream_chunk(parsed, optimizer, state=stream_state)
                        yield parsed
    finally:
        if stall_detect:
            self.stall_detector.end_stream(stream_id)
        self.tacc.after_response(provider_name, elapsed_ms_total, tokens_total, status_code=200, retry_after=None)

# Public methods become thin wrappers:
async def completion_stream(self, request, **kw):
    async for chunk in self._stream(request, stall_detect=False, **kw):
        yield chunk

async def completion_stream_with_stall_detect(self, request, *, stream_state=None, **kw):
    async for chunk in self._stream(request, stall_detect=True, stream_state=stream_state, **kw):
        yield chunk
```

Or — preferred — replace the two public methods with a single `completion_stream(request, *, stall_detect=True, stream_state=None, ...)`. The default `stall_detect=True` matches the more sophisticated method's behaviour. Audit all callers (mostly `proxy/routes.py` and `gateway/compat.py`) and update.

`streaming.py` ends up at ~600 LoC — replacing ~1300 LoC across the two old methods.

---

## 4. The tool sanitizer consolidation

Today (per audit) `providers/tool_sanitizer.py` has `AnthropicToolSanitizer` and `BedrockToolSanitizer` ~95% identical. Extract base:

```python
# providers/tool_sanitizer.py — new shape, ~250 LoC total

import contextvars
import re
from typing import ClassVar, Optional

class ToolSanitizer:
    """Base class: validate tool IDs against a regex; maintain bidirectional ID map per task.

    Subclasses set TOOL_ID_PATTERN and override behaviour where needed.
    """
    TOOL_ID_PATTERN: ClassVar[re.Pattern] = re.compile(r"^[a-zA-Z0-9_-]+$")
    _ctx: ClassVar[contextvars.ContextVar[dict]] = contextvars.ContextVar(
        "tool_id_map", default={},
    )

    def validate_tool_id(self, tool_id: str) -> bool:
        return bool(self.TOOL_ID_PATTERN.match(tool_id or ""))

    def sanitize(self, tool_id: str) -> str:
        if self.validate_tool_id(tool_id):
            return tool_id
        new_id = self._make_safe(tool_id)
        mapping = self._ctx.get().copy()
        mapping[tool_id] = new_id
        mapping[new_id] = tool_id   # reverse for restore
        self._ctx.set(mapping)
        return new_id

    def restore(self, tool_id: str) -> str:
        return self._ctx.get().get(tool_id, tool_id)

    def cleanup(self) -> None:
        self._ctx.set({})

    def _make_safe(self, tool_id: str) -> str:
        # default: replace invalid chars with underscore, prefix with `tool_` if empty
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_id or "")
        return sanitized or "tool_unknown"


class AnthropicToolSanitizer(ToolSanitizer):
    # Anthropic-specific: nothing different in the base operations,
    # but Anthropic also re-maps tool_use IDs to tool_result IDs bidirectionally.
    pass


class BedrockToolSanitizer(ToolSanitizer):
    # Bedrock-specific: same base behaviour for v1.0.
    pass


def sanitize_tool_ids(tools: list[dict], sanitizer: ToolSanitizer) -> list[dict]:
    out = []
    for tool in tools:
        if "id" in tool:
            tool = {**tool, "id": sanitizer.sanitize(tool["id"])}
        out.append(tool)
    return out


def restore_tool_call_ids(tool_calls: list[dict], sanitizer: ToolSanitizer) -> list[dict]:
    out = []
    for call in tool_calls:
        if "id" in call:
            call = {**call, "id": sanitizer.restore(call["id"])}
        out.append(call)
    return out

__all__ = [
    "ToolSanitizer", "AnthropicToolSanitizer", "BedrockToolSanitizer",
    "sanitize_tool_ids", "restore_tool_call_ids",
]
```

If the audit later finds genuine per-provider differences, add them as method overrides; the base stays clean.

---

## 5. The rate-limit eviction

`RateLimitTracker` today maintains a `dict[provider_name, RateLimitState]` that's appended to indefinitely. Add a `last_seen_at` timestamp per entry and evict entries older than `RATE_LIMIT_TTL_S = 3600.0`:

```python
# providers/transport/rate_limits.py — new shape

import time
from dataclasses import dataclass, field

RATE_LIMIT_TTL_S = 3600.0  # 1 hour

@dataclass
class RateLimitState:
    requests_limit: int = 0
    requests_remaining: int = 0
    requests_reset_at: float = 0.0
    tokens_limit: int = 0
    tokens_remaining: int = 0
    tokens_reset_at: float = 0.0
    last_seen_at: float = field(default_factory=time.time)

class RateLimitTracker:
    def __init__(self) -> None:
        self._state: dict[str, RateLimitState] = {}
        self._last_cleanup_at: float = time.time()

    def record(self, provider_name: str, headers: dict[str, str]) -> None:
        # parse x-ratelimit-* headers, update state, set last_seen_at
        ...
        self._maybe_cleanup()

    def get(self, provider_name: str) -> RateLimitState | None:
        return self._state.get(provider_name)

    def _maybe_cleanup(self) -> None:
        now = time.time()
        if now - self._last_cleanup_at < 300.0:   # cleanup at most every 5 min
            return
        cutoff = now - RATE_LIMIT_TTL_S
        self._state = {k: v for k, v in self._state.items() if v.last_seen_at >= cutoff}
        self._last_cleanup_at = now
```

`MaintenanceCoordinator` (Phase 8 destination: `telemetry/maintenance.py`) can also register a periodic `_maybe_cleanup` callback for proactive eviction.

---

## 6. Step-by-step

### 6.1 Create new directory skeletons

```bash
mkdir -p src/lattice/providers/transport
mkdir -p src/lattice/providers/adapters
```

### 6.2 Move stall_detector + adapter files

```bash
# stall_detector → transport/
git mv src/lattice/providers/stall_detector.py src/lattice/providers/transport/stall_detector.py

# adapters → adapters/
git mv src/lattice/providers/base.py               src/lattice/providers/adapters/base.py
git mv src/lattice/providers/openai.py             src/lattice/providers/adapters/openai.py
git mv src/lattice/providers/openai_compatible.py  src/lattice/providers/adapters/openai_compatible.py
git mv src/lattice/providers/anthropic.py          src/lattice/providers/adapters/anthropic.py
git mv src/lattice/providers/azure.py              src/lattice/providers/adapters/azure.py
git mv src/lattice/providers/bedrock.py            src/lattice/providers/adapters/bedrock.py
git mv src/lattice/providers/gemini.py             src/lattice/providers/adapters/gemini.py
git mv src/lattice/providers/ollama.py             src/lattice/providers/adapters/ollama.py
```

### 6.3 Split `providers/transport.py`

```bash
# Move aside, then create the 7 new files
mv src/lattice/providers/transport.py /tmp/providers_transport.py.bak
```

Create each file under `providers/transport/`, copying the relevant chunks from the backup:

| Source LoC range (in backup) | Target file |
|---|---|
| `ProviderRegistry` class + `_PROVIDER_ALIASES` + `_resolve_provider_name` | `registry.py` |
| `ConnectionPoolManager` class | `pool.py` |
| `RateLimitTracker` class | `rate_limits.py` (rewrite with TTL — §5) |
| `DirectHTTPProvider.__init__`, `health_check`, `get_transport_metadata`, `configure_resilience`, `cleanup_stale_streams`, `completion()` (non-streaming with `_build_request`, retry loop) | `completion.py` |
| `DirectHTTPProvider._stream`, `completion_stream`, `completion_stream_with_stall_detect` (merged per §3) | `streaming.py` |
| `_resolve_base_url`, `_resolve_api_key`, `_build_request`, `_parse_sse_line`, `_optimize_stream_chunk`, `_process_sse_line_with_state`, `_stream_chunk_text`, `_next_stream_line`, `_should_retry`, `_stream_retry_policy`, `_await_tacc_admission`, `_tacc_reservation` | `helpers.py` |

The `DirectHTTPProvider` class itself is now split across `completion.py` and `streaming.py`. Two implementation choices:

**Option A — DirectHTTPProvider in `completion.py` with streaming mixed in via composition:**

```python
# completion.py
from lattice.providers.transport.streaming import StreamingMixin

class DirectHTTPProvider(StreamingMixin):
    def __init__(self, registry, pool, ...):
        ...
    async def completion(self, request, ...):
        ...
```

**Option B — Split into protocols and a facade:**

```python
# completion.py
class CompletionExecutor: ...

# streaming.py
class StreamingExecutor: ...

# __init__.py
class DirectHTTPProvider(CompletionExecutor, StreamingExecutor):
    pass
```

**Chosen:** Option A (mixin). It preserves the existing `DirectHTTPProvider` class identity and minimises consumer changes.

### 6.4 Write `providers/transport/__init__.py`

```python
"""HTTP transport layer for provider dispatch.

Public API:
    DirectHTTPProvider         — the orchestrator; main entry point
    ProviderRegistry           — model-string → adapter routing
    ConnectionPoolManager      — httpx pool with HTTP/2 fallback
    RateLimitTracker           — provider rate-limit state with TTL eviction
    StreamStallDetector        — per-stream stall detection

Helpers (re-exported for advanced users):
    _resolve_provider_name     — public for backward compatibility
"""

from lattice.providers.transport.completion import DirectHTTPProvider
from lattice.providers.transport.registry import (
    ProviderRegistry, _resolve_provider_name, _PROVIDER_ALIASES,
)
from lattice.providers.transport.pool import ConnectionPoolManager
from lattice.providers.transport.rate_limits import RateLimitTracker, RateLimitState
from lattice.providers.transport.stall_detector import StreamStallDetector

__all__ = [
    "DirectHTTPProvider", "ProviderRegistry", "ConnectionPoolManager",
    "RateLimitTracker", "RateLimitState", "StreamStallDetector",
    "_resolve_provider_name", "_PROVIDER_ALIASES",
]
```

### 6.5 Write `providers/adapters/__init__.py`

```python
"""Provider adapters — one per provider family.

17 providers across 8 files (some adapters live in the same module — e.g. Gemini+Vertex).
"""

from lattice.providers.adapters.base import (
    ProviderAdapter,
    _pop_system, _remap_tool_choice, _remap_tools,
    _strip_provider_prefix, _format_sse_event,
)
from lattice.providers.adapters.openai import OpenAIAdapter
from lattice.providers.adapters.openai_compatible import (
    OpenAICompatibleAdapter,
    GroqAdapter, TogetherAdapter, DeepSeekAdapter, PerplexityAdapter,
    MistralAdapter, FireworksAdapter, OpenRouterAdapter,
    CohereAdapter, AI21Adapter,
)
from lattice.providers.adapters.anthropic import AnthropicAdapter
from lattice.providers.adapters.azure import AzureAdapter
from lattice.providers.adapters.bedrock import BedrockAdapter
from lattice.providers.adapters.gemini import GeminiAdapter, VertexAdapter
from lattice.providers.adapters.ollama import OllamaAdapter, OllamaCloudAdapter

__all__ = [
    "ProviderAdapter",
    "OpenAIAdapter", "OpenAICompatibleAdapter",
    "GroqAdapter", "TogetherAdapter", "DeepSeekAdapter", "PerplexityAdapter",
    "MistralAdapter", "FireworksAdapter", "OpenRouterAdapter",
    "CohereAdapter", "AI21Adapter",
    "AnthropicAdapter", "AzureAdapter", "BedrockAdapter",
    "GeminiAdapter", "VertexAdapter",
    "OllamaAdapter", "OllamaCloudAdapter",
    # helper functions (public for adapter authors):
    "_pop_system", "_remap_tool_choice", "_remap_tools",
    "_strip_provider_prefix", "_format_sse_event",
]
```

### 6.6 Update `providers/__init__.py`

```python
"""LATTICE provider layer.

Public API for users:
    from lattice.providers import DirectHTTPProvider, ProviderRegistry
    from lattice.providers import (
        OpenAIAdapter, AnthropicAdapter, ..., AI21Adapter,
    )
    from lattice.providers import (
        Capability, CacheMode, CacheSemantics, ProviderCapability,
    )
"""

# Transport layer (HTTP dispatch)
from lattice.providers.transport import (
    DirectHTTPProvider, ProviderRegistry, ConnectionPoolManager,
    RateLimitTracker, StreamStallDetector,
    _resolve_provider_name,
)

# All 17 adapters re-exported
from lattice.providers.adapters import (
    ProviderAdapter,
    OpenAIAdapter, OpenAICompatibleAdapter,
    GroqAdapter, TogetherAdapter, DeepSeekAdapter, PerplexityAdapter,
    MistralAdapter, FireworksAdapter, OpenRouterAdapter,
    CohereAdapter, AI21Adapter,
    AnthropicAdapter, AzureAdapter, BedrockAdapter,
    GeminiAdapter, VertexAdapter,
    OllamaAdapter, OllamaCloudAdapter,
)

# Per-provider utilities
from lattice.providers.capabilities import (
    Capability, CacheMode, CacheSemantics, RateLimitSemantics, ProviderCapability,
)
from lattice.providers.stream_state import AnthropicStreamState, StreamStateResult
from lattice.providers.tool_sanitizer import (
    ToolSanitizer, AnthropicToolSanitizer, BedrockToolSanitizer,
    sanitize_tool_ids, restore_tool_call_ids,
)
from lattice.providers.schema_filter import sanitize_json_schema, sanitize_tool_definitions
from lattice.providers.mcp_to_anthropic import convert_mcp_to_anthropic, is_mcp_tool
from lattice.providers.credentials import (
    ProviderCredentials, CredentialResolver, get_credential_resolver,
)

__all__ = [
    # transport
    "DirectHTTPProvider", "ProviderRegistry", "ConnectionPoolManager",
    "RateLimitTracker", "StreamStallDetector", "_resolve_provider_name",
    # adapters (all 17)
    "ProviderAdapter",
    "OpenAIAdapter", "OpenAICompatibleAdapter",
    "GroqAdapter", "TogetherAdapter", "DeepSeekAdapter", "PerplexityAdapter",
    "MistralAdapter", "FireworksAdapter", "OpenRouterAdapter",
    "CohereAdapter", "AI21Adapter",
    "AnthropicAdapter", "AzureAdapter", "BedrockAdapter",
    "GeminiAdapter", "VertexAdapter",
    "OllamaAdapter", "OllamaCloudAdapter",
    # capabilities
    "Capability", "CacheMode", "CacheSemantics", "RateLimitSemantics", "ProviderCapability",
    # stream state
    "AnthropicStreamState", "StreamStateResult",
    # tool sanitization
    "ToolSanitizer", "AnthropicToolSanitizer", "BedrockToolSanitizer",
    "sanitize_tool_ids", "restore_tool_call_ids",
    # schema
    "sanitize_json_schema", "sanitize_tool_definitions",
    # MCP
    "convert_mcp_to_anthropic", "is_mcp_tool",
    # credentials
    "ProviderCredentials", "CredentialResolver", "get_credential_resolver",
]
```

### 6.7 Fix internal imports in moved adapter files

Each adapter previously imported `from lattice.providers.base import ...`. After move:

```bash
sd 'from lattice\.providers\.base import' 'from lattice.providers.adapters.base import' src/lattice/providers/adapters/*.py
```

Or equivalently use relative imports (preferred for sibling files):

```python
# inside providers/adapters/openai.py:
from .base import _pop_system, _remap_tool_choice, _remap_tools, _strip_provider_prefix, _format_sse_event
```

Each adapter file's internal references to other adapters (e.g. `from lattice.providers.openai import OpenAIAdapter` inside `openai_compatible.py`):

```python
# inside providers/adapters/openai_compatible.py:
from .openai import OpenAIAdapter
```

Same pattern for `azure.py` (which inherits from `OpenAIAdapter`):

```python
from .openai import OpenAIAdapter
```

### 6.8 Fix imports in `transport/` modules

`providers/transport/registry.py` must import all 17 adapter classes for the `ProviderRegistry.__init__` to register them. Update:

```python
from lattice.providers.adapters import (
    AnthropicAdapter, AzureAdapter, BedrockAdapter,
    GeminiAdapter, VertexAdapter,
    OllamaAdapter, OllamaCloudAdapter, OpenAIAdapter,
    AI21Adapter, CohereAdapter, DeepSeekAdapter,
    FireworksAdapter, GroqAdapter, MistralAdapter,
    OpenRouterAdapter, PerplexityAdapter, TogetherAdapter,
)
```

`providers/transport/completion.py` imports the registry and pool:

```python
from .registry import ProviderRegistry, _resolve_provider_name
from .pool import ConnectionPoolManager
from .rate_limits import RateLimitTracker
from .helpers import _resolve_base_url, _resolve_api_key, _build_request, _should_retry
from lattice.transport.congestion import TACCController
from lattice.transport.types import Request, Response
from lattice.providers.credentials import CredentialResolver
from lattice.core.errors import ProviderError, ProviderTimeoutError
```

`providers/transport/streaming.py` imports helpers, stall detector, stream state:

```python
from .registry import ProviderRegistry
from .pool import ConnectionPoolManager
from .stall_detector import StreamStallDetector
from .helpers import (
    _resolve_base_url, _resolve_api_key, _parse_sse_line,
    _optimize_stream_chunk, _process_sse_line_with_state,
    _stream_chunk_text, _next_stream_line, _stream_retry_policy,
)
from lattice.transport.congestion import TACCController
from lattice.transport.types import Request
from lattice.providers.stream_state import AnthropicStreamState
from lattice.core.errors import ProviderError, ProviderTimeoutError
```

### 6.9 Rewrite consumer imports across the codebase

```bash
# Adapter imports
sd 'from lattice\.providers\.base import'              'from lattice.providers.adapters.base import'             $(rg -l "from lattice.providers.base import")
sd 'from lattice\.providers\.openai import'            'from lattice.providers.adapters.openai import'           $(rg -l "from lattice.providers.openai import")
sd 'from lattice\.providers\.openai_compatible import' 'from lattice.providers.adapters.openai_compatible import' $(rg -l "from lattice.providers.openai_compatible import")
sd 'from lattice\.providers\.anthropic import'         'from lattice.providers.adapters.anthropic import'        $(rg -l "from lattice.providers.anthropic import")
sd 'from lattice\.providers\.azure import'             'from lattice.providers.adapters.azure import'            $(rg -l "from lattice.providers.azure import")
sd 'from lattice\.providers\.bedrock import'           'from lattice.providers.adapters.bedrock import'          $(rg -l "from lattice.providers.bedrock import")
sd 'from lattice\.providers\.gemini import'            'from lattice.providers.adapters.gemini import'           $(rg -l "from lattice.providers.gemini import")
sd 'from lattice\.providers\.ollama import'            'from lattice.providers.adapters.ollama import'           $(rg -l "from lattice.providers.ollama import")

# Stall detector move
sd 'from lattice\.providers\.stall_detector import' 'from lattice.providers.transport.stall_detector import' $(rg -l "from lattice.providers.stall_detector import")
```

Note: `from lattice.providers.transport import DirectHTTPProvider` is **unchanged** — it still works because `providers/transport/` is now a package whose `__init__.py` re-exports `DirectHTTPProvider`. This is the deliberate design.

### 6.10 Update callers of `completion_stream_with_stall_detect`

If the merge in §3 chooses to keep two public method names: both methods stay; no caller change.

If the merge collapses to a single `completion_stream(request, *, stall_detect=True, stream_state=None)`:

```bash
sd 'completion_stream_with_stall_detect\(' 'completion_stream(' $(rg -l "completion_stream_with_stall_detect\\(")
# Then add `stall_detect=True, stream_state=stream_state` arguments where needed
```

The callers are: `proxy/routes.py`, `gateway/compat.py`, `gateway/server.py`. Manual fix at each call site to add the `stall_detect` and `stream_state` kwargs.

**Decision: keep two public methods** for v1.0.0 to minimise consumer change. Internally, both delegate to the merged `_stream`. v1.1 can collapse to a single public method via a deprecation cycle.

### 6.11 Refactor `tool_sanitizer.py` per §4

Apply the consolidation. `tests/unit/providers/test_tool_sanitizer.py` covers Anthropic and Bedrock paths.

### 6.12 Refactor `rate_limits.py` per §5

Add TTL field, `_maybe_cleanup` method, optional integration with `MaintenanceCoordinator`.

### 6.13 Final verification

```bash
uv run ruff check src/ tests/
uv run mypy src/lattice/
uv run pytest tests/ -q
uv run pytest tests/contract/ -q

# Provider-specific sanity: run a single chat completion against each provider that has API keys
# (skipped in CI without keys; manual local check before merge)
LATTICE_PROVIDER_BASE_URL=http://localhost:11434/v1 \
    uv run python -c "from lattice.providers import DirectHTTPProvider, ProviderRegistry, ConnectionPoolManager; from lattice.transport.types import Request, Message, Role; ..."

# Benchmark
uv run python benchmarks/evals/cli.py --suite all --providers ollama-cloud --provider-model ollama-cloud=kimi-k2.6:cloud --iterations 1 --warmup 0 --provider-warmup 0 --output-json benchmarks/results/phase-5.json
python scripts/compare_benchmarks.py benchmarks/results/phase-0-baseline.json benchmarks/results/phase-5.json --tolerance-pct 2
```

---

## 7. Symbol migration table

| Old fully-qualified | New fully-qualified |
|---|---|
| `lattice.providers.base.ProviderAdapter` | `lattice.providers.adapters.base.ProviderAdapter` (or `lattice.providers.ProviderAdapter` via re-export) |
| `lattice.providers.base._pop_system` | `lattice.providers.adapters.base._pop_system` |
| `lattice.providers.openai.OpenAIAdapter` | `lattice.providers.adapters.openai.OpenAIAdapter` |
| `lattice.providers.openai_compatible.GroqAdapter` | `lattice.providers.adapters.openai_compatible.GroqAdapter` |
| `lattice.providers.anthropic.AnthropicAdapter` | `lattice.providers.adapters.anthropic.AnthropicAdapter` |
| `lattice.providers.azure.AzureAdapter` | `lattice.providers.adapters.azure.AzureAdapter` |
| `lattice.providers.bedrock.BedrockAdapter` | `lattice.providers.adapters.bedrock.BedrockAdapter` |
| `lattice.providers.gemini.GeminiAdapter` | `lattice.providers.adapters.gemini.GeminiAdapter` |
| `lattice.providers.gemini.VertexAdapter` | `lattice.providers.adapters.gemini.VertexAdapter` |
| `lattice.providers.ollama.OllamaAdapter` | `lattice.providers.adapters.ollama.OllamaAdapter` |
| `lattice.providers.ollama.OllamaCloudAdapter` | `lattice.providers.adapters.ollama.OllamaCloudAdapter` |
| `lattice.providers.stall_detector.StreamStallDetector` | `lattice.providers.transport.stall_detector.StreamStallDetector` |
| `lattice.providers.transport.DirectHTTPProvider` | `lattice.providers.transport.DirectHTTPProvider` *(unchanged import path; now from package)* |
| `lattice.providers.transport.ProviderRegistry` | `lattice.providers.transport.ProviderRegistry` *(unchanged)* |
| `lattice.providers.transport.ConnectionPoolManager` | `lattice.providers.transport.ConnectionPoolManager` *(unchanged)* |
| `lattice.providers.transport.RateLimitTracker` | `lattice.providers.transport.RateLimitTracker` *(unchanged)* |
| `lattice.providers.transport._resolve_provider_name` | `lattice.providers.transport._resolve_provider_name` *(unchanged)* |
| `lattice.providers.tool_sanitizer.AnthropicToolSanitizer` | unchanged (now inherits from `ToolSanitizer`) |
| `lattice.providers.tool_sanitizer.BedrockToolSanitizer` | unchanged (now inherits from `ToolSanitizer`) |
| `lattice.providers.tool_sanitizer.ToolSanitizer` | **NEW** base class |

The `providers.transport` import path stays stable because it was already that string — only it's a package now, not a file.

---

## 8. Import-rewrite cheatsheet

```bash
# Adapter moves (file path change)
for adapter in base openai openai_compatible anthropic azure bedrock gemini ollama; do
    sd "from lattice\.providers\.${adapter} import" "from lattice.providers.adapters.${adapter} import" $(rg -l "from lattice.providers.${adapter} import")
done

# Stall detector move
sd 'from lattice\.providers\.stall_detector import' 'from lattice.providers.transport.stall_detector import' $(rg -l "from lattice.providers.stall_detector import")
```

After:

```bash
uv run ruff check src/ tests/
rg "from lattice.providers.base|from lattice.providers.openai|from lattice.providers.anthropic|from lattice.providers.azure|from lattice.providers.bedrock|from lattice.providers.gemini|from lattice.providers.ollama|from lattice.providers.openai_compatible" src/ tests/ benchmarks/
# Expected: 0 matches
rg "from lattice.providers.stall_detector" src/ tests/ benchmarks/
# Expected: 0 matches
```

---

## 9. Tests

### 9.1 Move existing tests

```bash
mkdir -p tests/unit/providers/transport tests/unit/providers/adapters
git mv tests/unit/providers/test_stall_detector.py tests/unit/providers/transport/test_stall_detector.py
# Per-adapter tests
git mv tests/unit/providers/test_openai.py        tests/unit/providers/adapters/test_openai.py
git mv tests/unit/providers/test_anthropic.py     tests/unit/providers/adapters/test_anthropic.py
git mv tests/unit/providers/test_azure.py         tests/unit/providers/adapters/test_azure.py
git mv tests/unit/providers/test_bedrock.py       tests/unit/providers/adapters/test_bedrock.py
git mv tests/unit/providers/test_gemini.py        tests/unit/providers/adapters/test_gemini.py
git mv tests/unit/providers/test_ollama.py        tests/unit/providers/adapters/test_ollama.py
git mv tests/unit/providers/test_openai_compatible.py tests/unit/providers/adapters/test_openai_compatible.py
# Transport-level
git mv tests/unit/test_transport*.py    tests/unit/providers/transport/
git mv tests/unit/test_provider_registry.py tests/unit/providers/transport/test_registry.py
git mv tests/unit/test_connection_pool.py   tests/unit/providers/transport/test_pool.py
git mv tests/unit/test_rate_limit.py        tests/unit/providers/transport/test_rate_limits.py
```

### 9.2 New tests

**`tests/unit/providers/transport/test_streaming_single_path.py`** — verify both public stream methods route through `_stream`:

```python
async def test_completion_stream_uses_unified_path(monkeypatch, provider_factory):
    provider = provider_factory()
    seen = []
    async def fake_stream(*, stall_detect, stream_state, **kw):
        seen.append({"stall_detect": stall_detect, "stream_state": stream_state})
        yield {"choices": [{"delta": {"content": "x"}}]}
    monkeypatch.setattr(provider, "_stream", fake_stream)
    async for _ in provider.completion_stream(request, provider_name="openai"):
        pass
    assert seen[0]["stall_detect"] is False
    seen.clear()
    async for _ in provider.completion_stream_with_stall_detect(request, provider_name="openai"):
        pass
    assert seen[0]["stall_detect"] is True
```

**`tests/unit/providers/transport/test_rate_limits_ttl.py`** — verify TTL eviction:

```python
def test_rate_limit_evicts_stale(monkeypatch):
    tracker = RateLimitTracker()
    tracker.record("openai", {"x-ratelimit-remaining-requests": "100"})
    assert tracker.get("openai") is not None
    # Fast-forward time past TTL
    import time
    fake_now = time.time() + RATE_LIMIT_TTL_S + 1
    monkeypatch.setattr(time, "time", lambda: fake_now)
    # Force cleanup
    tracker._last_cleanup_at = 0
    tracker.record("anthropic", {})
    assert tracker.get("openai") is None    # evicted
    assert tracker.get("anthropic") is not None
```

**`tests/unit/providers/test_tool_sanitizer_inheritance.py`**:

```python
def test_sanitizers_share_base():
    from lattice.providers.tool_sanitizer import ToolSanitizer, AnthropicToolSanitizer, BedrockToolSanitizer
    assert issubclass(AnthropicToolSanitizer, ToolSanitizer)
    assert issubclass(BedrockToolSanitizer, ToolSanitizer)

def test_anthropic_sanitizer_validates():
    s = AnthropicToolSanitizer()
    assert s.validate_tool_id("good_tool-name")
    assert not s.validate_tool_id("bad tool!")
    safe = s.sanitize("bad tool!")
    assert s.validate_tool_id(safe)
    assert s.restore(safe) == "bad tool!"
```

**`tests/unit/providers/test_registry_complete.py`**:

```python
def test_all_17_providers_registered():
    from lattice.providers import ProviderRegistry
    reg = ProviderRegistry()
    expected = {
        "openai", "anthropic", "azure", "bedrock", "gemini", "vertex",
        "groq", "deepseek", "mistral", "cohere", "perplexity",
        "fireworks", "together", "openrouter", "ai21",
        "ollama", "ollama-cloud",
    }
    found = {a.name for a in reg.adapters}
    assert expected.issubset(found), f"Missing: {expected - found}"
```

### 9.3 Contract tests

`tests/contract/test_python_api_contract.py` — verify `from lattice.providers import ...` resolves all 17 adapter names:

```python
def test_all_adapters_importable_at_top_level():
    from lattice.providers import (
        OpenAIAdapter, AnthropicAdapter, AzureAdapter, BedrockAdapter,
        GeminiAdapter, VertexAdapter,
        GroqAdapter, TogetherAdapter, DeepSeekAdapter, PerplexityAdapter,
        MistralAdapter, FireworksAdapter, OpenRouterAdapter,
        CohereAdapter, AI21Adapter,
        OllamaAdapter, OllamaCloudAdapter,
    )
```

`tests/contract/test_http_contract.py` — already exists; running it against a live proxy proves the streaming merge didn't break SSE shape.

---

## 10. Acceptance criteria

- [ ] `src/lattice/providers/transport.py` (single file) does not exist.
- [ ] `src/lattice/providers/transport/` (package) exists with 7 files matching §2.1.
- [ ] `src/lattice/providers/adapters/` exists with 8 files.
- [ ] `src/lattice/providers/stall_detector.py` does not exist (moved to `transport/`).
- [ ] `src/lattice/providers/base.py` does not exist (moved to `adapters/`).
- [ ] No file in `src/lattice/providers/` exceeds 850 LoC.
- [ ] `rg "completion_stream_with_stall_detect|completion_stream" src/lattice/providers/transport/streaming.py | wc -l` — both methods present, but the body of each is < 30 LoC (delegates to `_stream`).
- [ ] `from lattice.providers import DirectHTTPProvider, ProviderRegistry, ConnectionPoolManager` works.
- [ ] `from lattice.providers import OpenAIAdapter, AnthropicAdapter, ..., AI21Adapter` (all 17) works.
- [ ] `from lattice.providers.transport import DirectHTTPProvider, StreamStallDetector` works.
- [ ] `from lattice.providers.adapters import OpenAIAdapter, AnthropicAdapter` works.
- [ ] `from lattice.providers.tool_sanitizer import ToolSanitizer, AnthropicToolSanitizer, BedrockToolSanitizer` works; both subclasses inherit from `ToolSanitizer`.
- [ ] `RateLimitTracker` has a `last_seen_at` field and a `_maybe_cleanup` method; new TTL test passes.
- [ ] `tests/unit/providers/transport/test_streaming_single_path.py` passes.
- [ ] `tests/unit/providers/test_tool_sanitizer_inheritance.py` passes.
- [ ] `tests/unit/providers/test_registry_complete.py` passes (all 17 adapters registered).
- [ ] `tests/contract/test_http_contract.py` passes (live proxy SSE shape verified).
- [ ] `uv run ruff check src/ tests/` clean.
- [ ] `uv run mypy src/lattice/` clean.
- [ ] `uv run pytest tests/ -q` passes.
- [ ] `python scripts/compare_benchmarks.py benchmarks/results/phase-0-baseline.json benchmarks/results/phase-5.json --tolerance-pct 2` exits 0.

---

## 11. Risks specific to this phase

| Risk | Mitigation |
|---|---|
| Streaming merge subtly changes SSE chunk ordering or buffering for one provider | `tests/contract/test_http_contract.py` records the **byte stream** for a known prompt from each provider (Phase 0 captures these as fixtures). Compare byte-for-byte after the merge. Any diff blocks merge. |
| The mixin pattern (Option A in §6.3) causes mypy to lose track of `DirectHTTPProvider.completion_stream` type | Define `StreamingMixin` as an abstract Protocol; have `DirectHTTPProvider(StreamingMixin)` explicit; mypy can resolve via MRO. If issues persist, switch to Option B (facade pattern). |
| Splitting the file across `completion.py` and `streaming.py` introduces import-cycle: `completion.py` needs `_stream` from `streaming.py`, but `streaming.py` may need `_build_request` from `completion.py` | Both shared helpers move to `helpers.py`. Neither `completion.py` nor `streaming.py` imports from the other. |
| Anthropic's tool sanitizer was using a *thread-local* contextvar that the base extracts. Anthropic's per-task isolation breaks if base's ContextVar default is shared across tasks | `ContextVar(default={})` is per-context; each async task gets its own. Verify by running `tests/integration/test_redis_backend.py` (which runs concurrent requests) and check no tool ID leakage between requests. |
| `RateLimitTracker._maybe_cleanup` mutates the dict during iteration of `_state.items()` | Use `{k: v for k, v in self._state.items() if ...}` (creates new dict) — safe. |
| `OllamaCloudAdapter` inherits from `OpenAIAdapter` but the move puts both in `adapters/` — relative import must work | `providers/adapters/ollama.py` does `from .openai import OpenAIAdapter` — works. |
| `gateway/compat.py` directly calls `DirectHTTPProvider.completion_stream_with_stall_detect(...)` — the method must still exist | Decision in §6.10: keep both public methods. No gateway change needed. |
| `benchmarks/evals/runner.py` instantiates a `DirectHTTPProvider` directly with specific kwargs | Verify the constructor signature is unchanged. The split should not change `__init__`. |
| MCP support: `convert_mcp_to_anthropic` is imported by `anthropic.py` which now lives in `adapters/`. The path `from lattice.providers.mcp_to_anthropic import` still resolves because `mcp_to_anthropic.py` stays at `providers/` top level | Verify with `rg "from lattice.providers.mcp_to_anthropic import"` post-move — should still resolve. |
| 17 provider integration tests in `tests/integration/` all instantiate `DirectHTTPProvider` directly — they may rely on attributes that move | Each integration test runs after the split; if any breaks, the offending attribute is likely a `_private` that became `_resolve_helper()` in `helpers.py`. Re-export it from `providers/transport/__init__.py` if needed for tests. |

---

## 12. Rollback plan

```bash
# Restore the monolith
git checkout main -- src/lattice/providers/transport.py
git checkout main -- src/lattice/providers/stall_detector.py
git checkout main -- src/lattice/providers/base.py
git checkout main -- src/lattice/providers/openai.py
git checkout main -- src/lattice/providers/anthropic.py
git checkout main -- src/lattice/providers/azure.py
git checkout main -- src/lattice/providers/bedrock.py
git checkout main -- src/lattice/providers/gemini.py
git checkout main -- src/lattice/providers/ollama.py
git checkout main -- src/lattice/providers/openai_compatible.py
git checkout main -- src/lattice/providers/__init__.py
git rm -r src/lattice/providers/transport/ src/lattice/providers/adapters/
git checkout main -- src/ tests/   # revert sd-rewritten imports
```

This phase has the highest blast radius. Consider splitting into:

1. PR 5a: Move adapters to `adapters/`, no other changes.
2. PR 5b: Move `stall_detector.py` into `transport/`, refactor `tool_sanitizer.py` base class, fix `RateLimitTracker` TTL.
3. PR 5c: Split `providers/transport.py` into the 7-file package + merge streaming.

Each is independently revertible.

---

## 13. PR shape

Three PRs:

```
refactor(providers): move adapter files into providers/adapters/ [Phase 5a]
- 8 adapter files moved
- providers/__init__.py re-exports unchanged from consumer perspective
- ~50 import sites rewritten by sd

refactor(providers): tool_sanitizer base class; RateLimitTracker TTL; stall_detector → transport/ [Phase 5b]
- AnthropicToolSanitizer and BedrockToolSanitizer inherit from ToolSanitizer base
- RateLimitTracker evicts entries older than 1 hour
- providers/stall_detector.py → providers/transport/stall_detector.py
- ~10 import sites rewritten

refactor(providers/transport): split 1539-LoC monolith into 7-file package; merge streaming retry [Phase 5c]
- providers/transport.py → providers/transport/ package (7 files)
- DirectHTTPProvider mixin pattern: completion.py + streaming.py
- _stream() unified — both public stream methods delegate (saves ~300 LoC of duplicated retry/SSE/buffering)
- providers/__init__.py re-exports unchanged
- contract test verifies byte-for-byte SSE compatibility

Net: -1 file (transport.py), +7 files (transport/ pkg), +1 directory (adapters/), -300 LoC of duplication.
All 1600+ tests green. Contract tests green. Benchmarks ±2%.
```
