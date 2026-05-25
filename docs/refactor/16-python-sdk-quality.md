# Phase 16 — Python SDK as a Thin Client (zero algorithm duplication)

> **Footprint impact.** The Python SDK adds ~1 MB to the existing wheel (typed surfaces + thin clients). No new runtime deps beyond what the proxy already requires (`httpx`, `pydantic`).
>
> **Algorithm location.** Reverse-pass, alias substitution, streaming chunk buffering, IR fingerprinting all live in `src/lattice/` (the runtime). The SDK in `src/lattice/sdk/` is allowed to *call* them; it must not reimplement them. This is enforced by `scripts/check_sdk_no_algorithm_duplication.sh` from [Phase 31](31-edge-wasm-core.md).
>
> **External-service requirement.** None.
>

> **Transport role.** SDK is a thin HTTP client to the proxy transport layer — zero transport/retry/pool code in SDK.
> **Registry.** §12 SDK surfaces.

> **Guidelines.** [PHASE_GUIDELINES.md](PHASE_GUIDELINES.md) — constraints 1–6.
> **Outcome.** The README quick-start actually works on the first try (closes a confirmed credibility hole). `py.typed` ships in the wheel. Sync wrappers work (`wrap_openai` detects sync vs async clients). Streaming responses already arrive decoded from the proxy (Phase 27's server-side streaming reverse), so the SDK has nothing to decode itself — it's a thin pass-through. Hooks compose. `wrap_anthropic` is exported at top-level. `LatticeProxyClient.chat.create()` returns typed objects instead of `dict[str, Any]`.
>
> **Estimated effort.** 3 days (1 PR, ~+1100/-600 LoC — the LoC drops because we delete the duplicated stream-decode and reverse-pass code that the prior plan called for).

---

## 1. Why this phase exists, and what changed from the prior draft

Two reasons:

### 1.1 The audit findings still stand

[STATUS.md §6](STATUS.md), agent transcript:

1. **README quick-start is wrong.** [README.md](../../README.md) shows `LatticeClient().chat.completions.create(...)`. The class has no such attribute. First 30 seconds → `AttributeError`.
2. **No `py.typed`.** `mypy` users get `Cannot find implementation or library stub for module lattice`.
3. **Async-only.** Sync OpenAI SDK users — still the majority — can't wrap their clients.
4. **`wrap_openai` is async-only.** Patches with `async def`; a sync client breaks at call time.
5. **`wrap_anthropic` not re-exported at top-level.**
6. **`LatticeProxyClient.chat.create()` returns untyped dict.** No IDE completion.
7. **No middleware/hook API.** Users must monkey-patch.

### 1.2 The brutal change vs the prior draft

The prior draft proposed implementing in the SDK:

- A streaming reverse-pass with sliding-window placeholder handling
- A `LatticeHooks` machinery with five hook points
- Sync/async sliding-window chunk buffer

That was the wrong architectural call. **Reverse-pass already runs in the proxy** ([Phase 27 streaming-native compression](22-compression-intelligence.md)). Chunks arrive at the SDK already decoded. The SDK has no decoding to do.

This rewrite removes the duplicated logic and keeps the SDK to its true responsibilities:

- HTTP plumbing
- Type-safe request/response surfaces
- Orchestration hooks (run user-supplied functions at well-defined points)
- Sync / async parity at the **interface** level

If — and only if — the user runs the SDK without a proxy ([Phase 31](31-edge-wasm-core.md) in-process mode), the SDK calls into the *runtime's* implementations (or PyO3-bound `lattice-core-py`). Never reimplements them.

---

## 2. Files touched

### 2.1 Created

```
src/lattice/py.typed                             # PEP 561 marker
src/lattice/sdk/_types/__init__.py
src/lattice/sdk/_types/openai_chat.py            # TypedDicts, re-export from openai package when available
src/lattice/sdk/_types/openai_embeddings.py
src/lattice/sdk/_types/openai_responses.py
src/lattice/sdk/_types/anthropic_messages.py
src/lattice/sdk/_types/streaming.py
src/lattice/sdk/sync_proxy_client.py             # sync sibling of LatticeProxyClient
src/lattice/sdk/hooks.py                         # LatticeHooks dataclass; orchestration only
src/lattice/sdk/hooks_builtin.py                 # log_to_stderr, redact_keys, attach_request_id
src/lattice/sdk/_proxy_protocol.py               # shared request-body builder used by both sync + async
src/lattice/sdk/_in_process.py                   # OPTIONAL: in-process mode delegating to runtime
tests/contract/test_readme_examples.py
tests/contract/test_sdk_types_surface.py
tests/contract/test_sync_async_parity.py
tests/contract/test_sdk_no_algorithm_duplication.py    # CI gate companion
tests/unit/sdk/test_hooks.py
tests/unit/sdk/test_wrap_openai_sync.py
tests/unit/sdk/test_in_process_delegation.py
```

### 2.2 Modified

| File | Change |
|---|---|
| [src/lattice/__init__.py](../../src/lattice/__init__.py) | Add `SyncLatticeProxyClient`, `wrap_anthropic_client`, `LatticeHooks` to top-level `__all__` |
| [src/lattice/sdk/wrappers.py](../../src/lattice/sdk/wrappers.py) | Detect sync vs async client; in proxy mode, just redirect the `base_url`; in in-process mode, delegate to runtime |
| [src/lattice/sdk/proxy_client.py](../../src/lattice/sdk/proxy_client.py) | Typed return values; accept `hooks`; delegate request-body building to `_proxy_protocol` |
| [src/lattice/client.py](../../src/lattice/client.py) | Documents explicitly that `LatticeClient` is NOT an HTTP client; points users at proxy client / wrap helpers / in-process mode |
| [README.md](../../README.md) | Quick Start rewritten — three working snippets, each backed by a contract test |
| [pyproject.toml](../../pyproject.toml) | `package-data = {"lattice" = ["py.typed"]}`; explicit hatch include for `py.typed` |
| [docs/getting-started/quickstart.md](../../docs/getting-started/quickstart.md) | Mirror new README |

### 2.3 Deleted

| File | Why |
|---|---|
| [src/lattice/sdk/client.py](../../src/lattice/sdk/client.py) | v1.0 deprecation shim — removal date arrives in v1.1 |
| Any draft `sdk/streaming.py` / `sdk/_response_decoder.py` from the prior plan | Doesn't exist in the codebase yet, but explicitly out of scope — the runtime handles decoding |

---

## 3. The thin-client doctrine, applied

### 3.1 Proxy mode (default)

```python
# src/lattice/sdk/wrappers.py
def wrap_openai(client: Any, *, base_url: str = "http://localhost:8787/v1",
                hooks: LatticeHooks | None = None) -> Any:
    """Wrap an OpenAI client by pointing its base_url at the LATTICE proxy.

    In proxy mode (default), this is the ENTIRE wrap operation. The proxy does
    all compression, all reverse-pass, all streaming decode. The SDK is a URL redirector.

    Hooks are still useful for client-side observability (e.g. timing, logging).
    They never participate in request body manipulation in proxy mode — the proxy owns that.
    """
    is_async = _is_async_openai_client(client)
    client.base_url = base_url  # the entire wrap

    if hooks:
        _install_observability_hooks(client, hooks, is_async=is_async)
    return client
```

That's it. Five lines of meaningful code. The compressed request flows to the proxy, the decoded response flows back. No SDK-side streaming buffer. No SDK-side reverse-pass. No SDK-side alias-table tracking.

### 3.2 In-process mode (advanced)

For users who explicitly opt out of running a proxy — for example, a script that wants compression without a separate process — the SDK calls into the LATTICE runtime:

```python
# src/lattice/sdk/_in_process.py
from lattice.pipeline.runner import Pipeline
from lattice.planner.unified_planner import UnifiedPlanner

def wrap_openai_in_process(client: Any, *, config: LatticeConfig | None = None,
                            hooks: LatticeHooks | None = None) -> Any:
    """Wrap an OpenAI client to run compression IN-PROCESS via the LATTICE runtime.

    This mode runs the *exact* same Pipeline that the proxy runs. There is no
    parallel implementation. If lattice-core-py is installed, transforms are
    accelerated by the native Rust core — automatically.

    Use only when:
      - A proxy is not running and is undesirable (one-off script, isolated test)
      - You want zero network hops to your provider
    Otherwise prefer the proxy mode: it shares cache, telemetry, and bandit state
    across all clients.
    """
    pipeline = Pipeline.from_config(config or LatticeConfig.from_env())
    planner = UnifiedPlanner.from_config(pipeline.config)

    original_create = client.chat.completions.create

    async def acreate(**kwargs):
        request = _kwargs_to_request(kwargs)
        plan = planner.plan(request)
        compressed = await pipeline.compress(request, plan)
        new_kwargs = _request_to_kwargs(compressed.request, kwargs)
        if kwargs.get("stream"):
            upstream = await original_create(**new_kwargs)
            return pipeline.stream_reverse(upstream, compressed.alias_table)
        response = await original_create(**new_kwargs)
        return pipeline.reverse_response(response, compressed.alias_table)

    client.chat.completions.create = acreate
    return client
```

Note what is **not** in this file:

- No reverse-pass implementation. `pipeline.reverse_response` lives in the runtime.
- No streaming decoder. `pipeline.stream_reverse` lives in the runtime.
- No alias-table machinery. `AliasTable` is a runtime class.
- No transform code.

The in-process mode is glue. The algorithms stay in the runtime.

### 3.3 The choice surfaces explicitly

```python
from lattice import wrap_openai_client            # proxy mode (default; one-liner)
from lattice import wrap_openai_in_process        # in-process mode (explicit; advanced)
```

There's no ambiguity. If you call `wrap_openai_client`, you get a URL redirector. If you call `wrap_openai_in_process`, you opt into running the pipeline in your own process. Both produce a client with the same external behaviour — the difference is where compression executes.

---

## 4. The other audit fixes

### 4.1 `py.typed` packaged in the wheel

```bash
touch src/lattice/py.typed
```

```toml
# pyproject.toml
[tool.hatch.build.targets.wheel]
packages = ["src/lattice"]

[tool.hatch.build.targets.wheel.force-include]
"src/lattice/py.typed" = "lattice/py.typed"
```

Contract test verifies the file ends up in the wheel.

### 4.2 TypedDict surfaces (re-export when possible)

```python
# src/lattice/sdk/_types/openai_chat.py
try:
    from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage  # type: ignore
except ImportError:
    # Minimal vendored TypedDicts when openai isn't installed.
    # NOT a re-implementation of OpenAI logic; just shape declarations.
    class ChatCompletionMessage(TypedDict):
        role: Literal["assistant", "user", "system", "tool"]
        content: str | None
        # ... small set of fields, structural only
```

Same approach for `embeddings`, `responses`, `anthropic.messages`. Where the upstream package's types are available we re-export; the vendored fallback is structural only.

### 4.3 Sync proxy client

`src/lattice/sdk/sync_proxy_client.py`:

```python
class SyncLatticeProxyClient:
    """Synchronous variant of LatticeProxyClient. Identical API; blocking I/O."""

    def __init__(self, base_url: str = "http://localhost:8787", *, api_key: str | None = None,
                 timeout: float = 30.0, hooks: LatticeHooks | None = None) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=timeout)
        self._api_key = api_key
        self._hooks = hooks or LatticeHooks()
        self.chat = SyncChatCompletionsResource(self)
        self.sessions = SyncSessionsResource(self)

    def health(self) -> dict[str, Any]: ...
    def stats(self) -> dict[str, Any]: ...
    def close(self) -> None: self._http.close()
    def __enter__(self): return self
    def __exit__(self, *a): self.close()
```

Both clients construct identical HTTP requests via `_proxy_protocol.build_chat_request_body(...)`. Contract test `test_sync_async_parity.py` runs the same fixture through both, captures HTTP requests with `respx`, and diffs them.

### 4.4 Sync detection in `wrap_openai`

```python
def _is_async_openai_client(client: Any) -> bool:
    """Detect sync vs async OpenAI client by inspecting the create method.

    OpenAI ships separate AsyncOpenAI / OpenAI classes whose methods differ in shape.
    """
    create = getattr(getattr(getattr(client, "chat", None), "completions", None), "create", None)
    if create is None:
        return False
    return inspect.iscoroutinefunction(create)
```

In the sync case the wrapper installs a sync function; in the async case an async function. Both bodies are 5 lines (proxy mode) or delegate to `_in_process` (in-process mode).

### 4.5 `wrap_anthropic` at top-level

```python
# src/lattice/__init__.py
from lattice.sdk.wrappers import wrap_openai, wrap_anthropic
wrap_openai_client = wrap_openai
wrap_anthropic_client = wrap_anthropic
__all__ += ["wrap_openai_client", "wrap_anthropic_client", "wrap_openai", "wrap_anthropic",
            "SyncLatticeProxyClient", "LatticeHooks"]
```

### 4.6 Hooks as orchestration, not algorithm

```python
# src/lattice/sdk/hooks.py
HookOnRequest = Callable[[dict[str, Any]], dict[str, Any] | None]
HookOnResponse = Callable[[Any], None]
HookOnError = Callable[[BaseException], None]

@dataclass(frozen=True, slots=True)
class LatticeHooks:
    """User-supplied callbacks at well-defined SDK extension points.

    Note: in proxy mode, request bodies are NOT modified before sending to the proxy.
    The proxy is the source of truth for what gets sent to the upstream provider.
    Hooks here are for SDK-side observability (logging, metrics, request tagging).

    In in-process mode, on_request runs BEFORE compression and may modify the body.
    """
    on_request: tuple[HookOnRequest, ...] = ()
    on_response: tuple[HookOnResponse, ...] = ()
    on_error: tuple[HookOnError, ...] = ()
    strict: bool = False

    def __or__(self, other: "LatticeHooks") -> "LatticeHooks":
        return LatticeHooks(
            on_request=self.on_request + other.on_request,
            on_response=self.on_response + other.on_response,
            on_error=self.on_error + other.on_error,
            strict=self.strict or other.strict,
        )
```

Hooks are 3 callback lists. No "before compress / after compress / before send" — those decisions happen inside the runtime in proxy mode, and the runtime has its own hook surfaces if a user needs to plug deeper.

`hooks_builtin.py` ships three useful presets: `log_to_stderr()`, `redact_keys({"email", "ssn"})`, `attach_request_id()`. Compose with `|`.

### 4.7 README fix

[README.md](../../README.md) Quick Start gets three working snippets:

```python
# Path 1 — drop-in proxy (works for any OpenAI-compatible SDK in any language)
# Shell: lattice proxy run --port 8787
import os, openai
os.environ["OPENAI_BASE_URL"] = "http://localhost:8787/v1"
r = openai.OpenAI().chat.completions.create(model="openai/gpt-4o", messages=[...])

# Path 2 — wrap an existing OpenAI client (sync or async; the wrapper detects)
from lattice import wrap_openai_client
import openai
client = wrap_openai_client(openai.OpenAI())   # one-liner; sync supported
r = client.chat.completions.create(model="openai/gpt-4o", messages=[...])

# Path 3 — typed native LATTICE client (recommended for new code)
from lattice import LatticeProxyClient
async with LatticeProxyClient() as client:
    r = await client.chat.create(model="openai/gpt-4o", messages=[...])
```

Anthropic path:

```python
from lattice import wrap_anthropic_client
import anthropic
client = wrap_anthropic_client(anthropic.Anthropic())
r = client.messages.create(model="claude-3-5-sonnet-latest", messages=[...])
```

Each runs in `tests/contract/test_readme_examples.py` via a `respx`-mocked upstream + a real proxy fixture.

---

## 5. The CI gate against algorithm duplication in SDK

`tests/contract/test_sdk_no_algorithm_duplication.py`:

```python
"""Companion to scripts/check_sdk_no_algorithm_duplication.sh (Phase 31).

This Python test exists so the contract is enforceable on every PR, not just
on the unified gate.
"""
FORBIDDEN_IN_SDK = (
    r"def\s+reverse_substitute\b",
    r"def\s+build_canonical_ir\b",
    r"def\s+apply_reference_sub\b",
    r"class\s+ChunkBuffer\b",
    r"def\s+xxh3\b",
    r"def\s+canonical_serialize\b",
    r"def\s+compute_fingerprint\b",
    r"UUID_REGEX\s*=",
    r"URL_REGEX\s*=",
)

SDK_PATHS = ["src/lattice/sdk"]

def test_sdk_contains_no_algorithm_implementations():
    failures = []
    for path in SDK_PATHS:
        for py_file in pathlib.Path(path).rglob("*.py"):
            text = py_file.read_text()
            for pat in FORBIDDEN_IN_SDK:
                if re.search(pat, text):
                    failures.append(f"{py_file}: matches {pat}")
    assert not failures, (
        "SDK source must not reimplement runtime algorithms. "
        "Move the implementation to src/lattice/<module>/ or crates/lattice-core/ "
        "and have the SDK call it. Findings:\n" + "\n".join(failures)
    )
```

This test ships with Phase 19 and tightens with each subsequent phase.

---

## 6. Test plan

| Check | Command | Threshold |
|---|---|---|
| Lint / format / types | usual | clean |
| Unit | `uv run pytest tests/unit -q` | + new tests pass |
| Contract: readme | `pytest tests/contract/test_readme_examples.py` | All three paths execute |
| Contract: SDK no-duplication | `pytest tests/contract/test_sdk_no_algorithm_duplication.py` | exit 0 |
| Contract: sync/async parity | `pytest tests/contract/test_sync_async_parity.py` | byte-identical HTTP requests |
| Contract: py.typed in wheel | `pytest tests/contract/test_py_typed_in_wheel.py` | passes |
| Lean install | `pip install lattice-transport && python -c "from lattice import wrap_openai_client; wrap_openai_client(openai.OpenAI())"` | no error |
| Footprint | `tests/integration/footprint/test_4gb_laptop.py` | proxy + SDK fit in 1.5 GB virtual memory |

---

## 7. Acceptance criteria

1. The exact README quick-start snippets, copy-pasted into a fresh Python file, run successfully against a local proxy. (Contract test enforces.)
2. `python -c "from lattice import wrap_openai_client, wrap_anthropic_client, LatticeProxyClient, SyncLatticeProxyClient, LatticeHooks"` succeeds with no warnings.
3. A `wrap_openai_client(openai.OpenAI())` call (sync) on a streaming request returns an iterator whose chunks **arrive already-decoded from the proxy** — verified by capturing the raw upstream SSE bytes and confirming the SDK does no further transformation.
4. `python -c "import lattice.sdk; assert not any(... matches reverse_substitute ...)"` — i.e. the CI gate passes.
5. `git grep "from lattice.sdk.client"` returns 0 results in `src/` and `tests/` (deletion of v1.0 shim).
6. With `lattice-core-py` installed and `wrap_openai_in_process(client)` used, the in-process pipeline produces the same compressed request as the proxy on the same input (cross-mode parity test).
7. Footprint test confirms SDK adds ≤ 1 MB to the wheel and ≤ 5 MB RSS.

---

## 8. Out of scope

| Topic | Phase |
|---|---|
| Streaming reverse-pass in the SDK | Doesn't exist; lives in the runtime ([Phase 27](22-compression-intelligence.md)). The SDK never decodes. |
| OpenTelemetry hook | Lives in [Phase 22](19-otel-genai.md), exposed as a built-in hook factory `from lattice.sdk.hooks_builtin import emit_otel`. |
| Sync wrappers for the Vercel AI SDK | TypeScript ecosystem; [Phase 24](20-typescript-sdk.md). |
| Embedding / batch / audio / files wrappers | Surfaces ship in [Phase 31](24-non-chat-surfaces.md); wrap helpers follow with that phase. |
