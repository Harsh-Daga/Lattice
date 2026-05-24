# Phase 6 — Proxy / Gateway / SDK / CLI

> **Goal.** The proxy, gateway, SDK, and CLI layers are already structurally clean per the audit. Phase 6 (a) wires the orphaned `proxy/health.py` endpoints into the FastAPI app, (b) clarifies the three-client situation by making `LatticeClient`, `LatticeProxyClient`, `wrap_openai_client` the **only** top-level imports and deprecating the `lattice.sdk.client` back-compat shim with a one-release sunset, (c) deletes the orphaned `proxy/compat_exports.py` if not already gone, (d) confirms the public contract suite from Phase 0 catches every CLI command and HTTP endpoint, (e) lifts response-side header emission into a single place. The CLI stays a single 1011-LoC `cli.py` (well under the 800-LoC-per-concern guideline because it's all command parsing and is easier to read in one file).
>
> **Outcome.** Every CLI subcommand and every HTTP endpoint listed in §2 of the master plan is covered by a contract test that runs in CI. `/healthz`, `/readyz`, `/startupz`, `/metrics`, `/stats` are wired explicitly. The Python API has exactly three classes: `LatticeClient` (local), `LatticeProxyClient` (HTTP), and the `wrap_openai_client` function. No more `lattice.sdk.client` indirection.
>
> **Estimated effort.** 1 day.

---

## 1. Why this phase exists

The audit found four issues in this layer:

1. **`proxy/health.py` defines `HealthManager` with `healthz`, `readyz`, `startupz`, `stats`, `metrics` methods, but no caller wires it up.** The endpoints exist *as code* but the FastAPI app doesn't expose them. The contract test from Phase 0 would catch this (and *should* catch this).
2. **Three client classes for two semantics.**
   - `lattice/client.py` (326 LoC) — canonical local compression client; what users actually use.
   - `lattice/sdk/proxy_client.py` (373 LoC) — HTTP client for a running proxy.
   - `lattice/sdk/client.py` (15 LoC) — pure re-export shim of `LatticeClient` so `from lattice.sdk import LatticeClient` works.
   The shim is back-compat from a refactor that happened before v0.1. Cut it; consumers import from `lattice` directly.
3. **`src/lattice/proxy/compat_exports.py` is deleted in working tree but not committed.** Commit.
4. **Response-side header emission is scattered.** Headers like `x-lattice-compression`, `x-lattice-session-id`, `x-lattice-delta`, `x-lattice-cost-usd`, `x-lattice-provider`, `x-lattice-transforms-applied` are set in `gateway/compat.py`, `proxy/server.py` middleware, and per-handler in `proxy/routes.py`. Consolidate into a single middleware that reads from `TransformContext` after the response is built.

The CLI itself doesn't need restructuring. 1011 LoC in `cli.py` is monolithic but cohesive — it's all command-line argument parsing and pretty-printing. Splitting per-command would scatter help-text formatting and shared utilities. **Keep `cli.py` as a single file** until it grows past 1500 LoC.

---

## 2. Files touched

### 2.1 Modified

- `src/lattice/proxy/server.py` — wire health/metrics routes via `routes.py`; install the consolidated response-header middleware.
- `src/lattice/proxy/routes.py` — add `register_health_routes(app, health_manager)` function.
- `src/lattice/proxy/health.py` — minor adjustments to method signatures so they slot into FastAPI route handlers cleanly.
- `src/lattice/proxy/bootstrap.py` — instantiate `HealthManager` in `ProxyRuntime`; pass to routes.
- `src/lattice/__init__.py` — top-level re-exports for the three client surfaces.
- `src/lattice/sdk/__init__.py` — re-export the same three; mark `sdk.client` import path deprecated.
- `src/lattice/gateway/compat.py` — remove per-handler header setting; rely on the new middleware.
- `src/lattice/cli.py` — no structural change. Verify all subcommands are reachable. Add `lattice version` as an alias for `--version` (a common UX request).

### 2.2 Created

```
src/lattice/proxy/middleware.py     # NEW — consolidated header / context middleware
tests/unit/proxy/test_health_routes_registered.py
tests/unit/proxy/test_response_headers.py
```

### 2.3 Deleted

```
src/lattice/proxy/compat_exports.py     # already gone in working tree; commit the deletion
src/lattice/sdk/client.py               # 15-LoC re-export shim — deprecated in v1.0.0, removed in v1.1
                                        # Leave as a deprecation-warning module in v1.0.0; delete only the body.
```

**v1.0.0 final treatment of `sdk/client.py`** — replace the file with:

```python
"""DEPRECATED: import LatticeClient from `lattice` directly.

In v1.1 this module will be removed.
"""
import warnings
from lattice.client import LatticeClient, CompressResult

warnings.warn(
    "lattice.sdk.client is deprecated; import LatticeClient from `lattice` "
    "or `lattice.sdk` instead. This module will be removed in v1.1.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["LatticeClient", "CompressResult"]
```

Phase 11 schedules deletion for v1.1.

---

## 3. Step-by-step

### 3.1 Commit the deletion of `proxy/compat_exports.py`

```bash
git rm src/lattice/proxy/compat_exports.py     # working tree already shows it as deleted
git commit -m "chore(proxy): commit removal of orphaned compat_exports.py"
```

Verify nothing imports it:

```bash
rg "compat_exports" src/ tests/ benchmarks/    # 0 matches
```

### 3.2 Wire `HealthManager` into `bootstrap.py`

Today `proxy/bootstrap.py` builds a `ProxyRuntime` dataclass that does NOT include `HealthManager`. Add it:

```python
# proxy/bootstrap.py — additions

from lattice.proxy.health import HealthManager
from lattice.telemetry.metrics import MetricsCollector   # Phase 8 moves this; Phase 6 still uses core/metrics

@dataclass
class ProxyRuntime:
    # existing fields...
    health_manager: HealthManager
    metrics: MetricsCollector

def build_proxy_runtime(config: LatticeConfig) -> ProxyRuntime:
    # existing wiring...
    metrics = MetricsCollector()
    health_manager = HealthManager(
        config=config,
        metrics=metrics,
        version=__version__,
        pipeline=pipeline,
        store=session_store,
    )
    return ProxyRuntime(
        # existing fields...
        health_manager=health_manager,
        metrics=metrics,
    )
```

### 3.3 Add `register_health_routes` to `proxy/routes.py`

```python
# proxy/routes.py — addition

from fastapi import FastAPI, Response
from lattice.proxy.health import HealthManager

def register_health_routes(app: FastAPI, health: HealthManager) -> None:
    """Register /healthz, /readyz, /startupz, /metrics, /stats."""

    @app.get("/healthz")
    async def _healthz():
        return health.healthz()

    @app.get("/readyz")
    async def _readyz():
        body, status = health.readyz()
        return Response(content=body, status_code=status, media_type="application/json")

    @app.get("/startupz")
    async def _startupz():
        return health.startupz()

    @app.get("/metrics")
    async def _metrics():
        return Response(content=health.metrics(), media_type="text/plain; version=0.0.4")

    @app.get("/stats")
    async def _stats():
        return health.stats()
```

### 3.4 Call `register_health_routes` from `proxy/server.py`

```python
# proxy/server.py — change

from lattice.proxy.routes import (
    register_native_lattice_routes,
    register_provider_compat_routes,
    register_health_routes,           # NEW
)

def create_app(config: LatticeConfig) -> FastAPI:
    runtime = build_proxy_runtime(config)
    app = FastAPI(...)
    configure_cors(app, config)
    configure_lifecycle(app, ...)
    install_middleware(app)           # see §3.5
    register_health_routes(app, runtime.health_manager)        # NEW
    register_native_lattice_routes(app, runtime.gateway)
    register_provider_compat_routes(app, runtime.compat, deps)
    return app
```

### 3.5 Create `proxy/middleware.py` for consolidated response headers

```python
"""Consolidated x-lattice-* response header middleware.

Every per-request side-effect is recorded on TransformContext (or its descendant
ResponseContext). After the route handler runs, this middleware reads from the
context object stored in request.state and emits the headers in a single place.

Replaces per-handler header setting in gateway/compat.py.
"""

from fastapi import Request as FastAPIRequest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

_HEADER_KEYS = {
    "compression_pct":    "x-lattice-compression",
    "session_id":         "x-lattice-session-id",
    "delta_used":         "x-lattice-delta",
    "cost_usd":           "x-lattice-cost-usd",
    "provider":           "x-lattice-provider",
    "transforms_applied": "x-lattice-transforms-applied",
}

class LatticeHeaderMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: FastAPIRequest, call_next):
        response = await call_next(request)
        ctx = getattr(request.state, "transform_context", None)
        if ctx is None:
            return response
        for ctx_key, header_name in _HEADER_KEYS.items():
            value = ctx.get_summary().get(ctx_key)
            if value is None:
                continue
            if isinstance(value, bool):
                value = "true" if value else "false"
            elif isinstance(value, float):
                value = f"{value:.3f}"
            elif isinstance(value, (list, tuple)):
                value = ",".join(map(str, value))
            else:
                value = str(value)
            response.headers[header_name] = value
        return response
```

And in `proxy/server.py`:

```python
def install_middleware(app: FastAPI) -> None:
    from lattice.proxy.middleware import LatticeHeaderMiddleware
    app.add_middleware(LatticeHeaderMiddleware)
```

### 3.6 Remove per-handler header setting in `gateway/compat.py`

`gateway/compat.py` currently does things like:

```python
response.headers["x-lattice-compression"] = f"{ratio:.2f}"
response.headers["x-lattice-session-id"] = session_id
```

Replace with attaching to context:

```python
context.set("compression_pct", ratio * 100)
context.set("session_id", session_id)
context.set("delta_used", was_delta_encoded)
context.set("cost_usd", cost_estimate.total_cost_usd)
context.set("provider", provider_name)
context.set("transforms_applied", applied_transforms)
```

Then attach `context` to `request.state` in the handler entry:

```python
async def chat_completions_handler(request: fastapi.Request, body: dict, deps: ...):
    context = TransformContext(...)
    request.state.transform_context = context
    # ... existing logic ...
    return response   # middleware will add headers from context after this returns
```

Audit hits: ~12 sites in `gateway/compat.py` set these headers manually. After this phase, those lines are deleted.

### 3.7 Top-level re-exports in `lattice/__init__.py`

```python
"""LATTICE — LLM Transport & Efficiency Layer.

Three client surfaces:

    LatticeClient        — local compression (no proxy required)
    LatticeProxyClient   — HTTP client for a running LATTICE proxy
    wrap_openai_client   — context manager to route OpenAI SDK through LATTICE

Also exports:
    CompressResult       — return type of LatticeClient.compress
    __version__          — package version string
    PromptIR, PromptIRV2 — IR types for power users
    build_ir             — convenience to build IR from a Request
    serialize_ir_to_text — render IR back to a string
"""

from lattice._version import __version__
from lattice.client import LatticeClient, CompressResult
from lattice.sdk.proxy_client import LatticeProxyClient
from lattice.sdk.wrappers import wrap_openai_client
from lattice.ir import PromptIR, PromptIRV2, build_ir, serialize_ir_to_text

__all__ = [
    "__version__",
    "LatticeClient", "CompressResult",
    "LatticeProxyClient",
    "wrap_openai_client",
    "PromptIR", "PromptIRV2", "build_ir", "serialize_ir_to_text",
]
```

### 3.8 Update `sdk/__init__.py`

```python
"""LATTICE SDK — client surfaces for using LATTICE from Python code.

For most users:
    from lattice import LatticeClient, LatticeProxyClient, wrap_openai_client

This module re-exports the same names. Importing from `lattice.sdk.client` is
deprecated in v1.0.0 and will be removed in v1.1.
"""

from lattice.client import LatticeClient, CompressResult
from lattice.sdk.proxy_client import LatticeProxyClient
from lattice.sdk.wrappers import wrap_openai_client

__all__ = [
    "LatticeClient", "CompressResult",
    "LatticeProxyClient",
    "wrap_openai_client",
]
```

### 3.9 Replace `sdk/client.py` with deprecation shim

Already shown in §2.3. The file:

```python
import warnings
from lattice.client import LatticeClient, CompressResult

warnings.warn(
    "lattice.sdk.client is deprecated; import from `lattice` or `lattice.sdk` "
    "instead. This module will be removed in v1.1.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["LatticeClient", "CompressResult"]
```

### 3.10 Add `lattice version` alias

In `cli.py`'s main dispatcher:

```python
def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        _print_help()
        return
    if args[0] in ("-v", "--version", "version"):     # add "version"
        console.print(f"lattice {__version__}")
        return
    ...
```

### 3.11 Verify all CLI subcommands route correctly

Run:

```bash
for cmd in proxy init lace unlace info config benchmark health status doctor; do
    uv run lattice $cmd --help > /dev/null && echo "$cmd: OK" || echo "$cmd: FAIL"
done

# Proxy subcommands
for sub in run start stop restart status; do
    uv run lattice proxy $sub --help > /dev/null && echo "proxy $sub: OK" || echo "proxy $sub: FAIL"
done
```

All must print OK.

### 3.12 Final verification

```bash
uv run ruff check src/ tests/
uv run mypy src/lattice/
uv run pytest tests/ -q
uv run pytest tests/contract/ -q   # NOW includes the health-endpoints check

# Manual: start proxy, hit each endpoint
uv run lattice proxy start --port 8787
for ep in healthz readyz startupz metrics stats; do
    curl -s http://localhost:8787/$ep -o /dev/null -w "%{http_code}\n"
done
uv run lattice proxy stop
```

Each must return 200 (or the documented status; `readyz` returns 503 if pipeline not warm).

---

## 4. Per-file disposition

| File | LoC | Action |
|---|---|---|
| `proxy/bootstrap.py` | 300 | MODIFY — add `HealthManager` and `MetricsCollector` to `ProxyRuntime` |
| `proxy/server.py` | 338 | MODIFY — call `register_health_routes`; install `LatticeHeaderMiddleware` |
| `proxy/routes.py` | 328 | MODIFY — add `register_health_routes(app, health)` function |
| `proxy/health.py` | 112 | MINOR — adjust method signatures to return `(body, status)` tuples for `readyz` (so route handler can pass status code) |
| `proxy/middleware.py` | NEW (~50) | CREATE — `LatticeHeaderMiddleware` |
| `proxy/lifecycle.py` | 359 | UNCHANGED |
| `proxy/__init__.py` | 21 | MINOR — re-export `HealthManager` for advanced use |
| `proxy/compat_exports.py` | — | **DELETED** (already in working tree) |
| `gateway/compat.py` | 3158 | MODIFY — remove ~12 per-handler header setters; attach context to `request.state` instead |
| `gateway/server.py` | 587 | MINOR — same context-attach pattern in native handlers |
| `gateway/routing.py` | 231 | UNCHANGED |
| `gateway/detect_helpers.py` | 174 | UNCHANGED |
| `client.py` | 326 | UNCHANGED |
| `sdk/__init__.py` | 19 | MODIFY — explicit re-exports per §3.8 |
| `sdk/client.py` | 15 | REPLACE — deprecation shim |
| `sdk/proxy_client.py` | 373 | UNCHANGED |
| `sdk/wrappers.py` | 150 | UNCHANGED |
| `__init__.py` | (small) | MODIFY — public re-exports per §3.7 |
| `cli.py` | 1011 | MINOR — add `lattice version` alias |
| `ui.py` | 152 | UNCHANGED |
| `_compat.py` | 35 | UNCHANGED |

---

## 5. Tests

### 5.1 New tests

**`tests/unit/proxy/test_health_routes_registered.py`**:

```python
def test_health_routes_in_app():
    from lattice.proxy.server import create_app
    from lattice.core.config import LatticeConfig
    app = create_app(LatticeConfig.auto())
    paths = {route.path for route in app.routes if hasattr(route, "path")}
    assert "/healthz" in paths
    assert "/readyz" in paths
    assert "/startupz" in paths
    assert "/metrics" in paths
    assert "/stats" in paths
```

**`tests/unit/proxy/test_response_headers.py`** — uses `fastapi.testclient.TestClient`:

```python
from fastapi.testclient import TestClient
from lattice.proxy.server import create_app
from lattice.core.config import LatticeConfig

def test_headers_emitted_by_middleware():
    app = create_app(LatticeConfig.auto())
    client = TestClient(app)
    # Send a minimal valid chat request (mock the upstream provider with respx in conftest)
    r = client.post("/v1/chat/completions", json={
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert r.status_code in (200, 401)   # 401 if no API key, still has headers
    # Even on 401, the middleware should have emitted whatever context was populated
    # (e.g. provider, transforms_applied are set before auth)
    assert "x-lattice-provider" in r.headers
```

**`tests/unit/sdk/test_deprecation_shim.py`**:

```python
import warnings
def test_sdk_client_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import importlib
        if "lattice.sdk.client" in sys.modules:
            del sys.modules["lattice.sdk.client"]
        importlib.import_module("lattice.sdk.client")
        deprecations = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert deprecations, "expected DeprecationWarning from lattice.sdk.client"
```

**`tests/unit/cli/test_subcommands_reachable.py`**:

```python
import subprocess
import pytest

@pytest.mark.parametrize("cmd", [
    ["lattice", "--version"],
    ["lattice", "version"],          # alias
    ["lattice", "--help"],
    ["lattice", "proxy", "--help"],
    ["lattice", "proxy", "run", "--help"],
    ["lattice", "proxy", "start", "--help"],
    ["lattice", "proxy", "stop", "--help"],
    ["lattice", "proxy", "restart", "--help"],
    ["lattice", "proxy", "status", "--help"],
    ["lattice", "init", "--help"],
    ["lattice", "lace", "--help"],
    ["lattice", "unlace", "--help"],
    ["lattice", "info", "--help"],
    ["lattice", "config", "--help"],
    ["lattice", "benchmark", "--help"],
    ["lattice", "health", "--help"],
    ["lattice", "status", "--help"],
    ["lattice", "doctor", "--help"],
])
def test_subcommand_help(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"{cmd}: stderr={result.stderr}"
```

### 5.2 Contract tests already cover

`tests/contract/test_cli_contract.py` (Phase 0) — exhaustive CLI test, including `lattice proxy start/stop/status` lifecycle.

`tests/contract/test_http_contract.py` (Phase 0) — every `/v1/*` and `/lattice/*` endpoint. Phase 6 extends this to also assert `/healthz`, `/readyz`, `/startupz`, `/metrics`, `/stats` return their documented shape:

```python
def test_healthz_shape():
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("healthy", "ok")

def test_metrics_prometheus_format():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    # Prometheus exposition: lines start with HELP/TYPE or metric names
    for line in r.text.splitlines():
        assert line == "" or line.startswith("#") or line[0].isalpha()

def test_stats_json_shape():
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    # documented top-level keys:
    for key in ("version", "transforms", "sessions", "providers", "tacc", "maintenance"):
        assert key in body
```

`tests/contract/test_headers_contract.py` — verifies the six `x-lattice-*` headers emit on a successful chat completion. Phase 6 makes sure they emit *via the middleware*, not per-handler.

`tests/contract/test_python_api_contract.py` — already verifies:

```python
from lattice import (
    LatticeClient, LatticeProxyClient, CompressResult,
    wrap_openai_client, __version__,
    PromptIR, PromptIRV2, build_ir, serialize_ir_to_text,
)
from lattice.sdk import LatticeClient, LatticeProxyClient, wrap_openai_client
# Deprecated but still works (with warning):
import lattice.sdk.client
```

---

## 6. Symbol migration table

| Old | New |
|---|---|
| `from lattice.sdk.client import LatticeClient` | DEPRECATED — use `from lattice import LatticeClient` |
| `from lattice.sdk.client import CompressResult` | DEPRECATED — use `from lattice import CompressResult` |
| `from lattice.proxy.compat_exports import *` | **DELETED** — no replacement |
| `response.headers["x-lattice-*"] = value` (set in handlers) | `context.set("...", value)` (middleware emits) |

---

## 7. Acceptance criteria

- [x] `src/lattice/proxy/compat_exports.py` does not exist (committed deletion).
- [x] `src/lattice/proxy/middleware.py` exists.
- [x] `app.routes` includes `/healthz`, `/readyz`, `/startupz`, `/metrics`, `/stats`.
- [x] `tests/unit/proxy/test_health_routes_registered.py` passes.
- [x] `tests/unit/proxy/test_response_headers.py` passes.
- [x] `tests/contract/test_http_contract.py` covers all 5 health/metrics endpoints.
- [x] `from lattice import LatticeClient, LatticeProxyClient, wrap_openai_client, CompressResult, __version__` works.
- [x] `from lattice.sdk import LatticeClient, LatticeProxyClient, wrap_openai_client` works (same names).
- [x] `import lattice.sdk.client` emits a DeprecationWarning.
- [x] `lattice version` (no dashes) prints the version string identically to `lattice --version`.
- [x] `gateway/compat.py` and `gateway/server.py` contain no `response.headers["x-lattice-...]` assignments; lattice response headers are stashed on `request.state` and emitted by `LatticeHeaderMiddleware`.
- [x] `rg 'x-lattice-' src/lattice/proxy/middleware.py` returns 6 matches (canonical `_HEADER_KEYS`).
- [x] `uv run ruff check src/ tests/` clean.
- [x] `uv run mypy src/lattice/` clean.
- [x] `uv run pytest tests/ -q` passes.
- [x] `uv run pytest tests/contract/ -q` passes.
- [ ] `python scripts/compare_benchmarks.py benchmarks/results/phase-0-baseline.json benchmarks/results/phase-7-proxy.json --tolerance-pct 2` exits 0 (operator; `OLLAMA_CLOUD_API_KEY`).

---

## 8. Risks specific to this phase

| Risk | Mitigation |
|---|---|
| Middleware reads `request.state.transform_context` but some handlers (e.g. `/v1/models`) don't attach one | Middleware guards with `if ctx is None: return response`. The headers simply aren't emitted on those routes — acceptable. |
| Header values containing commas or special chars break Prometheus or downstream parsers | The middleware serialises with explicit type handling (booleans, floats, lists). Strings pass through. If a comma appears in `transforms_applied` (which is itself comma-separated), it's by design. Verify with `tests/contract/test_headers_contract.py`. |
| `HealthManager.readyz()` previously raised on uninitialised pipeline; now wrapped in a route that swallows the exception | `readyz()` returns `(body, status)` tuple where status is 503 on not-ready. Route handler uses the status code. Test asserts 503 when pipeline isn't warm yet. |
| Removing `sdk/client.py` body breaks an existing user's import path on a `pip install --upgrade` | The shim emits DeprecationWarning but still re-exports the symbols. The CHANGELOG and MIGRATION.md must call this out clearly. |
| Some integration tests bypass FastAPI and call gateway handlers directly; they don't go through middleware and won't see the new headers | This is a test-only concern. Update those tests to assert on `context.get_summary()` directly instead of response headers. |
| Adding middleware adds per-request overhead | The middleware does ~10 dict lookups and string conversions per response. Benchmark impact <0.5 ms — within tolerance. |

---

## 9. Rollback plan

```bash
git revert <phase-6-merge-commit>
```

Phase 6 is small enough to be a single PR. Reverting restores: per-handler header setting, no health endpoints (back to Phase 0 state for those), and the unused `proxy/health.py` file. The contract test for health endpoints would then fail, surfacing the regression immediately.

---

## 10. PR shape

```
refactor(proxy,sdk,cli): wire health routes; consolidate headers; tidy SDK surface [Phase 6]

- Wire HealthManager into bootstrap; register /healthz /readyz /startupz /metrics /stats routes
- New proxy/middleware.py: LatticeHeaderMiddleware emits all x-lattice-* response headers from TransformContext
- Remove per-handler header setting in gateway/compat.py and gateway/server.py
- Top-level: `from lattice import LatticeClient, LatticeProxyClient, wrap_openai_client, CompressResult`
- sdk/__init__.py: explicit re-exports (same names)
- sdk/client.py: replaced with DeprecationWarning shim; deletion scheduled for v1.1
- CLI: add `lattice version` alias for `--version`
- Delete proxy/compat_exports.py (already gone in working tree)

Net: -1 file (compat_exports), +1 file (middleware), +2 tests.
All 1600+ tests green. Contract tests cover all CLI commands + all HTTP endpoints (incl. health).
Benchmarks ±2%.
```
