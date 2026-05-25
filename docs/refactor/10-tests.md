# Phase 11 — Test Reshape & Feature-Parity Matrix (STATUS Phase 11; file `10-tests.md`)

> **Goal.** Restructure `tests/unit/` so its layout exactly mirrors `src/lattice/`. Every src module has a corresponding tests subdirectory; every public class has at least one test file. Produce a `docs/refactor/FEATURE_PARITY.md` checklist proving every v0.x user-visible feature still works at v1.0.0. Add the missing contract tests for HTTP endpoints, headers, and Python API surface. Verify the actual `pytest --collect-only` count matches what `README.md` claims. Add `pytest-xdist` for parallel test runs to keep CI under 5 minutes despite the suite growing.
>
> **Outcome.** Test discovery is grep-able: looking at `tests/unit/transforms/format_converter/` reveals every test for the format converter. Feature parity is provable by running `FEATURE_PARITY.md`'s checklist. CI completes in <5 min.
>
> **Estimated effort.** 2 days.

---

## 1. Why this phase exists

The audit found:

1. **`tests/unit/` is flat** — 80+ test files at the root, plus only `core/` and `providers/` subdirectories. Finding "tests for the pipeline" requires `grep -l Pipeline tests/unit/*.py`. With the new flat domain layout in `src/lattice/`, the mirror in `tests/unit/` makes discoverability automatic.
2. **Test count disagrees across documentation** — README says 1584, AGENTS says 1839+, actual `pytest --collect-only` (per Phase 0) gives a different number. Phase 10 captures the authoritative count and pins it everywhere.
3. **No `FEATURE_PARITY.md`** — there's no single document proving "every v0.x feature still works in v1.0.0". When we delete `prefix_opt`, `constraint_lifting`, `strategy_selector`, parts of `context_selector`, the proof of preservation lives only across many tests. Consolidate the checklist.
4. **Phase 0's `tests/contract/` was a skeleton.** Phase 10 ensures it's exhaustive — every CLI subcommand, every HTTP endpoint, every public Python symbol.
5. **Test runtime climbs** — at ~1600 tests today and growing through phases 1–9, sequential CI is touching 4 minutes. `pytest-xdist` with `-n auto` brings it to ~90 seconds on 8 cores.

---

## 2. Files touched

### 2.1 Created

```
docs/refactor/FEATURE_PARITY.md
tests/unit/test_test_count_pinned.py       # asserts pytest collected count == README badge
tests/contract/test_full_cli_matrix.py     # extends Phase 0 contract tests
tests/contract/test_full_http_matrix.py    # extends Phase 0 contract tests
tests/contract/test_headers_full.py        # extends Phase 0 contract tests
tests/contract/test_python_api_full.py     # extends Phase 0 contract tests
.github/workflows/test.yml                 # parallel test workflow (if not already configured for xdist)
```

### 2.2 Modified

- `pyproject.toml` — add `pytest-xdist` to dev deps.
- `tests/conftest.py` — common fixtures (httpx mocks, ephemeral proxy fixture, tmp_path-based config).
- All test files — moved into per-domain subdirectories matching `src/lattice/`.

### 2.3 Deleted

Tests for deleted code:

```
tests/unit/test_pipeline.py                 # v1 pipeline gone (Phase 2)
tests/unit/test_pipeline_v2_wrapper.py      # bridge gone (Phase 2)
tests/unit/test_scheduler.py                # gone (Phase 3)
tests/unit/test_optimizer_scheduler.py      # gone (Phase 3)
tests/unit/test_structure_optimizer.py      # gone (Phase 3 — text-based variant)
tests/unit/test_prefix_opt.py               # gone (Phase 4)
tests/unit/test_constraint_lifting.py       # gone (Phase 4)
tests/unit/test_strategy_selector.py        # gone (Phase 4, gated)
tests/unit/test_information_theoretic_selector.py    # gone (Phase 4, gated)
tests/unit/test_router.py                   # renamed to test_tier_classifier (Phase 3)
tests/unit/test_format_conv.py              # replaced by tests/unit/transforms/format_converter/*
tests/unit/test_content_profiler.py         # replaced by tests/unit/transforms/content_profiler/*
tests/unit/test_compiler.py                 # gone (Phase 1)
```

---

## 3. The target test tree

Every test file lives at the path mirroring its src target:

```
tests/
├── conftest.py
├── unit/
│   ├── core/
│   │   ├── test_config.py
│   │   ├── test_context.py
│   │   ├── test_errors.py
│   │   ├── test_result.py
│   │   └── test_segmentation.py
│   ├── safety/
│   │   └── test_risk_scoring.py
│   ├── ir/
│   │   ├── test_types.py
│   │   ├── test_primitives.py
│   │   ├── test_builder.py
│   │   ├── test_builder_stores_metadata.py        # from Phase 1
│   │   ├── test_normalizer.py
│   │   ├── test_serializer.py
│   │   ├── test_transform.py                      # IRTransform protocol + CandidateSearch
│   │   ├── test_native_optimizer.py
│   │   ├── test_validation.py
│   │   ├── test_quality.py
│   │   └── test_semantic_graph.py
│   ├── planner/
│   │   ├── test_request_classifier.py
│   │   ├── test_task_classifier.py
│   │   ├── test_execution_plan.py
│   │   ├── test_execution_builder.py
│   │   ├── test_unified_planner.py
│   │   ├── test_unified_planner_is_canonical.py   # from Phase 3
│   │   ├── test_runtime_state.py
│   │   ├── test_provider_strategy.py
│   │   ├── test_transport_planner.py
│   │   └── test_fallback_executor.py
│   ├── pipeline/
│   │   ├── test_runner.py
│   │   ├── test_factory.py
│   │   ├── test_policy.py
│   │   ├── test_guardrails.py
│   │   ├── test_milv.py
│   │   ├── test_auto_continuation.py
│   │   ├── test_batch_accumulator.py
│   │   ├── test_representation_optimizer.py
│   │   ├── test_no_legacy_process_paths.py        # from Phase 2
│   │   ├── test_transport_types_canonical_path.py # from Phase 2
│   │   └── test_response_side_dispatch.py         # from Phase 4
│   ├── transforms/
│   │   ├── test_registry.py
│   │   ├── test_registry_complete.py              # every spec resolves (Phase 4)
│   │   ├── test_reputation.py
│   │   ├── test_patterns.py
│   │   ├── test_no_prefix_opt.py                  # from Phase 4
│   │   ├── test_delta_encode_config_flag.py       # from Phase 4
│   │   ├── test_runtime_contract.py
│   │   ├── test_cache_arbitrage.py
│   │   ├── test_message_dedup.py
│   │   ├── test_reference_sub.py
│   │   ├── test_tool_filter.py
│   │   ├── test_tool_projection.py
│   │   ├── test_output_cleanup.py
│   │   ├── test_path_prefix.py
│   │   ├── test_json_shape.py
│   │   ├── test_columnar_pack.py
│   │   ├── test_extractive_compress.py
│   │   ├── test_rate_distortion.py
│   │   ├── test_diagnostic_rle.py
│   │   ├── test_causal_chain.py
│   │   ├── test_context_selector.py
│   │   ├── test_batching.py
│   │   ├── test_speculative.py
│   │   ├── test_delta_encode.py
│   │   ├── content_profiler/
│   │   │   ├── test_classifier.py
│   │   │   ├── test_risk_scorer.py
│   │   │   ├── test_planner_bridge.py
│   │   │   └── test_profiler_integration.py
│   │   ├── format_converter/
│   │   │   ├── test_format_converter.py
│   │   │   ├── test_table_converter.py
│   │   │   └── test_json_converter.py
│   │   └── optimizers/
│   │       ├── test_ir_structure_optimizer.py
│   │       ├── test_reference_optimizer.py
│   │       ├── test_tool_optimizer.py
│   │       ├── test_diagnostic_optimizer.py
│   │       ├── test_context_optimizer.py
│   │       └── test_no_text_structure_optimizer.py  # from Phase 3
│   ├── providers/
│   │   ├── test_capabilities.py
│   │   ├── test_stream_state.py
│   │   ├── test_tool_sanitizer.py
│   │   ├── test_tool_sanitizer_inheritance.py     # from Phase 5
│   │   ├── test_schema_filter.py
│   │   ├── test_mcp_to_anthropic.py
│   │   ├── test_credentials.py
│   │   ├── test_registry_complete.py              # all 17 adapters (Phase 5)
│   │   ├── transport/
│   │   │   ├── test_registry.py
│   │   │   ├── test_pool.py
│   │   │   ├── test_rate_limits.py
│   │   │   ├── test_rate_limits_ttl.py            # from Phase 5
│   │   │   ├── test_completion.py
│   │   │   ├── test_streaming.py
│   │   │   ├── test_streaming_single_path.py      # from Phase 5
│   │   │   ├── test_stall_detector.py
│   │   │   └── test_helpers.py
│   │   └── adapters/
│   │       ├── test_base.py
│   │       ├── test_openai.py
│   │       ├── test_openai_compatible.py
│   │       ├── test_anthropic.py
│   │       ├── test_azure.py
│   │       ├── test_bedrock.py
│   │       ├── test_gemini.py
│   │       └── test_ollama.py
│   ├── transport/
│   │   ├── test_types.py
│   │   ├── test_serialization.py
│   │   ├── test_congestion.py
│   │   ├── test_congestion_simulation.py
│   │   └── test_delta_wire.py
│   ├── protocol/
│   │   ├── test_framing.py
│   │   ├── test_reliability.py
│   │   ├── test_resume.py
│   │   ├── test_manifest.py
│   │   ├── test_segments.py
│   │   ├── test_content.py
│   │   ├── test_boundaries.py
│   │   ├── test_multiplex.py
│   │   ├── test_dictionary_codec.py
│   │   ├── test_dictionary_static.py
│   │   ├── test_prefix_canonicalization.py
│   │   └── test_cache_planner.py
│   ├── state/
│   │   ├── test_session.py
│   │   ├── test_store.py
│   │   └── test_segment_store.py
│   ├── cache/
│   │   └── test_semantic.py
│   ├── telemetry/
│   │   ├── test_metrics.py
│   │   ├── test_downgrade.py
│   │   ├── test_agent_stats.py
│   │   ├── test_cost_estimator.py
│   │   ├── test_maintenance.py
│   │   └── test_streaming_sketches.py
│   ├── runtime/
│   │   ├── test_tier_classifier.py
│   │   └── test_tier_classifier_naming.py         # from Phase 3
│   ├── proxy/
│   │   ├── test_bootstrap.py
│   │   ├── test_server.py
│   │   ├── test_routes.py
│   │   ├── test_lifecycle.py
│   │   ├── test_health.py
│   │   ├── test_health_routes_registered.py       # from Phase 6
│   │   ├── test_response_headers.py               # from Phase 6
│   │   └── test_middleware.py
│   ├── gateway/
│   │   ├── test_server.py
│   │   ├── test_compat.py
│   │   ├── test_routing.py
│   │   └── test_detect_helpers.py
│   ├── sdk/
│   │   ├── test_client.py                         # local LatticeClient
│   │   ├── test_proxy_client.py
│   │   ├── test_wrappers.py
│   │   └── test_deprecation_shim.py               # from Phase 6
│   ├── cli/
│   │   ├── test_main.py
│   │   ├── test_subcommands_reachable.py          # from Phase 6
│   │   ├── test_benchmark_wrapper.py              # from Phase 9
│   │   └── test_doctor_covers_all_agents.py       # from integrations phase (07-integrations.md)
│   ├── integrations/
│   │   ├── test_agents.py
│   │   ├── test_agent_protocol.py                 # from integrations phase
│   │   ├── test_init.py
│   │   ├── test_lace.py
│   │   ├── test_unlace.py
│   │   ├── test_mutation_store.py
│   │   ├── test_mutation_store_records_lace.py    # from integrations phase
│   │   ├── test_jsonfile_raises_when_config_missing.py   # from integrations phase
│   │   ├── test_tunnel.py
│   │   ├── claude/    {test_install, test_runtime}
│   │   ├── codex/     {test_install, test_runtime, test_auth, test_ws_handler}
│   │   ├── cursor/    {test_install, test_runtime}
│   │   ├── opencode/  {test_install, test_runtime}
│   │   └── copilot/   {test_install, test_runtime}
│   ├── utils/
│   │   └── test_token_count.py
│   ├── test_core_is_leaf.py                       # from Phase 8
│   ├── test_no_old_paths.py                       # from Phase 8
│   ├── test_no_lattice_evals.py                   # from Phase 9
│   └── test_test_count_pinned.py                  # Phase 10 — see §3.6
├── integration/
│   ├── test_proxy_end_to_end.py
│   ├── test_session_flow.py
│   ├── test_redis_backend.py
│   ├── test_streaming_flow.py
│   ├── test_tacc_admission.py
│   ├── test_delta_wire.py
│   ├── test_binary_protocol.py
│   ├── test_mcp_integration.py
│   ├── test_speculative_execution.py
│   ├── test_batching_pipeline.py
│   └── test_compression_end_to_end.py
├── e2e/
│   ├── test_agent_lace_claude.py
│   ├── test_agent_lace_codex.py
│   └── test_full_pipeline.py
├── contract/                                       # from Phase 0; Phase 10 extends
│   ├── test_cli_contract.py
│   ├── test_full_cli_matrix.py                     # new in Phase 10
│   ├── test_http_contract.py
│   ├── test_full_http_matrix.py                    # new
│   ├── test_headers_contract.py
│   ├── test_headers_full.py                        # new
│   ├── test_python_api_contract.py
│   └── test_python_api_full.py                     # new
└── security/
    └── test_security.py
```

---

## 4. Step-by-step

### 4.1 Capture the authoritative test count

```bash
uv run pytest tests/ --collect-only -q 2>&1 | tail -3
# Record the number, e.g.: 1634 tests collected
```

Write it into `tests/unit/test_test_count_pinned.py`:

```python
import subprocess

EXPECTED_TEST_COUNT = 1634   # update on Phase 10 capture; CI fails if this drifts

def test_test_count_matches_expected():
    """Pin the test collection count.

    If this test fails after adding tests, update EXPECTED_TEST_COUNT and the
    README + AGENTS.md test badge in the same PR.
    """
    result = subprocess.run(
        ["uv", "run", "pytest", "tests/", "--collect-only", "-q"],
        capture_output=True, text=True,
    )
    # parse "1634 tests collected" line
    last_lines = result.stdout.strip().splitlines()[-3:]
    line = next(l for l in last_lines if "tests collected" in l or "tests" in l)
    count = int(line.split()[0])
    assert count == EXPECTED_TEST_COUNT, (
        f"Test count drifted: expected {EXPECTED_TEST_COUNT}, got {count}. "
        f"Update EXPECTED_TEST_COUNT and the README badge."
    )
```

### 4.2 Mass `git mv` to mirror src/

For each file in `tests/unit/test_*.py`, move it to the subdirectory mirroring its src target. Most of the moves were anticipated in earlier phases (Phases 1–8 each include their own `git mv` block in the "Tests" section). Phase 10 audits the result:

```bash
# After all earlier phases have moved their tests, this should be empty:
ls tests/unit/test_*.py
```

If any files remain at `tests/unit/test_*.py` root (and aren't `test_test_count_pinned.py`, `test_core_is_leaf.py`, `test_no_old_paths.py`, `test_no_lattice_evals.py` — those are deliberately root-level because they're cross-cutting), move them:

```bash
# Example pattern — repeat for each remaining file
git mv tests/unit/test_some_module.py tests/unit/<domain>/test_some_module.py
```

### 4.3 Add `conftest.py` with shared fixtures

```python
# tests/conftest.py — common test infrastructure

import contextlib
import socket
import subprocess
import time
from pathlib import Path
import pytest
import httpx

@pytest.fixture
def free_port() -> int:
    """Bind a socket to port 0 and return the OS-assigned port."""
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port

@pytest.fixture
def temp_config_home(tmp_path, monkeypatch) -> Path:
    """Isolate XDG_CONFIG_HOME so integration mutations don't touch the dev's machine."""
    cfg = tmp_path / ".config"
    cfg.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg))
    return cfg

@pytest.fixture
def proxy_subprocess(free_port):
    """Spin up a real `lattice proxy run` for contract tests. Cleans up on exit."""
    proc = subprocess.Popen(
        ["lattice", "proxy", "run", "--port", str(free_port), "--no-ui"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    # Wait for readiness
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://localhost:{free_port}/healthz", timeout=0.5)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.1)
    else:
        proc.terminate()
        raise TimeoutError("proxy did not start within 10s")
    yield free_port
    proc.terminate()
    proc.wait(timeout=5)

@pytest.fixture
def httpx_mock_provider(monkeypatch):
    """Mock an upstream provider so tests don't need real API keys."""
    import respx
    with respx.mock(base_url="https://api.openai.com") as m:
        m.post("/v1/chat/completions").respond(json={
            "id": "chatcmpl-test", "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })
        yield m
```

### 4.4 Install pytest-xdist for parallel CI

Update `pyproject.toml`:

```toml
[dependency-groups]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "pytest-xdist>=3.6.0",      # NEW — parallel test execution
    "ruff>=0.6.0",
    "mypy>=1.11.0",
    "pre-commit>=3.8.0",
    "respx>=0.21.0",
    "fakeredis>=2.26.0",
]
```

Update `pyproject.toml`'s pytest config to default to xdist:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
addopts = "-v --tb=short -n auto"   # parallel by default
pythonpath = ["."]
markers = [
    "contract: contract tests (slow; spin up real proxy)",
    "integration: integration tests (require Redis or real provider)",
    "e2e: end-to-end tests (require agent binaries on PATH)",
]
```

Mark contract tests:

```python
# tests/contract/conftest.py
import pytest
pytestmark = pytest.mark.contract
```

CI workflow can then choose: `pytest -m "not contract"` for fast PR feedback, `pytest -m contract` for merge gate.

### 4.5 Write `docs/refactor/FEATURE_PARITY.md`

```markdown
# LATTICE v0.x → v1.0.0 Feature Parity Checklist

Every feature claimed in v0.x's README must still work at v1.0.0. This document
is the proof. Each row links to the test that proves the feature works.

| # | Feature (v0.x claim) | v1.0.0 evidence | Test |
|---|---|---|---|
| 1 | CLI: `lattice proxy run --port 8787` starts a foreground proxy | unchanged behavior | `tests/contract/test_cli_contract.py::test_proxy_run_help`, `tests/integration/test_proxy_end_to_end.py::test_proxy_run` |
| 2 | CLI: `lattice proxy start/stop/restart/status` daemon lifecycle | unchanged | `tests/contract/test_full_cli_matrix.py::test_proxy_lifecycle` |
| 3 | CLI: `lattice init <agent>` patches durable config | unchanged | `tests/integration/test_init_agent.py` |
| 4 | CLI: `lattice lace <agent>` transient routing + sidecar tunnel | unchanged | `tests/e2e/test_agent_lace_claude.py` |
| 5 | CLI: `lattice unlace <agent>` restores original config | unchanged | `tests/unit/integrations/test_unlace.py` |
| 6 | CLI: `lattice info` / `lattice config` / `lattice status` / `lattice doctor` / `lattice health` | unchanged | `tests/contract/test_full_cli_matrix.py` |
| 7 | HTTP: `POST /v1/chat/completions` OpenAI format, streaming + non-streaming | unchanged shape | `tests/contract/test_http_contract.py::test_chat_completions`, `tests/integration/test_streaming_flow.py` |
| 8 | HTTP: `POST /v1/messages` Anthropic format | unchanged | `tests/contract/test_http_contract.py::test_anthropic_messages` |
| 9 | HTTP: `GET /v1/models` | unchanged | `tests/contract/test_http_contract.py::test_models` |
| 10 | HTTP: `POST/GET/DELETE /v1/responses`, `WS /v1/responses` | unchanged | `tests/contract/test_full_http_matrix.py::test_responses_api` |
| 11 | HTTP: native LATTICE protocol on `/lattice/gateway`, session endpoints | unchanged | `tests/integration/test_binary_protocol.py` |
| 12 | HTTP: `/healthz /readyz /startupz /metrics /stats` | NEW WIRING (Phase 6) — already returned proper shape; now reachable | `tests/unit/proxy/test_health_routes_registered.py`, `tests/contract/test_http_contract.py::test_health_endpoints` |
| 13 | Response headers: `x-lattice-compression`, `x-lattice-session-id`, `x-lattice-delta`, `x-lattice-cost-usd`, `x-lattice-provider`, `x-lattice-transforms-applied` | NEW middleware (Phase 6); same names + values | `tests/contract/test_headers_full.py` |
| 14 | Python API: `from lattice import LatticeClient` (local compression) | unchanged | `tests/unit/sdk/test_client.py`, `tests/contract/test_python_api_full.py` |
| 15 | Python API: `from lattice import LatticeProxyClient` | unchanged | `tests/unit/sdk/test_proxy_client.py` |
| 16 | Python API: `wrap_openai_client(...)` context manager | unchanged | `tests/unit/sdk/test_wrappers.py` |
| 17 | Transform: content_profiler (priority 1, default) | refactored into package; same behavior | `tests/unit/transforms/content_profiler/*` |
| 18 | Transform: runtime_contract | unchanged | `tests/unit/transforms/test_runtime_contract.py` |
| 19 | Transform: cache_arbitrage | unchanged (legacy process() removed) | `tests/unit/transforms/test_cache_arbitrage.py` |
| 20 | Transform: prefix_optimizer | **deleted (folded into content_profiler)** — public config flag is a no-op | `tests/unit/transforms/test_no_prefix_opt.py` |
| 21 | Transform: message_dedup | unchanged | `tests/unit/transforms/test_message_dedup.py` |
| 22 | Transform: reference_sub | unchanged | `tests/unit/transforms/test_reference_sub.py` |
| 23 | Transform: rate_distortion | unchanged | `tests/unit/transforms/test_rate_distortion.py` |
| 24 | Transform: path_prefix | unchanged | `tests/unit/transforms/test_path_prefix.py` |
| 25 | Transform: format_conversion | refactored into format_converter/ package | `tests/unit/transforms/format_converter/*` |
| 26 | Transform: tool_projection | unchanged | `tests/unit/transforms/test_tool_projection.py` |
| 27 | Transform: tool_filter | unchanged | `tests/unit/transforms/test_tool_filter.py` |
| 28 | Transform: output_cleanup | unchanged (is_response_side=True flag added) | `tests/unit/transforms/test_output_cleanup.py` |
| 29 | Transform: columnar_pack | unchanged | `tests/unit/transforms/test_columnar_pack.py` |
| 30 | Transform: json_shape | unchanged | `tests/unit/transforms/test_json_shape.py` |
| 31 | Transform: extractive_compress | unchanged | `tests/unit/transforms/test_extractive_compress.py` |
| 32 | Transform: diagnostic_rle | unchanged | `tests/unit/transforms/test_diagnostic_rle.py` |
| 33 | Transform: causal_chain | unchanged | `tests/unit/transforms/test_causal_chain.py` |
| 34 | Transform: constraint_lifting | **deleted** (no production consumer) | n/a — config flag is no-op |
| 35 | Transform: context_selector | simplified to submodular-only (or split, gated by Phase 4 benchmarks) | `tests/unit/transforms/test_context_selector.py` |
| 36 | Transform: strategy_selector | **deleted unless benchmark gate kept it** (Phase 4) | n/a or `tests/unit/transforms/strategy_selector/*` |
| 37 | Execution transform: batching | unchanged | `tests/integration/test_batching_pipeline.py` |
| 38 | Execution transform: speculative | unchanged | `tests/integration/test_speculative_execution.py` |
| 39 | Execution transform: delta_encode | config_flag bug fixed | `tests/unit/transforms/test_delta_encode_config_flag.py` |
| 40 | TACC: token-aware congestion control | unchanged | `tests/integration/test_tacc_admission.py` |
| 41 | Binary framing protocol: 15-byte header, 17 frame types, CRC32 | unchanged | `tests/unit/protocol/test_framing.py` |
| 42 | Delta encoding: turn-1+ sends only new messages, CAS versioning | unchanged | `tests/integration/test_delta_wire.py` |
| 43 | Stream architecture: per-provider stall detection, phase-aware multipliers | unchanged | `tests/unit/providers/transport/test_stall_detector.py` |
| 44 | HMAC resume tokens | unchanged | `tests/unit/protocol/test_resume.py` |
| 45 | Semantic cache: exact + approximate fingerprint, in-memory + Redis | moved to `lattice.cache.SemanticCache` | `tests/unit/cache/test_semantic.py`, `tests/integration/test_redis_backend.py` |
| 46 | MILV: multi-input loss validation | unchanged | `tests/unit/pipeline/test_milv.py` |
| 47 | Provider: openai | unchanged | `tests/unit/providers/adapters/test_openai.py` |
| 48 | Provider: anthropic (with OAuth, tool sanitization, thinking) | unchanged | `tests/unit/providers/adapters/test_anthropic.py` |
| 49 | Provider: azure | unchanged | `tests/unit/providers/adapters/test_azure.py` |
| 50 | Provider: bedrock | unchanged | `tests/unit/providers/adapters/test_bedrock.py` |
| 51 | Provider: gemini + vertex | unchanged | `tests/unit/providers/adapters/test_gemini.py` |
| 52 | Provider: ollama + ollama-cloud | unchanged | `tests/unit/providers/adapters/test_ollama.py` |
| 53 | Providers: groq, together, deepseek, perplexity, mistral, fireworks, openrouter, cohere, ai21 (via openai_compatible) | unchanged | `tests/unit/providers/adapters/test_openai_compatible.py` |
| 54 | MCP integration with Anthropic | unchanged | `tests/integration/test_mcp_integration.py` |
| 55 | Agent integration: Claude Code (env-file) | unchanged + doctor() added | `tests/unit/integrations/claude/*`, `tests/e2e/test_agent_lace_claude.py` |
| 56 | Agent integration: Codex (env + TOML) | unchanged + doctor() | `tests/unit/integrations/codex/*` |
| 57 | Agent integration: Cursor (JSON, 3 providers) | unchanged + raises AgentNotInstalledError on missing config | `tests/unit/integrations/cursor/*` |
| 58 | Agent integration: OpenCode (JSON, multi-provider) | unchanged | `tests/unit/integrations/opencode/*` |
| 59 | Agent integration: GitHub Copilot | unchanged + doctor() added | `tests/unit/integrations/copilot/*` |
| 60 | Compression modes: safe / balanced / aggressive (`--mode` flag) | unchanged | `tests/unit/core/test_config.py::test_compression_modes` |
| 61 | Config: `LatticeConfig` field names + `lattice.yaml` keys + env vars | unchanged | `tests/unit/core/test_config.py` |

If any row's "test" column is empty when this checklist is finalised, that feature has no automated proof. **Phase 10 does not ship until every row has a passing test.**
```

### 4.6 Write the extended contract tests

The Phase 0 contract suite was minimal. Phase 10 makes it exhaustive:

**`tests/contract/test_full_cli_matrix.py`**:

```python
import subprocess
import pytest

ALL_CLI_SHAPES = [
    # (argv, expected_exit_code, must_be_in_stdout)
    (["lattice"], 0, "Commands"),
    (["lattice", "--help"], 0, "Commands"),
    (["lattice", "-h"], 0, "Commands"),
    (["lattice", "--version"], 0, "lattice"),
    (["lattice", "-v"], 0, "lattice"),
    (["lattice", "version"], 0, "lattice"),
    (["lattice", "proxy", "--help"], 0, "Usage"),
    (["lattice", "proxy", "run", "--help"], 0, "Start the proxy"),
    (["lattice", "proxy", "start", "--help"], 0, "background"),
    (["lattice", "proxy", "stop", "--help"], 0, "Stop the"),
    (["lattice", "proxy", "restart", "--help"], 0, "Stop"),
    (["lattice", "proxy", "status", "--help"], 0, "Show"),
    (["lattice", "init", "--help"], 0, "Detect"),
    (["lattice", "lace", "--help"], 0, "Route"),
    (["lattice", "unlace", "--help"], 0, "Restore"),
    (["lattice", "info", "--help"], 0, "Usage"),
    (["lattice", "config", "--help"], 0, "Usage"),
    (["lattice", "benchmark", "--help"], 0, "--suite"),
    (["lattice", "health", "--help"], 0, "Usage"),
    (["lattice", "status", "--help"], 0, "Usage"),
    (["lattice", "doctor", "--help"], 0, "Diagnose"),
    # error cases
    (["lattice", "nonexistent"], 1, "Unknown"),
]

@pytest.mark.parametrize("argv,code,needle", ALL_CLI_SHAPES, ids=[" ".join(a[1:]) or "lattice" for a, _, _ in ALL_CLI_SHAPES])
def test_cli_shape(argv, code, needle):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    assert result.returncode == code, f"{argv}: exit {result.returncode}\nstderr:\n{result.stderr}"
    out = result.stdout + result.stderr
    assert needle.lower() in out.lower(), f"{argv}: '{needle}' not in output"
```

**`tests/contract/test_full_http_matrix.py`** — uses the `proxy_subprocess` fixture from `conftest.py`:

```python
import httpx
import pytest

ALL_ENDPOINTS = [
    # (method, path, body, expected_status)
    ("GET", "/healthz", None, 200),
    ("GET", "/readyz", None, (200, 503)),    # 503 OK while warming
    ("GET", "/startupz", None, 200),
    ("GET", "/metrics", None, 200),
    ("GET", "/stats", None, 200),
    ("GET", "/v1/models", None, (200, 401, 403)),
    ("POST", "/v1/chat/completions", {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
    }, (200, 401, 403, 502)),    # may fail upstream without keys; CLI must still respond
    # ... add every documented endpoint
]

@pytest.mark.parametrize("method,path,body,expected", ALL_ENDPOINTS)
def test_endpoint_responds(proxy_subprocess, method, path, body, expected):
    expected_set = expected if isinstance(expected, tuple) else (expected,)
    url = f"http://localhost:{proxy_subprocess}{path}"
    with httpx.Client(timeout=10) as client:
        if method == "GET":
            r = client.get(url)
        else:
            r = client.post(url, json=body)
    assert r.status_code in expected_set, f"{method} {path}: {r.status_code}\n{r.text[:300]}"
```

**`tests/contract/test_headers_full.py`**:

```python
import httpx

def test_all_six_lattice_headers_present(proxy_subprocess, httpx_mock_provider):
    """A successful chat completion must emit all six x-lattice-* headers."""
    url = f"http://localhost:{proxy_subprocess}/v1/chat/completions"
    r = httpx.post(url, json={
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
    }, timeout=10)
    REQUIRED = {
        "x-lattice-compression", "x-lattice-session-id",
        "x-lattice-delta", "x-lattice-cost-usd",
        "x-lattice-provider", "x-lattice-transforms-applied",
    }
    seen = {h.lower() for h in r.headers}
    missing = REQUIRED - seen
    assert not missing, f"Missing headers: {missing}\nAll: {sorted(seen)}"
```

**`tests/contract/test_python_api_full.py`** — every documented symbol importable:

```python
def test_public_api_complete():
    # Top-level
    from lattice import (
        __version__,
        LatticeClient, LatticeProxyClient, CompressResult,
        wrap_openai_client,
        PromptIR, PromptIRV2, build_ir, serialize_ir_to_text,
        MetricsCollector, DowngradeCategory,
        Session, SessionManager, SegmentStore,
        SemanticCache,
        SemanticRiskScore, compute_risk_score,
    )
    # Package surfaces
    from lattice import ir, planner, pipeline, transforms, providers
    from lattice import transport, protocol, state, cache, telemetry, safety
    from lattice import proxy, gateway, sdk, integrations, cli, runtime
    # Type-level
    from lattice.core import (
        LatticeConfig, TransformContext,
        Result, Ok, Err, is_ok, is_err, unwrap, unwrap_err,
        Request, Response, Message, Role,
        Transform, SyncTransform, ReversibleSyncTransform,
        TransformError, ConfigurationError, ProviderError, SessionError,
    )

def test_no_internal_leaks():
    """v1.0.0 public API should NOT include former internal names."""
    import pytest
    with pytest.raises(ImportError):
        from lattice import CompressorPipeline    # v1, deleted
    with pytest.raises(ImportError):
        from lattice import PipelineV2Wrapper     # deleted
    with pytest.raises(ImportError):
        from lattice import decide_schedule       # deleted
    with pytest.raises(ImportError):
        from lattice import RuntimeRouter         # renamed
```

### 4.7 Verify

```bash
# Test count pinned
uv run pytest tests/ --collect-only -q | tail -3
# Should print the EXPECTED_TEST_COUNT exactly

# Full suite, parallel
uv run pytest tests/ -q

# Contract suite (slower)
uv run pytest tests/contract/ -q -m contract

# Feature parity proof
uv run pytest tests/contract/test_full_http_matrix.py tests/contract/test_full_cli_matrix.py tests/contract/test_headers_full.py tests/contract/test_python_api_full.py -q

# Confirm CI time
time uv run pytest tests/ -q
# Should be ~90 seconds on 8-core machine
```

---

## 5. Per-file disposition

| File | Action |
|---|---|
| `tests/conftest.py` | CREATE — shared fixtures |
| `tests/unit/*` | RESHAPED — entire directory structure per §3 |
| `tests/contract/*` | EXTENDED — new exhaustive matrices |
| `tests/integration/*` | UNCHANGED — already organised |
| `tests/e2e/*` | UNCHANGED |
| `tests/security/*` | UNCHANGED |
| `pyproject.toml` | MODIFIED — `pytest-xdist`, `markers`, parallel default |
| `docs/refactor/FEATURE_PARITY.md` | CREATE — single-page parity checklist |
| `README.md` test-count badge | UPDATED to authoritative number (final number set in Phase 11 release) |
| `AGENTS.md` test-count row | UPDATED similarly |

---

## 6. Acceptance criteria

- [ ] `tests/unit/` mirrors `src/lattice/` exactly — every src module has a tests subdirectory.
- [ ] No test files remain at `tests/unit/test_*.py` root except: `test_test_count_pinned.py`, `test_core_is_leaf.py`, `test_no_old_paths.py`, `test_no_lattice_evals.py` (cross-cutting).
- [ ] `tests/conftest.py` exists with at minimum: `free_port`, `temp_config_home`, `proxy_subprocess`, `httpx_mock_provider` fixtures.
- [ ] `pytest-xdist` added to dev deps; `addopts = "-v --tb=short -n auto"` in pyproject.
- [ ] `tests/unit/test_test_count_pinned.py` passes.
- [ ] `tests/contract/test_full_cli_matrix.py` passes (all 22+ CLI shapes).
- [ ] `tests/contract/test_full_http_matrix.py` passes (every documented endpoint responds).
- [ ] `tests/contract/test_headers_full.py` passes (all 6 x-lattice headers emit).
- [ ] `tests/contract/test_python_api_full.py` passes (every documented symbol importable; former internals are gone).
- [ ] `docs/refactor/FEATURE_PARITY.md` exists; every row has a passing-test link.
- [ ] `README.md` test-count badge and `AGENTS.md` "Tests passed" row match `EXPECTED_TEST_COUNT`.
- [ ] `time uv run pytest tests/ -q` completes in under 5 minutes (on dev's machine; CI may differ).
- [ ] `uv run ruff check src/ tests/` clean.
- [ ] `uv run mypy src/lattice/` clean.
- [ ] `uv run pytest tests/ -q` passes.
- [ ] `uv run pytest tests/contract/ -q` passes.

---

## 7. Risks specific to this phase

| Risk | Mitigation |
|---|---|
| `pytest-xdist` exposes flaky tests that passed sequentially due to ordering | Run `pytest -n auto --randomly-seed=42` once before merge; fix any genuine ordering bugs. The integration tests (`tests/integration/test_redis_backend.py`) likely need a fresh Redis db per worker — use `fakeredis` to isolate. |
| Contract tests with `proxy_subprocess` are slow (~3-5 s setup each) | Use a session-scoped `proxy_subprocess` fixture: one proxy serves all contract tests. Per-test cleanup uses `lattice proxy stop --grace 2`. |
| `test_test_count_pinned.py` fails on every PR that adds a test | This is by design — the PR author updates `EXPECTED_TEST_COUNT` and the README badge in the same commit. |
| `httpx_mock_provider` doesn't cover Anthropic, Bedrock, etc. | The mock is for OpenAI only (the most-tested provider). Other-provider contract tests use real keys in optional CI matrix; skipped if keys absent. |
| FEATURE_PARITY.md has 61 rows; if even one feature regresses, ship is blocked | This is the point. The doc is the gate. |
| Moving 80+ test files takes a long time; git history line-attribution suffers | Use `git mv` for every move; line history is preserved. |
| Some integration tests have hardcoded paths into old `tests/unit/` directory layout (e.g. importing from `tests.unit.test_session`) | Grep for `from tests.unit` in tests/; fix to new paths. |

---

## 8. Rollback plan

```bash
git revert <phase-10-merge-commit>
```

Restores the flat tests/ layout. No test loss. Just a discoverability regression.

---

## 9. PR shape

Two PRs:

```
test: reshape tests/unit/ to mirror src/; add conftest fixtures; pytest-xdist [Phase 10a]
- Move ~80 test files into tests/unit/<domain>/
- Create tests/conftest.py with shared fixtures
- pyproject.toml: pytest-xdist, parallel default, contract marker
- tests/unit/test_test_count_pinned.py: pin authoritative count

test: extend contract tests; FEATURE_PARITY.md; remove dead test files [Phase 10b]
- tests/contract/test_full_cli_matrix.py: all 22+ CLI shapes
- tests/contract/test_full_http_matrix.py: all endpoints respond
- tests/contract/test_headers_full.py: all 6 x-lattice-* headers
- tests/contract/test_python_api_full.py: every public symbol importable; former internals gone
- docs/refactor/FEATURE_PARITY.md: 61-row gate
- Delete tests for deleted code (prefix_opt, constraint_lifting, scheduler, etc.)

Net: tests/unit/ now grep-able by domain; CI time ~90s (down from ~4min sequential);
every v0.x feature has a passing-test link in FEATURE_PARITY.md.
```
