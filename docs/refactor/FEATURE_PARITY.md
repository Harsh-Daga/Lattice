# LATTICE v0.x → v1.0.0 Feature Parity Checklist

> **Phase 11.** Every user-visible v0.x feature must have a passing test link below.
> Authoritative spec: `docs/refactor/10-tests.md` §4.5.

| # | Feature (v0.x claim) | v1.0.0 evidence | Test |
|---|----------------------|-----------------|------|
| 1 | CLI: `lattice proxy run --port 8787` starts a foreground proxy | unchanged | `tests/contract/test_cli_contract.py`, `tests/contract/test_full_cli_matrix.py` |
| 2 | CLI: `lattice proxy start/stop/restart/status` daemon lifecycle | unchanged | `tests/contract/test_full_cli_matrix.py`, `tests/unit/cli/test_subcommands_reachable.py` |
| 3 | CLI: `lattice init <agent>` patches durable config | unchanged | `tests/unit/integrations/test_agents.py` |
| 4 | CLI: `lattice lace <agent>` transient routing + sidecar tunnel | unchanged | `tests/unit/integrations/test_lace_transient.py`, `tests/unit/integrations/test_tunnel.py` |
| 5 | CLI: `lattice unlace <agent>` restores original config | unchanged | `tests/unit/integrations/test_mutation_store.py` |
| 6 | CLI: `lattice info` / `config` / `status` / `doctor` / `health` | unchanged | `tests/contract/test_full_cli_matrix.py`, `tests/unit/integrations/test_doctor_covers_all_agents.py` |
| 7 | HTTP: `POST /v1/chat/completions` OpenAI format, streaming + non-streaming | unchanged shape | `tests/contract/test_http_contract.py`, `tests/contract/test_full_http_matrix.py` |
| 8 | HTTP: `POST /v1/messages` Anthropic format | unchanged | `tests/contract/test_full_http_matrix.py` |
| 9 | HTTP: `GET /v1/models` | unchanged | `tests/contract/test_full_http_matrix.py` |
| 10 | HTTP: `POST/GET/DELETE /v1/responses`, `WS /v1/responses` | route table locked | `tests/contract/test_http_contract.py::test_every_endpoint_in_route_table` |
| 11 | HTTP: native LATTICE protocol on `/lattice/gateway`, session endpoints | unchanged | `tests/integration/test_proxy_sessions.py` |
| 12 | HTTP: `/healthz` `/readyz` `/startupz` `/metrics` `/stats` | wired + reachable | `tests/unit/proxy/test_health_routes_registered.py`, `tests/contract/test_full_http_matrix.py` |
| 13 | Response headers: six `x-lattice-*` contract headers | middleware emits all six (incl. `x-lattice-cost-usd` default `0.000`) | `tests/contract/test_headers_full.py`, `tests/unit/proxy/test_response_headers.py` |
| 14 | Python API: `from lattice import LatticeClient` | unchanged | `tests/unit/sdk/test_sdk.py`, `tests/contract/test_python_api_full.py` |
| 15 | Python API: `from lattice import LatticeProxyClient` | unchanged | `tests/contract/test_python_api_full.py` |
| 16 | Python API: `wrap_openai_client(...)` | unchanged | `tests/contract/test_python_api_contract.py` |
| 17 | Transform: content_profiler (priority 1, default) | package split; same behavior | `tests/unit/transforms/content_profiler/` |
| 18 | Transform: runtime_contract | unchanged | `tests/unit/transforms/test_transform_registry.py` |
| 19 | Transform: cache_arbitrage | IR-native `optimize()` | `tests/unit/transforms/test_cache_arbitrage.py` |
| 20 | Transform: prefix_optimizer | **deleted** — config no-op | `tests/unit/transforms/test_no_prefix_opt.py` |
| 21 | Transform: message_dedup | unchanged | `tests/unit/transforms/test_message_dedup_ir.py` |
| 22 | Transform: reference_sub | unchanged | `tests/unit/transforms/test_transform_registry.py` |
| 23 | Transform: rate_distortion | unchanged | `tests/unit/transforms/test_rate_distortion_ir.py` |
| 24 | Transform: path_prefix | unchanged | `tests/unit/transforms/test_transform_registry.py` |
| 25 | Transform: format_conversion | `format_converter/` package | `tests/unit/transforms/format_converter/` |
| 26 | Transform: tool_projection | unchanged | `tests/unit/transforms/test_tool_output_compiler.py` |
| 27 | Transform: tool_filter | unchanged | `tests/unit/transforms/test_tool_filter_ir_native.py` |
| 28 | Transform: output_cleanup | response-side only | `tests/unit/pipeline/test_response_side_dispatch.py` |
| 29 | Transform: columnar_pack | unchanged | `tests/unit/transforms/test_transform_registry.py` |
| 30 | Transform: json_shape | unchanged | `tests/unit/transforms/test_transforms_e2e.py` |
| 31 | Transform: extractive_compress | unchanged | `tests/unit/transforms/test_new_transforms.py` |
| 32 | Transform: diagnostic_rle | unchanged | `tests/unit/transforms/test_transform_registry.py` |
| 33 | Transform: causal_chain | unchanged | `tests/unit/ir/test_production_quality.py` |
| 34 | Transform: constraint_lifting | **deleted** | `tests/unit/transforms/test_no_prefix_opt.py::test_constraint_lifting_module_removed` |
| 35 | Transform: context_selector | submodular path kept | `tests/unit/transforms/test_context_selector.py` |
| 36 | Transform: strategy_selector | **deleted** (Phase 5 gate) | `tests/unit/transforms/test_legacy_only.py`, `docs/refactor/phase-5-decisions.md` |
| 37 | Execution transform: batching | unchanged | `tests/integration/test_end_to_end.py::test_batching_single_request` |
| 38 | Execution transform: speculative | unchanged | `tests/integration/test_end_to_end.py::test_speculative_miss` |
| 39 | Execution transform: delta_encode | config flag fixed | `tests/unit/transforms/test_delta_encode_config_flag.py` |
| 40 | TACC: token-aware congestion control | unchanged | `tests/unit/providers/transport/test_stall_detector.py::test_tacc_uses_stall_signal` |
| 41 | Binary framing protocol | unchanged | `tests/unit/protocol/test_binary_framing.py` |
| 42 | Delta encoding: turn-1+ CAS | unchanged | `tests/unit/transport/test_delta_wire.py`, `tests/integration/test_session_correctness.py` |
| 43 | Stream stall detection | unchanged | `tests/unit/providers/transport/test_stall_detector.py` |
| 44 | HMAC resume tokens | unchanged | `tests/unit/protocol/test_resume.py` |
| 45 | Semantic cache: in-memory + Redis | `lattice.cache.SemanticCache` | `tests/unit/cache/test_semantic_cache.py`, `tests/integration/test_redis_store_integration.py` |
| 46 | MILV: multi-input loss validation | unchanged | `tests/unit/pipeline/test_flow_e2e.py` |
| 47 | Provider: openai | unchanged | `tests/unit/providers/test_providers.py` |
| 48 | Provider: anthropic | unchanged | `tests/unit/providers/test_anthropic_v2.py` |
| 49 | Provider: azure | unchanged | `tests/unit/providers/test_providers.py` |
| 50 | Provider: bedrock | unchanged | `tests/unit/providers/test_providers.py` |
| 51 | Provider: gemini + vertex | unchanged | `tests/unit/providers/test_providers.py` |
| 52 | Provider: ollama + ollama-cloud | unchanged | `tests/unit/providers/test_providers.py` |
| 53 | Providers: openai_compatible family | unchanged | `tests/unit/providers/test_providers.py` |
| 54 | MCP integration with Anthropic | unchanged | `tests/unit/providers/test_mcp_to_anthropic.py` |
| 55 | Agent integration: Claude Code | doctor + lace | `tests/unit/integrations/test_agents.py`, `tests/e2e/test_agent_wrappers.py::test_claude_wrap_dry_run` |
| 56 | Agent integration: Codex | unchanged | `tests/e2e/test_agent_wrappers.py::test_codex_wrap_dry_run` |
| 57 | Agent integration: Cursor | `AgentNotInstalledError` on missing config | `tests/unit/integrations/test_jsonfile_raises_when_config_missing.py` |
| 58 | Agent integration: OpenCode | unchanged | `tests/e2e/test_agent_wrappers.py::test_opencode_wrap_dry_run` |
| 59 | Agent integration: GitHub Copilot | doctor coverage | `tests/unit/integrations/test_doctor_covers_all_agents.py` |
| 60 | Compression modes: safe / balanced / aggressive | `--mode` + config | `tests/unit/core/test_config.py`, `tests/unit/core/test_compression_mode.py` |
| 61 | Config: `LatticeConfig` + `lattice.yaml` + env | unchanged | `tests/unit/core/test_config.py` |

**Also covered (layout mirror):** `tests/unit/utils/test_token_count.py` for `lattice.utils.token_count`.

**Status:** Phase 11 — all 61 rows linked. Run `uv run pytest tests/ -q` to verify.
