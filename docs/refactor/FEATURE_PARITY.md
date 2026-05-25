# LATTICE v0.x → v1.0.0 Feature Parity Checklist (Phase 0 scaffold)

> **Purpose.** Single index linking user-visible features to automated proof tests.
> Phase 11 expands this to the full 61-row matrix in `10-tests.md` §4.5.
> **Rule:** Each row must reference a test that exists and passes on `main` / the phase branch.

| # | Feature | Evidence (v1.0.0) | Test(s) |
|---|---------|-------------------|---------|
| 1 | CLI help / version | Unchanged surface | `tests/contract/test_cli_contract.py` |
| 2 | HTTP chat completions contract | OpenAI-shaped API | `tests/contract/test_http_contract.py` |
| 3 | Response headers contract | Lattice headers enumerated | `tests/contract/test_headers_contract.py` |
| 4 | Python public API imports | `lattice`, `lattice.core`, `lattice.ir` | `tests/contract/test_python_api_contract.py` |
| 5 | Canonical transport types | `lattice.transport.types` | `tests/unit/transport/test_transport_types_canonical_path.py` |
| 6 | Pipeline compress entry | `Pipeline.compress()` + gates | `tests/unit/test_pipeline_compress.py` |
| 7 | IR package | `lattice.ir` | `tests/unit/ir/test_builder_stores_metadata.py`, `tests/unit/test_ir.py` |
| 8 | Unified planner | Single scheduler | `tests/unit/test_sig_rats_psg.py` (planner integration) |
| 9 | Transform registry complete | Every spec resolves | `tests/unit/transforms/test_registry_complete.py` |
| 10 | content_profiler split | Package + metadata | `tests/unit/transforms/content_profiler/` |
| 11 | prefix_opt removed | No module / registry entry | `tests/unit/transforms/test_no_prefix_opt.py` |
| 12 | delta_encode config flag | `transform_delta_encode` | `tests/unit/transforms/test_delta_encode_config_flag.py` |
| 13 | Response-side dispatch | `output_cleanup` only on reverse path | `tests/unit/pipeline/test_response_side_dispatch.py` |
| 14 | IR-native transforms | No dual `process()` | `tests/unit/pipeline/test_ir_native_no_process.py` |
| 15 | Proxy integration | End-to-end proxy | `tests/integration/test_proxy.py` |
| 16 | Provider HTTP transport | DirectHTTPProvider | `tests/unit/providers/test_providers.py` |
| 17 | Compression modes (config) | safe / balanced / aggressive | `tests/unit/test_config.py` |
| 18 | strategy_selector deleted | Benchmark gate default cut | `docs/refactor/phase-5-decisions.md` |

**Status:** Scaffold satisfies Phase 0 `00-audit-baseline.md` §2. Full row coverage is Phase 11 (`10-tests.md`).
