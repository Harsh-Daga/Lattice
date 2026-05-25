# Scripts

Ad-hoc and CI helper scripts. Canonical benchmark entrypoint: `scripts/run_canonical_benchmark.sh` (see `docs/refactor/09-benchmarks.md`).

| Script | Purpose |
|--------|---------|
| `run_canonical_benchmark.sh` | CI gate / local canonical suite (`--suite all`, 1 iter) |
| `compare_benchmarks.py` | Compare two `benchmarks/results/*.json` phase artifacts |
| `benchmark_compression.py` | Ad-hoc compression profiling during development |
| `benchmark_e2e_through_proxy.py` | E2E through running proxy |
| `profile_format_conv.py` | Dev-only: cProfile `FormatConverter` (not CI) |
| `test_e2e_real.py` | Dev-only: local proxy + Ollama smoke (not CI) |
