# <h1 align="center">LATTICE</h1>

<p align="center">
  <strong>LLM Transport & Efficiency Layer</strong><br>
  <em>Make every LLM call cheaper, faster, and safe — without changing your model.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/lattice-transport/"><img src="https://img.shields.io/pypi/v/lattice-transport" alt="PyPI"></a>
  <a href="https://github.com/Harsh-Daga/lattice/actions"><img src="https://img.shields.io/github/actions/workflow/status/Harsh-Daga/lattice/ci.yml?branch=main&label=CI" alt="CI"></a>
  <a href="https://github.com/Harsh-Daga/lattice/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://github.com/Harsh-Daga/lattice"><img src="https://img.shields.io/badge/tests-2039%20collected%20%7C%201824%20passed-brightgreen" alt="Tests"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python"></a>
</p>

---

**LATTICE** sits between your application and any LLM provider. It compresses prompts, caches responses, manages concurrency (TACC), supports a native binary protocol, and routes coding agents through one self-hosted proxy. Your app sends standard OpenAI-format requests; LATTICE makes them smaller, faster, and cache-friendlier.

**It is not a router.** LATTICE never changes your model, never falls back between providers, never guesses. One provider per request. LATTICE optimises transport and execution.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Novel Technology](#novel-technology)
- [Compression Pipeline](#compression-pipeline)
- [Safety](#safety)
- [Observability](#observability)
- [Supported Providers](#supported-providers)
- [CLI Reference](#cli-reference)
- [Agent Integration](#agent-integration)
- [Development](#development)
- [Migrating from v0.x](#migrating-from-v0x)
- [Documentation](#documentation)
- [License](#license)

---

## Installation

```bash
pip install lattice-transport
```

Optional extras:

```bash
pip install "lattice-transport[redis]"   # Multi-process session store
pip install "lattice-transport[mcp]"     # MCP tool support
pip install "lattice-transport[all]"     # Everything
```

Requirements: Python 3.10+. No external services required for single-process mode.

## Quick Start

```bash
# Start the proxy
lattice proxy run --port 8787

# Point any OpenAI SDK at it
export OPENAI_BASE_URL=http://localhost:8787/v1

# Or route an agent through it
lattice lace claude
```

```python
from lattice import LatticeClient

client = LatticeClient()
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Explain transport protocols"}],
)
print(response.choices[0].message.content)
```

Every request is automatically compressed, cached, and optimized in proxy mode — zero application code changes.

---

## Architecture

```
Application
   │ OpenAI / Anthropic API format
   ▼
LATTICE Proxy :8787
   │
   ├── state/     Session, segments, SemanticCache
   ├── planner/   RequestClassifier → UnifiedPlanner → ExecutionPlan
   ├── pipeline/  Pipeline.compress() — IR-native transforms + gates
   ├── telemetry/ Metrics, downgrade, cost, agent stats
   └── providers/ adapters/ (17) + transport/ (HTTP pool, TACC, streaming)
            │
            ▼
       LLM Provider (exactly one per request)
```

### Request flow

1. Client sends `POST /v1/chat/completions` (or Anthropic `/v1/messages`).
2. `SessionManager` creates or loads a session (CAS versioning).
3. `content_profiler` builds a semantic profile; `UnifiedPlanner` produces an `ExecutionPlan`.
4. `Pipeline.compress()` runs transforms with policy, budget, risk, and MILV gates.
5. `SemanticCache` checks exact hash, then approximate fingerprint.
6. On miss, the provider adapter sends via HTTP/2 pool; TACC manages admission.
7. Response reverse pass + `x-lattice-*` headers via `LatticeHeaderMiddleware`.

→ [Runtime architecture](docs/architecture/runtime.md)

---

## Novel Technology

LATTICE applies classical systems techniques to LLM workloads — transport and execution, not model features.

| Capability | Summary | Deep dive |
|------------|---------|-----------|
| **TACC** | Token-aware AIMD congestion control | [docs/novel/tacc.md](docs/novel/tacc.md) |
| **Binary framing** | 15-byte headers, 17 frame types, CRC32 | [docs/novel/binary-framing.md](docs/novel/binary-framing.md) |
| **Delta encoding** | Turn 2+ sends deltas only; CAS sessions | [docs/novel/delta-encoding.md](docs/novel/delta-encoding.md) |
| **Streaming** | Stall detection, resume tokens, multiplex | [docs/novel/streaming.md](docs/novel/streaming.md) |
| **Batching** | Groups compatible requests | [docs/novel/batching-speculation.md](docs/novel/batching-speculation.md) |
| **Speculation** | Rule-based next-turn precompute | [docs/novel/batching-speculation.md](docs/novel/batching-speculation.md) |

Batching overhead reduction and long-conversation dedup savings are measured in the canonical benchmark suite — see [Claim traceability](benchmarks/results/CLAIMS.md) [^batch] [^dedup].

---

## Compression Pipeline

LATTICE ships **20 transforms** (`list_transform_names()` in `lattice.transforms.registry`). Six run in the default pipeline; three are execution-only (batching, speculative, delta); the rest are planner-selected or off by default. Every transform is safety-classified and risk-gated.

| P | Transform | Safety | What it does | Default |
|---|-----------|--------|--------------|:-------:|
| 1 | content_profiler | SAFE | Classifies content, computes semantic risk score | yes |
| 2 | runtime_contract | SAFE | Per-transform budget and timeout | yes |
| 2 | speculative | SAFE | Speculative token generation | exec |
| 3 | batching | SAFE | Request batching for multi-turn workloads | exec |
| 5 | delta_encoder | SAFE | Session-based delta encoding | exec |
| 9 | cache_arbitrage | SAFE | KV-cache alignment reorder | yes |
| 9 | causal_chain | SAFE | Causal chain extraction | no |
| 15 | message_dedup | CONDITIONAL | Exact/near-duplicate turn removal | no |
| 17 | diagnostic_rle | SAFE | Diagnostic repetition RLE | no |
| 18 | context_selector | SAFE | Submodular context selection | no |
| 19 | columnar_pack | SAFE | Columnar table packing | no |
| 20 | reference_sub | CONDITIONAL | UUID/URL/hash → short refs | yes |
| 21 | json_shape | SAFE | JSON shape factoring | no |
| 22 | extractive_compress | SAFE | Extractive compression | no |
| 22 | rate_distortion | CONDITIONAL | Rate-distortion semantic compression | no |
| 23 | path_prefix | SAFE | Filesystem path prefix compression | no |
| 25 | format_conversion | CONDITIONAL | Table/JSON format conversion | no |
| 29 | tool_projection | SAFE | Query-aware tool field projection | no |
| 30 | tool_filter | SAFE | Tool output filtering | yes |
| 40 | output_cleanup | SAFE | Response-side whitespace/JSON cleanup | yes |

**exec** = execution-only transform (outside default pipeline list).

Headline compression on the canonical feature suite (ollama-cloud / kimi-k2.6:cloud): **40.3%** average reduction [^compress]. Pipeline latency ~**36 ms** [^latency].

→ [Transform reference](docs/compression/transforms.md) · [Claim traceability](benchmarks/results/CLAIMS.md) · [Feature parity checklist](docs/refactor/FEATURE_PARITY.md) (61 rows)

---

## Safety

Transforms are classified **SAFE**, **CONDITIONAL**, or **DANGEROUS**. A 0–100 semantic risk score gates lossy transforms; expansion guards cap token growth.

→ [Safety guide](docs/concepts/safety.md) · [SIG · RATS · PSG · MILV](docs/concepts/sig-rats-psg-milv.md)

---

## Observability

```bash
curl http://localhost:8787/stats | jq
curl http://localhost:8787/metrics
```

- **/stats** — transforms, sessions, pools, TACC, maintenance, downgrades
- **/metrics** — Prometheus counters and histograms
- **Response headers** — `x-lattice-compression`, `x-lattice-session-id`, `x-lattice-delta`, `x-lattice-cost-usd`, `x-lattice-provider`, `x-lattice-transforms-applied`

→ [Observability](docs/concepts/observability.md)

---

## Supported Providers

**17** direct adapters. No routing — one provider per request.

| Provider | Prefix | HTTP/2 | Streaming |
|----------|--------|--------|-----------|
| OpenAI | `openai/` | yes | SSE |
| Anthropic | `anthropic/`, `claude-` | yes | SSE |
| Azure | `azure/` | yes | SSE |
| Bedrock | `bedrock/` | yes | SSE |
| Gemini | `gemini/`, `google/` | yes | SSE |
| Vertex AI | `vertex/` | yes | SSE |
| Groq | `groq/` | yes | SSE |
| DeepSeek | `deepseek/` | yes | SSE |
| Mistral | `mistral/` | yes | SSE |
| Cohere | `cohere/` | yes | SSE |
| Ollama | `ollama/` | — | SSE |
| Ollama Cloud | `ollama-cloud/` | yes | SSE |
| OpenRouter | `openrouter/` | yes | SSE |
| Fireworks | `fireworks/` | yes | SSE |
| Together | `together/` | yes | SSE |
| Perplexity | `perplexity/` | yes | SSE |
| AI21 | `ai21/` | yes | SSE |

→ [Provider details](docs/providers/providers.md)

---

## CLI Reference

```bash
lattice proxy run --port 8787
lattice proxy start|stop|restart|status
lattice init
lattice lace|unlace <agent>
lattice info|config|status|health|doctor
lattice benchmark --suite feature   # wraps benchmarks/evals/cli.py
```

→ [CLI reference](docs/getting-started/cli.md)

---

## Agent Integration

```bash
lattice lace claude    # Claude Code
lattice lace codex     # OpenAI Codex
lattice lace cursor    # Cursor
lattice lace opencode  # OpenCode
lattice lace copilot   # GitHub Copilot
```

`lattice doctor` (no args) checks all five agents. `lattice init` applies durable config; `lattice lace` uses transient routing + tunnel sidecar.

→ [Integrations](docs/operations/integrations.md)

---

## Development

```bash
git clone https://github.com/Harsh-Daga/lattice
cd lattice
uv sync
uv run pytest tests/ -q              # 2039 collected, 1824 passed (215 skipped)
uv run pytest tests/contract/ -q
uv run ruff check src/ tests/ benchmarks/
uv run ruff format --check src/ tests/ benchmarks/
uv run mypy src/lattice/

uv run python benchmarks/evals/cli.py --suite all \
  --providers ollama-cloud \
  --provider-model ollama-cloud=kimi-k2.6:cloud \
  --iterations 3 --warmup 1
```

→ [AGENTS.md](AGENTS.md) for AI agent contributors

---

## Migrating from v0.x

Internal Python imports changed in 1.0.0. CLI and HTTP are stable.

→ [Migration guide](docs/refactor/MIGRATION.md) · [CHANGELOG](CHANGELOG.md)

---

## Documentation

| Section | Documents |
|---------|-----------|
| **Getting Started** | [Quick Start](docs/getting-started/quickstart.md) · [Installation](docs/getting-started/installation.md) · [CLI](docs/getting-started/cli.md) |
| **Architecture** | [Runtime](docs/architecture/runtime.md) · [Safety](docs/concepts/safety.md) · [SDK](docs/concepts/sdk.md) |
| **Novel Tech** | [TACC](docs/novel/tacc.md) · [Binary framing](docs/novel/binary-framing.md) · [Delta](docs/novel/delta-encoding.md) · [Streaming](docs/novel/streaming.md) |
| **Compression** | [Transforms](docs/compression/transforms.md) · [Caching](docs/compression/caching.md) |
| **Providers** | [17 providers](docs/providers/providers.md) |
| **Operations** | [Agent integrations](docs/operations/integrations.md) |

→ [Full index](docs/index.md)

---

## Claim footnotes

[^compress]: `benchmarks/results/CLAIMS.md` — feature suite `avg_reduction_ratio` from `v1.0.0.json`
[^latency]: `benchmarks/results/CLAIMS.md` — `avg_pipeline_latency_ms` from `v1.0.0.json`
[^batch]: `benchmarks/results/CLAIMS.md` — batching overhead row
[^dedup]: `benchmarks/results/CLAIMS.md` — message_dedup row

---

## License

MIT © Harsh Daga

[GitHub](https://github.com/Harsh-Daga/lattice) · [Issues](https://github.com/Harsh-Daga/lattice/issues) · [PyPI](https://pypi.org/project/lattice-transport/) · [Changelog](CHANGELOG.md)
