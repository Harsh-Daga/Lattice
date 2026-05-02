# <h1 align="center">LATTICE</h1>

<p align="center">
  <strong>LLM Transport & Efficiency Layer</strong><br>
  <em>Make every LLM call cheaper, faster, and safe — without changing your model.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/lattice-transport/"><img src="https://img.shields.io/pypi/v/lattice-transport" alt="PyPI"></a>
  <a href="https://github.com/Harsh-Daga/lattice/actions"><img src="https://img.shields.io/github/actions/workflow/status/Harsh-Daga/lattice/ci.yml?branch=main&label=CI" alt="CI"></a>
  <a href="https://github.com/Harsh-Daga/lattice/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://github.com/Harsh-Daga/lattice"><img src="https://img.shields.io/badge/tests-1584%20passed-brightgreen" alt="Tests"></a>
  <a href="#"><img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue" alt="Python"></a>
</p>

---

**LATTICE** is an intelligent transport proxy that sits between your application and any LLM provider. It applies network-layer optimizations — congestion control, binary framing, delta encoding, speculation, batching — plus a safety-gated compression pipeline with 18 transforms. Your app sends standard OpenAI API requests; LATTICE makes them smaller, faster, safer, and cache-friendly.

**It is not a router.** LATTICE never changes your model, never falls back between providers, never guesses. You route to exactly one provider per request. LATTICE optimizes the transport and execution.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Novel Technology](#novel-technology)
  - [TACC — Token-Aware Congestion Control](#tacc)
  - [Binary Framing Protocol](#binary-framing)
  - [Delta Encoding](#delta-encoding)
  - [Stream Architecture](#stream-architecture)
  - [Request Batching](#request-batching)
  - [Speculative Execution](#speculative-execution)
- [Compression Pipeline](#compression-pipeline)
- [Safety](#safety)
- [Observability](#observability)
- [Supported Providers](#supported-providers)
- [CLI Reference](#cli-reference)
- [Agent Integration](#agent-integration)
- [Development](#development)
- [Documentation](#documentation)
- [License](#license)

---

## Installation

```bash
pip install lattice-transport
```

Optional dependencies:

```bash
pip install "lattice-transport[redis]"   # Multi-process session store
pip install "lattice-transport[mcp]"    # MCP tool support
pip install "lattice-transport[all]"     # Everything
```

Requirements: Python 3.10+. No external services needed for single-process mode.

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
# Or use the SDK
from lattice import LatticeClient

client = LatticeClient()
response = client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Explain transport protocols"}],
)
print(response.choices[0].message.content)
```

Every request is automatically compressed, cached, and optimized. Zero code changes in proxy mode.

---

## Architecture

```
                          ┌──────────────────────────┐
                          │   Application / Agent     │
                          │  (Claude, Cursor, Codex,  │
                          │   OpenAI SDK, curl)       │
                          └────────────┬─────────────┘
                                       │ OpenAI API format
                          ┌────────────▼─────────────┐
                          │   LATTICE PROXY :8787     │
                          │                           │
         ┌────────────────┼───────────────────────┐   │
         │                │                       │   │
         ▼                ▼                       ▼   │
┌─────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Session   │  │    Transform    │  │    Semantic     │
│   Manager   │  │    Pipeline     │  │     Cache       │
│             │  │                 │  │                 │
│ Memory or   │  │ 18 transforms   │  │ Exact-hash      │
│ Redis store │  │ priority-ordered│  │ + approximate   │
│             │  │ risk-gated       │  │ semantic match  │
│ CAS version │  │ expansion-capped│  │ LRU + TTL       │
└──────┬──────┘  └────────┬────────┘  └────────┬────────┘
       │                  │                    │
       └──────────────────┼────────────────────┘
                          │
              ┌───────────▼──────────────────┐
              │      DirectHTTPProvider      │
              ├──────────────────────────────┤
              │  ProviderRegistry (17 adapt)  │
              │  ConnectionPool (HTTP/2)      │
              │  StreamStallDetector          │
              │  TACC Congestion Controller   │
              └───────────┬──────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │     LLM Provider      │
              │  (exactly one per req)│
              └───────────────────────┘
```

### Request Flow

```
1. Client sends OpenAI-compatible POST /v1/chat/completions
2. SessionManager creates or retrieves session (with CAS versioning)
3. 18 transforms run in priority order, each gated by:
   config → policy → runtime budget → risk gate → expansion guard
4. SemanticCache checks exact hash, then approximate fingerprint
5. [cache miss] Provider adapter serializes → HTTP/2 pool → provider
6. [streaming] StallDetector monitors per-provider tolerance windows
7. TACC controller manages concurrency window (token-based, not request-count)
8. Response deserialized → pipeline reverse pass → OpenAI JSON → client
9. Session updated, response cached, headers attached
```

---

## Novel Technology

LATTICE adapts classical systems techniques for LLM workloads. These are not LLM features — they are transport, network, and execution innovations.

### TACC

**Token-Aware Congestion Control** — AIMD-style adaptive concurrency. Manages per-provider admission using token pressure (not request counts: a 100K-token request uses more provider capacity than a 10-token one). Priority-ordered waiting queue, stall-aware window collapse, cache-aware latency smoothing. → [Deep Dive](docs/novel/tacc.md)

### Binary Framing

15-byte fixed header format. 17 semantic frame types (PING, REQUEST, STREAM_CHUNK, RESUME_TOKEN...). CRC32 per-frame integrity. Semantic boundary flags for sentence/tool/reasoning boundaries. O(1) parsing — no JSON overhead per chunk. → [Deep Dive](docs/novel/binary-framing.md)

### Delta Encoding

After turn 1, sends only new messages — server reconstructs full context from session store. CAS-style optimistic concurrency via anchor versioning prevents lost updates. Graceful fallback on version/sequence mismatch. → [Deep Dive](docs/novel/delta-encoding.md)

### Stream Architecture

Per-provider dynamic stall detection with phase-aware tolerance multipliers (first_chunk=1.5×, streaming=1.0×, thinking=2.0×, tool_call=1.2×). Token velocity tracking catches trickle-stalls. Multi-stream multiplex (QUIC-inspired) with independent lifecycle per stream. HMAC-signed resume tokens with circular replay windows. → [Deep Dive](docs/novel/streaming.md)

### Request Batching

Groups independent requests sharing model/temperature/tools into single provider calls. 30-60% per-request overhead reduction from shared prompts. Streaming requests excluded. Compatibility-keyed grouping. → [Deep Dive](docs/novel/batching-speculation.md)

### Speculative Execution

Sidecar prediction of next-turn content. Rule-based (zero-cost). Runs in parallel with real request — discard if wrong, instant if right. Never blocks the main request. Confidence threshold ≥0.7. → [Deep Dive](docs/novel/batching-speculation.md)

---

## Compression Pipeline

18 transforms in priority order. Every transform is safety-classified and risk-gated.

| P | Transform | Safety | What It Does |
|---|-----------|--------|--------------|
| 1 | **content_profiler** | SAFE | Classifies content type, computes 0-100 risk score |
| 2 | **runtime_contract** | SAFE | Enforces transform time budget per-complexity tier |
| 9 | **cache_arbitrage** | SAFE | Reorders for KV-cache alignment, sets provider hints |
| 10 | **prefix_optimizer** | SAFE | Deduplicates common message prefixes |
| 15 | **message_dedup** | CONDITIONAL | Removes exact/near-duplicate turns |
| 20 | **reference_sub** | CONDITIONAL | UUIDs, URLs, paths → `<ref_N>` short references |
| 22 | **rate_distortion** | CONDITIONAL | Extractive text compression of long-form content |
| 24 | **grammar_compress** | CONDITIONAL | Grammar-based structured data compression |
| 25 | **dictionary_compress** | CONDITIONAL | Learned phrase dictionary (HPACK-style) |
| 25 | **format_conversion** | CONDITIONAL | Markdown tables, JSON → compact CSV/TSV |
| 30 | **tool_filter** | SAFE | Strips internal fields from tool output |
| 40 | **output_cleanup** | SAFE | Normalizes whitespace, trims boilerplate |

**Execution transforms** (outside main pipeline): batching, speculative execution, delta encoding, auto-continuation.

→ [Full Transform Reference](docs/compression/transforms.md)

---

## Safety

Every transform is classified into one of three buckets. A 0-100 semantic risk score (8 dimensions) gates CONDITIONAL and DANGEROUS transforms.

```
Risk Score     SAFE       CONDITIONAL    DANGEROUS     Expansion Guard
─────────      ────       ───────────    ──────────    ───────────────
LOW (0-20)     ✓          ✓              ✓             tokens × 1.5 max
MEDIUM (20-40)  ✓          ✓              ✗             tokens × 1.5 max
HIGH (40-60)   ✓          ✗              ✗             tokens × 1.5 max
CRITICAL (>60) ✓          ✗              ✗             tokens × 1.5 max
```

→ [Safety Deep Dive](docs/concepts/safety.md)

---

## Observability

Every request returns routing metadata. Full runtime state in `/stats`.

```bash
curl http://localhost:8787/stats | jq
```

Key surfaces:
- **/stats** — Full JSON: transforms, sessions, pools, TACC state, maintenance, downgrades, ignored chunks
- **/metrics** — Prometheus format: counters, gauges, latency histograms per provider
- **Response headers** — `x-lattice-compression`, `x-lattice-session-id`, `x-lattice-delta`, `x-lattice-cost-usd`
- **Maintenance** — Background cleanup every 60s (stale streams, cache expiry), visible in /stats/maintenance

→ [Observability Guide](docs/concepts/observability.md)

---

## Supported Providers

17 direct adapters. No routing — one provider per request.

| Provider | Prefix | HTTP/2 | Cache | Streaming |
|----------|--------|--------|-------|-----------|
| OpenAI | `openai/` | ✅ | AUTO_PREFIX | SSE delta |
| Anthropic | `anthropic/`, `claude-` | ✅ | EXPLICIT_BREAKPOINT | SSE |
| Groq | `groq/` | ✅ | — | SSE |
| DeepSeek | `deepseek/` | ✅ | — | SSE |
| Mistral | `mistral/` | ✅ | — | SSE |
| Cohere | `cohere/` | ✅ | — | SSE |
| Gemini | `gemini/`, `google/` | ✅ | EXPLICIT_CONTEXT | SSE |
| Vertex AI | `vertex/` | ✅ | EXPLICIT_CONTEXT | SSE |
| Azure | `azure/` | ✅ | AUTO_PREFIX | SSE |
| Bedrock | `bedrock/` | ✅ | EXPLICIT_BREAKPOINT | SSE |
| Ollama | `ollama/` | — | — | SSE |
| Ollama Cloud | `ollama-cloud/` | ✅ | — | SSE |
| OpenRouter | `openrouter/` | ✅ | — | SSE |
| Fireworks | `fireworks/` | ✅ | — | SSE |
| Together | `together/` | ✅ | — | SSE |
| Perplexity | `perplexity/` | ✅ | — | SSE |
| AI21 | `ai21/` | ✅ | — | SSE |

→ [Provider Details](docs/providers/providers.md)

---

## CLI Reference

```bash
lattice proxy run --port 8787          # Start foreground
lattice proxy start --port 8787        # Start daemon
lattice proxy stop                     # Graceful shutdown
lattice proxy status                   # PID, uptime, health

lattice init                           # Auto-detect + configure agents
lattice lace claude                    # Route agent through proxy
lattice unlace claude                  # Restore original config

lattice info                           # Version, transforms, config
lattice status                         # Proxy + agent health
lattice health                         # Connectivity check
lattice doctor                         # Diagnose routing issues
lattice config                         # Resolved configuration
```

→ [Full CLI Reference](docs/getting-started/cli.md)

---

## Agent Integration

Route coding agents through LATTICE with a single command:

```bash
lattice lace claude       # Claude Code
lattice lace codex        # OpenAI Codex  
lattice lace cursor       # Cursor
lattice lace opencode     # OpenCode
lattice lace copilot      # GitHub Copilot
```

`lattice lace` starts the proxy, configures the agent's environment, launches the agent, and cleans up on exit. No permanent changes.

For permanent configuration: `lattice init` patches agent config files. `lattice unlace` reverses.

→ [Integration Guide](docs/operations/integrations.md)

---

## Development

```bash
git clone https://github.com/Harsh-Daga/lattice
cd lattice
uv sync          # Install all deps
uv run pytest    # 1584 tests, 7 skipped

# Lint + typecheck
uv run ruff check src/
uv run mypy src/lattice/

# Run benchmarks
uv run python benchmarks/evals/cli.py --suite all \
  --providers ollama-cloud \
  --provider-model ollama-cloud=kimi-k2.6:cloud \
  --iterations 1 --warmup 0 --provider-warmup 0
```

---

## Documentation

| Section | Documents |
|---------|-----------|
| **Getting Started** | [Quick Start](docs/getting-started/quickstart.md) · [Installation](docs/getting-started/installation.md) · [CLI](docs/getting-started/cli.md) |
| **Concepts** | [Architecture](docs/concepts/architecture.md) · [Proxy](docs/concepts/proxy.md) · [SDK](docs/concepts/sdk.md) · [Observability](docs/concepts/observability.md) · [Safety](docs/concepts/safety.md) |
| **Novel Tech** | [TACC](docs/novel/tacc.md) · [Binary Framing](docs/novel/binary-framing.md) · [Delta Encoding](docs/novel/delta-encoding.md) · [Streaming](docs/novel/streaming.md) · [Batching & Speculation](docs/novel/batching-speculation.md) |
| **Compression** | [Transforms](docs/compression/transforms.md) · [Caching](docs/compression/caching.md) · [Protocol](docs/compression/protocol.md) |
| **Providers** | [17 Providers](docs/providers/providers.md) |
| **Evaluation** | [Benchmarks](docs/evaluation/benchmarks.md) |
| **Operations** | [Agent Integrations](docs/operations/integrations.md) |

→ [Full Documentation Index](docs/index.md)

---

## License

MIT © Harsh Daga

[GitHub](https://github.com/Harsh-Daga/lattice) · [Issues](https://github.com/Harsh-Daga/lattice/issues) · [PyPI](https://pypi.org/project/lattice-transport/) · [Changelog](https://github.com/Harsh-Daga/lattice/releases) · [Contributing](CONTRIBUTING.md)
