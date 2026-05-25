# LATTICE Documentation

## Getting Started

| Document | Description |
|----------|-------------|
| [Quick Start](getting-started/quickstart.md) | Get running in 5 minutes |
| [Installation](getting-started/installation.md) | pip, source, requirements |
| [CLI Reference](getting-started/cli.md) | Every `lattice` command |

## Core Concepts

| Document | Description |
|----------|-------------|
| [Runtime Architecture](architecture/runtime.md) | Module boundaries, lifecycles, request flow |
| [Proxy & agents](operations/integrations.md) | Lace, init, tunnel, five agents |
| [SDK](concepts/sdk.md) | LatticeClient API |
| [Observability](concepts/observability.md) | /stats, /metrics, headers, telemetry |
| [Safety](concepts/safety.md) | Risk scoring, transform buckets, gating |
| [SIG · RATS · PSG · MILV](concepts/sig-rats-psg-milv.md) | The safety architecture — four cooperating subsystems |

## Novel Transport Technology

| Document | Description |
|----------|-------------|
| [TACC Congestion Control](novel/tacc.md) | Token-aware AIMD controller |
| [Binary Framing](novel/binary-framing.md) | 15B headers, 17 frame types, CRC32 |
| [Delta Encoding](novel/delta-encoding.md) | 95% wire savings, CAS concurrency |
| [Streaming](novel/streaming.md) | Stall detection, resume, multiplex |
| [Batching & Speculation](novel/batching-speculation.md) | Request grouping, pre-execution |

## Compression & Caching

| Document | Description |
|----------|-------------|
| [All 20 Transforms](compression/transforms.md) | Registry-ordered pipeline |
| [Caching](compression/caching.md) | Semantic cache + KV-cache alignment |
| [Protocol](compression/protocol.md) | Manifests, delta, multiplex |

## Providers

| Document | Description |
|----------|-------------|
| [17 Supported Providers](providers/providers.md) | Adapters, pooling, streaming |

## Evaluation

| Document | Description |
|----------|-------------|
| [Benchmarks](evaluation/benchmarks.md) | Three-layer eval suite |

## Operations

| Document | Description |
|----------|-------------|
| [Agent Integrations](operations/integrations.md) | lace, unlace, init |

## Contributors & release

| Document | Description |
|----------|-------------|
| [Refactor & forward plan index](refactor/README.md) | Phases 0–12 shipped + v2.0 plan (12–27) |
| [Migration guide](refactor/MIGRATION.md) | v0.x → v1.0.0 import map |
| [Feature parity](refactor/FEATURE_PARITY.md) | 61-row shipped-feature checklist |
| [CHANGELOG](../CHANGELOG.md) | Release notes |

Historical phase specs (`docs/refactor/00-*.md` … `11-docs-release.md`) stay for audit traceability; user-facing narrative is **README**, **AGENTS.md**, and the sections above.
