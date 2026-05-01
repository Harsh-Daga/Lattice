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
| [Architecture](concepts/architecture.md) | System design, data flow, thread safety |
| [Proxy Server](concepts/proxy.md) | Endpoints, headers, config, /stats schema |
| [SDK](concepts/sdk.md) | LatticeClient API |
| [Observability](concepts/observability.md) | /stats, /metrics, headers, telemetry |
| [Safety](concepts/safety.md) | Risk scoring, transform buckets, gating |

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
| [All 18 Transforms](compression/transforms.md) | Priority-ordered pipeline |
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
