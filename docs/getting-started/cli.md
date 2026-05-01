# CLI Reference

LATTICE provides a unified CLI via the `lattice` command.

## Global Options

```
lattice [command] [subcommand] [options]
lattice --version
lattice --help
```

---

## `lattice proxy`

Manage the LATTICE proxy server lifecycle.

### `lattice proxy run`

Start the proxy in the foreground (blocks until Ctrl+C).

```bash
lattice proxy run --port 8787
lattice proxy run --port 8787 --mode aggressive --no-ui
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--port PORT` | 8787 | Listen port |
| `--host HOST` | from config | Bind address |
| `--workers N` | auto | Worker processes |
| `--mode MODE` | balanced | Compression mode: `safe`, `balanced`, `aggressive` |
| `--no-ui` | off | Disable live Rich display |
| `--reload` | off | Auto-reload for development |

### `lattice proxy start`

Start the proxy in the background (daemon).

```bash
lattice proxy start --port 8787
lattice proxy start --mode safe
```

### `lattice proxy stop`

Stop the background proxy.

```bash
lattice proxy stop
lattice proxy stop --grace 5
lattice proxy stop --force
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--grace N` | 10 | Seconds for graceful shutdown |
| `--force` | off | Kill immediately (SIGKILL) |

### `lattice proxy restart`

Stop then start the background proxy.

```bash
lattice proxy restart
lattice proxy restart --port 9999
```

### `lattice proxy status`

Show PID, uptime, and health of the background proxy.

```bash
lattice proxy status
```

---

## `lattice init`

One-step setup: detect installed agents and configure them for LATTICE.

```bash
lattice init                           # Auto-detect all agents
lattice init claude codex              # Configure specific agents
lattice init --start-proxy             # Auto-detect + start proxy
lattice init --port 9999               # Custom proxy port
lattice init --local                   # Local scope only
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--port PORT` | 8787 | Proxy port to configure |
| `--start-proxy` | off | Start proxy after init |
| `--local` | — | Local scope only |
| `--global` | on | Global scope |

Supported agents: `claude`, `codex`, `opencode`, `cursor`, `copilot`

---

## `lattice lace`

Route an agent through the LATTICE proxy. Starts the proxy if needed, configures the agent's environment, launches the agent, and cleans up on exit.

```bash
lattice lace claude
lattice lace codex --model o4-mini
lattice lace --port 9999 claude
lattice lace --no-start claude        # Proxy already running
lattice lace --no-patch claude        # Don't modify agent config
lattice lace --no-tunnel claude       # Skip sidecar tunnel
lattice lace --dry-run claude         # Preview only
```

Options:
| Flag | Default | Description |
|------|---------|-------------|
| `--port PORT` | 8787 | Proxy port |
| `--no-start` | — | Assume proxy already running |
| `--no-patch` | — | Use env vars only, don't modify config |
| `--no-tunnel` | — | Direct connection, no sidecar |
| `--dry-run` | — | Preview without executing |

---

## `lattice unlace`

Restore an agent's original configuration (reverses `lattice lace` or `lattice init`).

```bash
lattice unlace claude
lattice unlace codex
lattice unlace opencode
```

---

## `lattice status`

Show proxy health, detected agents, and routing readiness.

```bash
lattice status
```

Output shows:
- Proxy status (running / not running)
- Version and compression mode
- Mutated agents (via `lattice init`)
- Detected supported agents

---

## `lattice info`

Show version, enabled transforms, session store configuration.

```bash
lattice info
```

---

## `lattice config`

Display resolved configuration from environment and config files.

```bash
lattice config
lattice config --json               # JSON output
```

---

## `lattice doctor`

Diagnose proxy connectivity and agent routing issues.

```bash
lattice doctor
lattice doctor claude
```

Checks:
1. Is the agent laced (env/config patched)?
2. Are required env vars set?
3. Is the proxy running and healthy?
4. Can a test request reach the proxy?

---

## `lattice health`

Check if the proxy is reachable and healthy.

```bash
lattice health
lattice health --port 8787
lattice health --host 127.0.0.1 --port 9999
```

---

## `lattice benchmark`

Run the benchmark suite.

```bash
lattice benchmark
```

This delegates to the production eval CLI:

```bash
uv run python benchmarks/evals/cli.py --suite all \
  --providers ollama-cloud \
  --provider-model ollama-cloud=kimi-k2.6:cloud
```

See [Benchmarks](benchmarks.md) for full options.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LATTICE_PROVIDER_BASE_URL` | — | Default upstream provider URL |
| `LATTICE_PROVIDER_BASE_URLS` | — | JSON dict `{"provider": "url"}` |
| `OPENAI_API_KEY` | — | Used for OpenAI/Azure/compatible providers |
| `ANTHROPIC_API_KEY` | — | Anthropic provider |
| `LATTICE_PROXY_PORT` | 8787 | Proxy listen port |
| `LATTICE_SESSION_TTL_SECONDS` | 3600 | Session expiry |
| `LATTICE_SESSION_STORE` | memory | `memory` or `redis` |
| `LATTICE_REDIS_URL` | — | Redis connection URL |
| `LATTICE_GRACEFUL_DEGRADATION` | true | Continue on transform failure |
| `LATTICE_COMPRESSION_TIMEOUT_MS` | 100 | Max pipeline time |
| `LATTICE_MAX_TRANSFORM_EXPANSION_RATIO` | 1.5 | Abort threshold |
