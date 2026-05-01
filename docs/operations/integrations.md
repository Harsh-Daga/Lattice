# Integrations

LATTICE seamlessly integrates with coding agents via its proxy. Two approaches: one-time setup (`lattice init`) or per-session routing (`lattice lace`).

## Supported Agents

| Agent | Provider | Installation |
|-------|----------|-------------|
| **Claude Code** | Anthropic | `npm install -g @anthropic-ai/claude-code` |
| **OpenAI Codex** | OpenAI | `npm install -g @openai/codex` |
| **Cursor** | Various | `brew install cursor` |
| **OpenCode** | Various | `npm install -g @opencode-ai/cli` |
| **GitHub Copilot** | OpenAI | `npm install -g @github/copilot-cli` |
| **Generic** | Any | Any OpenAI-compatible client |

## One-Time Setup: `lattice init`

Detects installed agents and permanently configures them to route through LATTICE:

```bash
# Auto-detect all agents and configure
lattice init

# Configure specific agents only
lattice init claude codex

# Auto-detect + start the proxy
lattice init --start-proxy
```

What it does:
1. Detects supported agents on your PATH
2. Modifies agent configuration files to add `OPENAI_BASE_URL=http://localhost:8787/v1`
3. Stores mutation records for later reversal
4. Optionally starts the proxy

## Per-Session Routing: `lattice lace`

Routes a single agent session through LATTICE without permanent changes:

```bash
lattice lace claude
lattice lace codex --model o4-mini
lattice lace opencode
```

What it does:
1. Starts the proxy (or reuses existing)
2. Sets environment variables for the agent
3. Launches the agent
4. Cleans up on agent exit

### Options

```bash
lattice lace --port 9999 claude        # Custom port
lattice lace --no-start claude         # Proxy already running
lattice lace --no-patch claude         # Env vars only
lattice lace --no-tunnel claude        # Direct connection
lattice lace --dry-run claude          # Preview only
```

## Restoring Configuration: `lattice unlace`

```bash
lattice unlace claude
lattice unlace codex
lattice unlace --all
```

## Checking Status

```bash
# See which agents are configured
lattice status

# Diagnose routing issues
lattice doctor claude
```

## Environment Variables

Agents respect standard OpenAI environment variables:

```bash
export OPENAI_BASE_URL=http://localhost:8787/v1
export OPENAI_API_KEY=sk-your-key
```

LATTICE adds transport metadata headers automatically.

## Manual Configuration

For tools that don't support `lattice init`:

```bash
# Add to your shell profile or agent config
export OPENAI_BASE_URL=http://localhost:8787/v1

# For Anthropic-specific tools
export ANTHROPIC_BASE_URL=http://localhost:8787/v1
```
