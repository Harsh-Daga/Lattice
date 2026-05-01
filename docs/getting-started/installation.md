# Installation

## pip (Recommended)

```bash
# Core installation
pip install lattice-transport

# With Redis support (multi-process deployments)
pip install "lattice-transport[redis]"

# With MCP support (agent frameworks)
pip install "lattice-transport[mcp]"

# Everything
pip install "lattice-transport[all]"
```

## From Source

```bash
git clone https://github.com/Harsh-Daga/lattice
cd lattice
pip install -e .
```

## Development Install

```bash
git clone https://github.com/Harsh-Daga/lattice
cd lattice
uv sync
```

## Verify Installation

```bash
lattice --version
```

## Environment Setup

The proxy needs provider API keys to forward requests:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GROQ_API_KEY="gsk_..."
```

Or set the generic forwarding key:

```bash
export LATTICE_PROVIDER_API_KEY="sk-..."
export LATTICE_PROVIDER_BASE_URL="https://api.openai.com"
```

## Requirements

- Python 3.10+
- No external services needed (in-memory mode)
- Redis optional for multi-process deployments
