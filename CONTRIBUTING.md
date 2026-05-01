# Contributing to LATTICE

## Setup

```bash
git clone https://github.com/Harsh-Daga/lattice
cd lattice
uv sync
```

## Development

```bash
# Run all tests
uv run pytest tests/ -q

# Lint
uv run ruff check src/

# Typecheck
uv run mypy src/lattice/

# Format
uv run ruff format src/
```

## Adding a Transform

1. Extend `ReversibleSyncTransform` in `src/lattice/core/pipeline.py`
2. Implement `process()` and `reverse()` methods
3. Register in `src/lattice/core/pipeline_factory.py`
4. Add safety bucket entry in `src/lattice/utils/validation.py`

## Running Benchmarks

```bash
# Local only
uv run python benchmarks/evals/cli.py --suite feature

# Live provider
uv run python benchmarks/evals/cli.py --suite all \
  --providers ollama-cloud \
  --provider-model ollama-cloud=kimi-k2.6:cloud
```

## Releasing

1. Update version in `src/lattice/_version.py`
2. Commit and push
3. Create a GitHub Release with tag `vX.Y.Z`
4. CI builds and publishes to PyPI via Trusted Publishing
