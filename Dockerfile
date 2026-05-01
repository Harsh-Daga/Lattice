# =============================================================================
# LATTICE Proxy — Multi-stage Docker Build
# =============================================================================
# Production-grade container for the LATTICE LLM Transport & Efficiency Layer.
#
# Build:
#   docker build -t lattice-proxy:latest .
# Run:
#   docker run -p 8787:8787 -e LATTICE_PROVIDER_API_KEY=sk-... lattice-proxy:latest
#
# Stages:
#   1. builder — compile deps, run typecheck, install in virtualenv
#   2. runtime — minimal image with only runtime artifacts
# =============================================================================

# ------------------------------------------------------------------
# Stage 1: Builder
# ------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build deps (for packages with compiled extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy source and build
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --upgrade pip \
    && pip install build \
    && python -m build --wheel --outdir /build/dist

# Install into a temporary prefix for extraction
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install /build/dist/*.whl

# ------------------------------------------------------------------
# Stage 2: Runtime
# ------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="LATTICE Proxy"
LABEL org.opencontainers.image.description="LLM Transport & Efficiency Layer"
LABEL org.opencontainers.image.source="https://github.com/lattice-ai/lattice"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r lattice && useradd -r -g lattice lattice

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Health check — proxy readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/health')" || exit 1

# Expose proxy port + metrics port
EXPOSE 8787
EXPOSE 9090

USER lattice

# Default: run the proxy server
CMD ["lattice-proxy"]
