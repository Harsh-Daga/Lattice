"""Generate docs/refactor/api-surface.json — machine-readable public surface.

Captures the public contract from REFACTOR_PLAN.md §2:
  - §2.1 CLI commands
  - §2.2 HTTP endpoints
  - §2.3 Response headers
  - §2.4 Public Python symbols

The contents below mirror the master plan text. This file is the canonical
input for tests/contract/* and for Phase 6 / Phase 11 surface checks. When
the plan evolves, update the constants here and regenerate the JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "docs" / "refactor" / "api-surface.json"

# ---------------------------------------------------------------------------
# §2.1 CLI commands
# ---------------------------------------------------------------------------
CLI_COMMANDS = [
    {
        "name": "proxy",
        "subcommands": ["run", "start", "stop", "restart", "status"],
        "flags_per_subcommand": {
            "run":     ["--host", "--port", "--workers", "--mode", "--reload", "--no-ui"],
            "start":   ["--host", "--port", "--workers", "--mode"],
            "stop":    ["--grace", "--force"],
            "restart": [],
            "status":  [],
        },
    },
    {"name": "init",    "subcommands": [], "flags": ["--port", "--local", "--global", "--start-proxy"]},
    {"name": "lace",    "subcommands": [], "flags": ["--port", "--no-start", "--no-patch", "--no-tunnel", "--dry-run", "--"]},
    {"name": "unlace",  "subcommands": [], "flags": []},
    {"name": "info",    "subcommands": [], "flags": []},
    {"name": "config",  "subcommands": [], "flags": ["--json"]},
    {"name": "health",  "subcommands": [], "flags": ["--host", "--port"]},
    {"name": "status",  "subcommands": [], "flags": []},
    {"name": "doctor",  "subcommands": [], "flags": [],
     "supported_agent_args": ["claude", "codex", "cursor", "opencode", "copilot", "generic"]},
    {"name": "benchmark", "subcommands": [], "flags": []},
]
CLI_GLOBAL_FLAGS = ["--help", "-h", "-v", "--version"]
SUPPORTED_AGENTS = ["claude", "codex", "cursor", "opencode", "copilot", "generic"]

# ---------------------------------------------------------------------------
# §2.2 HTTP endpoints
# ---------------------------------------------------------------------------
HTTP_ENDPOINTS = [
    {"method": "POST",   "path": "/v1/chat/completions", "streaming_supported": True,  "format": "openai"},
    {"method": "POST",   "path": "/v1/messages",         "streaming_supported": True,  "format": "anthropic"},
    {"method": "GET",    "path": "/v1/models",           "streaming_supported": False, "format": "openai"},
    {"method": "POST",   "path": "/v1/responses",        "streaming_supported": True,  "format": "openai_responses"},
    {"method": "GET",    "path": "/v1/responses/{id}",   "streaming_supported": False, "format": "openai_responses"},
    {"method": "DELETE", "path": "/v1/responses/{id}",   "streaming_supported": False, "format": "openai_responses"},
    {"method": "WS",     "path": "/v1/responses",        "streaming_supported": True,  "format": "openai_responses"},
    {"method": "WS",     "path": "/v1/chat/completions", "streaming_supported": True,  "format": "openai_codex"},
    {"method": "POST",   "path": "/lattice/session/start",      "streaming_supported": False},
    {"method": "POST",   "path": "/lattice/session/append",     "streaming_supported": False},
    {"method": "GET",    "path": "/lattice/session/{session_id}", "streaming_supported": False},
    {"method": "POST",   "path": "/lattice/session/invalidate", "streaming_supported": False},
    {"method": "POST",   "path": "/lattice/gateway",            "streaming_supported": False, "format": "native"},
    {"method": "GET",    "path": "/healthz",   "streaming_supported": False, "kind": "health"},
    {"method": "GET",    "path": "/readyz",    "streaming_supported": False, "kind": "health"},
    {"method": "GET",    "path": "/startupz",  "streaming_supported": False, "kind": "health"},
    {"method": "GET",    "path": "/metrics",   "streaming_supported": False, "kind": "prometheus"},
    {"method": "GET",    "path": "/stats",     "streaming_supported": False, "kind": "snapshot"},
    # Codex aliases (§2.2 trailing paragraph).
    {"method": "POST",   "path": "/v1/codex/responses",   "streaming_supported": True, "format": "openai_responses_alias"},
    {"method": "POST",   "path": "/backend-api/responses", "streaming_supported": True, "format": "openai_responses_alias"},
]

# ---------------------------------------------------------------------------
# §2.3 Response headers — the contract subset.
# Implementation today emits additional informational headers; only the ones
# listed in REFACTOR_PLAN.md §2.3 are part of the public contract.
# ---------------------------------------------------------------------------
HEADERS_REQUIRED = [
    "x-lattice-compression",
    "x-lattice-session-id",
    "x-lattice-delta",
    "x-lattice-cost-usd",
    "x-lattice-provider",
    "x-lattice-transforms-applied",
]
HEADERS_PASSTHROUGH_PREFIX = ["x-ratelimit-"]

# ---------------------------------------------------------------------------
# §2.4 Public Python API
# ---------------------------------------------------------------------------
# Top-level lattice exports promised at v1.0.0. Today only __version__ is
# at the top level; LatticeClient et al. live under lattice.client and
# lattice.sdk. Phase 6 hoists these to the top level.
PYTHON_API_TOPLEVEL_TARGET = [
    "LatticeClient",        # phase 6 hoist (lives in lattice.client today)
    "LatticeProxyClient",   # phase 6 hoist (lives in lattice.sdk today)
    "CompressResult",       # phase 6 hoist (lives in lattice.client today)
    "wrap_openai_client",   # phase 6 hoist (called wrap_openai today; plan rename TBD)
    "__version__",          # already at top level
]

# What works today (used by contract tests to assert "working surface today").
PYTHON_API_TODAY = {
    "from_lattice":         ["__version__"],
    "from_lattice_client":  ["LatticeClient", "CompressResult"],
    "from_lattice_sdk":     ["LatticeClient", "LatticeProxyClient", "wrap_openai", "wrap_anthropic"],
    "from_lattice_core":    [
        "LatticeConfig", "TransformContext", "Result", "Ok", "Err",
        "is_ok", "is_err", "unwrap", "unwrap_err",
        "Request", "Response", "Message", "Role",
        "Transform", "SyncTransform", "ReversibleSyncTransform",
        "CompressorPipeline",
        "LatticeError", "ConfigurationError",
        "ProviderError", "ProviderTimeoutError",
        "RequestTooLargeError",
        "SessionError", "SessionExpiredError", "SessionNotFoundError", "SessionStoreError",
        "TransformError", "TransformNotFoundError", "ValidationError",
    ],
}

# Methods that LatticeClient must keep (REFACTOR_PLAN §2.4).
LATTICE_CLIENT_METHODS = [
    "compress", "compress_request", "decompress_response",
    "health", "compression_stats", "count_tokens",
]


def main() -> int:
    payload = {
        "schema_version": 1,
        "source": "docs/refactor/REFACTOR_PLAN.md §2 (public surface lock)",
        "cli": {
            "global_flags": CLI_GLOBAL_FLAGS,
            "commands": CLI_COMMANDS,
            "supported_agents": SUPPORTED_AGENTS,
        },
        "http": {
            "endpoints": HTTP_ENDPOINTS,
            "headers_required": HEADERS_REQUIRED,
            "headers_passthrough_prefix": HEADERS_PASSTHROUGH_PREFIX,
        },
        "python_api": {
            "toplevel_target": PYTHON_API_TOPLEVEL_TARGET,
            "today": PYTHON_API_TODAY,
            "lattice_client_methods": LATTICE_CLIENT_METHODS,
        },
        "notes": [
            "wrap_openai_client (plan §2.4) is named wrap_openai today; Phase 6/11 reconciles the name.",
            "Top-level LatticeClient/LatticeProxyClient/CompressResult imports are added in Phase 6.",
            "Many additional x-lattice-* headers are emitted today (cost/runtime/cached-tokens/etc.); only the §2.3 set is contract-required.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)}")
    print(f"  CLI commands: {len(CLI_COMMANDS)}, HTTP endpoints: {len(HTTP_ENDPOINTS)}")
    print(f"  Required headers: {len(HEADERS_REQUIRED)}")
    print(f"  Python API today (lattice.core): {len(PYTHON_API_TODAY['from_lattice_core'])} symbols")
    return 0


if __name__ == "__main__":
    sys.exit(main())
