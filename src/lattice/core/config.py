"""LATTICE configuration.

Single source of truth for all settings. Uses Pydantic BaseSettings with
environment variable override.

Loading order (highest priority wins):
    1. Default values (lowest)
    2. YAML config files: lattice.yaml in CWD, then ~/.config/lattice.yaml
    3. Environment variables
    4. Explicit constructor arguments (highest)
"""

from __future__ import annotations

import os
import pathlib

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class _ConfigFileSources:
    """Encapsulates config file discovery."""

    @classmethod
    def _find_config_files(cls) -> list[str]:
        files: list[str] = []
        # CWD
        cwd = pathlib.Path.cwd() / "lattice.yaml"
        if cwd.is_file():
            files.append(str(cwd))
        # User config dir
        home = pathlib.Path.home()
        user_config = home / ".config" / "lattice" / "config.yaml"
        if user_config.is_file():
            files.append(str(user_config))
        return files


class LatticeConfig(BaseSettings):
    """LATTICE configuration.

    Every field has a default and can be overridden via environment
    variable (with LATTICE_ prefix) or config file.

    Attributes:
        proxy_host: Host to bind proxy server to.
        proxy_port: Port to listen on.
        proxy_workers: Number of Uvicorn workers. None -> auto-detect.
        proxy_reload: Enable auto-reload in development.

        session_ttl_seconds: Session time-to-live.
        session_store: Backend for session storage ("memory" | "redis").
        redis_url: Redis connection URL.
        max_session_size_mb: Maximum session size before eviction.

        transform_reference_sub: Enable reference substitution.
        transform_tool_filter: Enable tool output filtering.
        transform_prefix_opt: Enable prefix optimization.
        transform_output_cleanup: Enable output cleanup.

        default_context_limit: Max tokens per request context.
        max_request_size_mb: Max HTTP request body size.
        request_timeout_seconds: Provider request timeout.
        compression_timeout_ms: Abort compression if > this.
        graceful_degradation: Continue on transform failure?

        metrics_enabled: Export Prometheus metrics.
        metrics_port: Prometheus metrics port.
        log_level: Python logging level.
        log_format: "json" or "console".
    """

    model_config = SettingsConfigDict(
        env_prefix="LATTICE_",
        env_nested_delimiter="__",
        extra="ignore",  # tolerate stale keys in config files
    )

    # ------------------------------------------------------------------
    # Proxy
    # ------------------------------------------------------------------
    proxy_host: str = Field(default="0.0.0.0")
    proxy_port: int = Field(default=8787, ge=1024, le=65535)
    proxy_workers: int | None = Field(default=None)
    proxy_reload: bool = Field(default=False)

    # Provider forwarding (required for proxy operation)
    provider_base_url: str = Field(
        default="",
        description="Base URL of the upstream LLM provider. "
        "Set this explicitly or via LATTICE_PROVIDER_BASE_URL.",
    )
    provider_api_key: str | None = Field(
        default=None,
        description="API key for the upstream provider. "
        "If not set, reads from OPENAI_API_KEY env var.",
    )
    provider_base_urls: dict[str, str] = Field(
        default_factory=dict,
        description="Per-provider base URLs. Overrides provider_base_url for specific providers. "
        "Example: {'ollama': 'http://127.0.0.1:11434', 'ollama-cloud': 'https://ollama.com/api'}",
    )

    http_proxy: str | None = Field(
        default=None,
        description="HTTP/HTTPS proxy URL for agent integrations. "
        "When set, LATTICE injects HTTP_PROXY and HTTPS_PROXY into agent env files.",
    )

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------
    session_ttl_seconds: int = Field(default=3600, ge=1)
    session_store: str = Field(default="memory")
    redis_url: str | None = Field(default=None)
    max_session_size_mb: int = Field(default=10, ge=1)

    # ------------------------------------------------------------------
    # Semantic Cache
    # ------------------------------------------------------------------
    semantic_cache_enabled: bool = Field(
        default=True,
        description="Enable proxy-side response caching. "
        "Caches complete responses keyed by request checksum.",
    )
    semantic_cache_ttl_seconds: int = Field(
        default=300,
        ge=1,
        description="TTL for cached responses in seconds.",
    )
    semantic_cache_max_entries: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of cached responses before LRU eviction.",
    )
    semantic_cache_max_entry_size_kb: int = Field(
        default=512,
        ge=1,
        description="Maximum size of a single cached response in KB.",
    )
    semantic_cache_backend: str = Field(
        default="memory",
        description="Cache backend: 'memory' or 'redis'.",
    )
    semantic_cache_backend_url: str | None = Field(
        default=None,
        description="Redis URL for semantic cache backend. Defaults to redis_url if not set.",
    )

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------
    compression_mode: str = Field(
        default="balanced",
        description="Compression mode: 'safe', 'balanced', or 'aggressive'. "
        "Maps to transform enablement flags internally.",
    )
    transform_reference_sub: bool = True
    transform_tool_filter: bool = True
    transform_prefix_opt: bool = True
    transform_output_cleanup: bool = True
    transform_format_conversion: bool = True
    transform_message_dedup: bool = True
    transform_semantic_compress: bool = True
    transform_content_profiler: bool = True
    transform_runtime_contract: bool = True
    transform_structural_fingerprint: bool = True
    transform_self_information: bool = True
    transform_hierarchical_summary: bool = True
    transform_strategy_selector: bool = True
    transform_context_selector: bool = True
    transform_cache_arbitrage: bool = True
    transform_dictionary_compress: bool = True
    transform_grammar_compress: bool = True
    rate_distortion_budget: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable distortion (0-1) for RateDistortionCompressor. "
        "Higher = more compression, lower = higher fidelity.",
    )
    submodular_token_budget: int = Field(
        default=4096,
        ge=1,
        description="Token budget for SubmodularContextSelector.",
    )
    strategy_selection_mode: str = Field(
        default="bandit",
        description="Strategy selection mode: 'bandit' (LinUCB) or 'fixed'.",
    )

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------
    connection_migration: bool = Field(
        default=True,
        description="Allow sessions to migrate across connections.",
    )

    # ------------------------------------------------------------------
    # TACC (Token-Aware Congestion Control)
    # ------------------------------------------------------------------
    tacc_enabled: bool = Field(
        default=True,
        description="Enable per-provider AIMD congestion control.",
    )
    tacc_initial_window: int = Field(
        default=1,
        ge=1,
        description="Initial request-window size for TACC slow start.",
    )

    # ------------------------------------------------------------------
    # Stall detection
    # ------------------------------------------------------------------
    provider_stall_detection_enabled: bool = Field(
        default=True,
        description="Enable stream stall detection in the transport layer.",
    )

    # ------------------------------------------------------------------
    # Execution transforms (proxy-only features)
    # ------------------------------------------------------------------
    transform_batching: bool = True
    transform_speculation: bool = True

    # ------------------------------------------------------------------
    # Policy / Limits
    # ------------------------------------------------------------------
    default_input_token_budget: int | None = Field(
        default=None,
        description="Max input tokens (messages + tools) per request. "
        "Rejects requests that exceed this budget.",
    )
    min_max_tokens: int = Field(
        default=64,
        ge=1,
        description="Minimum allowed max_tokens value. Requests with "
        "max_tokens below this are rejected. 64 ensures "
        "models can produce a meaningful response.",
    )
    default_context_limit: int = Field(default=128_000, ge=1)
    max_request_size_mb: int = Field(default=10, ge=1)
    request_timeout_seconds: int = Field(default=120, ge=1)
    compression_timeout_ms: int = Field(default=100, ge=1)
    max_transform_expansion_ratio: float = Field(
        default=1.5,
        ge=1.0,
        description="Hard cap on per-transform intermediate token growth. "
        "Any transform that expands input tokens by more than "
        "this ratio is aborted.",
    )
    graceful_degradation: bool = True

    # ------------------------------------------------------------------
    # Resilience (retry only — LATTICE is NOT a router)
    # ------------------------------------------------------------------
    provider_stall_timeout_seconds: int = Field(default=30, ge=1)
    provider_max_retries: int = Field(
        default=3,
        ge=0,
        description="Max retries for the SAME model on transient errors "
        "(429, 502, 503, 504). LATTICE never changes the model.",
    )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    metrics_enabled: bool = True
    metrics_port: int = Field(default=9090, ge=1024, le=65535)
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return v_upper

    @field_validator("log_format")
    @classmethod
    def _validate_log_format(cls, v: str) -> str:
        allowed = {"json", "console"}
        if v.lower() not in allowed:
            raise ValueError(f"log_format must be one of {allowed}, got {v!r}")
        return v.lower()

    @field_validator("session_store", "semantic_cache_backend")
    @classmethod
    def _validate_backend_choice(cls, v: str) -> str:
        allowed = {"memory", "redis"}
        if v.lower() not in allowed:
            raise ValueError(f"backend must be one of {allowed}, got {v!r}")
        return v.lower()

    @field_validator("compression_mode")
    @classmethod
    def _validate_compression_mode(cls, v: str) -> str:
        allowed = {"safe", "balanced", "aggressive"}
        v_lower = v.lower()
        if v_lower not in allowed:
            raise ValueError(f"compression_mode must be one of {allowed}, got {v!r}")
        return v_lower

    @model_validator(mode="after")
    def _validate_redis_url(self) -> LatticeConfig:
        if self.session_store == "redis" and not self.redis_url:
            raise ValueError("redis_url is required when session_store='redis'")
        if self.semantic_cache_backend == "redis":
            cache_url = self.semantic_cache_backend_url or self.redis_url
            if not cache_url:
                raise ValueError(
                    "semantic_cache_backend_url or redis_url is required "
                    "when semantic_cache_backend='redis'"
                )
        return self

    @model_validator(mode="after")
    def _validate_proxy_workers(self) -> LatticeConfig:
        if self.proxy_workers is not None and self.proxy_workers < 1:
            raise ValueError("proxy_workers must be >= 1")
        return self

    @model_validator(mode="after")
    def _apply_compression_mode_transforms(self) -> LatticeConfig:
        """compression_mode is the single source of truth for transform flags.

        Applies the mode mapping ONLY when the user explicitly set
        ``compression_mode`` and did NOT set any individual transform flags.
        This preserves backward compatibility for advanced users who pin
        transforms directly.
        """
        if "compression_mode" not in self.model_fields_set:
            return self
        # If any transform flag was explicitly set, respect it
        transform_fields = {f for f in self.model_fields_set if f.startswith("transform_")}
        if transform_fields:
            return self
        self.apply_compression_mode()
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def max_request_bytes(self) -> int:
        """Max request size in bytes."""
        return self.max_request_size_mb * 1024 * 1024

    @property
    def worker_count(self) -> int:
        """Resolved worker count."""
        return self.proxy_workers or (os.cpu_count() or 1)

    def apply_compression_mode(self) -> None:
        """Map compression_mode to transform enablement flags.

        safe      — non-lossy transforms only (dedup, prefix, cleanup, format)
        balanced  — safe + selective compression (profiler, strategy, cache)
        aggressive — balanced + lossy transforms (semantic, summary, selector)
        """
        mode = self.compression_mode

        # Base flags — all modes get SAFE transforms
        self.transform_content_profiler = True  # SIG metadata source — ALWAYS ON
        self.transform_runtime_contract = True
        self.transform_reference_sub = True
        self.transform_tool_filter = True
        self.transform_prefix_opt = True
        self.transform_output_cleanup = True
        self.transform_format_conversion = True
        self.transform_message_dedup = True
        self.transform_cache_arbitrage = True

        if mode == "safe":
            self.transform_structural_fingerprint = False
            self.transform_self_information = False
            self.transform_hierarchical_summary = False
            self.transform_strategy_selector = False
            self.transform_context_selector = False
            self.transform_dictionary_compress = False
            self.transform_grammar_compress = False
            self.transform_semantic_compress = False
            self.rate_distortion_budget = 0.0
        elif mode == "balanced":
            self.transform_structural_fingerprint = True
            self.transform_self_information = True
            self.transform_hierarchical_summary = False
            self.transform_strategy_selector = True
            self.transform_context_selector = False
            self.transform_dictionary_compress = True
            self.transform_grammar_compress = True
            self.transform_semantic_compress = False
            self.rate_distortion_budget = 0.02
        elif mode == "aggressive":
            self.transform_structural_fingerprint = True
            self.transform_self_information = True
            self.transform_hierarchical_summary = True
            self.transform_strategy_selector = True
            self.transform_context_selector = True
            self.transform_dictionary_compress = True
            self.transform_grammar_compress = True
            self.transform_semantic_compress = True
            self.rate_distortion_budget = 0.05

    def proxy_url(self) -> str:
        """Return the LATTICE proxy base URL (OpenAI-compatible endpoint)."""
        return f"http://{self.proxy_host}:{self.proxy_port}/v1"

    def is_transform_enabled(self, name: str) -> bool:
        """Check if a named transform is enabled.

        Transforms are identified by their canonical class name or
        short name. The mapping covers all built-in transforms.
        """
        mapping = {
            # Phase 0
            "reference_substitution": self.transform_reference_sub,
            "tool_output_filter": self.transform_tool_filter,
            "prefix_optimizer": self.transform_prefix_opt,
            "output_cleanup": self.transform_output_cleanup,
            # Short aliases
            "reference_sub": self.transform_reference_sub,
            "tool_filter": self.transform_tool_filter,
            "prefix_opt": self.transform_prefix_opt,
            # Phase 1
            "delta_encoder": True,
            "format_conversion": self.transform_format_conversion,
            # Phase 2
            "batching": True,
            "speculative": True,
            # Phase 3 — production-grade transforms
            "message_dedup": self.transform_message_dedup,
            "message_deduplicator": self.transform_message_dedup,
            "semantic_compress": self.transform_semantic_compress,
            "semantic_compressor": self.transform_semantic_compress,
            # Phase D — rate-distortion compressor
            "rate_distortion": self.transform_semantic_compress,
            "rate_distortion_compressor": self.transform_semantic_compress,
            "content_profiler": self.transform_content_profiler,
            "runtime_contract": self.transform_runtime_contract,
            "structural_fingerprint": self.transform_structural_fingerprint,
            "self_information": self.transform_self_information,
            "hierarchical_summary": self.transform_hierarchical_summary,
            "strategy_selector": self.transform_strategy_selector,
            "context_selector": self.transform_context_selector,
            "cache_arbitrage": self.transform_cache_arbitrage,
            "dictionary_compress": self.transform_dictionary_compress,
            "dictionary_compressor": self.transform_dictionary_compress,
            "grammar_compress": self.transform_grammar_compress,
            "grammar_compressor": self.transform_grammar_compress,
            # Deleted transforms (Phase 0 cleanup) — always False
            "stream_optimizer": False,
            "optimal_stopping": False,
            "fountain_codes": False,
            "convex_selector": False,
        }
        return mapping.get(name, False)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> LatticeConfig:
        """Load configuration from a YAML file."""
        import yaml  # type: ignore[import-untyped]

        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a dict, got {type(data).__name__}")

        return cls(**data)

    @classmethod
    def auto(cls) -> LatticeConfig:
        """Auto-discover configuration.

        Priority: env vars > ~/.config/lattice/config.yaml > ./lattice.yaml > defaults.
        This is the standard entry point.
        """
        # Check for explicit YAML files
        cwd_yaml = pathlib.Path.cwd() / "lattice.yaml"
        home_yaml = pathlib.Path.home() / ".config" / "lattice" / "config.yaml"

        for path in (cwd_yaml, home_yaml):
            if path.exists():
                return cls.from_yaml(path)

        # No YAML found — rely on env vars and defaults
        return cls()
