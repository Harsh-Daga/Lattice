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
    # ── Production core ──
    transform_content_profiler: bool = True
    transform_runtime_contract: bool = True
    transform_cache_arbitrage: bool = True
    transform_prefix_opt: bool = True
    transform_reference_sub: bool = True
    transform_tool_filter: bool = True
    transform_output_cleanup: bool = True
    # ── Useful extras ──
    transform_constraint_lifting: bool = True
    transform_causal_chain: bool = True
    transform_message_dedup: bool = True
    transform_context_selector: bool = True
    transform_strategy_selector: bool = True
    transform_format_conversion: bool = True
    transform_diagnostic_rle: bool = True
    transform_columnar_pack: bool = True
    transform_json_shape: bool = True
    transform_path_prefix: bool = True
    transform_extractive_compress: bool = True
    transform_tool_projection: bool = True
    transform_rate_distortion: bool = True
    # ── Optimizer flags (Phase 3 architecture)
    transform_representation_optimizer: bool = True
    transform_structure_optimizer: bool = True
    transform_reference_optimizer: bool = True
    transform_tool_optimizer: bool = True
    transform_context_optimizer: bool = True
    transform_diagnostic_optimizer: bool = True
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

    # Optimizer architecture (Phase 3)
    use_optimizer_pipeline: bool = Field(
        default=False,
        description="Enable the new optimizer-based pipeline (representation_optimizer, structure_optimizer, etc.) instead of individual transforms.",
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
    transform_delta_encode: bool = True
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

        Production pipeline has 7 core transforms. Everything else is
        opt-in via direct config flags (not mode-based).
        """
        mode = self.compression_mode

        # ── Production core (always enabled) ──
        self.transform_content_profiler = True
        self.transform_runtime_contract = True
        self.transform_prefix_opt = True
        self.transform_cache_arbitrage = True
        self.transform_reference_sub = True
        self.transform_tool_filter = True
        self.transform_output_cleanup = True

        # ── Conditional: extra transforms via mode ──
        # balanced + aggressive: enable additional useful transforms
        self.transform_constraint_lifting = mode in ("balanced", "aggressive")
        self.transform_causal_chain = mode in ("balanced", "aggressive")
        self.transform_message_dedup = mode == "aggressive"
        self.transform_context_selector = mode == "aggressive"
        self.transform_rate_distortion = mode == "aggressive"
        self.transform_format_conversion = mode == "aggressive"
        self.transform_diagnostic_rle = mode == "aggressive"
        self.transform_columnar_pack = mode == "aggressive"
        self.transform_json_shape = mode == "aggressive"
        self.transform_path_prefix = mode == "aggressive"
        self.transform_extractive_compress = mode == "aggressive"
        self.transform_tool_projection = mode == "aggressive"
        self.rate_distortion_budget = 0.05 if mode == "aggressive" else 0.02

    def proxy_url(self) -> str:
        """Return the LATTICE proxy base URL (OpenAI-compatible endpoint)."""
        return f"http://{self.proxy_host}:{self.proxy_port}/v1"

    def is_transform_enabled(self, name: str) -> bool:
        """Check if a named transform is enabled.

        Delegates to :func:`~lattice.transforms.registry.is_transform_enabled`
        so that config, pipeline, and safety metadata all share one registry.
        """
        # Hard-deleted transforms from Phase 0 cleanup — never enable
        if name in (
            "stream_optimizer",
            "optimal_stopping",
            "fountain_codes",
            "convex_selector",
        ):
            return False
        from lattice.transforms.registry import is_transform_enabled as _registry_is_enabled

        return _registry_is_enabled(self, name)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | pathlib.Path) -> LatticeConfig:
        """Load a standalone config from YAML — no env var overlay.

        Use :meth:`auto` for the standard priority stack (env > yaml > defaults).
        """
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
        """Auto-discover configuration. Priority: env vars > YAML > defaults.

        Builds from env vars and defaults, then layers YAML values for
        fields not already set via environment variables.
        """
        cwd_yaml = pathlib.Path.cwd() / "lattice.yaml"
        home_yaml = pathlib.Path.home() / ".config" / "lattice" / "config.yaml"

        # Collect YAML data from all discovered files
        yaml_data: dict = {}
        for path in (home_yaml, cwd_yaml):
            if path.exists():
                import yaml  # type: ignore[import-untyped]

                with path.open(encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    yaml_data.update(data)

        if not yaml_data:
            return cls()

        # Build from env vars + defaults first
        instance = cls()

        # Overlay YAML values for fields NOT explicitly set via env vars
        model_fields = cls.model_fields
        env_prefix = cls.model_config.get("env_prefix", "LATTICE_")
        for key, value in yaml_data.items():
            if key not in model_fields:
                continue
            env_var = f"{env_prefix}{key.upper()}"
            if env_var in os.environ:
                continue
            object.__setattr__(instance, key, value)

        return instance
