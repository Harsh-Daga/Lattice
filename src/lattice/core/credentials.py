"""Provider credential resolution.

Resolves API keys, base URLs, and provider-specific settings from:
1. Explicit config file (`~/.config/lattice/lattice.config.toml`)
2. Environment variables (standard names like OPENAI_API_KEY)
3. Fallback defaults

Design inspired by LiteLLM's `get_secret()` and FreeRouter's auth system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()

# =============================================================================
# Standard env var mappings (industry convention)
# =============================================================================

_PROVIDER_ENV_VARS: dict[str, dict[str, str]] = {
    "openai": {
        "api_key": "OPENAI_API_KEY",
        "base_url": "OPENAI_BASE_URL",
    },
    "anthropic": {
        "api_key": "ANTHROPIC_API_KEY",
        "base_url": "ANTHROPIC_BASE_URL",
    },
    "azure": {
        "api_key": "AZURE_OPENAI_API_KEY",
        "base_url": "AZURE_OPENAI_ENDPOINT",
        "api_version": "AZURE_OPENAI_API_VERSION",
    },
    "ollama": {
        "base_url": "OLLAMA_HOST",
    },
    "ollama-cloud": {
        "api_key": "OLLAMA_CLOUD_API_KEY",
        "base_url": "OLLAMA_CLOUD_BASE_URL",
    },
    "bedrock": {
        "aws_access_key_id": "AWS_ACCESS_KEY_ID",
        "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "aws_region": "AWS_REGION",
    },
    # OpenAI-compatible providers
    "groq": {
        "api_key": "GROQ_API_KEY",
        "base_url": "GROQ_BASE_URL",
    },
    "together": {
        "api_key": "TOGETHER_API_KEY",
        "base_url": "TOGETHER_BASE_URL",
    },
    "deepseek": {
        "api_key": "DEEPSEEK_API_KEY",
        "base_url": "DEEPSEEK_BASE_URL",
    },
    "perplexity": {
        "api_key": "PERPLEXITY_API_KEY",
        "base_url": "PERPLEXITY_BASE_URL",
    },
    "mistral": {
        "api_key": "MISTRAL_API_KEY",
        "base_url": "MISTRAL_BASE_URL",
    },
    "fireworks": {
        "api_key": "FIREWORKS_API_KEY",
        "base_url": "FIREWORKS_BASE_URL",
    },
    "openrouter": {
        "api_key": "OPENROUTER_API_KEY",
        "base_url": "OPENROUTER_BASE_URL",
    },
    "cohere": {
        "api_key": "COHERE_API_KEY",
        "base_url": "COHERE_BASE_URL",
    },
    "ai21": {
        "api_key": "AI21_API_KEY",
        "base_url": "AI21_BASE_URL",
    },
    "gemini": {
        "api_key": "GEMINI_API_KEY",
        "base_url": "GEMINI_BASE_URL",
    },
    "google": {
        "api_key": "GOOGLE_API_KEY",
        "base_url": "GOOGLE_BASE_URL",
    },
    "vertex": {
        "api_key": "VERTEX_API_KEY",
        "base_url": "VERTEX_BASE_URL",
    },
}


# =============================================================================
# Provider credentials dataclass
# =============================================================================


@dataclass
class ProviderCredentials:
    """Resolved credentials for a single provider."""

    api_key: str | None = None
    base_url: str | None = None
    # Provider-specific extras
    api_version: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str | None = None
    # Catch-all for custom provider fields
    extras: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get any credential field by name."""
        if hasattr(self, key):
            val = getattr(self, key)
            if val is not None:
                return val
        return self.extras.get(key, default)


# =============================================================================
# CredentialResolver
# =============================================================================


class CredentialResolver:
    """Resolve provider credentials from config + env vars.

    Priority (highest first):
    1. Runtime overrides passed to ``register()``
    2. Config file values (loaded from ``~/.config/lattice/lattice.config.toml``)
    3. Environment variables (standard names)
    4. Hardcoded defaults

    Usage
    -----
        resolver = CredentialResolver()
        creds = resolver.resolve("openai")
        print(creds.api_key)  # from OPENAI_API_KEY env var
        print(creds.base_url)  # from config or default
    """

    def __init__(self) -> None:
        self._overrides: dict[str, ProviderCredentials] = {}
        self._config: dict[str, dict[str, Any]] = {}
        self._log = logger.bind(module="credential_resolver")
        self._load_config_file()

    def _load_config_file(self) -> None:
        """Load TOML config from ``~/.config/lattice/lattice.config.toml``."""
        import pathlib

        config_path = pathlib.Path.home() / ".config" / "lattice" / "lattice.config.toml"
        if not config_path.exists():
            return

        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                self._log.warning("toml_not_installed", path=str(config_path))
                return

        try:
            with config_path.open("rb") as f:
                data = tomllib.load(f)
        except Exception as exc:
            self._log.warning("config_load_failed", path=str(config_path), error=str(exc))
            return

        providers = data.get("providers")
        if isinstance(providers, dict):
            self._config = {k.lower(): v for k, v in providers.items() if isinstance(v, dict)}
            self._log.info(
                "config_loaded",
                path=str(config_path),
                providers=list(self._config.keys()),
            )

    def register(self, provider: str, **kwargs: Any) -> None:
        """Register runtime credentials for a provider (highest priority)."""
        provider = provider.lower()
        creds = ProviderCredentials(**kwargs)
        self._overrides[provider] = creds
        self._log.debug("credentials_registered", provider=provider)

    def resolve(self, provider: str) -> ProviderCredentials:
        """Resolve credentials for a provider.

        Checks overrides → config file → env vars → defaults.
        """
        provider = provider.lower()

        # 1. Runtime overrides
        if provider in self._overrides:
            return self._overrides[provider]

        # Start with empty credentials
        creds = ProviderCredentials()

        # 2. Config file
        cfg = self._config.get(provider, {})

        # 3. Env var mapping
        env_map = _PROVIDER_ENV_VARS.get(provider, {})

        # Resolve each field
        creds.api_key = self._resolve_field("api_key", cfg, env_map)
        creds.base_url = self._resolve_field("base_url", cfg, env_map)
        creds.api_version = self._resolve_field("api_version", cfg, env_map)
        creds.aws_access_key_id = self._resolve_field("aws_access_key_id", cfg, env_map)
        creds.aws_secret_access_key = self._resolve_field("aws_secret_access_key", cfg, env_map)
        creds.aws_region = self._resolve_field("aws_region", cfg, env_map)

        # Extras (any config fields not in the standard schema)
        known = {
            "api_key",
            "base_url",
            "api_version",
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_region",
        }
        creds.extras = {k: v for k, v in cfg.items() if k not in known}

        return creds

    @staticmethod
    def _resolve_field(
        field_name: str,
        cfg: dict[str, Any],
        env_map: dict[str, str],
    ) -> str | None:
        """Resolve a single field: config file → env var → None."""
        # Config file takes priority over env vars
        if field_name in cfg and cfg[field_name]:
            return str(cfg[field_name])
        # Env var
        env_var = env_map.get(field_name)
        if env_var:
            val = os.environ.get(env_var)
            if val:
                return val
        return None

    def list_providers(self) -> list[str]:
        """Return all providers with known credentials."""
        providers = set(self._overrides.keys())
        providers.update(self._config.keys())
        providers.update(_PROVIDER_ENV_VARS.keys())
        return sorted(providers)
