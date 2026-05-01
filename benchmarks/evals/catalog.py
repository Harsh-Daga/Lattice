"""Catalog of providers and scenarios used by production evals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.scenarios.prompts import BenchmarkScenario, get_scenarios
from lattice.core.credentials import CredentialResolver
from lattice.providers.capabilities import CapabilityRegistry, get_capability_registry

_DEFAULT_PROVIDER_ORDER = [
    "openai",
    "anthropic",
    "ollama",
    "groq",
    "deepseek",
    "mistral",
    "cohere",
    "gemini",
    "vertex",
    "azure",
    "bedrock",
    "openrouter",
    "fireworks",
    "together",
    "perplexity",
    "ai21",
]

_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku",
    "ollama": "llama3.2",
    "ollama-cloud": "kimi-k2.6:cloud",
    "groq": "llama-3.1-8b-instant",
    "deepseek": "deepseek-chat",
    "mistral": "mistral-small-latest",
    "cohere": "command-r",
    "gemini": "gemini-2.5-flash",
    "vertex": "gemini-2.5-flash",
    "azure": "gpt-4o-mini",
    "bedrock": "anthropic.claude-3-haiku-20240307-v1:0",
    "openrouter": "openai/gpt-4o-mini",
    "fireworks": "accounts/fireworks/models/llama4-maverick-instruct-basic",
    "together": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "perplexity": "sonar",
    "ai21": "jamba-1.5-large",
}


@dataclass(frozen=True, slots=True)
class EvalTarget:
    """A provider/model target for live provider evals."""

    provider: str
    model: str
    base_url: str | None = None
    available: bool = True
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "available": self.available,
            "skip_reason": self.skip_reason,
        }


def default_scenarios(names: list[str] | None = None) -> list[BenchmarkScenario]:
    """Return the benchmark scenarios used in production evals."""
    return get_scenarios(names)


def default_provider_targets(
    providers: list[str] | None = None,
    *,
    model_overrides: dict[str, str] | None = None,
    strict_model_selection: bool = False,
    registry: CapabilityRegistry | None = None,
    resolver: CredentialResolver | None = None,
) -> list[EvalTarget]:
    """Return curated provider targets in preferred evaluation order."""
    registry = registry or get_capability_registry()
    resolver = resolver or CredentialResolver()
    model_overrides = model_overrides or {}
    selected = providers or list(_DEFAULT_PROVIDER_ORDER)
    targets: list[EvalTarget] = []

    if strict_model_selection and providers:
        missing = [provider for provider in selected if provider not in model_overrides]
        if missing:
            raise ValueError(
                "Exact model selection required for requested providers: "
                + ", ".join(missing)
                + ". Pass --provider-model PROVIDER=MODEL for each selected provider."
            )

    for provider in selected:
        caps = registry.get(provider)
        model_name = model_overrides.get(provider) or _DEFAULT_MODELS.get(provider)
        if model_name is None and caps and caps.supported_models:
            model_name = caps.supported_models[0]
        if model_name is None:
            model_name = "gpt-4o-mini"
        if "/" not in model_name and provider not in {"ollama"}:
            model = f"{provider}/{model_name}"
        elif provider == "ollama":
            model = model_name if "/" in model_name else f"ollama/{model_name}"
        else:
            model = model_name

        creds = resolver.resolve(provider)
        available = True
        reason = ""

        if provider == "ollama":
            base_url = creds.base_url
        else:
            base_url = creds.base_url or (caps.default_base_url if caps else None)

        if provider == "bedrock":
            if not (creds.aws_access_key_id and creds.aws_secret_access_key and creds.aws_region):
                available = False
                reason = "missing AWS credentials"
        elif provider in {"azure"}:
            if not (creds.api_key and base_url):
                available = False
                reason = "missing Azure OpenAI endpoint or key"
        elif provider in {"ollama"}:
            available = True
        elif provider in {"gemini", "vertex", "openai", "anthropic", "groq", "deepseek", "mistral", "cohere", "openrouter", "fireworks", "together", "perplexity", "ai21"}:
            if not creds.api_key:
                available = False
                reason = "missing API key"
        else:
            if not creds.api_key:
                available = False
                reason = "missing API key"

        targets.append(
            EvalTarget(
                provider=provider,
                model=model,
                base_url=base_url,
                available=available,
                skip_reason=reason,
            )
        )

    return targets
