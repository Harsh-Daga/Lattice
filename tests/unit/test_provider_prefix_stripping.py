"""Tests that every provider adapter correctly strips its own prefix.

This prevents bugs like `ollama-cloud/kimi-k2.6:cloud` being sent to the
provider as `ollama-cloud/kimi-k2.6:cloud` instead of `kimi-k2.6:cloud`.
"""

from __future__ import annotations

from typing import Any

import pytest

from lattice.providers.transport import ProviderRegistry


class TestProviderPrefixStripping:
    """Every adapter that uses a prefix must strip it in map_model_name."""

    def _prefixes_for_adapter(self, adapter: Any) -> set[str]:
        """Extract the set of prefixes an adapter claims to support."""
        # Direct _PREFIXES attribute (OpenAIAdapter, OpenAICompatibleAdapter, etc.)
        prefixes: set[str] = getattr(adapter, "_PREFIXES", set())
        if prefixes:
            return {p.lower() for p in prefixes}
        # OllamaAdapter uses _PREFIX string
        prefix: str | None = getattr(adapter, "_PREFIX", None)
        if prefix:
            return {prefix.lower()}
        # AnthropicAdapter uses _PREFIXES
        prefixes = getattr(adapter, "_PREFIXES", set())
        return {p.lower() for p in prefixes}

    def test_all_adapters_strip_their_prefix(self) -> None:
        registry = ProviderRegistry()
        failures: list[str] = []

        for adapter in registry._adapters:
            name = adapter.name
            prefixes = self._prefixes_for_adapter(adapter)

            # Skip adapters that do not declare a provider prefix set.
            if not prefixes:
                continue

            for prefix in prefixes:
                model = f"{prefix}/test-model"
                if not adapter.supports(model):
                    # Only report a failure if this prefix is the adapter's PRIMARY
                    # prefix (i.e., its own name). Adapters may inherit _PREFIXES
                    # from a parent for strip-only purposes.
                    if prefix == name.lower():
                        failures.append(f"{name}: supports('{model}') returned False")
                    continue

                mapped = adapter.map_model_name(model)
                if mapped != "test-model":
                    failures.append(
                        f"{name}: map_model_name('{model}') -> '{mapped}' (expected 'test-model')"
                    )

        if failures:
            pytest.fail("Prefix stripping failures:\n" + "\n".join(failures))

    def test_no_adapter_leaks_prefix_into_request(self) -> None:
        """Simulate the full transport path: adapter.serialize_request must not contain the prefix."""
        from lattice.core.transport import Message, Request

        registry = ProviderRegistry()
        failures: list[str] = []

        for adapter in registry._adapters:
            prefixes = self._prefixes_for_adapter(adapter)
            if not prefixes:
                continue

            for prefix in prefixes:
                model = f"{prefix}/test-model"
                if not adapter.supports(model):
                    continue

                req = Request(
                    model=model,
                    messages=[Message(role="user", content="hello")],
                )
                mapped_model = adapter.map_model_name(req.model)
                req.model = mapped_model
                payload = adapter.serialize_request(req)
                body_model = payload.get("model", "")

                if prefix in body_model.lower():
                    failures.append(
                        f"{adapter.name}: serialize_request still contains prefix "
                        f"'{prefix}' in model field: '{body_model}'"
                    )

        if failures:
            pytest.fail("Prefix leak in serialize_request:\n" + "\n".join(failures))

    def test_ollama_cloud_prefix_stripped(self) -> None:
        """Regression test for the ollama-cloud prefix bug."""
        from lattice.providers.ollama import OllamaCloudAdapter

        adapter = OllamaCloudAdapter()
        assert adapter.supports("ollama-cloud/kimi-k2.6:cloud")
        assert adapter.map_model_name("ollama-cloud/kimi-k2.6:cloud") == "kimi-k2.6:cloud"
        assert adapter.map_model_name("ollama-cloud/llama3.2") == "llama3.2"
        # Bare model should pass through unchanged
        assert adapter.map_model_name("gpt-4o") == "gpt-4o"
