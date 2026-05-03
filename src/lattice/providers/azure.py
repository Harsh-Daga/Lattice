"""Azure OpenAI adapter.

Azure OpenAI is mostly identical to standard OpenAI Chat Completions,
but with three critical differences:

1. **Deployment-based URL** — the model is NOT in the request body;
   instead the deployment name is in the URL path.
2. **Query parameter** — ``api-version`` is required (e.g. ``?api-version=2023-05-15``).
3. **Auth header** — ``api-key`` instead of ``Authorization: Bearer``.

Reference: https://learn.microsoft.com/en-us/azure/ai-services/openai/
"""

from __future__ import annotations

from typing import Any

from lattice.core.transport import Request, Response
from lattice.providers.openai import OpenAIAdapter


class AzureAdapter(OpenAIAdapter):
    """Azure OpenAI adapter.

    Inherits serialization / deserialization from OpenAIAdapter and only
    overrides endpoint construction, auth, and model handling.
    """

    name = "azure"
    _PREFIXES = {"azure"}

    def supports(self, model: str) -> bool:
        """Matches ``azure/...``."""
        if "/" not in model:
            return False
        prefix = model.split("/", 1)[0].lower()
        return prefix in self._PREFIXES

    def chat_endpoint(self, model: str, base_url: str) -> str:
        """Azure path includes the *deployment* name (stripped of ``azure/`` prefix).

        ``model`` here is actually the deployment name, e.g. ``azure/gpt-4o``
        becomes ``gpt-4o`` in the path.
        """
        deployment = model
        if "/" in deployment:
            deployment = deployment.split("/", 1)[1]
        return (
            f"{base_url.rstrip('/')}/openai/deployments/{deployment}/chat/completions"
            "?api-version=2024-02-01"
        )

    def auth_headers(self, api_key: str | None) -> dict[str, str]:
        h: dict[str, str] = {}
        if api_key:
            h["api-key"] = api_key
        return h

    def map_model_name(self, model: str) -> str:
        """Strip the ``azure/`` prefix."""
        if "/" in model:
            prefix, rest = model.split("/", 1)
            if prefix.lower() in self._PREFIXES:
                return rest
        return model

    def detect(self, signals: Any) -> Any:
        """Detect Azure from explicit signals and the ``api-key`` header.

        Azure OpenAI uses ``api-key`` for authentication — a header that is
        **not** used by standard OpenAI providers (which use
        ``Authorization: Bearer``).  This makes ``api-key`` a strong,
        Azure-specific signal.
        """
        from lattice.gateway.detect_helpers import (
            detect_explicit,
            detect_header_present,
            detect_model_prefix,
            highest_confidence,
        )

        return highest_confidence(
            self.name,
            detect_explicit(signals, self.name, aliases=self._PREFIXES),
            detect_header_present(
                signals,
                self.name,
                "api-key",
                "api-key header is Azure-specific (OpenAI uses Authorization: Bearer)",
            ),
            detect_model_prefix(signals, self.name, aliases=self._PREFIXES),
        )

    def extra_headers(self, _request: Request) -> dict[str, str]:
        return {}

    def retry_config(self) -> dict[str, Any]:
        return {
            "max_retries": 3,
            "backoff_factor": 1.0,
            "retry_on": (429, 502, 503, 504),
        }

    def serialize_request(self, request: Request) -> dict[str, Any]:
        """Azure ignores the ``model`` field in the body — deployment is URL-based."""
        body = super().serialize_request(request)
        # Azure ignores this key, but we keep it for compat with other layers
        body.pop("model", None)
        # Azure OpenAI supports reasoning_effort for o1/o3 deployments
        reasoning_effort = request.metadata.get("reasoning_effort") or request.extra_body.get(
            "reasoning_effort"
        )
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        return body

    def deserialize_response(self, data: dict[str, Any]) -> Response:
        """Azure response shape is identical to OpenAI."""
        return super().deserialize_response(data)

    def normalize_sse_chunk(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        return super().normalize_sse_chunk(chunk)

    def extract_content(self, msg: dict[str, Any]) -> str:
        return super().extract_content(msg)
