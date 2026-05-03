"""Google Gemini / Vertex AI adapter.

Google provides two API paths:
- **Gemini API** — developer API, API key auth, endpoint ``generativelanguage.googleapis.com``
- **Vertex AI** — enterprise GCP, service-account auth, endpoint ``<region>-aiplatform.googleapis.com``

This adapter targets the **Gemini API** (OpenAI-compatible) which is the
simplest path for most users. Vertex AI users can configure a custom
base URL.

Docs: https://ai.google.dev/gemini-api/docs/openai
"""

from __future__ import annotations

from typing import Any

from lattice.core.transport import Request
from lattice.providers.openai_compatible import OpenAICompatibleAdapter


class GeminiAdapter(OpenAICompatibleAdapter):
    """Google Gemini (OpenAI-compatible endpoint).

    Model format: ``gemini/gemini-1.5-pro`` or ``gemini/gemini-1.5-flash``
    """

    name = "gemini"
    _PREFIXES = {"gemini", "google"}
    _DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

    def detect(self, signals: Any) -> Any:
        """Detect Gemini from explicit signals and the ``x-goog-api-key`` header.

        The ``x-goog-api-key`` header is **exclusive** to Google APIs
        (Gemini and Vertex).  No other provider in the LATTICE ecosystem
        uses this header, making it a strong auth signal.
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
                "x-goog-api-key",
                "x-goog-api-key header is Google-specific (Gemini / Vertex)",
            ),
            detect_model_prefix(signals, self.name, aliases=self._PREFIXES),
        )

    def serialize_request(self, request: Request) -> dict[str, Any]:
        body = super().serialize_request(request)
        cached_content = (
            request.metadata.get("cachedContent")
            or request.metadata.get("cached_content")
            or request.metadata.get("gemini_cached_content")
            or request.extra_body.get("cachedContent")
            or request.extra_body.get("cached_content")
        )
        if cached_content is not None:
            body["cachedContent"] = cached_content
        return body


class VertexAdapter(OpenAICompatibleAdapter):
    """Google Vertex AI (OpenAI-compatible endpoint).

    Requires ``GOOGLE_APPLICATION_CREDENTIALS`` env var or ADC.
    Base URL format: ``https://<region>-aiplatform.googleapis.com/v1beta1/projects/<project>/locations/<region>/endpoints/openapi``
    """

    name = "vertex"
    _PREFIXES = {"vertex"}
    _DEFAULT_BASE_URL = ""

    def serialize_request(self, request: Request) -> dict[str, Any]:
        body = super().serialize_request(request)
        cached_content = (
            request.metadata.get("cachedContent")
            or request.metadata.get("cached_content")
            or request.metadata.get("vertex_cached_content")
            or request.extra_body.get("cachedContent")
            or request.extra_body.get("cached_content")
        )
        if cached_content is not None:
            body["cachedContent"] = cached_content
        return body

    def auth_headers(self, api_key: str | None) -> dict[str, str]:
        """Vertex uses GCP bearer tokens from ADC, not API keys."""
        h: dict[str, str] = {}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        return h
