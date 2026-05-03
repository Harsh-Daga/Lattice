"""ChatGPT / Codex provider adapter.

Codex is not a separate API format — it speaks OpenAI's wire protocol.
The only differences are:

1. **Upstream endpoint** — ``https://chatgpt.com/backend-api/codex/...``
   instead of ``https://api.openai.com/v1/...``.
2. **Authentication** — OAuth JWT with a nested ``chatgpt_account_id`` claim.

This adapter exists primarily for **routing detection**.  All serialization,
deserialization, streaming, and auth header formatting are inherited from
:class:`OpenAIAdapter`.

Registry placement
------------------
:class:`ChatGPTAdapter` is registered **before** :class:`OpenAIAdapter` in
:class:`ProviderRegistry` so that Codex JWTs are identified as ``chatgpt``
rather than falling through to the generic OpenAI handler.
"""

from __future__ import annotations

from typing import Any

from lattice.gateway.detect_helpers import detect_explicit, highest_confidence
from lattice.gateway.routing import DetectionConfidence, DetectionResult
from lattice.integrations.codex.auth import _is_codex_jwt
from lattice.providers.openai import OpenAIAdapter


class ChatGPTAdapter(OpenAIAdapter):
    """Detects ChatGPT / Codex requests and delegates I/O to OpenAI format."""

    name = "chatgpt"
    _PREFIXES = {"chatgpt", "codex"}

    def detect(self, signals: Any) -> Any:
        """Detect ChatGPT / Codex from explicit signals.

        Confidence priority (highest first):
        1. **EXPLICIT** — body field ``provider=chatgpt`` or header
           ``x-lattice-provider=chatgpt``.
        2. **AUTH** — ``Authorization`` header contains a Codex JWT
           (nested ``chatgpt_account_id`` claim).  This is a strong,
           ChatGPT-specific signal because no other provider issues
           these JWTs.
        3. **AUTH** — ``ChatGPT-Account-ID`` header is present.
        4. **MODEL** — model prefix ``chatgpt/...`` or ``codex/...``.

        Returns
        -------
        DetectionResult with provider ``"chatgpt"``.  ``NONE`` when no
        signal matches.
        """
        explicit = detect_explicit(signals, self.name, aliases=self._PREFIXES)

        # Codex JWT detection — unique to ChatGPT
        auth = signals.headers.get("authorization", "")
        jwt_result = DetectionResult(provider=self.name, confidence=DetectionConfidence.NONE)
        if _is_codex_jwt(auth):
            jwt_result = DetectionResult(
                provider=self.name,
                confidence=DetectionConfidence.AUTH,
                reason="Authorization header contains Codex JWT (chatgpt_account_id claim)",
                detail={"jwt_claim": "https://api.openai.com/auth.chatgpt_account_id"},
            )

        # ChatGPT-Account-ID header
        header_result = DetectionResult(provider=self.name, confidence=DetectionConfidence.NONE)
        if signals.headers.get("chatgpt-account-id"):
            header_result = DetectionResult(
                provider=self.name,
                confidence=DetectionConfidence.AUTH,
                reason="ChatGPT-Account-ID header present",
            )

        # Model prefix
        model_result = DetectionResult(provider=self.name, confidence=DetectionConfidence.NONE)
        if signals.model and "/" in signals.model:
            prefix = signals.model.split("/", 1)[0].lower()
            if prefix in self._PREFIXES:
                model_result = DetectionResult(
                    provider=self.name,
                    confidence=DetectionConfidence.MODEL,
                    reason=f"model prefix '{prefix}/' identifies ChatGPT/Codex",
                )

        return highest_confidence(self.name, explicit, jwt_result, header_result, model_result)
