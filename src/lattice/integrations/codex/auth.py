"""Codex authentication helpers for LATTICE proxy passthrough.

Codex sends an OpenAI-style Bearer JWT in the Authorization header.
We decode it (no verification — we are a passthrough proxy) to extract
``chatgpt_account_id`` for routing to the correct Codex upstream.
"""

from __future__ import annotations

import base64
import json
from typing import Any


def _decode_openai_bearer_payload(authorization: str) -> dict[str, Any] | None:
    """Decode the JWT payload from an ``Authorization: Bearer <jwt>`` header.

    Returns the payload dict, or *None* if the token is not a JWT or
    cannot be parsed.  No cryptographic verification is performed.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    parts = token.split(".")
    if len(parts) != 3:
        # Not a JWT — could be a plain API key
        return None
    payload_b64 = parts[1]
    # Pad base64 if needed
    padding_needed = 4 - len(payload_b64) % 4
    if padding_needed != 4:
        payload_b64 += "=" * padding_needed
    try:
        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        return json.loads(payload_bytes.decode("utf-8"))
    except Exception:
        return None


def _resolve_codex_routing_headers(
    authorization: str,
    openai_beta: str = "",
    chatgpt_account_id: str = "",
) -> dict[str, str]:
    """Build upstream headers for Codex WebSocket / HTTP passthrough.

    Prefers an explicit ``ChatGPT-Account-ID`` when provided, otherwise
    extracts ``chatgpt_account_id`` from the JWT payload and forwards it as
    ``ChatGPT-Account-ID`` so the upstream Codex endpoint can route correctly.
    """
    headers: dict[str, str] = {}
    if authorization:
        headers["Authorization"] = authorization
    if openai_beta:
        headers["OpenAI-Beta"] = openai_beta

    if chatgpt_account_id:
        headers["ChatGPT-Account-ID"] = chatgpt_account_id
        return headers

    payload = _decode_openai_bearer_payload(authorization)
    if payload is not None:
        account_id = payload.get("chatgpt_account_id")
        if account_id:
            headers["ChatGPT-Account-ID"] = str(account_id)

    return headers


def _is_codex_jwt(authorization: str) -> bool:
    """Return *True* if the Authorization header contains a Codex JWT.

    Heuristic: the decoded payload contains ``chatgpt_account_id``.
    """
    payload = _decode_openai_bearer_payload(authorization)
    if payload is None:
        return False
    return "chatgpt_account_id" in payload
