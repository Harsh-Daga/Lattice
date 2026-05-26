"""Shared HTTP/SSE helpers for provider transport."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from lattice.core.errors import ProviderError
from lattice.planner.runtime_state import get_canonical_request_value
from lattice.providers.adapters import ProviderAdapter
from lattice.transport.types import Request

if TYPE_CHECKING:
    pass


def should_retry(status_code: int, retry_on: tuple[int, ...]) -> bool:
    return status_code in retry_on


def stream_retry_policy(
    provider_name: str,
    metadata: dict[str, Any],
) -> tuple[float | None, int]:
    ttft_raw = metadata.get("ttft_timeout_seconds")
    retries_raw = metadata.get("no_first_chunk_retries")
    if ttft_raw is None:
        ttft_timeout: float | None = 8.0 if provider_name == "ollama-cloud" else None
    else:
        ttft_timeout = float(ttft_raw)
    if retries_raw is None:
        retries = 1 if provider_name == "ollama-cloud" else 0
    else:
        retries = max(0, int(retries_raw))
    return ttft_timeout, retries


def tacc_reservation(request: Any) -> int:
    prompt = max(1, int(getattr(request, "token_estimate", 0) or 0))
    completion = max(0, int(getattr(request, "max_tokens", 0) or 0))
    return max(1, prompt + completion)


def stream_chunk_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if isinstance(delta, dict):
        for key in ("content", "reasoning_content", "reasoning"):
            value = delta.get(key)
            if isinstance(value, str) and value:
                return value
    message = first.get("message")
    if isinstance(message, dict):
        for key in ("content", "reasoning_content", "reasoning"):
            value = message.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


async def next_stream_line(
    iterator: Any,
    *,
    deadline_monotonic: float | None,
) -> str:
    if deadline_monotonic is None:
        return await iterator.__anext__()  # type: ignore[no-any-return]
    remaining = deadline_monotonic - time.perf_counter()
    if remaining <= 0:
        raise TimeoutError
    return await asyncio.wait_for(iterator.__anext__(), timeout=remaining)


def process_sse_line_with_state(line: str, state: Any) -> list[dict[str, Any]]:
    line = line.strip()
    if not line or line.startswith(":"):
        return []
    if not line.startswith("data: "):
        return []
    payload = line[len("data: ") :]
    if payload.strip() == "[DONE]":
        return [{"choices": [], "done": True}]
    try:
        data: dict[str, Any] = __import__("json").loads(payload)
    except Exception:
        return []
    result = state.process(data)
    out: list[dict[str, Any]] = list(result.chunks)
    if result.done and out:
        out[-1]["done"] = True
    if result.done and result.metadata and out:
        out[-1].setdefault("_lattice_metadata", {}).update(result.metadata)
    return out


def parse_sse_line(line: str, adapter: ProviderAdapter) -> dict[str, Any] | None:
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data: "):
        return None
    payload = line[len("data: ") :]
    if payload.strip() == "[DONE]":
        return {"choices": [], "done": True}
    try:
        data: dict[str, Any] = __import__("json").loads(payload)
    except Exception:
        return None
    return adapter.normalize_sse_chunk(data)


def optimize_stream_chunk(
    chunk: dict[str, Any],
    optimizer: Any | None,
    state: Any | None,
) -> list[dict[str, Any]]:
    if optimizer is None or state is None:
        return [chunk]
    optimized_chunk, emit_done = optimizer.process_chunk(chunk, state)
    out: list[dict[str, Any]] = []
    if optimized_chunk is not None:
        out.append(optimized_chunk)
    if emit_done:
        out.append({"choices": [], "done": True})
    return out


def build_request(
    model: str,
    messages: list[dict[str, Any]],
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    stop: list[str] | str | None = None,
    stream: bool = False,
    metadata: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> Request:
    from lattice.transport.serialization import message_from_dict

    resolved_stop: list[str] | None = None
    if isinstance(stop, str):
        resolved_stop = [stop]
    elif stop is not None:
        resolved_stop = list(stop)

    return Request(
        messages=[message_from_dict(msg) for msg in messages],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        tools=tools,
        tool_choice=tool_choice,
        stream=stream,
        stop=resolved_stop,
        metadata=metadata or {},
        extra_headers=extra_headers or {},
        extra_body=extra_body or {},
    )


def resolve_base_url(
    provider: Any,
    provider_name: str,
    api_base: str | None = None,
) -> str:
    base_url = api_base or provider.provider_base_urls.get(provider_name)
    if not base_url:
        try:
            adapter = provider.registry.get_adapter(provider_name)
            base_url = getattr(adapter, "_DEFAULT_BASE_URL", "") or ""
        except ProviderError:
            pass
    if not base_url:
        from lattice.gateway.compat import _WELL_KNOWN_PROVIDER_URLS

        base_url = _WELL_KNOWN_PROVIDER_URLS.get(provider_name, "")
    if not base_url:
        raise ProviderError(
            provider=provider_name,
            status_code=400,
            message=(
                f"No base URL configured for provider '{provider_name}'. "
                f"Set it via provider_base_urls['{provider_name}'] or "
                f"LATTICE_PROVIDER_BASE_URLS env var."
            ),
        )
    return base_url


def resolve_api_key(
    provider: Any,
    provider_name: str,
    api_key: str | None = None,
) -> str:
    if api_key is not None:
        return api_key
    if provider._credentials is not None:
        creds = provider._credentials.resolve(provider_name)
        resolved_key: str | None = creds.api_key
        if resolved_key is not None:
            return resolved_key
    if provider.default_api_key is not None:
        return provider.default_api_key
    if provider_name in ("ollama", "bedrock"):
        return ""
    from lattice.providers.credentials import _PROVIDER_ENV_VARS

    env_var = _PROVIDER_ENV_VARS.get(provider_name, {}).get(
        "api_key", f"{provider_name.upper().replace('-', '_')}_API_KEY"
    )
    raise ProviderError(
        provider=provider_name,
        status_code=401,
        message=(
            f"No API key found for provider '{provider_name}'. "
            f"Set it via: 1) api_key parameter, 2) lattice config file "
            f"(~/.config/lattice/lattice.config.toml), or 3) environment variable "
            f"({env_var})"
        ),
    )


async def await_tacc_admission(provider: Any, provider_name: str, request: Request) -> bool:
    estimate = tacc_reservation(request)
    priority = int(request.metadata.get("tacc_priority", 0) or 0)
    cache_hit_expected = bool(
        get_canonical_request_value(request, None, "_lattice_cache_hit_expected", False)
    )
    is_speculative = bool(
        get_canonical_request_value(request, None, "_lattice_is_speculative", False)
    )
    is_batch = bool(get_canonical_request_value(request, None, "_lattice_is_batch", False))

    decision, _reason = provider.tacc.evaluate_admission(
        provider_name,
        estimate,
        priority,
        cache_hit_expected=cache_hit_expected,
        is_batch=is_batch,
        is_speculative=is_speculative,
    )
    if decision.value == "reject":
        return False
    if decision.value == "priority_downgrade":
        priority = max(0, priority - 2)

    return await provider.tacc.acquire_request(
        provider_name,
        estimated_tokens=estimate,
        priority=priority,
    )
