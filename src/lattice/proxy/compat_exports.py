"""Compatibility exports kept for proxy import stability."""

from __future__ import annotations

from typing import Any

import structlog

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.pipeline import CompressorPipeline
from lattice.core.errors import ProviderDetectionError as CompatProviderDetectionError
from lattice.gateway.compat import (
    build_routing_headers as compat_build_routing_headers,
)
from lattice.gateway.compat import (
    compress_anthropic_body as compat_compress_anthropic_body,
)
from lattice.gateway.compat import (
    compress_responses_body as compat_compress_responses_body,
)
from lattice.gateway.compat import (
    deserialize_anthropic_request as compat_deserialize_anthropic_request,
)
from lattice.gateway.compat import (
    deserialize_openai_request as compat_deserialize_openai_request,
)
from lattice.gateway.compat import (
    detect_new_messages as compat_detect_new_messages,
)
from lattice.gateway.compat import (
    extract_anthropic_text_blocks as compat_extract_anthropic_text_blocks,
)
from lattice.gateway.compat import (
    extract_responses_text_blocks as compat_extract_responses_text_blocks,
)
from lattice.gateway.compat import (
    is_local_origin as compat_is_local_origin,
)
from lattice.gateway.compat import (
    replace_anthropic_text_blocks as compat_replace_anthropic_text_blocks,
)
from lattice.gateway.compat import (
    replace_responses_text_blocks as compat_replace_responses_text_blocks,
)
from lattice.gateway.compat import (
    serialize_anthropic_response as compat_serialize_anthropic_response,
)
from lattice.gateway.compat import (
    serialize_messages as compat_serialize_messages,
)
from lattice.gateway.compat import (
    serialize_openai_response as compat_serialize_openai_response,
)

logger = structlog.get_logger()

ProviderDetectionError = CompatProviderDetectionError

_build_routing_headers = compat_build_routing_headers
_is_local_origin = compat_is_local_origin
_detect_new_messages = compat_detect_new_messages
_deserialize_openai_request = compat_deserialize_openai_request
_serialize_messages = compat_serialize_messages
_serialize_openai_response = compat_serialize_openai_response
_deserialize_anthropic_request = compat_deserialize_anthropic_request
_serialize_anthropic_response = compat_serialize_anthropic_response
_extract_anthropic_text_blocks = compat_extract_anthropic_text_blocks
_replace_anthropic_text_blocks = compat_replace_anthropic_text_blocks
_extract_responses_text_blocks = compat_extract_responses_text_blocks
_replace_responses_text_blocks = compat_replace_responses_text_blocks


async def _compress_anthropic_body(
    body: dict[str, Any],
    pipeline: CompressorPipeline,
    config: LatticeConfig,
    provider_name: str,
    model: str,
) -> tuple[dict[str, Any], TransformContext, int, int]:
    return await compat_compress_anthropic_body(
        body,
        pipeline,
        config,
        provider_name,
        model,
        logger,
    )


async def _compress_responses_body(
    body: dict[str, Any],
    pipeline: CompressorPipeline,
    config: LatticeConfig,
    model: str,
) -> tuple[dict[str, Any], TransformContext, int, int]:
    return await compat_compress_responses_body(
        body,
        pipeline,
        config,
        model,
        logger,
    )
