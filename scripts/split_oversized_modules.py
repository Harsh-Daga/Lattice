#!/usr/bin/env python3
"""Split oversized lattice modules (Phase 12 honesty). Run from repo root."""

from __future__ import annotations

import re
import shutil
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "lattice"

FUTURE = "from __future__ import annotations\n\n"


def read_lines(path: Path) -> list[str]:
    return path.read_text().splitlines(keepends=True)


def write_module(path: Path, header: str, body_lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(body_lines)
    if body and not body.endswith("\n"):
        body += "\n"
    path.write_text(header + body)


def slice_lines(lines: list[str], start: int, end: int) -> list[str]:
    """1-based inclusive start/end."""
    return lines[start - 1 : end]


def replace_compat_imports(content: str) -> str:
    content = content.replace(
        "from lattice.gateway.compat import ",
        "from lattice.gateway.compat.translation import ",
    )
    # WS handler imports deserialize from translation
    return content


def split_compat() -> None:
    src = SRC / "gateway" / "compat.py"
    lines = read_lines(src)
    pkg = SRC / "gateway" / "compat"
    pkg.mkdir(exist_ok=True)

    common_imports = textwrap.dedent(
        """\
        import asyncio
        import contextlib
        import dataclasses
        import json
        import time
        from collections.abc import AsyncIterator, Awaitable, Callable
        from typing import Any

        import httpx
        from fastapi import status
        from fastapi.responses import JSONResponse, StreamingResponse
        from starlette.responses import Response as StarletteResponse

        from lattice.cache.semantic import assemble_cached_response, compute_cache_key
        from lattice.core.context import TransformContext
        from lattice.core.result import is_err, unwrap
        from lattice.gateway.server import LLMTPGateway
        from lattice.pipeline.factory import pipeline_summary
        from lattice.planner.runtime_state import (
            get_canonical_request_value,
            persist_execution_plan_state,
            persist_session_plan_state,
            sum_expected_cached_tokens,
        )
        from lattice.protocol.manifest import manifest_summary
        from lattice.providers.capabilities import Capability, get_capability_registry
        from lattice.proxy.middleware import attach_routing_headers, stash_lattice_response_headers
        from lattice.telemetry.agent_stats import identify_agent
        from lattice.telemetry.cost_estimator import normalize_usage
        from lattice.telemetry.downgrade import TransportOutcome
        from lattice.transport.serialization import message_to_dict, request_from_dict, response_to_dict
        from lattice.transport.types import Message, Request, Response

        Handler = Callable[..., Awaitable[Any]]

        """
    )

    chunks: list[tuple[str, int, int, str]] = [
        ("handler.py", 41, 82, ""),
        (
            "providers.py",
            84,
            204,
            "from lattice.gateway.compat.providers import _WELL_KNOWN_PROVIDER_URLS\n",
        ),
        ("headers.py", 206, 377, ""),
        ("translation.py", 379, 417, ""),
        ("anthropic_messages.py", 419, 710, ""),
        ("responses_body.py", 711, 783, ""),
        ("responses_passthrough.py", 784, 1083, ""),
        ("anthropic_passthrough.py", 1084, 1371, ""),
        ("openai_chat.py", 1372, 2489, ""),
        ("anthropic_handler.py", 2491, 2723, ""),
        ("responses_handler.py", 2724, 2925, ""),
        ("operational.py", 2927, len(lines), ""),
    ]

    for fname, start, end, extra in chunks:
        body = slice_lines(lines, start, end)
        header = FUTURE + '"""Gateway compat — split from monolithic compat.py."""\n\n'
        if fname != "handler.py":
            header += common_imports
        if fname == "providers.py":
            header += extra
        elif fname in ("openai_chat.py", "anthropic_handler.py", "responses_handler.py"):
            header += (
                "from lattice.gateway.compat.headers import build_routing_headers\n"
                "from lattice.gateway.compat.providers import _WELL_KNOWN_PROVIDER_URLS\n"
                "from lattice.gateway.compat.translation import (\n"
                "    deserialize_anthropic_request,\n"
                "    deserialize_openai_request,\n"
                "    detect_new_messages,\n"
                "    serialize_messages,\n"
                "    serialize_openai_response,\n"
                ")\n"
            )
        if fname == "anthropic_handler.py":
            header += "from lattice.gateway.compat.anthropic_passthrough import anthropic_passthrough\n"
        if fname == "responses_handler.py":
            header += "from lattice.gateway.compat.responses_passthrough import responses_passthrough\n"
        if fname == "openai_chat.py":
            header += "from lattice.gateway.compat.translation import deserialize_openai_request\n"
        if fname == "anthropic_passthrough.py":
            header += (
                "from lattice.gateway.compat.anthropic_messages import (\n"
                "    deserialize_anthropic_request,\n"
                "    serialize_anthropic_response,\n"
                ")\n"
                "from lattice.gateway.compat.headers import build_routing_headers\n"
                "from lattice.gateway.compat.providers import (\n"
                "    _prepare_codex_upstream_headers,\n"
                "    _resolve_provider_upstream_url,\n"
                "    _resolve_passthrough_provider,\n"
                "    _WELL_KNOWN_PROVIDER_URLS,\n"
                ")\n"
                "from lattice.gateway.compat.translation import is_local_origin\n"
            )
        if fname == "responses_passthrough.py":
            header += (
                "from lattice.gateway.compat.headers import build_routing_headers\n"
                "from lattice.gateway.compat.providers import (\n"
                "    _prepare_codex_upstream_headers,\n"
                "    _resolve_provider_upstream_url,\n"
                "    _resolve_passthrough_provider,\n"
                "    _WELL_KNOWN_PROVIDER_URLS,\n"
                ")\n"
                "from lattice.gateway.compat.responses_body import (\n"
                "    compress_responses_body,\n"
                "    extract_responses_text_blocks,\n"
                "    replace_responses_text_blocks,\n"
                ")\n"
            )
        content = "".join(body)
        content = content.replace(
            "from lattice.gateway.compat import deserialize_openai_request",
            "from lattice.gateway.compat.translation import deserialize_openai_request",
        )
        write_module(pkg / fname, header, [content])

    # handler needs minimal imports
    handler_header = (
        FUTURE
        + '"""HTTP compat handler delegation."""\n\n'
        + "from typing import Any\n\n"
        + "from lattice.gateway.server import LLMTPGateway\n\n"
        + "Handler = Any  # re-exported from compat package __init__\n\n"
    )
    write_module(pkg / "handler.py", handler_header, slice_lines(lines, 41, 82))

    # Fix handler to use proper Handler type from __init__
    handler_text = (pkg / "handler.py").read_text()
    handler_text = handler_text.replace("Handler = Any  # re-exported", "")
    handler_text = handler_text.replace(
        "Handler | None",
        "Callable[..., Any] | None",
    )
    handler_text = handler_text.replace(
        "from typing import Any\n\nfrom lattice.gateway.server",
        "from collections.abc import Awaitable, Callable\nfrom typing import Any\n\nfrom lattice.gateway.server",
    )
    (pkg / "handler.py").write_text(handler_text)

    # providers chunk includes constants - export them
    prov = (pkg / "providers.py").read_text()
    if "_WELL_KNOWN_PROVIDER_URLS" not in prov.split("def ")[0]:
        pass

    # Build __init__.py re-exports
    init_body = textwrap.dedent(
        '''\
        """HTTP compatibility — public surface preserved at lattice.gateway.compat."""

        from lattice.gateway.compat.anthropic_handler import (
            AnthropicCompatDeps,
            make_anthropic_handler,
        )
        from lattice.gateway.compat.anthropic_messages import (
            compress_anthropic_body,
            deserialize_anthropic_request,
            deserialize_anthropic_response,
            extract_anthropic_text_blocks,
            replace_anthropic_text_blocks,
            serialize_anthropic_response,
        )
        from lattice.gateway.compat.anthropic_passthrough import anthropic_passthrough
        from lattice.gateway.compat.handler import HTTPCompatHandler
        from lattice.gateway.compat.headers import (
            _extract_cached_tokens,
            _runtime_header_values,
            _usage_total_tokens,
            build_routing_headers,
        )
        from lattice.gateway.compat.openai_chat import (
            ChatCompatDeps,
            chat_completions_websocket_passthrough,
            make_chat_completion_handler,
        )
        from lattice.gateway.compat.operational import (
            OperationalRouteDeps,
            build_proxy_stats_payload,
            register_operational_routes,
        )
        from lattice.gateway.compat.providers import (
            _PROVIDER_FALLBACK_BASE_URLS,
            _WELL_KNOWN_PROVIDER_URLS,
            _prepare_codex_upstream_headers,
            _resolve_passthrough_provider,
            _resolve_provider_upstream_url,
        )
        from lattice.gateway.compat.responses_body import (
            compress_responses_body,
            extract_responses_text_blocks,
            replace_responses_text_blocks,
        )
        from lattice.gateway.compat.responses_handler import (
            ResponsesCompatDeps,
            make_models_handler,
            make_responses_handler,
            models_passthrough,
        )
        from lattice.gateway.compat.responses_passthrough import (
            responses_passthrough,
            responses_websocket_passthrough,
        )
        from lattice.gateway.compat.translation import (
            deserialize_openai_request,
            detect_new_messages,
            is_local_origin,
            serialize_messages,
            serialize_openai_response,
        )

        Handler = __import__(
            "collections.abc", fromlist=["Callable"]
        ).Callable[..., __import__("typing").Awaitable[__import__("typing").Any]]

        __all__ = [
            "AnthropicCompatDeps",
            "ChatCompatDeps",
            "Handler",
            "HTTPCompatHandler",
            "OperationalRouteDeps",
            "ResponsesCompatDeps",
            "_PROVIDER_FALLBACK_BASE_URLS",
            "_WELL_KNOWN_PROVIDER_URLS",
            "_extract_cached_tokens",
            "_prepare_codex_upstream_headers",
            "_resolve_passthrough_provider",
            "_resolve_provider_upstream_url",
            "_runtime_header_values",
            "_usage_total_tokens",
            "anthropic_passthrough",
            "build_proxy_stats_payload",
            "build_routing_headers",
            "chat_completions_websocket_passthrough",
            "compress_anthropic_body",
            "compress_responses_body",
            "deserialize_anthropic_request",
            "deserialize_anthropic_response",
            "deserialize_openai_request",
            "detect_new_messages",
            "extract_anthropic_text_blocks",
            "extract_responses_text_blocks",
            "is_local_origin",
            "make_anthropic_handler",
            "make_chat_completion_handler",
            "make_models_handler",
            "make_responses_handler",
            "models_passthrough",
            "register_operational_routes",
            "replace_anthropic_text_blocks",
            "replace_responses_text_blocks",
            "responses_passthrough",
            "responses_websocket_passthrough",
            "serialize_anthropic_response",
            "serialize_messages",
            "serialize_openai_response",
        ]
        '''
    )
    (pkg / "__init__.py").write_text(FUTURE + init_body)
    src.unlink()


def split_cache_semantic() -> None:
    src = SRC / "cache" / "semantic.py"
    lines = read_lines(src)
    base_header = FUTURE + '"""Semantic cache package — split modules."""\n\n'

    # fingerprint: lines 47-539 (through _compute_similarity)
    fp_header = base_header + (
        "import dataclasses\nimport enum\nimport hashlib\nimport json\nimport re\n"
        "import time\nfrom dataclasses import dataclass, field\nfrom typing import Any\n\n"
        "import structlog\n\n"
    )
    write_module(SRC / "cache" / "fingerprint.py", fp_header, slice_lines(lines, 47, 539))

    # stores: backends 101-289 + assemble helpers at end
    stores_header = base_header + textwrap.dedent(
        """\
        import asyncio
        import time
        from collections import OrderedDict
        from collections.abc import Callable
        from dataclasses import dataclass, field
        from typing import Any, Protocol

        from lattice.cache.fingerprint import CachedResponse, ContentClass

        """
    )
    write_module(SRC / "cache" / "stores.py", stores_header, slice_lines(lines, 101, 289))

    # eviction + semantic cache class
    sem_header = base_header + textwrap.dedent(
        """\
        import asyncio
        import time
        from collections import OrderedDict
        from typing import Any

        import structlog

        from lattice.cache.fingerprint import (
            ContentClass,
            CachedResponse,
            _FingerprintEntry,
            _SemanticFingerprint,
            _compute_semantic_fingerprint,
            _compute_similarity,
            _detect_content_class,
            compute_cache_key,
        )
        from lattice.cache.stores import CacheBackend, InMemoryCacheBackend, RedisCacheBackend

        _logger = structlog.get_logger()

        """
    )
    write_module(SRC / "cache" / "semantic_core.py", sem_header, slice_lines(lines, 541, 1014))

    init = textwrap.dedent(
        '''\
        """Hybrid response cache — public imports unchanged."""

        from lattice.cache.fingerprint import (
            ContentClass,
            CachedResponse,
            compute_cache_key,
        )
        from lattice.cache.semantic_core import SemanticCache, assemble_cached_response, generate_sse_chunks
        from lattice.cache.stores import CacheBackend, InMemoryCacheBackend, RedisCacheBackend

        __all__ = [
            "CacheBackend",
            "CachedResponse",
            "ContentClass",
            "InMemoryCacheBackend",
            "RedisCacheBackend",
            "SemanticCache",
            "assemble_cached_response",
            "compute_cache_key",
            "generate_sse_chunks",
        ]
        '''
    )
    write_module(SRC / "cache" / "semantic.py", FUTURE + init, [])
    # semantic.py becomes shim - overwrite with re-exports only


def split_ir_builder() -> None:
    src = SRC / "ir" / "builder.py"
    lines = read_lines(src)
    pkg = SRC / "ir" / "builder"
    pkg.mkdir(exist_ok=True)

    regex_header = FUTURE + '"""IR builder regex and keyword tables."""\n\nimport re\n\n'
    write_module(pkg / "_patterns.py", regex_header, slice_lines(lines, 26, 220))

    core_header = FUTURE + textwrap.dedent(
        '''\
        """IR builder entry points."""

        from lattice.ir.builder._patterns import *  # noqa: F403
        from lattice.ir.types import PromptIR, Section, SectionType, Span, SpanRole
        from lattice.transport.types import Message, Request

        '''
    )
    write_module(pkg / "core.py", core_header, slice_lines(lines, 222, 444))

    partition_header = FUTURE + textwrap.dedent(
        '''\
        """Message partitioning helpers."""

        import json as _json

        from lattice.ir.builder._patterns import *  # noqa: F403
        from lattice.ir.types import Section, SectionType, Span, SpanRole

        '''
    )
    write_module(pkg / "partition.py", partition_header, slice_lines(lines, 445, 654))

    analyze_header = FUTURE + textwrap.dedent(
        '''\
        """Span analysis and protection."""

        import json as _json

        from lattice.ir.builder._patterns import *  # noqa: F403
        from lattice.ir.types import Section, SectionType, Span, SpanRole

        '''
    )
    write_module(pkg / "analyze.py", analyze_header, slice_lines(lines, 655, 821))

    init = textwrap.dedent(
        '''\
        from lattice.ir.builder.analyze import is_repeated_template
        from lattice.ir.builder.core import build_ir, compile_request_ir
        from lattice.ir.builder.partition import *  # noqa: F403

        __all__ = ["build_ir", "compile_request_ir", "is_repeated_template"]
        '''
    )
    (pkg / "__init__.py").write_text(FUTURE + init)
    src.unlink()


def split_anthropic_adapter() -> None:
    src = SRC / "providers" / "adapters" / "anthropic.py"
    lines = read_lines(src)
    pkg = SRC / "providers" / "adapters" / "anthropic"
    pkg.mkdir(exist_ok=True)

    doc = slice_lines(lines, 1, 37)
    ctx_header = FUTURE + "".join(doc) + "\nimport contextvars\n\n"
    write_module(pkg / "context.py", ctx_header, slice_lines(lines, 57, 65))

    base_header = FUTURE + "".join(doc) + textwrap.dedent(
        """\
        import json
        from typing import Any

        from lattice.planner.runtime_state import get_canonical_request_value
        from lattice.providers.mcp_to_anthropic import convert_mcp_to_anthropic, is_mcp_tool
        from lattice.providers.schema_filter import sanitize_json_schema, sanitize_tool_definitions
        from lattice.providers.stream_state import AnthropicStreamState
        from lattice.providers.tool_sanitizer import (
            AnthropicToolSanitizer,
            restore_tool_call_ids,
            sanitize_tool_ids,
        )
        from lattice.transport.types import Request, Response

        from lattice.providers.adapters.base import _pop_system, _remap_tool_choice, _remap_tools
        from lattice.providers.adapters.anthropic.context import _ctx_tool_id_mapping

        """
    )
    write_module(pkg / "core.py", base_header, slice_lines(lines, 72, 245))
    write_module(pkg / "serialization.py", base_header, slice_lines(lines, 246, 657))
    write_module(pkg / "deserialization.py", base_header, slice_lines(lines, 658, 756))
    write_module(pkg / "streaming.py", base_header, slice_lines(lines, 757, 813))

    init = textwrap.dedent(
        '''\
        from lattice.providers.adapters.anthropic.context import _ctx_tool_id_mapping
        from lattice.providers.adapters.anthropic.core import AnthropicAdapterCore
        from lattice.providers.adapters.anthropic.deserialization import AnthropicDeserializeMixin
        from lattice.providers.adapters.anthropic.serialization import AnthropicSerializeMixin
        from lattice.providers.adapters.anthropic.streaming import AnthropicStreamMixin


        class AnthropicAdapter(
            AnthropicStreamMixin,
            AnthropicDeserializeMixin,
            AnthropicSerializeMixin,
            AnthropicAdapterCore,
        ):
            """Anthropic Messages API adapter with full Claude Code parity."""

            name = "anthropic"


        __all__ = ["AnthropicAdapter", "_ctx_tool_id_mapping"]
        '''
    )
    (pkg / "__init__.py").write_text(FUTURE + init)
    # Rename classes in split files to mixins
    for fname, cls in [
        ("core.py", "AnthropicAdapterCore"),
        ("serialization.py", "AnthropicSerializeMixin"),
        ("deserialization.py", "AnthropicDeserializeMixin"),
        ("streaming.py", "AnthropicStreamMixin"),
    ]:
        p = pkg / fname
        t = p.read_text()
        t = t.replace("class AnthropicAdapter:", f"class {cls}:")
        t = t.replace("AnthropicAdapter.", f"{cls}.")
        p.write_text(t)
    src.unlink()


def split_agents() -> None:
    src = SRC / "integrations" / "agents.py"
    lines = read_lines(src)
    pkg = SRC / "integrations" / "agents"
    pkg.mkdir(exist_ok=True)

    shared_imports = slice_lines(lines, 66, 90)
    header = FUTURE + '"""Agent integrations package."""\n\n' + "".join(shared_imports)

    write_module(pkg / "protocol.py", header, slice_lines(lines, 94, 128))
    write_module(
        pkg / "doctor.py",
        header,
        slice_lines(lines, 129, 181) + ["\n", "from lattice.integrations.agents.protocol import AgentDoctorReport\n"],
    )
    write_module(pkg / "models.py", header, slice_lines(lines, 188, 232))
    write_module(pkg / "base.py", header + "from lattice.integrations.agents.models import AgentConfig\n", slice_lines(lines, 239, 289))
    write_module(
        pkg / "env_builder.py",
        header
        + "from lattice.integrations.agents.base import AgentIntegration\n"
        + "from lattice.integrations.agents.models import AgentConfig\n",
        slice_lines(lines, 290, 837),
    )
    write_module(
        pkg / "profiles.py",
        header
        + "from lattice.integrations.agents.base import AgentIntegration\n"
        + "from lattice.integrations.agents.env_builder import EnvFileIntegration, JsonFileIntegration\n"
        + "from lattice.integrations.agents.models import AgentConfig\n",
        slice_lines(lines, 838, 1540),
    )
    write_module(
        pkg / "registry.py",
        header
        + "from lattice.core.config import LatticeConfig\n"
        + "from lattice.integrations.agents.base import AgentIntegration\n"
        + "from lattice.integrations.agents.env_builder import (\n"
        + "    ClaudeCodeIntegration,\n"
        + "    CodexIntegration,\n"
        + "    EnvFileIntegration,\n"
        + "    GenericIntegration,\n"
        + "    JsonFileIntegration,\n"
        + "    VSCodeIntegration,\n"
        + ")\n"
        + "from lattice.integrations.agents.models import AgentConfig\n"
        + "from lattice.integrations.agents.profiles import (\n"
        + "    CopilotIntegration,\n"
        + "    CursorIntegration,\n"
        + "    OpenCodeIntegration,\n"
        + ")\n",
        slice_lines(lines, 1541, 1562),
    )
    write_module(
        pkg / "lifecycle.py",
        header
        + "from lattice.core.config import LatticeConfig\n"
        + "from lattice.integrations.agents.models import AgentConfig\n"
        + "from lattice.integrations.agents.protocol import AgentNotInstalledError\n"
        + "from lattice.integrations.agents.registry import _AGENT_REGISTRY, list_agents\n"
        + "from lattice.integrations.agents.env_builder import EnvFileIntegration, JsonFileIntegration\n",
        slice_lines(lines, 1563, 1690),
    )

    init = (SRC / "integrations" / "agents" / "__init__.py")
    init.write_text(
        FUTURE
        + textwrap.dedent(
            '''\
            from lattice.integrations.agents.base import AgentIntegration
            from lattice.integrations.agents.doctor import build_agent_doctor_report, list_primary_agents
            from lattice.integrations.agents.env_builder import (
                ClaudeCodeIntegration,
                CodexIntegration,
                EnvFileIntegration,
                GenericIntegration,
                JsonFileIntegration,
                VSCodeIntegration,
            )
            from lattice.integrations.agents.lifecycle import (
                agent_status,
                unwrap_agent,
                unwrap_all,
                wrap_agent,
                wrap_all,
            )
            from lattice.integrations.agents.models import AgentConfig
            from lattice.integrations.agents.profiles import (
                CopilotIntegration,
                CursorIntegration,
                OpenCodeIntegration,
            )
            from lattice.integrations.agents.protocol import (
                AgentDoctorReport,
                AgentIntegrationProtocol,
                AgentNotInstalledError,
            )
            from lattice.integrations.agents.registry import _AGENT_REGISTRY, get_agent_integration, list_agents

            __all__ = [
                "AgentConfig",
                "AgentDoctorReport",
                "AgentIntegration",
                "AgentIntegrationProtocol",
                "AgentNotInstalledError",
                "ClaudeCodeIntegration",
                "CodexIntegration",
                "CopilotIntegration",
                "CursorIntegration",
                "EnvFileIntegration",
                "GenericIntegration",
                "JsonFileIntegration",
                "OpenCodeIntegration",
                "VSCodeIntegration",
                "_AGENT_REGISTRY",
                "agent_status",
                "build_agent_doctor_report",
                "get_agent_integration",
                "list_agents",
                "list_primary_agents",
                "unwrap_agent",
                "unwrap_all",
                "wrap_agent",
                "wrap_all",
            ]
            '''
        )
    )
    src.unlink()


def split_cli() -> None:
    src = SRC / "cli.py"
    lines = read_lines(src)
    pkg = SRC / "cli"
    pkg.mkdir(exist_ok=True)

    shared = slice_lines(lines, 1, 98)
    write_module(pkg / "_console.py", FUTURE + "".join(shared), [])

    write_module(
        pkg / "proxy_cmds.py",
        FUTURE + 'from lattice.cli._console import console, logger\n\n',
        slice_lines(lines, 184, 494),
    )
    write_module(
        pkg / "init_cmds.py",
        FUTURE + 'from lattice.cli._console import console, logger\n\n',
        slice_lines(lines, 495, 746),
    )
    write_module(
        pkg / "info_cmds.py",
        FUTURE + 'from lattice.cli._console import console, logger\n\n',
        slice_lines(lines, 747, 949),
    )
    write_module(
        pkg / "doctor_cmds.py",
        FUTURE + 'from lattice.cli._console import console, logger\n\n',
        slice_lines(lines, 950, 1041),
    )

    init = textwrap.dedent(
        '''\
        """Command-line interface for LATTICE."""

        import sys
        from typing import Any

        from lattice._version import __version__
        from lattice.cli._console import _get_config, _get_pid_mgr, _lace_agent, _list_agents, _list_mutated_agents, _print_banner, _run_init, _unlace_agent, console, logger
        from lattice.cli.doctor_cmds import _cmd_agent_status, _cmd_doctor, _print_doctor_report
        from lattice.cli.info_cmds import _cmd_benchmark, _cmd_config, _cmd_health, _cmd_info, _enabled_transforms
        from lattice.cli.init_cmds import _cmd_init, _cmd_lace, _cmd_unlace, _format_init_empty_error, _print_init_help, _print_lace_help, _print_unlace_help
        from lattice.cli.proxy_cmds import _cmd_proxy, _cmd_proxy_restart, _cmd_proxy_run, _cmd_proxy_start, _cmd_proxy_status, _cmd_proxy_stop, _parse_proxy_args, _print_proxy_help, _start_background


        def _print_help() -> None:
            from rich.panel import Panel

            console.print(
                Panel.fit(
                    "[bold]lattice[/bold] — LLM transport proxy\\n\\n"
                    "  lattice proxy start|stop|status|run\\n"
                    "  lattice init [agent]\\n"
                    "  lattice lace <agent>\\n"
                    "  lattice unlace <agent>\\n"
                    "  lattice info | config | benchmark | health\\n"
                    "  lattice status | doctor",
                    title="Usage",
                )
            )


        def main() -> None:
            args = sys.argv[1:]
            if not args or args[0] in ("-h", "--help", "help"):
                _print_help()
                return
            if args[0] in ("-v", "--version", "version"):
                console.print(f"lattice {__version__}")
                return

            cmd = args[0]
            cmd_args = args[1:]

            if cmd == "proxy":
                _cmd_proxy(cmd_args)
            elif cmd == "init":
                _cmd_init(cmd_args)
            elif cmd == "lace":
                _cmd_lace(cmd_args)
            elif cmd == "unlace":
                _cmd_unlace(cmd_args)
            elif cmd == "info":
                _cmd_info(cmd_args)
            elif cmd == "config":
                _cmd_config(cmd_args)
            elif cmd == "benchmark":
                _cmd_benchmark(cmd_args)
            elif cmd == "health":
                _cmd_health(cmd_args)
            elif cmd == "status":
                _cmd_agent_status(cmd_args)
            elif cmd == "doctor":
                _cmd_doctor(cmd_args)
            else:
                console.print(f"[red]Unknown command: {cmd}[/red]")
                _print_help()
                sys.exit(1)


        __all__ = ["main"]
        '''
    )
    (pkg / "__init__.py").write_text(FUTURE + init)
    src.unlink()


def main() -> None:
    split_compat()
    split_cache_semantic()
    split_ir_builder()
    split_anthropic_adapter()
    split_agents()
    split_cli()
    print("Splits done — run ruff and pytest next.")


if __name__ == "__main__":
    main()
