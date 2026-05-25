from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lattice.integrations.agents.models import AgentConfig
from typing import Protocol, runtime_checkable

import structlog

logger = structlog.get_logger()


class AgentNotInstalledError(Exception):
    """Raised when an integration target (config file, env file, executable) is not present."""


@runtime_checkable
class AgentIntegrationProtocol(Protocol):
    """Stable protocol every integration subclass must satisfy."""

    @property
    def name(self) -> str: ...

    @property
    def proxy_url(self) -> str: ...

    def patch(self, dry_run: bool = False) -> "AgentConfig": ...

    def unpatch(self, dry_run: bool = False) -> "AgentConfig": ...

    def is_patched(self) -> bool: ...

    def doctor(self) -> "AgentDoctorReport": ...


@dataclasses.dataclass(frozen=True, slots=True)
class AgentDoctorReport:
    """Per-agent health matrix for ``lattice doctor``."""

    agent: str
    is_installed: bool
    is_patched_durable: bool
    is_patched_transient: bool
    proxy_reachable: bool
    diagnostic_lines: list[str] = dataclasses.field(default_factory=list)
