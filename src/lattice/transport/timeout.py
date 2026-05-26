"""Per-request timeout resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimeoutPolicy:
    connect: float
    read: float
    write: float
    pool: float

    def for_attempt(self, attempt_no: int) -> float:
        del attempt_no  # same budget per attempt today
        return self.read


class TimeoutResolver:
    def __init__(self, default_seconds: float = 120.0) -> None:
        env = os.environ.get("LATTICE_REQUEST_TIMEOUT")
        if env:
            try:
                default_seconds = float(env)
            except ValueError:
                pass
        self._default = default_seconds

    def resolve(self, ctx: object | None = None) -> TimeoutPolicy:
        del ctx
        return TimeoutPolicy(
            connect=10.0,
            read=self._default,
            write=30.0,
            pool=5.0,
        )
