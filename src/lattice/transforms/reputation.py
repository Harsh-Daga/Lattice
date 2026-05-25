"""Transform reputation system — rolling stats for quality/compression/rollback.

Maintains rolling window statistics per transform to drive automatic
enablement/disable decisions and post-transform guard triggering.
"""

from __future__ import annotations

import dataclasses
import threading
from collections import deque
from typing import Any

_MAX_HISTORY = 1000


@dataclasses.dataclass(slots=True)
class TransformStats:
    quality_avg: float = 1.0
    compression_avg: float = 0.0
    rollback_rate: float = 0.0
    sample_count: int = 0
    risk: str = "LOW"

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_avg": round(self.quality_avg, 3),
            "compression_avg": round(self.compression_avg, 3),
            "rollback_rate": round(self.rollback_rate, 3),
            "sample_count": self.sample_count,
            "risk": self.risk,
        }


class TransformReputation:
    __slots__ = ("_lock", "_quality", "_compression", "_rollbacks", "_total", "_stats")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._quality: deque[float] = deque(maxlen=_MAX_HISTORY)
        self._compression: deque[float] = deque(maxlen=_MAX_HISTORY)
        self._rollbacks: deque[bool] = deque(maxlen=_MAX_HISTORY)
        self._total = 0

    def record(
        self,
        quality: float,
        compression: float,
        rolled_back: bool = False,
    ) -> None:
        with self._lock:
            self._quality.append(quality)
            self._compression.append(compression)
            self._rollbacks.append(rolled_back)
            self._total += 1

    def stats(self) -> TransformStats:
        with self._lock:
            n = len(self._quality)
            if n == 0:
                return TransformStats(sample_count=self._total)
            q_avg = sum(self._quality) / n
            c_avg = sum(self._compression) / n
            rb_rate = sum(1 for r in self._rollbacks if r) / n

            risk = "LOW"
            # Require at least 5 samples before declaring HIGH risk.
            # Transforms with 1-2 rollback samples would get rollback_rate=1.00
            # and be permanently blocked, which is wrong for startup.
            if n >= 5 and rb_rate > 0.25:
                risk = "HIGH"
            elif rb_rate > 0.10 or q_avg < 0.85:
                risk = "MEDIUM"

            return TransformStats(
                quality_avg=q_avg,
                compression_avg=c_avg,
                rollback_rate=rb_rate,
                sample_count=self._total,
                risk=risk,
            )


class ReputationRegistry:
    __slots__ = ("_lock", "_reputations")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reputations: dict[str, TransformReputation] = {}

    def get(self, name: str) -> TransformReputation:
        with self._lock:
            if name not in self._reputations:
                self._reputations[name] = TransformReputation()
            return self._reputations[name]

    def record(
        self,
        name: str,
        quality: float,
        compression: float,
        rolled_back: bool = False,
    ) -> None:
        self.get(name).record(quality, compression, rolled_back)

    def stats(self, name: str) -> TransformStats:
        return self.get(name).stats()

    def all_stats(self) -> dict[str, TransformStats]:
        with self._lock:
            return {name: rep.stats() for name, rep in self._reputations.items()}

    def is_high_risk(self, name: str) -> bool:
        return self.stats(name).risk == "HIGH"

    def should_disable_by_default(self, name: str) -> bool:
        stats = self.stats(name)
        return stats.rollback_rate > 0.25


_GLOBAL_REPUTATION = ReputationRegistry()


def get_reputation_registry() -> ReputationRegistry:
    return _GLOBAL_REPUTATION
