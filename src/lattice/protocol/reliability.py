"""Selective reliability for LATTICE protocol frames.

Provides per-frame reliability decisions based on frame criticality flags.
"""

from __future__ import annotations

import enum
from typing import Any

from lattice.protocol.framing import FrameFlags


class ReliabilityMode(enum.Enum):
    """Delivery reliability guarantees."""

    RELIABLE = "reliable"
    PARTIAL = "partial"
    BEST_EFFORT = "best_effort"


class SelectiveReliability:
    """Per-frame reliability manager.

    Based on QUIC partial reliability concepts:
    - CRITICALITY_HIGH frames: guaranteed delivery, retransmit on loss
    - CRITICALITY_MEDIUM frames: at-least-once delivery, bounded retries
    - CRITICALITY_LOW / no criticality: best-effort, no retransmission
    """

    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries

    def should_retransmit(self, frame: Any, attempt: int) -> bool:
        """Decide whether a lost frame should be retransmitted.

        Args:
            frame: A frame object with a ``flags`` attribute (FrameFlags).
            attempt: Zero-based retransmission attempt number.

        Returns:
            True if the frame should be retransmitted.
        """
        flags = frame.flags

        if flags & FrameFlags.CRITICALITY_HIGH:
            return attempt < self.max_retries
        elif flags & FrameFlags.CRITICALITY_MEDIUM:
            return attempt < 2
        else:
            return False
