"""Tests for lattice.protocol.reliability.

Covers:
- SelectiveReliability decisions for each criticality level
- Retry budget enforcement
"""

from __future__ import annotations

import dataclasses

from lattice.protocol.framing import FrameFlags
from lattice.protocol.reliability import ReliabilityMode, SelectiveReliability


@dataclasses.dataclass(frozen=True)
class _FakeFrame:
    """Minimal stand-in for lattice.protocol.framing.Frame."""

    flags: FrameFlags


class TestSelectiveReliability:
    def test_high_criticality_retries_up_to_max(self) -> None:
        sr = SelectiveReliability(max_retries=3)
        frame = _FakeFrame(flags=FrameFlags.CRITICALITY_HIGH)
        assert sr.should_retransmit(frame, attempt=0) is True
        assert sr.should_retransmit(frame, attempt=1) is True
        assert sr.should_retransmit(frame, attempt=2) is True
        assert sr.should_retransmit(frame, attempt=3) is False

    def test_medium_criticality_limited_to_two(self) -> None:
        sr = SelectiveReliability(max_retries=5)
        frame = _FakeFrame(flags=FrameFlags.CRITICALITY_MEDIUM)
        assert sr.should_retransmit(frame, attempt=0) is True
        assert sr.should_retransmit(frame, attempt=1) is True
        assert sr.should_retransmit(frame, attempt=2) is False
        assert sr.should_retransmit(frame, attempt=3) is False

    def test_low_criticality_never_retransmits(self) -> None:
        sr = SelectiveReliability(max_retries=10)
        frame = _FakeFrame(flags=FrameFlags.CRITICALITY_LOW)
        assert sr.should_retransmit(frame, attempt=0) is False
        assert sr.should_retransmit(frame, attempt=9) is False

    def test_none_criticality_never_retransmits(self) -> None:
        sr = SelectiveReliability(max_retries=10)
        frame = _FakeFrame(flags=FrameFlags.NONE)
        assert sr.should_retransmit(frame, attempt=0) is False
        assert sr.should_retransmit(frame, attempt=5) is False

    def test_combined_flags_high_wins(self) -> None:
        """If both HIGH and MEDIUM are set, HIGH logic applies."""
        sr = SelectiveReliability(max_retries=2)
        frame = _FakeFrame(flags=FrameFlags.CRITICALITY_HIGH | FrameFlags.CRITICALITY_MEDIUM)
        assert sr.should_retransmit(frame, attempt=0) is True
        assert sr.should_retransmit(frame, attempt=1) is True
        assert sr.should_retransmit(frame, attempt=2) is False

    def test_default_max_retries(self) -> None:
        sr = SelectiveReliability()
        assert sr.max_retries == 3


class TestReliabilityModeEnum:
    def test_members(self) -> None:
        assert ReliabilityMode.RELIABLE.value == "reliable"
        assert ReliabilityMode.PARTIAL.value == "partial"
        assert ReliabilityMode.BEST_EFFORT.value == "best_effort"
