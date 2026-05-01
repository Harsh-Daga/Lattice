"""Tests for lattice.core.delta_wire compute_wire_savings and encoding."""

from __future__ import annotations

from lattice.core.delta_wire import (
    DeltaWireEncoder,
    compute_wire_savings,
    delta_wire_bytes,
)

SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
]

NEW_MESSAGES = [{"role": "user", "content": "How are you?"}]


def test_delta_wire_bytes() -> None:
    full_bytes, delta_bytes = delta_wire_bytes(SAMPLE_MESSAGES, NEW_MESSAGES, "sess_abc", 3)
    assert full_bytes > 0
    assert delta_bytes > 0
    assert delta_bytes < full_bytes


def test_compute_wire_savings() -> None:
    stats = compute_wire_savings(SAMPLE_MESSAGES, NEW_MESSAGES, "sess_abc", 3)
    assert stats["full_bytes"] > 0
    assert stats["delta_bytes"] > 0
    assert stats["savings_bytes"] > 0
    assert 0 < stats["savings_pct"] <= 100


def test_compute_wire_savings_empty_delta() -> None:
    """Empty delta still carries sentinel overhead, so savings may be small or zero."""
    stats = compute_wire_savings(SAMPLE_MESSAGES, [], "sess_abc", 3)
    # With empty delta, the overhead of sentinel/session/base_seq may
    # outweigh the savings from omitting messages. savings_bytes is clamped
    # to zero, so it should be >= 0.
    assert stats["savings_bytes"] >= 0
    assert stats["savings_pct"] >= 0


def test_compute_wire_savings_first_turn() -> None:
    """When base_seq is 0, delta == full, so savings should be 0 or negative (clamped)."""
    stats = compute_wire_savings(SAMPLE_MESSAGES, SAMPLE_MESSAGES, "sess_abc", 0)
    assert stats["savings_bytes"] >= 0
    assert stats["savings_pct"] >= 0


def test_encoder_includes_anchor_version() -> None:
    """DeltaWireEncoder includes anchor_version when provided."""
    encoder = DeltaWireEncoder()
    result = encoder.encode(SAMPLE_MESSAGES, "sess_abc", 2, anchor_version=5)
    assert result["_delta_wire"] is True
    assert result["_delta_session_id"] == "sess_abc"
    assert result["_delta_base_seq"] == 2
    assert result["_delta_anchor_version"] == 5
    assert result["_delta_messages"] == SAMPLE_MESSAGES[2:]


def test_encoder_omits_anchor_version_when_none() -> None:
    """DeltaWireEncoder omits anchor_version key when None."""
    encoder = DeltaWireEncoder()
    result = encoder.encode(SAMPLE_MESSAGES, "sess_abc", 2, anchor_version=None)
    assert "_delta_anchor_version" not in result
    assert result["_delta_messages"] == SAMPLE_MESSAGES[2:]
