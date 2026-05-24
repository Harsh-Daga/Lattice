"""Runtime tier classifier naming — LATTICE is not a provider router."""

from __future__ import annotations

import pytest

from lattice.transport.types import Message, Request


def test_router_is_gone() -> None:
    """README says 'LATTICE is not a router'. The runtime module must not contain one."""
    import lattice.runtime

    assert not hasattr(lattice.runtime, "RuntimeRouter")
    with pytest.raises(ImportError):
        from lattice.runtime.router import RuntimeRouter  # noqa: F401


def test_tier_classifier_works() -> None:
    from lattice.runtime import Tier, TierClassifier, TierDecision

    classifier = TierClassifier()
    req = Request(messages=[Message(role="user", content="hi")])
    decision = classifier.classify(req)
    assert isinstance(decision, TierDecision)
    assert decision.tier == Tier.SIMPLE
    assert decision.score < 20
