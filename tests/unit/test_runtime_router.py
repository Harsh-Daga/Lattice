"""Tests for runtime router."""

from lattice.core.context import TransformContext
from lattice.core.policy import OptimizationPolicy, Skip
from lattice.core.transport import Message, Request
from lattice.runtime.router import RuntimeRouter, Tier
from lattice.transforms.runtime_contract import RuntimeContractTransform


class TestRuntimeRouter:
    def test_simple_tier(self):
        router = RuntimeRouter()
        req = Request(messages=[Message(role="user", content="hi")])
        decision = router.classify(req)
        assert decision.tier == Tier.SIMPLE
        assert decision.score < 20

    def test_medium_tier_long_message(self):
        router = RuntimeRouter()
        req = Request(messages=[Message(role="user", content="a" * 2500)])
        decision = router.classify(req)
        assert decision.tier == Tier.MEDIUM
        assert decision.features["length"] == 20

    def test_complex_tier_tools(self):
        router = RuntimeRouter()
        req = Request(
            messages=[Message(role="user", content="use tools " * 500)],
            tools=[{"function": {"name": f"t{i}"}} for i in range(4)],
        )
        decision = router.classify(req)
        assert decision.tier == Tier.COMPLEX
        assert decision.features["tools"] == 20

    def test_reasoning_tier(self):
        router = RuntimeRouter()
        req = Request(
            messages=[
                Message(
                    role="user",
                    content="Prove the theorem step by step using formal logic and deduction. "
                    * 200,
                )
            ],
            tools=[{"function": {"name": "prover"}}],
        )
        decision = router.classify(req)
        assert decision.tier == Tier.REASONING
        assert decision.score >= 65

    def test_depth_score(self):
        router = RuntimeRouter()
        req = Request(messages=[Message(role="user", content=f"msg{i}") for i in range(12)])
        decision = router.classify(req)
        assert decision.features["depth"] == 10

    def test_code_score(self):
        router = RuntimeRouter()
        req = Request(messages=[Message(role="user", content="def foo(): pass" * 100)])
        decision = router.classify(req)
        assert decision.features["code"] > 0

    def test_select_model_never_overrides(self):
        """LATTICE is not a router — select_model always returns preferred."""
        router = RuntimeRouter()
        # High-confidence REASONING request
        req = Request(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Prove step by step using formal logic and deduction. "
                        "Implement the algorithm with function definitions and class structures. "
                    )
                    * 500,
                )
            ],
            tools=[{"function": {"name": "prover"}}, {"function": {"name": "solver"}}],
        )
        assert router.select_model(req, "preferred") == "preferred"

    def test_select_model_simple_request(self):
        router = RuntimeRouter()
        req = Request(messages=[Message(role="user", content="hi")])
        model = router.select_model(req, "my-model")
        assert model == "my-model"

    def test_select_model_respects_preferred_always(self):
        router = RuntimeRouter()
        req = Request(messages=[Message(role="user", content="a" * 550)])
        model = router.select_model(req, "preferred")
        assert model == "preferred"

    def test_to_dict(self):
        router = RuntimeRouter()
        req = Request(messages=[Message(role="user", content="test")])
        decision = router.classify(req)
        d = decision.to_dict()
        assert "tier" in d
        assert "score" in d
        assert "features" in d
        assert "confidence" in d
        assert "contract" in d
        assert d["contract"]["allow_model_override"] is False

    def test_custom_keywords(self):
        router = RuntimeRouter(
            reasoning_keywords=["custom_reasoning_kw"],
            code_indicators=["custom_code_kw"],
        )
        req = Request(messages=[Message(role="user", content="custom_reasoning_kw custom_code_kw")])
        decision = router.classify(req)
        assert decision.features["reasoning"] > 0
        assert decision.features["code"] > 0

    def test_simple_contract_skips_expensive_transforms(self):
        router = RuntimeRouter()
        decision = router.classify(Request(messages=[Message(role="user", content="hi")]))
        skipped = decision.contract["skipped_transforms"]
        assert decision.tier == Tier.SIMPLE
        assert "self_information" in skipped
        assert "rate_distortion" in skipped
        assert decision.contract["mode"] == "minimal"

    def test_reasoning_contract_keeps_all_transforms(self):
        router = RuntimeRouter()
        req = Request(
            messages=[
                Message(
                    role="user",
                    content="prove the theorem step by step with formal proof " * 300,
                )
            ],
            tools=[{"function": {"name": "prover"}}],
        )
        decision = router.classify(req)
        assert decision.tier == Tier.REASONING
        assert decision.contract["skipped_transforms"] == []
        assert decision.contract["mode"] == "max_fidelity"

    def test_runtime_contract_transform_writes_metadata(self):
        transform = RuntimeContractTransform()
        req = Request(messages=[Message(role="user", content="hi")])
        ctx = TransformContext()
        result = transform.process(req, ctx)
        assert result
        new_req = result.unwrap()
        runtime = new_req.metadata["_lattice_runtime"]
        assert runtime["tier"] == Tier.SIMPLE
        assert new_req.metadata["_lattice_runtime_contract"]["mode"] == "minimal"

    def test_policy_enforces_runtime_contract_skip(self):
        req = Request(
            messages=[Message(role="user", content="hi")],
            metadata={
                "_lattice_runtime_contract": {
                    "skipped_transforms": ["self_information"],
                }
            },
        )
        decision = OptimizationPolicy().should_run(
            "self_information",
            req,
            TransformContext(),
        )
        assert isinstance(decision, Skip)
        assert decision.reason == "disabled_by_runtime_contract"
