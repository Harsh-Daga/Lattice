"""Tests for StrategySelector (LinUCB contextual bandit)."""

from __future__ import annotations

import pytest

from lattice.core.context import TransformContext
from lattice.core.result import is_err, unwrap
from lattice.core.transport import Message, Request, Role
from lattice.transforms.strategy_selector import StrategySelector, _ArmState

# =============================================================================
# _ArmState unit tests
# =============================================================================

class TestArmState:
    """Test the per-arm LinUCB state machine."""

    def test_zero_initialization(self) -> None:
        s = _ArmState.zero(dim=3, lambda_reg=1.0)
        assert len(s.A) == 3
        assert len(s.A[0]) == 3
        assert s.A[0][0] == 1.0
        assert s.A[0][1] == 0.0
        assert len(s.b) == 3
        assert s.b == [0.0, 0.0, 0.0]
        assert s.pulls == 0
        assert s.cum_reward == 0.0

    def test_roundtrip(self) -> None:
        s = _ArmState.zero(dim=2, lambda_reg=0.5)
        s.update([1.0, 0.0], reward=1.0)
        d = s.to_dict()
        s2 = _ArmState.from_dict(d)
        assert s2.A == s.A
        assert s2.b == s.b
        assert s2.pulls == s.pulls
        assert s2.cum_reward == s.cum_reward

    def test_update_increments(self) -> None:
        s = _ArmState.zero(dim=2, lambda_reg=1.0)
        s.update([1.0, 2.0], reward=1.0)
        assert s.pulls == 1
        assert s.cum_reward == 1.0
        assert s.A[0][0] == 2.0  # 1 + 1*1
        assert s.A[0][1] == 2.0  # 0 + 1*2
        assert s.b[0] == 1.0    # 0 + 1*1
        assert s.b[1] == 2.0    # 0 + 1*2

    def test_theta_after_one_update(self) -> None:
        s = _ArmState.zero(dim=1, lambda_reg=1.0)
        s.update([2.0], reward=1.0)
        # A = [[5]], b = [2], theta = [0.4]
        theta = s.theta()
        assert pytest.approx(theta[0], 0.01) == 0.4

    def test_ucb_score(self) -> None:
        s = _ArmState.zero(dim=1, lambda_reg=1.0)
        # UCB before any pulls: theta=0 (no data), explore term = alpha * sqrt(x^T A^{-1} x)
        score = s.ucb_score([1.0], alpha=1.0)
        # theta = [0], A = [[1]], x^T A^{-1} x = 1*1*1 = 1, explore = 1.0
        assert pytest.approx(score, 0.01) == 1.0

    def test_ucb_score_rises_with_reward(self) -> None:
        s = _ArmState.zero(dim=1, lambda_reg=1.0)
        s.update([1.0], reward=1.0)
        # A = [[2]], b = [1], theta = [0.5]
        score = s.ucb_score([1.0], alpha=1.0)
        # exploit = 0.5 * 1 = 0.5, A^{-1} = [[0.5]], x^T A^{-1} x = 0.5
        # explore = 1.0 * sqrt(0.5) ≈ 0.707 -> total ≈ 1.207
        assert score > 0.5

    def test_ucb_score_decreases_with_pulls(self) -> None:
        """Exploration bonus should shrink as we pull more."""
        s = _ArmState.zero(dim=1, lambda_reg=1.0)
        s.update([1.0], reward=1.0)
        s.update([1.0], reward=0.0)
        # After 2 pulls, A = [[3]], b = [1], theta = [0.333], xAx = 1/3
        score = s.ucb_score([1.0], alpha=1.0)
        # explore = sqrt(1/3) ≈ 0.577, exploit = 0.333
        assert pytest.approx(score, 0.01) == 0.577 + 0.333

    def test_theta_zero_with_no_observations(self) -> None:
        s = _ArmState.zero(dim=3, lambda_reg=1.0)
        theta = s.theta()
        assert theta == [0.0, 0.0, 0.0]

    def test_theta_with_perfect_fit(self) -> None:
        """If b = A * [3, -1], then theta should be [3, -1]."""
        s = _ArmState.zero(dim=2, lambda_reg=0.0)
        # Build A manually and verify theta
        s.A = [[2.0, 1.0], [1.0, 2.0]]
        # [2 1; 1 2] * [3; -1] = [5; 1]
        s.b = [5.0, 1.0]
        theta = s.theta()
        assert pytest.approx(theta[0], 0.01) == 3.0
        assert pytest.approx(theta[1], 0.01) == -1.0


# =============================================================================
# StrategySelector unit tests
# =============================================================================

class TestStrategySelectorInit:
    def test_default_init(self) -> None:
        sel = StrategySelector()
        assert sel.arms == ("full", "submodular", "rd", "hybrid")
        assert sel.feature_dim == 8
        assert sel.alpha == 0.5
        assert sel.default_strategy == "full"

    def test_custom_arms(self) -> None:
        sel = StrategySelector(arms=("a", "b"), default_strategy="a")
        assert sel.arms == ("a", "b")
        assert sel.default_strategy == "a"

    def test_duplicate_arms_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            StrategySelector(arms=("a", "a"))

    def test_empty_arms_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one arm"):
            StrategySelector(arms=())

    def test_invalid_default_strategy(self) -> None:
        with pytest.raises(ValueError, match="default_strategy"):
            StrategySelector(default_strategy="nonexistent")

    def test_custom_params(self) -> None:
        sel = StrategySelector(feature_dim=5, alpha=2.0, lambda_reg=0.1, warmup_pulls=10)
        assert sel.feature_dim == 5
        assert sel.alpha == 2.0
        assert sel.lambda_reg == 0.1
        assert sel.warmup_pulls == 10


class TestFeatureExtraction:
    def test_extract_short_request(self) -> None:
        sel = StrategySelector()
        req = Request(messages=[Message(role=Role.USER, content="hi")])
        ctx = TransformContext(request_id="r1")
        features = sel._extract_features(req, ctx)
        assert len(features) == 8
        assert all(0.0 <= f <= 1.0 for f in features)

    def test_extract_with_tools(self) -> None:
        sel = StrategySelector()
        req = Request(
            messages=[Message(role=Role.USER, content="hello")],
            tools=[{"name": "f"}],
        )
        ctx = TransformContext(request_id="r1")
        features = sel._extract_features(req, ctx)
        assert features[1] == 1.0  # tool presence

    def test_extract_streaming(self) -> None:
        sel = StrategySelector()
        req = Request(
            messages=[Message(role=Role.USER, content="hello")],
            stream=True,
        )
        ctx = TransformContext(request_id="r1")
        features = sel._extract_features(req, ctx)
        assert features[5] == 1.0  # streaming

    def test_extract_code_fraction(self) -> None:
        sel = StrategySelector()
        req = Request(
            messages=[
                Message(role=Role.USER, content="```python\nx = 5\n```")
            ]
        )
        ctx = TransformContext(request_id="r1")
        features = sel._extract_features(req, ctx)
        assert features[3] > 0.0  # code fraction

    def test_extract_message_depth(self) -> None:
        sel = StrategySelector()
        messages = [Message(role=Role.USER, content=f"msg{i}") for i in range(25)]
        req = Request(messages=messages)
        ctx = TransformContext(request_id="r1")
        features = sel._extract_features(req, ctx)
        assert features[4] == 1.0  # clamped to 1.0 (>20 messages)


class TestProcess:
    def test_selects_arm(self) -> None:
        sel = StrategySelector()
        req = Request(
            messages=[Message(role=Role.USER, content="this is a somewhat longer test message so that the token estimate is large enough to avoid the short-circuit path and trigger algorithmic arm selection.")],
            metadata={},
        )
        ctx = TransformContext(request_id="r1")
        result = sel.process(req, ctx)
        assert is_err(result) is False
        new_req = unwrap(result)
        strategy = new_req.metadata.get("_lattice_strategy")
        assert isinstance(strategy, dict)
        assert strategy["name"] in sel.arms

    def test_sets_strategy_flags(self) -> None:
        sel = StrategySelector()
        req = Request(
            messages=[Message(role=Role.USER, content="a bit longer message to avoid the short-circuit path where nothing is selected.")],
            metadata={},
        )
        ctx = TransformContext(request_id="r1")
        result = sel.process(req, ctx)
        new_req = unwrap(result)
        s = new_req.metadata["_lattice_strategy"]
        assert "name" in s
        assert "semantic_compress" in s
        assert "submodular_select" in s
        assert "rd_compress" in s

    def test_runtime_contract_preferred_strategy_wins(self) -> None:
        sel = StrategySelector(warmup_pulls=10)
        req = Request(
            messages=[Message(role=Role.USER, content="long enough message " * 20)],
            metadata={
                "_lattice_runtime_contract": {
                    "preferred_strategy": "rd",
                }
            },
        )
        ctx = TransformContext(request_id="r1")
        result = sel.process(req, ctx)
        new_req = unwrap(result)
        assert new_req.metadata["_lattice_strategy"]["name"] == "rd"
        assert new_req.metadata["_lattice_strategy"]["rd_compress"] is True

    def test_invalid_runtime_contract_strategy_falls_back(self) -> None:
        sel = StrategySelector(arms=("full", "rd"), warmup_pulls=10)
        req = Request(
            messages=[Message(role=Role.USER, content="long enough message " * 20)],
            metadata={
                "_lattice_runtime_contract": {
                    "preferred_strategy": "missing",
                }
            },
        )
        ctx = TransformContext(request_id="r1")
        result = sel.process(req, ctx)
        new_req = unwrap(result)
        assert new_req.metadata["_lattice_strategy"]["name"] == "full"

    def test_short_request_can_process_false(self) -> None:
        sel = StrategySelector()
        req = Request(messages=[Message(role=Role.USER, content="hi")])
        ctx = TransformContext(request_id="r1")
        assert sel.can_process(req, ctx) is False

    def test_warmup_pulls_round_robin(self) -> None:
        """During warmup, arms should be selected in round-robin from least-pulled."""
        sel = StrategySelector(arms=("a", "b", "c"), warmup_pulls=2)
        ctx = TransformContext(request_id="r1")
        selected_arms: list[str] = []
        for i in range(6):
            req = Request(
                messages=[Message(role=Role.USER, content=f"x {i} " * 50)],
                metadata={},
            )
            result = sel.process(req, ctx)
            new_req = unwrap(result)
            selected_arms.append(new_req.metadata["_lattice_strategy"]["name"])
            # Update reward to advance warmup
            sel.update_reward(ctx, reward=0.5)

        assert "a" in selected_arms
        assert "b" in selected_arms
        assert "c" in selected_arms

    def test_bandit_state_persists_roundtrip(self) -> None:
        sel = StrategySelector()
        ctx = TransformContext(request_id="r1")
        req = Request(
            messages=[Message(role=Role.USER, content="hello " * 40)],
            metadata={},
        )
        sel.process(req, ctx)
        sel.update_reward(ctx, reward=0.8)

        state = ctx.get_transform_state(sel.name)
        assert "bandit" in state
        bandit = state["bandit"]
        assert bandit["total_pulls"] == 1
        assert bandit["arms"][state["last_arm"]]["pulls"] == 1
        assert bandit["arms"][state["last_arm"]]["cum_reward"] == 0.8

    def test_reward_clipping(self) -> None:
        sel = StrategySelector()
        ctx = TransformContext(request_id="r1")
        req = Request(
            messages=[Message(role=Role.USER, content="hello " * 40)],
            metadata={},
        )
        sel.process(req, ctx)
        sel.update_reward(ctx, reward=-0.5)  # should clip to 0
        sel.update_reward(ctx, reward=1.5)   # should clip to 1

        bandit = ctx.get_transform_state(sel.name)["bandit"]
        assert bandit["arms"]["full"]["cum_reward"] == 1.0  # 0 + 1

    def test_update_reward_no_arm(self) -> None:
        sel = StrategySelector()
        ctx = TransformContext(request_id="r1")
        # Never selected an arm
        sel.update_reward(ctx, reward=0.5)
        # Should not crash

    def test_reverse_is_noop(self) -> None:
        from lattice.core.transport import Response
        sel = StrategySelector()
        resp = Response(content="hello")
        ctx = TransformContext(request_id="r1")
        assert sel.reverse(resp, ctx) is resp

    def test_feature_dim_greater_than_8(self) -> None:
        """If feature_dim > 8, extra features should be zero-padded."""
        sel = StrategySelector(feature_dim=12)
        req = Request(
            messages=[Message(role=Role.USER, content="hello " * 40)],
            metadata={},
        )
        ctx = TransformContext(request_id="r1")
        features = sel._extract_features(req, ctx)
        assert len(features) == 12
        # Features beyond index 7 should be 0.0
        assert features[8] == 0.0
        assert features[11] == 0.0

    def test_feature_dim_less_than_8(self) -> None:
        """If feature_dim < 8, only that many features should be extracted."""
        sel = StrategySelector(feature_dim=3)
        req = Request(
            messages=[Message(role=Role.USER, content="hello " * 40)],
            metadata={},
        )
        ctx = TransformContext(request_id="r1")
        features = sel._extract_features(req, ctx)
        assert len(features) == 3

    def test_multiple_requests_same_session(self) -> None:
        """Bandit state should accumulate across requests in the same session."""
        sel = StrategySelector()
        ctx = TransformContext(request_id="r1")

        for _ in range(5):
            req = Request(
                messages=[Message(role=Role.USER, content="hello " * 50)],
                metadata={},
            )
            sel.process(req, ctx)
            sel.update_reward(ctx, reward=0.6)

        bandit = ctx.get_transform_state(sel.name)["bandit"]
        total_pulls = bandit["total_pulls"]
        assert total_pulls == 5

    def test_different_arms_selected_based_on_content(self) -> None:
        """Different content should sometimes select different arms over time."""
        # warmup_pulls=4 forces each arm to be tried once before exploitation
        sel = StrategySelector(warmup_pulls=4)
        ctx = TransformContext(request_id="r1")

        arms_seen: set[str] = set()
        contents = [
            "Step 1: define. " * 50,
            "```python\nfor i in range(10): print(i)\n``` " * 20,
            "|x|y|\n|:---:|:---:|\n|1|2|\n" * 30,
            "hello world " * 50,
            "Let me think. Therefore, foo. Bar " * 40,
        ]
        for i in range(20):
            content = contents[i % len(contents)]
            req = Request(
                messages=[Message(role=Role.USER, content=content)],
                metadata={},
            )
            sel.process(req, ctx)
            arms_seen.add(req.metadata["_lattice_strategy"]["name"])
            sel.update_reward(ctx, reward=0.7)

        # With warmup=4 we should have explored all 4 arms
        assert len(arms_seen) >= 2


class TestFeatureHelpers:
    def test_count_reasoning_text(self) -> None:
        text = "Let me think about this. Step 1: foo. Therefore, bar."
        count = StrategySelector._count_reasoning_text(text)
        assert count > 0

    def test_count_code_block_chars(self) -> None:
        text = "```python\nprint(1)\n```\n`inline`"
        count = StrategySelector._count_code_block_chars(text)
        assert count > 0
        assert "inline" not in text.split("```")[1]  # first block only

    def test_estimate_table_density_md(self) -> None:
        text = "|a|b|\n|:---|:---|\n|1|2|\n"
        d = StrategySelector._estimate_table_density(text)
        assert d > 0.0
        assert d <= 1.0

    def test_estimate_table_density_json(self) -> None:
        text = "[{\"a\":1}, {\"b\":2}]"
        d = StrategySelector._estimate_table_density(text)
        assert d >= 0.0


class TestSetStrategyFlags:
    def test_full_strategy_flags(self) -> None:
        sel = StrategySelector()
        req = Request(messages=[], metadata={})
        sel._set_strategy_flags(req, "full")
        s = req.metadata["_lattice_strategy"]
        assert s["submodular_select"] is True
        assert s["rd_compress"] is True

    def test_submodular_strategy_flags(self) -> None:
        sel = StrategySelector()
        req = Request(messages=[], metadata={})
        sel._set_strategy_flags(req, "submodular")
        s = req.metadata["_lattice_strategy"]
        assert s["submodular_select"] is True
        assert s["rd_compress"] is False

    def test_rd_strategy_flags(self) -> None:
        sel = StrategySelector()
        req = Request(messages=[], metadata={})
        sel._set_strategy_flags(req, "rd")
        s = req.metadata["_lattice_strategy"]
        assert s["submodular_select"] is False
        assert s["rd_compress"] is True

    def test_merge_string_to_dict(self) -> None:
        sel = StrategySelector()
        req = Request(messages=[], metadata={"_lattice_strategy": "old"})
        sel._set_strategy_flags(req, "rd")
        s = req.metadata["_lattice_strategy"]
        assert isinstance(s, dict)
        assert s["name"] == "rd"
        assert "rd_compress" in s

    def test_merge_dict(self) -> None:
        sel = StrategySelector()
        req = Request(messages=[], metadata={"_lattice_strategy": {"name": "old", "extra": True}})
        sel._set_strategy_flags(req, "full")
        s = req.metadata["_lattice_strategy"]
        assert s["name"] == "full"
        assert s["extra"] is True
