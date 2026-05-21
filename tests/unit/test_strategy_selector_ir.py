"""Unit tests for StrategySelector.optimize() (IR-native path)."""

from __future__ import annotations

from lattice.core.context import TransformContext
from lattice.core.result import Ok
from lattice.ir.primitives import PromptIRV2, SectionV2, SpanV2
from lattice.transforms.strategy_selector import StrategySelector
from lattice.transport.types import Request


class TestStrategySelectorIR:
    """Test the IR-native optimize() method of StrategySelector."""

    def _make_request(self, sections: list[SectionV2]) -> tuple[Request, TransformContext]:
        ir = PromptIRV2(sections=tuple(sections))
        request = Request(messages=[])
        request.metadata["_lattice_ir_v2"] = ir
        ctx = TransformContext()
        ctx.session_state["_lattice_ir_v2"] = ir
        return request, ctx

    def test_selects_arm_and_records_strategy(self) -> None:
        sec = SectionV2(type="context", spans=(SpanV2(span_id="s1", text="hello world"),))
        ir = PromptIRV2(sections=(sec,))
        request, ctx = self._make_request([sec])
        ss = StrategySelector()
        result = ss.optimize(ir, request, ctx)
        assert isinstance(result, Ok)
        new_ir = result.unwrap()
        # Strategy stored in metadata
        meta = dict(new_ir.metadata)
        assert "_lattice_strategy" in meta
        assert meta["_lattice_strategy"] in ss.arms

    def test_metrics_recorded(self) -> None:
        sec = SectionV2(type="context", spans=(SpanV2(span_id="s1", text="hello world"),))
        ir = PromptIRV2(sections=(sec,))
        request, ctx = self._make_request([sec])
        ss = StrategySelector()
        result = ss.optimize(ir, request, ctx)
        assert isinstance(result, Ok)
        metrics = ctx.metrics.get("transforms", {}).get("strategy_selector", {})
        assert metrics.get("selected_strategy") in ss.arms
        assert metrics.get("feature_norm", 0) >= 0

    def test_ir_features_computed(self) -> None:
        sections = [
            SectionV2(type="context", spans=(SpanV2(span_id="s1", text="a" * 100),)),
            SectionV2(type="json", spans=(SpanV2(span_id="s2", text='{"x": 1}'),)),
        ]
        ir = PromptIRV2(sections=tuple(sections))
        request, ctx = self._make_request(sections)
        ss = StrategySelector()
        features = ss._extract_features_ir(ir)
        assert len(features) == ss.feature_dim
        assert all(isinstance(f, float) for f in features)
        # bias term should be present
        assert features[-1] == 1.0
        # json presence should be 1.0
        assert features[3] == 1.0

    def test_strategy_flags_set(self) -> None:
        sec = SectionV2(type="context", spans=(SpanV2(span_id="s1", text="hello world"),))
        ir = PromptIRV2(sections=(sec,))
        request, ctx = self._make_request([sec])
        ss = StrategySelector()
        for arm in ss.arms:
            new_ir = ss._set_strategy_flags_ir(ir, arm, ctx)
            meta = dict(new_ir.metadata)
            assert "_lattice_strategy_submodular" in meta
            assert "_lattice_strategy_rate_distortion" in meta
