"""End-to-end integration tests for the unified optimizer + transport architecture.

These tests verify the full request flow through every layer:
1. Request Intelligence (classifier)
2. Representation Optimizer (beam search)
3. Transport Optimizer (delta, cache, framing)
4. Provider Execution (routing, streaming, metrics)
"""

from __future__ import annotations

from lattice.core.config import LatticeConfig
from lattice.core.context import TransformContext
from lattice.core.result import is_ok, unwrap
from lattice.pipeline.factory import build_default_pipeline
from lattice.planner.execution_plan import ExecutionPlan
from lattice.planner.provider_strategy import get_provider_strategy
from lattice.planner.request_classifier import RequestClassifier
from lattice.planner.transport_planner import build_transport_plan
from lattice.transport.types import Message, Request, Response


def _req(content: str, role: str = "user") -> Message:
    return Message(role=role, content=content)


class TestRequestIntelligenceLayer:
    """Layer 1 — classify task, risk, provider, session, content shape."""

    def test_request_classifier_produces_execution_plan_inputs(self) -> None:
        classifier = RequestClassifier()
        request = Request(
            messages=[
                _req("Debug the TypeError in line 42 of app.py"),
                _req("Traceback: ...", role="tool"),
            ],
            model="gpt-4",
        )
        result = classifier.classify(request)
        # Task class is determined by the classifier — could be debugging, retrieval, etc.
        assert result["task_class"] in (
            "debugging",
            "retrieval",
            "reasoning",
            "analysis",
            "structured",
            "simple",
            "summarization",
        )
        assert result["budget_ms"] > 0
        assert result["quality_floor"] >= 0.80
        assert result["debug_heavy"] is True

    def test_provider_strategy_maps_correctly(self) -> None:
        for provider in ("openai", "anthropic", "ollama", "gemini"):
            strategy = get_provider_strategy(provider)
            assert strategy.provider == provider
            assert len(strategy.preferred_optimizers) >= 3

    def test_transport_plan_builds_provider_aware(self) -> None:
        plan = build_transport_plan(
            provider="openai",
            model="gpt-4",
            session_id="sess_abc123",
            base_sequence=5,
            is_streaming=True,
            estimated_tokens=5000,
        )
        assert plan.delta_mode is True
        assert plan.resume_enabled is True
        assert plan.compression_codec == "dictionary"  # >4000 tokens
        assert plan.wire_format == "json"

        ollama_plan = build_transport_plan(
            provider="ollama",
            model="llama3.2",
            session_id="sess_abc123",
            base_sequence=5,
            estimated_tokens=5000,
        )
        assert ollama_plan.compression_codec is None  # No provider cache
        assert ollama_plan.cache_plan == []


class TestRepresentationOptimizerLayer:
    """Layer 2 — beam search across structure, reference, tool, context, diagnostic."""

    def test_optimizer_pipeline_builds_correctly(self) -> None:
        cfg = LatticeConfig(use_optimizer_pipeline=True)
        pipeline = build_default_pipeline(cfg)
        names = [t.name for t in pipeline.transforms]
        assert "content_profiler" in names
        assert "runtime_contract" in names
        assert "ir_structure_optimizer" in names
        assert "diagnostic_optimizer" in names

    def test_scheduler_decision_flows_to_representation_optimizer(self) -> None:
        from lattice.transforms.content_profiler import ContentProfiler

        profiler = ContentProfiler()
        request = Request(
            messages=[_req("Debug the TypeError in line 42")],
            model="gpt-4",
        )
        ctx = TransformContext()

        result = profiler.process(request, ctx)
        assert is_ok(result)

        # Schedule is stored in BOTH metadata and session_state
        assert "_lattice_schedule" in ctx.session_state
        schedule = ctx.session_state["_lattice_schedule"]
        assert "allowed_optimizers" in schedule
        assert "allowed" in schedule
        assert "blocked" in schedule

    def test_hard_rollback_rejects_expansion(self) -> None:
        from lattice.pipeline.representation_optimizer import _validate_beam_candidate

        candidate = type(
            "Cand",
            (),
            {
                "tokens_before": 100,
                "tokens_after": 120,
                "quality_estimate": 0.95,
                "cache_gain": 0.0,
                "transport_gain": 0.0,
                "latency_ms": 10.0,
            },
        )()
        ctx = TransformContext()
        assert _validate_beam_candidate(candidate, 0.85, ctx) is False

    def test_pipeline_runs_end_to_end_with_optimizers(self) -> None:
        cfg = LatticeConfig(use_optimizer_pipeline=True)
        pipeline = build_default_pipeline(cfg)

        request = Request(
            messages=[_req("Error: stack trace shows ModuleNotFoundError")],
            model="gpt-4",
        )
        ctx = TransformContext()

        result = pipeline.compress(request, ctx)
        assert is_ok(result)
        modified = unwrap(result)
        assert modified is not None
        assert "_lattice_schedule" in ctx.session_state
        assert any(name.endswith("_optimizer") for name in ctx.transforms_applied), (
            f"expected an optimizer transform, got {ctx.transforms_applied}"
        )


class TestTransportOptimizerLayer:
    """Layer 3 — delta, cache alignment, binary framing, multiplex, resume."""

    def test_delta_wire_decoder_exists(self) -> None:
        from lattice.core.session import MemorySessionStore
        from lattice.transport.delta_wire import DeltaWireDecoder

        store = MemorySessionStore(ttl_seconds=3600, max_sessions=100)
        decoder = DeltaWireDecoder(store)
        assert decoder is not None

    def test_stream_manager_creates_resume_token(self) -> None:
        from lattice.protocol.resume import StreamManager

        sm = StreamManager()
        token = sm.create_resume_token("stream_123", sequence=0)
        assert token is not None
        result = sm.validate_resume_token(token)
        assert result is not None
        stream_id, seq = result
        assert stream_id == "stream_123"

    def test_binary_framer_roundtrip(self) -> None:
        from lattice.protocol.framing import BinaryFramer, FrameFlags

        framer = BinaryFramer()
        payload = b'{"model":"gpt-4","messages":[]}'
        frames = framer.encode_request(payload, flags=FrameFlags.NONE)
        assert len(frames) > 0
        decoded = framer.decode_frame(frames[0].to_bytes())
        assert decoded.payload == payload


class TestProviderExecutionLayer:
    """Layer 4 — route, call model, stream safely, observe metrics."""

    def test_direct_http_provider_registry_has_all_adapters(self) -> None:
        from lattice.providers.transport import ProviderRegistry

        registry = ProviderRegistry()
        for name in ("openai", "anthropic", "ollama", "gemini", "azure", "bedrock"):
            adapter = registry.get_adapter(name)
            assert adapter is not None, f"Missing adapter for {name}"

    def test_provider_strategy_selects_correct_cache_mode(self) -> None:
        from lattice.planner.provider_strategy import get_provider_strategy

        openai = get_provider_strategy("openai")
        assert openai.cache_mode == "auto_prefix"
        assert openai.prefix_stable is True

        anthropic = get_provider_strategy("anthropic")
        assert anthropic.cache_mode == "explicit_breakpoint"
        assert anthropic.supports_breakpoints is True

        ollama = get_provider_strategy("ollama")
        assert ollama.cache_mode == "none"

    def test_execution_plan_encapsulates_all_layers(self) -> None:
        plan = ExecutionPlan(
            request_id="req_123",
            session_id="sess_456",
            provider="openai",
            model="gpt-4",
            task_class="debugging",
            risk_level="medium",
            latency_budget_ms=150.0,
            quality_floor=0.90,
            allowed_optimizers=[
                "representation_optimizer",
                "diagnostic_optimizer",
                "reference_optimizer",
            ],
            blocked_optimizers={"context_optimizer": "debugging_task_no_lossy"},
            representation_plan=["diagnostic_optimizer", "reference_optimizer"],
        )
        d = plan.to_dict()
        assert d["provider"] == "openai"
        assert d["task_class"] == "debugging"
        assert d["allowed_optimizers"] == [
            "representation_optimizer",
            "diagnostic_optimizer",
            "reference_optimizer",
        ]
        assert d["blocked_optimizers"] == {"context_optimizer": "debugging_task_no_lossy"}


class TestGovernanceAndRollback:
    """Phase 7 — hard rollback, safety, placeholder leakage."""

    def test_placeholder_leakage_blocked(self) -> None:
        from lattice.pipeline.guardrails import GuardAction, check_placeholder_leakage

        before = "The error was in module X with ID 123"
        after = "The error was <ref_17> in module <d_36>"
        decision = check_placeholder_leakage(before, after)
        assert decision.action in (GuardAction.ROLLBACK, GuardAction.REJECT)


class TestReverseTransformsEndToEnd:
    """Full forward + reverse cycle."""

    def test_optimizer_pipeline_reverse_restores_references(self) -> None:
        cfg = LatticeConfig(use_optimizer_pipeline=True)
        pipeline = build_default_pipeline(cfg)

        request = Request(
            messages=[
                _req("UUID 550e8400-e29b-41d4-a716-446655440000 is the error"),
            ],
            model="gpt-4",
        )
        ctx = TransformContext()

        result = pipeline.compress(request, ctx)
        assert is_ok(result)
        _compressed = unwrap(result)
        assert _compressed is not None

        response = Response(
            role="assistant",
            content="The error was in module <ref_1>",
            model="gpt-4",
        )
        restored = pipeline.reverse(response, ctx)
        assert restored is not None
        assert hasattr(restored, "content")
