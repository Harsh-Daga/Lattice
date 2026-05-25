"""Full public Python surface + deleted-symbol guard (Phase 11)."""

from __future__ import annotations

import importlib

import pytest


def test_public_api_complete() -> None:
    import lattice.cache
    import lattice.gateway
    import lattice.integrations
    import lattice.ir
    import lattice.pipeline
    import lattice.planner
    import lattice.protocol
    import lattice.providers
    import lattice.proxy
    import lattice.runtime
    import lattice.safety
    import lattice.sdk
    import lattice.state
    import lattice.telemetry
    import lattice.transforms
    import lattice.transport
    from lattice import (  # noqa: F401
        CompressResult,
        DowngradeCategory,
        LatticeClient,
        LatticeProxyClient,
        MetricsCollector,
        PromptIR,
        PromptIRV2,
        SegmentStore,
        SemanticCache,
        SemanticRiskScore,
        Session,
        SessionManager,
        __version__,
        build_ir,
        compute_risk_score,
        serialize_ir_to_text,
        wrap_openai_client,
    )

    for mod in (
        lattice.ir,
        lattice.planner,
        lattice.pipeline,
        lattice.transforms,
        lattice.providers,
        lattice.transport,
        lattice.protocol,
        lattice.state,
        lattice.cache,
        lattice.telemetry,
        lattice.safety,
        lattice.proxy,
        lattice.gateway,
        lattice.sdk,
        lattice.integrations,
        lattice.runtime,
    ):
        assert hasattr(mod, "__name__")


def test_lattice_core_result_monad() -> None:
    from lattice.core import (  # noqa: F401
        ConfigurationError,
        Err,
        LatticeConfig,
        LatticeError,
        Message,
        Ok,
        ProviderError,
        Request,
        Response,
        Result,
        ReversibleSyncTransform,
        Role,
        SessionError,
        SyncTransform,
        Transform,
        TransformContext,
        TransformError,
        is_err,
        is_ok,
        unwrap,
        unwrap_err,
    )


def test_no_internal_leaks() -> None:
    """Former internals must not be importable from lattice top-level."""
    lattice = importlib.import_module("lattice")
    for name in (
        "CompressorPipeline",
        "PipelineV2Wrapper",
        "decide_schedule",
        "RuntimeRouter",
    ):
        assert not hasattr(lattice, name), f"lattice still exports deleted symbol {name}"
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("lattice.evals")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("lattice.core.scheduler")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("lattice.runtime.router")
