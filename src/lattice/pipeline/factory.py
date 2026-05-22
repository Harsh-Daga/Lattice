"""Pipeline construction helpers.

Single entry point: :func:`build_default_pipeline` returns a
:class:`Pipeline` backed by the lazy ``PipelineTransformRegistry``. The
v2 Pipeline owns gate orchestration via ``Pipeline.compress``; the
factory is therefore reduced to wiring config + policy.

Execution-only transforms (delta_encoder, batching, speculative) are no
longer the factory's concern — the proxy/bootstrap registers them
post-build via ``pipeline.registry.register_instance`` because they need
session-scoped dependencies (e.g. ``SessionManager``).
"""

from __future__ import annotations

from typing import Any

from lattice.core.config import LatticeConfig
from lattice.pipeline.policy import OptimizationPolicy
from lattice.pipeline.runner import Pipeline, PipelineTransformRegistry


def build_default_pipeline(config: LatticeConfig) -> Pipeline:
    """Build the standard LATTICE pipeline.

    Returns a :class:`Pipeline` with a fresh lazy registry plus a
    config-bound :class:`OptimizationPolicy`. The same instance is used
    by proxy, SDK, MCP, and benchmark code paths.
    """
    registry = PipelineTransformRegistry()
    policy = OptimizationPolicy(config)
    return Pipeline(registry=registry, config=config, policy=policy)


def build_benchmark_pipeline(config: LatticeConfig) -> Pipeline:
    """Build a pipeline configured for benchmark runs.

    Forces ``use_optimizer_pipeline=True`` on a copy of the caller's
    config so production callers are not mutated.
    """
    cfg = config.model_copy() if hasattr(config, "model_copy") else config
    cfg.use_optimizer_pipeline = True
    return build_default_pipeline(cfg)


def pipeline_summary(pipeline: Pipeline) -> dict[str, Any]:
    """Return a stable operational summary for a pipeline."""
    names = pipeline.registry.get_transform_names()
    optimizers = [n for n in names if n.endswith("_optimizer")]
    core = [n for n in names if n not in optimizers]
    return {
        "count": len(names),
        "transforms": names,
        "core_transforms": core,
        "optimizers": optimizers,
        "execution_transforms": [],
        "runtime_contract_enabled": "runtime_contract" in names,
        "optimizer_pipeline": len(optimizers) > 0,
    }


__all__ = [
    "build_default_pipeline",
    "build_benchmark_pipeline",
    "pipeline_summary",
]
