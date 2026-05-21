"""Shared pipeline construction helpers.

Proxy, SDK, and MCP modes should use the same compression pipeline ordering so
runtime contracts and strategy selection do not drift across integration modes.
Execution-only transforms such as batching/speculation/delta remain opt-in.

Two paths:
  1. Legacy: build_default_pipeline() — 7 core transforms, no optimizer orchestration.
  2. Optimizer: build_optimizer_pipeline() — 7 core + representation_optimizer
     (which internally orchestrates structure/reference/tool/context/diagnostic).

The optimizer path is the future.  Legacy path exists for backward compat only.
"""

from __future__ import annotations

from typing import Any

from lattice.core.config import LatticeConfig
from lattice.core.pipeline import CompressorPipeline
from lattice.core.transform_registry import (
    BUILTIN_TRANSFORMS,
    OPTIMIZER_SPECS,
    TransformSpec,
    build_transform_instance,
    list_execution_only_names,
)

_EXECUTION_TRANSFORMS = set(list_execution_only_names())

# Optimizer specs map — canonical_name -> spec
_OPTIMIZER_MAP: dict[str, TransformSpec] = {s.canonical_name: s for s in OPTIMIZER_SPECS}

# Default production optimizers registered in the optimizer pipeline.
# representation_optimizer is the GLOBAL orchestrator.
_DEFAULT_OPTIMIZERS = [
    "representation_optimizer",  # priority 19
    "diagnostic_optimizer",         # priority 17
]


def build_default_pipeline(
    config: LatticeConfig,
    *,
    include_execution_transforms: bool = False,
    session_manager: Any | None = None,
) -> CompressorPipeline:
    """Build the standard (legacy) LATTICE transform pipeline.

    Only registers transforms with ``default_pipeline=True``.  Advanced
    constituent transforms must be registered explicitly or via the optimizer
    pipeline.
    """
    pipeline = CompressorPipeline(config=config)

    for spec in BUILTIN_TRANSFORMS:
        if spec.execution_only:
            continue
        if not spec.default_pipeline:
            continue
        if not config.is_transform_enabled(spec.canonical_name):
            continue
        instance = build_transform_instance(config, spec)
        pipeline.register(instance)

    if include_execution_transforms:
        for spec in BUILTIN_TRANSFORMS:
            if not spec.execution_only:
                continue
            if not config.is_transform_enabled(spec.canonical_name):
                continue
            if spec.canonical_name == "delta_encoder":
                if session_manager is None:
                    continue
                from lattice.transforms.delta_encode import DeltaEncoder

                instance = DeltaEncoder(session_manager=session_manager)
            else:
                instance = build_transform_instance(config, spec)
            pipeline.register(instance)

    return pipeline


def build_v2_pipeline(
    config: LatticeConfig,
    *,
    include_execution_transforms: bool = False,
    session_manager: Any | None = None,
) -> CompressorPipeline:
    """Build the V2 immutable pipeline.

    When config.use_v2_pipeline is True, this pipeline:
      - Uses UnifiedPlanner inside content_profiler for scheduling
      - Uses PipelineV2 for verbatim execution of ExecutionPlan
      - Has a flattened optimizer hierarchy (no nested representation_optimizer)

    If use_v2_pipeline is False, falls back to build_optimizer_pipeline().
    """
    if not getattr(config, "use_v2_pipeline", False):
        return build_optimizer_pipeline(
            config,
            include_execution_transforms=include_execution_transforms,
            session_manager=session_manager,
        )

    from lattice.core.pipeline import CompressorPipeline
    from lattice.transforms.content_profiler import ContentProfiler

    pipeline = CompressorPipeline(config=config)

    # 1. Content profiler
    profiler = ContentProfiler()
    pipeline.register(profiler)

    # 2. Runtime contract
    from lattice.transforms.runtime_contract import RuntimeContractTransform

    pipeline.register(RuntimeContractTransform())

    # 3. Cache arbitrage
    from lattice.transforms.cache_arbitrage import CacheArbitrageOptimizer

    if config.is_transform_enabled("cache_arbitrage"):
        pipeline.register(CacheArbitrageOptimizer())

    # 4. Prefix optimizer
    from lattice.transforms.prefix_opt import PrefixOptimizer

    if config.is_transform_enabled("prefix_optimizer"):
        pipeline.register(PrefixOptimizer())

    # 5. PipelineV2 wrapper — reads ExecutionPlan from context set by content_profiler.
    # This handles ALL transforms via beam search and serializes results.
    # No legacy output tail needed — PipelineV2 executes everything from the plan.
    from lattice.core.pipeline_v2_wrapper import PipelineV2Wrapper

    pipeline.register(PipelineV2Wrapper())

    # 6. Execution-only transforms
    if include_execution_transforms:
        for spec in BUILTIN_TRANSFORMS:
            if not spec.execution_only:
                continue
            if not config.is_transform_enabled(spec.canonical_name):
                continue
            if spec.canonical_name == "delta_encoder":
                if session_manager is None:
                    continue
                from lattice.transforms.delta_encode import DeltaEncoder

                instance = DeltaEncoder(session_manager=session_manager)
            else:
                instance = build_transform_instance(config, spec)
            pipeline.register(instance)

    return pipeline


def build_optimizer_pipeline(
    config: LatticeConfig,
    *,
    include_execution_transforms: bool = False,
    session_manager: Any | None = None,
) -> CompressorPipeline:
    """Build the optimizer-based pipeline.

    Core transforms (content_profiler, runtime_contract, prefix_optimizer,
    cache_arbitrage, message_dedup) always run first.  Then optimizers run in
    priority order.  Each optimizer internally selects the best constituent
    transforms via beam search.

    representation_optimizer is the global orchestrator — it coordinates
    structure_optimizer, reference_optimizer, tool_optimizer, context_optimizer,
    and diagnostic_optimizer internally.  Do NOT register those separately or
    they will run twice.
    """
    pipeline = CompressorPipeline(config=config)

    # 1. Core transforms
    core_in_optimizer = {
        "content_profiler",
        "runtime_contract",
        "prefix_optimizer",
        "cache_arbitrage",
        "strategy_selector",
    }
    for spec in BUILTIN_TRANSFORMS:
        if spec.execution_only:
            continue
        if not spec.default_pipeline:
            continue
        if spec.canonical_name not in core_in_optimizer:
            continue
        if not config.is_transform_enabled(spec.canonical_name):
            continue
        instance = build_transform_instance(config, spec)
        pipeline.register(instance)

    # 2. Optimizers
    for opt_name in _DEFAULT_OPTIMIZERS:
        _spec = _OPTIMIZER_MAP.get(opt_name)
        if _spec is None:
            continue
        if not getattr(config, _spec.config_flag, True):
            continue
        instance = build_transform_instance(config, _spec)
        pipeline.register(instance)

    # 3. Execution-only transforms when requested
    if include_execution_transforms:
        for spec in BUILTIN_TRANSFORMS:
            if not spec.execution_only:
                continue
            if not config.is_transform_enabled(spec.canonical_name):
                continue
            if spec.canonical_name == "delta_encoder":
                if session_manager is None:
                    continue
                from lattice.transforms.delta_encode import DeltaEncoder

                instance = DeltaEncoder(session_manager=session_manager)
            else:
                instance = build_transform_instance(config, spec)
            pipeline.register(instance)

    return pipeline


def build_benchmark_pipeline(config: LatticeConfig) -> CompressorPipeline:
    """Build a pipeline that exercises ALL enabled transforms + optimizers.

    Used by benchmarks and evals.  If ``use_v2_pipeline`` is True, builds
    the v2 path (UnifiedPlanner + PipelineV2).  Otherwise builds the optimizer
    path (representation_optimizer beam search).
    """
    # Clone config so we don't mutate the caller's instance
    cfg = config.model_copy() if hasattr(config, "model_copy") else config
    if getattr(cfg, "use_v2_pipeline", False):
        return build_v2_pipeline(cfg)
    cfg.use_optimizer_pipeline = True
    return build_optimizer_pipeline(cfg)


def pipeline_summary(pipeline: CompressorPipeline) -> dict[str, Any]:
    """Return a stable operational summary for a pipeline."""
    transforms = [t.name for t in pipeline.transforms]
    execution = [name for name in transforms if name in _EXECUTION_TRANSFORMS]
    core = [name for name in transforms if name not in _EXECUTION_TRANSFORMS]
    optimizers = [name for name in core if "_optimizer" in name]
    legacy = [name for name in core if name not in optimizers]
    v2 = "pipeline_v2" in transforms
    return {
        "count": len(transforms),
        "transforms": transforms,
        "core_transforms": legacy,
        "optimizers": optimizers,
        "execution_transforms": execution,
        "runtime_contract_enabled": "runtime_contract" in transforms,
        "optimizer_pipeline": len(optimizers) > 0,
        "v2_pipeline": v2,
    }


__all__ = [
    "build_default_pipeline",
    "build_optimizer_pipeline",
    "build_v2_pipeline",
    "build_benchmark_pipeline",
    "pipeline_summary",
]
