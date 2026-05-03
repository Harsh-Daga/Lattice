"""Shared pipeline construction helpers.

Proxy, SDK, and MCP modes should use the same compression pipeline ordering so
runtime contracts and strategy selection do not drift across integration modes.
Execution-only transforms such as batching/speculation/delta remain opt-in.

The pipeline is now built from :data:`~lattice.core.transform_registry.BUILTIN_TRANSFORMS`
so that config, ordering, and safety metadata all share one source of truth.
"""

from __future__ import annotations

from typing import Any

from lattice.core.config import LatticeConfig
from lattice.core.pipeline import CompressorPipeline
from lattice.core.transform_registry import (
    BUILTIN_TRANSFORMS,
    build_transform_instance,
    list_execution_only_names,
)

_EXECUTION_TRANSFORMS = set(list_execution_only_names())


def build_default_pipeline(
    config: LatticeConfig,
    *,
    include_execution_transforms: bool = False,
    session_manager: Any | None = None,
) -> CompressorPipeline:
    """Build the standard LATTICE transform pipeline from the registry.

    Args:
        config: Runtime configuration.
        include_execution_transforms: Include proxy-only batching,
            speculative, and delta transforms.
        session_manager: Required when execution transforms include delta
            encoding.
    """
    pipeline = CompressorPipeline(config=config)

    # Register default (non-execution) transforms
    for spec in BUILTIN_TRANSFORMS:
        if spec.execution_only:
            continue
        if not spec.default_pipeline:
            continue
        if not config.is_transform_enabled(spec.canonical_name):
            continue
        instance = build_transform_instance(config, spec)
        pipeline.register(instance)

    # Register execution-only transforms when requested
    if include_execution_transforms:
        for spec in BUILTIN_TRANSFORMS:
            if not spec.execution_only:
                continue
            if not config.is_transform_enabled(spec.canonical_name):
                continue
            if spec.canonical_name == "delta_encoder":
                # DeltaEncoder needs a session_manager injected
                if session_manager is None:
                    continue
                from lattice.transforms.delta_encode import DeltaEncoder

                instance = DeltaEncoder(session_manager=session_manager)
            else:
                instance = build_transform_instance(config, spec)
            pipeline.register(instance)

    return pipeline


def pipeline_summary(pipeline: CompressorPipeline) -> dict[str, Any]:
    """Return a stable operational summary for a pipeline."""
    transforms = [t.name for t in pipeline.transforms]
    execution = [name for name in transforms if name in _EXECUTION_TRANSFORMS]
    core = [name for name in transforms if name not in _EXECUTION_TRANSFORMS]
    return {
        "count": len(transforms),
        "transforms": transforms,
        "core_transforms": core,
        "execution_transforms": execution,
        "runtime_contract_enabled": "runtime_contract" in transforms,
        "strategy_selector_enabled": "strategy_selector" in transforms,
    }


__all__ = ["build_default_pipeline", "pipeline_summary"]
