"""Execution-time pipeline package — phase 2b-1 structural skeleton.

Phase 2b-1 lands the file moves only: ``pipeline_v2`` → ``pipeline/runner.py``,
plus the policy/guardrails/milv/auto_continuation/batch_accumulator/representation_optimizer
modules from ``core/`` and ``optimizer/``. The class formerly named
``PipelineV2`` is exported as ``Pipeline`` and ``TransformRegistryV2`` as
``PipelineTransformRegistry``.

This ``__init__.py`` is intentionally empty for Phase 2b-1: eager re-exports
would trigger a circular import while the legacy ``core/pipeline.py`` (which
imports from ``lattice.pipeline.policy``) is still alive. Callers import
through submodule paths instead:

    from lattice.pipeline.runner import Pipeline, PipelineTransformRegistry
    from lattice.pipeline.policy import OptimizationPolicy
    from lattice.pipeline.guardrails import check_entity_preservation
    from lattice.pipeline.factory import build_default_pipeline

Phase 2b-2 will delete ``core/pipeline.py`` + ``core/pipeline_v2_wrapper.py``,
rewrite the factory, rewire ``client.py`` to use ``Pipeline`` directly,
and broaden this public surface accordingly.
"""
