"""Sanity check: production source must not use pre-refactor import paths."""

from __future__ import annotations

import subprocess

OLD_PATHS = [
    "lattice.core.metrics",
    "lattice.core.telemetry",
    "lattice.core.agent_stats",
    "lattice.core.cost_estimator",
    "lattice.core.maintenance",
    "lattice.core.session",
    "lattice.core.store",
    "lattice.core.semantic_cache",
    "lattice.utils.validation",
    "lattice.utils.streaming_sketches",
    "lattice.utils.patterns",
    "lattice.core.compiler",
    "lattice.core.pipeline",
    "lattice.core.pipeline_v2",
    "lattice.core.pipeline_v2_wrapper",
    "lattice.core.pipeline_factory",
    "lattice.core.policy",
    "lattice.core.guardrails",
    "lattice.core.milv",
    "lattice.core.auto_continuation",
    "lattice.core.batch_accumulator",
    "lattice.core.scheduler",
    "lattice.core.optimizer_scheduler",
    "lattice.core.unified_planner",
    "lattice.core.task_classifier",
    "lattice.core.runtime_state",
    "lattice.core.credentials",
    "lattice.core.transport",
    "lattice.core.serialization",
    "lattice.core.delta_wire",
    "lattice.core.ir",
    "lattice.core.ir_builder",
    "lattice.core.ir_normalizer",
    "lattice.core.ir_serializer",
    "lattice.core.ir_transform",
    "lattice.core.primitives",
    "lattice.core.semantic_graph",
    "lattice.core.transform_registry",
    "lattice.core.transform_reputation",
    "lattice.core.tunnel_sidecar",
    "lattice.optimizer.ir_native_optimizer",
    "lattice.optimizer.validation",
    "lattice.optimizer.quality_estimator",
    "lattice.optimizer.representation_optimizer",
    "lattice.optimizer.structure_optimizer",
    "lattice.optimizer.ir_structure_optimizer",
    "lattice.optimizer.reference_optimizer",
    "lattice.optimizer.tool_optimizer",
    "lattice.optimizer.diagnostic_optimizer",
    "lattice.optimizer.context_optimizer",
    "lattice.transforms.prefix_opt",
    "lattice.transforms.constraint_lifting",
    "lattice.transforms.semantic_segmenter",
    "lattice.transforms.format_conv",
    "lattice.providers.base",
    "lattice.providers.openai",
    "lattice.providers.openai_compatible",
    "lattice.providers.anthropic",
    "lattice.providers.azure",
    "lattice.providers.bedrock",
    "lattice.providers.gemini",
    "lattice.providers.ollama",
    "lattice.providers.stall_detector",
    "lattice.runtime.router",
    "lattice.sdk.client",
]


def test_no_old_imports_in_src() -> None:
    """Production source code must not import from deprecated paths."""
    for path in OLD_PATHS:
        if path == "lattice.sdk.client":
            continue
        hits: list[str] = []
        for pattern in (
            f"from {path} import",
            f"from {path}.",
            f"import {path}",
        ):
            result = subprocess.run(
                ["rg", "-F", "-c", pattern, "src/lattice/"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                hits.append(f"{pattern}:\n{result.stdout.strip()}")
        if hits:
            raise AssertionError(f"OLD PATH STILL USED: {path}\n" + "\n".join(hits))
