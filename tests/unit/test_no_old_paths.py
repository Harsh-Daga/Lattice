"""Sanity check: production source must not use pre-refactor import paths."""

from __future__ import annotations

import pathlib

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

_SRC_ROOT = pathlib.Path("src/lattice")


def _line_uses_old_path(line: str, path: str) -> bool:
    if f"from {path} import" in line or f"from {path}." in line:
        return True
    needle = f"import {path}"
    idx = line.find(needle)
    if idx == -1:
        return False
    end = idx + len(needle)
    if end >= len(line):
        return True
    # Avoid prefix false positives (e.g. format_conv vs format_converter).
    return line[end] not in "._abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def test_no_old_imports_in_src() -> None:
    """Production source code must not import from deprecated paths."""
    violations: list[str] = []
    for path in OLD_PATHS:
        if path == "lattice.sdk.client":
            continue
        for py_file in _SRC_ROOT.rglob("*.py"):
            for line_no, line in enumerate(py_file.read_text().splitlines(), start=1):
                if _line_uses_old_path(line, path):
                    rel = py_file.relative_to(pathlib.Path.cwd())
                    violations.append(f"{path} at {rel}:{line_no}: {line.strip()}")
    assert not violations, "OLD PATH STILL USED:\n" + "\n".join(violations)
