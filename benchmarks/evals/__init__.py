"""Production-grade evaluation suite for LATTICE."""

from .catalog import EvalTarget, default_provider_targets, default_scenarios
from .report import EvalSectionReport, ProductionEvalReport
from .runner import (
    run_capability_eval,
    run_control_plane_eval,
    run_feature_eval,
    run_feature_matrix_eval,
    run_integration_eval,
    run_production_evals,
    run_protocol_eval,
    run_provider_eval,
    run_replay_eval,
    run_tacc_eval,
)

__all__ = [
    "EvalTarget",
    "EvalSectionReport",
    "ProductionEvalReport",
    "default_provider_targets",
    "default_scenarios",
    "run_control_plane_eval",
    "run_capability_eval",
    "run_feature_eval",
    "run_feature_matrix_eval",
    "run_integration_eval",
    "run_provider_eval",
    "run_production_evals",
    "run_protocol_eval",
    "run_replay_eval",
    "run_tacc_eval",
]
