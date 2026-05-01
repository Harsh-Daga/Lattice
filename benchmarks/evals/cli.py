"""CLI entrypoint for production evals."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ruff: isort: off
from benchmarks.evals.catalog import default_scenarios  # noqa: E402
from benchmarks.evals.report import EvalSectionReport, ProductionEvalReport, render_markdown  # noqa: E402
from benchmarks.evals.runner import (  # noqa: E402
    run_capability_eval,
    run_control_plane_eval,
    run_feature_eval,
    run_feature_matrix_eval,
    run_integration_eval,
    run_production_evals,
    run_provider_eval,
    run_protocol_eval,
    run_transport_eval,
    run_replay_eval,
    run_replay_feature_isolated,
    run_replay_governance,
    run_tacc_eval,
    write_production_eval_outputs,
)
# ruff: isort: on


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run production-grade LATTICE evals")
    parser.add_argument(
        "--suite",
        default="all",
        choices=["all", "feature", "feature-matrix", "provider", "protocol", "transport", "integration", "capability", "replay", "replay-isolated", "replay-governance", "tacc", "control"],
    )
    parser.add_argument("--scenarios", nargs="*", default=[], help="Optional scenario filter")
    parser.add_argument("--providers", nargs="*", default=[], help="Optional provider filter")
    parser.add_argument(
        "--provider-model",
        action="append",
        default=[],
        metavar="PROVIDER=MODEL",
        help="Override the model for a specific provider. Example: --provider-model ollama=kimi-k2.6-cloud",
    )
    parser.add_argument("--replay-input", default="benchmarks/datasets/replay_traces.jsonl")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--provider-iterations", type=int, default=1)
    parser.add_argument("--provider-warmup", type=int, default=1)
    parser.add_argument("--regression-threshold-quality", type=float, default=0.05, help="Max acceptable quality drop (default 0.05 = 5%)")
    parser.add_argument("--regression-threshold-latency", type=float, default=1.5, help="Max acceptable latency multiplier (default 1.5 = 50% increase)")
    parser.add_argument("--output-json", default="benchmarks/results/production_evals.json")
    parser.add_argument("--output-md", default="benchmarks/results/production_evals.md")
    parser.add_argument("--json-only", action="store_true")
    return parser.parse_args()


def _parse_provider_models(items: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid provider-model override: {item!r}")
        provider, model = item.split("=", 1)
        provider = provider.strip()
        model = model.strip()
        if not provider or not model:
            raise ValueError(f"Invalid provider-model override: {item!r}")
        overrides[provider] = model
    return overrides


def _render_single_section(section: EvalSectionReport, runner_name: str) -> str:
    report = ProductionEvalReport(runner_name=runner_name, sections=[section])
    return render_markdown(report)


async def main() -> int:
    args = _parse_args()
    provider_models = _parse_provider_models(args.provider_model)
    scenarios = default_scenarios(args.scenarios or None)

    try:
        if args.suite == "feature":
            section = await run_feature_eval(
                scenarios=scenarios,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            payload = section.to_dict()
            if args.json_only:
                import json

                print(json.dumps(payload, indent=2))
            else:
                print(_render_single_section(section, "feature_eval"))
            return 0

        if args.suite == "feature-matrix":
            section = await run_feature_matrix_eval(
                input_path=args.replay_input,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            payload = section.to_dict()
            if args.json_only:
                import json

                print(json.dumps(payload, indent=2))
            else:
                print(_render_single_section(section, "feature_matrix_eval"))
            return 0

        if args.suite == "provider":
            section = await run_provider_eval(
                providers=args.providers or None,
                model_overrides=provider_models,
                scenarios=scenarios,
                iterations=args.provider_iterations,
                warmup=args.provider_warmup,
            )
            payload = section.to_dict()
            if args.json_only:
                import json

                print(json.dumps(payload, indent=2))
            else:
                print(_render_single_section(section, "provider_eval"))
            return 0

        if args.suite == "protocol":
            section = await run_protocol_eval()
            payload = section.to_dict()
            if args.json_only:
                import json

                print(json.dumps(payload, indent=2))
            else:
                print(_render_single_section(section, "protocol_eval"))
            return 0

        if args.suite == "transport":
            section = await run_transport_eval()
            payload = section.to_dict()
            if args.json_only:
                import json

                print(json.dumps(payload, indent=2))
            else:
                print(_render_single_section(section, "transport_eval"))
            return 0

        if args.suite == "integration":
            section = await run_integration_eval()
            payload = section.to_dict()
            if args.json_only:
                import json

                print(json.dumps(payload, indent=2))
            else:
                print(_render_single_section(section, "integration_eval"))
            return 0

        if args.suite == "capability":
            section = await run_capability_eval()
            payload = section.to_dict()
            if args.json_only:
                import json

                print(json.dumps(payload, indent=2))
            else:
                print(_render_single_section(section, "capability_eval"))
            return 0

        if args.suite == "replay":
            section = await run_replay_eval(
                input_path=args.replay_input,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            payload = section.to_dict()
            if args.json_only:
                import json

                print(json.dumps(payload, indent=2))
            else:
                print(_render_single_section(section, "replay_eval"))
            return 0

        if args.suite == "replay-isolated":
            sections = await run_replay_feature_isolated(
                input_path=args.replay_input,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            if args.json_only:
                import json

                print(json.dumps({name: s.to_dict() for name, s in sections.items()}, indent=2))
            else:
                for name, section in sections.items():
                    print(_render_single_section(section, f"replay_isolated_{name}"))
            return 0

        if args.suite == "replay-governance":
            section = await run_replay_governance(
                input_path=args.replay_input,
                iterations=args.iterations,
                warmup=args.warmup,
                regression_threshold_quality=args.regression_threshold_quality,
                regression_threshold_latency=args.regression_threshold_latency,
            )
            payload = section.to_dict()
            if args.json_only:
                import json

                print(json.dumps(payload, indent=2))
            else:
                print(_render_single_section(section, "replay_governance"))
            return 0

        if args.suite == "tacc":
            section = await run_tacc_eval()
            payload = section.to_dict()
            if args.json_only:
                import json

                print(json.dumps(payload, indent=2))
            else:
                print(_render_single_section(section, "tacc_eval"))
            return 0

        if args.suite == "control":
            section = await run_control_plane_eval()
            payload = section.to_dict()
            if args.json_only:
                import json

                print(json.dumps(payload, indent=2))
            else:
                print(_render_single_section(section, "control_plane_eval"))
            return 0

        report = await run_production_evals(
            scenarios=args.scenarios or None,
            providers=args.providers or None,
            model_overrides=provider_models,
            replay_input=args.replay_input,
            iterations=args.iterations,
            warmup=args.warmup,
            provider_iterations=args.provider_iterations,
            provider_warmup=args.provider_warmup,
        )
        write_production_eval_outputs(report, output_json=args.output_json, output_md=args.output_md)
        if args.json_only:
            import json

            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(render_markdown(report))
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
