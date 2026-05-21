"""Contract-test fixtures and marker setup.

Marks:
    contract        — slow contract tests; require an externally running proxy
                      or a live network. Skipped by default.

Set the env var LATTICE_CONTRACT_FULL=1 (or pass --run-contract) to opt in.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

API_SURFACE_PATH = Path(__file__).resolve().parents[2] / "docs" / "refactor" / "api-surface.json"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-contract",
        action="store_true",
        default=False,
        help="Run contract-marked tests (slow: spins a real proxy).",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "contract: slow contract tests requiring a running proxy")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    full = config.getoption("--run-contract") or os.environ.get("LATTICE_CONTRACT_FULL") == "1"
    if full:
        return
    skip = pytest.mark.skip(reason="contract test (use --run-contract or LATTICE_CONTRACT_FULL=1)")
    for item in items:
        if item.get_closest_marker("contract") is not None:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def api_surface() -> dict:
    """Return docs/refactor/api-surface.json as a dict."""
    return json.loads(API_SURFACE_PATH.read_text())
