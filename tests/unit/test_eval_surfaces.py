"""Tests for the deterministic benchmark surfaces."""

from __future__ import annotations

import pytest

from benchmarks.evals.runner import run_capability_eval, run_integration_eval, run_protocol_eval


@pytest.mark.asyncio
async def test_run_protocol_eval() -> None:
    section = await run_protocol_eval()
    assert section.name == "protocol_eval"
    assert section.summary["total"] >= 8
    assert section.summary["passed"] == section.summary["total"]
    assert section.details["manifest"]["segment_count"] >= 2


@pytest.mark.asyncio
async def test_run_integration_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-secret")
    monkeypatch.setenv("CUSTOM_SESSION_TOKEN", "test-session-secret")
    monkeypatch.setenv("SHARED_PASSWORD", "test-password-secret")
    section = await run_integration_eval()
    assert section.name == "integration_eval"
    assert section.summary["total"] >= 5
    assert section.summary["passed"] == section.summary["total"]
    assert section.details["checks"]["claude_env"] is True
    copilot_env = section.details["envs"]["copilot"]
    assert copilot_env["OPENAI_API_KEY"] == "<redacted>"
    assert copilot_env["CUSTOM_SESSION_TOKEN"] == "<redacted>"
    assert copilot_env["SHARED_PASSWORD"] == "<redacted>"
    assert copilot_env["COPILOT_PROVIDER_BASE_URL"] == "http://127.0.0.1:8787/v1"


@pytest.mark.asyncio
async def test_run_capability_eval() -> None:
    section = await run_capability_eval()
    assert section.name == "capability_eval"
    assert section.summary["total"] >= 4
    assert section.summary["passed"] == section.summary["total"]
    assert "openai" in {row["provider"] for row in section.details["providers"]}
