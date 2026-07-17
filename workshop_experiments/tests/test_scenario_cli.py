"""Offline tests for ``ws-scenario``: dry-run plan + injected-call execution.

No LLM/API calls anywhere in this file. The CLI's ``--run`` path (which would
build a live TSX data service and call the model) is exercised only indirectly,
through :func:`workshop_experiments.scenario.run_scenario_plan` with an
injected fake ``call`` and a fake data service — never through
``workshop_experiments.scenario.run(["--run", ...])``, which would hit the
network.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from workshop_experiments.scenario import (
    DEFAULT_ORIGINS,
    ScenarioRunResult,
    plan_scenario_run,
    run_scenario_plan,
)
from workshop_experiments.scenario_store import has_writeup, load_scenario_writeup

from workshop_experiments import scenario as scenario_cli


class _FakeContext:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def get_series(self, series_id: str) -> pd.DataFrame:  # noqa: ARG002
        return self._df


class _FakeDataService:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def context(self, as_of):  # noqa: ANN001, ARG002
        return _FakeContext(self._df)


def _synthetic_return_series(n: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    values = [0.001 * (i % 5 - 2) for i in range(n)]
    return pd.DataFrame({"timestamp": dates, "value": values})


def _fake_call_factory(calls: list[tuple[str, str]]):
    def _call(config, prompt: str, as_of: str) -> ScenarioRunResult:  # noqa: ANN001
        calls.append((config.name, as_of))
        assert as_of in prompt  # the prompt actually carries this origin's as_of
        return ScenarioRunResult(
            markdown=f"## Scenario: Soft landing (~0.5)\nwritten for {as_of}\n",
            agent_name=config.name,
            model=str(config.model),
            trace_id=f"trace-{as_of}",
            trace_url=f"https://example.test/trace-{as_of}",
        )

    return _call


# ---------------------------------------------------------------------------
# Dry-run plan
# ---------------------------------------------------------------------------


def test_plan_scenario_run_defaults_to_landmark_origins() -> None:
    """With no --origin override, the plan lists the four landmark origins."""
    plan = plan_scenario_run()
    assert [o.isoformat() for o in plan.origins] == list(DEFAULT_ORIGINS)


def test_plan_describe_lists_pending_origins(tmp_path: Path) -> None:
    """describe() reports every origin as pending in a fresh store."""
    plan = plan_scenario_run(store_dir=tmp_path)
    text = plan.describe()
    assert "tsx_analyst_scenario" in text
    for origin in DEFAULT_ORIGINS:
        assert f"{origin}: pending" in text
    assert "--run" in text  # spend gate is advertised
    assert "dry-run" in text.lower()


def test_cli_dry_run_prints_plan_and_makes_no_calls(capsys, tmp_path: Path) -> None:
    """`ws-scenario` with no --run prints the plan and returns 0; nothing written."""
    rc = scenario_cli.run(["--store-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ws-scenario plan (dry-run)" in out
    assert list(tmp_path.iterdir()) == []  # no side effects


def test_cli_dry_run_honors_origin_and_model_overrides(capsys, tmp_path: Path) -> None:
    """--origin (repeatable) and --model / --search-model override the plan."""
    rc = scenario_cli.run(
        [
            "--origin",
            "2025-04-01",
            "--origin",
            "2025-04-08",
            "--model",
            "claude-sonnet-4-6",
            "--search-model",
            "gemini-3.1-flash-lite-preview",
            "--store-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "2025-04-01" in out and "2025-04-08" in out
    assert "2026-02-25" not in out  # default origins not present when overridden
    assert "claude-sonnet-4-6" in out
    assert "gemini-3.1-flash-lite-preview" in out


# ---------------------------------------------------------------------------
# Injected-call execution (write-up persistence)
# ---------------------------------------------------------------------------


def test_run_scenario_plan_persists_writeup_and_metadata(tmp_path: Path) -> None:
    """run_scenario_plan persists writeup.md + meta.yaml per origin (injected call)."""
    calls: list[tuple[str, str]] = []
    plan = plan_scenario_run(origins=["2025-04-01"], store_dir=tmp_path)
    data_service = _FakeDataService(_synthetic_return_series())

    written = run_scenario_plan(plan, data_service, call=_fake_call_factory(calls))

    assert len(written) == 1
    assert calls == [("tsx_analyst_scenario", "2025-04-01")]
    assert has_writeup(date(2025, 4, 1), tmp_path)

    loaded = load_scenario_writeup(date(2025, 4, 1), tmp_path)
    assert loaded is not None
    assert "written for 2025-04-01" in loaded.markdown
    assert loaded.meta["origin"] == "2025-04-01"
    assert loaded.meta["agent_name"] == "tsx_analyst_scenario"
    assert loaded.meta["trace_id"] == "trace-2025-04-01"
    assert loaded.meta["trace_url"] == "https://example.test/trace-2025-04-01"
    assert "generated_at" in loaded.meta


def test_run_scenario_plan_skips_already_persisted_origins(tmp_path: Path) -> None:
    """A second run over the same origin is a no-op (resume) without --force."""
    calls: list[tuple[str, str]] = []
    plan = plan_scenario_run(origins=["2025-04-01"], store_dir=tmp_path)
    data_service = _FakeDataService(_synthetic_return_series())
    call = _fake_call_factory(calls)

    first = run_scenario_plan(plan, data_service, call=call)
    second = run_scenario_plan(plan, data_service, call=call)

    assert len(first) == 1
    assert second == []  # nothing new written
    assert len(calls) == 1  # the agent was only actually invoked once


def test_run_scenario_plan_force_refresh_reruns(tmp_path: Path) -> None:
    """force_refresh=True re-invokes the agent for an already-persisted origin."""
    calls: list[tuple[str, str]] = []
    plan = plan_scenario_run(origins=["2025-04-01"], store_dir=tmp_path)
    data_service = _FakeDataService(_synthetic_return_series())
    call = _fake_call_factory(calls)

    run_scenario_plan(plan, data_service, call=call)
    written = run_scenario_plan(plan, data_service, call=call, force_refresh=True)

    assert len(written) == 1
    assert len(calls) == 2


def test_run_scenario_plan_runs_every_origin_in_plan(tmp_path: Path) -> None:
    """Every configured origin gets its own write-up directory."""
    calls: list[tuple[str, str]] = []
    plan = plan_scenario_run(store_dir=tmp_path)  # default landmark set (4 origins)
    data_service = _FakeDataService(_synthetic_return_series())

    written = run_scenario_plan(plan, data_service, call=_fake_call_factory(calls))

    assert len(written) == len(DEFAULT_ORIGINS)
    for origin in DEFAULT_ORIGINS:
        assert has_writeup(date.fromisoformat(origin), tmp_path)
