"""Offline tests for ``ws-scenario-judge``: dry-run plan + injected-judge execution.

No LLM/API calls anywhere in this file: the judge call and the realized-series
lookup are both injected. ``run(["--run", ...])`` (which would build a live TSX
data service and call the judge model) is never invoked directly.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from workshop_experiments.data_tsx import build_cumulative_log_return_frame
from workshop_experiments.scenario import DEFAULT_ORIGINS
from workshop_experiments.scenario_judge import (
    DEFAULT_JUDGE_MODEL,
    ScenarioJudgeVerdict,
    plan_scenario_judge,
    run_scenario_judge_plan,
)
from workshop_experiments.scenario_outcome import JUDGE_HORIZONS
from workshop_experiments.scenario_store import (
    ScenarioWriteup,
    has_judge_verdict,
    load_judge_verdict,
    write_scenario_writeup,
)

from workshop_experiments import scenario_judge as judge_cli


#: The origin every test below judges against. The synthetic price series
#: below is anchored so this date lands at business-day index 64 — past the
#: window-60 warmup (>=60 rows before it) with 335 rows still ahead (>=60
#: after it) — so every JUDGE_HORIZONS window matures and the "Nth business
#: day after origin" indexing lines up (see test_scenario_outcome.py's module
#: docstring for why that margin matters).
_ORIGIN = date(2025, 4, 1)


def _synthetic_price_series(*, start: str = "2025-01-01", n: int = 400, daily_growth: float = 0.001) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n)
    closes = [100.0 * (1.0 + daily_growth) ** i for i in range(n)]
    return pd.DataFrame({"timestamp": dates, "value": closes})


def _origin_index(price: pd.DataFrame, origin: date = _ORIGIN) -> int:
    return int(pd.DatetimeIndex(price["timestamp"]).get_loc(pd.Timestamp(origin)))


def _windows_for(price: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {h: build_cumulative_log_return_frame(price, window=h) for h in JUDGE_HORIZONS}


def _fake_judge_factory(calls: list[dict]):
    def _judge(*, origin, writeup_markdown, realized_outcome_summary, model):  # noqa: ANN001
        calls.append(
            {
                "origin": origin,
                "writeup_markdown": writeup_markdown,
                "n_outcomes": len(realized_outcome_summary.outcomes),
                "model": model,
            }
        )
        return ScenarioJudgeVerdict(
            drivers_score=4,
            drivers_justification="Cited BoC and oil moves that match the realized period.",
            calibration_score=3,
            calibration_justification="The realized-direction scenario was not the highest-probability one.",
            specificity_score=5,
            specificity_justification="Named specific dated catalysts.",
            overall_justification="Solid but slightly overconfident in the wrong scenario.",
        )

    return _judge


# ---------------------------------------------------------------------------
# Dry-run plan
# ---------------------------------------------------------------------------


def test_plan_scenario_judge_defaults_to_landmark_origins() -> None:
    """With no --origin override, the plan lists ws-scenario's landmark origins."""
    plan = plan_scenario_judge()
    assert [o.isoformat() for o in plan.origins] == list(DEFAULT_ORIGINS)
    assert plan.model == DEFAULT_JUDGE_MODEL


def test_plan_describe_reports_missing_writeups(tmp_path: Path) -> None:
    """describe() flags every origin as MISSING write-up in a fresh store."""
    plan = plan_scenario_judge(origins=["2025-04-01"], store_dir=tmp_path)
    text = plan.describe()
    assert "claude-sonnet-4-6" in text
    assert "MISSING write-up" in text
    assert "not yet judged" in text
    assert "5" in text and "21" in text and "60" in text  # horizons advertised


def test_plan_describe_reports_present_and_judged(tmp_path: Path) -> None:
    """describe() distinguishes write-up-present / already-judged origins."""
    origin = date(2025, 4, 1)
    write_scenario_writeup(ScenarioWriteup(origin=origin, markdown="x", meta={}), tmp_path)
    plan = plan_scenario_judge(origins=["2025-04-01"], store_dir=tmp_path)
    text = plan.describe()
    assert "write-up present" in text
    assert "not yet judged" in text


def test_cli_dry_run_prints_plan_and_makes_no_calls(capsys, tmp_path: Path) -> None:
    """`ws-scenario-judge` with no --run prints the plan; nothing written."""
    rc = judge_cli.run(["--store-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ws-scenario-judge plan (dry-run)" in out
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Injected-judge execution (verdict persistence)
# ---------------------------------------------------------------------------


def test_run_scenario_judge_plan_persists_verdict_and_realized_outcome(tmp_path: Path) -> None:
    """The verdict + realized-outcome numbers are persisted to judge.yaml."""
    origin = date(2025, 4, 1)
    write_scenario_writeup(
        ScenarioWriteup(origin=origin, markdown="## Scenario: Soft landing (~0.5)\n...", meta={}), tmp_path
    )
    price = _synthetic_price_series()
    windows = _windows_for(price)
    calls: list[dict] = []

    plan = plan_scenario_judge(origins=["2025-04-01"], store_dir=tmp_path)
    written = run_scenario_judge_plan(
        plan, get_series=lambda h: windows[h], judge=_fake_judge_factory(calls), force_refresh=False
    )

    assert len(written) == 1
    assert has_judge_verdict(origin, tmp_path)
    assert len(calls) == 1
    assert calls[0]["model"] == DEFAULT_JUDGE_MODEL
    assert calls[0]["n_outcomes"] == len(JUDGE_HORIZONS)

    record = load_judge_verdict(origin, tmp_path)
    assert record is not None
    assert record["judge_model"] == DEFAULT_JUDGE_MODEL
    assert record["verdict"]["drivers_score"] == 4
    assert record["verdict"]["calibration_score"] == 3
    assert record["verdict"]["specificity_score"] == 5
    assert len(record["realized_outcome"]["horizons"]) == len(JUDGE_HORIZONS)
    # Hand-check the 5-business-day realized return against the price series.
    origin_idx = _origin_index(price)
    five_day = next(h for h in record["realized_outcome"]["horizons"] if h["horizon"] == 5)
    expected = math.log(price["value"].iloc[origin_idx + 5] / price["value"].iloc[origin_idx])
    assert five_day["log_return"] == pytest.approx(expected)
    assert five_day["forecast_date"] == price["timestamp"].iloc[origin_idx + 5].date().isoformat()


def test_run_scenario_judge_plan_skips_missing_writeup(tmp_path: Path) -> None:
    """An origin with no persisted write-up is skipped, not errored."""
    calls: list[dict] = []
    plan = plan_scenario_judge(origins=["2025-04-01"], store_dir=tmp_path)

    written = run_scenario_judge_plan(plan, get_series=lambda h: pd.DataFrame(), judge=_fake_judge_factory(calls))

    assert written == []
    assert calls == []
    assert not has_judge_verdict(date(2025, 4, 1), tmp_path)


def test_run_scenario_judge_plan_skips_already_judged(tmp_path: Path) -> None:
    """A second pass over an already-judged origin is a no-op without --force."""
    origin = date(2025, 4, 1)
    write_scenario_writeup(ScenarioWriteup(origin=origin, markdown="x", meta={}), tmp_path)
    price = _synthetic_price_series()
    windows = _windows_for(price)
    calls: list[dict] = []
    plan = plan_scenario_judge(origins=["2025-04-01"], store_dir=tmp_path)
    judge = _fake_judge_factory(calls)

    first = run_scenario_judge_plan(plan, get_series=lambda h: windows[h], judge=judge)
    second = run_scenario_judge_plan(plan, get_series=lambda h: windows[h], judge=judge)

    assert len(first) == 1
    assert second == []
    assert len(calls) == 1


def test_run_scenario_judge_plan_force_refresh_rejudges(tmp_path: Path) -> None:
    """force_refresh=True re-invokes the judge for an already-judged origin."""
    origin = date(2025, 4, 1)
    write_scenario_writeup(ScenarioWriteup(origin=origin, markdown="x", meta={}), tmp_path)
    price = _synthetic_price_series()
    windows = _windows_for(price)
    calls: list[dict] = []
    plan = plan_scenario_judge(origins=["2025-04-01"], store_dir=tmp_path)
    judge = _fake_judge_factory(calls)

    run_scenario_judge_plan(plan, get_series=lambda h: windows[h], judge=judge)
    written = run_scenario_judge_plan(plan, get_series=lambda h: windows[h], judge=judge, force_refresh=True)

    assert len(written) == 1
    assert len(calls) == 2
