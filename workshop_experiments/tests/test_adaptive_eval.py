"""Offline tests for the retrospective before/after eval wiring (fake predictor)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import workshop_experiments.adaptive.evaluate as ev
from workshop_experiments.adaptive.evaluate import (
    DEFAULT_ARMS,
    plan_adaptive_eval,
    run_adaptive_eval,
)
from workshop_experiments.runner import RunAccounting


class _FakePredictor:
    def __init__(self, predictor_id: str) -> None:
        """Store the predictor id."""
        self.predictor_id = predictor_id


def test_plan_lists_three_arms() -> None:
    """Plan lists three arms."""
    plan = plan_adaptive_eval()
    assert set(plan.arms) == {"untrained", "phase_a", "phase_ab"}
    text = plan.describe()
    assert "untrained" in text and "phase_ab" in text
    assert "--run" in text  # spend gate is advertised


def test_run_adaptive_eval_runs_each_arm(monkeypatch, tmp_path: Path) -> None:
    """Run adaptive eval runs each arm."""
    calls: list[tuple[str, str]] = []

    def fake_run(predictor, spec, data_service, *, store_dir, force_refresh):
        """Fake run."""
        calls.append((predictor.predictor_id, spec.spec_id))
        return RunAccounting(
            spec_id=spec.spec_id,
            predictor_id=predictor.predictor_id,
            ran_at=dt.datetime(2026, 7, 16),
            n_candidate_origins=0,
        )

    monkeypatch.setattr(ev, "run_predictor_on_spec", fake_run)

    def factory(arm: str, strategy_dir: Path, model: str) -> _FakePredictor:
        """Build a fake predictor for one arm."""
        return _FakePredictor(f"adaptive_{arm}")

    plan = plan_adaptive_eval(store_dir=tmp_path)
    results = run_adaptive_eval(plan, data_service=object(), predictor_factory=factory)

    # One run per arm; distinct predictors; the real eval spec was loaded.
    assert set(results) == {"untrained", "phase_a", "phase_ab"}
    assert {pid for pid, _ in calls} == {"adaptive_untrained", "adaptive_phase_a", "adaptive_phase_ab"}
    assert {spec_id for _, spec_id in calls} == {"sp500_ws_eval_2026_weekly"}


def test_default_arm_dirs_are_distinct() -> None:
    """Default arm dirs are distinct."""
    dirs = set(DEFAULT_ARMS.values())
    assert len(dirs) == len(DEFAULT_ARMS)
