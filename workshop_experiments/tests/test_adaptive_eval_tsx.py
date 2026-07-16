"""Offline tests for the TSX before/after eval wiring (fake predictor)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import workshop_experiments.adaptive.evaluate as ev
from workshop_experiments.adaptive.evaluate import plan_adaptive_eval, run_adaptive_eval
from workshop_experiments.runner import RunAccounting


class _FakePredictor:
    def __init__(self, predictor_id: str) -> None:
        """Store the predictor id."""
        self.predictor_id = predictor_id


def test_tsx_plan_lists_two_arms() -> None:
    """The TSX plan is the two-arm pre/post shape over the 2026 weekly eval spec."""
    plan = plan_adaptive_eval(domain="tsx")
    assert plan.domain == "tsx"
    assert set(plan.arms) == {"untrained", "trained"}
    assert plan.spec_name == "tsx_ws_eval_2026_weekly"
    text = plan.describe()
    assert "tsx" in text and "untrained" in text and "trained" in text
    assert "--run" in text  # spend gate advertised


def test_tsx_arm_dirs_are_distinct() -> None:
    """The untrained (seed) and trained arm dirs are distinct (cache separation)."""
    plan = plan_adaptive_eval(domain="tsx")
    assert plan.arms["untrained"] != plan.arms["trained"]
    assert plan.arms["untrained"].name == "tsx-strategy"
    assert plan.arms["trained"].name == "tsx-strategy-trained"


def test_tsx_run_expands_each_arm_with_distinct_predictor_ids(monkeypatch, tmp_path: Path) -> None:
    """Each TSX arm runs once against the tsx eval spec with a distinct predictor id."""
    calls: list[tuple[str, str]] = []

    def fake_run(predictor, spec, data_service, *, store_dir, force_refresh):
        """Record the (predictor_id, spec_id) and return an empty accounting."""
        calls.append((predictor.predictor_id, spec.spec_id))
        return RunAccounting(
            spec_id=spec.spec_id,
            predictor_id=predictor.predictor_id,
            ran_at=dt.datetime(2026, 7, 16),
            n_candidate_origins=0,
        )

    monkeypatch.setattr(ev, "run_predictor_on_spec", fake_run)

    def factory(arm: str, strategy_dir: Path, model: str) -> _FakePredictor:
        """Build a fake, arm-distinct predictor."""
        return _FakePredictor(f"tsx_adaptive_{arm}")

    plan = plan_adaptive_eval(domain="tsx", store_dir=tmp_path)
    results = run_adaptive_eval(plan, data_service=object(), predictor_factory=factory)

    assert set(results) == {"untrained", "trained"}
    assert {pid for pid, _ in calls} == {"tsx_adaptive_untrained", "tsx_adaptive_trained"}
    assert {spec_id for _, spec_id in calls} == {"tsx_ws_eval_2026_weekly"}
