"""Per-origin persistence, resume, and scoring tests with a fake predictor.

No network / no API: a synthetic DataService and a trivial in-process predictor
exercise the runner's persist-and-resume behaviour and the scoring path.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.data.features import StaticFrameAdapter
from aieng.forecasting.evaluation import MultiTargetBacktestSpec, Prediction
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from workshop_experiments.runner import (
    RunAccounting,
    load_origin_predictions,
    run_predictor_on_spec,
    run_spec,
)
from workshop_experiments.scoring import score_spec


_TARGET = "toy_target"


class _FakePredictor(Predictor):
    """Deterministic predictor: a tight symmetric grid around zero per horizon."""

    def __init__(self, predictor_id: str = "fake_zero") -> None:
        self.calls = 0
        self._predictor_id = predictor_id

    @property
    def predictor_id(self) -> str:
        return self._predictor_id

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        self.calls += 1
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        issued = datetime(2020, 1, 1)
        predictions: list[Prediction] = []
        for h in task.horizons:
            quantiles = {q: (q - 0.5) * 0.01 for q in STANDARD_QUANTILES}
            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued,
                    as_of=context.as_of,
                    forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
                    payload=ContinuousForecast(point_forecast=0.0, quantiles=quantiles),
                )
            )
        return predictions


def _toy_service() -> DataService:
    """Register one flat business-day return series with ample history."""
    idx = pd.date_range("2019-01-01", "2020-12-31", freq="B")
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "value": [0.0005 * ((i % 7) - 3) for i in range(len(idx))],
            "released_at": idx,
        }
    )
    svc = DataService()
    svc.register(
        _TARGET,
        StaticFrameAdapter(frame),
        SeriesMetadata(
            series_id=_TARGET,
            description="toy log return",
            source="synthetic",
            units="log-return",
            frequency="B",
            table_id="toy",
        ),
    )
    return svc


def _toy_spec() -> MultiTargetBacktestSpec:
    return MultiTargetBacktestSpec(
        spec_id="toy_resume",
        tasks=[
            ForecastingTask(
                task_id=_TARGET,
                target_series_id=_TARGET,
                horizons=[1],
                frequency="B",
                description="toy 1-step return",
            )
        ],
        start=datetime(2020, 6, 1),
        end=datetime(2020, 6, 10),
        stride=1,
        warmup=0,
    )


def test_runner_persists_each_origin(tmp_path: Path) -> None:
    """A fresh run writes one file per candidate origin and returns accounting."""
    spec = _toy_spec()
    predictor = _FakePredictor()
    acc = run_predictor_on_spec(predictor, spec, _toy_service(), store_dir=tmp_path)

    assert isinstance(acc, RunAccounting)
    assert acc.n_predicted == acc.n_candidate_origins > 0
    assert acc.n_cached == 0
    assert predictor.calls == acc.n_predicted

    origin_files = sorted((tmp_path / spec.spec_id / predictor.predictor_id / _TARGET).glob("*.yaml"))
    assert len(origin_files) == acc.n_predicted
    # Round-trip one persisted origin.
    loaded = load_origin_predictions(origin_files[0])
    assert loaded and isinstance(loaded[0], Prediction)

    accounting_file = tmp_path / spec.spec_id / predictor.predictor_id / "accounting.json"
    assert accounting_file.exists()


def test_runner_resumes_and_skips_persisted_origins(tmp_path: Path) -> None:
    """A second run skips every already-persisted origin (no re-predict)."""
    spec = _toy_spec()
    first = run_predictor_on_spec(_FakePredictor(), spec, _toy_service(), store_dir=tmp_path)

    second_predictor = _FakePredictor()
    second = run_predictor_on_spec(second_predictor, spec, _toy_service(), store_dir=tmp_path)

    assert second.n_predicted == 0
    assert second.n_cached == first.n_predicted
    assert second_predictor.calls == 0


def test_force_refresh_recomputes(tmp_path: Path) -> None:
    """force_refresh re-predicts even when files already exist."""
    spec = _toy_spec()
    run_predictor_on_spec(_FakePredictor(), spec, _toy_service(), store_dir=tmp_path)

    refreshed = _FakePredictor()
    acc = run_predictor_on_spec(refreshed, spec, _toy_service(), store_dir=tmp_path, force_refresh=True)
    assert acc.n_predicted > 0
    assert acc.n_cached == 0
    assert refreshed.calls == acc.n_predicted


def test_score_spec_builds_leaderboard(tmp_path: Path) -> None:
    """Scoring persisted predictions yields a leaderboard frame with finite CRPS."""
    spec = _toy_spec()
    service = _toy_service()
    run_predictor_on_spec(_FakePredictor(), spec, service, store_dir=tmp_path)

    frame, results = score_spec(spec, service, store_dir=tmp_path)
    assert not frame.empty
    assert "fake_zero" in results
    assert (frame["mean_crps"] >= 0).all()


def test_score_spec_empty_when_no_predictions(tmp_path: Path) -> None:
    """Scoring an unseen spec directory returns an empty frame, not an error."""
    frame, results = score_spec(_toy_spec(), _toy_service(), store_dir=tmp_path)
    assert frame.empty
    assert results == {}


@pytest.mark.parametrize("horizon", [1, 5])
def test_toy_predictor_multi_horizon(horizon: int, tmp_path: Path) -> None:
    """The runner handles arbitrary single horizons (sanity for h=1 and h=5)."""
    spec = MultiTargetBacktestSpec(
        spec_id=f"toy_h{horizon}",
        tasks=[
            ForecastingTask(
                task_id=_TARGET,
                target_series_id=_TARGET,
                horizons=[horizon],
                frequency="B",
                description="toy",
            )
        ],
        start=datetime(2020, 6, 1),
        end=datetime(2020, 6, 10),
        stride=1,
        warmup=0,
    )
    acc = run_predictor_on_spec(_FakePredictor(), spec, _toy_service(), store_dir=tmp_path)
    assert acc.n_predicted > 0


# ---------------------------------------------------------------------------
# Politeness pacing (--pace): sleeps between API-calling predictions only.
# ---------------------------------------------------------------------------


def test_pacing_sleeps_after_each_api_prediction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A positive delay sleeps once per successful predict() call."""
    sleeps: list[float] = []
    monkeypatch.setattr("workshop_experiments.runner.time.sleep", sleeps.append)

    spec = _toy_spec()
    predictor = _FakePredictor()
    acc = run_predictor_on_spec(predictor, spec, _toy_service(), store_dir=tmp_path, inter_prediction_delay_s=2.5)

    assert acc.n_predicted > 0
    assert sleeps == [2.5] * acc.n_predicted


def test_pacing_does_not_sleep_after_cached_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A resumed run hitting only cached origins makes no API call and never sleeps."""
    spec = _toy_spec()
    # First (unpaced) run populates the cache.
    run_predictor_on_spec(_FakePredictor(), spec, _toy_service(), store_dir=tmp_path)

    sleeps: list[float] = []
    monkeypatch.setattr("workshop_experiments.runner.time.sleep", sleeps.append)
    second = run_predictor_on_spec(
        _FakePredictor(), spec, _toy_service(), store_dir=tmp_path, inter_prediction_delay_s=5.0
    )

    assert second.n_predicted == 0
    assert second.n_cached > 0
    assert sleeps == []


def test_pacing_disabled_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The default inter_prediction_delay_s=0.0 never sleeps."""
    sleeps: list[float] = []
    monkeypatch.setattr("workshop_experiments.runner.time.sleep", sleeps.append)

    spec = _toy_spec()
    acc = run_predictor_on_spec(_FakePredictor(), spec, _toy_service(), store_dir=tmp_path)

    assert acc.n_predicted > 0
    assert sleeps == []


def test_pacing_disabled_for_conventional_predictor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """pace=False (conventional/local predictor) never sleeps, even with a delay set."""
    sleeps: list[float] = []
    monkeypatch.setattr("workshop_experiments.runner.time.sleep", sleeps.append)

    spec = _toy_spec()
    acc = run_predictor_on_spec(
        _FakePredictor(), spec, _toy_service(), store_dir=tmp_path, inter_prediction_delay_s=5.0, pace=False
    )

    assert acc.n_predicted > 0
    assert sleeps == []


def test_run_spec_exempts_conventional_predictor_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_spec paces predictors whose id is absent from conventional_predictor_ids."""
    sleeps: list[float] = []
    monkeypatch.setattr("workshop_experiments.runner.time.sleep", sleeps.append)

    spec = _toy_spec()
    service = _toy_service()
    conventional = _FakePredictor(predictor_id="conventional_fake")
    api_like = _FakePredictor(predictor_id="api_fake")

    accounting = run_spec(
        [conventional, api_like],
        spec,
        service,
        store_dir=tmp_path,
        inter_prediction_delay_s=3.0,
        conventional_predictor_ids=frozenset({"conventional_fake"}),
    )

    assert accounting["conventional_fake"].n_predicted > 0
    assert accounting["api_fake"].n_predicted > 0
    # Only the non-conventional (API-like) predictor's origins were paced.
    assert sleeps == [3.0] * accounting["api_fake"].n_predicted
