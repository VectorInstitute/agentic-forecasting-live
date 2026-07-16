"""Gap-log policy and single-run lockfile behavior (offline)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from workshop_experiments.live.config import LiveConfig, LivePredictor, RetryPolicy
from workshop_experiments.live.lockfile import LockHeld, RunLock
from workshop_experiments.live.log_store import has_prediction, iter_gap_entries
from workshop_experiments.live.predict import PredictionUnavailable, predict_step
from workshop_experiments.live.schema_validation import validate


ORIGIN = date(2025, 10, 20)


def _grid() -> dict[float, float]:
    """Return a valid symmetric 11-point grid."""
    return {q: round((q - 0.5) * 0.1, 6) for q in STANDARD_QUANTILES}


def _mini_config(tmp_path: Path) -> LiveConfig:
    """Build a two-rung config: one rung that succeeds, one that always fails."""
    good = LivePredictor("naive", None, "naive", None, "sp500_naive", "conventional")
    bad = LivePredictor("ets", None, "classical", "ets", "sp500_ets", "conventional")
    return LiveConfig(
        schema_version="1.0.0",
        target_ticker="^GSPC",
        horizons=(1,),
        submission_time_local="17:30",
        timezone="America/Toronto",
        retry=RetryPolicy(max_attempts=3, backoff_minutes=20, hard_stop_local="21:00"),
        log_dir=tmp_path / "log",
        aggregates_dir=tmp_path / "data",
        smoke_store=tmp_path / "smoke",
        predictors=(good, bad),
    )


@dataclass
class _MixedSource:
    """Serves a valid prediction for sp500_naive; fails for everything else."""

    origin: date

    def predictions_for(self, live: LivePredictor) -> list[Any]:
        """Return a good prediction for naive, else raise."""
        if live.predictor_id != "sp500_naive":
            raise PredictionUnavailable("simulated method failure")
        return [
            Prediction(
                predictor_id="registry",
                task_id="sp500_logret_1b",
                issued_at=datetime(2025, 10, 20),
                as_of=datetime(2025, 10, 20),
                forecast_date=datetime(2025, 10, 21),
                payload=ContinuousForecast(point_forecast=_grid()[0.5], quantiles=_grid()),
            )
        ]


def test_failed_method_is_gapped_and_run_continues(tmp_path: Path) -> None:
    """A failing rung logs a gap; the good rung still writes its record."""
    config = _mini_config(tmp_path)
    result = predict_step(config, _MixedSource(ORIGIN), submission_timestamp="2025-10-20T21:30:00Z")

    assert result.written == ["sp500_naive"]
    assert result.gapped == ["sp500_ets"]
    assert has_prediction(config.log_dir, ORIGIN, "sp500_naive")
    assert not has_prediction(config.log_dir, ORIGIN, "sp500_ets")

    gaps = iter_gap_entries(config.log_dir)
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap["scope"] == "sp500_ets"
    assert gap["retries_attempted"] == 3
    assert validate("gap_log", gap) == []


def test_rerun_skips_existing_records(tmp_path: Path) -> None:
    """A second predict pass is idempotent: existing records are skipped."""
    config = _mini_config(tmp_path)
    predict_step(config, _MixedSource(ORIGIN), submission_timestamp="2025-10-20T21:30:00Z")
    second = predict_step(config, _MixedSource(ORIGIN), submission_timestamp="2025-10-20T21:30:00Z")
    assert second.skipped == ["sp500_naive"]


def test_lock_blocks_a_second_holder(tmp_path: Path) -> None:
    """Acquiring a held lock raises LockHeld; releasing frees it."""
    lock_path = tmp_path / ".ws-live-run.lock"
    first = RunLock(lock_path)
    first.acquire()
    try:
        with pytest.raises(LockHeld):
            RunLock(lock_path).acquire()
    finally:
        first.release()

    # Once released, the lock is re-acquirable.
    with RunLock(lock_path):
        pass


def test_lock_released_on_context_exit(tmp_path: Path) -> None:
    """The context manager releases the lock on exit."""
    lock_path = tmp_path / ".ws-live-run.lock"
    with RunLock(lock_path):
        pass
    with RunLock(lock_path):
        pass  # would raise if still held
