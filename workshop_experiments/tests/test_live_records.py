"""Prediction-record construction, writer invariants, and schema conformance."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from workshop_experiments.live.config import LivePredictor, load_config
from workshop_experiments.live.records import (
    WriterInvariantError,
    build_prediction_record,
    build_resolution_record,
    curated_trace_summary,
    group_predictions_by_horizon,
    market_close_timestamp,
)
from workshop_experiments.live.schema_validation import validate


ORIGIN = date(2025, 10, 20)


def _symmetric_grid(center: float = 0.0, scale: float = 0.1) -> dict[float, float]:
    """Return a valid, non-decreasing 11-point grid centred on *center*."""
    return {q: round(center + (q - 0.5) * scale, 6) for q in STANDARD_QUANTILES}


def _prediction(task_id: str, quantiles: dict[float, float], *, metadata: dict | None = None) -> Prediction:
    """Build one Prediction with a ContinuousForecast payload."""
    return Prediction(
        predictor_id="registry_id",
        task_id=task_id,
        issued_at=datetime(2025, 10, 20, 21, 30),
        as_of=datetime(2025, 10, 20),
        forecast_date=datetime(2025, 10, 21),
        payload=ContinuousForecast(point_forecast=quantiles[0.50], quantiles=quantiles),
        metadata=metadata or {},
    )


def _live(schema_method: str = "llm_process", model_label: str | None = "gemini-3.5-flash") -> LivePredictor:
    """Return a minimal LivePredictor identity for record building."""
    return LivePredictor(
        registry_method="llmp_qgrid",
        model="gemini-3.5-flash",
        schema_method=schema_method,
        model_label=model_label,
        predictor_id="sp500_llm_process__gemini-3.5-flash",
        group="llmp",
    )


def test_market_close_timestamp_is_schema_shaped() -> None:
    """The origin close timestamp is a UTC schema timestamp ending in Z."""
    ts = market_close_timestamp(ORIGIN)
    assert ts.endswith("Z")
    assert ts.startswith("2025-10-20T")


def test_build_prediction_record_conforms_to_schema() -> None:
    """A full multi-horizon record validates against prediction.schema.json."""
    config = load_config()
    preds = {
        h: _prediction(config.task_id_for_horizon(h), _symmetric_grid(), metadata={"rationale": "steady"})
        for h in config.horizons
    }
    record = build_prediction_record(_live(), preds, origin=ORIGIN, submission_timestamp="2025-10-20T21:30:00Z")
    assert validate("prediction", record) == []
    assert record["rationale"] == "steady"
    assert len(record["horizons"]) == 3
    assert record["horizons"][0]["point_estimate"] == record["horizons"][0]["quantiles"][5]["value"]


def test_conventional_record_has_no_rationale_and_null_model() -> None:
    """Conventional rungs write null model label and omit rationale/trace."""
    live = LivePredictor("naive", None, "naive", None, "sp500_naive", "conventional")
    preds = {1: _prediction("sp500_logret_1b", _symmetric_grid())}
    record = build_prediction_record(live, preds, origin=ORIGIN, submission_timestamp="2025-10-20T21:30:00Z")
    assert record["model"] is None
    assert "rationale" not in record
    assert record["langfuse_trace_id"] is None
    assert validate("prediction", record) == []


def test_non_decreasing_invariant_enforced() -> None:
    """A grid whose values decrease is rejected before writing."""
    bad = _symmetric_grid()
    bad[0.90] = -1.0  # break monotonicity
    with pytest.raises(WriterInvariantError, match="non-decreasing"):
        build_prediction_record(
            _live(),
            {1: _prediction("sp500_logret_1b", bad)},
            origin=ORIGIN,
            submission_timestamp="2025-10-20T21:30:00Z",
        )


def test_point_estimate_must_equal_median() -> None:
    """point_estimate != q50 is rejected."""
    grid = _symmetric_grid()
    pred = Prediction(
        predictor_id="r",
        task_id="sp500_logret_1b",
        issued_at=datetime(2025, 10, 20),
        as_of=datetime(2025, 10, 20),
        forecast_date=datetime(2025, 10, 21),
        payload=ContinuousForecast(point_forecast=0.5, quantiles=grid),  # q50 is 0.0
    )
    with pytest.raises(WriterInvariantError, match="0.50 quantile"):
        build_prediction_record(_live(), {1: pred}, origin=ORIGIN, submission_timestamp="2025-10-20T21:30:00Z")


def test_incomplete_grid_rejected() -> None:
    """A grid that is not the standard 11-point set is rejected."""
    partial = {0.05: -0.1, 0.5: 0.0, 0.95: 0.1}
    pred = Prediction(
        predictor_id="r",
        task_id="sp500_logret_1b",
        issued_at=datetime(2025, 10, 20),
        as_of=datetime(2025, 10, 20),
        forecast_date=datetime(2025, 10, 21),
        payload=ContinuousForecast(point_forecast=0.0, quantiles=partial),
    )
    with pytest.raises(WriterInvariantError, match="standard 11-point grid"):
        build_prediction_record(_live(), {1: pred}, origin=ORIGIN, submission_timestamp="2025-10-20T21:30:00Z")


def test_curated_trace_summary_keeps_only_tool_and_title() -> None:
    """Curation drops everything but tool names and query titles."""
    metadata = {
        "tool_calls": [
            {"tool": "search_web", "query_title": "Fed guidance", "body": "SECRET ARTICLE TEXT"},
            {"query_title": "no tool -> dropped"},
        ],
        "prompt": "internal scaffolding",
    }
    summary = curated_trace_summary(metadata)
    assert summary == {"tool_calls": [{"tool": "search_web", "query_title": "Fed guidance"}]}


def test_group_predictions_by_horizon_uses_task_binding() -> None:
    """Predictions are keyed to horizons via the task-id binding."""
    config = load_config()
    preds = [_prediction(config.task_id_for_horizon(h), _symmetric_grid()) for h in config.horizons]
    by_h = group_predictions_by_horizon(config, preds)
    assert set(by_h) == set(config.horizons)


def test_resolution_record_conforms_to_schema() -> None:
    """A resolution record validates against resolution.schema.json."""
    record = build_resolution_record(
        _live(),
        origin=ORIGIN,
        horizon=5,
        forecast_date=date(2025, 10, 27),
        realized_value=-0.012,
        crps=0.004,
        resolved_at="2025-10-27T21:30:00Z",
    )
    assert validate("resolution", record) == []
