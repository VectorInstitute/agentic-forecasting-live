"""Resolver CRPS math and resolve-step behavior (offline)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from workshop_experiments.live.log_store import iter_resolution_records, write_prediction
from workshop_experiments.live.resolve import (
    LookupRealizedProvider,
    crps_from_grid,
    resolve_log,
)
from workshop_experiments.live.schema_validation import validate


def _grid(values: list[float]) -> list[dict[str, float]]:
    """Build a quantile grid (levels irrelevant to CRPS, values are the ensemble)."""
    return [{"quantile": 0.5, "value": v} for v in values]


def test_crps_two_member_ensemble_closed_form() -> None:
    """CRPS of the ensemble {0, 1} against 0 equals the hand-computed 0.25.

    CRPS = (1/m) Σ|x_i - y| − (1/(2 m^2)) Σ_i Σ_j |x_i − x_j|
         = (1/2)(0 + 1) − (1/8)(0 + 1 + 1 + 0) = 0.5 − 0.25 = 0.25.
    """
    assert crps_from_grid(_grid([0.0, 1.0]), 0.0) == 0.25


def test_crps_point_mass_is_absolute_error() -> None:
    """A degenerate (all-equal) ensemble reduces CRPS to |forecast − actual|."""
    assert crps_from_grid(_grid([0.3, 0.3, 0.3]), 0.5) == abs(0.3 - 0.5)


def test_crps_is_permutation_invariant() -> None:
    """The grid is sorted internally, so member order does not matter."""
    a = crps_from_grid(_grid([0.1, 0.2, 0.9]), 0.4)
    b = crps_from_grid(_grid([0.9, 0.1, 0.2]), 0.4)
    assert a == b


def _prediction_record(origin: str, predictor_id: str, horizon: int, grid_values: list[float]) -> dict:
    """Build a minimal schema-shaped prediction record for the resolver to read."""
    quantiles = [
        {"quantile": q, "value": v}
        for q, v in zip(
            [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
            grid_values,
        )
    ]
    return {
        "schema_version": "1.0.0",
        "origin_date": origin,
        "origin_timestamp": f"{origin}T20:00:00Z",
        "submission_timestamp": f"{origin}T21:30:00Z",
        "method": "llm_process",
        "model": "gemini-3.5-flash",
        "predictor_id": predictor_id,
        "horizons": [{"horizon": horizon, "point_estimate": grid_values[5], "quantiles": quantiles}],
        "curated_trace_summary": {"tool_calls": []},
        "langfuse_trace_id": None,
    }


def test_resolve_writes_scored_records_for_matured_horizons(tmp_path: Path) -> None:
    """resolve_log resolves only matured horizons and scores them correctly."""
    log_dir = tmp_path / "log"
    grid_values = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    record = _prediction_record("2025-10-06", "sp500_llm_process__x", 5, grid_values)
    write_prediction(log_dir, date(2025, 10, 6), record)

    # Realized 5b return known only at 2025-10-13 (= origin + 5 business days).
    realized = -0.012775
    provider = LookupRealizedProvider({5: {date(2025, 10, 13): realized}})

    written = resolve_log(log_dir, provider, resolved_at="2025-10-13T21:30:00Z")
    assert len(written) == 1
    res = written[0]
    assert res["forecast_date"] == "2025-10-13"
    assert res["realized_value"] == realized
    assert res["crps"] == crps_from_grid(record["horizons"][0]["quantiles"], realized)
    assert validate("resolution", res) == []


def test_resolve_is_idempotent(tmp_path: Path) -> None:
    """A second resolve pass writes nothing new (records are append-once)."""
    log_dir = tmp_path / "log"
    grid_values = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    write_prediction(log_dir, date(2025, 10, 6), _prediction_record("2025-10-06", "sp500_x", 5, grid_values))
    provider = LookupRealizedProvider({5: {date(2025, 10, 13): 0.01}})

    first = resolve_log(log_dir, provider, resolved_at="2025-10-13T21:30:00Z")
    second = resolve_log(log_dir, provider, resolved_at="2025-10-14T21:30:00Z")
    assert len(first) == 1
    assert second == []
    assert len(iter_resolution_records(log_dir)) == 1


def test_unmatured_horizon_is_not_resolved(tmp_path: Path) -> None:
    """A horizon whose target session has not closed stays unresolved."""
    log_dir = tmp_path / "log"
    grid_values = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    write_prediction(log_dir, date(2025, 10, 20), _prediction_record("2025-10-20", "sp500_x", 5, grid_values))
    provider = LookupRealizedProvider({5: {date(2025, 10, 13): 0.01}})  # nothing at 2025-10-27
    assert resolve_log(log_dir, provider, resolved_at="2025-10-27T21:30:00Z") == []
