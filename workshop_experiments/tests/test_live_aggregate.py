"""Aggregate-step determinism and schema conformance (offline)."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from workshop_experiments.live.aggregate import aggregate_step
from workshop_experiments.live.log_store import write_prediction, write_resolution
from workshop_experiments.live.schema_validation import validate


_QUANTILES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def _prediction_record(origin: str, predictor_id: str, method: str, model: str | None) -> dict:
    """Build a schema-shaped prediction record with a symmetric grid at h=1,5."""
    horizons = []
    for horizon in (1, 5):
        values = [round((q - 0.5) * 0.1, 6) for q in _QUANTILES]
        horizons.append(
            {
                "horizon": horizon,
                "point_estimate": values[5],
                "quantiles": [{"quantile": q, "value": v} for q, v in zip(_QUANTILES, values)],
            }
        )
    return {
        "schema_version": "1.1.0",
        "origin_date": origin,
        "origin_timestamp": f"{origin}T20:00:00Z",
        "submission_timestamp": f"{origin}T21:30:00Z",
        "method": method,
        "model": model,
        "predictor_id": predictor_id,
        "horizons": horizons,
        "curated_trace_summary": {"tool_calls": []},
        "langfuse_trace_id": None,
    }


def _resolution_record(
    origin: str, forecast_date: str, predictor_id: str, method: str, model: str | None, crps: float
) -> dict:
    """Build a schema-shaped resolution record at h=5."""
    return {
        "schema_version": "1.1.0",
        "origin_date": origin,
        "method": method,
        "model": model,
        "predictor_id": predictor_id,
        "horizon": 5,
        "forecast_date": forecast_date,
        "realized_value": 0.01,
        "crps": crps,
        "resolved_at": f"{forecast_date}T21:30:00Z",
    }


def _seed_log(log_dir: Path) -> None:
    """Write a small two-origin, two-method log with one resolution each."""
    for origin in ("2025-10-06", "2025-10-13"):
        write_prediction(log_dir, date.fromisoformat(origin), _prediction_record(origin, "sp500_naive", "naive", None))
        write_prediction(
            log_dir,
            date.fromisoformat(origin),
            _prediction_record(origin, "sp500_llm_process__m", "llm_process", "m"),
        )
    write_resolution(
        log_dir, date(2025, 10, 6), _resolution_record("2025-10-06", "2025-10-13", "sp500_naive", "naive", None, 0.02)
    )
    write_resolution(
        log_dir,
        date(2025, 10, 6),
        _resolution_record("2025-10-06", "2025-10-13", "sp500_llm_process__m", "llm_process", "m", 0.01),
    )


def _digest(root: Path) -> str:
    """Return a hash of every JSON file (path + bytes) under *root*."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.json")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_aggregates_validate(tmp_path: Path) -> None:
    """The generated manifest, leaderboard, and bundles all pass their schemas."""
    log_dir = tmp_path / "log"
    out_dir = tmp_path / "data"
    _seed_log(log_dir)
    aggregate_step(log_dir, out_dir)

    assert validate("manifest", json.load((out_dir / "manifest.json").open())) == []
    assert validate("leaderboard", json.load((out_dir / "leaderboard.json").open())) == []
    for bundle in (out_dir / "forecasts").glob("*.json"):
        assert validate("forecast_bundle", json.load(bundle.open())) == []


def test_aggregates_are_byte_identical_across_runs(tmp_path: Path) -> None:
    """Same log -> byte-identical aggregates (deterministic generated_at)."""
    log_dir = tmp_path / "log"
    _seed_log(log_dir)
    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    aggregate_step(log_dir, out1)
    aggregate_step(log_dir, out2)
    assert _digest(out1) == _digest(out2)


def test_leaderboard_skill_and_provenance(tmp_path: Path) -> None:
    """Skill-vs-naive is present for non-naive cells and provenance is harness."""
    log_dir = tmp_path / "log"
    out_dir = tmp_path / "data"
    _seed_log(log_dir)
    aggregate_step(log_dir, out_dir)
    leaderboard = json.load((out_dir / "leaderboard.json").open())
    assert leaderboard["generated_by"] == "harness"
    llm_cell = next(c for c in leaderboard["cells"] if c["method"] == "llm_process" and c["horizon"] == 5)
    # naive floor crps 0.02, llm crps 0.01 -> skill 1 - 0.01/0.02 = 0.5.
    assert abs(llm_cell["skill_vs_naive"] - 0.5) < 1e-9


def test_empty_log_produces_no_aggregates(tmp_path: Path) -> None:
    """Aggregating an empty log is a clean no-op."""
    assert aggregate_step(tmp_path / "log", tmp_path / "data") == {}
