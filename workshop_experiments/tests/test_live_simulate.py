"""End-to-end offline simulate pipeline (no API, no network) — runs in CI.

Deliberately NOT marked ``integration_test`` so the default CI job
(``pytest -m "not integration_test"``) collects it. It exercises the full
write -> resolve -> aggregate -> validate chain against the committed smoke
predictions in a temp dir.
"""

from __future__ import annotations

import json
from pathlib import Path

from workshop_experiments.live.config import load_config
from workshop_experiments.live.schema_validation import validate
from workshop_experiments.live.simulate import run_simulation, smoke_origins


def test_simulate_end_to_end(tmp_path: Path) -> None:
    """The offline pipeline writes, resolves, aggregates, and validates."""
    config = load_config()
    origins = smoke_origins(config)
    log_dir = tmp_path / "log"
    out_dir = tmp_path / "data"

    result = run_simulation(
        config,
        origins,
        log_dir=log_dir,
        out_dir=out_dir,
        submission_timestamp="2025-10-20T21:30:00Z",
        resolved_at="2025-10-27T21:30:00Z",
    )

    # 15 smoke-backed rungs x 3 origins written; 7 unbacked rungs x 3 gapped.
    assert result.n_predictions_written == 45
    assert result.n_gapped == 21
    # h=5 resolves at origins 2025-10-06 and 2025-10-13 for all 15 rungs.
    assert result.n_resolutions == 30

    # Every produced aggregate conforms to its schema.
    assert validate("manifest", json.load((out_dir / "manifest.json").open())) == []
    assert validate("leaderboard", json.load((out_dir / "leaderboard.json").open())) == []
    gaps = json.load((out_dir / "gaps.json").open())
    assert gaps["generated_by"] == "harness"
    for entry in gaps["gaps"]:
        assert validate("gap_log", entry) == []
    for bundle_path in (out_dir / "forecasts").glob("*.json"):
        assert validate("forecast_bundle", json.load(bundle_path.open())) == []


def test_simulate_manifest_is_harness_and_fresh(tmp_path: Path) -> None:
    """The manifest is flagged harness and enumerates the drill-down origins."""
    config = load_config()
    origins = smoke_origins(config)
    run_simulation(
        config,
        origins,
        log_dir=tmp_path / "log",
        out_dir=tmp_path / "data",
        submission_timestamp="2025-10-20T21:30:00Z",
        resolved_at="2025-10-27T21:30:00Z",
    )
    manifest = json.load((tmp_path / "data" / "manifest.json").open())
    assert manifest["generated_by"] == "harness"
    assert manifest["latest_origin"] == "2025-10-20"
    assert manifest["origin_count"] == 3
    assert {o["origin_date"] for o in manifest["origins"]} == {"2025-10-06", "2025-10-13", "2025-10-20"}
