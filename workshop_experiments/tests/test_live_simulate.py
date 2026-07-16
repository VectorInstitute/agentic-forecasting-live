"""End-to-end offline simulate pipeline (no API, no network) — runs in CI.

Deliberately NOT marked ``integration_test`` so the default CI job
(``pytest -m "not integration_test"``) collects it. It exercises the full
write -> resolve -> aggregate -> validate chain against the committed smoke
predictions in a temp dir.
"""

from __future__ import annotations

import json
from pathlib import Path

from workshop_experiments.live.config import load_config, registry_predictor_id
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

    # Expectations derive from the committed smoke store, so growing the store
    # (new smoke-backed rungs, or filling a previously failed origin) never
    # breaks this test. A (rung, origin) pair is backed when the fixture for
    # EVERY horizon task exists — simulate gaps the pair otherwise.
    def backed(rung: object, origin: object) -> bool:
        rid = registry_predictor_id(rung.registry_method, rung.model)  # type: ignore[attr-defined]
        return all(
            (config.smoke_store / rid / config.task_id_for_horizon(h) / f"{origin}.yaml").is_file()
            for h in config.horizons
        )

    backed_pairs = [(r, o) for r in config.predictors for o in origins if backed(r, o)]
    n_pairs = len(config.predictors) * len(origins)
    assert backed_pairs
    assert result.n_predictions_written == len(backed_pairs)
    assert result.n_gapped == n_pairs - len(backed_pairs)
    # h=5 resolves only at the first two of the three smoke origins (the third
    # origin's h=5 target is beyond the smoke window).
    assert result.n_resolutions == sum(1 for _, o in backed_pairs if o in origins[:2])

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
