"""Offline test: mutation events collate into the monitor's mutations.json."""

from __future__ import annotations

import json
from pathlib import Path

from workshop_experiments.adaptive.gates.events import MutationEventStore
from workshop_experiments.live.aggregate import aggregate_step
from workshop_experiments.live.config import load_config
from workshop_experiments.live.simulate import run_simulation, smoke_origins


def test_mutations_flow_into_aggregate(tmp_path: Path) -> None:
    """Mutations flow into aggregate."""
    config = load_config()
    origins = smoke_origins(config)[:1]
    log_dir = tmp_path / "log"
    out_dir = tmp_path / "out"

    # Populate the log with one real (smoke) origin's predictions + resolutions.
    run_simulation(
        config,
        origins,
        log_dir=log_dir,
        out_dir=out_dir,
        submission_timestamp="2026-07-16T21:30:00Z",
        resolved_at="2026-07-16T21:30:00Z",
    )

    # A learning twin writes a mutation event on that day.
    store = MutationEventStore(log_dir, "adaptive_learning")
    store.write(
        tier="observation",
        gate_outcome="appended",
        rationale="pattern across origins",
        occurred_on=origins[0],
        origin_date=origins[0],
    )

    # Re-aggregate; the event now appears in mutations.json and validates.
    aggregate_step(log_dir, out_dir)
    mutations = json.loads((out_dir / "mutations.json").read_text())
    assert mutations["generated_by"] == "harness"
    assert len(mutations["mutations"]) == 1
    assert mutations["mutations"][0]["twin_id"] == "adaptive_learning"
    assert mutations["mutations"][0]["tier"] == "observation"
