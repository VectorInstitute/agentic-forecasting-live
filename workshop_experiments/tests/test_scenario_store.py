"""Offline tests for the Track-2 write-up / judge-verdict persistence layout."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from workshop_experiments.scenario_store import (
    ScenarioWriteup,
    has_judge_verdict,
    has_writeup,
    list_scenario_origins,
    load_judge_verdict,
    load_scenario_writeup,
    scenario_dir,
    write_judge_verdict,
    write_scenario_writeup,
)


def test_scenario_dir_layout(tmp_path: Path) -> None:
    """One directory per origin, named YYYY-MM-DD."""
    origin = date(2025, 4, 1)
    assert scenario_dir(origin, tmp_path) == tmp_path / "2025-04-01"


def test_write_and_load_scenario_writeup_round_trips(tmp_path: Path) -> None:
    """The write-up markdown + metadata round-trip through writeup.md + meta.yaml."""
    origin = date(2025, 4, 1)
    writeup = ScenarioWriteup(
        origin=origin,
        markdown="## Scenario: Soft landing (~0.5)\n**Key drivers:** ...\n",
        meta={"model": "gemini-3.1-flash-lite-preview", "agent_name": "tsx_analyst_scenario", "trace_id": "abc123"},
    )

    target_dir = write_scenario_writeup(writeup, tmp_path)

    assert target_dir == tmp_path / "2025-04-01"
    assert (target_dir / "writeup.md").exists()
    assert (target_dir / "meta.yaml").exists()

    loaded = load_scenario_writeup(origin, tmp_path)
    assert loaded is not None
    assert loaded.markdown == writeup.markdown
    assert loaded.meta["model"] == "gemini-3.1-flash-lite-preview"
    assert loaded.meta["trace_id"] == "abc123"


def test_has_writeup_and_load_missing(tmp_path: Path) -> None:
    """has_writeup is False and load returns None before anything is written."""
    origin = date(2025, 4, 8)
    assert not has_writeup(origin, tmp_path)
    assert load_scenario_writeup(origin, tmp_path) is None

    write_scenario_writeup(ScenarioWriteup(origin=origin, markdown="x", meta={}), tmp_path)
    assert has_writeup(origin, tmp_path)


def test_list_scenario_origins_sorted(tmp_path: Path) -> None:
    """Origins with a persisted write-up are listed, sorted ascending."""
    for origin in (date(2026, 3, 31), date(2025, 4, 1), date(2026, 2, 25), date(2025, 4, 8)):
        write_scenario_writeup(ScenarioWriteup(origin=origin, markdown="x", meta={}), tmp_path)
    # An extra directory with no writeup.md must not be picked up.
    (tmp_path / "not-an-origin").mkdir()
    (tmp_path / "2099-01-01").mkdir()  # dir exists but no writeup.md written

    origins = list_scenario_origins(tmp_path)

    assert origins == [date(2025, 4, 1), date(2025, 4, 8), date(2026, 2, 25), date(2026, 3, 31)]


def test_list_scenario_origins_empty_store(tmp_path: Path) -> None:
    """An empty (or nonexistent) store lists no origins."""
    assert list_scenario_origins(tmp_path / "does-not-exist") == []
    assert list_scenario_origins(tmp_path) == []


def test_write_and_load_judge_verdict_round_trips(tmp_path: Path) -> None:
    """The judge verdict persists alongside the write-up, under judge.yaml."""
    origin = date(2025, 4, 1)
    write_scenario_writeup(ScenarioWriteup(origin=origin, markdown="x", meta={}), tmp_path)
    assert not has_judge_verdict(origin, tmp_path)

    verdict = {
        "origin": "2025-04-01",
        "judge_model": "claude-sonnet-4-6",
        "verdict": {"drivers_score": 4, "calibration_score": 3, "specificity_score": 5},
    }
    path = write_judge_verdict(origin, verdict, tmp_path)

    assert path == tmp_path / "2025-04-01" / "judge.yaml"
    assert has_judge_verdict(origin, tmp_path)
    loaded = load_judge_verdict(origin, tmp_path)
    assert loaded == verdict


def test_load_judge_verdict_missing_returns_none(tmp_path: Path) -> None:
    """No persisted verdict -> None, not an error."""
    assert load_judge_verdict(date(2025, 4, 1), tmp_path) is None
