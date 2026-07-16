"""Offline tests for the study driver (fake session — no model calls)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from workshop_experiments.adaptive.session import TurnResult, approx_tokens
from workshop_experiments.adaptive.study import (
    STUDY_HALL_PROMPT,
    Postmortem,
    build_postmortem_prompt,
    run_residency,
    run_study_hall,
)


class FakeSession:
    """Records prompts and returns deterministic, accounted turns."""

    def __init__(self) -> None:
        """Start with an empty prompt log."""
        self.prompts: list[str] = []

    def run_turn(self, prompt: str) -> TurnResult:
        """Record the prompt and return a deterministic accounted turn."""
        self.prompts.append(prompt)
        text = f"reply-{len(self.prompts)}"
        return TurnResult(
            text=text,
            input_tokens=approx_tokens(prompt),
            output_tokens=approx_tokens(text),
            wall_time_s=0.01,
        )

    def close(self) -> None:  # pragma: no cover - trivial
        """Close."""


def _transcript(run_dir: Path) -> list[dict]:
    """Read the persisted transcript entries."""
    return [json.loads(line) for line in (run_dir / "transcript.jsonl").read_text().splitlines()]


def test_study_hall_runs_budget_with_checkpoint_cadence(tmp_path: Path) -> None:
    """Study hall runs budget with checkpoint cadence."""
    session = FakeSession()
    result = run_study_hall(session, tmp_path, turn_budget=25, checkpoint_every=10)
    assert result.turns_run == 25
    assert result.accounting.turns == 25
    assert result.checkpoints == [10, 20]
    entries = _transcript(tmp_path)
    assert len(entries) == 25
    assert entries[0]["kind"] == "study_hall"
    assert entries[9]["kind"] == "distill"
    assert entries[1]["kind"] == "continue"
    # First prompt is the Study Hall agenda (suggested directions, not a curriculum).
    assert session.prompts[0] == STUDY_HALL_PROMPT
    assert "SUGGESTED directions" in STUDY_HALL_PROMPT


def test_study_hall_persists_state_and_accounting(tmp_path: Path) -> None:
    """Study hall persists state and accounting."""
    run_study_hall(FakeSession(), tmp_path, turn_budget=12, checkpoint_every=10)
    state = json.loads((tmp_path / "study_state.json").read_text())
    assert state["turns_completed"] == 12
    assert state["accounting"]["turns"] == 12
    assert state["checkpoints"] == [10]


def test_study_hall_resumes_from_persisted_state(tmp_path: Path) -> None:
    """Study hall resumes from persisted state."""
    first = run_study_hall(FakeSession(), tmp_path, turn_budget=5, checkpoint_every=10)
    assert first.turns_run == 5
    session2 = FakeSession()
    second = run_study_hall(session2, tmp_path, turn_budget=12, checkpoint_every=10)
    # Only the remaining 7 turns run on the resume.
    assert second.turns_run == 7
    assert len(session2.prompts) == 7
    # Transcript accumulates to the full budget with no duplicate turn numbers.
    entries = _transcript(tmp_path)
    assert [e["turn"] for e in entries] == list(range(1, 13))


def test_residency_runs_and_resumes_per_origin(tmp_path: Path) -> None:
    """Residency runs and resumes per origin."""
    pms = [
        Postmortem(origin=dt.date(2025, 4, 8), committed_forecast="f", realized="r", crps="0.5"),
        Postmortem(origin=dt.date(2025, 8, 1), committed_forecast="f", realized="r", crps="0.4"),
    ]
    session = FakeSession()
    result = run_residency(session, tmp_path, pms, turns_per_postmortem=2)
    assert result.accounting.turns == 4  # 2 origins x 2 turns
    state = json.loads((tmp_path / "study_state.json").read_text())
    assert state["origins_done"] == ["2025-04-08", "2025-08-01"]

    # Re-running with the same + a new origin only processes the new one.
    pms.append(Postmortem(origin=dt.date(2025, 11, 3), committed_forecast="f", realized="r", crps="0.6"))
    session2 = FakeSession()
    run_residency(session2, tmp_path, pms, turns_per_postmortem=2)
    assert len(session2.prompts) == 2  # only the new origin's 2 turns


def test_postmortem_prompt_is_date_bounded() -> None:
    """Postmortem prompt is date bounded."""
    prompt = build_postmortem_prompt(origin=dt.date(2025, 4, 8), committed_forecast="f", realized="r", crps="0.5")
    assert "2025-04-08" in prompt
    assert "cutoff_date = 2025-04-08" in prompt
