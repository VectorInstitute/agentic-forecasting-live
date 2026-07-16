"""Offline dry-run tests for the TSX study + eval CLIs (nothing spends)."""

from __future__ import annotations

from pathlib import Path

from workshop_experiments.adaptive import eval_cli, study_cli
from workshop_experiments.adaptive.domain_tsx import (
    TSX_DISTILL_PROMPT,
    TSX_STUDY_HALL_PROMPT,
)
from workshop_experiments.adaptive.session import TurnResult, approx_tokens
from workshop_experiments.adaptive.study import StudyHallPrompts, run_study_hall


class _FakeSession:
    """Records prompts and returns deterministic, accounted turns."""

    def __init__(self) -> None:
        """Start with an empty prompt log."""
        self.prompts: list[str] = []

    def run_turn(self, prompt: str) -> TurnResult:
        """Record the prompt; return a deterministic accounted turn."""
        self.prompts.append(prompt)
        text = f"reply-{len(self.prompts)}"
        return TurnResult(
            text=text, input_tokens=approx_tokens(prompt), output_tokens=approx_tokens(text), wall_time_s=0.01
        )

    def close(self) -> None:  # pragma: no cover - trivial
        """Close."""


def test_tsx_study_prompts_thread_through_driver(tmp_path: Path) -> None:
    """The single-session driver uses the TSX bootcamp prompts (opening + distill)."""
    session = _FakeSession()
    prompts = StudyHallPrompts(study_hall=TSX_STUDY_HALL_PROMPT, distill=TSX_DISTILL_PROMPT)
    result = run_study_hall(session, tmp_path, turn_budget=10, checkpoint_every=10, prompts=prompts)
    assert result.turns_run == 10
    assert result.checkpoints == [10]
    # Opening turn is the TSX Study Hall agenda (covers pre-2026 history + 2025 review).
    assert session.prompts[0] == TSX_STUDY_HALL_PROMPT
    assert "pre-2026 TSX history" in TSX_STUDY_HALL_PROMPT
    assert "review the 2025 period" in TSX_STUDY_HALL_PROMPT
    # Turn 10 is the distill checkpoint.
    assert session.prompts[9] == TSX_DISTILL_PROMPT


def test_study_cli_tsx_single_session_dry_run(capsys) -> None:
    """`ws-study --domain tsx` (no subcommand) prints the single-session plan."""
    rc = study_cli.run(["--domain", "tsx"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "single-session study" in out
    assert "S&P/TSX Composite (tsx)" in out
    assert "tsx-strategy" in out
    # The single session covers the whole study over a 50-turn budget by default.
    assert "turn budget      : 50" in out
    assert "distill turns    : [10, 20, 30, 40, 50]" in out
    assert "dry-run — no model calls" in out


def test_study_cli_tsx_phase_a_alias_dry_run(capsys) -> None:
    """`ws-study phase-a --domain tsx` also drives the TSX Study Hall (dry-run)."""
    rc = study_cli.run(["phase-a", "--domain", "tsx", "--turn-budget", "20"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "S&P/TSX Composite (tsx)" in out
    assert "turn budget      : 20" in out


def test_study_cli_sp500_default_unchanged(capsys) -> None:
    """The default domain stays S&P 500, so existing phase-a behaviour is unchanged."""
    rc = study_cli.run(["phase-a"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "S&P 500 (sp500)" in out
    assert "sp500-strategy" in out


def test_eval_cli_tsx_dry_run(capsys) -> None:
    """`ws-adaptive-eval --domain tsx` prints the two-arm plan and spends nothing."""
    rc = eval_cli.run(["--domain", "tsx"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "domain    : tsx" in out
    assert "spec      : tsx_ws_eval_2026_weekly" in out
    assert "untrained" in out and "trained" in out
    assert "pass run=True" in out


def test_eval_cli_tsx_arm_override_dry_run(capsys, tmp_path) -> None:
    """A --arm override replaces the trained dir while keeping the untrained seed."""
    rc = eval_cli.run(["--domain", "tsx", "--arm", f"trained={tmp_path / 'my-trained'}"])
    assert rc == 0
    out = capsys.readouterr().out
    assert str(tmp_path / "my-trained") in out
    assert "untrained" in out
