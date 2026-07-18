"""Offline dry-run tests for the TSX study + eval CLIs (nothing spends)."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from workshop_experiments.adaptive import eval_cli, study_cli
from workshop_experiments.adaptive import session as session_mod
from workshop_experiments.adaptive import study as study_mod
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


class _FakeStudyDomain:
    """Minimal domain config stub recording reseed calls."""

    def __init__(self, base: Path) -> None:
        """Point the seed/trained dirs under *base* and start with no reseeds."""
        self.key = "tsx"
        self.seed_dir = base / "seed"
        self.trained_dir = base / "trained"
        self.prompts = StudyHallPrompts(study_hall="go", distill="distill")
        self.reseed_calls: list[bool] = []

    def reseed(self, force: bool) -> None:
        """Record the reseed instead of touching any files."""
        self.reseed_calls.append(force)

    def config_builder(self, *, model: str | None, strategy_dir: Path) -> object:
        """Return a placeholder config."""
        return object()


def _drive_args() -> argparse.Namespace:
    return argparse.Namespace(domain="tsx", no_reseed=False, model=None, turn_budget=5, checkpoint_every=5)


def _stub_spend_path(monkeypatch, fake_dc) -> None:
    monkeypatch.setitem(study_cli._DOMAINS, "tsx", lambda: fake_dc)
    monkeypatch.setattr(session_mod, "AdkStudySession", lambda *a, **k: _FakeSession())
    fake_result = SimpleNamespace(
        phase="study_hall",
        turns_run=0,
        accounting=SimpleNamespace(input_tokens=0, output_tokens=0, wall_time_s=0.0),
    )
    monkeypatch.setattr(study_mod, "run_study_hall", lambda *a, **k: fake_result)


def test_study_resume_never_reseeds(tmp_path: Path, monkeypatch, capsys) -> None:
    """A run_dir with existing study state must not force-reseed the trained strategy.

    Regression: a completed study resumed by a scheduled re-invocation used to
    reseed first, wiping every accumulated strategy mutation while replaying
    zero turns.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "study_state.json").write_text("{}", encoding="utf-8")
    fake_dc = _FakeStudyDomain(tmp_path)
    _stub_spend_path(monkeypatch, fake_dc)
    rc = study_cli._drive_study_hall(_drive_args(), run_dir, phase="study")
    assert rc == 0
    assert fake_dc.reseed_calls == []
    assert "skipping reseed" in capsys.readouterr().out


def test_study_fresh_run_reseeds(tmp_path: Path, monkeypatch) -> None:
    """A fresh run_dir (no study state) still reseeds the trained strategy."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fake_dc = _FakeStudyDomain(tmp_path)
    _stub_spend_path(monkeypatch, fake_dc)
    rc = study_cli._drive_study_hall(_drive_args(), run_dir, phase="study")
    assert rc == 0
    assert fake_dc.reseed_calls == [True]
