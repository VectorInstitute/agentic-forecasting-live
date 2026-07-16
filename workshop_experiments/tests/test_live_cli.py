"""CLI-level behavior that is safe to exercise offline (dry-run + arg checks)."""

from __future__ import annotations

from workshop_experiments.live.cli import run


def test_dry_run_simulate_makes_no_writes(capsys) -> None:  # type: ignore[no-untyped-def]
    """``--dry-run --simulate`` prints a plan and exits 0 without side effects."""
    code = run(["--dry-run", "--simulate"])
    assert code == 0
    out = capsys.readouterr().out
    assert "ws-live-run plan (simulate)" in out
    assert "configured rungs : 26" in out


def test_origin_requires_simulate() -> None:
    """``--origin`` outside simulate mode is rejected with exit code 2."""
    assert run(["--origin", "2025-10-20"]) == 2
