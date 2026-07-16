"""Offline tests for postmortem origin selection."""

from __future__ import annotations

import datetime as dt

from workshop_experiments.adaptive.origins import (
    parse_origins_override,
    rank_origins,
)


def _d(day: int) -> dt.date:
    """Return a January-2025 date for the given day-of-month."""
    return dt.date(2025, 1, day)


def test_rank_selects_worst_then_best_disjoint() -> None:
    """Rank selects worst then best disjoint."""
    scores = {_d(1): 0.9, _d(2): 0.1, _d(3): 0.8, _d(4): 0.2, _d(5): 0.5}
    sel = rank_origins(scores, worst_n=2, best_n=2)
    # Worst = highest CRPS (0.9, 0.8); controls = lowest among the rest (0.1, 0.2).
    assert set(sel.worst) == {_d(1), _d(3)}
    assert set(sel.controls) == {_d(2), _d(4)}
    # Worst and controls never overlap; `all` de-duplicates worst-first.
    assert not (set(sel.worst) & set(sel.controls))
    assert sel.all[:2] == tuple(sorted(sel.worst))


def test_rank_handles_more_requested_than_available() -> None:
    """Rank handles more requested than available."""
    scores = {_d(1): 0.5, _d(2): 0.4}
    sel = rank_origins(scores, worst_n=12, best_n=4)
    # All origins go to worst; nothing left for controls.
    assert set(sel.worst) == {_d(1), _d(2)}
    assert sel.controls == ()


def test_rank_is_deterministic_on_ties() -> None:
    """Rank is deterministic on ties."""
    scores = {_d(3): 0.5, _d(1): 0.5, _d(2): 0.5}
    sel = rank_origins(scores, worst_n=1, best_n=1)
    # Ties break by date; worst takes the earliest of the tied high scores.
    assert sel.worst == (_d(1),)


def test_parse_origins_override() -> None:
    """Parse origins override."""
    got = parse_origins_override(["2025-04-01", " 2025-08-01 ", ""])
    assert got == (dt.date(2025, 4, 1), dt.date(2025, 8, 1))
