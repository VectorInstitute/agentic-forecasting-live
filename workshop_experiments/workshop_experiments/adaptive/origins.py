"""Postmortem origin selection for Phase B (Residency).

Residency runs postmortems over the *most instructive* 2025 origins, not all 52:
the worst-N by CRPS (the misses worth diagnosing) plus a few best-CRPS controls
(good calls, to check the postmortem does not overfit to failure). Selection
reads the persisted weekly-backtest predictions the workshop runner already
wrote and scores them per origin — it never calls a model API.

The ranking itself (:func:`rank_origins`) is pure and offline-testable; the
store-reading helper (:func:`origin_scores_from_results`) turns the workshop's
scored :class:`BacktestResult` objects into the ``{origin: mean_crps}`` map it
consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from statistics import fmean
from typing import Mapping

import pandas as pd
from aieng.forecasting.evaluation.backtest import BacktestResult


@dataclass(frozen=True)
class OriginSelection:
    """The selected postmortem origins, split into misses and controls."""

    worst: tuple[date, ...]
    controls: tuple[date, ...]

    @property
    def all(self) -> tuple[date, ...]:
        """Worst origins first, then controls, de-duplicated (sorted per group)."""
        seen: set[date] = set()
        ordered: list[date] = []
        for origin in (*self.worst, *self.controls):
            if origin not in seen:
                seen.add(origin)
                ordered.append(origin)
        return tuple(ordered)


def origin_scores_from_results(task_results: Mapping[str, BacktestResult]) -> dict[date, float]:
    """Reduce a predictor's per-task scored results to ``{origin: mean_crps}``.

    Averages a predictor's CRPS across every task (horizon) that resolved at an
    origin, so a single instructive-ness score per origin ranks the postmortems.
    """
    by_origin: dict[date, list[float]] = {}
    for result in task_results.values():
        for prediction, score in zip(result.predictions, result.scores):
            origin = pd.Timestamp(prediction.as_of).date()
            by_origin.setdefault(origin, []).append(float(score))
    return {origin: fmean(scores) for origin, scores in by_origin.items()}


def rank_origins(
    origin_scores: Mapping[date, float],
    *,
    worst_n: int = 12,
    best_n: int = 4,
) -> OriginSelection:
    """Select the worst-N (highest CRPS) and best-N (lowest CRPS) origins.

    Ties break by date (earlier first) for determinism. ``worst`` and
    ``controls`` never overlap: the best set is drawn from the origins not
    already claimed as worst.
    """
    if worst_n < 0 or best_n < 0:
        raise ValueError("worst_n and best_n must be non-negative")
    ordered = sorted(origin_scores.items(), key=lambda kv: (-kv[1], kv[0]))
    worst = [origin for origin, _ in ordered[:worst_n]]
    remaining = [origin for origin, _ in ordered[worst_n:]]
    # Best controls: lowest CRPS among the remaining.
    best = sorted(remaining, key=lambda o: (origin_scores[o], o))[:best_n]
    return OriginSelection(
        worst=tuple(sorted(worst)),
        controls=tuple(sorted(best)),
    )


def parse_origins_override(tokens: list[str]) -> tuple[date, ...]:
    """Parse a ``--origins`` override list of ``YYYY-MM-DD`` tokens."""
    return tuple(date.fromisoformat(token.strip()) for token in tokens if token.strip())


__all__ = [
    "OriginSelection",
    "origin_scores_from_results",
    "parse_origins_override",
    "rank_origins",
]
