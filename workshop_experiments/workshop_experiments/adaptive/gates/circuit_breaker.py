"""Circuit breaker: freeze adaptation when the learner degrades vs the frozen twin.

Pure, offline logic over two CRPS series (learner and frozen twin), both ordered
oldest-first. The breaker trips when the learner's trailing-window mean CRPS
exceeds ``ratio`` times the frozen twin's over the same window. When it trips the
gate freezes all adaptation and writes a ``frozen_circuit_breaker`` event; the
run continues with the learner frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True)
class BreakerReading:
    """The outcome of one circuit-breaker check."""

    tripped: bool
    learner_mean: float | None
    frozen_mean: float | None
    ratio: float | None
    window: int

    @property
    def summary(self) -> str:
        """Human-readable one-liner for the mutation-event rationale."""
        if self.learner_mean is None or self.frozen_mean is None:
            return f"insufficient history (< {self.window} resolved points) — breaker not evaluated"
        return (
            f"learner trailing-{self.window} mean CRPS {self.learner_mean:.5f} vs "
            f"frozen {self.frozen_mean:.5f} (ratio {self.ratio:.3f})"
        )


def trailing_mean(series: list[float], window: int) -> float | None:
    """Mean of the last *window* points, or ``None`` if fewer than *window* exist."""
    if window <= 0:
        raise ValueError("window must be positive")
    if len(series) < window:
        return None
    return fmean(series[-window:])


def evaluate_breaker(
    learner_crps: list[float],
    frozen_crps: list[float],
    *,
    window: int,
    ratio: float,
) -> BreakerReading:
    """Return whether the circuit breaker trips for the two CRPS series.

    Trips when both series have at least *window* points and
    ``mean(learner[-window:]) > ratio * mean(frozen[-window:])``. Insufficient
    history never trips (returns ``tripped=False`` with ``None`` means).
    """
    learner_mean = trailing_mean(learner_crps, window)
    frozen_mean = trailing_mean(frozen_crps, window)
    if learner_mean is None or frozen_mean is None:
        return BreakerReading(False, learner_mean, frozen_mean, None, window)
    computed_ratio = learner_mean / frozen_mean if frozen_mean > 0 else float("inf")
    tripped = learner_mean > ratio * frozen_mean
    return BreakerReading(tripped, learner_mean, frozen_mean, computed_ratio, window)


__all__ = ["BreakerReading", "evaluate_breaker", "trailing_mean"]
