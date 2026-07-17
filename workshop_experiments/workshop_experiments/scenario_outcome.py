"""Realized-outcome computation for Track-2 scenario judging.

The judge (``ws-scenario-judge``) needs to know what actually happened after a
scenario write-up's origin: the realized S&P/TSX Composite cumulative log
return over the next 5 / 21 / 60 business days. This module computes that
purely from a target return series — no LLM call, no network beyond whatever
the caller used to fetch the series — so it is unit-testable against a
synthetic series with hand-checked numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Callable

import pandas as pd


#: Horizons (business days) the judge grades against — the rubric's realized
#: return over 5/21/60 business days from the scenario's origin.
JUDGE_HORIZONS: tuple[int, ...] = (5, 21, 60)

#: Log-return magnitude below which a horizon counts as "flat" rather than
#: up/down (avoids a coin-flip direction label on a near-zero move).
_FLAT_EPSILON = 1e-4


@dataclass(frozen=True)
class RealizedHorizonOutcome:
    """One horizon's realized outcome, measured from a scenario's origin."""

    horizon: int
    forecast_date: date
    log_return: float
    direction: str  # "up" | "down" | "flat"

    @property
    def pct_return(self) -> float:
        """Realized return as a simple percentage (``(e^r - 1) * 100``)."""
        return (math.exp(self.log_return) - 1.0) * 100.0


@dataclass(frozen=True)
class RealizedOutcomeSummary:
    """Realized outcomes across every matured horizon for one scenario origin."""

    origin: date
    outcomes: tuple[RealizedHorizonOutcome, ...]

    def to_markdown(self) -> str:
        """Render a short markdown summary for the judge prompt."""
        if not self.outcomes:
            return f"No matured realized outcomes are available yet for origin {self.origin.isoformat()}."
        lines = [f"Realized S&P/TSX Composite cumulative log returns from {self.origin.isoformat()}:"]
        for outcome in self.outcomes:
            lines.append(
                f"- {outcome.horizon} business days (through {outcome.forecast_date.isoformat()}): "
                f"{outcome.pct_return:+.2f}% ({outcome.direction})"
            )
        return "\n".join(lines)


def realized_outcome_for_horizon(
    series: pd.DataFrame,
    *,
    origin: date,
    horizon: int,
) -> RealizedHorizonOutcome | None:
    """Return the realized outcome ``horizon`` business days after ``origin``.

    ``series`` is a cumulative-log-return frame for exactly this window (i.e.
    the ``tsx_logret_{horizon}b`` series produced by
    :func:`workshop_experiments.data_tsx.build_cumulative_log_return_frame`:
    ``value`` at session ``t`` is ``log(close[t] / close[t - horizon])``), with
    ``timestamp`` / ``value`` columns.

    Resolves to the *horizon*-th trading session strictly after ``origin``
    present in ``series`` — holiday-robust, mirroring
    :class:`workshop_experiments.live.resolve.DataServiceRealizedProvider`.
    Returns ``None`` when fewer than ``horizon`` sessions are available after
    ``origin`` (the horizon has not matured yet in the supplied series).
    """
    frame = series.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    after = frame.loc[frame["timestamp"] > pd.Timestamp(origin)].sort_values("timestamp")
    if len(after) < horizon:
        return None
    row = after.iloc[horizon - 1]
    value = float(row["value"])
    if value > _FLAT_EPSILON:
        direction = "up"
    elif value < -_FLAT_EPSILON:
        direction = "down"
    else:
        direction = "flat"
    return RealizedHorizonOutcome(
        horizon=horizon,
        forecast_date=pd.Timestamp(row["timestamp"]).date(),
        log_return=value,
        direction=direction,
    )


def compute_realized_outcome_summary(
    get_series: Callable[[int], pd.DataFrame],
    *,
    origin: date,
    horizons: tuple[int, ...] = JUDGE_HORIZONS,
) -> RealizedOutcomeSummary:
    """Compute the realized-outcome summary across every horizon.

    ``get_series(horizon)`` supplies the ``tsx_logret_{horizon}b`` frame for
    that horizon — injected so callers can pass a live ``DataService`` lookup
    in the CLI or a synthetic frame in tests. Horizons that have not matured
    yet (see :func:`realized_outcome_for_horizon`) are omitted from the
    summary, not errored.
    """
    outcomes = []
    for horizon in horizons:
        outcome = realized_outcome_for_horizon(get_series(horizon), origin=origin, horizon=horizon)
        if outcome is not None:
            outcomes.append(outcome)
    return RealizedOutcomeSummary(origin=origin, outcomes=tuple(outcomes))


__all__ = [
    "JUDGE_HORIZONS",
    "RealizedHorizonOutcome",
    "RealizedOutcomeSummary",
    "compute_realized_outcome_summary",
    "realized_outcome_for_horizon",
]
