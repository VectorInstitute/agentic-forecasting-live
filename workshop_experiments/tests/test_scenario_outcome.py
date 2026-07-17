"""Offline, hand-checked tests for the realized-outcome computation.

No network, no LLM: builds a synthetic close-price series with
:func:`workshop_experiments.data_tsx.build_cumulative_log_return_frame` and
hand-checks the resulting realized returns against the closed-form log-return
arithmetic.

Test origins are always chosen past every tested window's warmup length (the
window-N return series has no valid rows for the first N sessions) so the
"Nth business day after origin" indexing lines up the way it does for the real
``tsx_logret_{h}b`` series, whose window warmup is deep in pre-2010 history —
any real judged origin (2025+) is thousands of sessions past it.
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest
from workshop_experiments.data_tsx import build_cumulative_log_return_frame
from workshop_experiments.scenario_outcome import (
    JUDGE_HORIZONS,
    compute_realized_outcome_summary,
    realized_outcome_for_horizon,
)


def _synthetic_price_series(*, start: str = "2025-01-01", n: int = 100, daily_growth: float = 0.001) -> pd.DataFrame:
    """Build a deterministic business-day close series: close[t] = 100 * (1+g)^t."""
    dates = pd.bdate_range(start, periods=n)
    closes = [100.0 * (1.0 + daily_growth) ** i for i in range(n)]
    return pd.DataFrame({"timestamp": dates, "value": closes})


def test_realized_outcome_matches_hand_computed_log_return() -> None:
    """Origin = session 10; horizon = 5 -> value at session 15 vs session 10."""
    price = _synthetic_price_series(n=40)
    window = build_cumulative_log_return_frame(price, window=5)
    origin_ts = price["timestamp"].iloc[10]

    outcome = realized_outcome_for_horizon(window, origin=origin_ts.date(), horizon=5)

    assert outcome is not None
    expected_forecast_ts = price["timestamp"].iloc[15]
    assert outcome.forecast_date == expected_forecast_ts.date()
    expected_log_return = math.log(price["value"].iloc[15] / price["value"].iloc[10])
    assert outcome.log_return == pytest.approx(expected_log_return, rel=1e-9)
    assert outcome.direction == "up"  # daily_growth > 0 -> cumulative return is positive


def test_realized_outcome_direction_labels() -> None:
    """Up / down / flat labels follow the sign of the realized log return."""
    up_price = _synthetic_price_series(n=30, daily_growth=0.01)
    down_price = _synthetic_price_series(n=30, daily_growth=-0.01)
    flat_price = _synthetic_price_series(n=30, daily_growth=0.0)

    # Origin past the window-5 warmup (index 10) so "5th session after origin"
    # in the filtered series really is origin's own +5 business days.
    origin = up_price["timestamp"].iloc[10].date()
    up_outcome = realized_outcome_for_horizon(build_cumulative_log_return_frame(up_price, window=5), origin=origin, horizon=5)
    down_outcome = realized_outcome_for_horizon(
        build_cumulative_log_return_frame(down_price, window=5), origin=origin, horizon=5
    )
    flat_outcome = realized_outcome_for_horizon(
        build_cumulative_log_return_frame(flat_price, window=5), origin=origin, horizon=5
    )

    assert up_outcome is not None and up_outcome.direction == "up"
    assert down_outcome is not None and down_outcome.direction == "down"
    assert flat_outcome is not None and flat_outcome.direction == "flat"


def test_realized_outcome_pct_return_hand_checked() -> None:
    """pct_return is (e^r - 1) * 100, hand-checked against a known return."""
    price = _synthetic_price_series(n=30, daily_growth=0.0)
    # Force one exact, hand-picked cumulative return 5 sessions after origin=10:
    # close[15] = close[10] * e^0.05.
    price.loc[15, "value"] = price.loc[10, "value"] * math.exp(0.05)
    window = build_cumulative_log_return_frame(price, window=5)
    origin = price["timestamp"].iloc[10].date()

    outcome = realized_outcome_for_horizon(window, origin=origin, horizon=5)

    assert outcome is not None
    assert outcome.log_return == pytest.approx(0.05, abs=1e-9)
    assert outcome.pct_return == pytest.approx((math.exp(0.05) - 1.0) * 100.0, rel=1e-9)


def test_realized_outcome_returns_none_when_unmatured() -> None:
    """Fewer than `horizon` sessions after origin -> None, not an error."""
    price = _synthetic_price_series(n=12)
    window = build_cumulative_log_return_frame(price, window=5)
    origin = price["timestamp"].iloc[9].date()  # only 2 sessions remain after this origin

    outcome = realized_outcome_for_horizon(window, origin=origin, horizon=5)

    assert outcome is None


def test_compute_realized_outcome_summary_across_horizons() -> None:
    """The summary collects only matured horizons and renders readable markdown."""
    price = _synthetic_price_series(n=200, daily_growth=0.001)
    windows = {h: build_cumulative_log_return_frame(price, window=h) for h in JUDGE_HORIZONS}
    # Origin past the largest window's warmup (60) with 99 sessions still ahead
    # -> every horizon in JUDGE_HORIZONS (5/21/60) matures.
    origin = price["timestamp"].iloc[100].date()

    summary = compute_realized_outcome_summary(lambda h: windows[h], origin=origin)

    assert [o.horizon for o in summary.outcomes] == list(JUDGE_HORIZONS)
    markdown = summary.to_markdown()
    assert str(origin) in markdown
    for outcome in summary.outcomes:
        assert f"{outcome.horizon} business days" in markdown


def test_compute_realized_outcome_summary_omits_unmatured_horizons() -> None:
    """A horizon with insufficient future data is omitted, not an error."""
    price = _synthetic_price_series(n=70, daily_growth=0.001)
    windows = {h: build_cumulative_log_return_frame(price, window=h) for h in JUDGE_HORIZONS}
    # Origin past every window's warmup (60), but only 9 sessions remain after it
    # -> 5-day horizon matures (9 >= 5), 21- and 60-day do not.
    origin = price["timestamp"].iloc[60].date()

    summary = compute_realized_outcome_summary(lambda h: windows[h], origin=origin)

    assert [o.horizon for o in summary.outcomes] == [5]


def test_compute_realized_outcome_summary_empty_renders_placeholder_text() -> None:
    """An empty outcome set still renders a readable (non-crashing) markdown summary."""
    empty_summary = compute_realized_outcome_summary(
        lambda h: pd.DataFrame({"timestamp": [], "value": []}), origin=date(2025, 4, 1)
    )
    assert empty_summary.outcomes == ()
    assert "No matured realized outcomes" in empty_summary.to_markdown()
