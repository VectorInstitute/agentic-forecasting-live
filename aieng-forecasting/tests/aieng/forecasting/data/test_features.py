"""Tests for the calendar-alignment feature helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.data.features import (
    business_daily_ffill,
    business_daily_zero_fill,
)


def _series_with_holiday_gap() -> pd.DataFrame:
    # 2025-07-04 (Fri) is a US market holiday: no row, while Mon-Thu and the
    # following Monday trade.
    ts = pd.to_datetime(["2025-06-30", "2025-07-01", "2025-07-02", "2025-07-03", "2025-07-07"])
    return pd.DataFrame({"timestamp": ts, "value": [1.0, 2.0, 3.0, 4.0, 5.0], "released_at": ts})


def test_business_daily_ffill_carries_level_across_gap() -> None:
    """A level series carries the last observed value onto a missing business day."""
    out = business_daily_ffill(_series_with_holiday_gap())
    row = out[out["timestamp"] == pd.Timestamp("2025-07-04")]
    assert len(row) == 1
    assert row["value"].iloc[0] == pytest.approx(4.0)


def test_business_daily_zero_fill_inserts_zero_return_on_gap() -> None:
    """A missing session gets a 0.0 return; existing rows and the sum are kept."""
    out = business_daily_zero_fill(_series_with_holiday_gap())
    row = out[out["timestamp"] == pd.Timestamp("2025-07-04")]
    assert len(row) == 1
    assert row["value"].iloc[0] == 0.0
    # Existing rows are untouched.
    kept = out[out["timestamp"] == pd.Timestamp("2025-07-03")]
    assert kept["value"].iloc[0] == pytest.approx(4.0)
    # Zero-filling preserves the running sum of returns (no double-counting).
    assert out["value"].sum() == pytest.approx(15.0)


def test_business_daily_zero_fill_empty_frame() -> None:
    """An empty frame passes through unchanged."""
    empty = pd.DataFrame(columns=["timestamp", "value", "released_at"])
    assert business_daily_zero_fill(empty).empty


def test_business_daily_zero_fill_covers_full_business_calendar() -> None:
    """The output covers every Mon-Fri business day in the input range with no NaNs."""
    out = business_daily_zero_fill(_series_with_holiday_gap())
    expected = pd.bdate_range("2025-06-30", "2025-07-07")
    assert list(out["timestamp"]) == list(expected)
    assert not np.any(out["value"].isna())
