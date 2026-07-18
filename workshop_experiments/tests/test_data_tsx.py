"""Offline leak-safety tests for the TSX data layer (pure transforms, no network).

The service-level fetch (Yahoo target + covariates) is network-bound and lives in
the ``integration_test``-marked test at the bottom, excluded from the default CI
run. Everything above exercises the pure frame transforms directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from workshop_experiments.data_tsx import (
    DEFAULT_COVARIATE_SERIES_IDS,
    TSX_COVARIATE_PANEL,
    TSX_RETURN_WINDOWS,
    build_cumulative_log_return_frame,
    daily_level_feature_from_statcan,
    monthly_level_feature,
    monthly_mom_logdiff_feature,
    tsx_logret_series_id,
)


def _daily_prices(n: int = 40, start: str = "2024-01-01") -> pd.DataFrame:
    ts = pd.bdate_range(start, periods=n)
    vals = 20000.0 * np.exp(np.cumsum(np.full(n, 0.001)))
    return pd.DataFrame({"timestamp": ts, "value": vals, "released_at": ts})


def test_series_ids_are_tsx_distinct() -> None:
    """Target + covariate ids are all tsx-prefixed (never collide with sp500)."""
    assert tsx_logret_series_id(1) == "tsx_logret_1b"
    assert all(sid.startswith("tsx_") for sid in DEFAULT_COVARIATE_SERIES_IDS)
    assert TSX_COVARIATE_PANEL == DEFAULT_COVARIATE_SERIES_IDS
    assert len(set(TSX_COVARIATE_PANEL)) == len(TSX_COVARIATE_PANEL)
    assert set(TSX_RETURN_WINDOWS) == {1, 5, 21}


def test_cumulative_log_return_frame_window_1() -> None:
    """window=1 is the ordinary daily close-to-close log return."""
    prices = _daily_prices()
    out = build_cumulative_log_return_frame(prices, window=1)
    expected = np.log(prices["value"].iloc[1] / prices["value"].iloc[0])
    assert out["value"].iloc[0] == pytest.approx(expected)
    # released_at equals the session timestamp (return known at that close).
    assert (out["released_at"] == out["timestamp"]).all()
    assert len(out) == len(prices) - 1


def test_cumulative_log_return_frame_window_5_is_trailing_sum() -> None:
    """A 5-day cumulative return equals the sum of five daily returns."""
    prices = _daily_prices()
    daily = build_cumulative_log_return_frame(prices, window=1)["value"].to_numpy()
    out5 = build_cumulative_log_return_frame(prices, window=5)
    # First 5b value aligns to trailing sum of daily returns ending at that row.
    assert out5["value"].iloc[0] == pytest.approx(daily[:5].sum())


def test_cumulative_log_return_frame_rejects_bad_window() -> None:
    """A window < 1 is rejected."""
    with pytest.raises(ValueError, match="window must be >= 1"):
        build_cumulative_log_return_frame(_daily_prices(), window=0)


def test_daily_level_feature_is_lagged_one_business_day() -> None:
    """The daily StatCan level feature at t carries the value from t-1 (leak-safe)."""
    raw = _daily_prices(n=10)
    raw = raw.assign(value=[100.0 + i for i in range(10)])
    out = daily_level_feature_from_statcan(raw)
    # released_at == timestamp after the lag; value at row i is the prior level.
    assert (out["released_at"] == out["timestamp"]).all()
    merged = out.merge(raw[["timestamp", "value"]], on="timestamp", suffixes=("_feat", "_raw"))
    # Each feature value equals the previous session's raw level.
    prior = raw.set_index("timestamp")["value"].shift(1)
    for _, row in merged.iterrows():
        assert row["value_feat"] == pytest.approx(prior[row["timestamp"]])


def test_daily_level_feature_ffills_holiday_gap() -> None:
    """A missing mid-week business day is forward-filled before the lag."""
    raw = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-11-05", "2025-11-06", "2025-11-10"]),
            "value": [3.0, 3.1, 3.2],
            "released_at": pd.to_datetime(["2025-11-06", "2025-11-07", "2025-11-11"]),
        }
    )
    out = daily_level_feature_from_statcan(raw)
    ts = list(out["timestamp"])
    # 2025-11-07 (Fri) is a business day absent from the raw frame; it must appear.
    assert pd.Timestamp("2025-11-07") in ts


def test_market_return_covariate_defined_on_cross_calendar_holidays() -> None:
    """The lagged market-return covariate chain covers US holidays the TSX trades.

    Mirrors the panel's composition for WTI/gold/USDCAD/SPX/VIX-change: a US
    session gap (2025-07-04) must yield a zero return row, so a TSX origin on or
    after the holiday never fails Darts' past-covariate alignment.
    """
    from aieng.forecasting.data.features import (  # noqa: PLC0415
        apply_one_business_day_feature_lag,
        business_daily_zero_fill,
        to_log_return_feature,
    )

    ts = pd.to_datetime(["2025-07-01", "2025-07-02", "2025-07-03", "2025-07-07", "2025-07-08"])
    prices = pd.DataFrame({"timestamp": ts, "value": [100.0, 101.0, 102.0, 103.0, 104.0], "released_at": ts})
    feat = apply_one_business_day_feature_lag(business_daily_zero_fill(to_log_return_feature(prices)))
    by_ts = feat.set_index("timestamp")["value"]
    # The holiday itself carries the prior session's (lagged) return...
    assert pd.Timestamp("2025-07-04") in by_ts.index
    assert by_ts[pd.Timestamp("2025-07-04")] == pytest.approx(np.log(102.0 / 101.0))
    # ...and the session after the holiday sees a zero return for the closed day.
    assert by_ts[pd.Timestamp("2025-07-07")] == 0.0


def test_monthly_mom_logdiff_is_release_lagged() -> None:
    """CPI MoM log-diff becomes visible only well after the reference month ends."""
    months = pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"])
    raw = pd.DataFrame({"timestamp": months, "value": [100.0, 100.5, 101.0, 101.4], "released_at": months})
    out = monthly_mom_logdiff_feature(raw, release_bday_lag=18, start="2025-01-01", end="2025-07-01")
    assert not out.empty
    # The Feb reference value (first MoM diff) cannot be visible in early Feb — its
    # conservative release is ~end of March, so no daily row before then carries it.
    feb_diff = float(np.log(100.5 / 100.0))
    early = out[out["timestamp"] < pd.Timestamp("2025-03-01")]
    assert not np.any(np.isclose(early["value"].to_numpy(), feb_diff))
    assert (out["released_at"] == out["timestamp"]).all()


def test_monthly_level_feature_release_lagged() -> None:
    """The monthly unemployment level is expanded onto business days, release-lagged."""
    months = pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"])
    raw = pd.DataFrame({"timestamp": months, "value": [6.5, 6.6, 6.4], "released_at": months})
    out = monthly_level_feature(raw, release_bday_lag=12, start="2025-01-01", end="2025-06-01")
    assert not out.empty
    # The January value should not be visible in January (released ~mid-February).
    jan_rows = out[out["timestamp"] < pd.Timestamp("2025-02-01")]
    assert not np.any(np.isclose(jan_rows["value"].to_numpy(), 6.5))


@pytest.mark.integration_test
def test_build_tsx_service_registers_target_and_market_covariates(tmp_path) -> None:
    """Live: the TSX service fetches ^GSPTSE and the Yahoo market covariates.

    StatCan covariates may be absent when the WDS API is unreachable (they degrade
    with a warning under strict_covariates=False); the Yahoo-sourced factors and
    the three targets must always register.
    """
    from workshop_experiments.data_tsx import build_tsx_multivariate_service  # noqa: PLC0415

    svc = build_tsx_multivariate_service(
        start="2020-01-01",
        end="2025-01-01",
        yahoo_cache_dir=tmp_path / "yfinance",
        statcan_cache_dir=tmp_path / "statcan",
    )
    ids = set(svc.series_ids)
    assert {"tsx_logret_1b", "tsx_logret_5b", "tsx_logret_21b"} <= ids
    # The FRED-free Yahoo market covariates must all be present.
    assert {
        "tsx_vix_level_l1b",
        "tsx_wti_oil_log_ret_1b_l1b",
        "tsx_gold_log_ret_1b_l1b",
        "tsx_usdcad_log_ret_1b_l1b",
        "tsx_sp500_log_ret_1b_l1b",
        "tsx_us10y_level_l1b",
    } <= ids
