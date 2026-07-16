"""Leak-safe data-service construction for the **S&P/TSX Composite** experiments.

The Vector live forecasting experiment is Canada-focused: the primary target is
the **S&P/TSX Composite index** (``^GSPTSE``), forecast as close-to-close
cumulative log returns at horizons 1 / 5 / 21 business days — the same
construction the S&P 500 machinery uses (see
:func:`sp500_forecasting.data.build_sp500_log_return_service`). This module is
the single entry point for the leak-safe TSX target + covariate panel; the S&P
500 layer (:mod:`workshop_experiments.data`) is left untouched for retrospective
use.

Target
------
``tsx_logret_{1,5,21}b`` — ``log(close[t] / close[t-N])`` of ``^GSPTSE``, one
series per window ``N`` (Yahoo Finance, ``Adj Close``). Forecasting ``r^(N)``
``N`` steps ahead resolves to the forward cumulative return over the next ``N``
sessions — a clean single-marginal forecast per horizon.

Covariate panel (every covariate lagged one business day, conservative
``released_at`` before daily expansion — mirrors the S&P 500 pattern)
--------------------------------------------------------------------
The panel is **FRED-free**: it draws market factors from Yahoo Finance and
Canadian macro from Statistics Canada, so no ``FRED_API_KEY`` is required. Each
inclusion was verified to fetch through mid-2026 (Yahoo) or to expose the exact
member label used (StatCan) before being wired here.

Included
~~~~~~~~
- ``tsx_vix_level_l1b`` / ``tsx_vix_log_ret_1b_l1b`` — CBOE VIX level and log
  return (Yahoo ``^VIX``). Global risk-appetite gauge; kept as the risk anchor
  because no liquid daily TSX-specific implied-vol series is freely available.
- ``tsx_wti_oil_log_ret_1b_l1b`` — WTI crude front-month log return (Yahoo
  ``CL=F``). Energy is ~1/6 of the TSX. Sourced from Yahoo rather than FRED
  ``DCOILWTICO`` to avoid the FRED key dependency.
- ``tsx_gold_log_ret_1b_l1b`` — gold front-month log return (Yahoo ``GC=F``).
  Materials weight. **Note:** the FRED London gold fixing series are delisted
  (they no longer resolve); ``GC=F`` is the replacement source, so this does not
  repeat the S&P 500 gold-degradation failure.
- ``tsx_usdcad_log_ret_1b_l1b`` — USD/CAD log return (Yahoo ``CAD=X``). The loonie
  is the single most important FX pair for TSX earnings translation.
- ``tsx_sp500_log_ret_1b_l1b`` — S&P 500 log return (Yahoo ``^GSPC``). US
  co-movement; expected to be the strongest single covariate.
- ``tsx_us10y_level_l1b`` — US 10-year Treasury yield level (Yahoo ``^TNX``,
  quoted directly in percent). Global rates. Sourced from Yahoo to stay
  FRED-free (FRED ``DGS10`` would need a key).
- ``tsx_boc_policy_rate_l1b`` — Bank of Canada target for the overnight rate,
  daily step series (StatCan ``10-10-0139-01`` member ``"Target rate"``, the
  same source the ``boc_rate_decisions`` implementation uses).
- ``tsx_goc10y_level_l1b`` — Government of Canada 10-year benchmark bond yield,
  daily (StatCan ``10-10-0139-01`` member ``"Government of Canada benchmark bond
  yields, 10 year"``). This is a clean **daily** leak-safe source reusing the
  existing ``StatCanAdapter`` (``release_lag_days=1``) — no BoC Valet adapter or
  monthly FRED fallback needed.
- ``tsx_ca_cpi_mom_logdiff_l1b`` — Canada CPI month-over-month log change
  (StatCan ``18-10-0004-11`` member ``"All-items"``), monthly with a
  conservative publication proxy (~end of the following month) before daily
  expansion.
- ``tsx_ca_unemployment_l1b`` — Canada unemployment rate, seasonally adjusted
  (StatCan LFS ``14-10-0287-03``), monthly with a conservative release lag.

Excluded
~~~~~~~~
- FRED-sourced covariates (US 10Y ``DGS10``, oil ``DCOILWTICO``, gold fixings):
  superseded by the Yahoo/StatCan sources above so the panel needs no API key.
- A dedicated TSX implied-vol index (e.g. ``VIXC``): no reliable free daily
  history, so CBOE VIX carries the risk-regime signal instead.

Anti-leakage policy (identical to the S&P 500 layer)
----------------------------------------------------
- Every covariate is transformed then lagged one business day so the feature at
  session ``t`` uses information only through ``t-1``.
- Low-frequency macro series get a conservative ``released_at`` before being
  expanded onto the daily business calendar, so a value becomes visible only
  once plausibly published.
- The ``DataService`` cutoff then guarantees context views never include
  unavailable rows.

StatCan covariates fetch through the ``stats-can`` WDS API. When that API is
unreachable (or a series is unavailable), :func:`build_tsx_multivariate_service`
with ``strict_covariates=False`` skips the covariate with a warning — the
Yahoo-sourced market factors still populate the panel — exactly as the S&P 500
layer degrades an unavailable covariate.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters import YFinanceDailyAdapter
from aieng.forecasting.data.adapters.statcan import StatCanAdapter
from aieng.forecasting.data.features import (
    StaticFrameAdapter,
    apply_one_business_day_feature_lag,
    business_daily_expand_from_releases,
    business_daily_ffill,
    canonical_three_col,
    drop_weekend_timestamp_rows,
    to_level_feature_from_daily,
    to_log_return_feature,
)


# ---------------------------------------------------------------------------
# Repo-root / cache resolution
# ---------------------------------------------------------------------------


def _repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / "aieng-forecasting").is_dir():
            return p
    return None


def _default_cache_dir() -> Path:
    root = _repo_root()
    return (root / "data") if root is not None else Path("data")


DEFAULT_CACHE_DIR = _default_cache_dir()
DEFAULT_YAHOO_CACHE_DIR = DEFAULT_CACHE_DIR / "yfinance"
DEFAULT_STATCAN_CACHE_DIR = DEFAULT_CACHE_DIR / "statcan"


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------

TSX_TICKER = "^GSPTSE"

#: Cumulative-return horizons (business days) registered as targets.
TSX_RETURN_WINDOWS: tuple[int, ...] = (1, 5, 21)

#: Human-readable framing per horizon, surfaced in metadata.
TSX_WINDOW_LABELS: dict[int, str] = {
    1: "next-session",
    5: "forward 1-week (5 business days)",
    21: "forward 1-month (21 business days)",
}


def tsx_logret_series_id(window: int) -> str:
    """Return the canonical target series id for an ``N``-business-day return."""
    return f"tsx_logret_{window}b"


#: Mapping from horizon (business days) to target series id.
TSX_RETURN_TARGETS: dict[int, str] = {w: tsx_logret_series_id(w) for w in TSX_RETURN_WINDOWS}

#: The canonical next-session (1-business-day) return target.
TSX_LOG_RETURN_SERIES_ID = tsx_logret_series_id(1)


# ---------------------------------------------------------------------------
# Covariate series ids
# ---------------------------------------------------------------------------

SERIES_ID_VIX_LEVEL = "tsx_vix_level_l1b"
SERIES_ID_VIX_CHANGE = "tsx_vix_log_ret_1b_l1b"
SERIES_ID_OIL_RETURN = "tsx_wti_oil_log_ret_1b_l1b"
SERIES_ID_GOLD_RETURN = "tsx_gold_log_ret_1b_l1b"
SERIES_ID_USDCAD_RETURN = "tsx_usdcad_log_ret_1b_l1b"
SERIES_ID_SP500_RETURN = "tsx_sp500_log_ret_1b_l1b"
SERIES_ID_US10Y_YIELD = "tsx_us10y_level_l1b"
SERIES_ID_BOC_POLICY_RATE = "tsx_boc_policy_rate_l1b"
SERIES_ID_GOC10Y_YIELD = "tsx_goc10y_level_l1b"
SERIES_ID_CA_CPI_INFLATION_CHANGE = "tsx_ca_cpi_mom_logdiff_l1b"
SERIES_ID_CA_UNEMPLOYMENT = "tsx_ca_unemployment_l1b"


#: Yahoo Finance tickers for the market covariates.
VIX_TICKER = "^VIX"
WTI_TICKER = "CL=F"
GOLD_TICKER = "GC=F"
USDCAD_TICKER = "CAD=X"
SP500_TICKER = "^GSPC"
US10Y_TICKER = "^TNX"

#: StatCan tables + member labels for the Canadian macro covariates.
STATCAN_RATES_TABLE = "10-10-0139-01"
STATCAN_CPI_TABLE = "18-10-0004-11"
STATCAN_LFS_TABLE = "14-10-0287-03"

BOC_TARGET_RATE_MEMBER = "Target rate"
GOC_10Y_YIELD_MEMBER = "Government of Canada benchmark bond yields, 10 year"

#: Member filter isolating the single seasonally-adjusted Canada unemployment-rate
#: series in the LFS table (verified to select exactly one monthly series).
CA_UNEMPLOYMENT_FILTER: dict[str, str] = {
    "GEO": "Canada",
    "Labour force characteristics": "Unemployment rate",
    "Gender": "Total - Gender",
    "Age group": "15 years and over",
    "Statistics": "Estimate",
    "Data type": "Seasonally adjusted",
}


DEFAULT_COVARIATE_SERIES_IDS: list[str] = [
    SERIES_ID_VIX_LEVEL,
    SERIES_ID_VIX_CHANGE,
    SERIES_ID_OIL_RETURN,
    SERIES_ID_GOLD_RETURN,
    SERIES_ID_USDCAD_RETURN,
    SERIES_ID_SP500_RETURN,
    SERIES_ID_US10Y_YIELD,
    SERIES_ID_BOC_POLICY_RATE,
    SERIES_ID_GOC10Y_YIELD,
    SERIES_ID_CA_CPI_INFLATION_CHANGE,
    SERIES_ID_CA_UNEMPLOYMENT,
]

#: The panel the workshop package re-exports (mirrors ``SP500_COVARIATE_PANEL``).
TSX_COVARIATE_PANEL: list[str] = list(DEFAULT_COVARIATE_SERIES_IDS)

#: Forecast horizons scored by the workshop, matching the registered targets.
WORKSHOP_HORIZONS: tuple[int, ...] = TSX_RETURN_WINDOWS


# ---------------------------------------------------------------------------
# Pure frame transforms (network-free; unit-tested directly)
# ---------------------------------------------------------------------------


def build_cumulative_log_return_frame(price_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """One row per session: ``value = log(close[t] / close[t-window])``.

    ``window=1`` is the ordinary daily close-to-close return; larger windows are
    trailing cumulative returns. ``released_at`` is the session timestamp (the
    return is known at that session's close). Mirrors the S&P 500 construction.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")
    if "value" not in price_df.columns:
        raise RuntimeError("Price data must include the close as 'value'.")
    frame = price_df[["timestamp", "value"]].copy().sort_values("timestamp").reset_index(drop=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame[frame["value"] > 0].dropna(subset=["value"]).reset_index(drop=True)
    frame["value"] = np.log(frame["value"] / frame["value"].shift(window))
    frame = frame.dropna(subset=["value"]).reset_index(drop=True)
    frame["released_at"] = pd.to_datetime(frame["timestamp"])
    return frame[["timestamp", "value", "released_at"]]


def daily_level_feature_from_statcan(raw: pd.DataFrame) -> pd.DataFrame:
    """Leak-safe daily *level* feature from a raw daily StatCan frame.

    Drops weekend stamps, forward-fills onto the full business-day calendar (so a
    Canadian bond-market holiday does not leave the covariate short of a target
    origin), then applies the one-business-day feature lag. Used for the BoC
    policy rate and the GoC 10-year yield.
    """
    x = canonical_three_col(raw)
    x = drop_weekend_timestamp_rows(x)
    x = business_daily_ffill(x)
    return apply_one_business_day_feature_lag(x)


def monthly_mom_logdiff_feature(
    raw: pd.DataFrame,
    *,
    release_bday_lag: int,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    """Leak-safe daily feature from a monthly index: MoM log change, release-expanded.

    ``released_at`` is set conservatively to ``month_end + release_bday_lag``
    business days (later than the true StatCan publication), the series is
    expanded onto the daily business calendar from those release stamps, then
    lagged one business day. Used for Canada CPI.
    """
    x = canonical_three_col(raw).sort_values("timestamp").reset_index(drop=True)
    x["value"] = np.log(x["value"] / x["value"].shift(1))
    x = x.dropna(subset=["value"]).reset_index(drop=True)
    x["released_at"] = pd.to_datetime(x["timestamp"]) + pd.offsets.MonthEnd(1) + pd.offsets.BDay(release_bday_lag)
    daily = business_daily_expand_from_releases(x, start=start, end=end)
    return apply_one_business_day_feature_lag(daily)


def monthly_level_feature(
    raw: pd.DataFrame,
    *,
    release_bday_lag: int,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    """Leak-safe daily feature from a monthly *level* series (no differencing).

    Conservative ``released_at`` (``month_end + release_bday_lag`` business days),
    daily release-expansion, then a one-business-day lag. Used for the Canada
    unemployment rate.
    """
    x = canonical_three_col(raw).sort_values("timestamp").reset_index(drop=True)
    x["released_at"] = pd.to_datetime(x["timestamp"]) + pd.offsets.MonthEnd(1) + pd.offsets.BDay(release_bday_lag)
    daily = business_daily_expand_from_releases(x, start=start, end=end)
    return apply_one_business_day_feature_lag(daily)


#: Conservative publication proxies (business days after the reference month
#: ends). StatCan CPI for month M is released in the third week of M+1; the LFS
#: is released the first/second Friday of M+1. Both proxies are set *later* than
#: the true release so a backtest never sees a print early.
_CPI_RELEASE_BDAY_LAG = 18
_UNEMPLOYMENT_RELEASE_BDAY_LAG = 12


# ---------------------------------------------------------------------------
# Service builder
# ---------------------------------------------------------------------------


def _yahoo_close_frame(
    ticker: str,
    *,
    start: str,
    end: str | None,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    adapter = YFinanceDailyAdapter(
        ticker,
        field="Adj Close",
        start=start,
        end=end,
        cache_dir=str(cache_dir),
        refresh=refresh,
    )
    raw = adapter.fetch()
    frame = raw[["timestamp", "value"]].copy().sort_values("timestamp").reset_index(drop=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["value"]).reset_index(drop=True)


def build_tsx_log_return_service(
    *,
    windows: tuple[int, ...] = TSX_RETURN_WINDOWS,
    refresh: bool = False,
    start: str = "2000-01-01",
    end: str | None = None,
    cache_dir: Path | None = None,
) -> DataService:
    """Register one close-to-close cumulative log-return target per window."""
    yahoo_dir = cache_dir or DEFAULT_YAHOO_CACHE_DIR
    yahoo_dir.mkdir(parents=True, exist_ok=True)
    price_df = _yahoo_close_frame(TSX_TICKER, start=start, end=end, cache_dir=yahoo_dir, refresh=refresh)

    svc = DataService()
    for window in windows:
        series_id = tsx_logret_series_id(window)
        label = TSX_WINDOW_LABELS.get(window, f"{window} business days")
        svc.register(
            series_id,
            StaticFrameAdapter(build_cumulative_log_return_frame(price_df, window)),
            SeriesMetadata(
                series_id=series_id,
                description=(
                    f"S&P/TSX Composite close-to-close cumulative log return over {window} "
                    f"business day(s) ({label}) (Yahoo Finance {TSX_TICKER}, derived)"
                ),
                source=f"Yahoo Finance ({TSX_TICKER}), derived",
                units="log-return",
                frequency="B",
                table_id=f"yahoo:{TSX_TICKER}:logret-{window}b",
            ),
        )
    return svc


def build_tsx_multivariate_service(  # noqa: PLR0912, PLR0915
    *,
    windows: tuple[int, ...] = TSX_RETURN_WINDOWS,
    include_covariates: bool = True,
    covariate_series_ids: list[str] | None = None,
    strict_covariates: bool = False,
    refresh: bool = False,
    start: str = "2000-01-01",
    end: str | None = None,
    yahoo_cache_dir: Path | None = None,
    statcan_cache_dir: Path | None = None,
) -> DataService:
    """Build the TSX :class:`DataService` with the target plus leak-safe covariates.

    Parameters
    ----------
    strict_covariates : bool
        If ``True``, any covariate fetch/build failure raises. If ``False``
        (default), unavailable covariates (e.g. a StatCan series when the WDS API
        is unreachable) are skipped with a warning, mirroring the S&P 500 layer.
    """
    svc = build_tsx_log_return_service(
        windows=windows, refresh=refresh, start=start, end=end, cache_dir=yahoo_cache_dir
    )
    if not include_covariates:
        return svc

    desired = set(covariate_series_ids or DEFAULT_COVARIATE_SERIES_IDS)
    yahoo_dir = yahoo_cache_dir or DEFAULT_YAHOO_CACHE_DIR
    statcan_dir = statcan_cache_dir or DEFAULT_STATCAN_CACHE_DIR
    yahoo_dir.mkdir(parents=True, exist_ok=True)
    statcan_dir.mkdir(parents=True, exist_ok=True)

    def _handle_error(series_id: str, exc: Exception) -> None:
        if strict_covariates:
            raise RuntimeError(f"Failed to build required covariate {series_id!r}.") from exc
        warnings.warn(f"Skipping unavailable covariate {series_id!r}: {exc}", stacklevel=2)

    def _yahoo(ticker: str) -> pd.DataFrame:
        return _yahoo_close_frame(ticker, start=start, end=end, cache_dir=yahoo_dir, refresh=refresh)

    # ── VIX level + log return ───────────────────────────────────────────────
    if SERIES_ID_VIX_LEVEL in desired or SERIES_ID_VIX_CHANGE in desired:
        try:
            vix = _yahoo(VIX_TICKER)
            if SERIES_ID_VIX_LEVEL in desired:
                svc.register(
                    SERIES_ID_VIX_LEVEL,
                    StaticFrameAdapter(apply_one_business_day_feature_lag(to_level_feature_from_daily(vix))),
                    SeriesMetadata(
                        series_id=SERIES_ID_VIX_LEVEL,
                        description="CBOE VIX close level, lagged 1 business day",
                        source=f"Yahoo Finance ({VIX_TICKER})",
                        units="index-level",
                        frequency="B",
                        table_id=f"yahoo:{VIX_TICKER}:close-l1b",
                    ),
                )
            if SERIES_ID_VIX_CHANGE in desired:
                svc.register(
                    SERIES_ID_VIX_CHANGE,
                    StaticFrameAdapter(apply_one_business_day_feature_lag(to_log_return_feature(vix))),
                    SeriesMetadata(
                        series_id=SERIES_ID_VIX_CHANGE,
                        description="CBOE VIX close-to-close log return, lagged 1 business day",
                        source=f"Yahoo Finance ({VIX_TICKER}), derived",
                        units="log-return",
                        frequency="B",
                        table_id=f"yahoo:{VIX_TICKER}:log-return-l1b",
                    ),
                )
        except (RuntimeError, ValueError) as exc:
            if SERIES_ID_VIX_LEVEL in desired:
                _handle_error(SERIES_ID_VIX_LEVEL, exc)
            if SERIES_ID_VIX_CHANGE in desired:
                _handle_error(SERIES_ID_VIX_CHANGE, exc)

    # ── Yahoo daily log-return covariates ────────────────────────────────────
    _yahoo_return_specs: list[tuple[str, str, str]] = [
        (SERIES_ID_OIL_RETURN, WTI_TICKER, "WTI crude front-month log return, lagged 1 business day"),
        (SERIES_ID_GOLD_RETURN, GOLD_TICKER, "Gold front-month log return, lagged 1 business day"),
        (SERIES_ID_USDCAD_RETURN, USDCAD_TICKER, "USD/CAD log return, lagged 1 business day"),
        (SERIES_ID_SP500_RETURN, SP500_TICKER, "S&P 500 close-to-close log return, lagged 1 business day"),
    ]
    for series_id, ticker, description in _yahoo_return_specs:
        if series_id not in desired:
            continue
        try:
            frame = apply_one_business_day_feature_lag(to_log_return_feature(_yahoo(ticker)))
            svc.register(
                series_id,
                StaticFrameAdapter(frame),
                SeriesMetadata(
                    series_id=series_id,
                    description=description,
                    source=f"Yahoo Finance ({ticker}), derived",
                    units="log-return",
                    frequency="B",
                    table_id=f"yahoo:{ticker}:log-return-l1b",
                ),
            )
        except (RuntimeError, ValueError) as exc:
            _handle_error(series_id, exc)

    # ── US 10Y Treasury yield level (Yahoo ^TNX, already in percent) ──────────
    if SERIES_ID_US10Y_YIELD in desired:
        try:
            us10y = apply_one_business_day_feature_lag(to_level_feature_from_daily(_yahoo(US10Y_TICKER)))
            svc.register(
                SERIES_ID_US10Y_YIELD,
                StaticFrameAdapter(us10y),
                SeriesMetadata(
                    series_id=SERIES_ID_US10Y_YIELD,
                    description="US 10-year Treasury yield level, lagged 1 business day",
                    source=f"Yahoo Finance ({US10Y_TICKER})",
                    units="percent",
                    frequency="B",
                    table_id=f"yahoo:{US10Y_TICKER}:level-l1b",
                ),
            )
        except (RuntimeError, ValueError) as exc:
            _handle_error(SERIES_ID_US10Y_YIELD, exc)

    # ── BoC policy rate + GoC 10Y yield (StatCan daily) ──────────────────────
    _statcan_daily_specs: list[tuple[str, str, str, str]] = [
        (
            SERIES_ID_BOC_POLICY_RATE,
            BOC_TARGET_RATE_MEMBER,
            "Bank of Canada target for the overnight rate (policy rate), lagged 1 business day",
            "boc-target-rate",
        ),
        (
            SERIES_ID_GOC10Y_YIELD,
            GOC_10Y_YIELD_MEMBER,
            "Government of Canada 10-year benchmark bond yield, lagged 1 business day",
            "goc-10y",
        ),
    ]
    for series_id, member, description, tag in _statcan_daily_specs:
        if series_id not in desired:
            continue
        try:
            adapter = StatCanAdapter(
                table_id=STATCAN_RATES_TABLE,
                member_filter={"GEO": "Canada", "Financial market statistics": member},
                cache_dir=statcan_dir,
                release_lag_days=1,
            )
            frame = daily_level_feature_from_statcan(adapter.fetch())
            svc.register(
                series_id,
                StaticFrameAdapter(frame),
                SeriesMetadata(
                    series_id=series_id,
                    description=description,
                    source=f"StatCan ({STATCAN_RATES_TABLE})",
                    units="percent",
                    frequency="B",
                    table_id=f"statcan:{STATCAN_RATES_TABLE}:{tag}-l1b",
                ),
            )
        except (RuntimeError, ValueError) as exc:
            _handle_error(series_id, exc)

    # ── Canada CPI MoM log change (StatCan monthly) ──────────────────────────
    if SERIES_ID_CA_CPI_INFLATION_CHANGE in desired:
        try:
            adapter = StatCanAdapter(
                table_id=STATCAN_CPI_TABLE,
                member_filter={"GEO": "Canada", "Products and product groups": "All-items"},
                cache_dir=statcan_dir,
            )
            frame = monthly_mom_logdiff_feature(
                adapter.fetch(), release_bday_lag=_CPI_RELEASE_BDAY_LAG, start=start, end=end
            )
            svc.register(
                SERIES_ID_CA_CPI_INFLATION_CHANGE,
                StaticFrameAdapter(frame),
                SeriesMetadata(
                    series_id=SERIES_ID_CA_CPI_INFLATION_CHANGE,
                    description="Canada CPI (All-items) MoM log change, conservative release lag + 1B feature lag",
                    source=f"StatCan ({STATCAN_CPI_TABLE}), derived",
                    units="log-change",
                    frequency="B",
                    table_id=f"statcan:{STATCAN_CPI_TABLE}:cpi-mom-l1b",
                ),
            )
        except (RuntimeError, ValueError) as exc:
            _handle_error(SERIES_ID_CA_CPI_INFLATION_CHANGE, exc)

    # ── Canada unemployment rate (StatCan LFS monthly, SA) ───────────────────
    if SERIES_ID_CA_UNEMPLOYMENT in desired:
        try:
            adapter = StatCanAdapter(
                table_id=STATCAN_LFS_TABLE,
                member_filter=CA_UNEMPLOYMENT_FILTER,
                cache_dir=statcan_dir,
            )
            frame = monthly_level_feature(
                adapter.fetch(), release_bday_lag=_UNEMPLOYMENT_RELEASE_BDAY_LAG, start=start, end=end
            )
            svc.register(
                SERIES_ID_CA_UNEMPLOYMENT,
                StaticFrameAdapter(frame),
                SeriesMetadata(
                    series_id=SERIES_ID_CA_UNEMPLOYMENT,
                    description="Canada unemployment rate (SA), conservative release lag + 1B feature lag",
                    source=f"StatCan ({STATCAN_LFS_TABLE})",
                    units="percent",
                    frequency="B",
                    table_id=f"statcan:{STATCAN_LFS_TABLE}:unemployment-l1b",
                ),
            )
        except (RuntimeError, ValueError) as exc:
            _handle_error(SERIES_ID_CA_UNEMPLOYMENT, exc)

    return svc


def build_tsx_workshop_service(
    *,
    include_covariates: bool = True,
    end: str | None = None,
    refresh: bool = False,
    start: str = "2000-01-01",
) -> DataService:
    """Build the TSX :class:`DataService` for the workshop experiments.

    Mirrors :func:`workshop_experiments.data.build_workshop_service`: registers
    the ``tsx_logret_{1,5,21}b`` targets and (when requested) the leak-safe
    covariate panel. Unavailable covariates are skipped, never fatal.
    """
    return build_tsx_multivariate_service(
        include_covariates=include_covariates,
        strict_covariates=False,
        refresh=refresh,
        start=start,
        end=end,
    )


__all__ = [
    "DEFAULT_COVARIATE_SERIES_IDS",
    "SERIES_ID_BOC_POLICY_RATE",
    "SERIES_ID_CA_CPI_INFLATION_CHANGE",
    "SERIES_ID_CA_UNEMPLOYMENT",
    "SERIES_ID_GOC10Y_YIELD",
    "SERIES_ID_GOLD_RETURN",
    "SERIES_ID_OIL_RETURN",
    "SERIES_ID_SP500_RETURN",
    "SERIES_ID_US10Y_YIELD",
    "SERIES_ID_USDCAD_RETURN",
    "SERIES_ID_VIX_CHANGE",
    "SERIES_ID_VIX_LEVEL",
    "TSX_COVARIATE_PANEL",
    "TSX_LOG_RETURN_SERIES_ID",
    "TSX_RETURN_TARGETS",
    "TSX_RETURN_WINDOWS",
    "TSX_TICKER",
    "TSX_WINDOW_LABELS",
    "WORKSHOP_HORIZONS",
    "build_cumulative_log_return_frame",
    "build_tsx_log_return_service",
    "build_tsx_multivariate_service",
    "build_tsx_workshop_service",
    "daily_level_feature_from_statcan",
    "monthly_level_feature",
    "monthly_mom_logdiff_feature",
    "tsx_logret_series_id",
]
