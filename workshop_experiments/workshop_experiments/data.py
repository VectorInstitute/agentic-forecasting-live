"""Data-service construction for the workshop S&P 500 experiments.

Thin wrapper over the ``sp500_forecasting`` reference implementation so the
workshop package has a single entry point for the leak-safe target + covariate
panel. Nothing here re-derives data logic — it delegates to
:func:`sp500_forecasting.data.build_sp500_multivariate_service`, which registers
the ``sp500_logret_{1,5,21}b`` targets and the lagged macro/market covariates.

Price caches live under the repo-root ``data/`` tree (git-ignored); only the
persisted predictions under ``workshop_experiments/data/predictions/`` are
committed.
"""

from __future__ import annotations

from aieng.forecasting.data.service import DataService
from sp500_forecasting.data import (
    DEFAULT_COVARIATE_SERIES_IDS,
    SP500_RETURN_WINDOWS,
    build_sp500_multivariate_service,
)


#: The full leak-safe covariate panel used by the ``*_cov`` predictor variants
#: (VIX level/change, Treasury yields, curve slope, Fed funds, CPI, unemployment,
#: oil/gold/dollar returns, NASDAQ returns). Sourced from the sp500 reference
#: implementation so the workshop and the bootcamp share one definition.
SP500_COVARIATE_PANEL: list[str] = list(DEFAULT_COVARIATE_SERIES_IDS)

#: Forecast horizons (business days) the workshop scores at, matching the
#: registered target windows.
WORKSHOP_HORIZONS: tuple[int, ...] = SP500_RETURN_WINDOWS


def build_workshop_service(
    *,
    include_covariates: bool = True,
    end: str | None = None,
    refresh: bool = False,
    start: str = "1990-01-01",
) -> DataService:
    """Build the S&P 500 :class:`DataService` for the workshop experiments.

    Parameters
    ----------
    include_covariates : bool, default=True
        Register the full leak-safe covariate panel in addition to the targets.
        Set ``False`` for target-only runs (conventional univariate methods only
        need the targets, but registering covariates is cheap and harmless).
    end : str or None
        Optional exclusive upper bound (``YYYY-MM-DD``) on fetched data. ``None``
        fetches through the latest available observation. Backtest/eval cutoffs
        are enforced per-origin by the :class:`ForecastContext`, so this is only
        a fetch-scope convenience.
    refresh : bool, default=False
        Force a live re-fetch instead of reading the cache.
    start : str, default="1990-01-01"
        Inclusive lower bound on fetched history.

    Returns
    -------
    DataService
        Populated service with the ``sp500_logret_{1,5,21}b`` targets and (when
        requested) the covariate panel registered.
    """
    return build_sp500_multivariate_service(
        include_covariates=include_covariates,
        strict_covariates=False,
        refresh=refresh,
        start=start,
        end=end,
    )


__all__ = [
    "SP500_COVARIATE_PANEL",
    "WORKSHOP_HORIZONS",
    "build_workshop_service",
]
