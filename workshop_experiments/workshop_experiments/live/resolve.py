"""The resolve step: score matured horizons and append resolution records.

Finds unresolved ``(origin, horizon)`` pairs whose target session has closed,
computes the realized cumulative log return and the CRPS of the stored quantile
grid against it, and appends a resolution record per
``monitor/schemas/resolution.schema.json``.

CRPS reuses the repo's existing implementation — ``properscoring.crps_ensemble``
over the sorted quantile grid, exactly as
:func:`workshop_experiments.scoring._crps` scores backtest predictions — so live
and offline scores are computed the same way.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import properscoring as ps

from workshop_experiments.live import SCHEMA_VERSION
from workshop_experiments.live.log_store import (
    has_resolution,
    iter_prediction_records,
    write_resolution,
)


def crps_from_grid(grid: list[dict[str, float]], actual: float) -> float:
    """CRPS of a quantile grid against a realized value.

    The grid's values form the ensemble (sorted), matching how
    :func:`workshop_experiments.scoring._crps` scores a ``ContinuousForecast``.

    Parameters
    ----------
    grid : list[dict]
        The record's ``quantiles`` list (each ``{"quantile", "value"}``).
    actual : float
        The realized value.

    Returns
    -------
    float
        The CRPS (>= 0; lower is better).
    """
    ensemble = sorted(float(point["value"]) for point in grid)
    return float(ps.crps_ensemble(actual, ensemble))


class RealizedProvider(Protocol):
    """Resolves ``(horizon, origin)`` to a matured ``(forecast_date, value)``."""

    def realized(self, horizon: int, origin: date) -> tuple[date, float] | None:
        """Return the resolved outcome, or ``None`` if the horizon is unmatured."""
        ...


class LookupRealizedProvider:
    """Realized provider backed by an in-memory ``{horizon: {date: value}}`` map.

    ``forecast_date`` is the *horizon*-th business day after the origin (pandas
    ``BDay``); the outcome resolves only when that date is present in the lookup
    (i.e. that session has closed and its realized return is known). Used by
    ``--simulate`` with a lookup reconstructed offline from committed data.
    """

    def __init__(self, lookup: dict[int, dict[date, float]]) -> None:
        """Store the ``{horizon: {forecast_date: realized_value}}`` lookup."""
        self._lookup = lookup

    def realized(self, horizon: int, origin: date) -> tuple[date, float] | None:
        """Resolve via ``origin + horizon`` business days against the lookup."""
        forecast_date = (pd.Timestamp(origin) + pd.tseries.offsets.BDay(horizon)).date()
        by_date = self._lookup.get(horizon, {})
        if forecast_date in by_date:
            return forecast_date, by_date[forecast_date]
        return None


class DataServiceRealizedProvider:
    """Realized provider backed by the live data service (real runs).

    Resolves against the registered ``sp500_logret_{h}b`` target series: the
    forecast date is the *horizon*-th trading session at/after ``origin`` in the
    series index (holiday-robust), and the realized value is that session's
    target value. Returns ``None`` until that session has closed.
    """

    def __init__(self, data_service: Any, task_id_for_horizon: Any) -> None:
        """Store the data service and the ``horizon -> target series id`` map."""
        self._data_service = data_service
        self._task_id_for_horizon = task_id_for_horizon
        self._series_cache: dict[int, pd.Series] = {}

    def _series(self, horizon: int) -> pd.Series:
        """Return the timestamp-indexed realized series for a horizon (cached)."""
        if horizon not in self._series_cache:
            from datetime import datetime, timezone  # noqa: PLC0415

            series_id = self._task_id_for_horizon(horizon)
            now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
            frame = self._data_service.get_series(series_id, as_of=now).copy()
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            self._series_cache[horizon] = frame.set_index("timestamp")["value"]
        return self._series_cache[horizon]

    def realized(self, horizon: int, origin: date) -> tuple[date, float] | None:
        """Resolve against the *horizon*-th session after ``origin``, if closed."""
        series = self._series(horizon)
        after = series.index[series.index > pd.Timestamp(origin)]
        if len(after) < horizon:
            return None
        forecast_ts = after[horizon - 1]
        return forecast_ts.date(), float(series.loc[forecast_ts])


def resolve_log(
    log_dir: Path,
    provider: RealizedProvider,
    *,
    resolved_at: str,
    schema_version: str = SCHEMA_VERSION,
) -> list[dict[str, Any]]:
    """Resolve every matured, unresolved ``(origin, horizon)`` in the log.

    Reads prediction records, and for each horizon not yet resolved asks the
    *provider* for the realized outcome; when available, scores the stored grid
    and writes a resolution record. Returns the resolution records written this
    call (deterministically ordered).

    Parameters
    ----------
    log_dir : Path
        Root of the append-only log.
    provider : RealizedProvider
        Supplies realized outcomes for matured horizons.
    resolved_at : str
        Schema timestamp stamped on each new resolution.
    schema_version : str
        Data-contract version stamped on each record.

    Returns
    -------
    list[dict]
        The resolution records written this call.
    """
    written: list[dict[str, Any]] = []
    for record in iter_prediction_records(log_dir):
        origin = date.fromisoformat(record["origin_date"])
        for horizon_forecast in record["horizons"]:
            horizon = int(horizon_forecast["horizon"])
            if has_resolution(log_dir, origin, record["predictor_id"], horizon):
                continue
            outcome = provider.realized(horizon, origin)
            if outcome is None:
                continue
            forecast_date, realized_value = outcome
            resolution = {
                "schema_version": schema_version,
                "origin_date": record["origin_date"],
                "method": record["method"],
                "model": record["model"],
                "predictor_id": record["predictor_id"],
                "horizon": horizon,
                "forecast_date": forecast_date.isoformat(),
                "realized_value": float(realized_value),
                "crps": crps_from_grid(horizon_forecast["quantiles"], float(realized_value)),
                "resolved_at": resolved_at,
            }
            write_resolution(log_dir, origin, resolution)
            written.append(resolution)
    return sorted(written, key=lambda r: (r["origin_date"], r["predictor_id"], r["horizon"]))


__all__ = [
    "DataServiceRealizedProvider",
    "LookupRealizedProvider",
    "RealizedProvider",
    "crps_from_grid",
    "resolve_log",
]
