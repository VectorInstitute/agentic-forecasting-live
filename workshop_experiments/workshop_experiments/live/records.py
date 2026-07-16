"""Build and validate data-contract records from ``Prediction`` objects.

Turns the ``list[Prediction]`` a predictor emits for one origin (one per
horizon) into a single prediction record conforming to
``monitor/schemas/prediction.schema.json``, and builds resolution records for
matured horizons. Enforces the three non-schema writer invariants from
``monitor/DESIGN.md`` before any record is handed on:

1. the ``quantiles`` set equals the standard 11-point grid exactly;
2. values are non-decreasing across the grid;
3. ``point_estimate`` equals the 0.50 quantile value.

Known limitation (accepted): ``curated_trace_summary`` is populated *when
available* — the writer curates whatever structured ``tool_calls`` list the
agent path surfaces in prediction metadata, but the current agent runtime does
not yet emit one, so agent records legitimately carry an empty ``tool_calls``
list until that upstream wiring lands. An empty list means "no structured tool
calls captured", not "no tools used".
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast

from workshop_experiments.live import SCHEMA_VERSION
from workshop_experiments.live.config import LiveConfig, LivePredictor


#: NYSE regular-session close, US/Eastern. Converted to UTC per origin so
#: ``origin_timestamp`` reflects the true information cutoff across DST.
_MARKET_CLOSE_HOUR = 16
_NY_TZ = ZoneInfo("America/New_York")

#: Absolute tolerance when checking ``point_estimate == q50``.
_POINT_TOL = 1e-9


def utc_now_z() -> str:
    """Return the current UTC time as a schema timestamp (``...Z``, seconds)."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def market_close_timestamp(origin: date) -> str:
    """Return the UTC market-close timestamp for *origin* as a schema string."""
    close_local = datetime(origin.year, origin.month, origin.day, _MARKET_CLOSE_HOUR, tzinfo=_NY_TZ)
    return close_local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class WriterInvariantError(ValueError):
    """Raised when a forecast violates a non-schema writer invariant."""


def _grid_from_forecast(forecast: ContinuousForecast) -> list[dict[str, float]]:
    """Return the validated 11-point quantile grid for one continuous forecast.

    Enforces grid-completeness, monotonicity, and point==median. The returned
    list is ordered by ascending quantile level, matching the standard grid.
    """
    quantiles = forecast.quantiles
    keys = sorted(quantiles)
    if keys != sorted(STANDARD_QUANTILES):
        raise WriterInvariantError(f"quantile grid must equal the standard 11-point grid exactly; got {keys}")

    grid = [{"quantile": q, "value": float(quantiles[q])} for q in STANDARD_QUANTILES]
    values = [point["value"] for point in grid]
    if any(b < a for a, b in zip(values, values[1:])):
        raise WriterInvariantError(f"quantile values must be non-decreasing; got {values}")

    median = float(quantiles[0.50])
    if abs(float(forecast.point_forecast) - median) > _POINT_TOL:
        raise WriterInvariantError(
            f"point_estimate ({forecast.point_forecast}) must equal the 0.50 quantile ({median})"
        )
    return grid


def _continuous_payload(prediction: Any) -> ContinuousForecast:
    """Return the ``ContinuousForecast`` payload of a prediction, or raise."""
    payload = prediction.payload
    if not isinstance(payload, ContinuousForecast):
        raise WriterInvariantError(f"live records require ContinuousForecast payloads; got {type(payload).__name__}")
    return payload


def curated_trace_summary(metadata: dict[str, Any] | None) -> dict[str, list[dict[str, str]]]:
    """Curate the public trace summary: tool names + query titles ONLY.

    Reads an optional ``tool_calls`` list from prediction metadata and keeps only
    the ``tool`` and ``query_title`` of each entry. Never emits retrieved article
    bodies or prompt scaffolding — anything else in metadata is dropped. Returns
    an empty ``tool_calls`` list when no structured tool calls are present.
    """
    tool_calls: list[dict[str, str]] = []
    for call in (metadata or {}).get("tool_calls", []) or []:
        if not isinstance(call, dict):
            continue
        tool = call.get("tool")
        if not tool:
            continue
        tool_calls.append({"tool": str(tool), "query_title": str(call.get("query_title", ""))})
    return {"tool_calls": tool_calls}


def _rationale_and_trace(predictions: list[Any]) -> tuple[str | None, str | None, dict[str, Any]]:
    """Pull the public rationale, langfuse trace id, and merged metadata.

    Agents surface ``rationale`` in metadata; LLMP predictors may carry a brief
    forecast rationale under ``rationale`` too (else ``None``); conventional
    methods carry neither. The trace id is taken from any horizon's metadata.
    """
    merged: dict[str, Any] = {}
    for prediction in predictions:
        merged.update(prediction.metadata or {})
    rationale = merged.get("rationale")
    rationale = str(rationale) if rationale not in (None, "") else None
    trace_id = merged.get("langfuse_trace_id")
    trace_id = str(trace_id) if trace_id not in (None, "") else None
    return rationale, trace_id, merged


def build_prediction_record(
    live: LivePredictor,
    predictions_by_horizon: dict[int, Any],
    *,
    origin: date,
    submission_timestamp: str,
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, Any]:
    """Assemble one schema-conforming prediction record for a rung at an origin.

    Parameters
    ----------
    live : LivePredictor
        The rung (supplies method/model/predictor_id identity).
    predictions_by_horizon : dict[int, Prediction]
        One ``Prediction`` per configured horizon, keyed by horizon.
    origin : datetime.date
        The origin (data-through-close) trading day.
    submission_timestamp : str
        Wall-clock commit time (schema timestamp).
    schema_version : str
        Data-contract version stamped on the record.

    Returns
    -------
    dict
        A ``prediction.schema.json``-conforming record. Writer invariants are
        enforced per horizon; a violation raises :class:`WriterInvariantError`.
    """
    horizons: list[dict[str, Any]] = []
    for horizon in sorted(predictions_by_horizon):
        prediction = predictions_by_horizon[horizon]
        forecast = _continuous_payload(prediction)
        grid = _grid_from_forecast(forecast)
        entry: dict[str, Any] = {
            "horizon": int(horizon),
            "point_estimate": float(forecast.point_forecast),
            "quantiles": grid,
        }
        horizon_rationale = (prediction.metadata or {}).get("horizon_rationale")
        if horizon_rationale not in (None, ""):
            entry["horizon_rationale"] = str(horizon_rationale)
        horizons.append(entry)

    rationale, trace_id, merged = _rationale_and_trace(list(predictions_by_horizon.values()))
    record: dict[str, Any] = {
        "schema_version": schema_version,
        "origin_date": origin.isoformat(),
        "origin_timestamp": market_close_timestamp(origin),
        "submission_timestamp": submission_timestamp,
        "method": live.schema_method,
        "model": live.model_label,
        "predictor_id": live.predictor_id,
        "horizons": horizons,
        "curated_trace_summary": curated_trace_summary(merged),
        "langfuse_trace_id": trace_id,
    }
    if rationale is not None:
        record["rationale"] = rationale
    return record


def build_resolution_record(
    live: LivePredictor,
    *,
    origin: date,
    horizon: int,
    forecast_date: date,
    realized_value: float,
    crps: float,
    resolved_at: str,
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, Any]:
    """Assemble one ``resolution.schema.json``-conforming record."""
    return {
        "schema_version": schema_version,
        "origin_date": origin.isoformat(),
        "method": live.schema_method,
        "model": live.model_label,
        "predictor_id": live.predictor_id,
        "horizon": int(horizon),
        "forecast_date": forecast_date.isoformat(),
        "realized_value": float(realized_value),
        "crps": float(crps),
        "resolved_at": resolved_at,
    }


def group_predictions_by_horizon(config: LiveConfig, predictions: list[Any]) -> dict[int, Any]:
    """Map a rung's ``list[Prediction]`` to ``{horizon: Prediction}``.

    Uses the config's ``task_id`` <-> horizon binding, so a prediction's
    ``task_id`` (``sp500_logret_{h}b``) selects its horizon. Predictions for
    task ids outside the configured horizons are ignored.
    """
    task_to_horizon = {config.task_id_for_horizon(h): h for h in config.horizons}
    by_horizon: dict[int, Any] = {}
    for prediction in predictions:
        horizon = task_to_horizon.get(prediction.task_id)
        if horizon is not None:
            by_horizon[horizon] = prediction
    return by_horizon


__all__ = [
    "WriterInvariantError",
    "build_prediction_record",
    "build_resolution_record",
    "curated_trace_summary",
    "group_predictions_by_horizon",
    "market_close_timestamp",
    "utc_now_z",
]
