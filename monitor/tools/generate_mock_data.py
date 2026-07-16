"""Generate mock fixtures for the experiment-monitor static site.

This script writes deterministic, schema-conforming JSON fixtures into
``monitor/site/data/`` so the prototype renders with realistic-looking
content before the live harness exists. Every emitted file carries
``generated_by: "mock"`` so the site can show its "MOCK DATA" banner.

The numbers are *plausible*, not real: daily S&P 500 cumulative log-return
forecasts are drawn from per-method normal predictive distributions, and CRPS
is computed in closed form against a simulated realized path so the drill-down
fan charts and the leaderboard scores stay mutually consistent.

Run from anywhere::

    python monitor/tools/generate_mock_data.py

It reproduces byte-identical output on every run (fixed RNG seed).
"""

from __future__ import annotations

import json
import math
import random
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import NormalDist


SCHEMA_VERSION = "1.0.0"

# The canonical grid from aieng.forecasting.evaluation.prediction.STANDARD_QUANTILES.
STANDARD_QUANTILES: list[float] = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

HORIZONS: list[int] = [1, 5, 21]

CONVENTIONAL_METHODS = ["naive", "classical", "lightgbm"]
LLM_METHODS = ["llm_process", "analyst_agent", "code_agent"]
LLM_MODELS = ["gemini-3.1-flash-lite-preview", "gemini-3.5-flash", "claude-haiku-4.5"]

# Relative CRPS skill: 1.0 == well calibrated; >1 == inflated/miscalibrated bands.
METHOD_SKILL: dict[str, float] = {
    "naive": 1.35,
    "classical": 1.12,
    "lightgbm": 1.05,
    "llm_process": 1.08,
    "analyst_agent": 1.18,
    "code_agent": 1.10,
}
MODEL_SKILL: dict[str, float] = {
    "gemini-3.1-flash-lite-preview": 1.10,
    "gemini-3.5-flash": 0.98,
    "claude-haiku-4.5": 1.02,
}

# Realized daily log-return volatility used to simulate the outcome path.
DAILY_VOL = 0.0092

# Base predictive sigma per horizon (sqrt-of-time scaling of DAILY_VOL, roughly).
BASE_SIGMA: dict[int, float] = {h: DAILY_VOL * math.sqrt(h) * 1.05 for h in HORIZONS}

N_ORIGINS = 60
LAST_ORIGIN = date(2026, 7, 14)  # a Tuesday; "today" for the mock world.

# Origins that failed and were logged as gaps (no predictions written for them).
GAP_DATES = {date(2026, 6, 5), date(2026, 6, 24)}

DATA_DIR = Path(__file__).resolve().parents[1] / "site" / "data"


def business_days_back(anchor: date, count: int) -> list[date]:
    """Return ``count`` business days ending at ``anchor``, oldest first.

    Weekends are skipped; holidays are ignored (mock calendar).

    Parameters
    ----------
    anchor : date
        The most recent business day (inclusive).
    count : int
        Number of business days to return.

    Returns
    -------
    list[date]
        Business days in ascending order, ending at ``anchor``.
    """
    days: list[date] = []
    cursor = anchor
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return list(reversed(days))


def future_business_days(anchor: date, count: int) -> list[date]:
    """Return the next ``count`` business days strictly after ``anchor``.

    Parameters
    ----------
    anchor : date
        The day to walk forward from (exclusive).
    count : int
        Number of business days to return.

    Returns
    -------
    list[date]
        Business days in ascending order.
    """
    days: list[date] = []
    cursor = anchor
    while len(days) < count:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:
            days.append(cursor)
    return days


def crps_normal(mu: float, sigma: float, y: float) -> float:
    """Return the closed-form CRPS of a Gaussian predictive distribution.

    Parameters
    ----------
    mu : float
        Predictive mean.
    sigma : float
        Predictive standard deviation (> 0).
    y : float
        Realized outcome.

    Returns
    -------
    float
        Continuous Ranked Probability Score (lower is better).
    """
    std = NormalDist()
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * std.cdf(z) - 1.0) + 2.0 * std.pdf(z) - 1.0 / math.sqrt(math.pi))


def iso_z(moment: datetime) -> str:
    """Render a UTC datetime as an ISO-8601 string with a trailing ``Z``.

    Parameters
    ----------
    moment : datetime
        A timezone-aware UTC datetime.

    Returns
    -------
    str
        ISO-8601 timestamp, e.g. ``2026-07-14T20:15:00Z``.
    """
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def origin_close_dt(origin: date) -> datetime:
    """Return the market-close timestamp for an origin (NYSE 16:00 ET as UTC 20:00).

    Parameters
    ----------
    origin : date
        The trading day.

    Returns
    -------
    datetime
        Timezone-aware UTC datetime approximating the NYSE close.
    """
    return datetime.combine(origin, time(20, 0), tzinfo=timezone.utc)


def rationale_text(method: str, origin: date, direction: str) -> str:
    """Compose an agent-authored, public rationale string for a forecast.

    Parameters
    ----------
    method : str
        Method identifier (only LLM methods receive rationales).
    origin : date
        The forecast origin date.
    direction : str
        A coarse directional read ("modestly higher", "roughly flat", ...).

    Returns
    -------
    str
        Public rationale text (no retrieved article bodies, no prompt scaffolding).
    """
    if method == "analyst_agent":
        return (
            f"As of the {origin:%B %d, %Y} close I read the tape as {direction} over the next month. "
            "Retrieved coverage this week centers on the Fed's next move and steady mega-cap earnings; "
            "no scheduled catalyst before the h=5 horizon resolves. Realized vol has drifted lower, so I "
            "kept the central path near flat but widened the lower tail to respect headline risk into the "
            "h=21 window. Nothing in the narrative justified a directional bet beyond a small drift."
        )
    if method == "code_agent":
        return (
            f"Computed 20-day realized vol and a short-horizon momentum score on the panel through "
            f"{origin:%Y-%m-%d}. Realized vol sits below its trailing median and the trend state is mildly "
            f"positive, so I nudged the median {direction} and set band width from the empirical 20-day "
            "return distribution rather than a constant. Widened h=21 bands to cover the fatter monthly "
            "tail I measured on history."
        )
    return (
        f"Quantile grid drawn from the model's conditional distribution given returns through "
        f"{origin:%Y-%m-%d}; central path {direction}, bands scaled by horizon."
    )


def curated_trace(method: str, origin: date) -> dict[str, object]:
    """Return the public, curated trace summary for an LLM/agent forecast.

    Only tool names and query *titles* are exposed. Raw retrieved article text
    and internal prompt scaffolding are never included here.

    Parameters
    ----------
    method : str
        Method identifier.
    origin : date
        Forecast origin date.

    Returns
    -------
    dict[str, object]
        Curated trace summary with a ``tool_calls`` list.
    """
    if method == "analyst_agent":
        tool_calls = [
            {"tool": "news_search", "query_title": f"S&P 500 outlook and Fed path, week of {origin:%Y-%m-%d}"},
            {"tool": "news_search", "query_title": "mega-cap earnings reactions and forward guidance"},
        ]
    elif method == "code_agent":
        tool_calls = [
            {"tool": "code_execution", "query_title": "20-day realized volatility on the price panel"},
            {"tool": "code_execution", "query_title": "empirical 21-day forward-return distribution"},
        ]
    else:
        tool_calls = []
    return {"tool_calls": tool_calls}


def build_prediction(
    rng: random.Random,
    *,
    method: str,
    model: str | None,
    origin: date,
) -> dict[str, object]:
    """Build one prediction record (all horizons) for a method/model at an origin.

    Parameters
    ----------
    rng : random.Random
        Seeded RNG for reproducible jitter.
    method : str
        Method identifier.
    model : str or None
        Model identifier (``None`` for conventional methods).
    origin : date
        Forecast origin date.

    Returns
    -------
    dict[str, object]
        A prediction record conforming to ``prediction.schema.json``.
    """
    skill = METHOD_SKILL[method] * (MODEL_SKILL[model] if model is not None else 1.0)
    is_llm = method in LLM_METHODS

    horizon_records: list[dict[str, object]] = []
    for h in HORIZONS:
        sigma = BASE_SIGMA[h] * skill * (1.0 + rng.uniform(-0.05, 0.05))
        # naive is the last-value floor: it forecasts zero cumulative change.
        mu = 0.0 if method == "naive" else rng.uniform(-0.25, 0.35) * BASE_SIGMA[h]
        quantiles = [{"quantile": q, "value": round(NormalDist(mu, sigma).inv_cdf(q), 6)} for q in STANDARD_QUANTILES]
        horizon_records.append({"horizon": h, "point_estimate": round(mu, 6), "quantiles": quantiles})

    close_dt = origin_close_dt(origin)
    submit_dt = close_dt + timedelta(minutes=15 + rng.randint(0, 20))
    direction = rng.choice(["modestly higher", "roughly flat", "slightly lower"])

    predictor_id = f"sp500_{method}" if model is None else f"sp500_{method}__{model}"
    return {
        "schema_version": SCHEMA_VERSION,
        "origin_date": origin.isoformat(),
        "origin_timestamp": iso_z(close_dt),
        "submission_timestamp": iso_z(submit_dt),
        "method": method,
        "model": model,
        "predictor_id": predictor_id,
        "horizons": horizon_records,
        "rationale": rationale_text(method, origin, direction) if is_llm else "",
        "curated_trace_summary": curated_trace(method, origin),
        "langfuse_trace_id": (f"{rng.getrandbits(128):032x}" if is_llm else None),
    }


def build_resolution(
    *,
    method: str,
    model: str | None,
    origin: date,
    horizon: int,
    forecast_date: date,
    prediction: dict[str, object],
    realized: float,
) -> dict[str, object]:
    """Build one resolution record for a resolved (method, model, horizon, origin).

    Parameters
    ----------
    method : str
        Method identifier.
    model : str or None
        Model identifier.
    origin : date
        Forecast origin date.
    horizon : int
        Resolved horizon.
    forecast_date : date
        The business day the horizon resolved on.
    prediction : dict[str, object]
        The originating prediction record (for mu/sigma reconstruction).
    realized : float
        Realized cumulative log return at this horizon.

    Returns
    -------
    dict[str, object]
        A resolution record conforming to ``resolution.schema.json``.
    """
    horizons = prediction["horizons"]
    assert isinstance(horizons, list)
    hz = next(h for h in horizons if h["horizon"] == horizon)
    quantiles = {q["quantile"]: q["value"] for q in hz["quantiles"]}
    mu = float(hz["point_estimate"])
    # Recover sigma from the symmetric 0.05/0.95 span of the normal grid.
    sigma = (quantiles[0.95] - quantiles[0.05]) / (NormalDist().inv_cdf(0.95) - NormalDist().inv_cdf(0.05))
    crps = crps_normal(mu, sigma, realized)
    resolved_dt = origin_close_dt(forecast_date) + timedelta(hours=2)
    predictor_id = f"sp500_{method}" if model is None else f"sp500_{method}__{model}"
    return {
        "schema_version": SCHEMA_VERSION,
        "origin_date": origin.isoformat(),
        "method": method,
        "model": model,
        "predictor_id": predictor_id,
        "horizon": horizon,
        "forecast_date": forecast_date.isoformat(),
        "realized_value": round(realized, 6),
        "crps": round(crps, 6),
        "resolved_at": iso_z(resolved_dt),
    }


def method_model_pairs() -> list[tuple[str, str | None]]:
    """Enumerate every (method, model) combination in the mock model matrix.

    Returns
    -------
    list[tuple[str, str or None]]
        Conventional methods paired with ``None``; LLM methods paired with each model.
    """
    pairs: list[tuple[str, str | None]] = [(m, None) for m in CONVENTIONAL_METHODS]
    for method in LLM_METHODS:
        for model in LLM_MODELS:
            pairs.append((method, model))
    return pairs


def write_json(path: Path, payload: object) -> None:
    """Write ``payload`` to ``path`` as pretty-printed JSON with a trailing newline.

    Parameters
    ----------
    path : Path
        Destination file path.
    payload : object
        JSON-serializable content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


GENERATED_AT = "2026-07-15T02:30:00Z"  # fixed timestamp for reproducible fixtures.


def realized_cumret(
    origin: date, horizon: int, index_of: dict[date, int], daily_returns: dict[date, float], calendar: list[date]
) -> float | None:
    """Return the cumulative log return from ``origin`` over ``horizon`` days.

    A horizon has resolved only once its forecast date is on or before "today"
    (:data:`LAST_ORIGIN`); recent origins therefore have open long horizons.

    Parameters
    ----------
    origin : date
        Forecast origin date.
    horizon : int
        Business-day horizon.
    index_of : dict[date, int]
        Map from calendar date to its index.
    daily_returns : dict[date, float]
        Simulated daily log returns keyed by date.
    calendar : list[date]
        Ordered business-day calendar.

    Returns
    -------
    float or None
        Cumulative log return, or ``None`` when the horizon has not resolved.
    """
    start = index_of[origin]
    if start + horizon > index_of[LAST_ORIGIN]:
        return None
    return sum(daily_returns[calendar[start + k]] for k in range(1, horizon + 1))


def write_bundles(
    rng: random.Random,
    active_origins: list[date],
    index_of: dict[date, int],
    daily_returns: dict[date, float],
    calendar: list[date],
) -> tuple[dict[tuple[str, str | None, int], list[tuple[date, float]]], dict[tuple[str, str | None, int], list[bool]]]:
    """Write every per-origin forecast bundle and accumulate scoring stats.

    Parameters
    ----------
    rng : random.Random
        Seeded RNG.
    active_origins : list[date]
        Origins that submitted (gaps excluded).
    index_of : dict[date, int]
        Calendar index map.
    daily_returns : dict[date, float]
        Simulated daily log returns.
    calendar : list[date]
        Business-day calendar.

    Returns
    -------
    tuple
        ``(crps_records, coverage_hits)`` keyed by ``(method, model, horizon)``.
    """
    pairs = method_model_pairs()
    crps_records: dict[tuple[str, str | None, int], list[tuple[date, float]]] = {}
    coverage_hits: dict[tuple[str, str | None, int], list[bool]] = {}

    for origin in active_origins:
        realized_by_h = {h: realized_cumret(origin, h, index_of, daily_returns, calendar) for h in HORIZONS}
        predictions: list[dict[str, object]] = []
        resolutions: list[dict[str, object]] = []

        for method, model in pairs:
            prediction = build_prediction(rng, method=method, model=model, origin=origin)
            predictions.append(prediction)
            for h in HORIZONS:
                realized = realized_by_h[h]
                if realized is None:
                    continue
                forecast_date = calendar[index_of[origin] + h]
                resolution = build_resolution(
                    method=method,
                    model=model,
                    origin=origin,
                    horizon=h,
                    forecast_date=forecast_date,
                    prediction=prediction,
                    realized=realized,
                )
                resolutions.append(resolution)
                key = (method, model, h)
                crps_records.setdefault(key, []).append((origin, float(resolution["crps"])))
                horizons = prediction["horizons"]
                assert isinstance(horizons, list)
                hz = next(x for x in horizons if x["horizon"] == h)
                qmap = {q["quantile"]: q["value"] for q in hz["quantiles"]}
                coverage_hits.setdefault(key, []).append(qmap[0.05] <= realized <= qmap[0.95])

        realized_summary = [
            {
                "horizon": h,
                "forecast_date": calendar[index_of[origin] + h].isoformat(),
                "realized_value": round(value, 6),
            }
            for h, value in realized_by_h.items()
            if value is not None
        ]
        bundle = {
            "schema_version": SCHEMA_VERSION,
            "generated_by": "mock",
            "generated_at": GENERATED_AT,
            "origin_date": origin.isoformat(),
            "realized": realized_summary,
            "predictions": predictions,
            "resolutions": resolutions,
        }
        write_json(DATA_DIR / "forecasts" / f"{origin.isoformat()}.json", bundle)

    return crps_records, coverage_hits


def leaderboard_cell(
    key: tuple[str, str | None, int],
    series: list[tuple[date, float]],
    coverage_hits: dict[tuple[str, str | None, int], list[bool]],
) -> tuple[dict[str, object], dict[str, object]]:
    """Build one leaderboard cell and its cumulative series from resolved scores.

    Parameters
    ----------
    key : tuple
        ``(method, model, horizon)``.
    series : list[tuple[date, float]]
        ``(origin, crps)`` pairs for this cell.
    coverage_hits : dict
        Per-key list of 90%-interval hit booleans.

    Returns
    -------
    tuple[dict, dict]
        The cell record and the cumulative-series record.
    """
    method, model, horizon = key
    series_sorted = sorted(series, key=lambda t: t[0])
    scores = [s for _, s in series_sorted]
    hits = coverage_hits[key]
    cell = {
        "method": method,
        "model": model,
        "horizon": horizon,
        "mean_crps": round(sum(scores) / len(scores), 6),
        "n": len(scores),
        "coverage_90": round(sum(1 for hit in hits if hit) / len(hits), 4),
        "last_updated": max(o for o, _ in series_sorted).isoformat(),
    }
    running_sum = 0.0
    points: list[dict[str, object]] = []
    for i, (origin_i, score) in enumerate(series_sorted, start=1):
        running_sum += score
        points.append({"origin_date": origin_i.isoformat(), "cumulative_mean_crps": round(running_sum / i, 6), "n": i})
    return cell, {"method": method, "model": model, "horizon": horizon, "series": points}


def write_leaderboard(
    crps_records: dict[tuple[str, str | None, int], list[tuple[date, float]]],
    coverage_hits: dict[tuple[str, str | None, int], list[bool]],
) -> None:
    """Write the leaderboard aggregate from accumulated scores.

    Parameters
    ----------
    crps_records : dict
        Per-key ``(origin, crps)`` lists.
    coverage_hits : dict
        Per-key 90%-interval hit booleans.
    """
    cells: list[dict[str, object]] = []
    cumulative: list[dict[str, object]] = []
    for key, series in sorted(crps_records.items(), key=lambda kv: (kv[0][0], kv[0][1] or "", kv[0][2])):
        cell, cum = leaderboard_cell(key, series, coverage_hits)
        cells.append(cell)
        cumulative.append(cum)
    write_json(
        DATA_DIR / "leaderboard.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generated_by": "mock",
            "generated_at": GENERATED_AT,
            "horizons": HORIZONS,
            "cells": cells,
            "cumulative": cumulative,
        },
    )


def write_gap_log() -> None:
    """Write the gap-log fixture (two logged, never-backfilled misses)."""
    write_json(
        DATA_DIR / "gaps.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generated_by": "mock",
            "generated_at": GENERATED_AT,
            "gaps": [
                {
                    "schema_version": SCHEMA_VERSION,
                    "date": "2026-06-05",
                    "scope": "all_methods",
                    "reason": "Proxy API returned 503 for all model IDs; three same-evening retries exhausted.",
                    "retries_attempted": 3,
                    "logged_at": "2026-06-05T20:52:00Z",
                },
                {
                    "schema_version": SCHEMA_VERSION,
                    "date": "2026-06-24",
                    "scope": "agents",
                    "reason": "Code-execution sandbox provisioning timed out; conventional + LLMP submitted on time.",
                    "retries_attempted": 2,
                    "logged_at": "2026-06-24T20:41:00Z",
                },
            ],
        },
    )


def write_mutations() -> None:
    """Write strategy-mutation fixtures (twins view is stubbed; schema exercised)."""
    write_json(
        DATA_DIR / "mutations.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generated_by": "mock",
            "generated_at": GENERATED_AT,
            "mutations": [
                {
                    "schema_version": SCHEMA_VERSION,
                    "event_id": "mut-2026-06-30-001",
                    "twin_id": "twin_learning",
                    "occurred_at": "2026-06-30T21:10:00Z",
                    "origin_date": "2026-06-30",
                    "tier": "observation",
                    "gate_outcome": "appended",
                    "version": "strategy-v3",
                    "rationale": "Logged that the h=5 miss on 2026-06-18 coincided with an unpriced CPI surprise.",
                },
                {
                    "schema_version": SCHEMA_VERSION,
                    "event_id": "mut-2026-07-07-002",
                    "twin_id": "twin_learning",
                    "occurred_at": "2026-07-07T21:05:00Z",
                    "origin_date": "2026-07-07",
                    "tier": "hypothesis",
                    "gate_outcome": "confirmed",
                    "version": "strategy-v4",
                    "rationale": "Third confirmation that widening the h=21 lower tail in low-VIX regimes improves CRPS.",
                    "confirmations": 3,
                },
                {
                    "schema_version": SCHEMA_VERSION,
                    "event_id": "mut-2026-07-10-003",
                    "twin_id": "twin_learning",
                    "occurred_at": "2026-07-10T21:12:00Z",
                    "origin_date": "2026-07-10",
                    "tier": "behavioral",
                    "gate_outcome": "shadowing",
                    "version": "strategy-v5-candidate",
                    "rationale": "Calibration correction (+8% band width at h=21) entered a forward shadow gate for M=5 origins.",
                },
            ],
        },
    )


def write_manifest(
    active_origins: list[date],
    index_of: dict[date, int],
    daily_returns: dict[date, float],
    calendar: list[date],
) -> None:
    """Write the site manifest (index, freshness, mock flag, origin list).

    Parameters
    ----------
    active_origins : list[date]
        Origins that submitted.
    index_of : dict[date, int]
        Calendar index map.
    daily_returns : dict[date, float]
        Simulated daily log returns.
    calendar : list[date]
        Business-day calendar.
    """
    write_json(
        DATA_DIR / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generated_by": "mock",
            "generated_at": GENERATED_AT,
            "title": "S&P 500 live forecasting experiment",
            "horizons": HORIZONS,
            "quantiles": STANDARD_QUANTILES,
            "methods": CONVENTIONAL_METHODS + LLM_METHODS,
            "models": LLM_MODELS,
            "latest_origin": max(active_origins).isoformat(),
            "origin_count": len(active_origins),
            "gap_count": len(GAP_DATES),
            "origins": [
                {
                    "origin_date": d.isoformat(),
                    "resolved_horizons": [
                        h for h in HORIZONS if realized_cumret(d, h, index_of, daily_returns, calendar) is not None
                    ],
                }
                for d in active_origins
            ],
        },
    )


def main() -> None:
    """Generate the full fixture set under ``monitor/site/data/``."""
    rng = random.Random(20260714)
    origins = business_days_back(LAST_ORIGIN, N_ORIGINS)
    calendar = origins + future_business_days(LAST_ORIGIN, 30)
    index_of = {d: i for i, d in enumerate(calendar)}
    daily_returns = {d: rng.gauss(0.0002, DAILY_VOL) for d in calendar}
    active_origins = [d for d in origins if d not in GAP_DATES]

    crps_records, coverage_hits = write_bundles(rng, active_origins, index_of, daily_returns, calendar)
    write_leaderboard(crps_records, coverage_hits)
    write_gap_log()
    write_mutations()
    write_manifest(active_origins, index_of, daily_returns, calendar)

    print(f"Wrote fixtures for {len(active_origins)} origins to {DATA_DIR}")


if __name__ == "__main__":
    main()
