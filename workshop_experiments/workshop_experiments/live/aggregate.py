"""The aggregate step: regenerate the monitor's site-facing aggregates.

Collates the entire append-only log into the small JSONs the static monitor
fetches — ``manifest.json``, ``leaderboard.json``, ``gaps.json``,
``mutations.json``, and one ``forecasts/<origin>.json`` bundle per origin — all
carrying ``generated_by: "harness"``. Everything is validated against the
schemas before writing.

Deterministic: the same log produces byte-identical aggregates. Arrays are
sorted by stable keys and ``generated_at`` is derived from the log's own latest
timestamp (never the wall clock), so re-running over an unchanged log is a
no-op on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any

from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES

from workshop_experiments.live import SCHEMA_VERSION
from workshop_experiments.live.log_store import (
    iter_gap_entries,
    iter_prediction_records,
    iter_resolution_records,
)
from workshop_experiments.live.schema_validation import check


#: Canonical method-enum order the monitor renders rows in.
_METHOD_ORDER = [
    "naive",
    "ets",
    "kalman",
    "autoarima",
    "lightgbm",
    "lightgbm_cov",
    "llm_process",
    "llm_process_cov",
    "agent_news",
    "agent_code",
    "adaptive_frozen",
    "adaptive_learning",
]


def _method_rank(method: str) -> int:
    """Return the fixed render rank of a method enum value."""
    return _METHOD_ORDER.index(method) if method in _METHOD_ORDER else len(_METHOD_ORDER)


def _dump_json(path: Path, obj: Any) -> None:
    """Write *obj* as deterministic JSON (sorted keys, trailing newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _generated_at(predictions: list[dict[str, Any]], resolutions: list[dict[str, Any]]) -> str:
    """Return the latest timestamp in the log (deterministic ``generated_at``)."""
    stamps = [p["submission_timestamp"] for p in predictions]
    stamps += [r["resolved_at"] for r in resolutions]
    return max(stamps)


def _cell_key(record: dict[str, Any], horizon: int) -> tuple[str, str | None, int]:
    """Return the ``(method, model, horizon)`` leaderboard cell key."""
    return record["method"], record["model"], horizon


def _grid_lookup(predictions: list[dict[str, Any]]) -> dict[tuple[str, str, int], list[dict[str, float]]]:
    """Map ``(origin_date, predictor_id, horizon) -> quantile grid``."""
    lookup: dict[tuple[str, str, int], list[dict[str, float]]] = {}
    for record in predictions:
        for horizon_forecast in record["horizons"]:
            key = (record["origin_date"], record["predictor_id"], int(horizon_forecast["horizon"]))
            lookup[key] = horizon_forecast["quantiles"]
    return lookup


def _coverage_90(
    resolutions: list[dict[str, Any]], grids: dict[tuple[str, str, int], list[dict[str, float]]]
) -> float | None:
    """Empirical coverage of the nominal-90% (0.05-0.95) interval, or ``None``."""
    hits = 0
    total = 0
    for res in resolutions:
        grid = grids.get((res["origin_date"], res["predictor_id"], res["horizon"]))
        if grid is None:
            continue
        values = {round(float(p["quantile"]), 2): float(p["value"]) for p in grid}
        low, high = values.get(0.05), values.get(0.95)
        if low is None or high is None:
            continue
        total += 1
        if low <= res["realized_value"] <= high:
            hits += 1
    return hits / total if total else None


def _build_leaderboard(
    predictions: list[dict[str, Any]],
    resolutions: list[dict[str, Any]],
    horizons: list[int],
    generated_at: str,
) -> dict[str, Any]:
    """Build the leaderboard aggregate (cells + cumulative trend series)."""
    grids = _grid_lookup(predictions)
    by_cell: dict[tuple[str, str | None, int], list[dict[str, Any]]] = {}
    for res in resolutions:
        by_cell.setdefault(_cell_key(res, res["horizon"]), []).append(res)

    # Naive floor per horizon, for the optional skill score.
    naive_mean_crps: dict[int, float] = {}
    for (method, _model, horizon), group in by_cell.items():
        if method == "naive":
            naive_mean_crps[horizon] = fmean(r["crps"] for r in group)

    cells: list[dict[str, Any]] = []
    cumulative: list[dict[str, Any]] = []
    for key in sorted(by_cell, key=lambda k: (_method_rank(k[0]), k[1] or "", k[2])):
        method, model, horizon = key
        group = sorted(by_cell[key], key=lambda r: r["origin_date"])
        mean_crps = fmean(r["crps"] for r in group)
        cell: dict[str, Any] = {
            "method": method,
            "model": model,
            "horizon": horizon,
            "mean_crps": mean_crps,
            "n": len(group),
            "last_updated": max(r["origin_date"] for r in group),
        }
        coverage = _coverage_90(group, grids)
        if coverage is not None:
            cell["coverage_90"] = coverage
        floor = naive_mean_crps.get(horizon)
        if floor is not None and floor > 0 and method != "naive":
            cell["skill_vs_naive"] = 1.0 - mean_crps / floor
        cells.append(cell)

        running: list[float] = []
        series: list[dict[str, Any]] = []
        for res in group:
            running.append(res["crps"])
            series.append(
                {
                    "origin_date": res["origin_date"],
                    "cumulative_mean_crps": fmean(running),
                    "n": len(running),
                }
            )
        cumulative.append({"method": method, "model": model, "horizon": horizon, "series": series})

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "harness",
        "generated_at": generated_at,
        "horizons": horizons,
        "cells": cells,
        "cumulative": cumulative,
    }


def _build_manifest(
    predictions: list[dict[str, Any]],
    resolutions: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    horizons: list[int],
    generated_at: str,
) -> dict[str, Any]:
    """Build the site manifest (freshness, universe, drill-down origins)."""
    origins = sorted({p["origin_date"] for p in predictions})
    resolved_by_origin: dict[str, set[int]] = {}
    for res in resolutions:
        resolved_by_origin.setdefault(res["origin_date"], set()).add(res["horizon"])

    methods = sorted({p["method"] for p in predictions}, key=_method_rank)
    models = sorted({p["model"] for p in predictions if p["model"] is not None})
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "harness",
        "generated_at": generated_at,
        "title": "S&P 500 live forecasting monitor",
        "horizons": horizons,
        "quantiles": list(STANDARD_QUANTILES),
        "methods": methods,
        "models": models,
        "latest_origin": max(origins),
        "origin_count": len(origins),
        "gap_count": len(gaps),
        "origins": [
            {"origin_date": origin, "resolved_horizons": sorted(resolved_by_origin.get(origin, set()))}
            for origin in origins
        ],
    }


def _build_bundle(
    origin: str,
    predictions: list[dict[str, Any]],
    resolutions: list[dict[str, Any]],
    generated_at: str,
) -> dict[str, Any]:
    """Build the per-origin drill-down bundle for one origin date."""
    preds = sorted((p for p in predictions if p["origin_date"] == origin), key=lambda p: p["predictor_id"])
    resos = sorted(
        (r for r in resolutions if r["origin_date"] == origin),
        key=lambda r: (r["predictor_id"], r["horizon"]),
    )
    realized: dict[int, dict[str, Any]] = {}
    for res in resos:
        realized.setdefault(
            res["horizon"],
            {
                "horizon": res["horizon"],
                "forecast_date": res["forecast_date"],
                "realized_value": res["realized_value"],
            },
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "harness",
        "generated_at": generated_at,
        "origin_date": origin,
        "predictions": preds,
        "resolutions": resos,
        "realized": [realized[h] for h in sorted(realized)],
    }


def aggregate_step(log_dir: Path, out_dir: Path, *, validate: bool = True) -> dict[str, Path]:
    """Regenerate every monitor aggregate from the log into *out_dir*.

    Parameters
    ----------
    log_dir : Path
        Root of the append-only log.
    out_dir : Path
        Destination for the aggregates (the monitor's ``site/data`` in prod, a
        temp dir in tests).
    validate : bool
        Validate every aggregate/record against its schema before writing.

    Returns
    -------
    dict[str, Path]
        Map of aggregate name -> path written. Empty if the log has no
        predictions yet (nothing to aggregate).

    Raises
    ------
    ValueError
        If ``validate`` and any produced payload violates its schema.
    """
    predictions = iter_prediction_records(log_dir)
    if not predictions:
        return {}
    resolutions = iter_resolution_records(log_dir)
    gaps = iter_gap_entries(log_dir)
    horizons = sorted({int(hf["horizon"]) for p in predictions for hf in p["horizons"]})
    generated_at = _generated_at(predictions, resolutions)

    manifest = _build_manifest(predictions, resolutions, gaps, horizons, generated_at)
    leaderboard = _build_leaderboard(predictions, resolutions, horizons, generated_at)
    gaps_doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "harness",
        "generated_at": generated_at,
        "gaps": gaps,
    }
    mutations_doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": "harness",
        "generated_at": generated_at,
        "mutations": [],
    }

    if validate:
        check("manifest", manifest)
        check("leaderboard", leaderboard)
        for gap in gaps:
            check("gap_log", gap)

    written: dict[str, Path] = {}
    _dump_json(out_dir / "manifest.json", manifest)
    written["manifest"] = out_dir / "manifest.json"
    _dump_json(out_dir / "leaderboard.json", leaderboard)
    written["leaderboard"] = out_dir / "leaderboard.json"
    _dump_json(out_dir / "gaps.json", gaps_doc)
    written["gaps"] = out_dir / "gaps.json"
    _dump_json(out_dir / "mutations.json", mutations_doc)
    written["mutations"] = out_dir / "mutations.json"

    for origin in sorted({p["origin_date"] for p in predictions}):
        bundle = _build_bundle(origin, predictions, resolutions, generated_at)
        if validate:
            check("forecast_bundle", bundle)
        path = out_dir / "forecasts" / f"{origin}.json"
        _dump_json(path, bundle)
        written[f"forecasts/{origin}"] = path

    return written


__all__ = ["aggregate_step"]
