"""The append-only per-day log: the tamper-evident source of truth.

Layout (per ``monitor/data-contract.md``), rooted at ``log_dir``::

    log/YYYY/MM/DD/predictions/<predictor_id>.json
    log/YYYY/MM/DD/resolutions/<predictor_id>-h<H>.json
    log/YYYY/MM/DD/gap.json                 # one gap-log array per day/scope

One record per file so commits stay small and diffs auditable. Records are
written once and never revised; resolutions are appended as horizons mature.
Readers here collate the whole log for the aggregate step.

All JSON is written with sorted keys and a trailing newline so re-running the
harness over an unchanged log produces byte-identical files.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any


def _dump(path: Path, obj: Any) -> None:
    """Write *obj* as deterministic JSON (sorted keys, trailing newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _load(path: Path) -> Any:
    """Parse a JSON file."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def day_dir(log_dir: Path, origin: date) -> Path:
    """Return the ``YYYY/MM/DD`` directory for an origin under *log_dir*."""
    return log_dir / f"{origin.year:04d}" / f"{origin.month:02d}" / f"{origin.day:02d}"


def prediction_path(log_dir: Path, origin: date, predictor_id: str) -> Path:
    """Return the prediction-record path for a rung at an origin."""
    return day_dir(log_dir, origin) / "predictions" / f"{predictor_id}.json"


def resolution_path(log_dir: Path, origin: date, predictor_id: str, horizon: int) -> Path:
    """Return the resolution-record path for a rung x horizon at an origin."""
    return day_dir(log_dir, origin) / "resolutions" / f"{predictor_id}-h{horizon}.json"


def gap_path(log_dir: Path, origin: date) -> Path:
    """Return the per-day gap-log path for an origin."""
    return day_dir(log_dir, origin) / "gap.json"


def write_prediction(log_dir: Path, origin: date, record: dict[str, Any]) -> Path:
    """Write one prediction record; returns the path written."""
    path = prediction_path(log_dir, origin, record["predictor_id"])
    _dump(path, record)
    return path


def write_resolution(log_dir: Path, origin: date, record: dict[str, Any]) -> Path:
    """Write one resolution record; returns the path written."""
    path = resolution_path(log_dir, origin, record["predictor_id"], record["horizon"])
    _dump(path, record)
    return path


def append_gap(log_dir: Path, origin: date, entry: dict[str, Any]) -> Path:
    """Append one gap-log entry to the origin's ``gap.json`` array.

    Gaps accumulate (append-only): existing entries are preserved and the new
    entry added. Returns the gap-log path.
    """
    path = gap_path(log_dir, origin)
    entries: list[dict[str, Any]] = list(_load(path)) if path.exists() else []
    entries.append(entry)
    _dump(path, entries)
    return path


def has_prediction(log_dir: Path, origin: date, predictor_id: str) -> bool:
    """Return whether a prediction record already exists for a rung at origin."""
    return prediction_path(log_dir, origin, predictor_id).exists()


def has_resolution(log_dir: Path, origin: date, predictor_id: str, horizon: int) -> bool:
    """Return whether a resolution record already exists (rung x horizon)."""
    return resolution_path(log_dir, origin, predictor_id, horizon).exists()


def iter_prediction_records(log_dir: Path) -> list[dict[str, Any]]:
    """Read every prediction record in the log, sorted by (origin, predictor)."""
    records = [_load(p) for p in sorted(log_dir.glob("*/*/*/predictions/*.json"))]
    return sorted(records, key=lambda r: (r["origin_date"], r["predictor_id"]))


def iter_resolution_records(log_dir: Path) -> list[dict[str, Any]]:
    """Read every resolution record in the log, deterministically ordered."""
    records = [_load(p) for p in sorted(log_dir.glob("*/*/*/resolutions/*.json"))]
    return sorted(records, key=lambda r: (r["origin_date"], r["predictor_id"], r["horizon"]))


def iter_gap_entries(log_dir: Path) -> list[dict[str, Any]]:
    """Read every gap-log entry in the log, sorted by (date, scope, logged_at)."""
    entries: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*/*/*/gap.json")):
        entries.extend(_load(path))
    return sorted(entries, key=lambda e: (e["date"], e["scope"], e.get("logged_at", "")))


def origin_dates(log_dir: Path) -> list[date]:
    """Return the sorted list of distinct origins that have prediction records."""
    seen = {r["origin_date"] for r in iter_prediction_records(log_dir)}
    return sorted(date.fromisoformat(d) for d in seen)


__all__ = [
    "append_gap",
    "day_dir",
    "gap_path",
    "has_prediction",
    "has_resolution",
    "iter_gap_entries",
    "iter_prediction_records",
    "iter_resolution_records",
    "origin_dates",
    "prediction_path",
    "resolution_path",
    "write_prediction",
    "write_resolution",
]
