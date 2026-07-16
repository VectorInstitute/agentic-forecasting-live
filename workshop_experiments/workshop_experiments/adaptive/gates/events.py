"""Strategy-mutation event records — build, validate, persist, collate.

Every adoption / rejection / freeze the gate performs writes one
``mutation_event`` record (``monitor/schemas/mutation_event.schema.json``) to the
append-only log under ``<log_dir>/YYYY/MM/DD/mutations/<event_id>.json`` — the
layout the data contract reserves for twins. The monitor's twins view reads the
collated ``mutations.json`` aggregate the harness regenerates from these files.

Records are validated against the schema before they are written, using the same
monitor validator the prediction/resolution records go through.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from workshop_experiments.live import SCHEMA_VERSION
from workshop_experiments.live.log_store import day_dir
from workshop_experiments.live.schema_validation import check


Tier = Literal["observation", "hypothesis", "behavioral"]
GateOutcome = Literal[
    "appended",
    "proposed",
    "confirmed",
    "refuted",
    "graduated",
    "demoted",
    "shadowing",
    "adopted",
    "rejected",
    "frozen_circuit_breaker",
]


def utc_now_z() -> str:
    """Return the current UTC time as a schema timestamp (``...Z``, seconds)."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def mutations_dir(log_dir: Path, origin: date) -> Path:
    """Return the ``mutations/`` directory for an origin under *log_dir*."""
    return day_dir(log_dir, origin) / "mutations"


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


class MutationEventStore:
    """Append-only writer + reader for one twin's mutation events.

    Parameters
    ----------
    log_dir : Path
        Root of the live append-only log.
    twin_id : str
        The learning twin this store belongs to (the frozen twin never mutates,
        so it never owns a store).
    """

    def __init__(self, log_dir: Path, twin_id: str) -> None:
        self.log_dir = log_dir
        self.twin_id = twin_id

    def _existing_count(self) -> int:
        """Count already-written events for this twin (drives the version seq)."""
        return sum(1 for _ in self._iter_paths() if _.name.startswith(f"{self.twin_id}-"))

    def _iter_paths(self) -> list[Path]:
        return sorted(self.log_dir.glob("*/*/*/mutations/*.json"))

    def write(
        self,
        *,
        tier: Tier,
        gate_outcome: GateOutcome,
        rationale: str,
        occurred_on: date | None = None,
        origin_date: date | None = None,
        occurred_at: str | None = None,
        confirmations: int | None = None,
        version: str | None = None,
        validate: bool = True,
    ) -> dict[str, Any]:
        """Build, validate, and persist one mutation event; return the record.

        ``occurred_on`` selects the log day the event file lands under (defaults
        to today UTC). ``version`` defaults to a monotonic per-twin ``v{seq}``.
        """
        seq = self._existing_count() + 1
        now = datetime.now(tz=timezone.utc)
        occurred_on = occurred_on or now.date()
        # Tie the timestamp's DATE to the logical event day, keeping the real
        # time-of-day. In production occurred_on defaults to today (so this is the
        # true wall-clock time); in tests an explicit occurred_on makes the
        # timestamp deterministic, which the weekly rate-limit reads back.
        if occurred_at is None:
            occurred_at = datetime(
                occurred_on.year, occurred_on.month, occurred_on.day, now.hour, now.minute, now.second
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
        event_id = f"{self.twin_id}-{seq:04d}"
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "twin_id": self.twin_id,
            "occurred_at": occurred_at,
            "tier": tier,
            "gate_outcome": gate_outcome,
            "version": version or f"{self.twin_id}-v{seq}",
            "rationale": rationale,
        }
        if origin_date is not None:
            record["origin_date"] = origin_date.isoformat()
        if confirmations is not None:
            record["confirmations"] = int(confirmations)

        if validate:
            check("mutation_event", record)
        _dump(mutations_dir(self.log_dir, occurred_on) / f"{event_id}.json", record)
        return record

    def read_all(self) -> list[dict[str, Any]]:
        """Read every mutation event in the log for this twin, event-id ordered."""
        records: list[dict[str, Any]] = []
        for path in self._iter_paths():
            with path.open(encoding="utf-8") as handle:
                record = json.load(handle)
            if record.get("twin_id") == self.twin_id:
                records.append(record)
        return sorted(records, key=lambda r: r["event_id"])


def iter_mutation_events(log_dir: Path) -> list[dict[str, Any]]:
    """Read every mutation event in the log (all twins), deterministically ordered."""
    records: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*/*/*/mutations/*.json")):
        with path.open(encoding="utf-8") as handle:
            records.append(json.load(handle))
    return sorted(records, key=lambda r: (r.get("occurred_at", ""), r["event_id"]))


__all__ = [
    "GateOutcome",
    "MutationEventStore",
    "Tier",
    "iter_mutation_events",
    "mutations_dir",
    "utc_now_z",
]
