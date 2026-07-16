"""Tier-3 shadow champion/challenger gate.

A behavioral change (a graduated calibration correction, an approach-narrative
rewrite) does not alter the live strategy immediately. Instead the candidate runs
as a **shadow challenger** alongside the current champion for ``M`` trading days.
Its predictions are persisted under a *shadow* directory that the monitor
aggregate deliberately never globs (the aggregate reads ``log/*/*/*/predictions``;
the shadow store lives outside that tree), so shadow forecasts stay out of the
public leaderboard. Each resolved origin contributes a ``(champion_crps,
challenger_crps)`` pair. After ``M`` pairs the gate decides:

- **adopt** the candidate iff its mean CRPS ≤ the champion's over the window;
- otherwise **reject** it and archive the comparison stats.

Only one candidate is ever open at a time (the §4 "bounded to one candidate"
rule). State is a small JSON file so the lifecycle survives restarts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from statistics import fmean
from typing import Any


class ShadowError(RuntimeError):
    """Raised on an invalid shadow-gate transition (e.g. two open candidates)."""


@dataclass
class ShadowPair:
    """One resolved origin's champion-vs-challenger CRPS pair."""

    origin_date: str
    champion_crps: float
    challenger_crps: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise the pair to a plain dict."""
        return {
            "origin_date": self.origin_date,
            "champion_crps": self.champion_crps,
            "challenger_crps": self.challenger_crps,
        }


@dataclass
class ShadowCandidate:
    """A behavioral candidate under shadow evaluation."""

    candidate_id: str
    kind: str  # "calibration_correction" | "approach_narrative"
    description: str
    opened_on: str
    status: str = "open"  # "open" | "adopted" | "rejected"
    decided_on: str | None = None
    pairs: list[ShadowPair] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the candidate (and its pairs) to a plain dict."""
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "description": self.description,
            "opened_on": self.opened_on,
            "status": self.status,
            "decided_on": self.decided_on,
            "pairs": [p.to_dict() for p in self.pairs],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ShadowCandidate:
        """Reconstruct a candidate from its serialised dict."""
        return cls(
            candidate_id=raw["candidate_id"],
            kind=raw["kind"],
            description=raw["description"],
            opened_on=raw["opened_on"],
            status=raw.get("status", "open"),
            decided_on=raw.get("decided_on"),
            pairs=[ShadowPair(**p) for p in raw.get("pairs", [])],
        )


@dataclass(frozen=True)
class ShadowDecision:
    """The outcome of evaluating a completed shadow window."""

    candidate_id: str
    adopted: bool
    champion_mean_crps: float
    challenger_mean_crps: float
    n: int

    @property
    def summary(self) -> str:
        """One-line summary of the decision for a mutation-event rationale."""
        verb = "adopted" if self.adopted else "rejected"
        return (
            f"shadow {verb}: challenger mean CRPS {self.challenger_mean_crps:.5f} vs "
            f"champion {self.champion_mean_crps:.5f} over {self.n} origins"
        )


class ShadowGate:
    """One-candidate-at-a-time shadow champion/challenger lifecycle.

    Parameters
    ----------
    shadow_dir : Path
        Directory holding ``active.json`` (the open candidate) and
        ``archive/<candidate_id>.json`` (decided candidates). Created on demand.
    window : int
        ``M`` — resolved origins required before a decision.
    """

    _ACTIVE = "active.json"
    _ARCHIVE = "archive"

    def __init__(self, shadow_dir: Path, *, window: int) -> None:
        if window <= 0:
            raise ValueError("shadow window must be positive")
        self.shadow_dir = shadow_dir
        self.window = window

    # ---- persistence ----------------------------------------------------
    @property
    def _active_path(self) -> Path:
        return self.shadow_dir / self._ACTIVE

    def _archive_path(self, candidate_id: str) -> Path:
        return self.shadow_dir / self._ARCHIVE / f"{candidate_id}.json"

    def _save_active(self, candidate: ShadowCandidate) -> None:
        self.shadow_dir.mkdir(parents=True, exist_ok=True)
        self._active_path.write_text(json.dumps(candidate.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _archive(self, candidate: ShadowCandidate) -> None:
        path = self._archive_path(candidate.candidate_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(candidate.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._active_path.unlink(missing_ok=True)

    # ---- state ----------------------------------------------------------
    def active(self) -> ShadowCandidate | None:
        """Return the open candidate, or ``None`` if the gate is idle."""
        if not self._active_path.exists():
            return None
        return ShadowCandidate.from_dict(json.loads(self._active_path.read_text(encoding="utf-8")))

    def has_open_candidate(self) -> bool:
        """Return whether a shadow candidate is currently open."""
        return self.active() is not None

    # ---- lifecycle ------------------------------------------------------
    def open_challenger(self, *, candidate_id: str, kind: str, description: str, opened_on: date) -> ShadowCandidate:
        """Open a new shadow candidate; raise if one is already open."""
        if self.has_open_candidate():
            raise ShadowError("a shadow candidate is already open (one at a time)")
        candidate = ShadowCandidate(
            candidate_id=candidate_id,
            kind=kind,
            description=description,
            opened_on=opened_on.isoformat(),
        )
        self._save_active(candidate)
        return candidate

    def record_pair(self, *, origin: date, champion_crps: float, challenger_crps: float) -> ShadowCandidate:
        """Append one resolved origin's champion/challenger CRPS pair."""
        candidate = self.active()
        if candidate is None:
            raise ShadowError("no open shadow candidate to record a pair against")
        candidate.pairs.append(ShadowPair(origin.isoformat(), float(champion_crps), float(challenger_crps)))
        self._save_active(candidate)
        return candidate

    def ready(self) -> bool:
        """Return whether the open candidate has completed its window."""
        candidate = self.active()
        return candidate is not None and len(candidate.pairs) >= self.window

    def evaluate(self, *, decided_on: date) -> ShadowDecision:
        """Decide the open candidate once its window is complete; archive it.

        Adopts iff the challenger's mean CRPS is ≤ the champion's over the
        window. Either way the candidate is archived (with its stats) and the
        active slot is cleared.
        """
        candidate = self.active()
        if candidate is None:
            raise ShadowError("no open shadow candidate to evaluate")
        if len(candidate.pairs) < self.window:
            raise ShadowError(f"shadow window incomplete: {len(candidate.pairs)}/{self.window} pairs recorded")
        champ = fmean(p.champion_crps for p in candidate.pairs)
        chall = fmean(p.challenger_crps for p in candidate.pairs)
        adopted = chall <= champ
        candidate.status = "adopted" if adopted else "rejected"
        candidate.decided_on = decided_on.isoformat()
        self._archive(candidate)
        return ShadowDecision(
            candidate_id=candidate.candidate_id,
            adopted=adopted,
            champion_mean_crps=champ,
            challenger_mean_crps=chall,
            n=len(candidate.pairs),
        )


__all__ = [
    "ShadowCandidate",
    "ShadowDecision",
    "ShadowError",
    "ShadowGate",
    "ShadowPair",
]
