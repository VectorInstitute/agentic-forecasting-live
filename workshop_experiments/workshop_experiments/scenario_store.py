"""Persistence layout for Track-2 scenario write-ups and judge verdicts.

Layout::

    data/scenarios/<origin>/writeup.md   — the agent's free-text scenario write-up
    data/scenarios/<origin>/meta.yaml    — model, agent name, trace id, timestamps
    data/scenarios/<origin>/judge.yaml   — the judge's rubric verdict, written by
                                            ws-scenario-judge

One directory per forecast origin (``YYYY-MM-DD``), directly under
:data:`DEFAULT_SCENARIO_STORE_DIR` — mirrors the ``data/predictions/<spec>/...``
convention in :mod:`workshop_experiments.runner`, minus the spec/predictor
nesting: Track 2 has one scenario agent identity, not a predictor grid, and one
origin's write-up is graded by exactly one judge pass (re-judging overwrites in
place, same as re-running a scenario does).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


#: Default store, bundled with the package (mirrors DEFAULT_STORE_DIR in runner.py).
DEFAULT_SCENARIO_STORE_DIR = Path(__file__).resolve().parent / "data" / "scenarios"

_WRITEUP_FILENAME = "writeup.md"
_META_FILENAME = "meta.yaml"
_JUDGE_FILENAME = "judge.yaml"


def scenario_dir(origin: date, store_dir: Path = DEFAULT_SCENARIO_STORE_DIR) -> Path:
    """Return the per-origin directory for a scenario write-up."""
    return store_dir / origin.isoformat()


def has_writeup(origin: date, store_dir: Path = DEFAULT_SCENARIO_STORE_DIR) -> bool:
    """Whether a scenario write-up is already persisted for ``origin`` (resume gate)."""
    return (scenario_dir(origin, store_dir) / _WRITEUP_FILENAME).exists()


@dataclass(frozen=True)
class ScenarioWriteup:
    """A persisted (or about-to-be-persisted) Track-2 scenario write-up."""

    origin: date
    markdown: str
    meta: dict[str, Any]


def write_scenario_writeup(writeup: ScenarioWriteup, store_dir: Path = DEFAULT_SCENARIO_STORE_DIR) -> Path:
    """Persist ``writeup`` as ``writeup.md`` + ``meta.yaml``. Returns the directory."""
    target_dir = scenario_dir(writeup.origin, store_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / _WRITEUP_FILENAME).write_text(writeup.markdown, encoding="utf-8")
    with (target_dir / _META_FILENAME).open("w") as f:
        yaml.safe_dump(writeup.meta, f, default_flow_style=False, sort_keys=False)
    return target_dir


def load_scenario_writeup(origin: date, store_dir: Path = DEFAULT_SCENARIO_STORE_DIR) -> ScenarioWriteup | None:
    """Load a persisted write-up, or ``None`` if absent."""
    target_dir = scenario_dir(origin, store_dir)
    writeup_path = target_dir / _WRITEUP_FILENAME
    meta_path = target_dir / _META_FILENAME
    if not writeup_path.exists():
        return None
    markdown = writeup_path.read_text(encoding="utf-8")
    meta: dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open() as f:
            meta = yaml.safe_load(f) or {}
    return ScenarioWriteup(origin=origin, markdown=markdown, meta=meta)


def list_scenario_origins(store_dir: Path = DEFAULT_SCENARIO_STORE_DIR) -> list[date]:
    """List every origin with a persisted write-up, sorted ascending."""
    if not store_dir.exists():
        return []
    origins = []
    for child in store_dir.iterdir():
        if child.is_dir() and (child / _WRITEUP_FILENAME).exists():
            try:
                origins.append(date.fromisoformat(child.name))
            except ValueError:
                continue
    return sorted(origins)


def write_judge_verdict(origin: date, verdict: dict[str, Any], store_dir: Path = DEFAULT_SCENARIO_STORE_DIR) -> Path:
    """Persist a judge verdict alongside the write-up. Returns the file path."""
    target_dir = scenario_dir(origin, store_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / _JUDGE_FILENAME
    with path.open("w") as f:
        yaml.safe_dump(verdict, f, default_flow_style=False, sort_keys=False)
    return path


def load_judge_verdict(origin: date, store_dir: Path = DEFAULT_SCENARIO_STORE_DIR) -> dict[str, Any] | None:
    """Load a persisted judge verdict, or ``None`` if absent."""
    path = scenario_dir(origin, store_dir) / _JUDGE_FILENAME
    if not path.exists():
        return None
    with path.open() as f:
        return yaml.safe_load(f)


def has_judge_verdict(origin: date, store_dir: Path = DEFAULT_SCENARIO_STORE_DIR) -> bool:
    """Whether a judge verdict is already persisted for ``origin`` (resume gate)."""
    return (scenario_dir(origin, store_dir) / _JUDGE_FILENAME).exists()


__all__ = [
    "DEFAULT_SCENARIO_STORE_DIR",
    "ScenarioWriteup",
    "has_judge_verdict",
    "has_writeup",
    "list_scenario_origins",
    "load_judge_verdict",
    "load_scenario_writeup",
    "scenario_dir",
    "write_judge_verdict",
    "write_scenario_writeup",
]
