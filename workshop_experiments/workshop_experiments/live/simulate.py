"""Offline ``--simulate`` wiring: run the full pipeline with no API/network.

Uses the committed smoke predictions under
``workshop_experiments/data/predictions/sp500_ws_smoke/`` as fake "today"
outputs and reconstructs an offline realized-return lookup from the committed
naive predictions (the naive point forecast at an origin *is* the observed
target value at that origin, so it supplies the market outcome that later
horizons resolve against). Together these let the write -> resolve -> aggregate
-> validate chain run end to end without any model call.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from workshop_experiments.live.aggregate import aggregate_step
from workshop_experiments.live.config import LiveConfig
from workshop_experiments.live.predict import SimulatePredictionSource, predict_step
from workshop_experiments.live.records import utc_now_z
from workshop_experiments.live.resolve import LookupRealizedProvider, resolve_log


#: Registry id of the naive predictor whose committed point forecasts supply the
#: offline realized-return lookup.
_NAIVE_REGISTRY_ID = "last_value_naive"


def latest_smoke_origin(config: LiveConfig) -> date:
    """Return the most recent origin present in the committed smoke store."""
    task_dir = config.smoke_store / _NAIVE_REGISTRY_ID / config.task_id_for_horizon(config.horizons[0])
    origins = sorted(date.fromisoformat(p.stem) for p in task_dir.glob("*.yaml"))
    if not origins:
        raise FileNotFoundError(f"no smoke origins under {task_dir}")
    return origins[-1]


def smoke_origins(config: LiveConfig) -> list[date]:
    """Return every origin present in the committed smoke store, ascending."""
    task_dir = config.smoke_store / _NAIVE_REGISTRY_ID / config.task_id_for_horizon(config.horizons[0])
    return sorted(date.fromisoformat(p.stem) for p in task_dir.glob("*.yaml"))


def build_realized_lookup(config: LiveConfig) -> dict[int, dict[date, float]]:
    """Reconstruct ``{horizon: {origin_date: realized_value}}`` from naive files.

    The naive predictor stores, at each origin, the observed ``sp500_logret_{h}b``
    value as its point forecast — exactly the realized cumulative log return a
    matured horizon resolves against.
    """
    from workshop_experiments.runner import load_origin_predictions  # noqa: PLC0415

    lookup: dict[int, dict[date, float]] = {}
    for horizon in config.horizons:
        task_dir = config.smoke_store / _NAIVE_REGISTRY_ID / config.task_id_for_horizon(horizon)
        by_date: dict[date, float] = {}
        for path in sorted(task_dir.glob("*.yaml")):
            prediction = load_origin_predictions(path)[0]
            by_date[prediction.as_of.date()] = float(prediction.payload.point_forecast)
        lookup[horizon] = by_date
    return lookup


@dataclass(frozen=True)
class SimulationResult:
    """Counts from an end-to-end offline simulation."""

    origins: list[date]
    n_predictions_written: int
    n_gapped: int
    n_resolutions: int
    aggregates: dict[str, Path]


def run_simulation(
    config: LiveConfig,
    origins: list[date],
    *,
    log_dir: Path,
    out_dir: Path,
    submission_timestamp: str | None = None,
    resolved_at: str | None = None,
) -> SimulationResult:
    """Run predict (per origin) -> resolve -> aggregate offline into temp dirs.

    Parameters
    ----------
    config : LiveConfig
        The deployed configuration + ladder.
    origins : list[date]
        Simulated trading days to predict, in order.
    log_dir : Path
        Append-only log root (a temp dir in tests).
    out_dir : Path
        Aggregates destination (a temp dir in tests).
    submission_timestamp, resolved_at : str or None
        Fixed schema timestamps for determinism; ``None`` uses the current UTC.

    Returns
    -------
    SimulationResult
        Prediction/gap/resolution counts and the aggregate paths written.
    """
    submitted_at = submission_timestamp or utc_now_z()
    resolved_stamp = resolved_at or submitted_at

    n_written = 0
    n_gapped = 0
    for origin in origins:
        source = SimulatePredictionSource(config=config, origin=origin)
        result = predict_step(config, source, log_dir=log_dir, submission_timestamp=submitted_at)
        n_written += len(result.written)
        n_gapped += len(result.gapped)

    provider = LookupRealizedProvider(build_realized_lookup(config))
    resolutions = resolve_log(log_dir, provider, resolved_at=resolved_stamp)
    aggregates = aggregate_step(log_dir, out_dir)

    return SimulationResult(
        origins=list(origins),
        n_predictions_written=n_written,
        n_gapped=n_gapped,
        n_resolutions=len(resolutions),
        aggregates=aggregates,
    )


__all__ = [
    "SimulationResult",
    "build_realized_lookup",
    "latest_smoke_origin",
    "run_simulation",
    "smoke_origins",
]
