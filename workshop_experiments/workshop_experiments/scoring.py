"""Score persisted workshop predictions into leaderboard artifacts.

Reads the per-origin predictions written by :mod:`workshop_experiments.runner`,
resolves each against the realised return series, CRPS-scores it, and
reconstructs a :class:`~aieng.forecasting.evaluation.BacktestResult` per
(predictor, task). Those feed the pure-frame
:func:`sp500_forecasting.leaderboard.build_leaderboard` helper (reused directly
— no notebook/plotting imports) to produce the leaderboard frame, which is
written as CSV and Markdown under ``data/results/<spec_id>/``.

Scoring never calls any model API: it operates purely on the committed
prediction files and the (locally cached) data service.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import properscoring as ps
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation import MultiTargetBacktestSpec, Prediction
from aieng.forecasting.evaluation.backtest import BacktestResult, BacktestSpec
from aieng.forecasting.evaluation.prediction import ContinuousForecast
from sp500_forecasting.leaderboard import build_leaderboard

from workshop_experiments.runner import DEFAULT_STORE_DIR, load_origin_predictions


logger = logging.getLogger(__name__)

#: Default location for scored leaderboard artifacts.
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"


def _resolve_series(data_service: DataService, target_series_id: str) -> pd.Series:
    """Return a timestamp-indexed lookup of realised values for a target series."""
    as_of_now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    full = data_service.get_series(target_series_id, as_of=as_of_now).copy()
    full["timestamp"] = pd.to_datetime(full["timestamp"])
    return full.set_index("timestamp")["value"]


def _crps(prediction: Prediction, actual: float) -> float:
    """CRPS of one continuous prediction vs a realised value (quantile ensemble)."""
    if not isinstance(prediction.payload, ContinuousForecast):
        raise TypeError("CRPS scoring requires a ContinuousForecast payload.")
    ensemble = np.array(sorted(prediction.payload.quantiles.values()), dtype=float)
    return float(ps.crps_ensemble(actual, ensemble))


def _load_task_predictions(task_dir: Path) -> list[Prediction]:
    """Load and flatten every origin's predictions under a predictor/task dir."""
    predictions: list[Prediction] = []
    for origin_file in sorted(task_dir.glob("*.yaml")):
        predictions.extend(load_origin_predictions(origin_file))
    return predictions


def score_task(
    predictions: list[Prediction],
    single_spec: BacktestSpec,
    data_service: DataService,
) -> BacktestResult | None:
    """Resolve and CRPS-score one task's predictions into a :class:`BacktestResult`.

    Predictions whose ``forecast_date`` has no realised observation yet (future
    or unresolved) are dropped. Returns ``None`` when nothing resolved.
    """
    lookup = _resolve_series(data_service, single_spec.task.target_series_id)
    scored: list[Prediction] = []
    scores: list[float] = []
    for pred in predictions:
        ts = pd.Timestamp(pred.forecast_date)
        if ts not in lookup.index:
            continue
        scored.append(pred)
        scores.append(_crps(pred, float(lookup.loc[ts])))
    if not scored:
        return None
    return BacktestResult(
        spec=single_spec,
        predictor_id=scored[0].predictor_id,
        predictions=scored,
        scores=scores,
        metric="crps",
        mean_score=float(np.mean(scores)),
        ran_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
        skipped_origins=len(predictions) - len(scored),
    )


def score_spec(
    spec: MultiTargetBacktestSpec,
    data_service: DataService,
    *,
    store_dir: Path = DEFAULT_STORE_DIR,
) -> tuple[pd.DataFrame, dict[str, dict[str, BacktestResult]]]:
    """Score every persisted predictor for *spec* into a leaderboard frame.

    Discovers predictors from the store subdirectories under
    ``<store>/<spec_id>/`` and scores each task with persisted predictions.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        The leaderboard frame (one row per predictor×horizon) and the raw
        ``{predictor_id: {task_id: BacktestResult}}`` mapping it was built from.
    """
    spec_dir = store_dir / spec.spec_id
    single_specs = {s.task.task_id: s for s in spec.specs()}

    results_by_predictor: dict[str, dict[str, BacktestResult]] = {}
    covariates_by_predictor: dict[str, list[str]] = {}
    if not spec_dir.is_dir():
        logger.warning("No persisted predictions found under %s", spec_dir)
        return pd.DataFrame(), results_by_predictor

    for predictor_dir in sorted(p for p in spec_dir.iterdir() if p.is_dir()):
        predictor_id = predictor_dir.name
        task_results: dict[str, BacktestResult] = {}
        for task_id, single_spec in single_specs.items():
            task_dir = predictor_dir / task_id
            if not task_dir.is_dir():
                continue
            predictions = _load_task_predictions(task_dir)
            if not predictions:
                continue
            result = score_task(predictions, single_spec, data_service)
            if result is None:
                continue
            task_results[task_id] = result
            covariates = predictions[0].metadata.get("covariate_series_ids") if predictions[0].metadata else None
            if covariates:
                covariates_by_predictor[predictor_id] = list(covariates)
        if task_results:
            results_by_predictor[predictor_id] = task_results

    frame = build_leaderboard(
        results_by_predictor,
        data_service,
        covariates_by_predictor=covariates_by_predictor,
    )
    return frame, results_by_predictor


def write_leaderboard_artifacts(
    frame: pd.DataFrame,
    spec_id: str,
    *,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> dict[str, Path]:
    """Write the leaderboard frame as CSV and Markdown under ``results_dir``.

    Returns a mapping ``{"csv": path, "markdown": path}``.
    """
    out_dir = results_dir / spec_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "leaderboard.csv"
    md_path = out_dir / "leaderboard.md"

    frame.to_csv(csv_path, index=False)
    md_path.write_text(_frame_to_markdown(frame, title=f"{spec_id} leaderboard"))
    return {"csv": csv_path, "markdown": md_path}


def _frame_to_markdown(frame: pd.DataFrame, *, title: str) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no tabulate dep)."""
    if frame.empty:
        return f"# {title}\n\n_No scored predictions found._\n"
    columns = list(frame.columns)
    header = "| " + " | ".join(str(c) for c in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_format_cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return f"# {title}\n\n" + "\n".join([header, divider, *rows]) + "\n"


def _format_cell(value: object) -> str:
    """Format one table cell: round floats, blank out NaN, stringify the rest."""
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.5f}"
    return str(value)


__all__ = [
    "DEFAULT_RESULTS_DIR",
    "score_spec",
    "score_task",
    "write_leaderboard_artifacts",
]
