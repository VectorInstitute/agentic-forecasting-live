"""Resumable, per-origin persisting backtest runner with cost accounting.

The bootcamp's :func:`~aieng.forecasting.evaluation.cached_multi_backtest`
resumes at *task* granularity (one YAML per predictor×task, written after the
whole task completes). For the workshop's expensive agentic and LLMP runs that
is too coarse: a crash at origin 40 of 52 would discard 40 completed forecasts.

This runner persists **each origin's predictions immediately** and resumes by
skipping any origin whose file already exists, so an interrupted run costs at
most one origin of rework. Scoring (see :mod:`workshop_experiments.scoring`)
reads these persisted predictions and never re-calls the API.

Layout::

    data/predictions/<spec_id>/<predictor_id>/<task_id>/<YYYY-MM-DD>.yaml
    data/predictions/<spec_id>/<predictor_id>/accounting.json

Each origin file carries the origin's ``list[Prediction]`` plus wall-clock time;
``accounting.json`` summarises the run — new vs cached vs skipped origins, call
count, wall time, and any token/cost usage the predictors reported in
``Prediction.metadata`` (LLMP/agents populate ``cost_usd`` /
``input_tokens`` / ``output_tokens``; conventional methods report none, so the
runner falls back to call counts + wall time as the brief requires).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation import MultiTargetBacktestSpec, Prediction
from aieng.forecasting.evaluation.backtest import BacktestSpec
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

#: Default prediction store, bundled with the package (committed smoke runs live
#: here; large full runs are git-ignored by the caller's environment).
DEFAULT_STORE_DIR = Path(__file__).resolve().parent / "data" / "predictions"


class RunAccounting(BaseModel):
    """Per-(spec, predictor) accounting summary for one runner invocation."""

    spec_id: str
    predictor_id: str
    ran_at: datetime
    n_candidate_origins: int = Field(description="Grid origins per task, before warmup/resume.")
    n_predicted: int = Field(default=0, description="predict() calls made this run (new origins).")
    n_cached: int = Field(default=0, description="Origins skipped because a persisted file already existed.")
    n_skipped_warmup: int = Field(default=0, description="Origins skipped for insufficient history.")
    n_failed: int = Field(default=0, description="Origins where predict() raised and was skipped.")
    n_predictions_written: int = Field(default=0, description="Total Prediction records written this run.")
    wall_time_s: float = Field(default=0.0, description="Wall-clock seconds spent in predict() this run.")
    cost_usd: float = Field(default=0.0, description="Summed Prediction.metadata cost_usd (0 when unreported).")
    input_tokens: int = Field(default=0, description="Summed Prediction.metadata input_tokens.")
    output_tokens: int = Field(default=0, description="Summed Prediction.metadata output_tokens.")


def _predictor_dir(store_dir: Path, spec_id: str, predictor_id: str) -> Path:
    return store_dir / spec_id / predictor_id


def _origin_file(store_dir: Path, spec_id: str, predictor_id: str, task_id: str, origin: datetime) -> Path:
    return _predictor_dir(store_dir, spec_id, predictor_id) / task_id / f"{origin.date().isoformat()}.yaml"


def save_origin_predictions(path: Path, predictions: list[Prediction], *, wall_time_s: float, as_of: datetime) -> None:
    """Persist one origin's predictions (plus wall time) to a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "as_of": as_of.isoformat(),
        "wall_time_s": wall_time_s,
        "predictions": [p.model_dump(mode="json") for p in predictions],
    }
    with path.open("w") as f:
        yaml.safe_dump(record, f, default_flow_style=False, sort_keys=False)


def load_origin_predictions(path: Path) -> list[Prediction]:
    """Load one origin's persisted predictions."""
    with path.open() as f:
        record = yaml.safe_load(f)
    return [Prediction.model_validate(p) for p in record.get("predictions", [])]


def _usage_from_predictions(predictions: list[Prediction]) -> tuple[float, int, int]:
    """Sum (cost_usd, input_tokens, output_tokens) reported in prediction metadata."""
    cost = 0.0
    in_tok = 0
    out_tok = 0
    for pred in predictions:
        meta = pred.metadata or {}
        cost += float(meta.get("cost_usd", 0.0) or 0.0)
        in_tok += int(meta.get("input_tokens", 0) or 0)
        out_tok += int(meta.get("output_tokens", 0) or 0)
    return cost, in_tok, out_tok


def run_predictor_on_spec(
    predictor: Predictor,
    spec: MultiTargetBacktestSpec,
    data_service: DataService,
    *,
    store_dir: Path = DEFAULT_STORE_DIR,
    force_refresh: bool = False,
    inter_prediction_delay_s: float = 0.0,
    pace: bool = True,
) -> RunAccounting:
    """Run one predictor across every task/origin in *spec*, persisting per origin.

    Origins whose prediction file already exists are skipped (resume) unless
    ``force_refresh``. Warmup filtering matches the shared harness: an origin
    with fewer than ``spec.warmup`` cutoff-filtered target observations is
    skipped. Predict failures are logged and the origin is skipped (so a later
    resume retries it), never crashing the run.

    Parameters
    ----------
    inter_prediction_delay_s : float, default=0.0
        Seconds to sleep after each origin whose ``predict()`` actually ran
        (success or failure) — never after a cache-hit resume-skip or a
        warmup-skip, since no call was made in either case. This is politeness
        on a shared proxy quota, not a correctness requirement: it exists
        because the workshop runner otherwise fires predictions back-to-back,
        which is impolite on a shared quota and can itself trigger the 429
        bursts that :mod:`aieng.forecasting.methods.agentic.predictor` and the
        LLMP seam retry around. A value of ``0.0`` (the default) disables
        pacing entirely.
    pace : bool, default=True
        Whether this predictor is subject to ``inter_prediction_delay_s`` at
        all. Set ``False`` for conventional/local predictors that make no API
        call (see :data:`workshop_experiments.registry.CONVENTIONAL_METHODS`)
        — there is nothing to be polite about when no request left the
        process. :func:`run_spec` sets this from a predictor-id membership
        check; callers invoking this function directly default to ``True``.

    Returns
    -------
    RunAccounting
        Summary of this invocation; also written to ``accounting.json`` next to
        the predictor's predictions.
    """
    acc = RunAccounting(
        spec_id=spec.spec_id,
        predictor_id=predictor.predictor_id,
        ran_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
        n_candidate_origins=0,
    )
    single_specs: list[BacktestSpec] = spec.specs()
    if single_specs:
        acc.n_candidate_origins = len(single_specs[0].origins())

    def _maybe_pace() -> None:
        if pace and inter_prediction_delay_s > 0:
            time.sleep(inter_prediction_delay_s)

    for single_spec in single_specs:
        task: ForecastingTask = single_spec.task
        for origin in single_spec.origins():
            path = _origin_file(store_dir, spec.spec_id, predictor.predictor_id, task.task_id, origin)
            if not force_refresh and path.exists():
                acc.n_cached += 1
                continue

            ctx = data_service.context(as_of=origin)
            if single_spec.warmup > 0:
                series = ctx.get_series(task.target_series_id)
                if len(series) < single_spec.warmup:
                    acc.n_skipped_warmup += 1
                    continue

            start = time.perf_counter()
            try:
                predictions = predictor.predict(task, ctx)
            except Exception as exc:  # noqa: BLE001 — one bad origin must not kill the run
                logger.warning(
                    "predict() failed: predictor=%s task=%s origin=%s — skipping: %s",
                    predictor.predictor_id,
                    task.task_id,
                    origin.date(),
                    exc,
                )
                acc.n_failed += 1
                _maybe_pace()
                continue
            wall = time.perf_counter() - start

            if not predictions:
                logger.warning(
                    "predict() returned no predictions: predictor=%s task=%s origin=%s — skipping",
                    predictor.predictor_id,
                    task.task_id,
                    origin.date(),
                )
                acc.n_failed += 1
                _maybe_pace()
                continue

            save_origin_predictions(path, predictions, wall_time_s=wall, as_of=origin)
            cost, in_tok, out_tok = _usage_from_predictions(predictions)
            acc.n_predicted += 1
            acc.n_predictions_written += len(predictions)
            acc.wall_time_s += wall
            acc.cost_usd += cost
            acc.input_tokens += in_tok
            acc.output_tokens += out_tok
            _maybe_pace()

    _write_accounting(store_dir, spec.spec_id, predictor.predictor_id, acc)
    return acc


def _write_accounting(store_dir: Path, spec_id: str, predictor_id: str, acc: RunAccounting) -> None:
    path = _predictor_dir(store_dir, spec_id, predictor_id) / "accounting.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(acc.model_dump(mode="json"), f, indent=2, sort_keys=False)
        f.write("\n")


def run_spec(
    predictors: list[Predictor],
    spec: MultiTargetBacktestSpec,
    data_service: DataService,
    *,
    store_dir: Path = DEFAULT_STORE_DIR,
    force_refresh: bool = False,
    inter_prediction_delay_s: float = 0.0,
    conventional_predictor_ids: frozenset[str] = frozenset(),
) -> dict[str, RunAccounting]:
    """Run every predictor in *predictors* across *spec*, resuming per origin.

    Parameters
    ----------
    inter_prediction_delay_s : float, default=0.0
        Politeness pacing on a shared proxy quota — forwarded to
        :func:`run_predictor_on_spec`. See its docstring for the exact
        skip conditions (cache-hit / warmup-skip origins are never paced).
    conventional_predictor_ids : frozenset[str], default=frozenset()
        ``predictor_id``s that make no API call (e.g. built from
        :data:`workshop_experiments.registry.CONVENTIONAL_METHODS` /
        ``TSX_CONVENTIONAL_METHODS``). These are exempt from
        ``inter_prediction_delay_s`` regardless of its value.

    Returns
    -------
    dict[str, RunAccounting]
        Mapping from ``predictor_id`` to its :class:`RunAccounting`.
    """
    results: dict[str, RunAccounting] = {}
    for predictor in predictors:
        logger.info("Running predictor=%s on spec=%s", predictor.predictor_id, spec.spec_id)
        results[predictor.predictor_id] = run_predictor_on_spec(
            predictor,
            spec,
            data_service,
            store_dir=store_dir,
            force_refresh=force_refresh,
            inter_prediction_delay_s=inter_prediction_delay_s,
            pace=predictor.predictor_id not in conventional_predictor_ids,
        )
    return results


__all__ = [
    "DEFAULT_STORE_DIR",
    "RunAccounting",
    "load_origin_predictions",
    "run_predictor_on_spec",
    "run_spec",
    "save_origin_predictions",
]
