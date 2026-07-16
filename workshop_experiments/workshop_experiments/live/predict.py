"""The predict step: run every configured rung and write prediction records.

A *prediction source* supplies the ``list[Prediction]`` (one per horizon) for one
rung at the origin. Two sources exist:

- :class:`RealPredictionSource` — builds the predictor from the registry and
  calls ``predict`` against the live data service (makes API calls; used in
  production only).
- :class:`SimulatePredictionSource` — reads the committed smoke predictions as
  fake "today" outputs (no API, no network; used by ``--simulate`` and tests).

For each rung the step builds a schema-conforming record (enforcing the writer
invariants) and writes it to the append-only log. A rung that raises after its
retries are exhausted gets a per-method gap-log entry and the run continues.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

from workshop_experiments.live import SCHEMA_VERSION
from workshop_experiments.live.config import LiveConfig, LivePredictor, registry_predictor_id
from workshop_experiments.live.log_store import append_gap, has_prediction, write_prediction
from workshop_experiments.live.records import (
    build_prediction_record,
    group_predictions_by_horizon,
    utc_now_z,
)
from workshop_experiments.live.schema_validation import check


logger = logging.getLogger(__name__)


class PredictionUnavailable(Exception):  # noqa: N818 - names an availability outcome
    """Raised by a prediction source when a rung cannot produce a forecast."""


class PredictionSource(Protocol):
    """Supplies the ``list[Prediction]`` for one rung at the run's origin."""

    origin: date

    def predictions_for(self, live: LivePredictor) -> list[Any]:
        """Return one ``Prediction`` per horizon, or raise on failure."""
        ...


@dataclass
class SimulatePredictionSource:
    """Offline source: committed smoke predictions as fake "today" outputs.

    Loads ``<smoke_store>/<registry_id>/<task_id>/<origin>.yaml`` for each
    configured horizon. A rung with no committed fixture raises
    :class:`PredictionUnavailable` (so the run exercises the gap policy).
    """

    config: LiveConfig
    origin: date

    def predictions_for(self, live: LivePredictor) -> list[Any]:
        """Load the smoke predictions for *live* at the simulated origin."""
        from workshop_experiments.runner import load_origin_predictions  # noqa: PLC0415

        registry_id = registry_predictor_id(live.registry_method, live.model)
        predictions: list[Any] = []
        for horizon in self.config.horizons:
            path = (
                self.config.smoke_store
                / registry_id
                / self.config.task_id_for_horizon(horizon)
                / f"{self.origin.isoformat()}.yaml"
            )
            if not path.exists():
                raise PredictionUnavailable(f"no committed simulate fixture for {registry_id} at {self.origin}")
            predictions.extend(load_origin_predictions(path))
        return predictions


@dataclass
class RealPredictionSource:
    """Production source: build the predictor and call ``predict`` (uses API).

    Not exercised in tests — the offline suites use
    :class:`SimulatePredictionSource`. Builds one single-horizon
    :class:`ForecastingTask` per configured horizon and predicts against a
    context cut at the origin close.
    """

    config: LiveConfig
    data_service: Any
    origin: date
    covariate_panel: list[str] | None = None

    def predictions_for(self, live: LivePredictor) -> list[Any]:
        """Build *live* and predict every horizon at the origin close."""
        from aieng.forecasting.evaluation.task import ForecastingTask  # noqa: PLC0415

        from workshop_experiments.registry import build_predictor  # noqa: PLC0415

        predictor = build_predictor(
            live.registry_method,
            model=live.model or _default_model(),
            covariate_panel=self.covariate_panel,
        )
        ctx = self.data_service.context(as_of=datetime(self.origin.year, self.origin.month, self.origin.day))
        predictions: list[Any] = []
        for horizon in self.config.horizons:
            task = ForecastingTask(
                task_id=self.config.task_id_for_horizon(horizon),
                target_series_id=self.config.task_id_for_horizon(horizon),
                horizons=[horizon],
                frequency="B",
            )
            result = predictor.predict(task, ctx)
            if not result:
                raise PredictionUnavailable(f"{live.predictor_id} returned no predictions at h={horizon}")
            predictions.extend(result)
        return predictions


def _default_model() -> str:
    """Return the lite model id (used when a rung carries no explicit model)."""
    from aieng.forecasting.models import LITE_MODEL  # noqa: PLC0415

    return LITE_MODEL


@dataclass(frozen=True)
class PredictStepResult:
    """Outcome of one predict step over the configured ladder."""

    origin: date
    written: list[str]
    skipped: list[str]
    gapped: list[str]


def predict_step(
    config: LiveConfig,
    source: PredictionSource,
    *,
    log_dir: Path | None = None,
    submission_timestamp: str | None = None,
    validate: bool = True,
) -> PredictStepResult:
    """Run every configured rung through *source* and write prediction records.

    Rungs whose record already exists for this origin are skipped (idempotent
    re-run). A rung that raises :class:`PredictionUnavailable` — or any other
    exception — is logged as a per-method gap and the run continues.

    Parameters
    ----------
    config : LiveConfig
        The deployed configuration + ladder.
    source : PredictionSource
        Supplies each rung's predictions at ``source.origin``.
    log_dir : Path or None
        Log root; defaults to ``config.log_dir``.
    submission_timestamp : str or None
        Schema timestamp stamped on every record; ``None`` uses the current UTC
        time (pass a fixed value for deterministic tests).
    validate : bool
        Validate each record against the prediction schema before writing.

    Returns
    -------
    PredictStepResult
        The predictor ids written, skipped, and gapped.
    """
    log_root = log_dir if log_dir is not None else config.log_dir
    submitted_at = submission_timestamp if submission_timestamp is not None else utc_now_z()
    origin = source.origin

    written: list[str] = []
    skipped: list[str] = []
    gapped: list[str] = []

    for live in config.predictors:
        if has_prediction(log_root, origin, live.predictor_id):
            skipped.append(live.predictor_id)
            continue
        try:
            raw_predictions = source.predictions_for(live)
            by_horizon = group_predictions_by_horizon(config, raw_predictions)
            missing = [h for h in config.horizons if h not in by_horizon]
            if missing:
                raise PredictionUnavailable(f"{live.predictor_id} missing horizons {missing}")
            record = build_prediction_record(
                live,
                by_horizon,
                origin=origin,
                submission_timestamp=submitted_at,
                schema_version=SCHEMA_VERSION,
            )
            if validate:
                check("prediction", record)
        except Exception as exc:  # noqa: BLE001 - one bad rung must not kill the run
            logger.warning("predict failed for %s: %s", live.predictor_id, exc)
            append_gap(
                log_root,
                origin,
                {
                    "schema_version": SCHEMA_VERSION,
                    "date": origin.isoformat(),
                    "scope": live.predictor_id,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "retries_attempted": config.retry.max_attempts,
                    "logged_at": submitted_at,
                },
            )
            gapped.append(live.predictor_id)
            continue
        write_prediction(log_root, origin, record)
        written.append(live.predictor_id)

    return PredictStepResult(origin=origin, written=written, skipped=skipped, gapped=gapped)


__all__ = [
    "PredictStepResult",
    "PredictionSource",
    "PredictionUnavailable",
    "RealPredictionSource",
    "SimulatePredictionSource",
    "predict_step",
]
