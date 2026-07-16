"""Load ``live_config.yaml`` and expand the deployed predictor ladder.

The YAML holds the *deployment* knobs — horizons, submission time, retry policy,
paths, and the model matrices. The binding from a registry method name to its
data-contract ``method`` enum value and its public display label lives here in
code (it is a fixed taxonomy, not a tuning knob).

Each expanded :class:`LivePredictor` carries two identities:

- the **registry** id (:func:`registry_predictor_id`) — the
  :mod:`workshop_experiments.registry` predictor id, used to build the predictor
  for a real run and to locate the committed smoke store in ``--simulate``;
- the **live** id (:attr:`LivePredictor.predictor_id`) — the stable
  ``sp500_...`` join key the data contract records key on, plus the
  ``(schema_method, model_label)`` pair the monitor leaderboard keys its cells
  on. These are chosen so no two rungs collide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


#: Repo root: ``.../workshop_experiments/workshop_experiments/live`` -> parents[3].
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Default config shipped with the package.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "live_config.yaml"

#: Registry method name -> data-contract ``method`` enum value.
SCHEMA_METHOD: dict[str, str] = {
    "naive": "naive",
    "ets": "classical",
    "kalman": "classical",
    "autoarima": "classical",
    "lightgbm": "lightgbm",
    "lightgbm_cov": "lightgbm",
    "llmp_qgrid": "llm_process",
    "llmp_qgrid_cov": "llm_process",
    "agent_news": "analyst_agent",
    "agent_code": "code_agent",
}

#: Display label for conventional rungs. Several conventional methods share one
#: schema ``method`` value (ets/kalman/autoarima -> ``classical``; lightgbm and
#: lightgbm_cov -> ``lightgbm``), so a distinct ``model`` label is required to
#: keep each a separate leaderboard cell. ``None`` is used only where the schema
#: method is already unique to one rung.
_CONVENTIONAL_MODEL_LABEL: dict[str, str | None] = {
    "naive": None,
    "ets": "ets",
    "kalman": "kalman",
    "autoarima": "autoarima",
    "lightgbm": None,
    "lightgbm_cov": "cov",
}

#: Stable ``predictor_id`` for conventional rungs (matches ``sp500_<method>``).
_CONVENTIONAL_PREDICTOR_ID: dict[str, str] = {
    "naive": "sp500_naive",
    "ets": "sp500_ets",
    "kalman": "sp500_kalman",
    "autoarima": "sp500_autoarima",
    "lightgbm": "sp500_lightgbm",
    "lightgbm_cov": "sp500_lightgbm_cov",
}

#: ``registry method -> (predictor_id stem, model-label suffix)`` for the API
#: rungs. The live ``predictor_id`` is ``sp500_<stem>__<model>``; the model
#: label is ``<model><suffix>`` (the ``+cov`` suffix keeps covariate LLMP
#: variants distinct from their base variant in the leaderboard).
_API_PREDICTOR_STEM: dict[str, tuple[str, str]] = {
    "llmp_qgrid": ("llm_process", ""),
    "llmp_qgrid_cov": ("llm_process_cov", "+cov"),
    "agent_news": ("analyst_agent", ""),
    "agent_code": ("code_agent", ""),
}


@dataclass(frozen=True)
class LivePredictor:
    """One deployed rung: how to build it and how to key its records.

    Parameters
    ----------
    registry_method : str
        Key into :mod:`workshop_experiments.registry` (e.g. ``llmp_qgrid_cov``).
    model : str or None
        Model id passed to the registry for API methods; ``None`` for
        conventional methods.
    schema_method : str
        The data-contract ``method`` enum value written into records.
    model_label : str or None
        The public ``model`` label written into records and used (with
        ``schema_method``) as the leaderboard cell key.
    predictor_id : str
        Stable ``sp500_...`` join key the records are written under.
    group : str
        ``conventional`` | ``llmp`` | ``agent`` — the submission scope bucket.
    """

    registry_method: str
    model: str | None
    schema_method: str
    model_label: str | None
    predictor_id: str
    group: str


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded same-evening retry policy for a submission run."""

    max_attempts: int
    backoff_minutes: int
    hard_stop_local: str


@dataclass(frozen=True)
class LiveConfig:
    """Parsed ``live_config.yaml`` plus the expanded predictor ladder."""

    schema_version: str
    target_ticker: str
    horizons: tuple[int, ...]
    submission_time_local: str
    timezone: str
    retry: RetryPolicy
    log_dir: Path
    aggregates_dir: Path
    smoke_store: Path
    predictors: tuple[LivePredictor, ...] = field(default_factory=tuple)

    def task_id_for_horizon(self, horizon: int) -> str:
        """Return the workshop target/task id for a business-day *horizon*."""
        return f"sp500_logret_{horizon}b"

    def by_group(self, group: str) -> list[LivePredictor]:
        """Return the configured predictors in one scope *group*."""
        return [p for p in self.predictors if p.group == group]


def _expand_conventional(entries: list[dict[str, Any]]) -> list[LivePredictor]:
    """Expand the conventional block into :class:`LivePredictor` rungs."""
    out: list[LivePredictor] = []
    for entry in entries:
        method = entry["method"]
        out.append(
            LivePredictor(
                registry_method=method,
                model=None,
                schema_method=SCHEMA_METHOD[method],
                model_label=_CONVENTIONAL_MODEL_LABEL[method],
                predictor_id=_CONVENTIONAL_PREDICTOR_ID[method],
                group="conventional",
            )
        )
    return out


def _expand_api(block: dict[str, Any], group: str) -> list[LivePredictor]:
    """Expand an API (llmp/agent) method x model matrix into rungs."""
    out: list[LivePredictor] = []
    methods = list(block.get("methods", []))
    models = list(block.get("models", []))
    for method in methods:
        stem, suffix = _API_PREDICTOR_STEM[method]
        for model in models:
            out.append(
                LivePredictor(
                    registry_method=method,
                    model=model,
                    schema_method=SCHEMA_METHOD[method],
                    model_label=f"{model}{suffix}",
                    predictor_id=f"sp500_{stem}__{model}",
                    group=group,
                )
            )
    return out


def expand_predictors(raw: dict[str, Any]) -> tuple[LivePredictor, ...]:
    """Expand the ``predictors:`` block into the full, ordered ladder.

    Order is conventional, then LLMP, then agents — the fixed method order the
    monitor renders in. Raises ``ValueError`` if two rungs would collide on
    ``predictor_id`` or on ``(schema_method, model_label)``.
    """
    predictors: list[LivePredictor] = []
    predictors += _expand_conventional(raw.get("conventional", []))
    predictors += _expand_api(raw.get("llmp", {}), "llmp")
    predictors += _expand_api(raw.get("agent", {}), "agent")

    ids = [p.predictor_id for p in predictors]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate predictor_id in ladder: {sorted(_duplicates(ids))}")
    cells = [(p.schema_method, p.model_label) for p in predictors]
    if len(set(cells)) != len(cells):
        raise ValueError(f"Duplicate (schema_method, model_label) in ladder: {sorted(map(str, _duplicates(cells)))}")
    return tuple(predictors)


def _duplicates(items: list[Any]) -> set[Any]:
    """Return the set of values appearing more than once in *items*."""
    seen: set[Any] = set()
    dups: set[Any] = set()
    for item in items:
        if item in seen:
            dups.add(item)
        seen.add(item)
    return dups


def _resolve_path(root: Path, value: str) -> Path:
    """Resolve a config path against the repo *root* unless already absolute."""
    path = Path(value)
    return path if path.is_absolute() else (root / path)


def load_config(path: Path | str | None = None, *, repo_root: Path | None = None) -> LiveConfig:
    """Load and validate the live config from *path* (default: the shipped one).

    Parameters
    ----------
    path : Path or str or None
        Config YAML path; ``None`` loads :data:`DEFAULT_CONFIG_PATH`.
    repo_root : Path or None
        Root the ``paths:`` entries resolve against; ``None`` uses
        :data:`REPO_ROOT`.

    Returns
    -------
    LiveConfig
        The parsed configuration with the predictor ladder expanded.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    root = repo_root if repo_root is not None else REPO_ROOT
    with cfg_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    submission = raw["submission"]
    retry = raw["retry"]
    paths = raw["paths"]
    return LiveConfig(
        schema_version=str(raw["schema_version"]),
        target_ticker=str(raw["target_ticker"]),
        horizons=tuple(int(h) for h in raw["horizons"]),
        submission_time_local=str(submission["time_local"]),
        timezone=str(submission["timezone"]),
        retry=RetryPolicy(
            max_attempts=int(retry["max_attempts"]),
            backoff_minutes=int(retry["backoff_minutes"]),
            hard_stop_local=str(retry["hard_stop_local"]),
        ),
        log_dir=_resolve_path(root, paths["log_dir"]),
        aggregates_dir=_resolve_path(root, paths["aggregates_dir"]),
        smoke_store=_resolve_path(root, paths["smoke_store"]),
        predictors=expand_predictors(raw["predictors"]),
    )


@lru_cache(maxsize=None)
def registry_predictor_id(registry_method: str, model: str | None) -> str:
    """Return the :mod:`workshop_experiments.registry` id for a rung.

    This is the smoke-store directory name and the real-run cache key. Computed
    by actually constructing the predictor (cheap, offline, no API call) so it
    never drifts from the registry's own id logic. Cached per (method, model).
    """
    from workshop_experiments.registry import build_predictor  # noqa: PLC0415

    return build_predictor(registry_method, model=model or _default_model()).predictor_id


def _default_model() -> str:
    """Return the lite model id used when a rung carries no model."""
    from aieng.forecasting.models import LITE_MODEL  # noqa: PLC0415

    return LITE_MODEL


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "REPO_ROOT",
    "SCHEMA_METHOD",
    "LiveConfig",
    "LivePredictor",
    "RetryPolicy",
    "expand_predictors",
    "load_config",
    "registry_predictor_id",
]
