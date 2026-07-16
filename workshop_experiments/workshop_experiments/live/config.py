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

#: Registry method name -> data-contract ``method`` enum value (schema 1.1.0:
#: one enum value per deployed rung, so ``(method, model, horizon)`` uniquely
#: keys every leaderboard cell and ``model`` is never overloaded as a variant
#: label).
SCHEMA_METHOD: dict[str, str] = {
    "naive": "naive",
    "ets": "ets",
    "kalman": "kalman",
    "autoarima": "autoarima",
    "lightgbm": "lightgbm",
    "lightgbm_cov": "lightgbm_cov",
    "llmp_qgrid": "llm_process",
    "llmp_qgrid_cov": "llm_process_cov",
    "agent_news": "agent_news",
    "agent_code": "agent_code",
    # Stage-2c adaptive twins — same data-contract method enum values (1.1.0).
    "adaptive_frozen": "adaptive_frozen",
    "adaptive_learning": "adaptive_learning",
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
    twin_id: str | None = None


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
    #: Stage-2c adaptive twins (deployed later; kept OUT of ``predictors`` so the
    #: stateless-ladder counts stay stable). Empty until a ``twins:`` block is set.
    twins: tuple[LivePredictor, ...] = field(default_factory=tuple)
    #: Raw ``gates:`` block (fed to ``GateConfig.from_mapping`` by the twin
    #: runtime). Kept as a plain mapping so ``config.py`` never imports the gate
    #: package (which would import ``config`` back).
    gate_params: dict[str, Any] = field(default_factory=dict)

    def task_id_for_horizon(self, horizon: int) -> str:
        """Return the workshop target/task id for a business-day *horizon*."""
        return f"sp500_logret_{horizon}b"

    def by_group(self, group: str) -> list[LivePredictor]:
        """Return the configured predictors in one scope *group*."""
        return [p for p in self.predictors if p.group == group]

    def twin(self, twin_id: str) -> LivePredictor:
        """Return the twin rung with the given ``twin_id`` (raises if absent)."""
        for rung in self.twins:
            if rung.twin_id == twin_id:
                return rung
        raise KeyError(f"no twin {twin_id!r} configured (have: {[t.twin_id for t in self.twins]})")


def _expand_conventional(entries: list[dict[str, Any]]) -> list[LivePredictor]:
    """Expand the conventional block into :class:`LivePredictor` rungs.

    Conventional rungs carry ``model = None`` (they consume no model) and the
    contract's ``sp500_<method>`` predictor-id convention.
    """
    out: list[LivePredictor] = []
    for entry in entries:
        method = entry["method"]
        schema_method = SCHEMA_METHOD[method]
        out.append(
            LivePredictor(
                registry_method=method,
                model=None,
                schema_method=schema_method,
                model_label=None,
                predictor_id=f"sp500_{schema_method}",
                group="conventional",
            )
        )
    return out


def _expand_api(block: dict[str, Any], group: str) -> list[LivePredictor]:
    """Expand an API (llmp/agent) method x model matrix into rungs.

    API rungs carry the plain backing-model id as their public ``model`` label
    (no variant suffixes — the covariate variant is expressed by the method)
    and the contract's ``sp500_<method>__<model>`` predictor-id convention.
    """
    out: list[LivePredictor] = []
    methods = list(block.get("methods", []))
    models = list(block.get("models", []))
    for method in methods:
        schema_method = SCHEMA_METHOD[method]
        for model in models:
            out.append(
                LivePredictor(
                    registry_method=method,
                    model=model,
                    schema_method=schema_method,
                    model_label=model,
                    predictor_id=f"sp500_{schema_method}__{model}",
                    group=group,
                )
            )
    return out


#: The two adaptive twins, both built from the same adaptive predictor config
#: pointed at the trained strategy dir. The frozen twin is read-only; the
#: learning twin carries the mutation tools + gates. Their ``predictor_id`` join
#: keys follow the ``sp500_<method>__<model>`` convention like the agent rungs.
TWIN_IDS: tuple[str, ...] = ("adaptive_frozen", "adaptive_learning")


def _expand_twins(block: dict[str, Any]) -> tuple[LivePredictor, ...]:
    """Expand the optional ``twins:`` block into the two twin rungs.

    The block carries a single ``model`` (both twins share it — the only degree
    of freedom between them is experience) and an ``enabled`` flag (twins deploy
    later than the stateless ladder, so they default off). Returns an empty tuple
    when absent or disabled.
    """
    if not block or not block.get("enabled", False):
        return ()
    model = str(block["model"])
    out: list[LivePredictor] = []
    for twin_id in TWIN_IDS:
        schema_method = SCHEMA_METHOD[twin_id]
        out.append(
            LivePredictor(
                registry_method=twin_id,
                model=model,
                schema_method=schema_method,
                model_label=model,
                predictor_id=f"sp500_{schema_method}__{model}",
                group="twins",
                twin_id=twin_id,
            )
        )
    return tuple(out)


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
        twins=_expand_twins(raw.get("twins", {})),
        gate_params=dict(raw.get("gates", {})),
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
    "TWIN_IDS",
    "LiveConfig",
    "LivePredictor",
    "RetryPolicy",
    "expand_predictors",
    "load_config",
    "registry_predictor_id",
]
