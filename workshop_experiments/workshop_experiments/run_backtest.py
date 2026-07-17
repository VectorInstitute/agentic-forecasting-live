r"""CLI: run workshop predictors over a spec, resuming and persisting per origin.

Examples
--------
Smoke-first, conventional only (no API spend)::

    ws-run-backtest --spec sp500_ws_smoke --methods naive ets autoarima
    python -m workshop_experiments.run_backtest \
        --spec sp500_ws_smoke --methods conventional

Weekly backtest, LLMP quantile-grid across two models::

    ws-run-backtest --spec sp500_ws_backtest_2025_weekly \
        --methods llmp_qgrid llmp_qgrid_cov \
        --models gemini-3.1-flash-lite-preview gemini-3.5-flash

``--methods`` accepts individual names or the groups ``all``, ``conventional``,
``llmp``, ``agent``, ``api``. Runs resume automatically: origins with a persisted
prediction file are skipped unless ``--force-refresh`` is given.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from aieng.forecasting.evaluation import MultiTargetBacktestSpec
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.models import LITE_MODEL

from workshop_experiments.data import SP500_COVARIATE_PANEL, build_workshop_service
from workshop_experiments.data_tsx import TSX_COVARIATE_PANEL, build_tsx_workshop_service
from workshop_experiments.registry import (
    CONVENTIONAL_METHODS,
    TSX_CONVENTIONAL_METHODS,
    build_predictor,
    resolve_methods,
)
from workshop_experiments.runner import DEFAULT_STORE_DIR, run_spec
from workshop_experiments.specs import load_spec, origin_count


logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run workshop S&P 500 predictors over a spec (resumable).")
    parser.add_argument("--spec", required=True, help="Spec name (e.g. sp500_ws_smoke) or path to a YAML file.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["conventional"],
        help="Method names and/or groups (all, conventional, llmp, agent, api). Default: conventional.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[LITE_MODEL],
        help="Model ids for LLMP/agent methods (ignored by conventional). Default: the lite model.",
    )
    parser.add_argument("--store-dir", default=str(DEFAULT_STORE_DIR), help="Prediction store root.")
    parser.add_argument("--end", default=None, help="Optional data-fetch end (YYYY-MM-DD, exclusive).")
    parser.add_argument(
        "--no-covariates",
        action="store_true",
        help="Do not register the covariate panel (target-only; *_cov methods will then be unavailable).",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Recompute even where a persisted file exists.")
    parser.add_argument(
        "--pace",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "Sleep this many seconds after each API-calling prediction (LLMP/agent "
            "methods) before starting the next. Politeness on a shared proxy quota, "
            "not a correctness requirement — conventional methods and cached/resumed "
            "origins are never paced. Default: 0 (no pacing)."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable INFO logging.")
    return parser


#: Registry method names that consume no model (one predictor regardless of --models)
#: and make no API call. Doubles as the ``--pace`` exemption set: conventional
#: methods have nothing to be polite about since no request leaves the process.
_MODEL_FREE_METHODS: frozenset[str] = frozenset(CONVENTIONAL_METHODS + TSX_CONVENTIONAL_METHODS)


def _spec_is_tsx(spec: MultiTargetBacktestSpec) -> bool:
    """Return whether *spec* targets the S&P/TSX Composite (vs the S&P 500)."""
    return any(task.target_series_id.startswith("tsx_") for task in spec.tasks)


def _build_predictor_list(
    methods: list[str], models: list[str], covariate_panel: list[str]
) -> tuple[list[Predictor], frozenset[str]]:
    """Instantiate predictors: conventional once; model-parameterised once per model.

    Returns
    -------
    tuple[list[Predictor], frozenset[str]]
        The predictor list, and the subset of their ``predictor_id``s built
        from a conventional (no-API) method — used to exempt those runs from
        ``--pace``.
    """
    predictors: list[Predictor] = []
    seen_ids: set[str] = set()
    conventional_ids: set[str] = set()
    for method in methods:
        method_models = [models[0]] if method in _MODEL_FREE_METHODS else models
        for model in method_models:
            predictor = build_predictor(method, model=model, covariate_panel=covariate_panel)
            if predictor.predictor_id in seen_ids:
                continue
            seen_ids.add(predictor.predictor_id)
            predictors.append(predictor)
            if method in _MODEL_FREE_METHODS:
                conventional_ids.add(predictor.predictor_id)
    return predictors, frozenset(conventional_ids)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``ws-run-backtest`` console script."""
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    methods = resolve_methods(args.methods)
    spec = load_spec(args.spec)
    n_origins = origin_count(spec)
    print(f"Spec {spec.spec_id}: {len(spec.tasks)} tasks x {n_origins} candidate origins (stride={spec.stride}).")

    # Select the data service + covariate panel from the spec's target family so
    # a single CLI serves both the S&P 500 and the S&P/TSX Composite experiments.
    is_tsx = _spec_is_tsx(spec)
    full_panel = TSX_COVARIATE_PANEL if is_tsx else SP500_COVARIATE_PANEL
    if is_tsx:
        data_service = build_tsx_workshop_service(include_covariates=not args.no_covariates, end=args.end)
    else:
        data_service = build_workshop_service(include_covariates=not args.no_covariates, end=args.end)

    # Filter the covariate panel to series the service actually registered —
    # some covariates (e.g. StatCan macro when the WDS API is unreachable) can be
    # unavailable upstream and are skipped at registration; predictors must not
    # reference them.
    registered = set(data_service.series_ids)
    covariate_panel = [c for c in full_panel if c in registered]
    dropped = [c for c in full_panel if c not in registered]
    if dropped and not args.no_covariates:
        print(f"Covariates unavailable and skipped: {', '.join(dropped)}")

    predictors, conventional_predictor_ids = _build_predictor_list(methods, args.models, covariate_panel)
    print("Predictors:")
    for predictor in predictors:
        print(f"  - {predictor.predictor_id}")
    accounting = run_spec(
        predictors,
        spec,
        data_service,
        store_dir=Path(args.store_dir),
        force_refresh=args.force_refresh,
        inter_prediction_delay_s=args.pace,
        conventional_predictor_ids=conventional_predictor_ids,
    )

    print("\nRun accounting:")
    for predictor_id, acc in accounting.items():
        print(
            f"  {predictor_id}: predicted={acc.n_predicted} cached={acc.n_cached} "
            f"warmup_skipped={acc.n_skipped_warmup} failed={acc.n_failed} "
            f"wall={acc.wall_time_s:.1f}s cost=${acc.cost_usd:.4f} "
            f"tokens_in={acc.input_tokens} tokens_out={acc.output_tokens}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
