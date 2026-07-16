"""CLI: score persisted workshop predictions into leaderboard artifacts.

Reads the per-origin predictions written by ``ws-run-backtest``, CRPS-scores
them, and writes ``leaderboard.csv`` / ``leaderboard.md`` under
``data/results/<spec_id>/``. Never calls a model API.

Examples
--------
::

    ws-score --spec sp500_ws_smoke
    python -m workshop_experiments.score --spec sp500_ws_backtest_2025_weekly
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from workshop_experiments.data import build_workshop_service
from workshop_experiments.data_tsx import build_tsx_workshop_service
from workshop_experiments.runner import DEFAULT_STORE_DIR
from workshop_experiments.scoring import (
    DEFAULT_RESULTS_DIR,
    score_spec,
    write_leaderboard_artifacts,
)
from workshop_experiments.specs import load_spec


logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score persisted workshop predictions into a leaderboard.")
    parser.add_argument("--spec", required=True, help="Spec name (e.g. sp500_ws_smoke) or path to a YAML file.")
    parser.add_argument("--store-dir", default=str(DEFAULT_STORE_DIR), help="Prediction store root to read from.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Where to write leaderboard artifacts.")
    parser.add_argument("--end", default=None, help="Optional data-fetch end (YYYY-MM-DD, exclusive).")
    parser.add_argument(
        "--no-covariates",
        action="store_true",
        help="Build the data service without the covariate panel (targets resolve either way).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable INFO logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``ws-score`` console script."""
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    spec = load_spec(args.spec)
    is_tsx = any(task.target_series_id.startswith("tsx_") for task in spec.tasks)
    build_service = build_tsx_workshop_service if is_tsx else build_workshop_service
    data_service = build_service(include_covariates=not args.no_covariates, end=args.end)

    frame, _results = score_spec(spec, data_service, store_dir=Path(args.store_dir))
    paths = write_leaderboard_artifacts(frame, spec.spec_id, results_dir=Path(args.results_dir))

    if frame.empty:
        print(f"No scored predictions found for spec {spec.spec_id}.")
    else:
        print(f"Scored {len(frame)} predictor x horizon rows for spec {spec.spec_id}:\n")
        columns = [c for c in ("model", "horizon", "mean_crps", "n_scores", "dir_accuracy") if c in frame.columns]
        print(frame[columns].to_string(index=False))
    print(f"\nWrote:\n  {paths['csv']}\n  {paths['markdown']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
