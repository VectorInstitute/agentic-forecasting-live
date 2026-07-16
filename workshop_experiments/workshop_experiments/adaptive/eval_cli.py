"""``ws-adaptive-eval`` — retrospective before/after eval CLI.

Runs the three frozen strategy arms (untrained / phase_a / phase_ab) across the
2026 protected weekly grid via the resumable runner. **Dry-run by default**;
pass ``--run`` to make model + E2B calls. Scoring the persisted predictions is a
separate offline step (``ws-score --spec sp500_ws_eval_2026_weekly``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aieng.forecasting.models import ADVANCED_MODEL

from workshop_experiments.adaptive.evaluate import plan_adaptive_eval, run_adaptive_eval
from workshop_experiments.runner import DEFAULT_STORE_DIR


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrospective before/after adaptive-strategy eval.")
    parser.add_argument("--spec", default="sp500_ws_eval_2026_weekly", help="Eval spec name.")
    parser.add_argument("--model", default=ADVANCED_MODEL, help="Agent model (default: the advanced model).")
    parser.add_argument("--store-dir", default=str(DEFAULT_STORE_DIR), help="Prediction store root.")
    parser.add_argument(
        "--arm",
        action="append",
        default=None,
        metavar="NAME=DIR",
        help="Override an arm's strategy dir (repeatable), e.g. phase_ab=/path/to/sp500-strategy-trained.",
    )
    parser.add_argument("--run", action="store_true", help="Actually run the arms (makes model/E2B calls).")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit (the default).")
    return parser


def _parse_arm_overrides(items: list[str] | None) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--arm expects NAME=DIR, got {item!r}")
        name, _, path = item.partition("=")
        overrides[name.strip()] = Path(path.strip())
    return overrides


def run(argv: list[str] | None = None) -> int:
    """Entry point for the ``ws-adaptive-eval`` console script."""
    args = _build_arg_parser().parse_args(argv)
    overrides = _parse_arm_overrides(args.arm)
    plan = plan_adaptive_eval(
        spec_name=args.spec,
        arms={**dict(plan_adaptive_eval().arms), **overrides} if overrides else None,
        model=args.model,
        store_dir=Path(args.store_dir),
    )

    if not args.run:
        print(plan.describe())
        return 0

    from workshop_experiments.data import build_workshop_service  # noqa: PLC0415

    data_service = build_workshop_service(include_covariates=True)
    results = run_adaptive_eval(plan, data_service)
    for arm, acc in results.items():
        print(f"{arm:10s}: predicted {acc.n_predicted}, cached {acc.n_cached}, cost ${acc.cost_usd:.4f}")
    print(f"Score with: ws-score --spec {plan.spec_name}")
    return 0


def main() -> None:
    """Console-script wrapper that exits with the run's status code."""
    sys.exit(run())


if __name__ == "__main__":
    main()
