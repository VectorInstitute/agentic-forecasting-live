"""``ws-study`` — the Study Hall / Residency driver CLI (non-notebook, resumable).

Two subcommands:

- ``ws-study phase-a`` — Phase A (Study Hall) over pre-2025 history.
- ``ws-study phase-b`` — Phase B (Residency) postmortems over selected 2025 origins.

**Dry-run by default** (nothing spends): prints the resolved plan and exits.
Pass ``--run`` to actually drive the study (this makes model + E2B calls). A
``--run`` invocation RESEEDs the trained strategy from the seed first unless
``--no-reseed`` is given (RESEED never mutates the seed), then drives the phase
over a sticky session, persisting transcript + accounting per turn so an
interrupted run resumes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aieng.forecasting.models import ADVANCED_MODEL

from workshop_experiments.adaptive.domain import SEED_STRATEGY_DIR, TRAINED_STRATEGY_DIR
from workshop_experiments.adaptive.origins import parse_origins_override
from workshop_experiments.adaptive.study import _prompt_for_turn


_DEFAULT_RUN_DIR = TRAINED_STRATEGY_DIR.parent.parent / "study_runs"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive S&P 500 Study Hall / Residency driver.")
    sub = parser.add_subparsers(dest="phase", required=True)

    a = sub.add_parser("phase-a", help="Study Hall over pre-2025 history.")
    a.add_argument("--turn-budget", type=int, default=50, help="Total Study Hall turns (default 50).")
    a.add_argument("--checkpoint-every", type=int, default=10, help="Distill-checkpoint cadence (default 10).")

    b = sub.add_parser("phase-b", help="Residency postmortems over selected 2025 origins.")
    b.add_argument("--spec", default="sp500_ws_backtest_2025_weekly", help="Backtest spec to select origins from.")
    b.add_argument("--predictor-id", default=None, help="Predictor whose CRPS ranks the origins.")
    b.add_argument("--worst-n", type=int, default=12, help="Worst-CRPS origins to postmortem (default 12).")
    b.add_argument("--best-n", type=int, default=4, help="Best-CRPS control origins (default 4).")
    b.add_argument("--origins", default=None, help="Comma-separated YYYY-MM-DD override for the origin selection.")
    b.add_argument("--turns-per-postmortem", type=int, default=3, help="Bounded turns per postmortem (default 3).")

    for p in (a, b):
        p.add_argument("--model", default=ADVANCED_MODEL, help="Agent model (default: the advanced model).")
        p.add_argument("--run-dir", default=None, help="Where to persist transcript + state.")
        p.add_argument("--run", action="store_true", help="Actually drive the study (makes model/E2B calls).")
        p.add_argument("--no-reseed", action="store_true", help="Do NOT reseed the trained strategy before a --run.")
        p.add_argument("--dry-run", action="store_true", help="Print the plan and exit (the default).")
    return parser


def _print_phase_a_plan(args: argparse.Namespace, run_dir: Path) -> None:
    print("ws-study phase-a plan (Study Hall)")
    print(f"  seed strategy    : {SEED_STRATEGY_DIR}")
    print(f"  trained strategy : {TRAINED_STRATEGY_DIR}  (RESEEDed from seed unless --no-reseed)")
    print(f"  turn budget      : {args.turn_budget}")
    print(f"  checkpoint every : {args.checkpoint_every} turns")
    checkpoints = [
        t for t in range(1, args.turn_budget + 1) if _prompt_for_turn(t, args.checkpoint_every)[0] == "distill"
    ]
    print(f"  distill turns    : {checkpoints}")
    print(f"  run dir          : {run_dir}")
    print(f"  model            : {args.model}")
    print("  cost accounting  : per-turn tokens (est.) + wall time persisted; authoritative")
    print("                     token/USD from per-turn Langfuse traces at --run time.")
    print("  NOTE: dry-run — no model calls. Pass --run to drive the session.")


def _print_phase_b_plan(args: argparse.Namespace, run_dir: Path) -> None:
    print("ws-study phase-b plan (Residency)")
    print(f"  trained strategy    : {TRAINED_STRATEGY_DIR}")
    print(f"  spec                : {args.spec}")
    if args.origins:
        origins = parse_origins_override(args.origins.split(","))
        print(f"  origins (override)  : {[o.isoformat() for o in origins]}")
    else:
        print(f"  origin selection    : worst {args.worst_n} + best {args.best_n} by CRPS")
        print(f"                        predictor: {args.predictor_id or '(first in store)'} (computed at --run)")
    print(f"  turns/postmortem    : {args.turns_per_postmortem}")
    print(f"  run dir             : {run_dir}")
    print(f"  model               : {args.model}")
    print("  NOTE: dry-run — no model calls. Pass --run to drive the postmortems.")


def run(argv: list[str] | None = None) -> int:
    """Entry point for the ``ws-study`` console script."""
    args = _build_arg_parser().parse_args(argv)
    run_dir = Path(args.run_dir) if args.run_dir else (_DEFAULT_RUN_DIR / args.phase)

    if not args.run:
        if args.phase == "phase-a":
            _print_phase_a_plan(args, run_dir)
        else:
            _print_phase_b_plan(args, run_dir)
        return 0

    return _drive(args, run_dir)


def _drive(args: argparse.Namespace, run_dir: Path) -> int:
    """Spend-bearing path (constructs the real session). Not exercised offline."""
    from workshop_experiments.adaptive.domain import build_sp500_adaptive_config  # noqa: PLC0415
    from workshop_experiments.adaptive.reseed import reseed_strategy  # noqa: PLC0415
    from workshop_experiments.adaptive.session import AdkStudySession  # noqa: PLC0415

    if not args.no_reseed:
        reseed_strategy(force=True)
        print(f"RESEEDed {TRAINED_STRATEGY_DIR} from {SEED_STRATEGY_DIR}.")

    config = build_sp500_adaptive_config(model=args.model, strategy_dir=TRAINED_STRATEGY_DIR)
    session = AdkStudySession(
        config,
        app_name=f"sp500_study_{args.phase}",
        langfuse_tags=["sp500", "adaptive-agent", args.phase],
        trace_name=f"sp500-adaptive-{args.phase}",
    )
    try:
        if args.phase == "phase-a":
            from workshop_experiments.adaptive.study import run_study_hall  # noqa: PLC0415

            result = run_study_hall(
                session, run_dir, turn_budget=args.turn_budget, checkpoint_every=args.checkpoint_every
            )
        else:
            result = _drive_phase_b(args, run_dir, session)
        print(
            f"{result.phase}: ran {result.turns_run} turns "
            f"({result.accounting.input_tokens + result.accounting.output_tokens} est. tokens, "
            f"{result.accounting.wall_time_s:.1f}s)."
        )
        return 0
    finally:
        session.close()


def _drive_phase_b(args: argparse.Namespace, run_dir: Path, session: object) -> object:
    """Select origins, build postmortems, and drive Residency (spend-bearing)."""
    from workshop_experiments.adaptive.study import Postmortem, run_residency  # noqa: PLC0415

    if not args.origins:
        raise SystemExit(
            "phase-b --run currently requires --origins (comma-separated YYYY-MM-DD); "
            "CRPS-based auto-selection needs scored backtest artifacts. Pass the origins "
            "from `ws-score --spec {spec}` output."
        )
    origins = parse_origins_override(args.origins.split(","))
    # Minimal postmortems: the agent retrieves its own committed forecast/realized
    # context via code + search at run time; the driver supplies the origin bounds.
    postmortems = [
        Postmortem(origin=o, committed_forecast="(retrieve from the live log)", realized="(compute at run)", crps="")
        for o in origins
    ]
    return run_residency(session, run_dir, postmortems, turns_per_postmortem=args.turns_per_postmortem)  # type: ignore[arg-type]


def main() -> None:
    """Console-script wrapper that exits with the run's status code."""
    sys.exit(run())


if __name__ == "__main__":
    main()
