"""``ws-study`` — the Study Hall / Residency / bootcamp driver CLI (resumable).

Invocation shapes:

- ``ws-study --domain tsx`` (no subcommand) — the **single-session bootcamp
  study**: one self-directed session over the full pre-2026 history *and* a review
  of the 2025 period, distilled into the strategy via the gated mutation tools.
  This is the whole demonstration study for the TSX pre/post shape.
- ``ws-study phase-a`` — S&P 500 Phase A (Study Hall) over pre-2025 history.
- ``ws-study phase-b`` — S&P 500 Phase B (Residency) postmortems over 2025 origins.

Both phases and the bootcamp study share the same driver machinery (scheduling,
checkpointing, per-turn transcript + accounting); only the seed/trained dirs, the
branded strategy state, the adaptive config factory, and the turn prompts differ
per ``--domain``.

**Dry-run by default** (nothing spends): prints the resolved plan and exits.
Pass ``--run`` to actually drive the study (this makes model + E2B calls). A
``--run`` invocation RESEEDs the trained strategy from the seed first unless
``--no-reseed`` is given (RESEED never mutates the seed), then drives over a
sticky session, persisting transcript + accounting per turn so an interrupted run
resumes.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from aieng.forecasting.models import ADVANCED_MODEL

from workshop_experiments.adaptive.origins import parse_origins_override
from workshop_experiments.adaptive.study import StudyHallPrompts, _prompt_for_turn


@dataclass(frozen=True)
class _DomainStudyConfig:
    """Per-domain wiring the single-session / Phase-A driver needs."""

    key: str
    label: str
    seed_dir: Path
    trained_dir: Path
    prompts: StudyHallPrompts
    config_builder: Callable[..., object]
    reseed: Callable[..., Path]


def _sp500_study_config() -> _DomainStudyConfig:
    from workshop_experiments.adaptive.domain import (  # noqa: PLC0415
        SEED_STRATEGY_DIR,
        TRAINED_STRATEGY_DIR,
        build_sp500_adaptive_config,
    )
    from workshop_experiments.adaptive.reseed import reseed_strategy  # noqa: PLC0415

    return _DomainStudyConfig(
        key="sp500",
        label="S&P 500",
        seed_dir=SEED_STRATEGY_DIR,
        trained_dir=TRAINED_STRATEGY_DIR,
        prompts=StudyHallPrompts(),  # the sp500 defaults
        config_builder=build_sp500_adaptive_config,
        reseed=lambda force: reseed_strategy(force=force),
    )


def _tsx_study_config() -> _DomainStudyConfig:
    from workshop_experiments.adaptive.domain_tsx import (  # noqa: PLC0415
        TSX_CONTINUE_PROMPT,
        TSX_DISTILL_PROMPT,
        TSX_SEED_STRATEGY_DIR,
        TSX_STUDY_HALL_PROMPT,
        TSX_TRAINED_STRATEGY_DIR,
        build_tsx_adaptive_config,
    )
    from workshop_experiments.adaptive.reseed import reseed_tsx_strategy  # noqa: PLC0415

    return _DomainStudyConfig(
        key="tsx",
        label="S&P/TSX Composite",
        seed_dir=TSX_SEED_STRATEGY_DIR,
        trained_dir=TSX_TRAINED_STRATEGY_DIR,
        prompts=StudyHallPrompts(
            study_hall=TSX_STUDY_HALL_PROMPT, cont=TSX_CONTINUE_PROMPT, distill=TSX_DISTILL_PROMPT
        ),
        config_builder=build_tsx_adaptive_config,
        reseed=lambda force: reseed_tsx_strategy(force=force),
    )


#: Domain key -> lazy config factory (imports are deferred so a dry-run of one
#: domain never imports the other's data/domain modules).
_DOMAINS: dict[str, Callable[[], _DomainStudyConfig]] = {
    "sp500": _sp500_study_config,
    "tsx": _tsx_study_config,
}


def _default_run_dir(domain: str, phase: str) -> Path:
    """Default run dir, rooted next to the domain's skills package."""
    dc = _DOMAINS[domain]()
    return dc.trained_dir.parent.parent / "study_runs" / f"{domain}_{phase}"


def _add_study_hall_args(p: argparse.ArgumentParser) -> None:
    """Turn-budget / checkpoint args shared by the bootcamp study and Phase A."""
    p.add_argument("--turn-budget", type=int, default=50, help="Total study turns (default 50).")
    p.add_argument("--checkpoint-every", type=int, default=10, help="Distill-checkpoint cadence (default 10).")


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Args shared by every invocation shape."""
    p.add_argument("--domain", choices=sorted(_DOMAINS), default="sp500", help="Target domain (default sp500).")
    p.add_argument("--model", default=ADVANCED_MODEL, help="Agent model (default: the advanced model).")
    p.add_argument("--run-dir", default=None, help="Where to persist transcript + state.")
    p.add_argument("--run", action="store_true", help="Actually drive the study (makes model/E2B calls).")
    p.add_argument("--no-reseed", action="store_true", help="Do NOT reseed the trained strategy before a --run.")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit (the default).")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive Study Hall / Residency / bootcamp driver.")
    # Top-level args back the no-subcommand single-session bootcamp study.
    _add_common_args(parser)
    _add_study_hall_args(parser)

    sub = parser.add_subparsers(dest="phase", required=False)

    a = sub.add_parser("phase-a", help="Study Hall over pre-2025 history (S&P 500).")
    _add_common_args(a)
    _add_study_hall_args(a)

    b = sub.add_parser("phase-b", help="Residency postmortems over selected 2025 origins (S&P 500).")
    _add_common_args(b)
    b.add_argument("--spec", default="sp500_ws_backtest_2025_weekly", help="Backtest spec to select origins from.")
    b.add_argument("--predictor-id", default=None, help="Predictor whose CRPS ranks the origins.")
    b.add_argument("--worst-n", type=int, default=12, help="Worst-CRPS origins to postmortem (default 12).")
    b.add_argument("--best-n", type=int, default=4, help="Best-CRPS control origins (default 4).")
    b.add_argument("--origins", default=None, help="Comma-separated YYYY-MM-DD override for the origin selection.")
    b.add_argument("--turns-per-postmortem", type=int, default=3, help="Bounded turns per postmortem (default 3).")
    return parser


# ---------------------------------------------------------------------------
# Dry-run plan printers
# ---------------------------------------------------------------------------


def _print_study_hall_plan(args: argparse.Namespace, run_dir: Path, *, title: str) -> None:
    dc = _DOMAINS[args.domain]()
    print(title)
    print(f"  domain           : {dc.label} ({dc.key})")
    print(f"  seed strategy    : {dc.seed_dir}")
    print(f"  trained strategy : {dc.trained_dir}  (RESEEDed from seed unless --no-reseed)")
    print(f"  turn budget      : {args.turn_budget}")
    print(f"  checkpoint every : {args.checkpoint_every} turns")
    checkpoints = [
        t
        for t in range(1, args.turn_budget + 1)
        if _prompt_for_turn(t, args.checkpoint_every, dc.prompts)[0] == "distill"
    ]
    print(f"  distill turns    : {checkpoints}")
    print(f"  run dir          : {run_dir}")
    print(f"  model            : {args.model}")
    print("  cost accounting  : per-turn tokens (est.) + wall time persisted; authoritative")
    print("                     token/USD from per-turn Langfuse traces at --run time.")
    print("  NOTE: dry-run — no model calls. Pass --run to drive the session.")


def _print_phase_b_plan(args: argparse.Namespace, run_dir: Path) -> None:
    dc = _DOMAINS[args.domain]()
    print("ws-study phase-b plan (Residency)")
    print(f"  domain              : {dc.label} ({dc.key})")
    print(f"  trained strategy    : {dc.trained_dir}")
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(argv: list[str] | None = None) -> int:
    """Entry point for the ``ws-study`` console script."""
    args = _build_arg_parser().parse_args(argv)
    # No subcommand => the single-session bootcamp study (the whole demonstration).
    phase = args.phase or "study"
    run_dir = Path(args.run_dir) if args.run_dir else _default_run_dir(args.domain, phase)

    if not args.run:
        if phase == "study":
            _print_study_hall_plan(args, run_dir, title="ws-study bootcamp plan (single-session study)")
        elif phase == "phase-a":
            _print_study_hall_plan(args, run_dir, title="ws-study phase-a plan (Study Hall)")
        else:
            _print_phase_b_plan(args, run_dir)
        return 0

    if phase == "phase-b":
        return _drive_residency(args, run_dir)
    return _drive_study_hall(args, run_dir, phase=phase)


def _drive_study_hall(args: argparse.Namespace, run_dir: Path, *, phase: str) -> int:
    """Spend-bearing path for the single-session study / Phase A (real session)."""
    from workshop_experiments.adaptive.session import AdkStudySession  # noqa: PLC0415
    from workshop_experiments.adaptive.study import run_study_hall  # noqa: PLC0415

    dc = _DOMAINS[args.domain]()
    if not args.no_reseed:
        dc.reseed(True)
        print(f"RESEEDed {dc.trained_dir} from {dc.seed_dir}.")

    config = dc.config_builder(model=args.model, strategy_dir=dc.trained_dir)
    session = AdkStudySession(
        config,
        app_name=f"{dc.key}_study_{phase}",
        langfuse_tags=[dc.key, "adaptive-agent", phase],
        trace_name=f"{dc.key}-adaptive-{phase}",
    )
    try:
        result = run_study_hall(
            session,
            run_dir,
            turn_budget=args.turn_budget,
            checkpoint_every=args.checkpoint_every,
            prompts=dc.prompts,
        )
        print(
            f"{result.phase}: ran {result.turns_run} turns "
            f"({result.accounting.input_tokens + result.accounting.output_tokens} est. tokens, "
            f"{result.accounting.wall_time_s:.1f}s)."
        )
        return 0
    finally:
        session.close()


def _drive_residency(args: argparse.Namespace, run_dir: Path) -> int:
    """Spend-bearing S&P 500 Residency path (Phase B)."""
    from workshop_experiments.adaptive.session import AdkStudySession  # noqa: PLC0415
    from workshop_experiments.adaptive.study import Postmortem, run_residency  # noqa: PLC0415

    dc = _DOMAINS[args.domain]()
    if not args.no_reseed:
        dc.reseed(True)
        print(f"RESEEDed {dc.trained_dir} from {dc.seed_dir}.")

    if not args.origins:
        raise SystemExit(
            "phase-b --run currently requires --origins (comma-separated YYYY-MM-DD); "
            "CRPS-based auto-selection needs scored backtest artifacts. Pass the origins "
            "from `ws-score --spec {spec}` output."
        )
    origins = parse_origins_override(args.origins.split(","))
    config = dc.config_builder(model=args.model, strategy_dir=dc.trained_dir)
    session = AdkStudySession(
        config,
        app_name=f"{dc.key}_study_phase-b",
        langfuse_tags=[dc.key, "adaptive-agent", "phase-b"],
        trace_name=f"{dc.key}-adaptive-phase-b",
    )
    try:
        postmortems = [
            Postmortem(
                origin=o, committed_forecast="(retrieve from the live log)", realized="(compute at run)", crps=""
            )
            for o in origins
        ]
        result = run_residency(session, run_dir, postmortems, turns_per_postmortem=args.turns_per_postmortem)
        print(
            f"{result.phase}: ran {result.turns_run} turns "
            f"({result.accounting.input_tokens + result.accounting.output_tokens} est. tokens, "
            f"{result.accounting.wall_time_s:.1f}s)."
        )
        return 0
    finally:
        session.close()


def main() -> None:
    """Console-script wrapper that exits with the run's status code."""
    sys.exit(run())


if __name__ == "__main__":
    main()
