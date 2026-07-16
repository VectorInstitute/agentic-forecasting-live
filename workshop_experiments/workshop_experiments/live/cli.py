"""``ws-live-run`` — the umbrella CLI for one live trading-day harness run.

Pipeline: trading-day check -> predict -> resolve -> aggregate -> commit + push,
guarded by a single-run lockfile.

Usage
-----
- ``ws-live-run --simulate --no-push`` — offline end-to-end exercise (no API, no
  network), using the latest committed smoke origin.
- ``ws-live-run --dry-run`` — print the plan and exit before any write or call.
- ``ws-live-run --no-push`` — production run (real data + model calls), commit
  but do not push.

``--simulate`` replaces the real data service and model calls with the committed
smoke predictions; ``--dry-run`` prints the plan and exits before any write or
call; ``--no-push`` stops after the local commit.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from workshop_experiments.live.aggregate import aggregate_step
from workshop_experiments.live.config import LiveConfig, load_config
from workshop_experiments.live.gitops import commit_message, push, stage_and_commit
from workshop_experiments.live.lockfile import LockHeld, RunLock
from workshop_experiments.live.predict import (
    RealPredictionSource,
    SimulatePredictionSource,
    predict_step,
)
from workshop_experiments.live.records import utc_now_z
from workshop_experiments.live.resolve import (
    DataServiceRealizedProvider,
    LookupRealizedProvider,
    resolve_log,
)
from workshop_experiments.live.simulate import build_realized_lookup, latest_smoke_origin


logger = logging.getLogger(__name__)

#: US-Eastern: the origin is the NYSE regular-session close date.
_NY_TZ = ZoneInfo("America/New_York")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one live daily S&P 500 forecasting harness cycle.")
    parser.add_argument("--config", default=None, help="Path to live_config.yaml (default: the shipped config).")
    parser.add_argument(
        "--simulate", action="store_true", help="Offline: use committed smoke predictions, no API/network."
    )
    parser.add_argument("--origin", default=None, help="Override the origin date (YYYY-MM-DD); simulate mode only.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan and exit; no writes, no API, no network."
    )
    parser.add_argument("--no-push", action="store_true", help="Commit locally but do not push to the live remote.")
    parser.add_argument("--remote", default="live", help="Git remote to push the daily commit to (default: live).")
    parser.add_argument("--branch", default="main", help="Remote branch to push to (default: main).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable INFO logging.")
    return parser


def _today_eastern() -> date:
    """Return today's date in US/Eastern (the market-close reference day)."""
    return datetime.now(tz=_NY_TZ).date()


def _print_plan(config: LiveConfig, origin: date, mode: str, log_dir: Path, out_dir: Path) -> None:
    """Print the run plan for ``--dry-run`` (no side effects)."""
    print(f"ws-live-run plan ({mode})")
    print(f"  origin           : {origin}")
    print(f"  horizons         : {list(config.horizons)}")
    print(f"  submission       : {config.submission_time_local} {config.timezone}")
    print(
        f"  retry            : max={config.retry.max_attempts} backoff={config.retry.backoff_minutes}m stop={config.retry.hard_stop_local}"
    )
    print(f"  configured rungs : {len(config.predictors)}")
    for group in ("conventional", "llmp", "agent"):
        rungs = config.by_group(group)
        print(
            f"    - {group:12s}: {len(rungs)} ({', '.join(p.predictor_id for p in rungs[:3])}{', ...' if len(rungs) > 3 else ''})"
        )
    print(f"  log dir          : {log_dir}")
    print(f"  aggregates dir   : {out_dir}")


def _latest_close_date(data_service: object, series_id: str) -> date:
    """Return the latest available close date for a target series (real mode)."""
    import pandas as pd  # noqa: PLC0415

    now = datetime.now(tz=ZoneInfo("UTC")).replace(tzinfo=None)
    frame = data_service.get_series(series_id, as_of=now)  # type: ignore[attr-defined]
    return pd.to_datetime(frame["timestamp"]).max().date()


def _run_simulate(config: LiveConfig, origin: date, log_dir: Path, out_dir: Path) -> tuple[int, int, str]:
    """Predict (offline) -> resolve -> aggregate for one simulated origin.

    Returns ``(n_methods, n_resolutions, submitted_at)`` — the UTC submission
    timestamp is surfaced so the commit subject can carry it (see
    :func:`~workshop_experiments.live.gitops.commit_message`).
    """
    submitted_at = utc_now_z()
    source = SimulatePredictionSource(config=config, origin=origin)
    result = predict_step(config, source, log_dir=log_dir, submission_timestamp=submitted_at)
    provider = LookupRealizedProvider(build_realized_lookup(config))
    resolutions = resolve_log(log_dir, provider, resolved_at=submitted_at)
    aggregate_step(log_dir, out_dir)
    print(f"predict: wrote {len(result.written)}, gapped {len(result.gapped)}, skipped {len(result.skipped)}")
    print(f"resolve: {len(resolutions)} new resolutions")
    return len(result.written), len(resolutions), submitted_at


def _run_real(config: LiveConfig, log_dir: Path, out_dir: Path) -> tuple[int, int, str] | None:
    """Real run: trading-day check, then predict -> resolve -> aggregate.

    Returns ``(n_methods, n_resolutions, submitted_at)``, or ``None`` when today
    is not a trading session (clean no-op exit). The UTC submission timestamp is
    surfaced so the commit subject can carry it (see
    :func:`~workshop_experiments.live.gitops.commit_message`).
    """
    from workshop_experiments.data import SP500_COVARIATE_PANEL, build_workshop_service  # noqa: PLC0415

    data_service = build_workshop_service(include_covariates=True)
    target_series = config.task_id_for_horizon(config.horizons[0])
    latest_close = _latest_close_date(data_service, target_series)
    today = _today_eastern()
    if latest_close != today:
        print(f"non-session day: latest close {latest_close} != today {today}; exiting cleanly (not a gap).")
        return None

    registered = set(data_service.series_ids)  # type: ignore[attr-defined]
    covariate_panel = [c for c in SP500_COVARIATE_PANEL if c in registered]
    submitted_at = utc_now_z()
    source = RealPredictionSource(
        config=config, data_service=data_service, origin=latest_close, covariate_panel=covariate_panel
    )
    result = predict_step(config, source, log_dir=log_dir, submission_timestamp=submitted_at)
    provider = DataServiceRealizedProvider(data_service, config.task_id_for_horizon)
    resolutions = resolve_log(log_dir, provider, resolved_at=submitted_at)
    aggregate_step(log_dir, out_dir)
    print(f"predict: wrote {len(result.written)}, gapped {len(result.gapped)}, skipped {len(result.skipped)}")
    print(f"resolve: {len(resolutions)} new resolutions")
    return len(result.written), len(resolutions), submitted_at


def run(argv: list[str] | None = None) -> int:  # noqa: PLR0911 - linear step gating with early exits
    """Entry point for the ``ws-live-run`` console script."""
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    log_dir = config.log_dir
    out_dir = config.aggregates_dir

    if args.origin and not args.simulate:
        print("--origin is only valid with --simulate", file=sys.stderr)
        return 2

    origin = (
        date.fromisoformat(args.origin)
        if args.origin
        else (latest_smoke_origin(config) if args.simulate else _today_eastern())
    )

    if args.dry_run:
        _print_plan(config, origin, "simulate" if args.simulate else "real", log_dir, out_dir)
        return 0

    lock = RunLock(log_dir.parent / ".ws-live-run.lock")
    try:
        lock.acquire()
    except LockHeld as exc:
        print(f"another run is in progress: {exc}", file=sys.stderr)
        return 1
    try:
        if args.simulate:
            n_methods, n_resolutions, submitted_at = _run_simulate(config, origin, log_dir, out_dir)
        else:
            outcome = _run_real(config, log_dir, out_dir)
            if outcome is None:
                return 0
            n_methods, n_resolutions, submitted_at = outcome

        message = commit_message(origin.isoformat(), n_methods, n_resolutions, submitted_at)
        created = stage_and_commit(_repo_root(config), [log_dir, out_dir], message)
        if not created:
            print("nothing to commit.")
            return 0
        print(f"committed: {message}")
        if args.no_push:
            print("--no-push: skipping push.")
            return 0
        push(_repo_root(config), remote=args.remote, branch=args.branch)
        print(f"pushed to {args.remote}/{args.branch}.")
        return 0
    finally:
        lock.release()


def _repo_root(config: LiveConfig) -> Path:
    """Return the repo root the git operations run in (log dir's repo)."""
    from workshop_experiments.live.config import REPO_ROOT  # noqa: PLC0415

    return REPO_ROOT


def main() -> None:
    """Console-script wrapper that exits with the run's status code."""
    sys.exit(run())


if __name__ == "__main__":
    main()
