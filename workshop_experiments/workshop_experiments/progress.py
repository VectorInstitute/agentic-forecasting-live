"""Compact experiment-progress monitor for retrospective runs.

Reads the persisted prediction store for a spec and prints one line per
predictor: done/expected, percent, mean wall time, and a naive ETA for the
remaining predictions. Pure filesystem — no API calls, safe to run (or
``--watch``) alongside an active ``ws-run-backtest``.

Usage::

    uv run python -m workshop_experiments.progress --spec tsx_ws_backtest_2025_weekly
    uv run python -m workshop_experiments.progress --spec tsx_ws_smoke --watch 60

Designed for a tmux side pane: ``--watch N`` refreshes every N seconds.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from workshop_experiments.runner import DEFAULT_STORE_DIR
from workshop_experiments.specs import load_spec, origin_count


def _wall_times(predictor_dir: Path) -> list[float]:
    times: list[float] = []
    for f in predictor_dir.glob("*/*.yaml"):
        try:
            meta = yaml.safe_load(f.read_text()) or {}
            wall = meta.get("wall_time_s")
            if isinstance(wall, (int, float)):
                times.append(float(wall))
        except Exception:  # noqa: BLE001 - a corrupt file must not kill the monitor
            continue
    return times


def render(spec_name: str, store_dir: Path) -> str:
    """Render one progress snapshot as a plain-text table."""
    spec = load_spec(spec_name)
    expected = origin_count(spec) * len(spec.tasks)
    spec_dir = store_dir / spec.spec_id
    lines = [
        f"{spec.spec_id} — {datetime.now(timezone.utc).strftime('%H:%M:%S')}Z  (expected {expected} predictions/rung)",
        f"{'predictor':<58} {'done':>9} {'pct':>5} {'avg':>7} {'eta':>8}",
    ]
    if not spec_dir.is_dir():
        lines.append("  (no predictions yet)")
        return "\n".join(lines)
    total_done = total_expected = 0
    for pdir in sorted(spec_dir.iterdir()):
        if not pdir.is_dir():
            continue
        done = sum(1 for _ in pdir.glob("*/*.yaml"))
        walls = _wall_times(pdir)
        avg = sum(walls) / len(walls) if walls else 0.0
        remaining = max(expected - done, 0)
        eta_s = remaining * avg
        eta = f"{eta_s / 3600:.1f}h" if eta_s >= 5400 else f"{eta_s / 60:.0f}m"  # noqa: PLR2004
        marker = "✓" if done >= expected else " "
        lines.append(
            f"{marker}{pdir.name:<57} {done:>5}/{expected:<3} {100 * done // expected:>4}% {avg:>6.0f}s {eta:>8}"
        )
        total_done += min(done, expected)
        total_expected += expected
    if total_expected:
        lines.append(f"{'TOTAL':<58} {total_done:>5}/{total_expected:<3} {100 * total_done // total_expected:>4}%")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m workshop_experiments.progress``."""
    parser = argparse.ArgumentParser(description="Progress monitor for persisted prediction stores.")
    parser.add_argument("--spec", required=True, help="Spec name or YAML path.")
    parser.add_argument("--store-dir", default=str(DEFAULT_STORE_DIR), help="Prediction store root.")
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS", help="Refresh every N seconds (0 = once).")
    args = parser.parse_args(argv)
    store = Path(args.store_dir)
    while True:
        output = render(args.spec, store)
        if args.watch:
            print("\033[2J\033[H", end="")  # clear screen for watch mode
        print(output, flush=True)
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
