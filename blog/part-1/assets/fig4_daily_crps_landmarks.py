"""fig4: per-origin CRPS over time on the daily grid, landmark windows shaded.

Every business-day origin (2025-01-02 -> 2026-06-15) scored at h = 1 / 5 / 21 for
the free conventional methods plus the lite LLM-Process. The landmark windows are
shaded; the point is that every method's error spikes at the same moments -- they
are all blind to the same regime breaks.

All five methods now resolve on the full daily grid (h=1/5/21 ~= 365/365/364
origins each), including lightgbm_cov, which was backfilled in the refreshed
store; every series is drawn across all three horizons.

Run: ``python blog/part-1/assets/fig4_daily_crps_landmarks.py``
"""

from __future__ import annotations

import _blogdata as bd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


SPEC = "tsx_ws_daily_2025_2026"

# (model dir, label, color, linestyle)
SERIES = [
    ("last_value_naive", "Naive", bd.INK["muted"], "-"),
    ("darts_autoarima", "AutoARIMA", bd.CAT["aqua"], "-"),
    ("darts_lightgbm", "LightGBM", bd.CAT["blue"], "-"),
    ("darts_lightgbm_cov", "LightGBM +cov", bd.CAT["blue"], (0, (3, 2))),
    ("llmp_quantile_grid_tsx_ws[gemini-3.1-flash-lite-preview]", "LLMP flash-lite", bd.CAT["violet"], "-"),
]


def main() -> None:
    bd.apply_style()
    lms = bd.landmarks()

    fig, axes = plt.subplots(3, 1, figsize=(12, 9.2), sharex=True)
    n_origins = {}  # horizon -> {model label: resolved-origin count}
    for ax, target in zip(axes, bd.TARGETS):
        h = bd.HORIZON_OF[target]
        # Landmark shading.
        wash = {"drawdown": bd.STATUS["critical"], "rebound": bd.STATUS["good"]}
        for lm in lms:
            ax.axvspan(lm["start"], lm["end"], color=wash[lm["kind"]], alpha=0.09, lw=0, zorder=0)
        counts = {}
        for model, label, color, ls in SERIES:
            s = bd.crps_series(SPEC, model, target)
            if s.empty:
                continue
            counts[label] = int(s.size)
            ax.plot(s.index, s.values, color=color, lw=1.1, ls=ls, alpha=0.85, label=label, zorder=3)
        n_origins[h] = counts
        ax.set_ylabel(f"CRPS  (h={h})")
        ax.margins(x=0.01)
        ax.set_ylim(bottom=0)

    bd.figure_title(fig, 5, "Daily CRPS per origin through the landmark windows", x=0.125, y=0.98)
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [
        Patch(facecolor=bd.STATUS["critical"], alpha=0.35, lw=0),
        Patch(facecolor=bd.STATUS["good"], alpha=0.35, lw=0),
    ]
    labels += ["Drawdown", "Rebound"]
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.125, 0.945),
        ncol=7,
        columnspacing=1.2,
        handlelength=1.5,
    )
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    fig.subplots_adjust(top=0.895, bottom=0.075, hspace=0.16)
    out = bd.savefig(fig, "fig4_daily_crps_landmarks.png")
    n_by_h = {h: (min(c.values()), max(c.values())) for h, c in n_origins.items() if c}
    n_str = ", ".join(
        f"h={h}: n={lo}" if lo == hi else f"h={h}: n={lo}–{hi}" for h, (lo, hi) in sorted(n_by_h.items())
    )
    print(
        "CAPTION: Per-origin CRPS on the daily grid from the daily prediction store "
        "(tsx_ws_daily_2025_2026), scored with properscoring.crps_ensemble. Shaded bands are the "
        "TSX landmark windows (red = drawdown, green = rebound; identities as in Figure 1). "
        f"All five methods resolve across the full daily grid ({n_str} origins per method).",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
