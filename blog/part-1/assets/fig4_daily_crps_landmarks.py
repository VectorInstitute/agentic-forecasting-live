"""fig4: per-origin CRPS over time on the daily grid, landmark windows shaded.

Every business-day origin (2025-01-02 -> 2026-06-15) scored at h = 1 / 5 / 21 for
the free conventional methods plus the lite LLM-Process. The landmark windows are
shaded; the point is that every method's error spikes at the same moments -- they
are all blind to the same regime breaks.

lightgbm_cov is intentionally incomplete on the daily grid (h=1: 369 origins,
h=5: 226, h=21: none); it is drawn where present and flagged in the caption.

Run: ``python blog/part-1/assets/fig4_daily_crps_landmarks.py``
"""

from __future__ import annotations

import _blogdata as bd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


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
    for ax, target in zip(axes, bd.TARGETS):
        h = bd.HORIZON_OF[target]
        # Landmark shading.
        wash = {"drawdown": bd.STATUS["critical"], "rebound": bd.STATUS["good"]}
        for lm in lms:
            ax.axvspan(lm["start"], lm["end"], color=wash[lm["kind"]], alpha=0.09, lw=0, zorder=0)
        for model, label, color, ls in SERIES:
            s = bd.crps_series(SPEC, model, target)
            if s.empty:
                continue
            ax.plot(s.index, s.values, color=color, lw=1.1, ls=ls, alpha=0.85, label=label, zorder=3)
        ax.set_ylabel(f"CRPS  (h={h})")
        ax.margins(x=0.01)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "Every method goes blind at the same moments: daily CRPS through the landmark windows",
        fontsize=13,
        fontweight="bold",
        x=0.125,
        ha="left",
        y=0.975,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.125, 0.945),
        ncol=5,
        fontsize=9,
        columnspacing=1.4,
        handlelength=1.8,
    )
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    fig.text(
        0.125,
        0.035,
        "Source: daily prediction store (tsx_ws_daily_2025_2026), CRPS per origin via "
        "properscoring.crps_ensemble. Shaded: TSX landmark windows (red = drawdown, green = rebound; "
        "identities in fig. 1). LightGBM +cov is incomplete on the daily grid and absent at h=21.",
        fontsize=7.5,
        color=bd.INK["muted"],
        ha="left",
    )

    fig.subplots_adjust(top=0.9, bottom=0.1, hspace=0.16)
    out = bd.savefig(fig, "fig4_daily_crps_landmarks.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
