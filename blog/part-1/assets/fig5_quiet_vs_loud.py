"""fig5: the market is loud, the forecasts are quiet.

Median h=1 forecasts vs the realised next-day return over the 2025 tariff
drawdown-and-rebound stretch. The realised series swings several percent a day;
the forecast medians (a conventional model and the lite LLM-Process) barely leave
zero -- efficiency made visual. Both are aligned on the day being predicted
(forecast_date).

Run: ``python blog/part-1/assets/fig5_quiet_vs_loud.py``
"""

from __future__ import annotations

import _blogdata as bd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


SPEC = "tsx_ws_daily_2025_2026"
WIN_START = "2025-02-01"
WIN_END = "2025-06-30"

FORECASTS = [
    ("darts_lightgbm", "LightGBM median", bd.CAT["blue"]),
    ("llmp_quantile_grid_tsx_ws[gemini-3.1-flash-lite-preview]", "LLMP flash-lite median", bd.CAT["violet"]),
]


def _median_by_forecast_date(model: str) -> "object":
    preds = bd.load_predictions(SPEC, model, "tsx_logret_1b")
    s = preds.reset_index().set_index("forecast_date")["median"].sort_index()
    return s[WIN_START:WIN_END]


def main() -> None:
    bd.apply_style()

    realized = bd.realized("tsx_logret_1b")[WIN_START:WIN_END]
    fig, ax = plt.subplots(figsize=(11.5, 5.4))

    # Landmark shading for context.
    for lm in bd.landmarks():
        if lm["end"] < realized.index.min() or lm["start"] > realized.index.max():
            continue
        wash = {"drawdown": bd.STATUS["critical"], "rebound": bd.STATUS["good"]}[lm["kind"]]
        ax.axvspan(lm["start"], lm["end"], color=wash, alpha=0.08, lw=0, zorder=0)

    # Realised return: loud vertical stems.
    ax.bar(
        realized.index,
        realized.values,
        width=1.6,
        color=bd.INK["muted"],
        alpha=0.55,
        label="Realised next-day return",
        zorder=2,
    )

    med_ranges = []
    for model, label, color in FORECASTS:
        s = _median_by_forecast_date(model)
        ax.plot(s.index, s.values, color=color, lw=1.7, label=label, zorder=4)
        med_ranges.append((s.min(), s.max()))

    ax.axhline(0, color=bd.INK["axis"], lw=1.0, zorder=1)

    r_lo, r_hi = realized.min() * 100, realized.max() * 100
    m_lo = min(m[0] for m in med_ranges) * 100
    m_hi = max(m[1] for m in med_ranges) * 100

    bd.figure_title(ax, 6, "Realised next-day returns vs. forecast medians (h = 1)")
    ax.set_ylabel("Next-day log return")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.margins(x=0.01)
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Patch(facecolor=bd.STATUS["critical"], alpha=0.35, lw=0),
        Patch(facecolor=bd.STATUS["good"], alpha=0.35, lw=0),
    ]
    labels += ["Drawdown", "Rebound"]
    ax.legend(handles, labels, loc="lower right")

    out = bd.savefig(fig, "fig5_quiet_vs_loud.png")
    print(
        f"CAPTION: The realised next-day return swung between {r_lo:+.1f}% and {r_hi:+.1f}% over the "
        f"window, while both forecast medians stayed within {m_lo:+.2f}% to {m_hi:+.2f}%.",
    )
    print(
        "CAPTION: Median h = 1 forecasts vs. the realised next-day return, aligned on the predicted "
        "day, from the daily prediction store (tsx_ws_daily_2025_2026). Shaded bands are the 2025 "
        "tariff drawdown (red) and rebound (green).",
    )
    print(f"wrote {out}  realised [{r_lo:.2f},{r_hi:.2f}]%  medians [{m_lo:.3f},{m_hi:.3f}]%")


if __name__ == "__main__":
    main()
