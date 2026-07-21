"""fig1: S&P/TSX Composite price path with landmark event windows annotated.

Level path over 2025-01 -> 2026-06 (Yahoo ``^GSPTSE`` adjusted close, the same
series the data service builds its returns from). Landmark windows are shaded
(drawdowns warm, rebounds cool) and labelled with the TSX percentage move
computed by :func:`_blogdata.landmarks`.

Run: ``python blog/part-1/assets/fig1_tsx_landmarks.py``
"""

from __future__ import annotations

import _blogdata as bd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def main() -> None:
    bd.apply_style()
    close = bd.tsx_close()
    close = close["2025-01-01":"2026-06-30"]
    lms = bd.landmarks()

    fig, ax = plt.subplots(figsize=(11, 5.2))

    # Landmark shading + annotation.
    wash = {"drawdown": bd.STATUS["critical"], "rebound": bd.STATUS["good"]}
    label_y = {"drawdown": 0.10, "rebound": 0.95}
    for lm in lms:
        ax.axvspan(lm["start"], lm["end"], color=wash[lm["kind"]], alpha=0.08, lw=0)
        mid = lm["start"] + (lm["end"] - lm["start"]) / 2
        ax.annotate(
            f"{lm['label']}\n{lm['pct']:+.1f}%",
            xy=(mid, ax.get_ylim()[0]),
            xycoords=("data", "axes fraction"),
            xytext=(mid, label_y[lm["kind"]]),
            textcoords=("data", "axes fraction"),
            ha="center",
            va="center",
            color=bd.INK["primary"],
            fontweight="bold",
            linespacing=1.3,
        )

    # Peak / trough reference dots.
    for lm in lms:
        for key in ("peak_date", "trough_date"):
            d = lm[key]
            if d in close.index:
                ax.plot(
                    d,
                    close.loc[d],
                    marker="o",
                    ms=4.5,
                    mfc=bd.INK["surface"],
                    mec=bd.CAT["blue"],
                    mew=1.4,
                    zorder=5,
                )

    ax.plot(close.index, close.values, color=bd.CAT["blue"], lw=1.8, zorder=4)

    bd.figure_title(ax, 2, "S&P/TSX Composite with landmark drawdowns and rebounds, 2025–2026")
    ax.set_ylabel("Index level (adj. close)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.margins(x=0.01)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))

    out = bd.savefig(fig, "fig1_tsx_landmarks.png")
    print(f"wrote {out}")
    moves = "; ".join(f"{lm['label']} {lm['pct']:+.1f}%" for lm in lms)
    print(
        "CAPTION: Yahoo ^GSPTSE adjusted close via the workshop_experiments TSX data service. "
        "Shaded windows mark landmark moves, computed on the close path as peak-to-trough "
        f"(drawdowns) and trough-to-recovery (rebounds): {moves}.",
    )


if __name__ == "__main__":
    main()
