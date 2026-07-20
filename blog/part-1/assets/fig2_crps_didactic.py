"""fig2: CRPS in one intuitive pass -- sharp vs wide, same median.

Didactic (synthetic) figure at daily log-return scale. Two forecast
distributions share a median but differ in width; both face the same realised
value. CRPS (lower = better) is computed analytically for each Gaussian with
``properscoring.crps_gaussian`` and printed on the panel, showing that CRPS
rewards a sharp forecast when it is also right.

Run: ``python blog/part-1/assets/fig2_crps_didactic.py``
"""

from __future__ import annotations

import _blogdata as bd
import matplotlib.pyplot as plt
import numpy as np
import properscoring as ps


# Didactic parameters, chosen at daily log-return scale.
MEDIAN = 0.000  # both forecasts share this median (0% expected daily move)
REALIZED = 0.004  # the day actually returned +0.4%, near the shared median
SD_SHARP = 0.004
SD_WIDE = 0.012


def main() -> None:
    bd.apply_style()

    crps_sharp = float(ps.crps_gaussian(REALIZED, MEDIAN, SD_SHARP))
    crps_wide = float(ps.crps_gaussian(REALIZED, MEDIAN, SD_WIDE))

    x = np.linspace(-0.05, 0.05, 1200)

    def pdf(sd):
        return np.exp(-0.5 * ((x - MEDIAN) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

    fig, ax = plt.subplots(figsize=(9.5, 5.0))

    for sd, color, name, crps in (
        (SD_SHARP, bd.CAT["blue"], "Sharp forecast", crps_sharp),
        (SD_WIDE, bd.CAT["orange"], "Wide forecast", crps_wide),
    ):
        y = pdf(sd)
        ax.fill_between(x, 0, y, color=color, alpha=0.14, lw=0)
        ax.plot(x, y, color=color, lw=2.0, label=f"{name}   CRPS = {crps:.4f}")

    # Shared median and the realised value.
    ax.axvline(MEDIAN, color=bd.INK["muted"], lw=1.2, ls=(0, (4, 3)))
    ax.annotate(
        "shared median",
        xy=(MEDIAN, 0),
        xytext=(MEDIAN - 0.0015, 70),
        color=bd.INK["secondary"],
        ha="right",
    )
    ax.axvline(REALIZED, color=bd.INK["primary"], lw=1.6)
    ax.annotate(
        f"realised\n{REALIZED:+.1%}",
        xy=(REALIZED, 0),
        xytext=(REALIZED + 0.0015, 95),
        fontweight="bold",
        color=bd.INK["primary"],
        ha="left",
        linespacing=1.3,
    )

    bd.figure_title(ax, 3, "Sharp vs. wide forecast against one realised value")
    ax.set_xlabel("Next-day log return")
    ax.set_ylabel("Forecast density")
    ax.set_ylim(0, None)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(loc="upper left", handlelength=1.6)
    ax.margins(x=0)

    out = bd.savefig(fig, "fig2_crps_didactic.png")
    print(f"wrote {out}  (sharp={crps_sharp:.5f}, wide={crps_wide:.5f})")
    print(
        "CAPTION: Didactic, synthetic example at daily log-return scale. Both Gaussian forecasts "
        f"share a {MEDIAN:.1%} median; the realised next-day return is {REALIZED:+.1%}. CRPS is the "
        "integrated squared gap between the forecast CDF and the step at the realised value; lower "
        f"is better. The sharp forecast scores {crps_sharp:.4f} vs. {crps_wide:.4f} for the wide "
        "one — the sharper forecast wins here only because it also placed its mass near what "
        "happened.",
    )


if __name__ == "__main__":
    main()
