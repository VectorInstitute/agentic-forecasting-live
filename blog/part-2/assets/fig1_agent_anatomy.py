"""Part-2 fig1: anatomy of one news-agent forecast (origin 2026-03-30, h=21).

Three panels, left -> right, tracing the agent from evidence to distribution:

* left   -- the six web searches it issued, as a vertical tool trail;
* center -- the load-bearing rationale factors it wrote;
* right  -- the emitted quantile grid as a distribution strip, with the median
            and the realised +21bd outcome marked.

Every value is read from the persisted prediction store:
``predictions/tsx_ws_eval_2026_weekly/
   agent_predictor_tsx_analyst_news_claude-sonnet-4-6_continuous/
   tsx_logret_21b/2026-03-30.yaml`` (tool_calls, rationale, quantiles). The
realised value is the ``tsx_logret_21b`` for forecast_date 2026-04-28 from the
leak-safe TSX data service (identical to Part-1's ``_blogdata.realized``).

Style matches Part-1 (shared ``_blogdata`` palette + rcParams, 220 dpi).

Run: ``python blog/part-2/assets/fig1_agent_anatomy.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

_P1_ASSETS = Path(__file__).resolve().parents[2] / "part-1" / "assets"
sys.path.insert(0, str(_P1_ASSETS))

import _blogdata as bd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


SPEC = "tsx_ws_eval_2026_weekly"
MODEL = "agent_predictor_tsx_analyst_news_claude-sonnet-4-6_continuous"
TARGET = "tsx_logret_21b"
ORIGIN = "2026-03-30"

AGENT = bd.CAT["orange"]  # the agent accent (Part-1 fig3 family colour for agents)

# The six search queries, paraphrased to short trail labels (verbatim source in
# the caption / EDITORS notes). Order = emission order in tool_calls.
SEARCHES = [
    "BoC policy rate & forward guidance",
    "CPI + jobs-report market reaction",
    "Oil & gold, energy & materials",
    "USD/CAD & GoC 10-year yields",
    "Fed policy & tariff spillovers",
    "TSX sector earnings (Q1 2026)",
]

# The load-bearing rationale factors (from the written rationale).
FACTORS = [
    ("BoC held at 2.25%", "accommodative hold, no hike risk"),
    ("CPI 1.8% vs −84K Feb jobs", "soft inflation, deteriorating labour"),
    ("Middle-East commodity vol", "oil mixed, gold bid — net small +"),
    ("25% US tariffs", "structural headwind, negative skew"),
]

# Quantile grid as emitted (log-return space).
QUANT = {
    0.05: -0.075, 0.10: -0.055, 0.20: -0.030, 0.30: -0.010, 0.40: 0.0,
    0.50: 0.010, 0.60: 0.022, 0.70: 0.035, 0.80: 0.050, 0.90: 0.072, 0.95: 0.090,
}


def _realised() -> float:
    r = bd.realized(TARGET)
    fd = pd.Timestamp("2026-04-28")
    return float(r.loc[fd])


def main() -> None:
    bd.apply_style()
    realised = _realised()  # log-return, +0.0496

    fig = plt.figure(figsize=(13.4, 7.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.06, 0.92], wspace=0.10,
                          left=0.015, right=0.965, top=0.82, bottom=0.11)
    ax_l = fig.add_subplot(gs[0])
    ax_c = fig.add_subplot(gs[1])
    ax_r = fig.add_subplot(gs[2])
    for ax in (ax_l, ax_c):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # ---- Title band --------------------------------------------------------
    bd.figure_title(fig, 1, "Anatomy of one agent forecast", x=0.015, y=0.955)

    # Column headers.
    ax_l.text(0.0, 1.055, "1  GATHER", fontsize=bd.FS["axtitle"], fontweight="bold",
              color=AGENT, ha="left", transform=ax_l.transAxes)
    ax_l.text(0.0, 1.010, "six web searches",
              color=bd.INK["muted"], ha="left", transform=ax_l.transAxes)
    ax_c.text(0.0, 1.055, "2  REASON", fontsize=bd.FS["axtitle"], fontweight="bold",
              color=bd.CAT["violet"], ha="left", transform=ax_c.transAxes)
    ax_c.text(0.0, 1.010, "load-bearing factors",
              color=bd.INK["muted"], ha="left", transform=ax_c.transAxes)
    ax_r.text(0.0, 1.055, "3  FORECAST", fontsize=bd.FS["axtitle"], fontweight="bold",
              color=bd.CAT["blue"], ha="left", transform=ax_r.transAxes)
    ax_r.text(0.0, 1.010, "quantile grid (21-day log return)",
              color=bd.INK["muted"], ha="left", transform=ax_r.transAxes)

    # ---- Left: tool trail --------------------------------------------------
    n = len(SEARCHES)
    ys = np.linspace(0.90, 0.10, n)
    xdot = 0.055
    ax_l.plot([xdot, xdot], [ys[-1], ys[0]], color=bd.INK["grid"], lw=2.2, zorder=1)
    for i, (y, label) in enumerate(zip(ys, SEARCHES), start=1):
        ax_l.add_patch(plt.Circle((xdot, y), 0.028, facecolor=AGENT,
                                  edgecolor=bd.INK["surface"], lw=1.6, zorder=3))
        ax_l.text(xdot, y, str(i), ha="center", va="center",
                  color="#ffffff", fontweight="bold", zorder=4)
        ax_l.text(xdot + 0.080, y, label, ha="left", va="center",
                  fontsize=bd.FS["label"], color=bd.INK["primary"])

    # ---- Center: rationale factor cards ------------------------------------
    m = len(FACTORS)
    top, bot, gap = 0.94, 0.06, 0.035
    card_h = (top - bot - (m - 1) * gap) / m
    for i, (head, sub) in enumerate(FACTORS):
        y1 = top - i * (card_h + gap)
        y0 = y1 - card_h
        ax_c.add_patch(plt.Rectangle((0.02, y0), 0.96, card_h,
                                     facecolor=bd.INK["page"],
                                     edgecolor=bd.INK["grid"], lw=1.0, zorder=1))
        ax_c.add_patch(plt.Rectangle((0.02, y0), 0.02, card_h,
                                     facecolor=bd.CAT["violet"], lw=0, zorder=2))
        ax_c.text(0.075, y0 + card_h * 0.62, head, ha="left", va="center",
                  fontsize=bd.FS["label"], fontweight="bold", color=bd.INK["primary"])
        ax_c.text(0.075, y0 + card_h * 0.26, sub, ha="left", va="center",
                  color=bd.INK["secondary"])

    # ---- Right: distribution strip -----------------------------------------
    levels = sorted(QUANT)
    vals = np.array([QUANT[q] * 100 for q in levels])  # percent
    med = QUANT[0.50] * 100
    q05, q95 = QUANT[0.05] * 100, QUANT[0.95] * 100
    q20, q80 = QUANT[0.20] * 100, QUANT[0.80] * 100
    real_pc = realised * 100

    ax_r.set_xlim(0, 1)
    lo, hi = -9.0, 10.5
    ax_r.set_ylim(lo, hi)
    ax_r.set_facecolor(bd.INK["surface"])
    for s in ("top", "right", "bottom"):
        ax_r.spines[s].set_visible(False)
    ax_r.spines["left"].set_color(bd.INK["axis"])
    ax_r.set_xticks([])
    ax_r.grid(False)
    ax_r.axhline(0, color=bd.INK["axis"], lw=0.9, zorder=1)
    ax_r.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(decimals=0))

    xc = 0.34          # strip centre
    half90 = 0.16      # 90% interval half-width
    half60 = 0.16      # 60% interval half-width (drawn wider visually)
    # 90% interval (light) and 60% interval (darker) as vertical bands.
    ax_r.add_patch(plt.Rectangle((xc - half90, q05), 2 * half90, q95 - q05,
                                 facecolor=AGENT, alpha=0.16, lw=0, zorder=2))
    ax_r.add_patch(plt.Rectangle((xc - half60 * 1.32, q20), 2 * half60 * 1.32, q80 - q20,
                                 facecolor=AGENT, alpha=0.34, lw=0, zorder=3))
    # Quantile tick marks across the strip.
    for q, v in zip(levels, vals):
        ax_r.plot([xc - half90, xc + half90], [v, v], color=AGENT,
                  lw=0.8, alpha=0.55, zorder=4)
    # Median.
    ax_r.plot([xc - half90 * 1.42, xc + half90 * 1.42], [med, med],
              color=bd.INK["primary"], lw=2.4, zorder=6)
    ax_r.text(xc + half90 * 1.5, med, f"median {med:+.0f}%", ha="left", va="center",
              fontsize=bd.FS["label"], fontweight="bold", color=bd.INK["primary"])
    # Key-quantile labels, tucked just left of the strip (clear of the % axis).
    for q in (0.05, 0.80, 0.95):
        v = QUANT[q] * 100
        ax_r.text(xc - half90 * 1.04, v, f"Q{q:.2f}", ha="right", va="center",
                  color=bd.INK["muted"])

    # Realised outcome marker (neutral, high-contrast) landing at ~Q0.80.
    xr = xc + half90 * 1.42
    ax_r.annotate(
        "",
        xy=(xc + half90 + 0.01, real_pc), xytext=(xr + 0.30, real_pc),
        arrowprops=dict(arrowstyle="-|>", color=bd.INK["primary"], lw=1.8,
                        shrinkA=0, shrinkB=0),
        zorder=8,
    )
    ax_r.plot([xr + 0.30], [real_pc], marker="D", ms=10,
              color=bd.INK["primary"], zorder=9, clip_on=False)
    real_simple = (np.exp(realised) - 1) * 100  # simple-return headline (+5.1%)
    ax_r.text(xr + 0.30, real_pc + 1.35,
              f"realised {real_simple:+.1f}%\n(21-day)  ≈ Q0.80",
              ha="center", va="bottom", fontsize=bd.FS["label"], fontweight="bold",
              color=bd.INK["primary"], linespacing=1.25)

    out = bd.savefig(fig, "fig1_agent_anatomy.png")
    print(f"wrote {out}  realised={real_pc:+.3f}% (log)  median={med:+.1f}%")
    print(
        "CAPTION: News analyst agent (Claude Sonnet-4.6) forecasting the S&P/TSX "
        "21-business-day log return from origin 2026-03-30: six web searches and "
        "86 s of wall clock, four load-bearing rationale factors, and an 11-point "
        f"quantile grid (median {med:+.0f}%)."
    )
    print(
        f"CAPTION: The realised 21-day return of {real_simple:+.1f}% "
        f"(log {real_pc:+.2f}%) landed at roughly the agent's Q0.80. Search "
        "queries are paraphrased; verbatim queries, rationale, and quantiles are "
        "in the persisted prediction store "
        "(predictions/tsx_ws_eval_2026_weekly/agent_predictor_tsx_analyst_news_"
        "claude-sonnet-4-6_continuous/tsx_logret_21b/2026-03-30.yaml), with the "
        "realised value from the leak-safe TSX data service."
    )


if __name__ == "__main__":
    main()
