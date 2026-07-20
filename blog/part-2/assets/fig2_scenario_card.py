"""Part-2 fig2: one scenario write-up, graded against what happened.

The 2026-03-31 TSX scenario card: three named scenarios (probabilities and
60-business-day outlook ranges) on the left; the LLM judge's verdict scores and
the realised cumulative returns on the right; and one annotation naming the
mechanism mismatch -- the base case called the direction right for the wrong
reason.

Sources (read-only):
* data/scenarios/2026-03-31/writeup.md   -- scenario names, probabilities, ranges
* data/scenarios/2026-03-31/judge.yaml   -- verdict scores + realised_outcome

Style matches Part-1 (shared ``_blogdata`` palette + rcParams, 220 dpi).

Run: ``python blog/part-2/assets/fig2_scenario_card.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

_P1_ASSETS = Path(__file__).resolve().parents[2] / "part-1" / "assets"
sys.path.insert(0, str(_P1_ASSETS))

import _blogdata as bd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


# (name, probability, 60-day outlook range, one-line driver, is_base)
SCENARIOS = [
    ("Commodity-Led Defensive Rotation", 0.55, "+3% to +5%",
     "Middle-East friction persists; rotate into Energy & Materials", True),
    ("US-Trade Policy Stalls / Tariff Risk", 0.30, "−4% to −7%",
     "Trade barriers bite; a 'tariff premium' widens the discount", False),
    ("BoC Policy Disappointment", 0.15, "−2% to +1%",
     "BoC pivots to easing; market reads a deeper recession risk", False),
]

# Judge verdict (out of 5) from judge.yaml.
VERDICT = [("Calibration", 5, bd.STATUS["good"]),
           ("Drivers", 3, bd.STATUS["warning"]),
           ("Specificity", 4, bd.CAT["blue"])]

# Realised cumulative log returns from 2026-03-31 (judge.yaml realized_outcome).
REALISED = [("5-day", "2026-04-08", "+2.60%"),
            ("21-day", "2026-04-30", "+3.65%"),
            ("60-day", "2026-06-25", "+6.35%")]


def main() -> None:
    bd.apply_style()

    fig, ax = plt.subplots(figsize=(13.4, 7.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- Title -------------------------------------------------------------
    fig.text(0.012, 0.955, "One scenario write-up, graded against what happened",
             fontsize=18, fontweight="bold", color=bd.INK["primary"], ha="left")
    fig.text(0.012, 0.912,
             "S&P/TSX 60-business-day outlook  ·  scenario set issued 2026-03-31  ·  "
             "judged by Claude Sonnet-4.6 against realised returns",
             fontsize=13, color=bd.INK["secondary"], ha="left")

    # ================= LEFT: scenario cards =================================
    lx0, lx1 = 0.012, 0.60
    ax.text(lx0, 0.85, "SCENARIOS", fontsize=13, fontweight="bold",
            color=bd.INK["secondary"], ha="left")
    top, bot, gap = 0.83, 0.205, 0.030
    n = len(SCENARIOS)
    card_h = (top - bot - (n - 1) * gap) / n
    for i, (name, prob, outlook, driver, is_base) in enumerate(SCENARIOS):
        y1 = top - i * (card_h + gap)
        y0 = y1 - card_h
        accent = bd.CAT["aqua"] if is_base else bd.INK["muted"]
        face = bd.INK["page"] if is_base else bd.INK["surface"]
        ax.add_patch(plt.Rectangle((lx0, y0), lx1 - lx0, card_h, facecolor=face,
                                   edgecolor=(accent if is_base else bd.INK["grid"]),
                                   lw=(2.0 if is_base else 1.0), zorder=1))
        ax.add_patch(plt.Rectangle((lx0, y0), 0.006, card_h, facecolor=accent, lw=0, zorder=2))

        # Probability block.
        px = lx0 + 0.045
        ax.text(px, y0 + card_h * 0.60, f"{prob:.2f}", ha="center", va="center",
                fontsize=27, fontweight="bold", color=accent)
        ax.text(px, y0 + card_h * 0.22, "probability", ha="center", va="center",
                fontsize=10, color=bd.INK["muted"])
        # Probability bar.
        bar_x0, bar_w = lx0 + 0.088, 0.055
        ax.add_patch(plt.Rectangle((bar_x0, y0 + card_h * 0.30), bar_w, card_h * 0.42,
                                   facecolor=bd.INK["grid"], lw=0, zorder=2))
        ax.add_patch(plt.Rectangle((bar_x0, y0 + card_h * 0.30), bar_w * prob, card_h * 0.42,
                                   facecolor=accent, alpha=0.85, lw=0, zorder=3))

        # Text block.
        tx = lx0 + 0.165
        ax.text(tx, y0 + card_h * 0.72, name, ha="left", va="center",
                fontsize=13.5, fontweight="bold", color=bd.INK["primary"])
        if is_base:
            ax.text(lx1 - 0.012, y0 + card_h * 0.72, "BASE CASE", ha="right", va="center",
                    fontsize=11, fontweight="bold", color=accent)
        ax.text(tx, y0 + card_h * 0.44, f"60-day outlook:  {outlook}", ha="left", va="center",
                fontsize=13, color=bd.INK["secondary"])
        ax.text(tx, y0 + card_h * 0.19, driver, ha="left", va="center",
                fontsize=12, color=bd.INK["muted"])

    # ================= RIGHT: verdict + realised ============================
    rx0, rx1 = 0.635, 0.988

    # ---- Judge verdict panel ----
    v_top, v_bot = 0.83, 0.545
    ax.add_patch(plt.Rectangle((rx0, v_bot), rx1 - rx0, v_top - v_bot,
                               facecolor=bd.INK["surface"], edgecolor=bd.INK["grid"],
                               lw=1.0, zorder=1))
    ax.text(rx0 + 0.018, v_top - 0.035, "JUDGE VERDICT", fontsize=13,
            fontweight="bold", color=bd.INK["secondary"], ha="left")
    rows_y = [v_top - 0.095, v_top - 0.170, v_top - 0.245]
    for (label, score, color), ry in zip(VERDICT, rows_y):
        ax.text(rx0 + 0.018, ry, label, ha="left", va="center",
                fontsize=13, color=bd.INK["primary"])
        # 5 dots.
        dot_x0 = rx0 + 0.150
        for k in range(5):
            filled = k < score
            ax.add_patch(plt.Circle((dot_x0 + k * 0.030, ry), 0.0092,
                                    facecolor=color if filled else bd.INK["surface"],
                                    edgecolor=color if filled else bd.INK["grid"],
                                    lw=1.2, zorder=3))
        ax.text(rx1 - 0.020, ry, f"{score}/5", ha="right", va="center",
                fontsize=13, fontweight="bold", color=color)

    # ---- Realised panel ----
    a_top, a_bot = 0.515, 0.175
    ax.add_patch(plt.Rectangle((rx0, a_bot), rx1 - rx0, a_top - a_bot,
                               facecolor=bd.INK["surface"], edgecolor=bd.INK["grid"],
                               lw=1.0, zorder=1))
    ax.text(rx0 + 0.018, a_top - 0.035, "REALISED  (from 2026-03-31)", fontsize=13,
            fontweight="bold", color=bd.INK["secondary"], ha="left")
    ry_list = [a_top - 0.105, a_top - 0.190, a_top - 0.275]
    for (horizon, fdate, ret), ry in zip(REALISED, ry_list):
        ax.text(rx0 + 0.018, ry + 0.018, horizon, ha="left", va="center",
                fontsize=13.5, fontweight="bold", color=bd.INK["primary"])
        ax.text(rx0 + 0.018, ry - 0.030, f"through {fdate}", ha="left", va="center",
                fontsize=11, color=bd.INK["muted"])
        ax.text(rx1 - 0.050, ry, ret, ha="right", va="center",
                fontsize=17, fontweight="bold", color=bd.STATUS["good"])
        ax.plot([rx1 - 0.028], [ry], marker="^", ms=11, color=bd.STATUS["good"],
                zorder=4, clip_on=False)

    # ================= Mechanism-mismatch banner ============================
    # Body wrapped to two lines with inner padding so no word touches the box edge.
    ax.add_patch(plt.Rectangle((0.012, 0.020), 0.976, 0.150,
                               facecolor=bd.INK["page"], edgecolor=bd.STATUS["warning"],
                               lw=1.4, zorder=1))
    ax.add_patch(plt.Rectangle((0.012, 0.020), 0.006, 0.150,
                               facecolor=bd.STATUS["warning"], lw=0, zorder=2))
    ax.text(0.032, 0.143, "Right call, wrong reason", fontsize=15, fontweight="bold",
            color=bd.INK["primary"], ha="left", va="center")
    ax.text(0.032, 0.070,
            "Base-case mechanism — persistent Middle-East friction keeping oil bid — did not hold; the rally\n"
            "came from a ceasefire relief bounce. Direction and magnitude landed (calibration 5/5), but the\n"
            "causal chain did not (drivers 3/5).",
            fontsize=12, color=bd.INK["secondary"], ha="left", va="center", linespacing=1.5)

    out = _P1_ASSETS.parent.parent / "part-2" / "assets" / "fig2_scenario_card.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=bd.INK["surface"])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
