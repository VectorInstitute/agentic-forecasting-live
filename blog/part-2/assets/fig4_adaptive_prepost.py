"""Part-2 fig4: one adaptive study session, graded as a null result.

LEFT -- pre/post paired CRPS on the protected 2026 eval. Grouped bars by horizon
(h=1, 5, 21): PRE = the seed strategy rung
(``agent_predictor_tsx_adaptive_analyst_tsx_strategy_gemini-3.5-flash_continuous``)
vs POST = the trained strategy rung
(``..._tsx_strategy_trained_gemini-3.5-flash_continuous``). Per-origin CRPS via the
same ``properscoring.crps_ensemble`` over the sorted 11-point quantile grid as
Part-1, over the origins each rung resolved (n = 24 / 22 / 24). Per-horizon
post-win counts are annotated; small open markers show the war-window cut
(origins 2026-02-09..2026-04-13) at h=5 and h=21; a subtle reference line marks
darts_lightgbm_cov's h=21 mean for rank context. The horizon-mean differences run
both ways and are all within noise (n<=24) -- an honest null.

RIGHT -- a faithful, condensed excerpt of the trained strategy file
(``adaptive/skills_tsx/tsx-strategy-trained/skill_state.yaml``): two graduated
calibration corrections and one recorded NEGATIVE result (the day-of-week ANOVA
finding), styled as Part-2 fig2's cards.

The script recomputes every plotted CRPS and asserts the six horizon means, the
two war-window cuts, the reference mean, and the three win counts reproduce the
pinned values to 1e-5 before writing the PNG.

Run (from the worktree root, with the refreshed store on hand):
``BLOG_WS_DATA_ROOT=.../workshop_experiments/workshop_experiments/data \
  uv run python blog/part-2/assets/fig4_adaptive_prepost.py``
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
PRE = "agent_predictor_tsx_adaptive_analyst_tsx_strategy_gemini-3.5-flash_continuous"
POST = "agent_predictor_tsx_adaptive_analyst_tsx_strategy_trained_gemini-3.5-flash_continuous"
TREE = "darts_lightgbm_cov"
TASK = {1: "tsx_logret_1b", 5: "tsx_logret_5b", 21: "tsx_logret_21b"}
HS = [1, 5, 21]

WAR_START, WAR_END = pd.Timestamp("2026-02-09"), pd.Timestamp("2026-04-13")

# Pinned values the recompute must reproduce (tolerance 1e-5).
REF_PRE = {1: 0.0052246, 5: 0.0120594, 21: 0.0185683}
REF_POST = {1: 0.0052720, 5: 0.0119320, 21: 0.0187625}
REF_WARPRE = {5: 0.0155245, 21: 0.0246009}
REF_WARPOST = {5: 0.0151359, 21: 0.0241618}
REF_TREE21 = 0.0171834
REF_WINS = {1: (13, 24), 5: (12, 22), 21: (10, 24)}

C_PRE = bd.INK["muted"]     # seed strategy (baseline grey)
C_POST = bd.CAT["blue"]     # trained strategy (accent)


def _paired(h: int):
    """Return (pre_mean, post_mean, n, post_wins, war_pre, war_post) for horizon h."""
    task = TASK[h]
    pre = bd.crps_series(SPEC, PRE, task)
    post = bd.crps_series(SPEC, POST, task)
    common = pre.index.intersection(post.index)
    pre, post = pre.loc[common], post.loc[common]
    n = int(len(common))
    wins = int((post.values < pre.values).sum())
    mask = (common >= WAR_START) & (common <= WAR_END)
    war_pre = float(pre.loc[mask].mean()) if mask.any() else float("nan")
    war_post = float(post.loc[mask].mean()) if mask.any() else float("nan")
    return float(pre.mean()), float(post.mean()), n, wins, war_pre, war_post


def main() -> None:
    bd.apply_style()

    pre_m, post_m, n_of, wins_of, warpre, warpost = {}, {}, {}, {}, {}, {}
    for h in HS:
        pm, qm, n, w, wp, wq = _paired(h)
        pre_m[h], post_m[h], n_of[h], wins_of[h] = pm, qm, n, w
        if h in (5, 21):
            warpre[h], warpost[h] = wp, wq

    tree21 = float(bd.crps_series(SPEC, TREE, TASK[21]).mean())

    # ---- Asserts: recompute must reproduce the pinned references -----------
    for h in HS:
        assert abs(pre_m[h] - REF_PRE[h]) <= 1e-5, f"pre h={h}: {pre_m[h]:.7f}"
        assert abs(post_m[h] - REF_POST[h]) <= 1e-5, f"post h={h}: {post_m[h]:.7f}"
        assert (wins_of[h], n_of[h]) == REF_WINS[h], f"wins h={h}: {wins_of[h]}/{n_of[h]}"
    for h in (5, 21):
        assert abs(warpre[h] - REF_WARPRE[h]) <= 1e-5, f"warpre h={h}: {warpre[h]:.7f}"
        assert abs(warpost[h] - REF_WARPOST[h]) <= 1e-5, f"warpost h={h}: {warpost[h]:.7f}"
    assert abs(tree21 - REF_TREE21) <= 1e-5, f"tree21: {tree21:.7f}"

    # ---- Figure ------------------------------------------------------------
    fig, (ax, axc) = plt.subplots(
        1, 2, figsize=(13.2, 6.7), gridspec_kw={"width_ratios": [1.42, 1.0], "wspace": 0.10},
    )

    # ================= LEFT: paired pre/post bars ===========================
    xs = np.arange(len(HS), dtype=float)
    bw = 0.34
    pre_k = [pre_m[h] * 1000 for h in HS]
    post_k = [post_m[h] * 1000 for h in HS]

    ax.bar(xs - bw / 2, pre_k, width=bw, color=C_PRE, alpha=0.9, zorder=3, label="PRE — seed strategy")
    ax.bar(xs + bw / 2, post_k, width=bw, color=C_POST, alpha=0.9, zorder=3, label="POST — trained strategy")
    for x, v in zip(xs - bw / 2, pre_k):
        ax.text(x, v + 0.3, f"{v:.2f}", ha="center", va="bottom", fontsize=8.4,
                fontweight="bold", color=C_PRE)
    for x, v in zip(xs + bw / 2, post_k):
        ax.text(x, v + 0.3, f"{v:.2f}", ha="center", va="bottom", fontsize=8.4,
                fontweight="bold", color=C_POST)

    # War-window cut: open markers over h=5 and h=21 (means over origins in window).
    for j, h in enumerate(HS):
        if h not in warpre:
            continue
        ax.plot(xs[j] - bw / 2, warpre[h] * 1000, marker="D", ms=6.2, mfc=bd.INK["surface"],
                mec=C_PRE, mew=1.5, zorder=5, clip_on=False)
        ax.plot(xs[j] + bw / 2, warpost[h] * 1000, marker="D", ms=6.2, mfc=bd.INK["surface"],
                mec=C_POST, mew=1.5, zorder=5, clip_on=False)
    ax.annotate("war-window cut\n(2026-02-09 – 04-13)",
                xy=(xs[2] + bw / 2, warpost[21] * 1000),
                xytext=(xs[1] + 0.16, 26.3), fontsize=8.0, color=bd.INK["secondary"],
                ha="center", va="top", linespacing=1.25,
                arrowprops=dict(arrowstyle="-", color=bd.INK["axis"], lw=0.9))

    # Reference line: darts_lightgbm_cov h=21 mean, for rank context (h=21 group only).
    ax.plot([xs[2] - 0.52, xs[2] + 0.50], [tree21 * 1000, tree21 * 1000],
            color=bd.CAT["aqua"], lw=1.2, ls=(0, (4, 3)), zorder=4)
    ax.text(xs[2] - 0.40, tree21 * 1000 + 0.45, f"LightGBM+cov  ({tree21 * 1000:.1f})",
            ha="right", va="bottom", fontsize=7.8, color=bd.CAT["aqua"])

    # Post-win counts under each horizon group.
    for j, h in enumerate(HS):
        ax.text(xs[j], -2.35, f"post wins {wins_of[h]}/{n_of[h]}", ha="center", va="top",
                fontsize=8.6, color=bd.INK["secondary"], fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"h = {h}" for h in HS], fontsize=10.5)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_ylim(0, 27.5)
    ax.set_ylabel("Mean CRPS ×10⁻³  (lower is better)")
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="x", visible=False)
    ax.margins(x=0.06)
    ax.set_title("One study session: nothing moved", fontsize=13.5,
                 fontweight="bold", loc="left", pad=22)
    ax.text(0.0, 1.018, "PRE vs POST mean CRPS on the protected 2026 eval  ·  "
            "horizon-mean differences run both ways, all within noise (n ≤ 24)",
            transform=ax.transAxes, fontsize=8.6, color=bd.INK["muted"],
            ha="left", va="bottom")
    ax.legend(loc="upper left", fontsize=8.8, handlelength=1.1,
              bbox_to_anchor=(0.005, 0.965), borderaxespad=0)

    # ================= RIGHT: strategy-file excerpt card ====================
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1)
    axc.axis("off")

    axc.text(0.0, 0.985, "WHAT GRADUATED", fontsize=10.5, fontweight="bold",
             color=bd.INK["secondary"], ha="left", va="top")
    axc.text(0.0, 0.945, "trained strategy file  ·  skill_state.yaml (condensed, faithful)",
             fontsize=8.6, color=bd.INK["muted"], ha="left", va="top")

    # (accent, kind_tag, tag_color, title, body)
    CARDS = [
        (bd.CAT["blue"], "CALIBRATION CORRECTION", bd.CAT["blue"],
         "Low-volatility interval widening",
         "When realised vol < 10%, widen prediction intervals by 12% (h=1), "
         "18% (h=5), 23% (h=21) vs standard-normal — corrects for volatility "
         "mean-reversion from calm.  [hyp-001]"),
        (bd.CAT["blue"], "CALIBRATION CORRECTION", bd.CAT["blue"],
         "Negative-anomaly widening",
         "When a daily return z-score < −2.5 (vs trailing 63 days) fires, widen "
         "the 1-day intervals by 40% (and 5-day by 20%) for the leverage/vol-of-vol "
         "kick.  [hyp-003]"),
        (bd.STATUS["critical"], "NEGATIVE RESULT — RULED OUT", bd.STATUS["critical"],
         "No day-of-week adjustment",
         "Unconditional + sub-period ANOVA tests: daily returns are statistically "
         "identical across weekdays (p = 0.5392) and averages sign-flip across "
         "epochs, so apply no day-of-week drift.  [hyp-009]"),
    ]

    top, bot, gap = 0.905, 0.115, 0.030
    n = len(CARDS)
    ch = (top - bot - (n - 1) * gap) / n
    for i, (accent, tag, tagc, title, body) in enumerate(CARDS):
        y1 = top - i * (ch + gap)
        y0 = y1 - ch
        neg = tag.startswith("NEGATIVE")
        face = bd.INK["page"] if neg else bd.INK["surface"]
        axc.add_patch(plt.Rectangle((0.0, y0), 1.0, ch, facecolor=face,
                                    edgecolor=bd.INK["grid"], lw=1.0, zorder=1))
        axc.add_patch(plt.Rectangle((0.0, y0), 0.008, ch, facecolor=accent, lw=0, zorder=2))
        axc.text(0.030, y1 - 0.028, tag, ha="left", va="top", fontsize=7.8,
                 fontweight="bold", color=tagc)
        axc.text(0.030, y1 - 0.072, title, ha="left", va="top", fontsize=11.4,
                 fontweight="bold", color=bd.INK["primary"])
        axc.text(0.030, y1 - 0.118, _wrap(body, 52), ha="left", va="top",
                 fontsize=8.6, color=bd.INK["secondary"], linespacing=1.42)

    # Footer label.
    axc.add_patch(plt.Rectangle((0.0, 0.008), 1.0, 0.088, facecolor=bd.INK["page"],
                                edgecolor=bd.STATUS["warning"], lw=1.3, zorder=1))
    axc.add_patch(plt.Rectangle((0.0, 0.008), 0.008, 0.088, facecolor=bd.STATUS["warning"],
                                lw=0, zorder=2))
    axc.text(0.030, 0.052,
             _wrap("25 corrections graduated in one session — all confirmations "
                   "same-session (see text).", 62),
             ha="left", va="center", fontsize=8.8, color=bd.INK["secondary"],
             linespacing=1.35)

    fig.text(
        0.058, 0.005,
        "Source: predictions/tsx_ws_eval_2026_weekly/ — seed rung "
        "agent_predictor_tsx_adaptive_analyst_tsx_strategy_gemini-3.5-flash_continuous (PRE) vs "
        "..._tsx_strategy_trained_... (POST), tsx_logret_{1b,5b,21b}. Per-origin CRPS via "
        "properscoring.crps_ensemble on the sorted 11-point quantile grid vs the realised value "
        "(_blogdata.realized), resolved origins only (n = 24 / 22 / 24). War-window cut = origins "
        "2026-02-09 to 04-13. Card text condensed from adaptive/skills_tsx/tsx-strategy-trained/skill_state.yaml.",
        fontsize=6.9, color=bd.INK["muted"], ha="left",
    )

    out = Path(__file__).resolve().parent / "fig4_adaptive_prepost.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=bd.INK["surface"])
    print(f"wrote {out}")
    for h in HS:
        extra = ""
        if h in warpre:
            extra = f"  war: pre={warpre[h]:.5f} post={warpost[h]:.5f}"
        print(f"h={h:2d}  pre={pre_m[h]:.5f}  post={post_m[h]:.5f}  wins={wins_of[h]}/{n_of[h]}{extra}")
    print(f"darts_lightgbm_cov h=21 = {tree21:.5f}")


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width=width))


if __name__ == "__main__":
    main()
