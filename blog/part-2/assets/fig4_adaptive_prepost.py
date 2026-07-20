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
    # The three horizons live on very different CRPS scales (~5 / ~12 / ~18.5
    # ×10⁻³), so a single shared axis cannot zoom. We give each horizon its own
    # y-zoomed panel with a non-zero baseline (and a break cue) so the ~2%
    # pre-vs-post difference reads as near-identical-but-visible, not as two
    # identical bars. War-window means stay as open diamonds above each pair.
    fig = plt.figure(figsize=(12.9, 7.9))
    # Left three horizon panels and the right card are decoupled so the card can
    # use nearly the full height while the panels leave room for their labels.
    gs_left = fig.add_gridspec(1, 3, left=0.06, right=0.605, top=0.845, bottom=0.205, wspace=0.62)
    axes_h = [fig.add_subplot(gs_left[i]) for i in range(3)]
    axc = fig.add_axes([0.655, 0.075, 0.33, 0.86])

    # Per-horizon zoomed y-window (×10⁻³): brackets the pre/post bars, the war
    # markers, and (h=21) the LightGBM+cov reference, from a non-zero baseline.
    YLIM = {1: (5.02, 5.52), 5: (11.55, 16.1), 21: (16.9, 25.4)}
    bw = 0.62
    xpre, xpost = 0.0, 1.0

    for ax, h in zip(axes_h, HS):
        y0, y1 = YLIM[h]
        rng = y1 - y0
        pv, qv = pre_m[h] * 1000, post_m[h] * 1000
        ax.bar(xpre, pv, width=bw, color=C_PRE, alpha=0.9, zorder=3)
        ax.bar(xpost, qv, width=bw, color=C_POST, alpha=0.9, zorder=3)
        ax.text(xpre, pv + rng * 0.015, f"{pv:.2f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=C_PRE)
        ax.text(xpost, qv + rng * 0.015, f"{qv:.2f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=C_POST)

        # War-window cut markers (h=5, h=21).
        if h in warpre:
            ax.plot(xpre, warpre[h] * 1000, marker="D", ms=9, mfc=bd.INK["surface"],
                    mec=C_PRE, mew=1.8, zorder=5, clip_on=False)
            ax.plot(xpost, warpost[h] * 1000, marker="D", ms=9, mfc=bd.INK["surface"],
                    mec=C_POST, mew=1.8, zorder=5, clip_on=False)
            ax.text(0.5, (warpre[h] * 1000 + rng * 0.03), "war-window cut",
                    ha="center", va="bottom", fontsize=11, color=bd.INK["secondary"])

        # h=21: LightGBM+cov reference line for rank context. It is labelled in the
        # top legend (the panel center is filled by bars, so no in-panel label fits
        # near the line without colliding).
        if h == 21:
            ax.axhline(tree21 * 1000, color=bd.CAT["aqua"], lw=1.6, ls=(0, (4, 3)), zorder=4)

        ax.set_xlim(-0.72, 1.72)
        ax.set_ylim(y0, y1)
        ax.set_xticks([0.5])
        ax.set_xticklabels([f"h = {h}"], fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", length=0, pad=10)
        ax.tick_params(axis="y", labelsize=11.5)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", visible=True, color=bd.INK["grid"], lw=0.6)
        ax.set_axisbelow(True)
        # Post-win count under each panel (below the h = N label).
        ax.text(0.5, -0.105, f"post wins {wins_of[h]}/{n_of[h]}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=11.5, color=bd.INK["secondary"], fontweight="bold")
        # Break cue: diagonal slashes at the non-zero baseline.
        _bk = dict(transform=ax.transAxes, color=bd.INK["axis"], clip_on=False, lw=1.4)
        ax.plot([-0.05, 0.05], [-0.012, 0.012], **_bk)
        ax.plot([-0.05, 0.05], [0.016, 0.040], **_bk)

    axes_h[0].set_ylabel("Mean CRPS ×10⁻³  (lower is better)", fontsize=13)

    # Legend (PRE / POST / war marker) + title/subtitle spanning the left block.
    from matplotlib.patches import Patch  # noqa: E402
    from matplotlib.lines import Line2D as _L2D  # noqa: E402
    handles = [
        Patch(facecolor=C_PRE, alpha=0.9, label="PRE — seed strategy"),
        Patch(facecolor=C_POST, alpha=0.9, label="POST — trained strategy"),
        _L2D([0], [0], marker="D", ls="none", mfc=bd.INK["surface"], mec=bd.INK["secondary"],
             mew=1.6, ms=9, label="war-window mean"),
        _L2D([0], [0], color=bd.CAT["aqua"], lw=1.6, ls=(0, (4, 3)),
             label=f"LightGBM+cov (h=21 ref, {tree21 * 1000:.1f})"),
    ]
    axes_h[0].legend(handles=handles, loc="lower left", fontsize=11.5, handlelength=1.4,
                     bbox_to_anchor=(-0.02, 1.02), ncol=4, frameon=False,
                     columnspacing=1.3, handletextpad=0.5, borderaxespad=0)
    fig.text(0.06, 0.958, "One study session: nothing moved", fontsize=18,
             fontweight="bold", color=bd.INK["primary"], ha="left")
    fig.text(0.06, 0.915,
             "PRE vs POST mean CRPS, protected 2026 eval  ·  panels y-zoomed  ·  all within noise (n ≤ 24)",
             fontsize=11.5, color=bd.INK["muted"], ha="left", va="top")

    # ================= RIGHT: strategy-file excerpt card ====================
    axc.set_xlim(0, 1)
    axc.set_ylim(0, 1)
    axc.axis("off")

    axc.text(0.0, 0.998, "WHAT GRADUATED", fontsize=13, fontweight="bold",
             color=bd.INK["secondary"], ha="left", va="top")
    axc.text(0.0, 0.960, "trained strategy file  ·  skill_state.yaml (condensed, faithful)",
             fontsize=11, color=bd.INK["muted"], ha="left", va="top")

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

    top, bot, gap = 0.925, 0.150, 0.022
    n = len(CARDS)
    ch = (top - bot - (n - 1) * gap) / n
    for i, (accent, tag, tagc, title, body) in enumerate(CARDS):
        y1 = top - i * (ch + gap)
        y0 = y1 - ch
        neg = tag.startswith("NEGATIVE")
        face = bd.INK["page"] if neg else bd.INK["surface"]
        axc.add_patch(plt.Rectangle((0.0, y0), 1.0, ch, facecolor=face,
                                    edgecolor=bd.INK["grid"], lw=1.0, zorder=1))
        axc.add_patch(plt.Rectangle((0.0, y0), 0.010, ch, facecolor=accent, lw=0, zorder=2))
        axc.text(0.036, y1 - 0.024, tag, ha="left", va="top", fontsize=10.5,
                 fontweight="bold", color=tagc)
        axc.text(0.036, y1 - 0.068, title, ha="left", va="top", fontsize=12.5,
                 fontweight="bold", color=bd.INK["primary"])
        axc.text(0.036, y1 - 0.112, _wrap(body, 46), ha="left", va="top",
                 fontsize=10, color=bd.INK["secondary"], linespacing=1.34)

    # Footer label.
    axc.add_patch(plt.Rectangle((0.0, 0.020), 1.0, 0.098, facecolor=bd.INK["page"],
                                edgecolor=bd.STATUS["warning"], lw=1.3, zorder=1))
    axc.add_patch(plt.Rectangle((0.0, 0.020), 0.010, 0.098, facecolor=bd.STATUS["warning"],
                                lw=0, zorder=2))
    axc.text(0.036, 0.069,
             _wrap("25 corrections graduated in one session — all confirmations "
                   "same-session (see text).", 54),
             ha="left", va="center", fontsize=10.5, color=bd.INK["secondary"],
             linespacing=1.32)

    # Footnote wrapped to the LEFT panel-block width only (x 0.06 .. ~0.60) so it
    # never runs horizontally under the right-hand strategy cards / yellow box.
    _foot = (
        "Source: predictions/tsx_ws_eval_2026_weekly/ — seed rung "
        "agent_predictor_tsx_adaptive_analyst_tsx_strategy_gemini-3.5-flash_continuous "
        "(PRE) vs ..._tsx_strategy_trained_... (POST), tsx_logret_{1b,5b,21b}. Per-origin "
        "CRPS via properscoring.crps_ensemble on the sorted 11-point quantile grid vs the "
        "realised value (_blogdata.realized), resolved origins only (n = 24 / 22 / 24). "
        "War-window cut = origins 2026-02-09 to 04-13. Card text condensed from "
        "adaptive/skills_tsx/tsx-strategy-trained/skill_state.yaml."
    )
    fig.text(
        0.06, 0.055, _wrap(_foot, 92),
        fontsize=9.5, color=bd.INK["muted"], ha="left", va="top", linespacing=1.4,
    )

    out = Path(__file__).resolve().parent / "fig4_adaptive_prepost.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=bd.INK["surface"])
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
