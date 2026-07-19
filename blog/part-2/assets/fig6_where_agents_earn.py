"""Part-2 fig6: where the agent's judgement earns its keep -- and where it doesn't.

LEFT -- "war window vs quiet weeks" (h=21). Grouped bars: the gemini news agent
vs darts_lightgbm_cov, mean CRPS split over (a) the 2026 war-window origins
(as_of 2026-02-09..2026-04-13, n=10) and (b) the quiet origins (n=14). The agent
beats the tree by 11% through the regime break and loses to it by 22% in the calm
-- the whole "agents read the news" case is one regime event sampled weekly, not
ten independent breaks.

RIGHT -- "does agency help the same model?" (h=21). Paired dumbbells from a frozen
LLM-Process rung to the agent built on the *same* base model. Improvement vs
regression is encoded by the arrow direction and colour. Only one pair is
statistically distinguishable: sonnet-4.6 LLMP -> code agent wins on 18 of 24
origins (one-sided sign test p ~ 0.01). All others -- including the two adaptive
arms, which regress off their gemini-3.5 base -- are within noise (n <= 24).

Per-origin CRPS is recomputed from the prediction stores with the same
``properscoring.crps_ensemble`` over the sorted 11-point quantile grid as Part-1,
over the common resolved origins of each pair. The script asserts every plotted
mean, the two war/quiet splits, and the code-agent win count reproduce the pinned
values before writing the PNG.

Run (from the worktree root, with the refreshed store on hand):
``BLOG_WS_DATA_ROOT=.../workshop_experiments/workshop_experiments/data \
  uv run python blog/part-2/assets/fig6_where_agents_earn.py``
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
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

SPEC = "tsx_ws_eval_2026_weekly"
T21 = "tsx_logret_21b"
WAR_START, WAR_END = pd.Timestamp("2026-02-09"), pd.Timestamp("2026-04-13")

NEWS_GEM = "agent_predictor_tsx_analyst_news_gemini-3.5-flash_continuous"
NEWS_SON = "agent_predictor_tsx_analyst_news_claude-sonnet-4-6_continuous"
CODE_SON = "agent_predictor_tsx_analyst_code_claude-sonnet-4-6_continuous"
ADAPT_PRE = "agent_predictor_tsx_adaptive_analyst_tsx_strategy_gemini-3.5-flash_continuous"
ADAPT_POST = "agent_predictor_tsx_adaptive_analyst_tsx_strategy_trained_gemini-3.5-flash_continuous"
LLMP_GEM = "llmp_quantile_grid_tsx_ws[gemini-3.5-flash]"
LLMP_SON = "llmp_quantile_grid_tsx_ws[claude-sonnet-4-6]"
TREE = "darts_lightgbm_cov"

# Pinned references (tolerance 1e-5 on means; exact on counts). Recomputed on the
# common resolved set of 24 origins (war n=10, quiet n=14).
REF_WAR = {"agent": 0.0217804, "tree": 0.0244914, "n": 10}
REF_QUIET = {"agent": 0.0146015, "tree": 0.0119633, "n": 14}
REF_MEAN = {
    LLMP_GEM: 0.0182594, NEWS_GEM: 0.0175927,
    LLMP_SON: 0.0209920, NEWS_SON: 0.0209890, CODE_SON: 0.0203487,
    ADAPT_PRE: 0.0185683, ADAPT_POST: 0.0187625,
}
REF_CODE_WINS = (18, 24)

C_TREE = bd.CAT["blue"]
C_AGENT = bd.CAT["orange"]
C_LLMP = bd.CAT["violet"]
C_UP = bd.STATUS["good"]      # improvement (agent better)
C_DOWN = bd.STATUS["critical"]  # regression (agent worse)


def _cs(model: str) -> pd.Series:
    return bd.crps_series(SPEC, model, T21)


def _paired(base: str, arm: str):
    b, m = _cs(base), _cs(arm)
    c = b.index.intersection(m.index)
    b, m = b.loc[c], m.loc[c]
    wins = int((m.values < b.values).sum())
    return float(b.mean()), float(m.mean()), int(len(c)), wins


def main() -> None:
    bd.apply_style()

    # ---- LEFT: war/quiet split (recompute) ---------------------------------
    a, t = _cs(NEWS_GEM), _cs(TREE)
    common = a.index.intersection(t.index)
    a, t = a.loc[common], t.loc[common]
    mask = (common >= WAR_START) & (common <= WAR_END)
    war = {"agent": float(a[mask].mean()), "tree": float(t[mask].mean()), "n": int(mask.sum())}
    quiet = {"agent": float(a[~mask].mean()), "tree": float(t[~mask].mean()), "n": int((~mask).sum())}
    assert len(common) == 24, len(common)
    for k in ("agent", "tree", "n"):
        assert (abs(war[k] - REF_WAR[k]) <= 1e-5) if k != "n" else (war[k] == REF_WAR[k]), (k, war[k])
        assert (abs(quiet[k] - REF_QUIET[k]) <= 1e-5) if k != "n" else (quiet[k] == REF_QUIET[k]), (k, quiet[k])
    war_delta = war["agent"] / war["tree"] - 1.0     # agent vs tree
    quiet_delta = quiet["agent"] / quiet["tree"] - 1.0

    # ---- RIGHT: paired dumbbells (recompute) -------------------------------
    PAIRS = [
        (LLMP_GEM, "LLMP gemini-3.5", NEWS_GEM, "News agent (gemini-3.5)"),
        (LLMP_GEM, "LLMP gemini-3.5", ADAPT_PRE, "Adaptive agent (pre-study)"),
        (LLMP_GEM, "LLMP gemini-3.5", ADAPT_POST, "Adaptive agent (post-study)"),
        (LLMP_SON, "LLMP sonnet-4.6", NEWS_SON, "News agent (sonnet-4.6)"),
        (LLMP_SON, "LLMP sonnet-4.6", CODE_SON, "Code agent (sonnet-4.6)"),
    ]
    dumbs = []
    for base, blab, arm, alab in PAIRS:
        bm, am, n, wins = _paired(base, arm)
        assert abs(bm - REF_MEAN[base]) <= 1e-5, (base, bm)
        assert abs(am - REF_MEAN[arm]) <= 1e-5, (arm, am)
        dumbs.append({"blab": blab, "alab": alab, "base": bm, "arm": am, "n": n, "wins": wins, "arm_id": arm})
    code = next(d for d in dumbs if d["arm_id"] == CODE_SON)
    assert (code["wins"], code["n"]) == REF_CODE_WINS, (code["wins"], code["n"])
    code_p = binomtest(code["wins"], code["n"], alternative="greater").pvalue

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(13.6, 6.4), gridspec_kw={"width_ratios": [1.0, 1.28], "wspace": 0.22},
    )

    # ================= LEFT panel: grouped bars =============================
    groups = ["War window\n(2026-02-09 – 04-13)", "Quiet weeks"]
    xg = np.arange(2, dtype=float)
    bw = 0.32
    agent_k = [war["agent"] * 1000, quiet["agent"] * 1000]
    tree_k = [war["tree"] * 1000, quiet["tree"] * 1000]

    axl.bar(xg - bw / 2, agent_k, width=bw, color=C_AGENT, zorder=3, label="News agent (gemini-3.5)")
    axl.bar(xg + bw / 2, tree_k, width=bw, color=C_TREE, zorder=3, label="LightGBM +cov")
    for x, v in zip(xg - bw / 2, agent_k):
        axl.text(x, v + 0.25, f"{v:.2f}", ha="center", va="bottom", fontsize=9.0,
                 fontweight="bold", color=C_AGENT)
    for x, v in zip(xg + bw / 2, tree_k):
        axl.text(x, v + 0.25, f"{v:.2f}", ha="center", va="bottom", fontsize=9.0,
                 fontweight="bold", color=C_TREE)

    # Delta annotations (agent vs tree), coloured by who wins.
    def _delta_note(x, ytop, delta, n):
        better = delta < 0  # agent lower CRPS = agent better
        col = C_UP if better else C_DOWN
        verb = "agent " + (f"{delta * 100:+.0f}%")
        axl.annotate(verb, xy=(x, ytop), xytext=(x, ytop + 2.6), ha="center", va="bottom",
                     fontsize=9.6, fontweight="bold", color=col,
                     arrowprops=dict(arrowstyle="-", color=bd.INK["axis"], lw=0.8))
        axl.text(x, ytop + 2.0, f"n = {n}", ha="center", va="bottom", fontsize=7.8,
                 color=bd.INK["muted"])

    _delta_note(0, max(agent_k[0], tree_k[0]), war_delta, war["n"])
    _delta_note(1, max(agent_k[1], tree_k[1]), quiet_delta, quiet["n"])

    axl.set_xticks(xg)
    axl.set_xticklabels(groups, fontsize=9.4)
    axl.tick_params(axis="x", length=0, pad=6)
    axl.set_ylim(0, 30)
    axl.set_ylabel("Mean CRPS ×10⁻³  (h = 21, lower is better)", fontsize=9.4)
    axl.tick_params(axis="y", labelsize=9)
    axl.grid(axis="x", visible=False)
    axl.margins(x=0.12)
    axl.set_title("The agent earns its keep in the break — and gives it back in the calm",
                  fontsize=11.6, fontweight="bold", loc="left", pad=20)
    axl.text(0.0, 1.017, "News agent (gemini-3.5) vs LightGBM +cov, mean CRPS by regime",
             transform=axl.transAxes, fontsize=8.8, color=bd.INK["muted"], ha="left", va="bottom")
    axl.legend(loc="upper right", fontsize=8.6, handlelength=1.1, borderaxespad=0.4)

    # ================= RIGHT panel: paired dumbbells =======================
    ny = len(dumbs)
    axr.set_xlim(16.9, 21.9)
    axr.set_ylim(-0.7, ny - 0.3)
    axr.invert_yaxis()

    for i, d in enumerate(dumbs):
        y = i
        bx, ax_ = d["base"] * 1000, d["arm"] * 1000
        diff = d["arm"] - d["base"]
        flat = abs(diff) < 5e-5
        better = (diff < 0) and not flat
        seg_col = bd.INK["muted"] if flat else (C_UP if better else C_DOWN)

        # connector with arrowhead pointing base -> arm (omitted when flat)
        if not flat:
            axr.annotate("", xy=(ax_, y), xytext=(bx, y),
                         arrowprops=dict(arrowstyle="-|>", color=seg_col, lw=2.4,
                                         shrinkA=6, shrinkB=6, mutation_scale=14), zorder=3)
        # base (frozen LLMP) marker
        axr.plot(bx, y, marker="o", ms=9.5, color=C_LLMP, mec=bd.INK["surface"], mew=1.1, zorder=4)
        # arm (agent) marker
        axr.plot(ax_, y, marker="o", ms=11.5, color=C_AGENT, mec=bd.INK["surface"], mew=1.2, zorder=5)

        # arm label (left gutter) + base note
        axr.text(-0.015, y - 0.17, d["alab"], transform=axr.get_yaxis_transform(),
                 ha="right", va="center", fontsize=9.0, fontweight="bold", color=C_AGENT)
        axr.text(-0.015, y + 0.19, f"from frozen {d['blab']}", transform=axr.get_yaxis_transform(),
                 ha="right", va="center", fontsize=7.8, color=bd.INK["muted"])

        # value labels at each end (single centered label when flat/coincident)
        if flat:
            axr.text(ax_, y - 0.34, f"{ax_:.2f}", ha="center", va="bottom", fontsize=7.9,
                     fontweight="bold", color=bd.INK["secondary"])
        else:
            axr.text(bx, y - 0.34, f"{bx:.2f}", ha="center", va="bottom", fontsize=7.6, color=C_LLMP)
            axr.text(ax_, y - 0.34, f"{ax_:.2f}", ha="center", va="bottom", fontsize=7.9,
                     fontweight="bold", color=C_AGENT)

        # outcome tag on the right
        if flat:
            tag, tcol = "flat", bd.INK["muted"]
        elif better:
            tag, tcol = f"agent {diff / d['base'] * 100:+.0f}%", C_UP
        else:
            tag, tcol = f"agent {diff / d['base'] * 100:+.0f}%", C_DOWN
        axr.text(21.82, y, tag, ha="right", va="center", fontsize=8.8, fontweight="bold", color=tcol)
        if d["arm_id"] == CODE_SON:
            axr.text(21.82, y + 0.30, f"{d['wins']}/{d['n']} origins · sign test p ≈ {code_p:.2f}",
                     ha="right", va="center", fontsize=7.6, color=bd.INK["secondary"])

    axr.set_yticks([])
    axr.tick_params(axis="x", labelsize=9)
    axr.set_xlabel("Mean CRPS ×10⁻³  (h = 21)", fontsize=9.4)
    axr.grid(axis="y", visible=False)
    axr.grid(axis="x", visible=True, color=bd.INK["grid"], lw=0.6)
    axr.set_axisbelow(True)
    axr.spines["left"].set_visible(False)
    axr.set_title("Does agency help the same model? Sometimes — once significantly.",
                  fontsize=11.6, fontweight="bold", loc="left", pad=20)
    axr.text(0.0, 1.017, "Frozen LLM-Process rung  to  agent on the same base model (h = 21 means)",
             transform=axr.transAxes, fontsize=8.8, color=bd.INK["muted"], ha="left", va="bottom")

    handles = [
        Line2D([0], [0], marker="o", ls="none", ms=9, color=C_LLMP, mec=bd.INK["surface"],
               mew=1.0, label="frozen LLM-Process"),
        Line2D([0], [0], marker="o", ls="none", ms=10, color=C_AGENT, mec=bd.INK["surface"],
               mew=1.0, label="agent"),
        Line2D([0], [0], color=C_UP, lw=2.4, label="agent better"),
        Line2D([0], [0], color=C_DOWN, lw=2.4, label="agent worse"),
    ]
    axr.legend(handles=handles, loc="lower left", fontsize=8.0, ncol=1,
               frameon=False, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.3,
               bbox_to_anchor=(0.01, 0.02))

    # caveat, tucked below the right-panel title
    axr.text(0.0, 0.965, "n ≤ 24 origins; one regime event in the window.",
             transform=axr.transAxes, ha="left", va="bottom", fontsize=7.8,
             color=bd.INK["muted"], style="italic")

    fig.text(
        0.008, 0.012,
        "Source: predictions/tsx_ws_eval_2026_weekly/, tsx_logret_21b. Per-origin CRPS via properscoring.crps_ensemble on the sorted 11-point quantile grid\n"
        "vs the realised value, resolved origins only. LEFT: news agent gemini-3.5-flash vs darts_lightgbm_cov over the 24 common resolved origins, split\n"
        "war-window (as_of 2026-02-09..04-13, n=10) vs quiet (n=14). RIGHT: frozen LLM-Process rung to agent on the same base model, leaderboard h=21 means;\n"
        "code-agent win count and one-sided sign test recomputed per origin. Caveat: the war/quiet split is one regime event sampled weekly, not ten\n"
        "independent breaks; dumbbells are horizon-mean deltas.",
        ha="left", va="bottom", fontsize=6.8, color=bd.INK["muted"], linespacing=1.5,
    )

    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.23, wspace=0.42)
    out = Path(__file__).resolve().parent / "fig6_where_agents_earn.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=bd.INK["surface"])
    print(f"wrote {out}")
    print(f"WAR   (n={war['n']}): agent={war['agent']:.7f}  tree={war['tree']:.7f}  agent delta={war_delta * 100:+.1f}%")
    print(f"QUIET (n={quiet['n']}): agent={quiet['agent']:.7f}  tree={quiet['tree']:.7f}  agent delta={quiet_delta * 100:+.1f}%")
    for d in dumbs:
        print(f"{d['blab']:16s} -> {d['alab']:28s}  {d['base']:.6f} -> {d['arm']:.6f}  "
              f"({(d['arm'] / d['base'] - 1) * 100:+.1f}%)  wins={d['wins']}/{d['n']}")
    print(f"code-agent one-sided sign test p = {code_p:.4f}")


if __name__ == "__main__":
    main()
