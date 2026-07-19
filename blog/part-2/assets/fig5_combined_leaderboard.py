"""Part-2 fig5: the full protected-eval scoreboard, 21 methods x 3 horizons.

One panel per horizon (h = 1, 5, 21). Within each panel the 21 methods are
ranked by mean CRPS (best on top) and drawn as a dot on a zoomed value axis, so
the fine ordering at the top of the ladder stays legible where every non-floor
method sits within a few percent of the leader. The three far-worse floors
(naive at every horizon; ETS at h=5 and h=21) are flagged off-scale at the right
margin with their true value rather than blowing out the axis.

Dots are colour-coded by method family (naive floor grey, classical green,
LightGBM blue, LLM-Process purple, LLM-Process +covariates pink -- the Part-1
fig3 legend) and the five AGENTS (news gemini / news sonnet / code sonnet /
adaptive pre / adaptive post) get a distinct orange highlight with bold labels.

Every value is read straight from the final leaderboard CSV
(results/tsx_ws_eval_2026_weekly/leaderboard.csv, 21 rungs per horizon). The
script asserts the anchor facts (h=1 lightgbm_cov leads with the code agent 2nd;
h=5 top five are four LLMP rungs + the adaptive-trained agent 4th, both LightGBMs
14th/17th; h=21 lightgbm_cov leads, news-gemini 3rd, three agents in the top
seven) before writing the PNG.

Run (from the worktree root, with the refreshed store on hand):
``BLOG_WS_DATA_ROOT=.../workshop_experiments/workshop_experiments/data \
  uv run python blog/part-2/assets/fig5_combined_leaderboard.py``
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


LEADERBOARD = bd.RESULTS_DIR / "tsx_ws_eval_2026_weekly" / "leaderboard.csv"

HS = [1, 5, 21]
TARGET_OF = {1: "tsx_logret_1b", 5: "tsx_logret_5b", 21: "tsx_logret_21b"}

# model id -> (compact label, family)
META = {
    "last_value_naive": ("Naive (last value)", "naive"),
    "darts_ets": ("ETS", "classical"),
    "darts_kalman": ("Kalman", "classical"),
    "darts_autoarima": ("AutoARIMA", "classical"),
    "darts_lightgbm": ("LightGBM", "gbm"),
    "darts_lightgbm_cov": ("LightGBM +cov", "gbm"),
    "llmp_quantile_grid_tsx_ws[gemini-3.1-flash-lite-preview]": ("LLMP flash-lite", "llmp"),
    "llmp_quantile_grid_tsx_ws[gemini-3.5-flash]": ("LLMP gemini-3.5", "llmp"),
    "llmp_quantile_grid_tsx_ws[gpt-5.4]": ("LLMP gpt-5.4", "llmp"),
    "llmp_quantile_grid_tsx_ws[claude-sonnet-4-6]": ("LLMP sonnet-4.6", "llmp"),
    "llmp_quantile_grid_tsx_ws[claude-sonnet-5]": ("LLMP sonnet-5", "llmp"),
    "llmp_quantile_grid_tsx_ws_cov[gemini-3.1-flash-lite-preview]": ("LLMP flash-lite +cov", "llmp_cov"),
    "llmp_quantile_grid_tsx_ws_cov[gemini-3.5-flash]": ("LLMP gemini-3.5 +cov", "llmp_cov"),
    "llmp_quantile_grid_tsx_ws_cov[gpt-5.4]": ("LLMP gpt-5.4 +cov", "llmp_cov"),
    "llmp_quantile_grid_tsx_ws_cov[claude-sonnet-4-6]": ("LLMP sonnet-4.6 +cov", "llmp_cov"),
    "llmp_quantile_grid_tsx_ws_cov[claude-sonnet-5]": ("LLMP sonnet-5 +cov", "llmp_cov"),
    "agent_predictor_tsx_analyst_news_gemini-3.5-flash_continuous": ("News agent (gemini-3.5)", "agent"),
    "agent_predictor_tsx_analyst_news_claude-sonnet-4-6_continuous": ("News agent (sonnet-4.6)", "agent"),
    "agent_predictor_tsx_analyst_code_claude-sonnet-4-6_continuous": ("Code agent (sonnet-4.6)", "agent"),
    "agent_predictor_tsx_adaptive_analyst_tsx_strategy_gemini-3.5-flash_continuous": ("Adaptive agent (pre-study)", "agent"),
    "agent_predictor_tsx_adaptive_analyst_tsx_strategy_trained_gemini-3.5-flash_continuous": ("Adaptive agent (post-study)", "agent"),
}

# Family colours: identical to the Part-1 fig3 legend; AGENTS get the orange
# highlight used for the agent throughout Part 2.
FAMILY_COLOR = {
    "naive": bd.INK["muted"],
    "classical": bd.CAT["aqua"],
    "gbm": bd.CAT["blue"],
    "llmp": bd.CAT["violet"],
    "llmp_cov": bd.CAT["magenta"],
    "agent": bd.CAT["orange"],
}
FAMILY_LABEL = {
    "naive": "Naive floor",
    "classical": "Classical (ETS / Kalman / ARIMA)",
    "gbm": "LightGBM (tree)",
    "llmp": "LLM-Process",
    "llmp_cov": "LLM-Process +covariates",
    "agent": "Agent (news / code / adaptive)",
}

# Per-panel zoomed x-window (mean CRPS x10^-3). Anything worse than the cap is a
# far floor and is flagged off-scale at the right margin.
XWIN = {1: (4.88, 5.52), 5: (11.5, 13.62), 21: (16.7, 25.9)}


def _panel_frame(df: pd.DataFrame, h: int) -> pd.DataFrame:
    sub = df[df["horizon"] == h].copy()
    sub["crps_k"] = sub["mean_crps"] * 1000.0
    sub = sub.sort_values("crps_k").reset_index(drop=True)
    sub["label"] = sub["model"].map(lambda m: META[m][0])
    sub["family"] = sub["model"].map(lambda m: META[m][1])
    return sub


def _check_anchors(frames: dict[int, pd.DataFrame]) -> None:
    f1, f5, f21 = frames[1], frames[5], frames[21]
    # h=1: lightgbm_cov leads, code agent 2nd.
    assert f1.loc[0, "model"] == "darts_lightgbm_cov"
    assert abs(f1.loc[0, "crps_k"] - 4.975) < 0.01, f1.loc[0, "crps_k"]
    assert f1.loc[1, "model"] == "agent_predictor_tsx_analyst_code_claude-sonnet-4-6_continuous"
    assert abs(f1.loc[1, "crps_k"] - 4.991) < 0.01, f1.loc[1, "crps_k"]
    # h=5: top five = four LLMP rungs + adaptive-trained agent 4th.
    top5 = f5.head(5)
    assert (top5["family"].isin({"llmp", "llmp_cov", "agent"})).all()
    assert (top5["family"].isin({"llmp", "llmp_cov"})).sum() == 4
    assert f5.loc[3, "model"].endswith("tsx_strategy_trained_gemini-3.5-flash_continuous")
    assert abs(f5.loc[3, "crps_k"] - 11.932) < 0.01, f5.loc[3, "crps_k"]
    lgbm5 = {f5.index[f5["model"] == "darts_lightgbm_cov"][0], f5.index[f5["model"] == "darts_lightgbm"][0]}
    assert lgbm5 == {13, 16}, lgbm5  # 0-based -> ranks 14 and 17
    # h=21: lightgbm_cov leads, news-gemini 3rd, three agents in top seven.
    assert f21.loc[0, "model"] == "darts_lightgbm_cov"
    assert abs(f21.loc[0, "crps_k"] - 17.183) < 0.01, f21.loc[0, "crps_k"]
    assert f21.loc[2, "model"] == "agent_predictor_tsx_analyst_news_gemini-3.5-flash_continuous"
    assert abs(f21.loc[2, "crps_k"] - 17.593) < 0.01, f21.loc[2, "crps_k"]
    assert (f21.head(7)["family"] == "agent").sum() == 3, f21.head(7)["family"].tolist()


def main() -> None:
    bd.apply_style()
    raw = pd.read_csv(LEADERBOARD)
    frames = {h: _panel_frame(raw, h) for h in HS}
    _check_anchors(frames)

    fig, axes = plt.subplots(
        1, 3, figsize=(15.4, 8.9), gridspec_kw={"wspace": 0.62},
    )

    n = 21
    for ax, h in zip(axes, HS):
        sub = frames[h]
        xmin, cap = XWIN[h]
        ax.set_xlim(xmin, cap)
        ax.set_ylim(-0.8, n - 0.2)
        ax.invert_yaxis()  # rank 1 on top

        for rank, row in sub.iterrows():
            y = rank
            fam = row["family"]
            color = FAMILY_COLOR[fam]
            is_agent = fam == "agent"
            v = row["crps_k"]
            off = v > cap

            # Faint leader line across the row.
            ax.plot([xmin, cap], [y, y], color=bd.INK["grid"], lw=0.6, zorder=1)

            # Left-side rank + label (label coloured for agents).
            lab_color = bd.CAT["orange"] if is_agent else bd.INK["primary"]
            weight = "bold" if is_agent else "normal"
            ax.text(-0.012, y, f"{rank + 1:>2}", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=8.0, color=bd.INK["muted"])
            ax.text(-0.055, y, row["label"], transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=8.9, color=lab_color, fontweight=weight)

            if off:
                # Off-scale floor: chevron at the right edge + true value.
                ax.plot(cap - 0.012 * (cap - xmin), y, marker=">", ms=7.5,
                        color=color, mec="none", zorder=4, clip_on=False)
                ax.annotate(f"{v:.2f}", xy=(cap, y), xytext=(-4, 0),
                            textcoords="offset points", ha="right", va="center",
                            fontsize=8.0, color=color, fontweight="bold")
            else:
                ms = 10.5 if is_agent else 7.6
                z = 6 if is_agent else 5
                if is_agent:  # halo ring behind agent dots
                    ax.plot(v, y, marker="o", ms=ms + 5.5, mfc="none",
                            mec=bd.CAT["orange"], mew=1.1, alpha=0.55, zorder=z - 1)
                ax.plot(v, y, marker="o", ms=ms, color=color, mec=bd.INK["surface"],
                        mew=1.1 if is_agent else 0.8, zorder=z)
                # Value label just right of the dot.
                dx = 0.024 * (cap - xmin)
                ax.text(v + dx, y, f"{v:.2f}", ha="left", va="center",
                        fontsize=7.6, color=bd.INK["secondary"])

        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=8.4, length=3)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True, color=bd.INK["grid"], lw=0.6)
        ax.set_axisbelow(True)
        for s in ("left",):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("Mean CRPS  ×10⁻³   (lower is better)", fontsize=8.8)
        ax.set_title(f"h = {h}", fontsize=13.5, fontweight="bold", loc="left", pad=8)
        # small note when floors are clipped off the right
        floors = sub[sub["crps_k"] > cap]
        if len(floors):
            names = ", ".join(f"{r.label} {r.crps_k:.1f}" for _, r in floors.iterrows())
            ax.text(1.0, 1.008, f"off scale:  {names}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.0, color=bd.INK["muted"])

    # ---- Title + legend ----------------------------------------------------
    fig.suptitle("The protected-eval scoreboard: 21 methods, three horizons",
                 x=0.062, y=0.985, ha="left", fontsize=17, fontweight="bold",
                 color=bd.INK["primary"])
    fig.text(0.062, 0.947,
             "Mean CRPS on the leak-safe 2026 S&P/TSX eval, ranked within each horizon. "
             "No family owns every horizon — and the agents (orange) mix in with the leaders "
             "at h=1 and h=21.",
             ha="left", va="top", fontsize=9.6, color=bd.INK["secondary"])

    handles = [
        Line2D([0], [0], marker="o", ls="none", ms=8, mec=bd.INK["surface"],
               mew=0.8, color=FAMILY_COLOR[f], label=FAMILY_LABEL[f])
        for f in ("naive", "classical", "gbm", "llmp", "llmp_cov", "agent")
    ]
    leg = fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8.8,
                     frameon=False, bbox_to_anchor=(0.53, -0.006),
                     handletextpad=0.35, columnspacing=1.4)
    for txt, f in zip(leg.get_texts(), ("naive", "classical", "gbm", "llmp", "llmp_cov", "agent")):
        if f == "agent":
            txt.set_color(bd.CAT["orange"])
            txt.set_fontweight("bold")

    fig.text(
        0.062, 0.028,
        "Source: results/tsx_ws_eval_2026_weekly/leaderboard.csv — mean CRPS per predictor × horizon "
        "(21 rungs each; n_scores = 24 / 22 / 24 resolved weekly origins). Values ×10⁻³. Far-worse floors "
        "(naive at every horizon; ETS at h=5 & h=21) are flagged off-scale at the right rather than compressing "
        "the ladder. Family colours as Part-1 fig. 3; agents highlighted in orange.",
        ha="left", va="top", fontsize=7.0, color=bd.INK["muted"],
    )

    fig.subplots_adjust(left=0.11, right=0.985, top=0.905, bottom=0.10)
    out = Path(__file__).resolve().parent / "fig5_combined_leaderboard.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=bd.INK["surface"])
    print(f"wrote {out}")
    for h in HS:
        print(f"--- h={h} ---")
        for rank, row in frames[h].iterrows():
            print(f"  {rank + 1:2d}  {row['crps_k']:7.3f}  {row['family']:9s}  {row['label']}")


if __name__ == "__main__":
    main()
