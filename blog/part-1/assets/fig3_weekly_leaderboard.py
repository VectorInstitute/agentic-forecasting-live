"""fig3: weekly leaderboards -- 2025 backtest vs 2026 protected eval.

Mean CRPS by predictor x horizon for the two weekly rolling-origin runs. Every
cell is recomputed from the persisted prediction store so the values reproduce
``leaderboard.csv`` exactly. This is a Part-1 figure: it shows the numbers-only
ladder (naive / classical / LightGBM) plus the LLM-Process rungs only -- the news
and code agents are Part-2 material and are deliberately excluded here. Rows are
ordered by mean backtest rank; cell shading encodes within-column rank (darker =
better), so the eval columns visibly reshuffle the backtest order -- the findable
story. Cell text is the mean CRPS.

Run: ``python blog/part-1/assets/fig3_weekly_leaderboard.py``
"""

from __future__ import annotations

import _blogdata as bd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# Predictor -> (display label, family). Order within family is cosmetic; rows are
# re-sorted by performance below.
MODELS = [
    ("last_value_naive", "Naive (last value)", "naive"),
    ("darts_ets", "ETS", "classical"),
    ("darts_kalman", "Kalman", "classical"),
    ("darts_autoarima", "AutoARIMA", "classical"),
    ("darts_lightgbm", "LightGBM", "gbm"),
    ("darts_lightgbm_cov", "LightGBM +cov", "gbm"),
    ("llmp_quantile_grid_tsx_ws[gemini-3.1-flash-lite-preview]", "LLMP flash-lite", "llmp"),
    ("llmp_quantile_grid_tsx_ws[gemini-3.5-flash]", "LLMP gemini-3.5", "llmp"),
    ("llmp_quantile_grid_tsx_ws[gpt-5.4]", "LLMP gpt-5.4", "llmp"),
    ("llmp_quantile_grid_tsx_ws[claude-sonnet-4-6]", "LLMP sonnet-4.6", "llmp"),
    ("llmp_quantile_grid_tsx_ws[claude-sonnet-5]", "LLMP sonnet-5", "llmp"),
    ("llmp_quantile_grid_tsx_ws_cov[gemini-3.1-flash-lite-preview]", "LLMP flash-lite +cov", "llmp_cov"),
    ("llmp_quantile_grid_tsx_ws_cov[gemini-3.5-flash]", "LLMP gemini-3.5 +cov", "llmp_cov"),
    ("llmp_quantile_grid_tsx_ws_cov[gpt-5.4]", "LLMP gpt-5.4 +cov", "llmp_cov"),
    ("llmp_quantile_grid_tsx_ws_cov[claude-sonnet-4-6]", "LLMP sonnet-4.6 +cov", "llmp_cov"),
    ("llmp_quantile_grid_tsx_ws_cov[claude-sonnet-5]", "LLMP sonnet-5 +cov", "llmp_cov"),
]

FAMILY_COLOR = {
    "naive": bd.INK["muted"],
    "classical": bd.CAT["aqua"],
    "gbm": bd.CAT["blue"],
    "llmp": bd.CAT["violet"],
    "llmp_cov": bd.CAT["magenta"],
}
FAMILY_LABEL = {
    "naive": "Naive floor",
    "classical": "Classical (ETS / Kalman / ARIMA)",
    "gbm": "LightGBM",
    "llmp": "LLM-Process",
    "llmp_cov": "LLM-Process +covariates",
}

SPECS = [
    ("tsx_ws_backtest_2025_weekly", "2025 backtest"),
    ("tsx_ws_eval_2026_weekly", "2026 protected eval"),
]
COLS = [(spec, tgt) for spec, _ in SPECS for tgt in bd.TARGETS]
COL_LABELS = [f"h={bd.HORIZON_OF[t]}" for _ in SPECS for t in bd.TARGETS]

# Blue ordinal ramp: dark (best rank) -> light (worst rank).
RANK_CMAP = LinearSegmentedColormap.from_list("rank_blue", ["#0d366b", "#256abf", "#86b6ef", "#cde2fb"])


def main() -> None:
    bd.apply_style()

    n_models = len(MODELS)
    crps = np.full((n_models, len(COLS)), np.nan)
    for i, (model, _, _) in enumerate(MODELS):
        for j, (spec, tgt) in enumerate(COLS):
            mc, n = bd.mean_crps(spec, model, tgt)
            crps[i, j] = mc if (n > 0 and np.isfinite(mc)) else np.nan

    # Within-column ranks (1 = best); NaN ranked last.
    ranks = np.full_like(crps, np.nan)
    for j in range(crps.shape[1]):
        col = crps[:, j]
        order = np.argsort(np.where(np.isnan(col), np.inf, col))
        r = np.empty(n_models)
        r[order] = np.arange(1, n_models + 1)
        ranks[:, j] = r

    # Row order: mean backtest rank (columns 0..2), best on top.
    bt_rank = np.nanmean(ranks[:, :3], axis=1)
    row_order = np.argsort(bt_rank)

    fig, ax = plt.subplots(figsize=(12.2, 8.0))
    for rr, i in enumerate(row_order):
        y = n_models - 1 - rr
        for j in range(len(COLS)):
            v = crps[i, j]
            rank = ranks[i, j]
            frac = (rank - 1) / (n_models - 1)
            color = RANK_CMAP(frac)
            ax.add_patch(
                plt.Rectangle(
                    (j, y),
                    1.0,
                    1.0,
                    facecolor=color,
                    edgecolor=bd.INK["surface"],
                    lw=2.4,
                ),
            )
            txt = "n/a" if np.isnan(v) else f"{v * 1000:.1f}"
            ax.text(
                j + 0.5,
                y + 0.5,
                txt,
                ha="center",
                va="center",
                fontsize=9.2,
                color="#ffffff" if frac < 0.45 else bd.INK["primary"],
                fontweight="bold" if rank <= 3 else "normal",
            )
        # Row label + family swatch.
        _, label, family = MODELS[i]
        ax.add_patch(plt.Rectangle((-0.24, y + 0.3), 0.12, 0.4, facecolor=FAMILY_COLOR[family], lw=0))
        ax.text(-0.34, y + 0.5, label, ha="right", va="center", fontsize=9.3, color=bd.INK["primary"])

    # Column headers (two grouped windows).
    for j, cl in enumerate(COL_LABELS):
        ax.text(j + 0.5, n_models + 0.1, cl, ha="center", va="bottom", fontsize=9.8, color=bd.INK["secondary"])
    ax.text(
        1.5,
        n_models + 0.85,
        SPECS[0][1],
        ha="center",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color=bd.INK["primary"],
    )
    ax.text(
        4.5,
        n_models + 0.85,
        SPECS[1][1],
        ha="center",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color=bd.INK["primary"],
    )

    # Divider between the two windows.
    ax.plot([3, 3], [0, n_models], color=bd.INK["axis"], lw=1.8, zorder=6)

    # Family legend as a manual swatch grid below the grid (2 columns x 3 rows).
    fams = list(FAMILY_LABEL)
    xpos = [-3.4, -3.4, -3.4, 1.9, 1.9, 1.9]
    ypos = [-1.15, -1.75, -2.35, -1.15, -1.75, -2.35]
    for f, xp, yp in zip(fams, xpos, ypos):
        ax.add_patch(plt.Rectangle((xp, yp), 0.16, 0.4, facecolor=FAMILY_COLOR[f], lw=0))
        ax.text(xp + 0.28, yp + 0.2, FAMILY_LABEL[f], ha="left", va="center", fontsize=9, color=bd.INK["secondary"])

    ax.set_xlim(-3.6, len(COLS) + 0.1)
    ax.set_ylim(-2.7, n_models + 2.4)
    ax.axis("off")

    ax.text(
        -3.6,
        n_models + 1.95,
        "The backtest order does not survive the protected window",
        fontsize=14,
        fontweight="bold",
        color=bd.INK["primary"],
        ha="left",
    )

    fig.text(
        0.5,
        0.005,
        "Mean CRPS ×10⁻³ (lower = better); shading = within-column rank (darker = better). "
        "Recomputed from the prediction store with properscoring.crps_ensemble (reproduces "
        "leaderboard.csv). 'n/a' = one degenerate origin gave a non-finite score.",
        fontsize=7.5,
        color=bd.INK["muted"],
        ha="center",
    )

    out = bd.savefig(fig, "fig3_weekly_leaderboard.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
