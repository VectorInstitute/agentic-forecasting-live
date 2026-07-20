"""Part-2 fig7 (Figure 4): the sentinel, rolled out over time.

Both forecasters' 10-90 prediction intervals for the 21-day TSX log return,
origin by origin across the 24 weekly origins of the protected 2026 window
(2026-01-05 .. 2026-06-15), with the realised 21-day return overlaid and the
war window (origins 2026-02-09 .. 2026-04-13) shaded.

The point of the figure is the *shape* of the two bands, not their level.
darts_lightgbm_cov builds its interval from trailing volatility, so its band is
nearly flat all half-year. The gemini-3.5 news agent widens sharply as the
regime breaks: its 10-90 width runs ~1.63x the tree's at the median origin and
peaks at ~3.0x at the war trough (origin 2026-03-16). That is the "agent as
sentinel" claim in one picture -- the agent's useful signal is how uncertain it
says it is, not which way it leans.

Widths are recomputed from the persisted 11-point quantile grids (q0.9 - q0.1
off the sorted grid) over the common resolved origins; the script asserts the
median ratio, the peak ratio and its origin, and n = 24 before writing the PNG.

House rule (chrome-out): the image carries only title, axis labels/ticks,
legend and data marks. Source line and caveats live in the markdown caption.

Run (from the worktree root, with the refreshed store on hand):
``BLOG_WS_DATA_ROOT=.../workshop_experiments/workshop_experiments/data \
  uv run python blog/part-2/assets/fig7_sentinel_bands.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

_P1_ASSETS = Path(__file__).resolve().parents[2] / "part-1" / "assets"
sys.path.insert(0, str(_P1_ASSETS))

import _blogdata as bd  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SPEC = "tsx_ws_eval_2026_weekly"
T21 = "tsx_logret_21b"
WAR_START, WAR_END = pd.Timestamp("2026-02-09"), pd.Timestamp("2026-04-13")

AGENT = "agent_predictor_tsx_analyst_news_gemini-3.5-flash_continuous"
TREE = "darts_lightgbm_cov"

# Pinned references, recomputed below on the 24 common resolved origins.
REF_N = 24
REF_MEDIAN_RATIO = 1.634
REF_PEAK_RATIO = 3.001
REF_PEAK_ORIGIN = pd.Timestamp("2026-03-16")

C_AGENT = bd.CAT["orange"]
C_TREE = bd.CAT["blue"]

# Index of q0.1 / q0.9 in the sorted 11-point grid
# ('0.05','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','0.95').
I_Q10, I_Q90 = 1, 9


def _band(model: str) -> pd.DataFrame:
    """Per-origin q0.1 / q0.9 / median and forecast date for one model."""
    preds = bd.load_predictions(SPEC, model, T21)
    assert not preds.empty, model
    q = np.vstack(preds["quantiles"].to_numpy())
    assert q.shape[1] == 11, q.shape
    return pd.DataFrame(
        {
            "lo": q[:, I_Q10],
            "hi": q[:, I_Q90],
            "median": preds["median"].to_numpy(),
            "forecast_date": preds["forecast_date"].to_numpy(),
        },
        index=preds.index,
    )


def main() -> None:
    bd.apply_style()

    agent, tree = _band(AGENT), _band(TREE)
    origins = agent.index.intersection(tree.index)
    agent, tree = agent.loc[origins], tree.loc[origins]

    w_agent = (agent["hi"] - agent["lo"]).astype(float)
    w_tree = (tree["hi"] - tree["lo"]).astype(float)
    ratio = w_agent / w_tree

    median_ratio = float(ratio.median())
    peak_ratio = float(ratio.max())
    peak_origin = pd.Timestamp(ratio.idxmax())

    assert len(origins) == REF_N, len(origins)
    assert abs(median_ratio - REF_MEDIAN_RATIO) <= 5e-3, median_ratio
    assert abs(peak_ratio - REF_PEAK_RATIO) <= 5e-3, peak_ratio
    assert peak_origin == REF_PEAK_ORIGIN, peak_origin

    # Realised 21-day log return for each origin's forecast date.
    look = bd.realized(T21)
    real = pd.Series(
        [float(look.loc[d]) for d in agent["forecast_date"]],
        index=origins,
        dtype=float,
    )

    x = origins.to_pydatetime()
    pct = 100.0  # plot in percent

    fig, ax = plt.subplots(figsize=(12.0, 6.5))

    # War-window shading (behind everything).
    ax.axvspan(WAR_START, WAR_END, color=bd.INK["grid"], alpha=0.55, lw=0, zorder=0)

    # Agent band (wide) first, tree band on top so both edges stay readable.
    ax.fill_between(x, agent["lo"] * pct, agent["hi"] * pct, color=C_AGENT,
                    alpha=0.22, lw=0, zorder=2)
    ax.plot(x, agent["lo"] * pct, color=C_AGENT, lw=1.9, zorder=3)
    ax.plot(x, agent["hi"] * pct, color=C_AGENT, lw=1.9, zorder=3)

    ax.fill_between(x, tree["lo"] * pct, tree["hi"] * pct, color=C_TREE,
                    alpha=0.30, lw=0, zorder=4)
    ax.plot(x, tree["lo"] * pct, color=C_TREE, lw=1.9, zorder=5)
    ax.plot(x, tree["hi"] * pct, color=C_TREE, lw=1.9, zorder=5)

    # Realised outcome.
    ax.plot(x, real * pct, color=bd.INK["primary"], lw=1.9, marker="o", ms=4.6,
            mec=bd.INK["surface"], mew=0.8, zorder=7)
    ax.axhline(0.0, color=bd.INK["axis"], lw=0.9, alpha=0.7, zorder=1)

    # Peak-ratio callout at the war trough.
    px = peak_origin.to_pydatetime()
    ptop = float(agent.loc[peak_origin, "hi"]) * pct
    # Two stacked lines: headline above, context line below it, both clear of
    # the band top (va anchors chosen so the pair can never overlap).
    ax.annotate(
        f"{peak_ratio:.1f}× the tree's width",
        xy=(px, ptop), xytext=(px, ptop + 7.2),
        ha="center", va="bottom", fontsize=13, fontweight="bold", color=C_AGENT,
        arrowprops=dict(arrowstyle="-", color=C_AGENT, lw=1.2, shrinkB=3),
        zorder=8,
    )
    ax.text(px, ptop + 6.9, f"war trough · origin {peak_origin:%Y-%m-%d}",
            ha="center", va="top", fontsize=12, color=bd.INK["muted"], zorder=8)

    ax.set_xlim(origins.min() - pd.Timedelta(days=7), origins.max() + pd.Timedelta(days=7))
    ax.set_ylim(-13.5, 21.5)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlabel("Forecast origin (weekly, 2026)", fontsize=13.5)
    ax.set_ylabel("21-day log return  (%)", fontsize=13.5)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(False)
    ax.set_title(
        "Figure 4. The tree's uncertainty barely moves; the agent's runs wider — and spikes at the trough",
        fontsize=17.5, fontweight="bold", loc="left", pad=14,
    )

    handles = [
        Patch(facecolor=C_AGENT, alpha=0.30, edgecolor=C_AGENT, lw=1.6,
              label=f"News agent 10–90 interval (median {median_ratio:.2f}× the tree's)"),
        Patch(facecolor=C_TREE, alpha=0.38, edgecolor=C_TREE, lw=1.6,
              label="LightGBM +cov 10–90 interval"),
        Line2D([0], [0], color=bd.INK["primary"], lw=1.9, marker="o", ms=5,
               mec=bd.INK["surface"], mew=0.8, label="Realized 21-day return"),
        Patch(facecolor=bd.INK["grid"], alpha=0.55, lw=0,
              label="War window (origins 2026-02-09 – 04-13)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=12, frameon=True,
              framealpha=0.92, edgecolor=bd.INK["grid"], handlelength=1.5,
              borderaxespad=0.5, labelspacing=0.45)

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "fig7_sentinel_bands.png"
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=bd.INK["surface"])
    print(f"wrote {out}")
    print(f"n origins            = {len(origins)}  ({origins.min():%Y-%m-%d} .. {origins.max():%Y-%m-%d})")
    print(f"median width ratio   = {median_ratio:.4f}")
    print(f"peak   width ratio   = {peak_ratio:.4f}  at origin {peak_origin:%Y-%m-%d}")
    print(f"agent 10-90 width    : {w_agent.min():.4f} .. {w_agent.max():.4f}")
    print(f"tree  10-90 width    : {w_tree.min():.4f} .. {w_tree.max():.4f}")


if __name__ == "__main__":
    main()
