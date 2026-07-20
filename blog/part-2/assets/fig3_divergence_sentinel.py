"""Part-2 fig3: agent-vs-tree divergence as a regime sentinel (exploratory).

Main panel: per-origin divergence D between the gemini news agent's and
LightGBM+cov's 11-point quantile grids on the protected 2026 eval, h=21, where
D = mean absolute difference across the 11 quantile levels. The 2026 war window
is shaded; the four highest-D origins all sit in or at it.

Side panel: mean CRPS of three policies over the same origins -- always
LightGBM+cov, always agent, and a divergence-gated router (use the agent when D
exceeds its in-sample median, else LightGBM+cov). Exploratory: n=24 and the
threshold is in-sample.

Population: all 24 weekly origins with resolved 21b outcomes, 2026-01-05 ->
2026-06-15 (matches the refreshed eval leaderboard's n_scores=24 at h=21).
Three of the four highest-D origins sit in/at the war window; the fourth,
2026-06-08, is the agent pricing a post-record-high correction (annotated).

Everything is recomputed from the prediction stores with the same
``crps_ensemble`` call as Part-1; the script asserts the three policy means
reproduce the pinned n=24 values to 1e-5 before writing the PNG.

Run: ``python blog/part-2/assets/fig3_divergence_sentinel.py``
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
import properscoring as ps  # noqa: E402


SPEC = "tsx_ws_eval_2026_weekly"
TARGET = "tsx_logret_21b"
AGENT = "agent_predictor_tsx_analyst_news_gemini-3.5-flash_continuous"
TREE = "darts_lightgbm_cov"

WAR_START, WAR_END = pd.Timestamp("2026-02-09"), pd.Timestamp("2026-04-13")

# Pinned n=24 values the recompute must reproduce (tolerance 1e-5).
REF = {"tree": 0.0171834, "agent": 0.0175927, "router": 0.0168775}

C_TREE = bd.CAT["blue"]
C_AGENT = bd.CAT["orange"]
C_ROUTER = bd.CAT["violet"]


def _table() -> pd.DataFrame:
    pa = bd.load_predictions(SPEC, AGENT, TARGET)
    pt = bd.load_predictions(SPEC, TREE, TARGET)
    look = bd.realized(TARGET)
    rows = []
    for origin in sorted(pa.index.intersection(pt.index)):
        ra, rt = pa.loc[origin], pt.loc[origin]
        fdate = ra["forecast_date"]
        if fdate not in look.index:
            continue
        qa, qt = ra["quantiles"], rt["quantiles"]
        assert len(qa) == 11 and len(qt) == 11
        y = float(look.loc[fdate])
        rows.append(
            {
                "origin": origin,
                "D": float(np.mean(np.abs(qa - qt))),
                "crps_agent": float(ps.crps_ensemble(y, qa)),
                "crps_tree": float(ps.crps_ensemble(y, qt)),
            },
        )
    return pd.DataFrame(rows).set_index("origin")


def main() -> None:
    bd.apply_style()
    df = _table()
    assert len(df) == 24, f"expected 24 origins, got {len(df)}"

    med = float(df["D"].median())
    gated = np.where(df["D"] > med, df["crps_agent"], df["crps_tree"])
    bars = {
        "tree": float(df["crps_tree"].mean()),
        "agent": float(df["crps_agent"].mean()),
        "router": float(gated.mean()),
    }
    for key, ref in REF.items():
        assert abs(bars[key] - ref) <= 1e-5, f"{key}: {bars[key]:.7f} != ref {ref}"

    top4 = df.sort_values("D", ascending=False).head(4)

    fig, (ax, axb) = plt.subplots(
        1, 2, figsize=(12.6, 7.6), gridspec_kw={"width_ratios": [2.05, 1.0], "wspace": 0.30},
    )

    # ---- Main panel: D per origin -----------------------------------------
    ax.axvspan(WAR_START, WAR_END, color=bd.STATUS["critical"], alpha=0.09, lw=0, zorder=0)
    ax.text(WAR_START + (WAR_END - WAR_START) / 2, 0.0485, "2026 war window",
            ha="center", va="top", fontsize=12.5, color=bd.STATUS["critical"], alpha=0.85)

    ax.plot(df.index, df["D"], color=C_ROUTER, lw=1.8, zorder=3,
            marker="o", ms=4.4, mfc=C_ROUTER, mec="none")
    ax.scatter(top4.index, top4["D"], s=96, facecolor="none",
               edgecolor=bd.INK["primary"], lw=1.5, zorder=5)
    ax.axhline(med, color=bd.INK["muted"], lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.text(df.index[-1], med - 0.0022, f"in-sample median threshold ({med:.3f})",
            fontsize=12, color=bd.INK["muted"], ha="right", va="top", zorder=6,
            bbox=dict(facecolor=bd.INK["surface"], edgecolor="none", pad=1.2))

    ax.set_ylabel("Divergence D  (mean |Δq| across 11 quantiles)", fontsize=13)
    ax.set_ylim(0, 0.050)
    ax.margins(x=0.03)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(labelsize=12)
    ax.set_title("When the agent and the tree disagree",
                 fontsize=18, fontweight="bold", loc="left", pad=10)
    ax.annotate("4 highest-D origins", xy=(top4.index[0], top4["D"].iloc[0]),
                xytext=(pd.Timestamp("2026-04-20"), 0.042),
                fontsize=12, color=bd.INK["secondary"],
                arrowprops=dict(arrowstyle="-", color=bd.INK["axis"], lw=0.9))
    # The one top-4 firing outside the war window: 2026-06-08, where the agent
    # priced a correction off the record high (sticky 3.2% CPI, Q05/Q95 widened
    # to -7.2%/+6.8% in its rationale).
    jun8 = pd.Timestamp("2026-06-08")
    ax.annotate("post-record-high\ncorrection priced",
                xy=(jun8, float(df.loc[jun8, "D"])),
                xytext=(pd.Timestamp("2026-05-12"), 0.0305),
                fontsize=11.5, color=bd.INK["secondary"], ha="center",
                linespacing=1.25,
                arrowprops=dict(arrowstyle="-", color=bd.INK["axis"], lw=0.9))

    # ---- Side panel: three policy bars ------------------------------------
    # The three means differ by only ~4%, so a zero-baseline axis would hide the
    # router's win. We zoom the y-axis to start at 16.5 (×10⁻³) and draw an
    # explicit broken-axis cue so the ordering reads honestly.
    labels = ["Always\nLightGBM\n+cov", "Always\nnews\nagent", "Divergence-\ngated\nrouter"]
    vals = [bars["tree"], bars["agent"], bars["router"]]
    colors = [C_TREE, C_AGENT, C_ROUTER]
    xs = np.arange(3)
    vals_k = [v * 1000 for v in vals]
    Y0 = 16.5  # zoomed baseline (×10⁻³)
    axb.bar(xs, vals_k, width=0.62, color=colors, zorder=3)
    for x, v in zip(xs, vals_k):
        axb.text(x, v + 0.022, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=12, fontweight="bold", color=bd.INK["primary"])
    axb.set_xticks(xs)
    axb.set_xticklabels(labels, fontsize=11.5)
    axb.set_ylim(Y0, 17.95)
    axb.set_yticks([16.5, 17.0, 17.5])
    axb.set_ylabel("Mean CRPS ×10⁻³  (h=21)", fontsize=13)
    axb.tick_params(axis="y", labelsize=12)
    axb.grid(axis="x", visible=False)
    # Broken-axis cue: diagonal slashes at the zoomed baseline + an explicit,
    # in-plot label (kept off the crowded x-tick region below the bars).
    _bk = dict(transform=axb.transAxes, color=bd.INK["axis"], clip_on=False, lw=1.4)
    axb.plot([-0.045, 0.045], [-0.010, 0.010], **_bk)
    axb.plot([-0.045, 0.045], [0.016, 0.036], **_bk)
    axb.set_title("Route on disagreement (in-sample)",
                  fontsize=13, fontweight="bold", loc="left", pad=12)
    axb.text(0.03, 0.965,
             "exploratory: n=24, in-sample threshold\n"
             "y-axis zoomed — starts at 16.5 ×10⁻³ (bars ≈4% span)",
             transform=axb.transAxes, fontsize=10.5, color=bd.INK["muted"],
             ha="left", va="top", style="italic", linespacing=1.3)

    fig.text(
        0.02, -0.02,
        "Source: predictions/tsx_ws_eval_2026_weekly/ (news agent gemini-3.5-flash vs darts_lightgbm_cov), tsx_logret_21b, all resolved weekly\n"
        "origins 2026-01-05 to 2026-06-15 (n=24). D = mean absolute difference between the two 11-point quantile grids; CRPS per origin via\n"
        "properscoring.crps_ensemble. Values ×10⁻³ on bars. Shaded: 2026 war window (style as Part-1 fig. 4).",
        fontsize=10, color=bd.INK["muted"], ha="left", linespacing=1.4,
    )

    out = Path(__file__).resolve().parent / "fig3_divergence_sentinel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=bd.INK["surface"])
    print(f"wrote {out}")
    print(f"median D = {med:.6f}")
    print("top-4 D origins:")
    for o, r in top4.iterrows():
        print(f"  {o.date()}  D={r['D']:.6f}")
    for k in ("tree", "agent", "router"):
        print(f"{k:7s} mean CRPS = {bars[k]:.7f} (ref {REF[k]})")


if __name__ == "__main__":
    main()
