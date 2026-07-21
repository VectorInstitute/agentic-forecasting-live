# Part-2 figure captions and data provenance

Reader-facing captions live in `../post.md`. This file keys every figure by its
**reading-order number** (which no longer matches the `figN_` script prefixes —
the table below is the mapping), records the provenance and computed values
that were deliberately removed from the rendered PNGs (maximal chrome-out), and
pins the numbers.

All six scripts import the Part-1 `_blogdata.py` helper, so palette, fonts, and
the 220-dpi output match Part 1 exactly. Realised returns come from the same
leak-safe TSX data service (`workshop_experiments.data_tsx.
build_tsx_workshop_service`, Yahoo `^GSPTSE` adjusted close) used throughout.
The data store is co-located in this clone; no env var is needed (set
`BLOG_WS_DATA_ROOT` only when rendering against a different store). Paths below
are relative to that store.

| Reading order | Script / PNG |
|---|---|
| Figure 1 | `fig1_agent_anatomy` |
| Figure 2 | `fig5_combined_leaderboard` |
| Figure 3 | `fig6_where_agents_earn` |
| Figure 4 | `fig7_sentinel_bands` |
| Figure 5 | `fig3_divergence_sentinel` |
| Figure 6 | `fig2_scenario_card` |
| — retired | `fig4_adaptive_prepost` (no longer referenced by the post; script and PNG kept on disk) |

---

## Figure 1 — `fig1_agent_anatomy.png` (`fig1_agent_anatomy.py`)

PNG title: *Anatomy of one agent forecast*. News analyst agent (Claude
Sonnet-4.6) forecasting the S&P/TSX **21-business-day log return** from origin
**2026-03-30** (forecast date 2026-04-28): six `search_web` calls (86.5 s
wall), four load-bearing rationale factors (BoC hold at **2.25 %**; CPI
**1.8 %** vs **−84K** February jobs; Middle-East commodity volatility; **25 %
US tariffs** driving the negative skew), and the emitted 11-point quantile grid
— **Q0.05 = −7.5 %**, **median = +1.0 %**, **Q0.80 = +5.0 %**, **Q0.95 =
+9.0 %**. Realised 21-day move **+5.1 %** (simple; log +4.96 %) lands
essentially at Q0.80. Search-trail labels paraphrase the verbatim `tool_calls`
titles.

**Source:** `data/predictions/tsx_ws_eval_2026_weekly/`
`agent_predictor_tsx_analyst_news_claude-sonnet-4-6_continuous/tsx_logret_21b/`
`2026-03-30.yaml` (`tool_calls`, `metadata.rationale`, `payload.quantiles`);
realised value from the TSX data service (`_blogdata.realized`).

## Figure 2 — `fig5_combined_leaderboard.png` (`fig5_combined_leaderboard.py`)

PNG title: *Mean CRPS leaderboard: 21 methods, three horizons*. All 21 methods
× 3 horizons read straight from the final `leaderboard.csv`, ranked per panel
on a zoomed value axis. Far-worse floors are held off-scale with chevron marks
carrying their true values — h=1: naive **9.28**; h=5: ETS **16.21**, naive
**23.30**; h=21: ETS **32.57**, naive **42.33** (all ×10⁻³). Family colours as
the Part-1 fig-3 legend; the five agents (news gemini-3.5, news sonnet-4.6,
code sonnet-4.6, adaptive pre/post) highlighted in orange.

Anchor facts (asserted in-script before render): h=1 leader
`darts_lightgbm_cov` **4.975** with the **code agent 2nd at 4.991**; h=5 top
five all LLM-based with both LightGBMs at **14th/17th**; h=21 leader
`darts_lightgbm_cov` **17.18** with **news-gemini 3rd at 17.59** and three
agents in the top seven. n_scores = **24 / 22 / 24** resolved weekly origins.

**Source:** `results/tsx_ws_eval_2026_weekly/leaderboard.csv` (`mean_crps` per
`model` × `horizon`; the CSV is the authoritative scoreboard).

## Figure 3 — `fig6_where_agents_earn.png` (`fig6_where_agents_earn.py`)

PNG title: *Mean CRPS at h = 21: war vs quiet split, and agent vs frozen base*.

**Left** — news agent (`..._analyst_news_gemini-3.5-flash_continuous`) vs
`darts_lightgbm_cov` over the 24 common resolved h=21 origins, split on the war
window (origins 2026-02-09 → 04-13, n = 10) vs quiet weeks (n = 14): war
window agent **0.02178** vs tree **0.02449** (**−11 %**, agent better); quiet
agent **0.01460** vs tree **0.01196** (**+22 %**, agent worse).

**Right** — paired same-model dumbbells, frozen LLMP → agent, leaderboard h=21
means: news gemini-3.5 **−3.7 %**; adaptive pre **+1.7 %**; adaptive post
**+2.8 %**; news sonnet-4.6 **−0.0 %** (flat); code sonnet-4.6 **−3.1 %**,
better at **18/24** origins (one-sided sign test p ≈ 0.011 — the only pair that
separates, and it does not survive a magnitude-weighting or overlap-respecting
test; see post).

**Caveats (now caption-side):** n ≤ 24 origins at every cut; the war/quiet
split is one regime event sampled weekly, not ten independent breaks.
**Source:** `predictions/tsx_ws_eval_2026_weekly/`, `tsx_logret_21b/`;
per-origin CRPS via `properscoring.crps_ensemble`; the script pins the
war/quiet means, all dumbbell means, and the 18/24 count, and asserts its
recompute reproduces them before writing the PNG.

## Figure 4 — `fig7_sentinel_bands.png` (`fig7_sentinel_bands.py`)

PNG title: *10–90 forecast bands vs the realized 21-day return*. Both methods'
10–90 prediction intervals per forecast origin over the 24 common resolved
weekly origins (2026-01-05 → 2026-06-15), realised 21-day return overlaid,
war-window origins (2026-02-09 → 04-13) shaded.

Computed stats (printed by the script; caption-side by design): the tree's band
width varies only **1.7×** min-to-max across the half-year while the agent's
varies **3.5×**; the agent's band runs **1.63×** the tree's width at the median
origin and peaks at **3.00×** at the war-trough origin **2026-03-16**. The
agent's *median* width inside the war window is **not** elevated vs quiet weeks
— the break is distinguished by the spike, not a level shift.

**Source:** `predictions/tsx_ws_eval_2026_weekly/`, both rungs (news gemini
agent and `darts_lightgbm_cov`), `tsx_logret_21b/<origin>.yaml` — bands read
off the persisted 11-point quantile grids (q0.1–q0.9); realised values from the
TSX data service.

## Figure 5 — `fig3_divergence_sentinel.png` (`fig3_divergence_sentinel.py`)

PNG title: *When the agent and the tree disagree*. **Main panel:** per-origin
divergence D between the gemini news agent's and LightGBM+cov's 11-point
quantile grids (D = mean absolute difference across the 11 levels), war window
shaded. Top-4 origins: **2026-03-16 (D=0.0461)**, **2026-03-23 (0.0338)**,
**2026-06-08 (0.0228)**, **2026-02-23 (0.0223)**. The June origin is the agent
pricing a post-record-high correction (record 35,217; sticky 3.2 % CPI;
Q0.05/Q0.95 widened to −7.2 %/+6.8 % in its rationale) that never confirmed.
**Inset:** mean h=21 CRPS — always LightGBM+cov **0.01718**, always agent
**0.01759**, divergence-gated router **0.01688** (in-sample median threshold
D = 0.0170). The inset's zoomed non-zero axis start (16.5 ×10⁻³) is encoded on
the figure itself (break glyphs + "16.5 (axis start)" tick) so a screenshot
cannot mislead.

**Exploratory caveats (caption-side):** n = 24 origins; the router threshold is
the in-sample median, so the CRPS gain is illustrative, not out-of-sample —
about 28 % of random gates of the same size do as well (see post).
**Source:** `predictions/tsx_ws_eval_2026_weekly/`, `tsx_logret_21b/`; the
script pins the three policy means (0.0171834 / 0.0175927 / 0.0168775) and
asserts its recompute reproduces them to 1e-5 before writing the PNG.

## Figure 6 — `fig2_scenario_card.png` (`fig2_scenario_card.py`)

PNG title: *Right call, wrong reason — a scenario write-up graded against what
happened* (the mechanism-mismatch headline lives in the title by design).
Scenario set issued **2026-03-31**, 60-business-day outlook:

| Scenario | Prob. | 60-day outlook |
|---|---|---|
| Commodity-Led Defensive Rotation *(base case)* | **0.55** | +3 % to +5 % |
| US-Trade Policy Stalls / Tariff Risk | **0.30** | −4 % to −7 % |
| BoC Policy Disappointment | **0.15** | −2 % to +1 % |

Judge (Claude Sonnet-4.6): **calibration 5/5, drivers 3/5, specificity 4/5**.
Realised cumulative log returns from 2026-03-31: **+2.60 %** at 5 days
(through 2026-04-08), **+3.65 %** at 21 days (2026-04-30), **+6.35 %** at 60
days (2026-06-25). The mechanism mismatch: the base case assumed persistent
Middle-East friction keeping oil bid; the rally came from a ceasefire relief
bounce — hence the withheld drivers credit.

**Source:** `data/scenarios/2026-03-31/writeup.md` and
`data/scenarios/2026-03-31/judge.yaml` (`verdict` scores +
`realized_outcome`).

---

### Retired — `fig4_adaptive_prepost.png` (`fig4_adaptive_prepost.py`)

Removed from the post in the final compression pass (a second null scoreboard
after Figure 2 added cost without information; the adaptive section now carries
its one transferable finding in prose). The script still runs and its pinned
numbers remain valid: pre/post mean CRPS 0.00522/0.00527 (h=1, n=24),
0.01206/0.01193 (h=5, n=22), 0.01857/0.01876 (h=21, n=24); per-origin wins a
coin flip at every horizon; all 25 graduated corrections confirmed same-session
(`opened_on` = `confirmed_on` = 2026-07-18). Note: it has **not** been migrated
to the chrome-out `bd.savefig` contract.

### Regenerating

From the repo root (data store co-located; no env var needed):

```
uv run python blog/part-2/assets/fig1_agent_anatomy.py
uv run python blog/part-2/assets/fig2_scenario_card.py
uv run python blog/part-2/assets/fig3_divergence_sentinel.py
uv run python blog/part-2/assets/fig5_combined_leaderboard.py
uv run python blog/part-2/assets/fig6_where_agents_earn.py
uv run python blog/part-2/assets/fig7_sentinel_bands.py
```

Each writes its PNG beside the script through `_blogdata.savefig`, which
enforces the shared contract: 220 dpi (Part-1 parity), no figure-level text
below the axes block (footnotes are banned — they belong in the post caption
and here), no text under the 10 pt floor, and saved width ≤ 1.02 × figsize ×
220 dpi. Scripts print `CAPTION:` lines at save time with any computed values a
caption needs — paste from there, never hand-transcribe.
