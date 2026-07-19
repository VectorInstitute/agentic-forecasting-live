# Part-2 figure captions and data provenance

Both figures read directly from repo artefacts and invent no numbers. The
generating scripts (`fig1_agent_anatomy.py`, `fig2_scenario_card.py`) import the
Part-1 `_blogdata.py` helper, so the palette, fonts, and 220-dpi output match
Part-1 exactly. Realised returns come from the same leak-safe TSX data service
(`workshop_experiments.data_tsx.build_tsx_workshop_service`, Yahoo `^GSPTSE`
adjusted close) used throughout Part 1.

Data-root note: these were rendered against the refreshed prediction/scenario
store via `BLOG_WS_DATA_ROOT=.../workshop_experiments/workshop_experiments/data`
(env override added to `_blogdata.py`); paths below are relative to that store.

---

## fig1 — `fig1_agent_anatomy.png`

Anatomy of a single news-analyst agent forecast (Claude Sonnet-4.6), issued at
origin **2026-03-30** for the S&P/TSX Composite **21-business-day log return**
(forecast date 2026-04-28), read left→right as gather → reason → forecast:

- **Gather** — the six `search_web` calls the agent issued (86.5 s wall), shown
  as a vertical tool trail. Trail labels paraphrase the verbatim `tool_calls`
  titles: (1) *Bank of Canada policy rate decision and forward guidance for
  Canadian equities*; (2) *Canada CPI inflation and Labour Force Survey jobs
  report market reaction 2026*; (3) *oil and gold commodity price outlook impact
  on the TSX energy and materials sectors 2026*; (4) *USD/CAD Canadian dollar and
  Government of Canada 10-year bond yield moves March 2026*; (5) *US Federal
  Reserve policy and tariff trade spillovers into Canadian stocks 2026*; (6)
  *S&P TSX Composite outlook Canadian bank energy mining sector earnings Q1 2026*.
- **Reason** — four load-bearing factors lifted from the written `rationale`:
  BoC held at **2.25%** (accommodative hold); CPI **1.8%** vs **−84K** February
  jobs (soft inflation, weakening labour); Middle-East commodity volatility (oil
  mixed, gold bid); **25% US tariffs** as the structural headwind driving the
  negative skew.
- **Forecast** — the emitted 11-point quantile grid as a distribution strip
  (log-return space): **Q0.05 = −7.5%**, **median = +1.0%**, **Q0.80 = +5.0%**,
  **Q0.95 = +9.0%**; darker band = the 60% interval (Q0.20–Q0.80). The realised
  21-day move, **+5.1%** (simple return; log **+4.96%**), lands essentially at
  **Q0.80** — the agent's positive-median, negative-skew call was well-placed.

**Source:** `data/predictions/tsx_ws_eval_2026_weekly/`
`agent_predictor_tsx_analyst_news_claude-sonnet-4-6_continuous/tsx_logret_21b/`
`2026-03-30.yaml` (`tool_calls`, `metadata.rationale`, `payload.quantiles`).
Realised value: `tsx_logret_21b` at forecast date 2026-04-28 from the TSX data
service (`_blogdata.realized`).

## fig2 — `fig2_scenario_card.png`

One narrative scenario write-up (origin **2026-03-31**, 60-business-day outlook)
graded against what happened. **Left:** the three scenarios with their assigned
probabilities and 60-day outlook ranges —

| Scenario | Prob. | 60-day outlook |
|---|---|---|
| Commodity-Led Defensive Rotation *(base case)* | **0.55** | +3% to +5% |
| US-Trade Policy Stalls / Tariff Risk | **0.30** | −4% to −7% |
| BoC Policy Disappointment | **0.15** | −2% to +1% |

**Right, top:** the LLM judge's verdict (Claude Sonnet-4.6) — **calibration 5/5,
drivers 3/5, specificity 4/5**. **Right, bottom:** the realised cumulative log
returns from 2026-03-31 — **+2.60%** at 5 days (through 2026-04-08), **+3.65%**
at 21 days (through 2026-04-30), **+6.35%** at 60 days (through 2026-06-25), all
up. **Annotation:** the mechanism mismatch — the base case earned its high
calibration by calling the direction (and roughly the magnitude) right, but for
the wrong reason: it assumed *persistent Middle-East friction keeping oil bid*,
whereas the rally came from a *ceasefire relief bounce* — hence the lower
drivers score.

**Source:** `data/scenarios/2026-03-31/writeup.md` (scenario names,
probabilities, outlook ranges) and `data/scenarios/2026-03-31/judge.yaml`
(`verdict` scores + `realized_outcome` horizons/returns).

## fig3 — `fig3_divergence_sentinel.png`

Agent-vs-tree divergence as a regime sentinel (protected 2026 eval, h=21).
**Main panel:** per-origin divergence D between the gemini news agent's
(`agent_predictor_tsx_analyst_news_gemini-3.5-flash_continuous`) and LightGBM
+cov's (`darts_lightgbm_cov`) 11-point quantile grids, where D = mean absolute
difference across the 11 quantile levels; the 2026 war window (origins
2026-02-09 → 2026-04-13) is shaded in the Part-1 fig-4 landmark style. Three of
the four highest-D origins sit in or at the window: **2026-03-16 (D=0.0461)**,
**2026-03-23 (0.0338)**, **2026-02-23 (0.0223)** — the two forecasters disagree
most exactly when the regime breaks. The fourth, **2026-06-08 (0.0228)**, fires
outside the confirmed window: the agent's rationale at that origin prices a
correction off the TSX's record high (35,217) with sticky 3.2% CPI, widening its
Q0.05/Q0.95 to −7.2%/+6.8% — a genuine perceived-risk signal, annotated on the
figure. **Side panel:** mean CRPS of three policies over the same origins —
always LightGBM +cov **0.01718**, always agent **0.01759**, and a
divergence-gated router (use the agent when D exceeds its median, else LightGBM
+cov) **0.01688** (in-sample median threshold D = 0.0170).

**Population:** all **24** weekly origins with resolved 21-day outcomes,
**2026-01-05 → 2026-06-15** — the same n=24 the refreshed eval leaderboard
reports at h=21 (no resolved origin is excluded).

**Exploratory caveats (also flagged on the figure):** n = 24 origins; the router
threshold is the *in-sample* median, so the CRPS gain is illustrative, not an
out-of-sample result. A companion construction that sets the threshold on the
2025 backtest fires at the 2025 tariff window as well — but does not pay for the
sonnet news agent, so the gated-routing gain is rung-specific, not a law.

**Source:** `data/predictions/tsx_ws_eval_2026_weekly/` (both rungs),
`tsx_logret_21b/<origin>.yaml`; realised values from the TSX data service
(`_blogdata.realized`); CRPS per origin via `properscoring.crps_ensemble` on the
sorted quantile grids. The script pins the three policy means (0.0171834 /
0.0175927 / 0.0168775) and asserts its recompute reproduces them to 1e-5 before
writing the PNG.

## fig4 — `fig4_adaptive_prepost.png`

One adaptive study session, graded honestly as a **null result**. The adaptive
analyst studies the market, opens hypotheses, backtests them, and graduates the
confirmed ones into a strategy file that the forecasting rung then reads. This
figure asks the only question that matters: did a session of study actually move
the protected-eval score?

**Left — pre/post paired CRPS.** Grouped bars by horizon (h = 1, 5, 21) of mean
CRPS on the protected 2026 weekly eval: **PRE** = the seed-strategy rung
(`agent_predictor_tsx_adaptive_analyst_tsx_strategy_gemini-3.5-flash_continuous`)
vs **POST** = the trained-strategy rung (`..._tsx_strategy_trained_...`).

| Horizon | PRE mean CRPS | POST mean CRPS | n | POST wins |
|---|---|---|---|---|
| h = 1 | **0.00522** | **0.00527** | 24 | 13/24 |
| h = 5 | **0.01206** | **0.01193** | 22 | 12/22 |
| h = 21 | **0.01857** | **0.01876** | 24 | 10/24 |

The horizon means move both directions — POST is very slightly better at h=5,
very slightly worse at h=1 and h=21 — and every gap is a fraction of a
CRPS×10⁻³, well inside the noise floor for n ≤ 24. The per-origin win split is
essentially a coin flip at every horizon. Small open diamonds mark the
**war-window cut** (origins 2026-02-09 → 04-13 only): h=5 pre **0.01552** / post
**0.01514**, h=21 pre **0.02460** / post **0.02416** — the trained strategy's
wider low-vol/anomaly intervals help a touch in the turbulent stretch, but not
enough to register in the pooled mean. The subtle dashed reference line is
`darts_lightgbm_cov`'s h=21 mean (**0.01718**), for rank context: both agent
rungs sit above the plain covariate tree at the long horizon.

**Right — what graduated.** A faithful, condensed excerpt of the trained
strategy file, styled as fig2's cards: (1) the low-volatility calibration
correction (realised vol < 10% → widen intervals 12% / 18% / 23% by horizon,
hyp-001); (2) the negative-anomaly correction (daily return z-score < −2.5 →
widen the 1-day intervals 40%, 5-day 20%, hyp-003); and (3) one recorded
**NEGATIVE result** — the day-of-week finding, where unconditional and
sub-period ANOVA tests (overall p = 0.5392, averages sign-flipping across
epochs) rule out any day-of-week adjustment (hyp-009). Card footer: **25
corrections graduated in one session — all confirmations same-session.**

**The honest caveat (framed on the figure and here):** all 25 graduated
corrections were *confirmed within the same session that opened them* — the
strategy file's `hypotheses` all carry `opened_on` = `confirmed_on` =
`2026-07-18`, three same-day confirmations each. There is no held-out
confirmation and no out-of-sample graduation gate, so the strategy file is a
record of what the session found self-consistent, not of what survived
independent replication. Combined with the null pre/post result, the takeaway is
deliberately deflationary: one study session produced a plausible, readable
strategy document but did **not** move the protected-eval score beyond noise.

**Source:** `data/predictions/tsx_ws_eval_2026_weekly/` — the two adaptive rungs
above, tasks `tsx_logret_{1b,5b,21b}`, `<origin>.yaml`. Per-origin CRPS via
`properscoring.crps_ensemble` on the sorted 11-point quantile grid vs the
realised value (`_blogdata.realized`), resolved origins only (n = 24 / 22 / 24;
paired on the origins both rungs resolved). War-window cut = origins 2026-02-09
→ 04-13. Reference line from `darts_lightgbm_cov/tsx_logret_21b/`. Card text
condensed (faithfully) from
`workshop_experiments/.../adaptive/skills_tsx/tsx-strategy-trained/skill_state.yaml`
(`calibration_corrections` for hyp-001/hyp-003, `hypotheses`/`observations` for
hyp-009). The script pins all six horizon means, both war-window cuts, the
reference mean, and the three win counts, and asserts its recompute reproduces
them to 1e-5 before writing the PNG.

---

### Regenerating

From the worktree root, with the refreshed data store on hand:

```
export BLOG_WS_DATA_ROOT=.../workshop_experiments/workshop_experiments/data
uv run python blog/part-2/assets/fig1_agent_anatomy.py
uv run python blog/part-2/assets/fig2_scenario_card.py
uv run python blog/part-2/assets/fig3_divergence_sentinel.py
uv run python blog/part-2/assets/fig4_adaptive_prepost.py
```

Each writes its PNG (220 dpi) beside the script. Omit `BLOG_WS_DATA_ROOT` to use
this checkout's own `data/` store.
