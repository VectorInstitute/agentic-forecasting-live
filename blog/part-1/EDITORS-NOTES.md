# Editor's notes — Part 1

Word count: **1,921 readable words** (1,979 raw incl. image alt-text). Within the
hard range 1,700–2,000; near the blueprint's ~1,850 target. §5 (the ladder) is the
heavy section because it carries all three result figures.

## Every number in the post → its artifact source

All CRPS values are mean CRPS from the two `leaderboard.csv` files (paths below);
all landmark % moves and the two didactic CRPS values are from `assets/CAPTIONS.md`.

- **BT** = `workshop_experiments/workshop_experiments/data/results/tsx_ws_backtest_2025_weekly/leaderboard.csv`
- **EV** = `.../tsx_ws_eval_2026_weekly/leaderboard.csv`

| Claim in text | Value | Source |
|---|---|---|
| Horizons 1 / 5 / 21 business days | — | BT/EV `target` col (`tsx_logret_1b/5b/21b`) |
| 2025 tariff drawdown −12.8%, rebound +19.3% | −12.8 / +19.3 | CAPTIONS landmark table (fig1) |
| 2026 war drawdown −9.3%, rebound +8.5% | −9.3 / +8.5 | CAPTIONS landmark table (fig1) |
| CRPS didactic: sharp σ=0.4% → 0.0024; wide σ=1.2% → 0.0033 | 0.0024 / 0.0033 | CAPTIONS fig2 |
| "~50 resolved per horizon" (backtest) | 51/47/51 | BT `n_scores` (naive ref) |
| "~24 per horizon" (protected eval) | 24/22/24 | EV `n_scores` |
| naive floor h=1 eval CRPS 0.0093 | 0.009275 | EV `last_value_naive` h=1 |
| "best method roughly halves that" | 0.004996 / 0.009275 ≈ 0.54× | EV lightgbm_cov h=1 vs naive |
| plain LightGBM tops h=1 **backtest** at 0.0038 | 0.0038188 | BT `darts_lightgbm` h=1 (rank 1) |
| LightGBM+cov takes h=1 eval at 0.00500 | 0.0049957 | EV `darts_lightgbm_cov` h=1 (rank 1) |
| flash-lite LLMP tied at 0.00501 | 0.0050115 | EV `llmp_quantile_grid_tsx_ws[gemini-3.1-flash-lite-preview]` h=1 (rank 2) |
| plain LightGBM "middle of the pack" in eval | 0.0051603 (8th/16) | EV `darts_lightgbm` h=1 |
| Sonnet-5 (thinking) tops h=5 and h=21 **backtest** | 0.0084581 / 0.0203576 | BT `llmp_...cov[claude-sonnet-5]` h=5, h=21 (rank 1 each) |
| covariate panel members (BoC rate, CPI, unemployment, WTI, gold, USD/CAD, VIX, SPX) | 11 covariates | BT/EV `covariates` col on any `_cov` row |
| fig5: market swung −4.8% to +5.3%/day; medians −0.83% to +1.37% | as stated | CAPTIONS fig5 |
| classical methods "within a hair" at short horizon (eval) | ets 0.005211, autoarima 0.005217, kalman 0.005236 vs best 0.004996 | EV h=1 rows |

## Claims I was tempted to make but could not source (and how I handled them)

1. **Sonnet-5 per-forecast cost "an order of magnitude above a flash-lite call."**
   Neither leaderboard carries token or dollar cost. This is the one number in the
   post not traceable to an artifact — it rests on general model-pricing knowledge
   (a frontier thinking model vs. a lite model). I deliberately wrote "an order of
   magnitude" (loose, defensible) rather than a specific multiple. The brief
   *allowed* this one aside "if it earns it"; if comms wants zero unsourced numbers,
   cut the clause (see cut-list #1). Part 2's ~100× figure is for *agents*, not LLMP,
   so I did not borrow it here.
2. **"Indistinguishable" for 0.00500 vs 0.00501.** Justified by magnitude, not a
   significance test — we have no per-origin variance/CI artifact to cite, so I
   avoided any statistical-significance language and kept it to the plain-number
   comparison. If challenged, this is a descriptive claim, not an inferential one.
3. **ForecastBench "climbing toward the superforecaster line."** External, cited by
   link only (forecastbench.org/explore); I quoted no specific ForecastBench number,
   so nothing here needs a repo artifact. It is framing, not a result.
4. **News/analyst agent.** It exists in the fig3 prediction store (and the backtest
   leaderboard) but is a Part-2 subject; I cited none of its numbers here to avoid
   pre-empting Part 2.

## Tensions with the blueprint I resolved

- **Figure placement.** Blueprint §5 assigns three result figures (leaderboard,
  daily-CRPS, median-vs-realized) to the ladder and gives §6 (frozen LLM) none. I
  followed that literally: fig3/4/5 all sit in §5, §6 runs figure-free. Consequence:
  §5 exceeds its ~450w text budget and §6 sits under its ~300w; the *section* budgets
  bend but the *total* holds. I judged fidelity to the figure assignment more
  important than per-section word parity.
- **Missing ForecastBench figure.** Blueprint hook calls for an "attributed figure,
  forecastbench.org/explore," but the asset set is fig1–5 only. Fabricating or
  screenshotting one would violate the real-artifact rule, so I referenced
  ForecastBench as an inline attributed link with no embedded image.
- **SPX vs TSX magnitudes.** Used the TSX values from CAPTIONS throughout (−12.8 /
  +19.3 / −9.3 / +8.5), never the SPX-based `market-timeline-2025-2026.md` numbers,
  per the brief.
- **h=1 nuance precision.** The brief cites "0.005012"; the artifact value is
  0.0050115 (→ 0.00501) and LightGBM+cov is 0.0049957 (→ 0.00500). I reported both
  at 5 d.p.-rounded form to make "indistinguishable" self-evident.

## The 3 sentences I'd cut first if comms wants it shorter

1. **§6, the Sonnet-5 cost aside** — "...though its per-forecast cost is an order of
   magnitude above a flash-lite call — a trade-off worth naming out loud." (Least
   artifact-anchored; a clean, self-contained cut of ~20 words.)
2. **§5, the horizon-reshuffle tail** — "At h=5 and h=21 the ordering reshuffles
   again — no single family owns every horizon — while the classical methods stay
   competitive at the short end and the covariate panel earns its keep unevenly,
   helping at some horizons and adding noise at others." (Densest sentence; the
   headline h=1 story survives without it.)
3. **§2, the Toronto rationale** — "We forecast it because we're in Toronto and it's
   the market on our doorstep — but it is also a genuinely useful stress test."
   (Charming, not load-bearing; the energy/materials sentence carries the real
   justification.)


## Orchestrator pass (2026-07-17)
- Sonnet-5 aside re-sourced: token/wall multiples computed from run accounting
  (weekly eval: sonnet-5 target-only 399,507 out-tokens / 72 preds ≈ 5.5k/pred vs
  flash-lite ≈ 1.3k/pred; wall ≈ 92s vs ≈ 4s) — no longer an unsourced number.
- fig0_forecastbench added to the hook with attribution (source: learn-days asset,
  per the original capture; caption credits forecastbench.org/explore).
- [REPO] resolved to the public bootcamp repo per plan §6 (bootcamp CTAs → original repo).
