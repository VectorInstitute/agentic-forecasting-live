# Part-1 figure captions and data provenance

Every number below is computed from repo artefacts by the figure scripts in this
directory (all import `_blogdata.py`). Realised returns and the price path come
from the leak-safe TSX data service
(`workshop_experiments.data_tsx.build_tsx_workshop_service`, Yahoo `^GSPTSE`
adjusted close). CRPS is recomputed per origin with
`properscoring.crps_ensemble` over the sorted quantile grid — the identical call
the scoring layer uses; it reproduces `data/results/*/leaderboard.csv` (e.g.
weekly-backtest naive h=1 = 0.0078193, n=51, matches exactly).

**TSX landmark magnitudes** (computed on the `^GSPTSE` close path,
`_blogdata.landmarks()`; drawdowns are peak→trough, rebounds trough→recovery):

| Window | Identity | Peak (date, level) | Trough (date, level) | Move |
|---|---|---|---|---|
| 2025 tariff drawdown | pre-tariff high → 8 Apr crash low | 2025-01-30, 25,808.3 | 2025-04-08, 22,506.9 | **−12.8 %** |
| 2025 rebound | crash low → end-June recovery | 2025-06-30, 26,857.1 | 2025-04-08, 22,506.9 | **+19.3 %** |
| 2026 war drawdown | early-Mar high → 20 Mar low | 2026-03-02, 34,541.3 | 2026-03-20, 31,317.4 | **−9.3 %** |
| 2026 April rebound | 20 Mar low → end-Apr | 2026-04-30, 33,964.3 | 2026-03-20, 31,317.4 | **+8.5 %** |

(TSX magnitudes differ from the SPX-based numbers in
`planning-docs/workshop-paper/market-timeline-2025-2026.md`: the TSX pre-tariff
high is 30 Jan 2025, not the SPX 19 Feb, and the energy/materials-heavy index
fell less in the 2026 war month than the SPX’s −4.98 % total-return March. The
event *identities* are taken from that timeline; the *magnitudes* here are the
authoritative TSX values.)

---

## fig1 — `fig1_tsx_landmarks.png`

S&P/TSX Composite adjusted-close level, 2025-01 → 2026-06, with the four landmark
windows shaded (drawdowns red, rebounds green) and labelled with the TSX
percentage moves above. This is the series the whole scoreboard forecasts, shown
as levels so the events are legible; the forecasting target itself is its
close-to-close log return at 1/5/21 business days.
**Data:** `^GSPTSE` adj. close via the TSX data service (cache
`data/yfinance/gsptse_adj_close_1d.parquet`). **Numbers:** the landmark table above.

## fig2 — `fig2_crps_didactic.png`

Didactic / synthetic (not from repo data). Two forecast distributions share a
median (0 %) but differ in width, facing a realised next-day return of +0.4 %.
CRPS (lower = better) is computed analytically with `properscoring.crps_gaussian`:
**sharp (σ=0.4 %) CRPS = 0.0024**, **wide (σ=1.2 %) CRPS = 0.0033**. The sharp
forecast wins because it also placed its mass near what happened — CRPS rewards
sharpness only when it is calibrated. Values are chosen at daily log-return scale.

## fig3 — `fig3_weekly_leaderboard.png`

Mean CRPS by predictor × horizon for the two weekly rolling-origin runs — 2025
backtest vs 2026 protected eval — as a rank heatmap. Cells are mean CRPS ×10⁻³;
shading is the within-column rank (darker = better); rows are ordered by mean
backtest rank so the eval columns visibly reshuffle it (the findable story: e.g.
plain LightGBM tops the backtest at h=1 but LightGBM +cov takes the protected-eval
h=1 at 0.00497 — roughly half the naive floor — with LLMP flash-lite a hair behind
at 0.00501, and the classical/LLMP order scrambles at h=5/21). This is a Part-1
figure: it shows only the numbers-only ladder (naive / classical / LightGBM) plus
the LLM-Process rungs — **16 predictors** — and deliberately excludes the news and
code agents, which are Part-2 material. Every cell is recomputed from the
prediction store and reproduces `leaderboard.csv` exactly.
**Data:** `data/predictions/tsx_ws_backtest_2025_weekly/` and
`.../tsx_ws_eval_2026_weekly/`; leaderboards in the sibling `data/results/`.
**n (origins resolved):** backtest h=1/5/21 = 51/47/51; eval = 24/22/24 (per
predictor, naive reference). **`n/a`:** LLMP gemini-3.5 (no cov), backtest h=5,
has one degenerate origin (2025-02-03) with a non-finite CRPS — matches the `inf`
row in `leaderboard.csv`; it is ranked last in that column.

## fig4 — `fig4_daily_crps_landmarks.png`

Per-origin CRPS over the daily grid (every business-day origin 2025-01-02 →
2026-06-15) at h = 1/5/21 for the free conventional methods plus the lite
LLM-Process, with the landmark windows shaded. Every method’s error spikes at the
same moments — the 2025 tariff crash dominates all three horizons, and the 2026
war window lifts them again — because the cause of each regime break is exogenous
to the series. **Data:** `data/predictions/tsx_ws_daily_2025_2026/`; CRPS per
origin via `crps_ensemble`. **n:** all five methods now resolve across the full
daily grid — ~365/365/364 origins at h=1/5/21 (the last few longest-horizon
origins have not resolved). **Gap closed:** `darts_lightgbm_cov`, previously
incomplete on the daily grid and absent at h=21, was backfilled in the refreshed
store and is now drawn across all three horizons.

## fig5 — `fig5_quiet_vs_loud.png`

Median h=1 forecast vs the realised next-day return over the 2025 tariff
drawdown-and-rebound stretch (2025-02-01 → 2025-06-30), aligned on the day being
predicted. The realised series (grey bars) **swung −4.8 % to +5.3 % a day**; the
forecast medians (LightGBM, LLMP flash-lite) **stayed within −0.83 % to +1.37 %**
— a probabilistic forecaster hedges toward zero because the daily move is close to
unforecastable. **Data:** `data/predictions/tsx_ws_daily_2025_2026/`, h=1;
realised from the TSX data service. Efficiency made visual, and a preview of why
every rung shares the same blind spot.

---

### Regenerating

From the workspace root (with the `aieng-forecasting` venv active and the Yahoo
cache populated under `data/yfinance/`):

```
python blog/part-1/assets/fig1_tsx_landmarks.py
python blog/part-1/assets/fig2_crps_didactic.py
python blog/part-1/assets/fig3_weekly_leaderboard.py
python blog/part-1/assets/fig4_daily_crps_landmarks.py
python blog/part-1/assets/fig5_quiet_vs_loud.py
```

Each writes its PNG (220 dpi) beside the script.
