# workshop_experiments

Foundation package for the **workshop-paper** and **live-evaluation** S&P 500
forecasting experiments. This is research/experiment code — **not** bootcamp or
participant material, and there are **no notebooks**. Everything runs as an
importable Python package plus resumable CLI runners with immediately-persisted,
per-origin predictions.

It builds on the shared forecasting library (`aieng.forecasting`) and the
`sp500_forecasting` reference implementation (leak-safe target + covariate
panel, pure-frame leaderboard/analysis helpers), and on the domain-agnostic
agentic toolkit promoted in PR 1 (`build_analyst_config` over a `DomainConfig`).

## What's here

| Module | Purpose |
|--------|---------|
| `domain.py` | `SP500_DOMAIN` (`DomainConfig` for the equity index) + the return prompt builder and news / code agent config factories. |
| `data.py` | `build_workshop_service` — thin wrapper over `sp500_forecasting.data` for the `sp500_logret_{1,5,21}b` targets and the covariate panel. |
| `registry.py` | Named predictor factories (conventional, LLMP ± covariates, analyst / code agents), model-parameterised. |
| `specs/*.yaml` | Smoke, weekly 2025 backtest, weekly 2026 protected eval, and daily bonus-layer specs. |
| `runner.py` | Per-origin persisting, resumable backtest runner with token/cost accounting. |
| `scoring.py` | Reconstructs `BacktestResult`s from persisted predictions and builds leaderboard CSV/Markdown. |
| `run_backtest.py` / `score.py` | CLI entry points (`ws-run-backtest`, `ws-score`). |

## Task

Close-to-close **cumulative log returns** of `^GSPC` at horizons **h = 1 / 5 / 21
business days** (the same leak-safe construction as `sp500_forecasting`).
Probabilistic output on the standard quantile grid, scored by **CRPS**. Each spec
carries one single-horizon task per target.

## Predictors

Model-parameterised methods fold the model into their `predictor_id` so
persisted caches keep variants separate.

| Name | `predictor_id` shape |
|------|----------------------|
| `naive`, `ets`, `kalman`, `autoarima` | `last_value_naive`, `darts_ets`, `darts_kalman`, `darts_autoarima` |
| `lightgbm`, `lightgbm_cov` | `darts_lightgbm`, `darts_lightgbm_cov` |
| `llmp_qgrid`, `llmp_qgrid_cov` | `llmp_quantile_grid_sp500_ws[<model>]`, `..._sp500_ws_cov[<model>]` |
| `agent_news`, `agent_code` | `agent_predictor_sp500_analyst_{news,code}_<model>_continuous` |

`--methods` also accepts the groups `all`, `conventional`, `llmp`, `agent`, `api`.

## Usage

Smoke-first, conventional only (no API spend):

```bash
ws-run-backtest --spec sp500_ws_smoke --methods conventional
ws-score --spec sp500_ws_smoke
```

Weekly grid, LLMP across two models (spend-gated — smoke and cost-check first):

```bash
ws-run-backtest --spec sp500_ws_backtest_2025_weekly \
    --methods llmp_qgrid llmp_qgrid_cov \
    --models gemini-3.1-flash-lite-preview gemini-3.5-flash
ws-score --spec sp500_ws_backtest_2025_weekly
```

Runs **resume automatically**: an origin whose prediction file already exists is
skipped unless `--force-refresh`. Every origin is persisted the moment it
completes, so an interrupted run costs at most one origin of rework. Scoring
never re-calls any API — it reads the committed prediction files.

Predictions are written under `data/predictions/<spec_id>/<predictor_id>/<task_id>/<YYYY-MM-DD>.yaml`,
with a per-predictor `accounting.json` summary (new vs cached vs skipped origins,
call count, wall time, and any token/cost the predictors reported). Leaderboards
land under `data/results/<spec_id>/`.

Data caches (Yahoo/FRED parquet) live under the repo-root `data/` tree and are
git-ignored; only the committed smoke predictions under `data/predictions/` are
version-controlled.

## Design note — how the live daily harness (stage 2b) sits on this

The live daily-forecasting harness builds directly on this foundation, changing
only *where origins come from* and *when they resolve*:

- **A spec-less "today" origin.** Instead of a fixed backtest window, the live
  job runs a single origin = the latest trading day, one run per trading day at a
  fixed post-close time. The same `registry` predictors and the same per-origin
  persistence apply unchanged — a daily run is a one-origin backtest.
- **Append-only log.** The live layer writes into the same
  `data/predictions/<spec>/<predictor>/<task>/<date>.yaml` layout (an append-only
  committed log); the resume rule already guarantees a day is never overwritten
  or backfilled, matching the "logged gap, never backfilled" policy.
- **Deferred resolution.** Because prediction and scoring are decoupled here
  (`runner` predicts and persists; `scoring` resolves and scores later), the live
  resolution job is exactly today's `score` path run once each horizon resolves
  (h=1 next close, then h=5, h=21), appending outcomes to a self-updating live
  leaderboard.
- **Observability + accounting.** The token/cost accounting and (for agents) the
  Langfuse trace ids already stamped into `Prediction.metadata` carry straight
  into the live log, so every public prediction links to its full trace.

Stage 2b implements the scheduler, retry/gap policy, and model matrix on top of
these primitives. Stage 2c adds the adaptive study driver and the live twins.
This package is **design intent only** for those stages — no live scheduler or
adaptive driver is implemented here.
