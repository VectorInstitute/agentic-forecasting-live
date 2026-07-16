# Data contract — live harness (writer) ⇄ monitor (reader)

This is the **versioned interface** between the live forecasting harness (stage 2b,
`workshop_experiments/`, another agent's work) and the static monitor site. The harness
**writes** append-only records and small aggregates; the site **reads** them. Neither
side imports the other's code — the JSON schemas in [`schemas/`](schemas/) are the only
coupling.

- **Schema version:** `1.1.0` (semantic). Every record and aggregate carries
  `schema_version`. Additive, backward-compatible changes bump the minor version; a
  breaking change bumps the major and the site branches on it.
- **Validation:** [`validate_fixtures.py`](validate_fixtures.py) validates every fixture
  against these schemas (`jsonschema`, Draft 2020-12). The harness team should run the
  same schemas against real output before committing — see the README.
- **Quantile grid:** the canonical 11-point grid from
  `aieng.forecasting.evaluation.prediction.STANDARD_QUANTILES`:
  `[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]`.
- **Units:** forecast values and `realized_value` are **cumulative log returns** of
  `^GSPC` from origin close to horizon close. `crps` is in the same units (lower is
  better). All timestamps are UTC ISO-8601 with a trailing `Z`; all dates are
  `YYYY-MM-DD`.

## File layout

Two layouts coexist: an **append-only per-day log** (the tamper-evident source of truth,
git-timestamped) and **derived aggregates** the site fetches. The prototype ships the
aggregates plus one bundle per origin; the schemas apply to both layouts.

```
live/log/YYYY/MM/DD/                 ← append-only source of truth (harness writes)
  predictions/<predictor_id>.json    ← one prediction record  (schema a)
  resolutions/<predictor_id>-h<H>.json ← one resolution record (schema b), appended on resolve
  gap.json                           ← gap-log entry if the day/scope failed (schema d)
  mutations/<event_id>.json          ← strategy-mutation event (schema e), twins only

monitor/site/data/                   ← derived aggregates the static site fetches
  manifest.json                      ← site index: freshness, origins, mock flag (envelope)
  leaderboard.json                   ← overview aggregate (schema c)
  gaps.json                          ← { …, gaps: [ gap-log entry (d) ] }
  mutations.json                     ← { …, mutations: [ mutation event (e) ] }
  forecasts/<origin_date>.json       ← per-origin drill-down bundle: predictions (a) +
                                        resolutions (b) + realized summary (envelope)
```

The append-only files are one-record-per-file so commits are small and diffs are
auditable. The site never reads the per-day log directly (it can't glob statically); the
harness collates the log into the aggregates and the per-origin bundles on each run. Both
are byte-checkable against the schemas, so a bad aggregate is caught in CI, not in
production.

## The five core records

### (a) Prediction record — `schemas/prediction.schema.json`

One forecast submission for a single `(method, model)` at one origin, covering all
horizons. Written once at submission; **never revised** (the submission rule).

Key fields: `schema_version`, `origin_date`, `origin_timestamp` (market-close cutoff),
`submission_timestamp` (exact wall-clock commit time), `method` (enum), `model`
(string | null for conventional), `predictor_id`, `horizons[]` — each with `horizon`,
`point_estimate` (= 0.50 quantile), and the 11-point `quantiles[]` grid — `rationale`
(public), `curated_trace_summary.tool_calls[]` (`{tool, query_title}` only),
`langfuse_trace_id`. Optional `twin_id` for adaptive twins.

**`method` enum (1.1.0):** one value per deployed rung — `naive`, `ets`, `kalman`,
`autoarima`, `lightgbm`, `lightgbm_cov`, `llm_process`, `llm_process_cov`,
`agent_news`, `agent_code`, plus the stage-2c forward declarations
`adaptive_frozen` / `adaptive_learning`. `model` is `null` for the conventional
methods and carries the plain backing-model id for LLM/agent rungs
(`llm_process_cov` carries the same model id as `llm_process` — the covariate
variant is expressed by the method, never by a model-label suffix). Leaderboard
cells key on `(method, model, horizon)`, so per-rung methods keep every cell
unique without overloading `model`.

**`curated_trace_summary` is populated when available.** The writer curates
whatever structured tool-call list the agent path surfaces (tool names + query
titles only); until the agent runtime exposes one, records legitimately carry an
empty `tool_calls` list. An empty list means "no structured tool calls captured",
not "no tools used".

**Harness invariants not expressible in JSON Schema** (assert them in the writer): the
`quantiles` set equals the standard grid exactly; values are non-decreasing;
`point_estimate == quantile[0.50]` (mirrors the `ContinuousAgentForecastOutput`
validators).

### (b) Resolution record — `schemas/resolution.schema.json`

The outcome + score for one `(method, model, horizon, origin)`, appended when the horizon
resolves. Key fields: `origin_date`, `method`, `model`, `predictor_id`, `horizon`,
`forecast_date` (origin + horizon business days), `realized_value`, `crps`, `resolved_at`.
Joins back to the prediction on `(origin_date, predictor_id, horizon)`.

### (c) Leaderboard aggregate — `schemas/leaderboard.schema.json`

The overview's data. `cells[]` — one row per `(method, model, horizon)` with `mean_crps`,
`n`, optional `coverage_90` and `skill_vs_naive`, and `last_updated`. `cumulative[]` — one
running-mean-CRPS series per `(method, model, horizon)` for the trend lines. Envelope
carries `generated_by` (`mock` | `harness`) and `generated_at`.

### (d) Gap-log entry — `schemas/gap_log.schema.json`

A missed submission. `date`, `scope` (`all_methods` | `agents` | a `predictor_id`),
`reason`, `retries_attempted`, `logged_at`. Gaps are **documented facts, never
backfilled**.

### (e) Strategy-mutation event — `schemas/mutation_event.schema.json`

One tiered adaptation event for a live twin. `event_id`, `twin_id`, `occurred_at`,
optional `origin_date` (the triggering resolution), `tier`
(`observation` | `hypothesis` | `behavioral`), `gate_outcome` (`appended`, `proposed`,
`confirmed`, `refuted`, `graduated`, `demoted`, `shadowing`, `adopted`, `rejected`,
`frozen_circuit_breaker`), `version` (the `VersionEntry` this produces/targets),
`rationale` (mandatory, public), optional `confirmations` count for hypotheses.

## Envelope / site-facing schemas

- **`schemas/manifest.schema.json`** — the site index. Lists `horizons`, `quantiles`,
  `methods`, `models`, `latest_origin`, `origin_count`, `gap_count`, and `origins[]`
  (each `{origin_date, resolved_horizons}`) so the site can populate the drill-down
  picker and freshness badges without globbing. Carries the `generated_by` flag.
- **`schemas/forecast_bundle.schema.json`** — a per-origin drill-down bundle. `$ref`s the
  prediction and resolution schemas so composition is validated in one pass; adds a
  `realized[]` summary (`{horizon, forecast_date, realized_value}`) shared across methods.

## The mock flag & real-data switchover

Every aggregate/envelope carries `generated_by`: `"mock"` for the shipped fixtures,
`"harness"` for real output. The site shows a **MOCK DATA** banner whenever any loaded
payload is `mock`. Switchover is therefore just: point `monitor/site/data/` at the
harness's real aggregates (or have the harness write there) with `generated_by:
"harness"`. No site code changes. See the README.

## Notes for the harness team (stage 2b)

- **Write the aggregates, not just the log.** The static site cannot query; it needs
  `manifest.json`, `leaderboard.json`, `gaps.json`, and one `forecasts/<origin>.json`
  bundle per origin, regenerated each run from the append-only log. Keep them small.
- **Enforce the three non-schema invariants** (grid completeness, monotonic quantiles,
  point==median) in the writer — JSON Schema can't, and the drill-down fan charts assume
  them.
- **`predictor_id` is the join key and must be stable** — committed caches key on it
  (mirror the oil `predictor_id` invariant). Suggested convention, matching the fixtures:
  `sp500_<method>` for conventional, `sp500_<method>__<model>` for LLM/agent rungs.
- **Curation happens at write time.** Never write retrieved article bodies or prompt
  scaffolding into `rationale`/`curated_trace_summary`. Populate `langfuse_trace_id` so
  the record links to its trace.
- **Timestamps:** record both `origin_timestamp` (the information cutoff) and
  `submission_timestamp` (wall-clock commit) — the monitor and the paper both rely on the
  distinction for the leakage-free claim.
- **Version bumps:** additive fields → minor bump, site keeps working; anything renamed or
  removed → major bump and a heads-up so the site can branch.

## Changelog

- **1.1.0 (2026-07-15, pre-deployment)** — per-rung `method` enum. Replaced the
  coarse `classical` / `analyst_agent` / `code_agent` / `twin_*` values with one
  enum value per deployed rung (`ets`, `kalman`, `autoarima`, `lightgbm_cov`,
  `llm_process_cov`, `agent_news`, `agent_code`, `adaptive_frozen`,
  `adaptive_learning`), so `(method, model, horizon)` uniquely keys every
  leaderboard cell and `model` is never overloaded as a variant label. Applied
  to `prediction` and `resolution` schemas, the mock generator + fixtures, and
  the site's method ordering/labels. Made **before** any live record was
  committed, so no live artifact carries the 1.0.0 method values; treated as a
  minor bump because the contract's shape is unchanged.
- **1.0.0** — initial contract: five core records, envelope schemas, mock flag.
