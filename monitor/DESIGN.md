# Experiment monitor — information architecture

The monitor is the **public live dashboard** for the continuously running S&P 500
forecasting experiment (see `planning-docs/workshop-paper-plan.md` §2). It is a
**zero-backend static site**: the committed append-only log *is* the database, the
harness derives a handful of small aggregate JSONs, and the site fetches them over
plain HTTP. Anyone can open it; nothing is privileged except the internal Langfuse
traces linked by id.

This document defines the view hierarchy, what each view answers, and the data it
reads. The versioned interface between the harness (writer) and this site (reader)
is specified separately in [`data-contract.md`](data-contract.md) and enforced by
[`schemas/`](schemas/) + [`validate_fixtures.py`](validate_fixtures.py).

## Design principles

- **The log is the database.** No server, no query layer. Every view is a pure
  function of committed artifacts. Aggregates (leaderboard, manifest) are convenience
  derivations the harness commits; they can always be rebuilt from the records.
- **Honesty is the product.** Misses are shown as gaps, never hidden or backfilled.
  Sample size `n`, freshness, and horizon overlap are surfaced next to every score so
  a small-`n` cell can't masquerade as a settled result.
- **Progressive disclosure.** Overview → method/model detail → single-forecast
  drill-down. Each level answers a coarser-to-finer question.
- **Curation is explicit and uniform** (policy verbatim below): rationales are public;
  raw retrieval and prompt scaffolding are not; trace ids link authorized users to the
  full internal trace.
- **Charts follow the `dataviz` skill** — categorical hues in fixed order, a
  single-hue sequential ramp for the heatmap, thin marks, recessive grid, a legend for
  every multi-series chart, hover tooltips, and a companion table view. Light and dark
  are both explicitly designed (not an auto-flip).

## View hierarchy

### 1. Overview scoreboard  *(built)*

The "how are we doing" landing view. Answers: *which method/model is winning, at which
horizon, on how much evidence, and how fresh is it?*

- **KPI row** — methods tracked, origins committed, resolved-scores count, logged gaps.
  Stat tiles (not one-bar charts), per the form heuristic.
- **Leaderboard heatmap** — rows are `method × model` in fixed method order (LLM rungs
  expand across the model matrix; conventional methods are single rows); columns are
  horizons `h = 1, 5, 21`. Cell = **mean CRPS**, shaded on a single-hue blue ramp
  **normalized within each horizon column** (CRPS grows ~5× across horizons, so
  cross-column shading would be misleading). Hover reveals `n`, 90% coverage, and last
  updated. The numeric value is printed in every cell, so the heatmap doubles as the
  table view.
- **Cumulative mean CRPS over time** — one line per method (LLM rungs shown on a single
  representative model to keep the chart at six series), horizon selectable. Legend +
  crosshair tooltip carry identity; a "show data table" toggle exposes the underlying
  numbers. A line settling lower is the better forecaster; the compounding gap is the
  headline.
- **Gap log** — table of missed days/scopes with reason, retries, and log time.

Reads: `data/manifest.json`, `data/leaderboard.json`, `data/gaps.json`.

### 2. Method / model detail  *(partially built; folded into overview + drill-down)*

Answers: *how has one predictor done per origin, and where did it miss?* In this
prototype the per-cell hover (n, coverage, freshness) and the cumulative lines cover
the aggregate half; per-origin forecast-vs-realized inspection is provided by the
drill-down below, reachable for any origin. A dedicated single-predictor page
(per-origin CRPS strip + forecast-vs-realized for one method across all origins) is a
natural next increment and needs no new data — it reads the same resolution records.

### 3. Single-forecast drill-down  *(built)*

Answers: *at this one origin, what did every method predict, who was right, and why?*

- **Origin picker** (from the manifest) + **horizon selector**.
- **Predictive-distribution comparison** — one horizontal row per `method × model`,
  drawing the 90% (0.05–0.95) and 80% (0.10–0.90) intervals and the median, all against
  a shared value axis, with the **realized outcome** as a dashed reference line spanning
  every row. Rows sort best-CRPS-first once resolved. Hover gives the exact quantiles
  and CRPS. This is the honest side-by-side: fan width = confidence, distance of the
  realized line from each median = who was right.
- **Agent rationale & curated trace** — the committed, agent-authored rationale text for
  the selected method; the curated trace summary (tool names + query titles only); and
  the Langfuse trace id. Conventional methods honestly show "no rationale".
- **Curation policy** printed inline (verbatim below).

Reads: `data/forecasts/<origin_date>.json` (a per-origin bundle of prediction +
resolution records + realized summary).

### 4. Twins view  *(stubbed — "coming soon")*

For stage 2c, when the frozen and learning twins deploy. Will plot the two arms'
**cumulative CRPS side by side** with the trailing score gap the circuit breaker
watches, and annotate **strategy-mutation events** (observation → hypothesis →
behavioral, with gate outcomes) on the timeline, each linking to its version-controlled
rationale. The data contract already covers this: `schemas/mutation_event.schema.json`
and `data/mutations.json` exist, so this view is a rendering task, not a data one.

### 5. Calibration deep-dive  *(stubbed — "coming soon")*

The overview already surfaces 90% interval coverage per cell. This view will add
reliability diagrams (nominal vs. empirical coverage) and PIT histograms per method ×
horizon, plus coverage-over-time. All inputs come from resolution records already in the
contract — no new schema needed.

## Curation policy (verbatim)

> Agent-authored rationales and tool-call summaries are PUBLIC; raw retrieved article
> text and internal prompt scaffolding are NOT public; Langfuse trace ids are shown so
> authorized users can open full traces internally.

Concretely, in the data contract this means: the `rationale` and per-horizon
`horizon_rationale` fields and the `curated_trace_summary.tool_calls[]` list (each entry
is a `{tool, query_title}` pair — **names and titles only**) are safe to commit and
display. Retrieved article bodies and the agent's system/prompt scaffolding are never
written to the log. `langfuse_trace_id` is committed and shown as an identifier; opening
the trace itself requires internal Langfuse access.

## Data flow

```
live harness (writer)                     monitor site (reader, static)
─────────────────────                     ─────────────────────────────
predict job   ─┐                          fetch ./data/manifest.json   → freshness, origins, mock flag
resolve job   ─┼─► append-only log  ──►   fetch ./data/leaderboard.json → heatmap + cumulative lines
scoring job   ─┘   (git-timestamped)      fetch ./data/gaps.json        → gap log
                    │                      fetch ./data/mutations.json   → twins (later)
                    └─► derive aggregates  fetch ./data/forecasts/<d>.json → drill-down
```

The site never globs a directory (it can't, statically): the **manifest enumerates**
the origins that have a committed bundle, and every aggregate carries a `generated_by`
flag (`mock` | `harness`) that drives the **MOCK DATA** banner.

## What is built vs. stubbed

| View | Status |
|---|---|
| Overview: KPI row, leaderboard heatmap, cumulative lines, gap log | **Built** |
| Single-forecast drill-down: fan comparison + rationale + curated trace + trace id | **Built** |
| Method/model detail (dedicated per-predictor page) | Partial (hover + drill-down cover it; dedicated page is a no-new-data increment) |
| Twins view (frozen vs. learning, mutation timeline) | **Stubbed** (schema + fixtures ready) |
| Calibration deep-dive (reliability, PIT) | **Stubbed** (no new data needed) |
