---
name: meta-learning
description: >-
  Governs when and how the adaptive S&P 500 analyst updates its strategy skill.
  Consult this before calling any strategy mutation tool. The process is
  deliberately conservative — it resists updating on individual surprises and
  requires pattern-level evidence before revising strategy.
---

# Meta-learning: strategy update governance

## The four learning layers

`sp500-strategy` has four distinct layers, each with its own evidence bar and
mutation tool. Work bottom-up: always start with an observation before opening a
hypothesis, and always accumulate enough hypothesis outcomes before graduating
to a calibration correction.

| Layer | Tool | Evidence bar |
|-------|------|-------------|
| **Observations** | `record_observation` | Pattern visible across ≥2 forecasts — not a single surprise |
| **Hypotheses** | `open_hypothesis` | One strong observation suggesting a durable pattern |
| **Hypothesis outcomes** | `record_hypothesis_outcome` | Each resolution relevant to an open hypothesis |
| **Calibration corrections** | `graduate_hypothesis` | Tool enforces the confirmation threshold — rejects if not met |
| **Approach narrative** | `update_approach_narrative` | Only when the calibration record reveals a structural insight |

## Live gating you operate under

When deployed as the **learning twin**, your proposed updates pass through a
tiered gate on top of these tools. You cannot lower any bar, and you are
incentivised to propose thoughtfully rather than overzealously:

- **Observations** append freely (tier 1).
- **Hypotheses** are evidence-gated (tier 2): a hypothesis only graduates after
  the confirmation threshold `k` is met by *subsequent live resolutions*;
  refutations demote it.
- **Behavioral changes** — anything that directly alters forecasts (a graduated
  calibration correction, an approach-narrative rewrite) — enter a **forward
  shadow gate** (tier 3): the candidate runs in shadow for `M` trading days and
  is adopted only if it does not underperform the current strategy.
- Rate limits cap graduated behavioral changes (≤1 per week) and open
  hypotheses. A circuit breaker freezes all adaptation if your trailing 21-day
  CRPS degrades too far relative to the frozen twin.

## When to update

Engage the update process only when you have **pattern-level evidence** — not
after a single surprising outcome. Appropriate triggers:

- A self-review or backtesting exercise spanning five or more origins reveals a
  systematic bias (e.g. intervals consistently too narrow in the elevated/high
  VIX regime, or a directional skew that persists across horizons).
- A postmortem over a resolved forecast shows the miss was knowable from a
  narrative signal you ignored, and the same pattern recurs.
- A code-execution analysis on historical index data reveals a durable
  relationship not currently captured in your strategy.

**Do not update after a single resolution, even a large miss.** Markets have
noise; one bad forecast is not a signal.

## How to update: the tool call sequence

1. **Always** `record_observation(finding, linked_hypothesis?)` first.
2. **If a durable pattern is suspected** `open_hypothesis(claim, initial_evidence)`.
3. **On each relevant resolution** `record_hypothesis_outcome(hypothesis_id, outcome)`.
4. **When the threshold is reached** `graduate_hypothesis(hypothesis_id, condition, adjustment, horizon_scope)` — the tool enforces the threshold and will state exactly how many more confirmations are needed if you are short.
5. **Rarely** `update_approach_narrative(new_text, rationale)` — only when the calibration record reveals a structural insight the narrative no longer captures. Never call this during a live prediction task.

## Guarding against over-learning

The greatest risk in a self-updating strategy is chasing noise. Before opening a
hypothesis or proposing a graduation, ask:

- Is this pattern visible across multiple origins, or just one?
- Would this update have improved performance over the past ten forecasts, or
  only the most recent few?
- Am I reacting to a one-time market event (a geopolitical shock, an unscheduled
  Fed move) rather than a durable forecasting flaw?

If uncertain, call `record_observation` without opening a hypothesis and revisit
after more evidence accumulates.

## What NOT to update

- Do not open a hypothesis after a single resolution.
- Do not attempt to graduate a hypothesis the tool rejects — accumulate the
  required outcomes first.
- Do not update the approach narrative based on market opinions or macro views.
  Update only based on evidence about your own forecasting behaviour.
- Do not update during a live prediction task.
