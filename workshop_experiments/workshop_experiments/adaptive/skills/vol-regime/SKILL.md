---
name: vol-regime
description: >-
  Code patterns for classifying the current S&P 500 volatility regime and
  detecting anomalous recent moves. Always load alongside fetch-yfinance — the
  examples require a yfinance data fetch at the top of the same script. Load
  references/examples.md for working code.
---

# Volatility regime classification (equity index)

## What this skill provides

**`references/examples.md`** — Working code patterns for:
- Pattern 1: Regime from the VIX level (low / normal / elevated / high)
- Pattern 2: Regime from rolling 21-day realised vol of index returns (when the
  VIX is unavailable)
- Pattern 3: Anomaly detection — z-score of the most recent daily return
- Pattern 4: Adaptive dispersion-window selection based on regime and anomaly

These patterns are designed to be **combined with a data-fetch block** in a
single code execution. Do not call `run_code` separately for data fetching and
regime classification — combine them.

## Regime thresholds (S&P 500 / VIX)

| Regime   | VIX level | Rough realised-vol proxy (annualised %) |
|----------|-----------|-----------------------------------------|
| low      | < 15      | < 12                                    |
| normal   | 15 – 20   | 12 – 18                                 |
| elevated | 20 – 30   | 18 – 28                                 |
| high     | > 30      | > 28                                    |

These bands mirror the `vol_regime_bands` in the workshop `SP500_DOMAIN`
(15 / 20 / 30 / 45). The VIX is implied vol in annualised percentage points; the
realised-vol column is an approximate translation for when only the index close
is available.

## Output of Pattern 4

Pattern 4 returns a `dispersion_window` integer (10 or 21 trading days) to pass
to the `trend-projection` skill's interval-calibration step: use the shorter
window when a recent anomaly indicates the regime is shifting.
