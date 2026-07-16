---
name: vol-regime
description: >-
  Code patterns for classifying the current S&P/TSX Composite volatility regime
  and detecting anomalous recent moves. Always load alongside fetch-yfinance —
  the examples require a yfinance data fetch at the top of the same script. Load
  references/examples.md for working code.
---

# Volatility regime classification (S&P/TSX Composite)

## What this skill provides

**`references/examples.md`** — Working code patterns for:
- Pattern 1: Regime from rolling 21-day realised vol of `^GSPTSE` returns (the
  primary classifier — the TSX has no native implied-vol index)
- Pattern 2: US `^VIX` as a cross-market risk cross-check (spillover only)
- Pattern 3: Anomaly detection — z-score of the most recent daily return
- Pattern 4: Adaptive dispersion-window selection based on regime and anomaly

These patterns are designed to be **combined with a data-fetch block** in a
single code execution. Do not call `run_code` separately for data fetching and
regime classification — combine them.

## Regime thresholds (S&P/TSX Composite realised vol)

| Regime   | Realised vol (annualised %) |
|----------|-----------------------------|
| low      | < 10                        |
| normal   | 10 – 15                     |
| elevated | 15 – 22                     |
| high     | > 22                        |

These bands mirror the `vol_regime_bands` in the workshop `TSX_DOMAIN`
(10 / 15 / 22 / 30), calibrated empirically from 21-day rolling realised vol of
`^GSPTSE` since 2005 (median ~11%, 75th ~15%, 90th ~21%, 95th ~29%). The TSX runs
materially calmer than the S&P 500, so these bands sit **below** the SPX/VIX
levels — do not carry over US thresholds. There is no Canadian VIX; realised vol
of the index return series is the regime signal.

## Output of Pattern 4

Pattern 4 returns a `dispersion_window` integer (10 or 21 trading days) to pass
to the `trend-projection` skill's interval-calibration step: use the shorter
window when a recent anomaly indicates the regime is shifting.
