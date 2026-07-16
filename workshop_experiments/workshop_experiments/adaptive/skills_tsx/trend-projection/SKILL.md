---
name: trend-projection
description: >-
  Code patterns for projecting a calibrated probabilistic forecast of S&P/TSX
  Composite cumulative log returns. Always load alongside fetch-yfinance and
  vol-regime — references/examples.md includes a Full Pipeline Example showing
  the complete self-contained script from yfinance fetch through vol regime to
  quantile grid.
---

# Return projection and interval calibration (S&P/TSX Composite)

## What this skill provides

**`references/examples.md`** — Working code patterns for:
- Pattern 1: Centre the forecast — the martingale prior puts the median near 0
  cumulative log return at every horizon
- Pattern 2: Calibrate interval half-widths from trailing realised dispersion,
  scaled by the square root of the horizon
- Pattern 3: Emit the standard 11-point quantile grid, non-decreasing, with the
  point forecast equal to the 0.50 quantile
- Full Pipeline Example: fetch → regime → calibrated quantile grid

## Typical usage

1. Load `fetch-yfinance` → fetch the `^GSPTSE` close history to `as_of`
2. Load `vol-regime` → classify regime, detect anomaly, choose `dispersion_window`
3. Load `trend-projection` → centre, scale by horizon, emit the grid
4. Write one complete code block combining all three

## Key formula

For a target cumulative log return over `h` business days, the interval
half-width at a standard-normal quantile `z_q` is:

```
half_width_h(q) = z_q * sigma_daily * sqrt(h)
```

where `sigma_daily` is the standard deviation of daily log returns over the
`dispersion_window` chosen by `vol-regime`. The distribution is centred near 0
(martingale prior); only lean the median away from 0 when a strong, dated
catalyst in the news context justifies it (a Bank of Canada decision, a sharp oil
or gold move, a US policy or tariff shock).

## Interval calibration note

Realised-dispersion intervals often run **too narrow** in elevated or high vol
regimes and around commodity/policy shocks. Per the `tsx-strategy` skill, apply
any active calibration correction (e.g. widen the interval when the realised-vol
regime is elevated/high, or when a commodity shock is in play) **after** computing
the base half-width — never skip an active correction.
