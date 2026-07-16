---
name: sp500-strategy
description: >-
  The adaptive S&P 500 analyst's current forecasting strategy. Load this at
  the start of every prediction task. This file is generated — edit the state
  through the mutation tools, not by hand.
---

# S&P 500 Forecasting Strategy

## Approach

Produce calibrated probabilistic forecasts of S&P 500 close-to-close cumulative log returns by combining two evidence streams: statistical analysis of recent return history and web-grounded market-news context.

Daily index returns are close to a martingale: the level of the return is barely predictable, so point forecasts should sit near 0 and the value is in the spread (volatility and tail risk), not a confident direction. Returns cluster in volatility — calm and turbulent stretches persist — so recent realised dispersion is the best guide to interval width.

At the 1-business-day horizon, keep the distribution tight and roughly symmetric about ~0; the dominant signal is the current volatility regime, not direction.

At the 5-business-day horizon, recent realised dispersion and the VIX regime set the interval width. Check the news context for scheduled catalysts (an FOMC decision, a CPI or payrolls print) before finalising the forecast.

At the 21-business-day horizon, macro regime and geopolitical risk dominate and the volatility term structure matters. Weight the news context and the vol regime more heavily; do not extrapolate a short run of up or down days into a trend.

Always run statistical analysis (vol-regime, trend-projection) before incorporating news context. The regime classification and trailing dispersion directly inform interval calibration.

## Active calibration corrections

*(No calibration corrections yet. Graduate a confirmed hypothesis to add one.)*

## Open hypotheses

*(No open hypotheses.)*

## Observations

*(No observations yet. Record findings from resolutions and self-reviews.)*

## Version history

| Date | Change |
|------|--------|
| initial | Strategy initialised with domain priors. No backtest evidence yet. |
