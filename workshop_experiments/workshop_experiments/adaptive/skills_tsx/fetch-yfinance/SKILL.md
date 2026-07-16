---
name: fetch-yfinance
description: >-
  One-shot code patterns for downloading price and market data from yfinance
  inside the E2B sandbox. Load this skill whenever a task requires the S&P/TSX
  Composite or Canadian market data from Yahoo Finance. Load examples.md for
  working code.
---

# Fetching market data with yfinance

## E2B execution model

Each `run_code` call is a completely fresh Python process. There is no state,
no variables, and no files from any previous call. Every code block must be
fully self-contained: all imports, all data fetching, and all analysis in one
block.

yfinance is pre-installed in the sandbox. No `pip install` needed.

## What this skill provides

**`references/examples.md`** — Working code patterns for:
- Pattern 1: Single ticker, date range (e.g. the S&P/TSX Composite index `^GSPTSE`)
- Pattern 2: Applying a temporal cutoff for backtesting (do not use data after `as_of`)
- Pattern 3: Multiple tickers in one fetch (e.g. `^GSPTSE` alongside WTI crude `CL=F`
  and gold `GC=F`, the commodity drivers of the energy- and materials-heavy index)

## Workflow

1. Call `load_skill_resource("fetch-yfinance", "references/examples.md")` to load the patterns.
2. Identify which pattern fits your task.
3. Combine with other skill examples in the same code block.

## Common tickers

| Series                     | Ticker    |
|----------------------------|-----------|
| S&P/TSX Composite index    | `^GSPTSE` |
| WTI crude oil              | `CL=F`    |
| Gold                       | `GC=F`    |
| USD/CAD exchange rate      | `CAD=X`   |
| CBOE Volatility Index (US) | `^VIX`    |
| S&P 500 index (US spillover)| `^GSPC`  |

## Gotchas

- `ticker.history()` returns a timezone-aware DatetimeIndex on recent yfinance
  versions. Strip the timezone with `.dt.tz_localize(None)` after reset_index.
- For the index (`^GSPTSE`), use `auto_adjust=False` and take the `Close` column —
  the index carries no dividends so the adjusted close is the same series.
- The TSX carries no native implied-vol index (there is no Canadian VIX in
  yfinance). Classify the regime from realised vol of `^GSPTSE` returns; use the
  US `^VIX` only as a cross-market risk read, not as the TSX's own implied vol.
- Always sort by date ascending after fetching.
