---
name: fetch-yfinance
description: >-
  One-shot code patterns for downloading price and market data from yfinance
  inside the E2B sandbox. Load this skill whenever a task requires equity-index
  or market data from Yahoo Finance. Load examples.md for working code.
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
- Pattern 1: Single ticker, date range (e.g. the S&P 500 index `^GSPC`)
- Pattern 2: Applying a temporal cutoff for backtesting (do not use data after `as_of`)
- Pattern 3: Multiple tickers in one fetch (e.g. `^GSPC` alongside the VIX `^VIX`)

## Workflow

1. Call `load_skill_resource("fetch-yfinance", "references/examples.md")` to load the patterns.
2. Identify which pattern fits your task.
3. Combine with other skill examples in the same code block.

## Common tickers

| Series               | Ticker  |
|----------------------|---------|
| S&P 500 index        | `^GSPC` |
| CBOE Volatility Index | `^VIX`  |
| NASDAQ Composite     | `^IXIC` |
| 10Y Treasury yield   | `^TNX`  |
| WTI crude oil        | `CL=F`  |

## Gotchas

- `ticker.history()` returns a timezone-aware DatetimeIndex on recent yfinance
  versions. Strip the timezone with `.dt.tz_localize(None)` after reset_index.
- For the index (`^GSPC`), use `auto_adjust=False` and take the `Close` column —
  the index carries no dividends so the adjusted close is the same series.
- Always sort by date ascending after fetching.
