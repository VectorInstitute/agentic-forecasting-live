# fetch-yfinance — working code patterns

All patterns are self-contained: every `run_code` block must include its own
imports and data fetch. yfinance is pre-installed.

## Pattern 1 — Single ticker, date range

```python
import yfinance as yf
import pandas as pd

ticker = yf.Ticker("^GSPC")
raw = ticker.history(start="1990-01-01", end="2026-01-01", auto_adjust=False)
df = raw.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
df = df.sort_values("date").reset_index(drop=True)
```

## Pattern 2 — Temporal cutoff for backtesting

The `as_of` date is the information cutoff. Pass it as the **exclusive** `end`
so the sandbox never sees a session at or after the origin's close.

```python
import yfinance as yf
import pandas as pd

as_of = "2025-04-01"                       # from the prediction payload
raw = yf.Ticker("^GSPC").history(start="1990-01-01", end=as_of, auto_adjust=False)
df = raw.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
df = df[df["date"] < pd.Timestamp(as_of)].sort_values("date").reset_index(drop=True)
```

## Pattern 3 — Multiple tickers in one fetch

Fetch the index and the VIX together so you can classify the volatility regime
directly from the implied-vol level.

```python
import yfinance as yf
import pandas as pd

data = yf.download(["^GSPC", "^VIX"], start="1990-01-01", end="2026-01-01", auto_adjust=False)
close = data["Close"].reset_index()
close.columns = ["date", "gspc", "vix"]
close["date"] = pd.to_datetime(close["date"]).dt.tz_localize(None)
close = close.sort_values("date").reset_index(drop=True)
```

## Working with log returns

The workshop target is the close-to-close **cumulative log return** over `N`
business days. Build it from the close series:

```python
import numpy as np

df["logret_1b"] = np.log(df["close"] / df["close"].shift(1))
# N-business-day forward cumulative return (what horizon N resolves to):
N = 5
df["logret_5b_fwd"] = np.log(df["close"].shift(-N) / df["close"])
```
