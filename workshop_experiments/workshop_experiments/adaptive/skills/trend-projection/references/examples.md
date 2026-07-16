# trend-projection — working code patterns

Combine with `fetch-yfinance` and `vol-regime` in a single `run_code` call.

## Pattern 1 — Centre the forecast (martingale prior)

```python
# Median cumulative log return at every horizon starts at 0. Only shift it when
# a dated catalyst in the news context clearly justifies a directional lean.
median = 0.0
```

## Pattern 2 — Horizon-scaled interval half-width

```python
import numpy as np

logret = np.log(df["close"] / df["close"].shift(1)).dropna()
sigma_daily = float(logret.tail(dispersion_window).std())   # window from vol-regime

def half_width(z_q: float, horizon: int) -> float:
    return z_q * sigma_daily * np.sqrt(horizon)
```

## Pattern 3 — Emit the standard 11-point quantile grid

```python
from scipy.stats import norm

STANDARD_QUANTILES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

def quantile_grid(median: float, horizon: int, widen: float = 1.0) -> list[dict]:
    grid = []
    for q in STANDARD_QUANTILES:
        z_q = float(norm.ppf(q))
        value = median + widen * z_q * sigma_daily * np.sqrt(horizon)
        grid.append({"quantile": q, "value": value})
    # Non-decreasing by construction (z_q is monotone); point == 0.50 quantile.
    return grid
```

## Full Pipeline Example

```python
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

as_of = "2025-04-01"                       # from the prediction payload
horizons = [1, 5, 21]

# 1. fetch-yfinance
raw = yf.Ticker("^GSPC").history(start="1990-01-01", end=as_of, auto_adjust=False)
df = raw.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
df = df[df["date"] < pd.Timestamp(as_of)].sort_values("date").reset_index(drop=True)

# 2. vol-regime
logret = np.log(df["close"] / df["close"].shift(1)).dropna()
realised_vol_ann = float(logret.tail(21).std() * np.sqrt(252) * 100.0)
regime = "high" if realised_vol_ann > 28 else "elevated" if realised_vol_ann > 18 else "normal"
dispersion_window = 10 if regime in ("elevated", "high") else 21
sigma_daily = float(logret.tail(dispersion_window).std())

# 3. trend-projection — apply the sp500-strategy widen correction if the regime calls for it
widen = 1.12 if regime in ("elevated", "high") else 1.0
STANDARD_QUANTILES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
forecast = {}
for h in horizons:
    grid = [
        {"quantile": q, "value": widen * float(norm.ppf(q)) * sigma_daily * np.sqrt(h)}
        for q in STANDARD_QUANTILES
    ]
    forecast[h] = {"point": grid[5]["value"], "quantiles": grid}

print(regime, dispersion_window, sigma_daily)
```
