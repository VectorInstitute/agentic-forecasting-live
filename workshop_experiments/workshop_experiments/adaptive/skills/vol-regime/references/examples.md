# vol-regime — working code patterns

Combine each pattern with a `fetch-yfinance` block in a single `run_code` call.

## Pattern 1 — Regime from the VIX level

```python
def classify_vix(vix_level: float) -> str:
    if vix_level < 15.0:
        return "low"
    if vix_level < 20.0:
        return "normal"
    if vix_level < 30.0:
        return "elevated"
    return "high"

latest_vix = float(close["vix"].iloc[-1])   # from a ^GSPC + ^VIX fetch
regime = classify_vix(latest_vix)
```

## Pattern 2 — Regime from realised vol (VIX unavailable)

```python
import numpy as np

logret = np.log(df["close"] / df["close"].shift(1)).dropna()
realised_vol_ann = float(logret.tail(21).std() * np.sqrt(252) * 100.0)

def classify_realised(vol_ann: float) -> str:
    if vol_ann < 12.0:
        return "low"
    if vol_ann < 18.0:
        return "normal"
    if vol_ann < 28.0:
        return "elevated"
    return "high"

regime = classify_realised(realised_vol_ann)
```

## Pattern 3 — Anomaly detection (z-score of the latest return)

```python
import numpy as np

logret = np.log(df["close"] / df["close"].shift(1)).dropna()
recent = logret.tail(63)
z = float((logret.iloc[-1] - recent.mean()) / recent.std())
is_anomaly = abs(z) > 2.5
```

## Pattern 4 — Adaptive dispersion window

```python
dispersion_window = 10 if (regime in ("elevated", "high") or is_anomaly) else 21
```

Use the shorter window when the regime is elevated/high or a recent anomaly
indicates the regime is shifting, so the interval width tracks the current
turbulence rather than a stale calm stretch.
