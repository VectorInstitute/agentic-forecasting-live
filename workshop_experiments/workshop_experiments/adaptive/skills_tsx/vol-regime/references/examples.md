# vol-regime — working code patterns

Combine each pattern with a `fetch-yfinance` block in a single `run_code` call.

## Pattern 1 — Regime from realised vol (primary classifier)

The TSX has no native implied-vol index, so classify directly from 21-day
realised vol of the index returns.

```python
import numpy as np

logret = np.log(df["close"] / df["close"].shift(1)).dropna()   # from a ^GSPTSE fetch
realised_vol_ann = float(logret.tail(21).std() * np.sqrt(252) * 100.0)

def classify_realised(vol_ann: float) -> str:
    if vol_ann < 10.0:
        return "low"
    if vol_ann < 15.0:
        return "normal"
    if vol_ann < 22.0:
        return "elevated"
    return "high"

regime = classify_realised(realised_vol_ann)
```

## Pattern 2 — US VIX as a cross-market risk cross-check

The US `^VIX` is not the TSX's own implied vol, but a spike in US risk usually
spills into Canadian equities. Use it only to corroborate — never to override —
the realised-vol regime from Pattern 1.

```python
latest_vix = float(close["vix"].iloc[-1])   # from a ^GSPTSE + ^VIX fetch
vix_stress = latest_vix > 25.0              # elevated US risk backdrop
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
