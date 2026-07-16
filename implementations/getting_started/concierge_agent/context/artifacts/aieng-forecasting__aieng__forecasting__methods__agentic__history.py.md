# Source: aieng-forecasting/aieng/forecasting/methods/agentic/history.py

kind: python

```python
"""History compression for agent prompt payloads.

Serialising a full daily price history into an agent prompt is wasteful and can
overflow the context window.  :func:`compress_history` keeps recent daily
resolution while down-sampling older history to weekly averages, producing a
compact ``date,close`` CSV suitable for embedding in a JSON payload.
"""

from __future__ import annotations

import pandas as pd


def compress_history(df: pd.DataFrame, *, recent_months: int = 6) -> str:
    """Compress a daily price history to stay within context limits.

    Returns daily bars for the most recent ``recent_months`` months and weekly
    averages for older history.  The CSV header is ``date,close``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``timestamp`` and ``value``.
    recent_months : int, default=6
        Number of trailing months to keep at daily resolution.  Older history
        is resampled to weekly averages.

    Returns
    -------
    str
        CSV string with header ``date,close``.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = df["timestamp"].max() - pd.DateOffset(months=recent_months)

    recent = df[df["timestamp"] >= cutoff].copy()
    old = df[df["timestamp"] < cutoff].copy()

    rows: list[str] = ["date,close"]

    if not old.empty:
        old_indexed = old.set_index("timestamp")["value"]
        weekly: pd.Series = old_indexed.resample("W").mean().dropna()
        for date, val in weekly.items():
            rows.append(f"{date.date()},{val:.2f}")

    for _, row in recent.iterrows():
        rows.append(f"{row['timestamp'].date()},{row['value']:.2f}")

    return "\n".join(rows)
```
