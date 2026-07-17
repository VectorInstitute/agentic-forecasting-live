"""Shared data + style helpers for the Part-1 blog figures.

Every figure script imports from here so that the realised return series, the
prediction-store loaders, the CRPS definition, the landmark windows, and the
plotting palette are defined once and identically. All numbers ultimately come
from two repo artefacts:

* the leak-safe S&P/TSX Composite data service
  (:func:`workshop_experiments.data_tsx.build_tsx_workshop_service`), which
  supplies the realised ``tsx_logret_{1,5,21}b`` series used for both scoring and
  the price path; and
* the persisted prediction stores under
  ``workshop_experiments/workshop_experiments/data/predictions/<spec>/<model>/<task>/<origin>.yaml``.

CRPS is computed with the exact same call the scoring layer uses
(``properscoring.crps_ensemble`` over the sorted quantile grid), verified to
reproduce ``leaderboard.csv`` to the printed precision.
"""

from __future__ import annotations

import glob
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import properscoring as ps
import yaml


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[3]
WS = REPO_ROOT / "workshop_experiments" / "workshop_experiments"
PRED_DIR = WS / "data" / "predictions"
RESULTS_DIR = WS / "data" / "results"
ASSETS_DIR = _THIS.parent

TARGETS = ["tsx_logret_1b", "tsx_logret_5b", "tsx_logret_21b"]
HORIZON_OF = {"tsx_logret_1b": 1, "tsx_logret_5b": 5, "tsx_logret_21b": 21}


# --------------------------------------------------------------------------- #
# Realised series (authoritative, from the data service)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _service():
    from workshop_experiments.data_tsx import build_tsx_workshop_service

    # Covariate-free build: the forecast targets resolve either way and no macro
    # panel is needed to score. refresh=False -> use the local Yahoo cache.
    return build_tsx_workshop_service(include_covariates=False, refresh=False)


@lru_cache(maxsize=8)
def realized(target_series_id: str) -> pd.Series:
    """Realised value series for a target, indexed by (tz-naive) timestamp."""
    svc = _service()
    frame = svc.get_series(target_series_id, as_of=pd.Timestamp("2026-08-01")).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.set_index("timestamp")["value"].sort_index()


@lru_cache(maxsize=1)
def tsx_close() -> pd.Series:
    """S&P/TSX Composite adjusted-close level, reconstructed from realised 1b log returns.

    Anchored to the first cached close so the level is a true price path (the
    landmark *percentage* moves are anchor-invariant). Reconstructing from the
    data-service returns keeps the figure regenerable without the raw parquet.
    """
    r1 = realized("tsx_logret_1b")
    close_cache = REPO_ROOT / "data" / "yfinance" / "gsptse_adj_close_1d.parquet"
    if close_cache.exists():
        raw = pd.read_parquet(close_cache)
        s = raw.set_index("timestamp")["value"].sort_index()
        s.index = pd.to_datetime(s.index)
        return s
    # Fallback: reconstruct a relative price path from cumulative log returns.
    return np.exp(r1.cumsum())


# --------------------------------------------------------------------------- #
# Prediction-store loaders
# --------------------------------------------------------------------------- #
def _model_task_dir(spec: str, model: str, target: str) -> Path:
    return PRED_DIR / spec / model / target


def load_predictions(spec: str, model: str, target: str) -> pd.DataFrame:
    """Load one model/task's persisted predictions.

    Returns a frame indexed by origin (``as_of``) date with columns
    ``forecast_date``, ``median`` (q0.5 point) and ``quantiles`` (sorted ndarray).
    Empty frame if the directory is absent.
    """
    task_dir = _model_task_dir(spec, model, target)
    rows = []
    if not task_dir.is_dir():
        return pd.DataFrame(columns=["origin", "forecast_date", "median", "quantiles"]).set_index("origin")
    for origin_file in sorted(glob.glob(glob.escape(str(task_dir)) + "/*.yaml")):
        doc = yaml.safe_load(Path(origin_file).read_text())
        for pred in doc.get("predictions", []):
            payload = pred.get("payload") or {}
            quantiles = payload.get("quantiles")
            if not quantiles:
                continue
            values = np.array(sorted(float(v) for v in quantiles.values()), dtype=float)
            rows.append(
                {
                    "origin": pd.Timestamp(pred["as_of"]).normalize(),
                    "forecast_date": pd.Timestamp(pred["forecast_date"]).normalize(),
                    "median": float(quantiles.get("0.5", payload.get("point_forecast", np.nan))),
                    "quantiles": values,
                },
            )
    if not rows:
        return pd.DataFrame(columns=["origin", "forecast_date", "median", "quantiles"]).set_index("origin")
    return pd.DataFrame(rows).set_index("origin").sort_index()


def crps_series(spec: str, model: str, target: str) -> pd.Series:
    """Per-origin CRPS for one model/task, indexed by origin date.

    Uses the same ``crps_ensemble`` over the sorted quantile grid as the scoring
    layer. Origins whose forecast date has not resolved are dropped.
    """
    preds = load_predictions(spec, model, target)
    if preds.empty:
        return pd.Series(dtype=float)
    look = realized(target)
    out = {}
    for origin, row in preds.iterrows():
        fdate = row["forecast_date"]
        if fdate not in look.index:
            continue
        out[origin] = float(ps.crps_ensemble(float(look.loc[fdate]), row["quantiles"]))
    return pd.Series(out).sort_index()


def mean_crps(spec: str, model: str, target: str) -> tuple[float, int]:
    """Mean CRPS and resolved-origin count for one model/task."""
    s = crps_series(spec, model, target)
    if s.empty:
        return (float("nan"), 0)
    return (float(s.mean()), int(s.size))


# --------------------------------------------------------------------------- #
# Landmark windows (identities from the market timeline; TSX magnitudes computed)
# --------------------------------------------------------------------------- #
def _pct(a: float, b: float) -> float:
    return (b / a - 1.0) * 100.0


@lru_cache(maxsize=1)
def landmarks() -> list[dict]:
    """Landmark windows with TSX magnitudes computed from the close path.

    Each dict: label, start, end, kind ('drawdown'|'rebound'), peak/trough dates
    and levels, and pct (signed % move peak->trough or trough->recovery).
    """
    close = tsx_close()

    def seg(peak_lo, peak_hi, tr_lo, tr_hi):
        pk = close[peak_lo:peak_hi]
        tr = close[tr_lo:tr_hi]
        return pk.idxmax(), float(pk.max()), tr.idxmin(), float(tr.min())

    # 2025 tariff drawdown: pre-tariff high -> 08 Apr crash low.
    pk1_d, pk1_v, tr1_d, tr1_v = seg("2025-01-01", "2025-04-08", "2025-03-15", "2025-05-15")
    # 2025 rebound: crash low -> full recovery by end-June.
    reb1_d = pd.Timestamp("2025-06-30")
    reb1_v = float(close[:reb1_d].iloc[-1])
    # 2026 war drawdown: early-Mar high -> 20 Mar low.
    pk2_d, pk2_v, tr2_d, tr2_v = seg("2026-02-01", "2026-03-05", "2026-03-05", "2026-04-15")
    # 2026 April rebound: 20 Mar low -> end-Apr.
    reb2_d = pd.Timestamp("2026-04-30")
    reb2_v = float(close[:reb2_d].iloc[-1])

    return [
        {
            "label": "2025 tariff drawdown",
            "start": pk1_d,
            "end": tr1_d,
            "kind": "drawdown",
            "pct": _pct(pk1_v, tr1_v),
            "peak_date": pk1_d,
            "trough_date": tr1_d,
        },
        {
            "label": "2025 rebound",
            "start": tr1_d,
            "end": reb1_d,
            "kind": "rebound",
            "pct": _pct(tr1_v, reb1_v),
            "peak_date": reb1_d,
            "trough_date": tr1_d,
        },
        {
            "label": "2026 war drawdown",
            "start": pk2_d,
            "end": tr2_d,
            "kind": "drawdown",
            "pct": _pct(pk2_v, tr2_v),
            "peak_date": pk2_d,
            "trough_date": tr2_d,
        },
        {
            "label": "2026 April rebound",
            "start": tr2_d,
            "end": reb2_d,
            "kind": "rebound",
            "pct": _pct(tr2_v, reb2_v),
            "peak_date": reb2_d,
            "trough_date": tr2_d,
        },
    ]


# --------------------------------------------------------------------------- #
# Palette + matplotlib style (dataviz-skill reference instance, light mode)
# --------------------------------------------------------------------------- #
# Categorical hues in fixed slot order (never cycled).
CAT = {
    "blue": "#2a78d6",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
    "magenta": "#e87ba4",
    "orange": "#eb6834",
}
INK = {
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}


def apply_style() -> None:
    """Apply the shared light-mode matplotlib rcParams (recessive chrome)."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": INK["surface"],
            "axes.facecolor": INK["surface"],
            "savefig.facecolor": INK["surface"],
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "text.color": INK["primary"],
            "axes.edgecolor": INK["axis"],
            "axes.labelcolor": INK["secondary"],
            "axes.titlecolor": INK["primary"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": INK["grid"],
            "grid.linewidth": 0.7,
            "xtick.color": INK["muted"],
            "ytick.color": INK["muted"],
            "xtick.labelcolor": INK["secondary"],
            "ytick.labelcolor": INK["secondary"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 220,
            "savefig.dpi": 220,
        },
    )


def savefig(fig, name: str) -> Path:
    """Save a figure PNG into the assets dir at >=200 dpi."""
    out = ASSETS_DIR / name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    return out


if __name__ == "__main__":
    for lm in landmarks():
        print(f"{lm['label']:24s} {lm['start'].date()} -> {lm['end'].date()}  {lm['pct']:+.2f}%")
