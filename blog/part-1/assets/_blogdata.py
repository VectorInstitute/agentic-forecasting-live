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
import os
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

# Optional overrides so the figures can be re-rendered against a data root other
# than this checkout (e.g. a live runner clone holding refreshed leaderboards and
# prediction stores). Defaults preserve the in-repo behaviour.
#   BLOG_WS_DATA_ROOT -> dir that contains predictions/ and results/
#   BLOG_YF_CACHE     -> yfinance cache dir (holds gsptse_adj_close_1d.parquet, ...)
_WS_DATA_ROOT = Path(os.environ["BLOG_WS_DATA_ROOT"]) if os.environ.get("BLOG_WS_DATA_ROOT") else WS / "data"
_YF_CACHE = Path(os.environ["BLOG_YF_CACHE"]) if os.environ.get("BLOG_YF_CACHE") else REPO_ROOT / "data" / "yfinance"
PRED_DIR = _WS_DATA_ROOT / "predictions"
RESULTS_DIR = _WS_DATA_ROOT / "results"
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
    # panel is needed to score. refresh=False -> use the local Yahoo cache
    # (resolved by the data service from REPO_ROOT/data/yfinance).
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
    close_cache = _YF_CACHE / "gsptse_adj_close_1d.parquet"
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


# Font-size ladder for every blog figure. Sizes are points at the shared 220-dpi
# render. ``small`` is the floor: no text on any figure may be smaller, and
# savefig() enforces that. Tick/label/legend sizes are pushed into rcParams by
# apply_style(), so scripts should not pass ``fontsize=`` to routine axis calls.
FS = {
    "title": 17,  # figure_title() — the one "Figure N." headline
    "axtitle": 13,  # per-axes / panel headers
    "label": 12,  # axis labels
    "tick": 11,  # tick labels
    "legend": 12,  # legend entries
    "annot": 11,  # in-plot annotations (also the rc default font size)
    "small": 10,  # the floor — nothing smaller may exist
}

DPI = 220


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
            "font.size": FS["annot"],
            "text.color": INK["primary"],
            "axes.edgecolor": INK["axis"],
            "axes.labelcolor": INK["secondary"],
            "axes.titlecolor": INK["primary"],
            "axes.titlesize": FS["axtitle"],
            "axes.labelsize": FS["label"],
            "figure.titlesize": FS["title"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": INK["grid"],
            "grid.linewidth": 0.7,
            "xtick.color": INK["muted"],
            "ytick.color": INK["muted"],
            "xtick.labelsize": FS["tick"],
            "ytick.labelsize": FS["tick"],
            "xtick.labelcolor": INK["secondary"],
            "ytick.labelcolor": INK["secondary"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": FS["legend"],
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
        },
    )


def figure_title(target, number: int, text: str, **kw):
    """Draw the numbered headline — "Figure N.  Text" — on an Axes or a Figure.

    The number burned into the PNG must match the caption number in post.md;
    both parts' reading order is pinned in assets/CAPTIONS.md. Extra kwargs pass
    through to ``set_title`` / ``suptitle`` (e.g. ``x=``, ``y=`` to align a
    suptitle with the axes block).
    """
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    label = f"Figure {number}.  {text}"
    common = {"fontsize": FS["title"], "fontweight": "bold", "color": INK["primary"]}
    if isinstance(target, Axes):
        kw.setdefault("loc", "left")
        kw.setdefault("pad", 10)
        return target.set_title(label, **common, **kw)
    if isinstance(target, Figure):
        kw.setdefault("x", 0.01)
        kw.setdefault("ha", "left")
        return target.suptitle(label, **common, **kw)
    raise TypeError(f"figure_title target must be an Axes or Figure, got {type(target)!r}")


def _png_width_px(path: Path) -> int:
    import struct

    head = path.read_bytes()[:24]
    if head[12:16] != b"IHDR":  # pragma: no cover - corrupt write
        raise ValueError(f"{path.name}: not a valid PNG header")
    return struct.unpack(">I", head[16:20])[0]


def _figure_text_below_axes(fig) -> list[tuple[str, float]]:
    """Figure-level Text artists sitting below the axes block (y < 0.10)."""
    inv = fig.transFigure.inverted()
    bad = []
    for t in fig.texts:
        s = t.get_text().strip()
        if not s or not t.get_visible():
            continue
        _, fy = inv.transform(t.get_transform().transform(t.get_position()))
        if fy < 0.10:
            bad.append((s.replace("\n", " ")[:60], round(float(fy), 3)))
    return bad


def _undersized_texts(fig) -> list[tuple[str, float]]:
    """Visible, non-empty Text artists below the FS['small'] floor."""
    from matplotlib.text import Text

    bad = []
    for t in fig.findobj(Text):
        s = t.get_text()
        if not s or not s.strip() or not t.get_visible():
            continue
        size = float(t.get_fontsize())
        if size < FS["small"] - 1e-6:
            bad.append((s.strip().replace("\n", " ")[:50], size))
    return bad


def savefig(fig, name: str, out_dir: Path | None = None) -> Path:
    """Save a figure PNG at the shared 220 dpi, enforcing the chrome-out contract.

    Refuses to save when:

    * any figure-level text sits below the axes block (y < 0.10 in figure
      coords) — the footnote idiom. Footnotes belong in the markdown caption
      and in assets/CAPTIONS.md, never on the PNG;
    * any visible text is smaller than the ``FS['small']`` floor;
    * after writing, the PNG is more than 2% wider than ``figsize`` × 220 dpi —
      ``bbox_inches="tight"`` silently *expands* the canvas around over-wide
      artists, which then shrinks the plot when the PNG is scaled into the page.

    ``out_dir`` defaults to the calling script's own directory, so the one
    helper serves both parts' asset folders.
    """
    if out_dir is None:
        import inspect

        out_dir = Path(inspect.stack()[1].filename).resolve().parent
        if not out_dir.is_dir():  # e.g. interactive use
            out_dir = ASSETS_DIR

    low = _figure_text_below_axes(fig)
    if low:
        raise ValueError(
            f"{name}: figure-level text below the axes block {low} — footnotes are "
            "banned on rendered figures; move the text to the post caption/CAPTIONS.md",
        )
    small = _undersized_texts(fig)
    if small:
        raise ValueError(
            f"{name}: text below the FS['small']={FS['small']}pt floor: {small}",
        )

    out = Path(out_dir) / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")

    width_px = _png_width_px(out)
    nominal = fig.get_figwidth() * DPI
    if width_px / nominal > 1.02:
        raise ValueError(
            f"{name}: saved width {width_px}px is {width_px / nominal:.1%} of the "
            f"nominal {nominal:.0f}px ({fig.get_figwidth()}in × {DPI}dpi) — an "
            "over-wide artist is inflating the tight bbox; wrap or shrink it",
        )
    return out


if __name__ == "__main__":
    for lm in landmarks():
        print(f"{lm['label']:24s} {lm['start'].date()} -> {lm['end'].date()}  {lm['pct']:+.2f}%")
