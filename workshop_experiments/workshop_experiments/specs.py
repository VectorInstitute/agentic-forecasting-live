"""Spec loading for the workshop S&P 500 experiments.

The workshop specs are plain
:class:`~aieng.forecasting.evaluation.MultiTargetBacktestSpec` YAML files (one
single-horizon task per target: ``sp500_logret_{1,5,21}b`` at h = 1 / 5 / 21
business days). The protected 2026 window is modelled as a
backtest spec too: the workshop's leakage discipline is the smoke-gate plus
resumable, immediately-persisted per-origin predictions, and scoring never
re-calls the API — so the notebook-era ``EvalTracker`` run budget is obviated.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from aieng.forecasting.evaluation import MultiTargetBacktestSpec


#: Directory holding the committed workshop spec YAMLs.
SPECS_DIR = Path(__file__).resolve().parent / "specs"

#: The canonical S&P 500 workshop specs, in method-ladder-friendly order.
SPEC_NAMES: tuple[str, ...] = (
    "sp500_ws_smoke",
    "sp500_ws_backtest_2025_weekly",
    "sp500_ws_eval_2026_weekly",
    "sp500_ws_daily_2025_2026",
)

#: The Canada-focused S&P/TSX Composite workshop specs (the deployed primary
#: target), mirroring the S&P 500 spec conventions rung-for-rung.
TSX_SPEC_NAMES: tuple[str, ...] = (
    "tsx_ws_smoke",
    "tsx_ws_backtest_2025_weekly",
    "tsx_ws_eval_2026_weekly",
    "tsx_ws_daily_2025_2026",
)


def spec_path(name_or_path: str) -> Path:
    """Resolve a spec selector to a YAML path.

    Accepts a bare spec name (``sp500_ws_smoke``), a name with extension, or an
    explicit filesystem path.
    """
    candidate = Path(name_or_path)
    if candidate.exists():
        return candidate
    stem = candidate.name
    if not stem.endswith((".yaml", ".yml")):
        stem = f"{stem}.yaml"
    resolved = SPECS_DIR / stem
    if not resolved.exists():
        raise FileNotFoundError(
            f"No workshop spec {name_or_path!r} (looked for {resolved}). "
            f"Known: {', '.join((*SPEC_NAMES, *TSX_SPEC_NAMES))}."
        )
    return resolved


def load_spec(name_or_path: str) -> MultiTargetBacktestSpec:
    """Load and validate a workshop spec into a :class:`MultiTargetBacktestSpec`."""
    with spec_path(name_or_path).open() as f:
        data = yaml.safe_load(f)
    return MultiTargetBacktestSpec.model_validate(data)


def origin_count(spec: MultiTargetBacktestSpec) -> int:
    """Return the number of candidate origins the spec's stride generates.

    This is the pre-warmup grid size (the same for every task, since all tasks
    share the window and business-day frequency). The realised scored-origin
    count can be lower once warmup and unresolved future dates are applied.
    """
    from aieng.forecasting.evaluation.backtest import _compute_origins  # noqa: PLC0415

    first_task = spec.tasks[0]
    origins = _compute_origins(spec.start, spec.end, first_task.frequency, spec.stride)
    return len(origins)


__all__ = [
    "SPECS_DIR",
    "SPEC_NAMES",
    "TSX_SPEC_NAMES",
    "load_spec",
    "origin_count",
    "spec_path",
]
