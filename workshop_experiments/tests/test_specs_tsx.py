"""Spec loading / validation and origin-count tests for the four TSX workshop specs."""

from __future__ import annotations

import pandas as pd
import pytest
from aieng.forecasting.evaluation import MultiTargetBacktestSpec
from workshop_experiments.data_tsx import WORKSHOP_HORIZONS
from workshop_experiments.specs import TSX_SPEC_NAMES, load_spec, origin_count


EXPECTED_TARGETS = {"tsx_logret_1b", "tsx_logret_5b", "tsx_logret_21b"}
EXPECTED_HORIZONS = set(WORKSHOP_HORIZONS)


@pytest.mark.parametrize("name", TSX_SPEC_NAMES)
def test_spec_loads_and_validates(name: str) -> None:
    """Each committed TSX spec parses into a MultiTargetBacktestSpec."""
    spec = load_spec(name)
    assert isinstance(spec, MultiTargetBacktestSpec)
    assert spec.spec_id == name


@pytest.mark.parametrize("name", TSX_SPEC_NAMES)
def test_spec_targets_and_horizons(name: str) -> None:
    """Every TSX spec forecasts the three return targets at h = 1 / 5 / 21."""
    spec = load_spec(name)
    targets = {t.target_series_id for t in spec.tasks}
    horizons = {h for t in spec.tasks for h in t.horizons}
    assert targets == EXPECTED_TARGETS
    assert horizons == EXPECTED_HORIZONS
    for task in spec.tasks:
        assert task.frequency == "B"


def test_smoke_spec_has_three_origins() -> None:
    """The TSX smoke spec is exactly three weekly origins (fast proof-of-life)."""
    assert origin_count(load_spec("tsx_ws_smoke")) == 3


def test_smoke_origins_are_tsx_sessions() -> None:
    """The smoke origins avoid Canadian Thanksgiving (TSX-session-keyed, not NYSE)."""
    spec = load_spec("tsx_ws_smoke")
    origins = {pd.Timestamp(o).date().isoformat() for o in spec.specs()[0].origins()}
    assert origins == {"2025-11-03", "2025-11-10", "2025-11-17"}
    # Oct 13 2025 (Canadian Thanksgiving, TSX closed) must not be an origin.
    assert "2025-10-13" not in origins


def test_eval_2026_origins_resolve_by_mid_july() -> None:
    """The last 2026 eval origin's h=21 forecast resolves on/before 2026-07-15."""
    spec = load_spec("tsx_ws_eval_2026_weekly")
    last_origin = max(spec.specs()[0].origins())
    assert pd.Timestamp(last_origin) <= pd.Timestamp("2026-06-15")
    resolved = pd.Timestamp(last_origin) + pd.offsets.BDay(21)
    assert resolved <= pd.Timestamp("2026-07-15")


def test_weekly_backtest_is_about_52_origins() -> None:
    """The 2025 canonical backtest is a full year of weekly origins (~52)."""
    n = origin_count(load_spec("tsx_ws_backtest_2025_weekly"))
    assert 48 <= n <= 54


def test_daily_spec_is_dense() -> None:
    """The daily bonus layer spans ~18 months of business days."""
    assert origin_count(load_spec("tsx_ws_daily_2025_2026")) > 300
