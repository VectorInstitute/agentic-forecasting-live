"""Spec loading / validation and origin-count tests for the four workshop specs."""

from __future__ import annotations

import pytest
from aieng.forecasting.evaluation import MultiTargetBacktestSpec
from workshop_experiments.data import WORKSHOP_HORIZONS
from workshop_experiments.specs import SPEC_NAMES, load_spec, origin_count


EXPECTED_TARGETS = {"sp500_logret_1b", "sp500_logret_5b", "sp500_logret_21b"}
EXPECTED_HORIZONS = set(WORKSHOP_HORIZONS)


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_spec_loads_and_validates(name: str) -> None:
    """Each committed spec parses into a MultiTargetBacktestSpec."""
    spec = load_spec(name)
    assert isinstance(spec, MultiTargetBacktestSpec)
    assert spec.spec_id == name


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_spec_targets_and_horizons(name: str) -> None:
    """Every spec forecasts the three return targets at h = 1 / 5 / 21."""
    spec = load_spec(name)
    targets = {t.target_series_id for t in spec.tasks}
    horizons = {h for t in spec.tasks for h in t.horizons}
    assert targets == EXPECTED_TARGETS
    assert horizons == EXPECTED_HORIZONS
    for task in spec.tasks:
        assert task.frequency == "B"


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_spec_has_positive_origin_count(name: str) -> None:
    """Every spec generates at least one candidate origin."""
    assert origin_count(load_spec(name)) > 0


def test_smoke_spec_has_three_origins() -> None:
    """The smoke spec is exactly three weekly origins (fast proof-of-life)."""
    assert origin_count(load_spec("sp500_ws_smoke")) == 3


def test_eval_2026_origins_resolve_by_mid_july() -> None:
    """The last 2026 eval origin's h=21 forecast resolves on/before 2026-07-15."""
    import pandas as pd  # noqa: PLC0415

    spec = load_spec("sp500_ws_eval_2026_weekly")
    last_origin = max(spec.specs()[0].origins())
    resolved = pd.Timestamp(last_origin) + pd.offsets.BDay(21)
    assert resolved <= pd.Timestamp("2026-07-15")


def test_weekly_backtest_is_about_52_origins() -> None:
    """The 2025 canonical backtest is a full year of weekly origins (~52)."""
    n = origin_count(load_spec("sp500_ws_backtest_2025_weekly"))
    assert 48 <= n <= 54


def test_daily_spec_is_dense() -> None:
    """The daily bonus layer spans ~18 months of business days."""
    n = origin_count(load_spec("sp500_ws_daily_2025_2026"))
    assert n > 300
