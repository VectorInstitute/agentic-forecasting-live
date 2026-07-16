"""Registry construction and method-resolution tests (no network / no API)."""

from __future__ import annotations

import pytest
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from workshop_experiments.registry import (
    ALL_METHODS,
    API_METHODS,
    CONVENTIONAL_METHODS,
    build_predictor,
    resolve_methods,
)


@pytest.mark.parametrize("name", CONVENTIONAL_METHODS)
def test_conventional_predictors_construct(name: str) -> None:
    """Every conventional (non-API) method builds a Predictor with a stable id."""
    predictor = build_predictor(name)
    assert isinstance(predictor, Predictor)
    assert isinstance(predictor.predictor_id, str)
    assert predictor.predictor_id


def test_conventional_predictor_ids() -> None:
    """Conventional predictor ids match the documented, model-free conventions."""
    expected = {
        "naive": "last_value_naive",
        "ets": "darts_ets",
        "kalman": "darts_kalman",
        "autoarima": "darts_autoarima",
        "lightgbm": "darts_lightgbm",
        "lightgbm_cov": "darts_lightgbm_cov",
    }
    for name, predictor_id in expected.items():
        assert build_predictor(name).predictor_id == predictor_id


def test_covariate_variant_differs_from_target_only() -> None:
    """The covariate LightGBM variant has a distinct id (separate cache key)."""
    assert build_predictor("lightgbm").predictor_id != build_predictor("lightgbm_cov").predictor_id


def test_llmp_predictor_id_encodes_model_and_variant() -> None:
    """LLMP ids fold the model and covariate variant so caches stay separate."""
    qgrid = build_predictor("llmp_qgrid", model=LITE_MODEL)
    qgrid_cov = build_predictor("llmp_qgrid_cov", model=LITE_MODEL)
    qgrid_adv = build_predictor("llmp_qgrid", model=ADVANCED_MODEL)

    assert LITE_MODEL in qgrid.predictor_id
    assert "sp500_ws" in qgrid.predictor_id
    assert qgrid.predictor_id != qgrid_cov.predictor_id
    assert qgrid.predictor_id != qgrid_adv.predictor_id
    assert ADVANCED_MODEL in qgrid_adv.predictor_id


def test_unknown_method_raises() -> None:
    """An unregistered method name raises KeyError."""
    with pytest.raises(KeyError):
        build_predictor("does_not_exist")


def test_resolve_methods_expands_groups_and_dedupes() -> None:
    """Group aliases expand and duplicates collapse, preserving first-seen order."""
    assert resolve_methods(["conventional"]) == list(CONVENTIONAL_METHODS)
    assert resolve_methods(["all"]) == list(ALL_METHODS)
    assert set(resolve_methods(["api"])) == set(API_METHODS)
    assert resolve_methods(["naive", "naive", "ets"]) == ["naive", "ets"]


def test_resolve_methods_rejects_unknown_token() -> None:
    """An unknown selector token raises KeyError."""
    with pytest.raises(KeyError):
        resolve_methods(["not_a_method"])
