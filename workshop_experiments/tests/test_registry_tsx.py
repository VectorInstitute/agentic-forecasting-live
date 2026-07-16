"""TSX registry construction and method-resolution tests (no network / no API)."""

from __future__ import annotations

import pytest
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from workshop_experiments.registry import (
    TSX_AGENT_METHODS,
    TSX_CONVENTIONAL_METHODS,
    TSX_LLMP_METHODS,
    build_predictor,
    resolve_methods,
)


@pytest.mark.parametrize("name", TSX_CONVENTIONAL_METHODS)
def test_tsx_conventional_predictors_construct(name: str) -> None:
    """Every TSX conventional method builds a Predictor with a stable id."""
    predictor = build_predictor(name)
    assert isinstance(predictor, Predictor)
    assert predictor.predictor_id


def test_tsx_llmp_predictor_id_encodes_model_and_tsx_variant() -> None:
    """TSX LLMP ids fold the model and a tsx variant tag (distinct from sp500)."""
    qgrid = build_predictor("tsx_llmp_qgrid", model=LITE_MODEL)
    qgrid_cov = build_predictor("tsx_llmp_qgrid_cov", model=LITE_MODEL)
    qgrid_adv = build_predictor("tsx_llmp_qgrid", model=ADVANCED_MODEL)

    assert LITE_MODEL in qgrid.predictor_id
    assert "tsx_ws" in qgrid.predictor_id
    assert qgrid.predictor_id != qgrid_cov.predictor_id
    assert qgrid.predictor_id != qgrid_adv.predictor_id
    # Distinct from the S&P 500 LLMP rung with the same model (separate cache).
    assert qgrid.predictor_id != build_predictor("llmp_qgrid", model=LITE_MODEL).predictor_id


def test_tsx_agent_ids_are_tsx_distinct() -> None:
    """TSX agent predictor ids carry the tsx_analyst name (distinct from sp500)."""
    news = build_predictor("tsx_agent_news", model=LITE_MODEL)
    code = build_predictor("tsx_agent_code", model=LITE_MODEL)
    assert "tsx_analyst" in news.predictor_id
    assert "tsx_analyst" in code.predictor_id
    assert news.predictor_id != build_predictor("agent_news", model=LITE_MODEL).predictor_id


def test_tsx_covariate_variant_differs_from_target_only() -> None:
    """The covariate LightGBM variant has a distinct id (separate cache key)."""
    assert build_predictor("tsx_lightgbm").predictor_id != build_predictor("tsx_lightgbm_cov").predictor_id


def test_resolve_methods_expands_tsx_groups() -> None:
    """The tsx_* group aliases expand to the TSX method tuples."""
    assert resolve_methods(["tsx_conventional"]) == list(TSX_CONVENTIONAL_METHODS)
    assert resolve_methods(["tsx_llmp"]) == list(TSX_LLMP_METHODS)
    assert resolve_methods(["tsx_agent"]) == list(TSX_AGENT_METHODS)
    assert set(resolve_methods(["tsx_all"])) == set(TSX_CONVENTIONAL_METHODS + TSX_LLMP_METHODS + TSX_AGENT_METHODS)


def test_conventional_group_excludes_tsx() -> None:
    """The plain `conventional` alias stays S&P 500-only (no tsx bleed)."""
    assert all(not m.startswith("tsx_") for m in resolve_methods(["conventional"]))
