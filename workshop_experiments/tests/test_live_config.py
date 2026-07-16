"""Live-harness config loading and ladder-expansion tests (offline)."""

from __future__ import annotations

import pytest
from workshop_experiments.live.config import (
    LiveConfig,
    expand_predictors,
    load_config,
)


def test_default_config_loads() -> None:
    """The shipped live_config.yaml parses into a LiveConfig with a ladder."""
    config = load_config()
    assert isinstance(config, LiveConfig)
    assert config.horizons == (1, 5, 21)
    assert config.submission_time_local == "17:30"
    assert config.timezone == "America/Toronto"
    assert config.retry.max_attempts == 3
    assert config.retry.hard_stop_local == "21:00"
    assert len(config.predictors) == 22  # 6 conventional + 12 llmp + 4 agent


def test_ladder_groups_have_expected_counts() -> None:
    """Each scope group expands to the configured number of rungs."""
    config = load_config()
    assert len(config.by_group("conventional")) == 6
    assert len(config.by_group("llmp")) == 12
    assert len(config.by_group("agent")) == 4


def test_predictor_ids_are_unique() -> None:
    """Every rung has a distinct, stable sp500_ predictor_id join key."""
    config = load_config()
    ids = [p.predictor_id for p in config.predictors]
    assert len(set(ids)) == len(ids)
    assert all(pid.startswith("sp500_") for pid in ids)


def test_leaderboard_cell_keys_are_unique() -> None:
    """No two rungs collide on (schema_method, model_label) — distinct cells."""
    config = load_config()
    cells = [(p.schema_method, p.model_label) for p in config.predictors]
    assert len(set(cells)) == len(cells)


def test_schema_methods_are_valid_enum_values() -> None:
    """Every rung maps to a data-contract method enum value."""
    valid = {"naive", "classical", "lightgbm", "llm_process", "analyst_agent", "code_agent"}
    config = load_config()
    assert {p.schema_method for p in config.predictors} <= valid


def test_covariate_llmp_variant_is_distinct_cell() -> None:
    """The _cov LLMP variant is a separate leaderboard cell from its base."""
    config = load_config()
    base = next(p for p in config.predictors if p.predictor_id == "sp500_llm_process__gemini-3.5-flash")
    cov = next(p for p in config.predictors if p.predictor_id == "sp500_llm_process_cov__gemini-3.5-flash")
    assert base.schema_method == cov.schema_method == "llm_process"
    assert base.model_label != cov.model_label
    assert cov.model_label.endswith("+cov")


def test_duplicate_predictor_id_raises() -> None:
    """A ladder that would collide on predictor_id is rejected."""
    raw = {"conventional": [{"method": "naive"}, {"method": "naive"}]}
    with pytest.raises(ValueError, match="Duplicate predictor_id"):
        expand_predictors(raw)
