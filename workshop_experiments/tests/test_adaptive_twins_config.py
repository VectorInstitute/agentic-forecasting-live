"""Offline tests for twin config expansion and read-only twin construction."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from workshop_experiments.adaptive.domain import (
    SEED_STRATEGY_DIR,
    build_sp500_adaptive_config,
)
from workshop_experiments.adaptive.twins import (
    FROZEN_STRATEGY_DIR,
    LEARNING_STRATEGY_DIR,
    build_twin_predictor,
    strategy_dir_for_twin,
)
from workshop_experiments.live.config import DEFAULT_CONFIG_PATH, load_config


def _config_with_twins(tmp_path: Path, *, enabled: bool, model: str = "gemini-3.5-flash") -> Path:
    """Write a live config with the twins block set to *enabled*."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["twins"] = {"enabled": enabled, "model": model}
    path = tmp_path / "live_config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_default_config_has_no_twins_but_carries_gate_params() -> None:
    """Default config has no twins but carries gate params."""
    config = load_config()
    # Twins are off by default (deploy later); the stateless ladder is unchanged.
    assert config.twins == ()
    assert len(config.predictors) == 22
    assert config.gate_params["confirmation_threshold"] == 3
    assert config.gate_params["shadow_window_days"] == 10


def test_twins_expand_to_two_rungs(tmp_path: Path) -> None:
    """Twins expand to two rungs."""
    config = load_config(_config_with_twins(tmp_path, enabled=True))
    assert len(config.twins) == 2
    frozen = config.twin("adaptive_frozen")
    learning = config.twin("adaptive_learning")
    assert frozen.schema_method == "adaptive_frozen"
    assert learning.schema_method == "adaptive_learning"
    assert frozen.predictor_id == "sp500_adaptive_frozen__gemini-3.5-flash"
    assert learning.predictor_id == "sp500_adaptive_learning__gemini-3.5-flash"
    assert frozen.group == learning.group == "twins"
    assert frozen.model_label == learning.model_label == "gemini-3.5-flash"
    # Twins stay OUT of the main ladder so the stateless-record counts hold.
    assert len(config.predictors) == 22


def test_disabled_twins_expand_to_nothing(tmp_path: Path) -> None:
    """Disabled twins expand to nothing."""
    config = load_config(_config_with_twins(tmp_path, enabled=False))
    assert config.twins == ()


def test_twin_schema_methods_are_valid_contract_enum(tmp_path: Path) -> None:
    """Twin schema methods are valid contract enum."""
    config = load_config(_config_with_twins(tmp_path, enabled=True))
    assert {t.schema_method for t in config.twins} == {"adaptive_frozen", "adaptive_learning"}


def test_strategy_dir_for_twin() -> None:
    """Strategy dir for twin."""
    assert strategy_dir_for_twin("adaptive_frozen") == FROZEN_STRATEGY_DIR
    assert strategy_dir_for_twin("adaptive_learning") == LEARNING_STRATEGY_DIR
    with pytest.raises(ValueError):
        strategy_dir_for_twin("nope")


def test_frozen_config_has_no_mutation_tools() -> None:
    """The frozen twin's config is built without the five mutation tools."""
    frozen = build_sp500_adaptive_config(strategy_dir=SEED_STRATEGY_DIR, attach_mutation_tools=False)
    learning = build_sp500_adaptive_config(strategy_dir=SEED_STRATEGY_DIR, attach_mutation_tools=True)
    assert tuple(frozen.extra_tools) == ()
    assert len(learning.extra_tools) == 5


def test_build_twin_predictor_rejects_non_twin(tmp_path: Path) -> None:
    """Build twin predictor rejects non twin."""
    config = load_config(_config_with_twins(tmp_path, enabled=True))
    ladder_rung = config.predictors[0]  # a conventional rung, twin_id is None
    with pytest.raises(ValueError):
        build_twin_predictor(ladder_rung)
