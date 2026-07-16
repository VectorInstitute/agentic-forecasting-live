"""Offline tests for TSX RESEED semantics (the tsx seed is never mutated)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from workshop_experiments.adaptive.domain_tsx import TSX_SEED_STRATEGY_DIR, TsxStrategyState
from workshop_experiments.adaptive.reseed import ReseedError, is_seeded, reseed_strategy


def _reseed_tsx(trained: Path, *, force: bool = False) -> Path:
    """Reseed a TSX trained dir from the committed tsx seed."""
    return reseed_strategy(
        seed_dir=TSX_SEED_STRATEGY_DIR,
        trained_dir=trained,
        state_type=TsxStrategyState,
        force=force,
    )


def test_reseed_copies_tsx_seed_and_renders_branded_skill(tmp_path: Path) -> None:
    """Reseed copies the tsx seed and renders a valid, branded SKILL.md."""
    trained = tmp_path / "tsx-strategy-trained"
    assert not is_seeded(trained)
    _reseed_tsx(trained)
    assert is_seeded(trained)
    assert (trained / "skill_state.yaml").exists()
    md = (trained / "SKILL.md").read_text()
    # Frontmatter name matches the destination dir (ADK requirement).
    assert "name: tsx-strategy-trained" in md
    # Branding follows through the copy — the TSX title, not the generic one.
    assert "# S&P/TSX Composite Forecasting Strategy" in md


def test_reseed_refuses_to_clobber_without_force(tmp_path: Path) -> None:
    """A populated tsx trained dir is protected unless force=True."""
    trained = tmp_path / "tsx-strategy-trained"
    _reseed_tsx(trained)
    state = yaml.safe_load((trained / "skill_state.yaml").read_text())
    state["observations"] = [{"date": "2026-03-02", "finding": "learned TSX thing", "linked_hypothesis": None}]
    (trained / "skill_state.yaml").write_text(yaml.safe_dump(state))
    with pytest.raises(ReseedError):
        _reseed_tsx(trained)
    _reseed_tsx(trained, force=True)
    assert yaml.safe_load((trained / "skill_state.yaml").read_text())["observations"] == []


def test_reseed_never_mutates_tsx_seed(tmp_path: Path) -> None:
    """The committed tsx seed state is untouched by a reseed."""
    before = (TSX_SEED_STRATEGY_DIR / "skill_state.yaml").read_text()
    _reseed_tsx(tmp_path / "tsx-strategy-trained", force=True)
    assert (TSX_SEED_STRATEGY_DIR / "skill_state.yaml").read_text() == before
