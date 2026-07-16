"""Offline tests for RESEED semantics (seed is never mutated)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from workshop_experiments.adaptive.domain import SEED_STRATEGY_DIR
from workshop_experiments.adaptive.reseed import ReseedError, is_seeded, reseed_strategy


def test_reseed_copies_seed_and_renders_skill(tmp_path: Path) -> None:
    """Reseed copies seed and renders skill."""
    trained = tmp_path / "trained"
    assert not is_seeded(trained)
    reseed_strategy(trained_dir=trained)
    assert is_seeded(trained)
    assert (trained / "skill_state.yaml").exists()
    assert (trained / "SKILL.md").exists()
    # Frontmatter name matches the destination dir (ADK requirement).
    assert "name: trained" in (trained / "SKILL.md").read_text()


def test_reseed_refuses_to_clobber_without_force(tmp_path: Path) -> None:
    """Reseed refuses to clobber without force."""
    trained = tmp_path / "trained"
    reseed_strategy(trained_dir=trained)
    # Mutate the trained state, then a plain reseed must refuse.
    state = yaml.safe_load((trained / "skill_state.yaml").read_text())
    state["observations"] = [{"date": "2026-03-02", "finding": "learned thing", "linked_hypothesis": None}]
    (trained / "skill_state.yaml").write_text(yaml.safe_dump(state))
    with pytest.raises(ReseedError):
        reseed_strategy(trained_dir=trained)
    # force=True resets it to the clean seed.
    reseed_strategy(trained_dir=trained, force=True)
    assert yaml.safe_load((trained / "skill_state.yaml").read_text())["observations"] == []


def test_reseed_never_mutates_seed(tmp_path: Path) -> None:
    """Reseed never mutates seed."""
    before = (SEED_STRATEGY_DIR / "skill_state.yaml").read_text()
    trained = tmp_path / "trained"
    reseed_strategy(trained_dir=trained, force=True)
    # Write into the trained dir via a subsequent reseed; the seed is untouched.
    assert (SEED_STRATEGY_DIR / "skill_state.yaml").read_text() == before


def test_reseed_onto_self_is_refused(tmp_path: Path) -> None:
    """Reseed onto self is refused."""
    trained = tmp_path / "trained"
    reseed_strategy(trained_dir=trained)
    with pytest.raises(ReseedError):
        reseed_strategy(seed_dir=trained, trained_dir=trained, force=True)
