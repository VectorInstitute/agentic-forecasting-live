"""Characterization tests for the generic adaptive-skill persistence layer.

Exercises ``AdaptiveSkillStore`` against a minimal concrete
``AdaptiveSkillState`` subclass so the save / load / render / backup contract is
pinned before the strategy-state machinery is promoted and generalised.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from aieng.forecasting.methods.agentic.adaptive_skill import (
    AdaptiveSkillState,
    AdaptiveSkillStore,
)


class _DummyState(AdaptiveSkillState):
    """Minimal concrete state used only for testing the store."""

    approach: str
    notes: list[str] = []

    def build_markdown(self, skill_name: str | None = None) -> str:
        name = skill_name or "dummy-skill"
        lines = [
            "---",
            f"name: {name}",
            "---",
            "",
            "# Dummy",
            "",
            self.approach.strip(),
            "",
        ]
        lines += [f"- {n}" for n in self.notes]
        return "\n".join(lines)


def _make_store(tmp_path: Path, *, confirmation_threshold: int = 3) -> AdaptiveSkillStore[_DummyState]:
    return AdaptiveSkillStore(
        skill_dir=tmp_path,
        state_type=_DummyState,
        confirmation_threshold=confirmation_threshold,
    )


def test_rejects_missing_directory(tmp_path: Path) -> None:
    """The store must refuse a non-existent skill directory up front."""
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="does not exist"):
        AdaptiveSkillStore(skill_dir=missing, state_type=_DummyState)


def test_load_before_seed_raises(tmp_path: Path) -> None:
    """Loading before any state is seeded raises a helpful FileNotFoundError."""
    store = _make_store(tmp_path)
    with pytest.raises(FileNotFoundError, match="skill_state.yaml not found"):
        store.load()


def test_save_writes_yaml_and_markdown(tmp_path: Path) -> None:
    """``save`` writes both the YAML source of truth and rendered SKILL.md."""
    store = _make_store(tmp_path)
    state = _DummyState(approach="Trust the trend.", notes=["a", "b"])

    msg = store.save(state)

    assert store.state_path.exists()
    assert store.skill_md_path.exists()
    assert "SKILL.md re-rendered" in msg

    # SKILL.md is exactly what build_markdown renders with the dir name.
    assert store.skill_md_path.read_text(encoding="utf-8") == state.build_markdown(skill_name=tmp_path.name)


def test_round_trips_state(tmp_path: Path) -> None:
    """State survives a save/load cycle byte-for-byte in its fields."""
    store = _make_store(tmp_path)
    state = _DummyState(approach="Weight news at long horizons.", notes=["x"])
    store.save(state)

    loaded = store.load()

    assert loaded.approach == state.approach
    assert loaded.notes == state.notes
    assert loaded.schema_version == state.schema_version


def test_second_save_backs_up_previous_state(tmp_path: Path) -> None:
    """The prior YAML is copied into ``.history`` before being overwritten."""
    store = _make_store(tmp_path)
    store.save(_DummyState(approach="first", notes=[]))

    # No backup after the very first save (nothing to back up beforehand).
    assert not store.history_dir.exists()

    store.save(_DummyState(approach="second", notes=[]))

    backups = list(store.history_dir.glob("skill_state_*.yaml"))
    assert len(backups) == 1
    backed_up = yaml.safe_load(backups[0].read_text(encoding="utf-8"))
    assert backed_up["approach"] == "first"

    # The live file reflects the latest save.
    assert store.load().approach == "second"


def test_confirmation_threshold_is_a_store_parameter(tmp_path: Path) -> None:
    """The evidence bar lives on the store, not in serialised state."""
    store = _make_store(tmp_path, confirmation_threshold=5)
    store.save(_DummyState(approach="a", notes=[]))

    assert store.confirmation_threshold == 5
    # It must not leak into the persisted state.
    raw = yaml.safe_load(store.state_path.read_text(encoding="utf-8"))
    assert "confirmation_threshold" not in raw
