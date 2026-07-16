"""Characterization tests for the strategy-mutation tools.

Drives the five mutation callables produced by ``build_skill_tools`` against a
temporary strategy directory, pinning the evidence-governance behaviour (the
confirmation threshold guard, hypothesis lifecycle, and narrative update) before
the tools are promoted to the generic shared library.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from aieng.forecasting.methods.agentic.adaptive_skill import AdaptiveSkillStore
from energy_oil_forecasting.adaptive_agent.skill_state import WtiStrategyState
from energy_oil_forecasting.adaptive_agent.skill_tools import build_skill_tools


@pytest.fixture
def strategy_dir(tmp_path: Path) -> Path:
    """Seed a minimal strategy directory and return its path."""
    store: AdaptiveSkillStore[WtiStrategyState] = AdaptiveSkillStore(
        skill_dir=tmp_path,
        state_type=WtiStrategyState,
    )
    store.save(WtiStrategyState(approach_narrative="Seed approach."))
    return tmp_path


def _tools(strategy_dir: Path, *, confirmation_threshold: int = 2):
    """Unpack the five mutation tools by name."""
    (
        record_observation,
        open_hypothesis,
        record_hypothesis_outcome,
        graduate_hypothesis,
        update_approach_narrative,
    ) = build_skill_tools(strategy_dir, confirmation_threshold=confirmation_threshold)
    return {
        "record_observation": record_observation,
        "open_hypothesis": open_hypothesis,
        "record_hypothesis_outcome": record_hypothesis_outcome,
        "graduate_hypothesis": graduate_hypothesis,
        "update_approach_narrative": update_approach_narrative,
    }


def _load(strategy_dir: Path) -> WtiStrategyState:
    return AdaptiveSkillStore(skill_dir=strategy_dir, state_type=WtiStrategyState).load()


def test_record_observation_appends(strategy_dir: Path) -> None:
    tools = _tools(strategy_dir)
    tools["record_observation"]("Intervals too narrow in elevated vol.")

    state = _load(strategy_dir)
    assert len(state.observations) == 1
    assert state.observations[0].finding == "Intervals too narrow in elevated vol."
    assert state.observations[0].linked_hypothesis is None


def test_open_hypothesis_assigns_id_and_links_evidence(strategy_dir: Path) -> None:
    tools = _tools(strategy_dir)
    msg = tools["open_hypothesis"]("Trend hurts at long horizons.", "Backtest showed 3x MAE.")

    assert "hyp-001" in msg
    state = _load(strategy_dir)
    assert len(state.hypotheses) == 1
    assert state.hypotheses[0].id == "hyp-001"
    assert state.hypotheses[0].status == "open"
    # Initial evidence recorded as a linked observation.
    assert any(o.linked_hypothesis == "hyp-001" for o in state.observations)


def test_record_outcome_validation(strategy_dir: Path) -> None:
    tools = _tools(strategy_dir)
    tools["open_hypothesis"]("claim", "evidence")

    assert "Invalid outcome" in tools["record_hypothesis_outcome"]("hyp-001", "maybe")
    assert "not found" in tools["record_hypothesis_outcome"]("hyp-999", "confirmed")

    tools["record_hypothesis_outcome"]("hyp-001", "confirmed")
    state = _load(strategy_dir)
    assert state.hypotheses[0].confirmations == 1


def test_graduate_rejected_below_threshold(strategy_dir: Path) -> None:
    tools = _tools(strategy_dir, confirmation_threshold=2)
    tools["open_hypothesis"]("claim", "evidence")
    tools["record_hypothesis_outcome"]("hyp-001", "confirmed")

    msg = tools["graduate_hypothesis"]("hyp-001", "elevated vol", "widen CI", "all")

    assert "Cannot graduate" in msg
    state = _load(strategy_dir)
    assert state.calibration_corrections == []
    assert state.hypotheses[0].status == "open"


def test_graduate_succeeds_at_threshold(strategy_dir: Path) -> None:
    tools = _tools(strategy_dir, confirmation_threshold=2)
    tools["open_hypothesis"]("claim", "evidence")
    tools["record_hypothesis_outcome"]("hyp-001", "confirmed")
    tools["record_hypothesis_outcome"]("hyp-001", "confirmed")

    msg = tools["graduate_hypothesis"]("hyp-001", "elevated vol", "widen CI 12%", "21bd")

    assert "graduated" in msg.lower()
    state = _load(strategy_dir)
    assert len(state.calibration_corrections) == 1
    correction = state.calibration_corrections[0]
    assert correction.condition == "elevated vol"
    assert correction.adjustment == "widen CI 12%"
    assert correction.horizon_scope == "21bd"
    assert correction.source_hypothesis == "hyp-001"
    assert state.hypotheses[0].status == "confirmed"
    # Graduation is logged to version history.
    assert any("hyp-001" in v.description for v in state.version_history)


def test_update_narrative_requires_rationale(strategy_dir: Path) -> None:
    tools = _tools(strategy_dir)

    assert "rationale must not be empty" in tools["update_approach_narrative"]("New approach.", "")
    assert "new_text must not be empty" in tools["update_approach_narrative"]("", "because")

    tools["update_approach_narrative"]("Weight news heavily at long horizons.", "Calibration record shifted.")
    state = _load(strategy_dir)
    assert state.approach_narrative == "Weight news heavily at long horizons."
    assert any("approach narrative" in v.description.lower() for v in state.version_history)
