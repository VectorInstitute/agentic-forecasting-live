"""Offline tests for the reflection step + gated tools routing."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from workshop_experiments.adaptive.gates import GateConfig, GatePolicy
from workshop_experiments.adaptive.gates.gated_tools import build_gated_skill_tools
from workshop_experiments.adaptive.reflection import (
    ReflectionInputs,
    build_reflection_config,
    build_reflection_prompt,
    run_reflection,
)
from workshop_experiments.adaptive.reseed import reseed_strategy
from workshop_experiments.adaptive.session import TurnResult, approx_tokens


D = dt.date(2026, 3, 2)


def _policy(tmp_path: Path) -> GatePolicy:
    """Policy."""
    trained = tmp_path / "learning"
    reseed_strategy(trained_dir=trained)
    return GatePolicy(
        strategy_dir=trained,
        log_dir=tmp_path / "log",
        twin_id="adaptive_learning",
        config=GateConfig(confirmation_threshold=1),
    )


class _FakeSession:
    def __init__(self) -> None:
        """Start with an empty prompt log."""
        self.prompts: list[str] = []

    def run_turn(self, prompt: str) -> TurnResult:
        """Record the prompt and return a deterministic accounted turn."""
        self.prompts.append(prompt)
        return TurnResult("ok", approx_tokens(prompt), approx_tokens("ok"), 0.0)

    def close(self) -> None:  # pragma: no cover
        """Close."""


def test_gated_tools_route_observations_and_emit_events(tmp_path: Path) -> None:
    """Gated tools route observations and emit events."""
    pol = _policy(tmp_path)
    tools = {t.__name__: t for t in build_gated_skill_tools(pol, origin_date=D)}
    msg = tools["record_observation"]("intervals too narrow in high VIX at 5bd")
    assert "Observation recorded" in msg
    assert len(pol.events.read_all()) == 1
    assert pol.store.load().observations  # write landed


def test_gated_narrative_enters_shadow_not_direct(tmp_path: Path) -> None:
    """Gated narrative enters shadow not direct."""
    pol = _policy(tmp_path)
    tools = {t.__name__: t for t in build_gated_skill_tools(pol, origin_date=D)}
    before = pol.store.load().approach_narrative
    msg = tools["update_approach_narrative"]("a full rewrite of the approach", "structural insight")
    assert "shadow gate" in msg
    # Not applied yet — the narrative is unchanged until the shadow window decides.
    assert pol.store.load().approach_narrative == before
    assert pol.shadow.has_open_candidate()


def test_build_reflection_config_attaches_gated_tools(tmp_path: Path) -> None:
    """Build reflection config attaches gated tools."""
    pol = _policy(tmp_path)
    config = build_reflection_config(pol, origin_date=D, model="gemini-3.5-flash")
    assert len(config.extra_tools) == 5


def test_run_reflection_bounds_turns_and_accounts(tmp_path: Path) -> None:
    """Run reflection bounds turns and accounts."""
    session = _FakeSession()
    inputs = ReflectionInputs(origin=D, committed_forecast="f", rationale="because", realized="r", crps="0.5")
    result = run_reflection(session, inputs, max_turns=2, transcript_dir=tmp_path)
    assert len(result.turns) == 2
    assert result.accounting.turns == 2
    assert (tmp_path / f"reflection_{D.isoformat()}.jsonl").exists()
    assert "cutoff_date = 2026-03-02" in session.prompts[0]


def test_reflection_prompt_mentions_gates() -> None:
    """Reflection prompt mentions gates."""
    prompt = build_reflection_prompt(
        ReflectionInputs(origin=D, committed_forecast="f", rationale="x", realized="r", crps="0.5")
    )
    assert "shadow" in prompt
    assert "survive the gates" in prompt
