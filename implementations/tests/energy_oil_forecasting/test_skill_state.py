"""Snapshot test pinning the rendered WTI strategy SKILL.md.

Loads the committed ``wti-strategy-trained`` artifact, re-renders it through
:meth:`WtiStrategyState.build_markdown`, and asserts the output is byte-identical
to the committed ``SKILL.md``.  This guards the promotion of ``WtiStrategyState``
onto the generic shared-library ``StrategyState`` base: the rendered markdown
must stay render-identical so committed skill artifacts keep round-tripping.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from energy_oil_forecasting.adaptive_agent.skill_state import WtiStrategyState


_TRAINED_DIR = (
    Path(__file__).resolve().parents[2]
    / "energy_oil_forecasting"
    / "adaptive_agent"
    / "skills"
    / "wti-strategy-trained"
)


def test_trained_strategy_renders_identically() -> None:
    """The committed trained strategy re-renders byte-identically."""
    state = WtiStrategyState.model_validate(yaml.safe_load((_TRAINED_DIR / "skill_state.yaml").read_text()))

    rendered = state.build_markdown(skill_name=_TRAINED_DIR.name)
    committed = (_TRAINED_DIR / "SKILL.md").read_text(encoding="utf-8")

    assert rendered == committed


def test_default_skill_name_is_wti_strategy() -> None:
    """With no explicit name, the frontmatter and heading stay oil-branded."""
    state = WtiStrategyState.model_validate(yaml.safe_load((_TRAINED_DIR / "skill_state.yaml").read_text()))

    rendered = state.build_markdown()

    assert "name: wti-strategy" in rendered
    assert "# WTI Forecasting Strategy" in rendered
