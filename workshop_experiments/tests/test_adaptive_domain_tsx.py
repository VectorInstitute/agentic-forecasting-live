"""Offline tests for the TSX adaptive domain wiring (no model calls).

Covers: the adaptive instruction renders with the TSX persona and no S&P-500 /
oil-persona bleed; the branded strategy title; the seed skill loads and its
SKILL.md renders; and the config factories wire the mutation tools (or strip them
for the frozen eval arm) with tsx-distinct / arm-distinct agent names.
"""

from __future__ import annotations

from aieng.forecasting.methods.agentic.adaptive_skill import AdaptiveSkillStore
from aieng.forecasting.methods.agentic.domain import render_adaptive_analyst_instruction
from workshop_experiments.adaptive.domain_tsx import (
    TSX_ADAPTIVE_DOMAIN,
    TSX_META_LEARNING_SKILL_DIR,
    TSX_PIPELINE_SKILL_DIRS,
    TSX_SEED_STRATEGY_DIR,
    TSX_TRAINED_STRATEGY_DIR,
    TsxStrategyState,
    build_tsx_adaptive_config,
)


# Forbidden persona/index identities (commodity nouns like "oil"/"gold" as market
# factors are correct for the TSX and must NOT be flagged — only foreign personas
# and the S&P-500 index identity).
_FORBIDDEN = (
    "s&p 500",
    "sp500",
    "^gspc",
    "wti crude oil market analyst",
    "crude oil market analyst",
    "oil market analyst",
)


def _assert_no_bleed(text: str) -> None:
    """Assert no S&P-500 / oil-persona identity leaked into *text*."""
    lowered = text.lower()
    for term in _FORBIDDEN:
        assert term not in lowered, f"forbidden term {term!r} leaked into the rendered instruction"


def test_adaptive_domain_carries_tsx_skill_dirs() -> None:
    """The adaptive domain is TSX_DOMAIN + the four skill-dir fields, tsx-branded."""
    assert TSX_ADAPTIVE_DOMAIN.domain_name == "S&P/TSX Composite index"
    assert TSX_ADAPTIVE_DOMAIN.strategy_skill_name == "tsx-strategy"
    assert TSX_ADAPTIVE_DOMAIN.strategy_skill_title == "S&P/TSX Composite Forecasting Strategy"
    assert TSX_ADAPTIVE_DOMAIN.pipeline_skill_dirs == TSX_PIPELINE_SKILL_DIRS
    assert TSX_ADAPTIVE_DOMAIN.meta_learning_skill_dir == TSX_META_LEARNING_SKILL_DIR
    assert TSX_ADAPTIVE_DOMAIN.seed_strategy_dir == TSX_SEED_STRATEGY_DIR
    assert TSX_ADAPTIVE_DOMAIN.trained_strategy_dir == TSX_TRAINED_STRATEGY_DIR
    # TSX vol bands sit below the SPX/VIX levels (calmer index).
    assert TSX_ADAPTIVE_DOMAIN.vol_regime_bands == ((10.0, "low"), (15.0, "normal"), (22.0, "elevated"), (30.0, "high"))


def test_adaptive_instruction_renders_tsx_persona_no_bleed() -> None:
    """The adaptive analyst instruction renders TSX-framed with no persona bleed."""
    text = render_adaptive_analyst_instruction(TSX_ADAPTIVE_DOMAIN)
    assert "S&P/TSX Composite" in text
    assert "tsx-strategy" in text
    # Pipeline + governance skill names are referenced.
    for name in ("fetch-yfinance", "vol-regime", "trend-projection", "meta-learning"):
        assert name in text
    _assert_no_bleed(text)


def test_seed_strategy_state_is_branded() -> None:
    """TsxStrategyState pins the TSX title / skill name / description."""
    assert TsxStrategyState.markdown_title == "S&P/TSX Composite Forecasting Strategy"
    assert TsxStrategyState.default_skill_name == "tsx-strategy"


def test_seed_skill_loads_and_renders() -> None:
    """The committed seed skill loads and its SKILL.md round-trips the branding."""
    store = AdaptiveSkillStore(skill_dir=TSX_SEED_STRATEGY_DIR, state_type=TsxStrategyState)
    state = store.load()
    md = state.build_markdown(skill_name="tsx-strategy")
    assert "name: tsx-strategy" in md
    assert "# S&P/TSX Composite Forecasting Strategy" in md
    # The committed SKILL.md matches a fresh render of the state (byte-identical).
    committed = (TSX_SEED_STRATEGY_DIR / "SKILL.md").read_text()
    assert committed.strip() == md.strip()
    # Seed carries domain priors only — nothing learned yet.
    assert state.calibration_corrections == []
    assert state.hypotheses == []
    assert state.observations == []


def test_config_attaches_and_strips_mutation_tools() -> None:
    """Seed config has the five mutation tools; the frozen eval arm has none."""
    learning = build_tsx_adaptive_config(strategy_dir=TSX_SEED_STRATEGY_DIR, attach_mutation_tools=True)
    frozen = build_tsx_adaptive_config(strategy_dir=TSX_SEED_STRATEGY_DIR, attach_mutation_tools=False)
    assert len(learning.extra_tools) == 5
    assert tuple(frozen.extra_tools) == ()
    # The agent name is tsx-distinct so caches never collide with the sp500 agents.
    assert learning.name == "tsx_adaptive_analyst_tsx_strategy"
