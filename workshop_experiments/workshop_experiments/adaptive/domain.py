"""Adaptive S&P 500 domain wiring (workshop stage 2c).

Extends the stateless :data:`workshop_experiments.domain.SP500_DOMAIN` with the
adaptive skill-directory fields PR 1 left unset — the pipeline skills
(``fetch-yfinance``, ``vol-regime``, ``trend-projection``), the governance skill
(``meta-learning``), and the seed / trained strategy directories — producing
:data:`SP500_ADAPTIVE_DOMAIN`. A thin :class:`Sp500StrategyState` subclass pins
the rendered ``SKILL.md`` branding so the committed seed artifacts round-trip
byte-identically, exactly as ``WtiStrategyState`` does for oil.

The equity-generalized pipeline skill *content* lives under :mod:`.skills` — new
files under the workshop package. The oil skill dirs are never touched.

Nothing here calls a model API. The config/predictor factories delegate to the
shared, domain-agnostic
:func:`aieng.forecasting.methods.agentic.domain.build_adaptive_config`.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from aieng.forecasting.methods.agentic import (
    AgentPredictor,
    ContinuousAgentForecastOutput,
)
from aieng.forecasting.methods.agentic.agent_factory import AgentConfig
from aieng.forecasting.methods.agentic.domain import build_adaptive_config
from aieng.forecasting.methods.agentic.strategy_state import StrategyState
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL

from workshop_experiments.domain import SP500_DOMAIN, Sp500ReturnForecastPromptBuilder


# ---------------------------------------------------------------------------
# Skill directories (committed under this package)
# ---------------------------------------------------------------------------

#: Root of the seed skill directories shipped with the workshop package.
SKILLS_ROOT = Path(__file__).resolve().parent / "skills"

#: Ordered pipeline skills the adaptive prompt names (fetch → vol → trend).
PIPELINE_SKILL_DIRS: tuple[Path, ...] = (
    SKILLS_ROOT / "fetch-yfinance",
    SKILLS_ROOT / "vol-regime",
    SKILLS_ROOT / "trend-projection",
)

#: Governance skill loaded after the strategy skill.
META_LEARNING_SKILL_DIR = SKILLS_ROOT / "meta-learning"

#: The read-only seed strategy — never mutated by a study run (RESEED copies it).
SEED_STRATEGY_DIR = SKILLS_ROOT / "sp500-strategy"

#: The mutable trained strategy variant a study run writes into (created by
#: RESEED as a copy of the seed; see :mod:`workshop_experiments.adaptive.reseed`).
TRAINED_STRATEGY_DIR = SKILLS_ROOT / "sp500-strategy-trained"


# ---------------------------------------------------------------------------
# Branded strategy state
# ---------------------------------------------------------------------------


class Sp500StrategyState(StrategyState):
    """Equity-branded strategy state for the adaptive S&P 500 analyst.

    Identical in structure to :class:`StrategyState`; only the presentation
    strings are pinned so the ``sp500-strategy`` artifacts render exactly as
    committed.
    """

    markdown_title: ClassVar[str] = "S&P 500 Forecasting Strategy"
    default_skill_name: ClassVar[str] = "sp500-strategy"
    frontmatter_description_lines: ClassVar[tuple[str, ...]] = (
        "The adaptive S&P 500 analyst's current forecasting strategy. Load this at",
        "the start of every prediction task. This file is generated — edit the state",
        "through the mutation tools, not by hand.",
    )


# ---------------------------------------------------------------------------
# The adaptive domain — SP500_DOMAIN + skill-directory fields
# ---------------------------------------------------------------------------

#: The stateless :data:`SP500_DOMAIN` extended with every adaptive skill-dir
#: field the shared adaptive builder needs. Built by ``model_copy`` so the
#: identity strings, target series, vol-regime bands, and tool bounds stay
#: exactly as PR 1 fixed them.
SP500_ADAPTIVE_DOMAIN = SP500_DOMAIN.model_copy(
    update={
        "pipeline_skill_dirs": PIPELINE_SKILL_DIRS,
        "meta_learning_skill_dir": META_LEARNING_SKILL_DIR,
        "seed_strategy_dir": SEED_STRATEGY_DIR,
        "trained_strategy_dir": TRAINED_STRATEGY_DIR,
    }
)


# ---------------------------------------------------------------------------
# Config / predictor factories
# ---------------------------------------------------------------------------


def build_sp500_adaptive_config(
    *,
    model: str = ADVANCED_MODEL,
    search_model: str = LITE_MODEL,
    max_output_tokens: int = 16_384,
    strategy_dir: Path | None = None,
    confirmation_threshold: int = 3,
    attach_mutation_tools: bool = True,
) -> AgentConfig:
    """Build the adaptive S&P 500 analyst :class:`AgentConfig`.

    Delegates to the shared :func:`build_adaptive_config` over
    :data:`SP500_ADAPTIVE_DOMAIN` and :class:`Sp500StrategyState`.

    Parameters
    ----------
    model, search_model : str
        Top-level and web-search models.
    max_output_tokens : int
        Per-response token budget (generous, for a full ``run_code`` script).
    strategy_dir : Path or None
        Strategy skill directory. ``None`` uses the seed dir. Pass the trained
        dir to run the learned strategy; the same dir backs both the ADK skill
        load and the mutation-tool bindings.
    confirmation_threshold : int, default=3
        Hypothesis confirmations required before ``graduate_hypothesis`` fires
        (the tier-2 gate ``k``).
    attach_mutation_tools : bool, default=True
        When ``False``, the config is built **without** the five strategy
        mutation tools — the read-only *frozen twin* configuration. The
        strategy skill is still loaded (read), but nothing can write it.
    """
    config = build_adaptive_config(
        SP500_ADAPTIVE_DOMAIN,
        state_type=Sp500StrategyState,
        model=model,
        search_model=search_model,
        max_output_tokens=max_output_tokens,
        strategy_dir=strategy_dir,
        confirmation_threshold=confirmation_threshold,
    )
    if attach_mutation_tools:
        return config
    # Frozen twin: strip the mutation tools so the strategy is genuinely
    # read-only. The skill files are still loaded into context; only the write
    # path is removed.
    return config.model_copy(update={"extra_tools": ()})


def build_sp500_adaptive_predictor(
    *,
    config: AgentConfig | None = None,
    strategy_dir: Path | None = None,
    model: str = ADVANCED_MODEL,
    attach_mutation_tools: bool = True,
) -> AgentPredictor:
    """Wrap the adaptive S&P 500 agent in an :class:`AgentPredictor`.

    Uses :class:`~workshop_experiments.domain.Sp500ReturnForecastPromptBuilder`
    (the same returns-aware payload the stateless rungs use) and the continuous
    output schema.
    """
    if config is None:
        config = build_sp500_adaptive_config(
            model=model,
            strategy_dir=strategy_dir,
            attach_mutation_tools=attach_mutation_tools,
        )
    return AgentPredictor(
        agent_config=config,
        prompt_builder=Sp500ReturnForecastPromptBuilder(),
        output_schema=ContinuousAgentForecastOutput,
    )


__all__ = [
    "META_LEARNING_SKILL_DIR",
    "PIPELINE_SKILL_DIRS",
    "SEED_STRATEGY_DIR",
    "SKILLS_ROOT",
    "SP500_ADAPTIVE_DOMAIN",
    "TRAINED_STRATEGY_DIR",
    "Sp500StrategyState",
    "build_sp500_adaptive_config",
    "build_sp500_adaptive_predictor",
]
