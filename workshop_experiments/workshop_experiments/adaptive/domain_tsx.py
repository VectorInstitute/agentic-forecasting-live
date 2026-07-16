"""Adaptive S&P/TSX Composite domain wiring (bootcamp pre/post demonstration).

Extends the stateless :data:`workshop_experiments.domain_tsx.TSX_DOMAIN` with the
adaptive skill-directory fields — the pipeline skills (``fetch-yfinance``,
``vol-regime``, ``trend-projection``), the governance skill (``meta-learning``),
and the seed / trained strategy directories — producing
:data:`TSX_ADAPTIVE_DOMAIN`. A thin :class:`TsxStrategyState` subclass pins the
rendered ``SKILL.md`` branding so the committed seed artifacts round-trip
byte-identically, exactly as the S&P 500's ``Sp500StrategyState`` does in
:mod:`workshop_experiments.adaptive.domain`.

The TSX pipeline skill *content* lives under :mod:`.skills_tsx` — the equity
pipeline generalized to the S&P/TSX Composite (``^GSPTSE``), with TSX volatility
bands from :data:`TSX_DOMAIN` and no S&P-500 persona or ticker bleed. The S&P 500
skill dirs under :mod:`.skills` are never touched.

Unlike the twins / continual-learning wiring in :mod:`.domain`, this module
serves the single **pre/post** bootcamp shape: one self-directed study session
distils a strategy (``tsx-strategy`` → ``tsx-strategy-trained``), then a frozen
before/after eval scores the untrained seed against the trained strategy.

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

from workshop_experiments.domain_tsx import TSX_DOMAIN, TsxReturnForecastPromptBuilder


# ---------------------------------------------------------------------------
# Skill directories (committed under this package, separate from the sp500 set)
# ---------------------------------------------------------------------------

#: Root of the TSX seed skill directories shipped with the workshop package.
SKILLS_TSX_ROOT = Path(__file__).resolve().parent / "skills_tsx"

#: Ordered pipeline skills the adaptive prompt names (fetch → vol → trend).
TSX_PIPELINE_SKILL_DIRS: tuple[Path, ...] = (
    SKILLS_TSX_ROOT / "fetch-yfinance",
    SKILLS_TSX_ROOT / "vol-regime",
    SKILLS_TSX_ROOT / "trend-projection",
)

#: Governance skill loaded after the strategy skill.
TSX_META_LEARNING_SKILL_DIR = SKILLS_TSX_ROOT / "meta-learning"

#: The read-only seed strategy — never mutated by a study run (RESEED copies it).
TSX_SEED_STRATEGY_DIR = SKILLS_TSX_ROOT / "tsx-strategy"

#: The mutable trained strategy variant a study run writes into (created by
#: RESEED as a copy of the seed; see :mod:`workshop_experiments.adaptive.reseed`).
TSX_TRAINED_STRATEGY_DIR = SKILLS_TSX_ROOT / "tsx-strategy-trained"


# ---------------------------------------------------------------------------
# Branded strategy state
# ---------------------------------------------------------------------------


class TsxStrategyState(StrategyState):
    """Canadian-equity-index-branded strategy state for the adaptive TSX analyst.

    Identical in structure to :class:`StrategyState`; only the presentation
    strings are pinned so the ``tsx-strategy`` artifacts render exactly as
    committed. The title matches :data:`TSX_DOMAIN.strategy_skill_title`.
    """

    markdown_title: ClassVar[str] = "S&P/TSX Composite Forecasting Strategy"
    default_skill_name: ClassVar[str] = "tsx-strategy"
    frontmatter_description_lines: ClassVar[tuple[str, ...]] = (
        "The adaptive S&P/TSX Composite analyst's current forecasting strategy. Load",
        "this at the start of every prediction task. This file is generated — edit the",
        "state through the mutation tools, not by hand.",
    )


# ---------------------------------------------------------------------------
# The adaptive domain — TSX_DOMAIN + skill-directory fields
# ---------------------------------------------------------------------------

#: The stateless :data:`TSX_DOMAIN` extended with every adaptive skill-dir field
#: the shared adaptive builder needs. Built by ``model_copy`` so the identity
#: strings, target series, vol-regime bands, and tool bounds stay exactly as the
#: stateless TSX domain fixed them (``strategy_skill_title`` / ``strategy_skill_name``
#: are already set on ``TSX_DOMAIN``, so they are not overridden here).
TSX_ADAPTIVE_DOMAIN = TSX_DOMAIN.model_copy(
    update={
        "pipeline_skill_dirs": TSX_PIPELINE_SKILL_DIRS,
        "meta_learning_skill_dir": TSX_META_LEARNING_SKILL_DIR,
        "seed_strategy_dir": TSX_SEED_STRATEGY_DIR,
        "trained_strategy_dir": TSX_TRAINED_STRATEGY_DIR,
    }
)


# ---------------------------------------------------------------------------
# Single-session bootcamp study prompts (pre-2026 history + 2025 review)
# ---------------------------------------------------------------------------

#: Opening turn for the TSX bootcamp Study Hall. Unlike the S&P 500 two-phase
#: shape (Study Hall over pre-2025, Residency over 2025 postmortems), the TSX
#: demonstration folds the whole study into ONE self-directed session: explore
#: the full pre-2026 ^GSPTSE history AND review the 2025 period in the same run.
TSX_STUDY_HALL_PROMPT = """\
You are in a self-directed Study Hall for the S&P/TSX Composite (^GSPTSE). You
have code execution over the full pre-2026 TSX history and your pipeline +
strategy + meta-learning skills. Nothing here is scored — the point is skill
formation before the protected 2026 evaluation. Work the loop: explore ->
hypothesise -> test with code -> distill into your `tsx-strategy` through the
mutation tools, governed by your meta-learning skill.

This single session covers the whole demonstration study. These are SUGGESTED
directions, not a checklist — follow what the data makes interesting, in whatever
order:
- Stylized facts of the TSX: fat tails, volatility clustering, the leverage
  effect, autocorrelation of returns vs |returns|, drawdown/recovery profiles,
  seasonality. The TSX is energy- and materials-heavy, so test how oil, gold, and
  base-metal moves co-move with index returns.
- Conditional playbooks: empirical forward-return distributions conditional on
  regime (realised-vol bands, commodity-shock state, BoC policy phase, trend
  state) — "when realised vol > 22%, the 5-day forward distribution looks like X".
- Event studies: 2008–09, the 2014–16 oil crash (the TSX's defining shock, not
  the S&P's), COVID 2020, the 2022 hiking cycle, and any 2025 commodity or
  policy-driven breaks — shock-decay and vol-spike profiles around known breaks.
- Indicator validation: build realised-vol estimators, momentum/mean-reversion by
  horizon, moving-average states — keep what has distributional forecast value for
  the TSX, discard folklore carried over from US indices.
- Self-calibration: backtest your own quantile bands on TSX history and learn
  coverage corrections (e.g. "my 90% bands ran at 78% in the elevated-vol regime").

Then review the 2025 period specifically: walk the 2025 origins, compare how your
seed strategy would have forecast them against what realised, and diagnose any
systematic miss — was it knowable from the commodity/policy narrative, or genuinely
not forecastable? Fold durable 2025 lessons into your strategy the same way.

Always end a substantive finding by recording it through the appropriate mutation
tool if — and only if — it clears your meta-learning evidence bar. Begin now with
whatever you find most promising.
"""

#: Mid-session continuation turn (deeper on an open thread or a new direction).
TSX_CONTINUE_PROMPT = """\
Continue your self-directed TSX study. Build on what you have found so far — go
deeper on a promising thread, test a hypothesis you opened, review another stretch
of the 2025 period, or turn to a direction you have not yet explored. Use code
execution; record durable findings through the mutation tools per your
meta-learning skill.
"""

#: Checkpoint distillation turn (forces consolidation into the strategy).
TSX_DISTILL_PROMPT = """\
Checkpoint: distill what you have learned so far into your `tsx-strategy` NOW.
Review your observations and open hypotheses; where the evidence clears the bar,
record observations, open or update hypotheses, and — only where the confirmation
threshold is met — graduate calibration corrections. If your approach narrative no
longer captures how you actually forecast the TSX, update it with a rationale. Be
conservative: consolidate what the data supports, do not invent corrections.
"""


# ---------------------------------------------------------------------------
# Config / predictor factories
# ---------------------------------------------------------------------------


def build_tsx_adaptive_config(
    *,
    model: str = ADVANCED_MODEL,
    search_model: str = LITE_MODEL,
    max_output_tokens: int = 16_384,
    strategy_dir: Path | None = None,
    confirmation_threshold: int = 3,
    attach_mutation_tools: bool = True,
) -> AgentConfig:
    """Build the adaptive S&P/TSX Composite analyst :class:`AgentConfig`.

    Delegates to the shared :func:`build_adaptive_config` over
    :data:`TSX_ADAPTIVE_DOMAIN` and :class:`TsxStrategyState`.

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
        mutation tools — the read-only *frozen* eval configuration. The strategy
        skill is still loaded (read), but nothing can write it.
    """
    config = build_adaptive_config(
        TSX_ADAPTIVE_DOMAIN,
        state_type=TsxStrategyState,
        model=model,
        search_model=search_model,
        max_output_tokens=max_output_tokens,
        strategy_dir=strategy_dir,
        confirmation_threshold=confirmation_threshold,
    )
    if attach_mutation_tools:
        return config
    # Frozen eval arm: strip the mutation tools so the strategy is genuinely
    # read-only. The skill files are still loaded into context; only the write
    # path is removed.
    return config.model_copy(update={"extra_tools": ()})


def build_tsx_adaptive_predictor(
    *,
    config: AgentConfig | None = None,
    strategy_dir: Path | None = None,
    model: str = ADVANCED_MODEL,
    attach_mutation_tools: bool = True,
) -> AgentPredictor:
    """Wrap the adaptive S&P/TSX Composite agent in an :class:`AgentPredictor`.

    Uses :class:`~workshop_experiments.domain_tsx.TsxReturnForecastPromptBuilder`
    (the same returns-aware payload the stateless rungs use) and the continuous
    output schema. The resulting ``predictor_id`` folds the agent name
    (``tsx_adaptive_analyst_*``, distinct per strategy dir) and model, so the two
    eval arms and the S&P 500 agents never share a prediction cache.
    """
    if config is None:
        config = build_tsx_adaptive_config(
            model=model,
            strategy_dir=strategy_dir,
            attach_mutation_tools=attach_mutation_tools,
        )
    return AgentPredictor(
        agent_config=config,
        prompt_builder=TsxReturnForecastPromptBuilder(),
        output_schema=ContinuousAgentForecastOutput,
    )


__all__ = [
    "SKILLS_TSX_ROOT",
    "TSX_ADAPTIVE_DOMAIN",
    "TSX_CONTINUE_PROMPT",
    "TSX_DISTILL_PROMPT",
    "TSX_META_LEARNING_SKILL_DIR",
    "TSX_PIPELINE_SKILL_DIRS",
    "TSX_SEED_STRATEGY_DIR",
    "TSX_STUDY_HALL_PROMPT",
    "TSX_TRAINED_STRATEGY_DIR",
    "TsxStrategyState",
    "build_tsx_adaptive_config",
    "build_tsx_adaptive_predictor",
]
