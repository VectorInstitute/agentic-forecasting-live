"""WTI crude oil domain configuration.

Defines :data:`OIL_DOMAIN`, the single
:class:`~aieng.forecasting.methods.agentic.domain.DomainConfig` instance that
supplies every WTI-specific fragment the shared agent-building machinery needs.
The analyst, multitask, and adaptive agent factories render their instructions
from this instance, so all oil-specific prompt wording lives here rather than
scattered across the agent modules.
"""

from __future__ import annotations

from pathlib import Path

from aieng.forecasting.methods.agentic.domain import DomainConfig
from energy_oil_forecasting.data import WTI_SERIES_ID
from energy_oil_forecasting.paths import SHOCK_HORIZON, SHOCK_THRESHOLD


# ---------------------------------------------------------------------------
# Web-search sub-agent instruction (domain-specific verbatim)
# ---------------------------------------------------------------------------

_WTI_CONTEXT_RETRIEVAL_INSTRUCTION = """\
You are an oil market intelligence specialist with access to web search.

Search for information relevant to the query and return a concise structured \
markdown summary (3-5 paragraphs) covering relevant aspects of:
- WTI/Brent crude price level and recent trend
- OPEC+ production decisions and supply outlook
- Geopolitical risks in the Persian Gulf, Middle East, key shipping lanes
- US Strategic Petroleum Reserve and energy policy signals
- Notable tanker/shipping incidents or supply disruption signals
- Published analyst forecasts or unusual price-target revisions

Ground your summary in the search results you actually retrieve. \
When a cutoff date is specified, do not report or speculate about events \
that occurred after that date.

Before finalizing your summary, reason step by step: (1) for each candidate \
fact, judge its actual recency from the substance of the result itself, \
never from a source's claimed publish date or byline timestamp — those are \
frequently stale or updated after original publication; (2) discard \
anything you cannot confidently place before the cutoff date; (3) only then \
write your summary. Do not supplement the search results with your own \
background/training knowledge — if the results are insufficient, say so \
explicitly rather than filling gaps from memory.\
"""


# ---------------------------------------------------------------------------
# Skill directories (adaptive agent)
# ---------------------------------------------------------------------------

_ADAPTIVE_SKILLS_ROOT = Path(__file__).parent / "adaptive_agent" / "skills"


# ---------------------------------------------------------------------------
# The WTI domain
# ---------------------------------------------------------------------------

OIL_DOMAIN = DomainConfig(
    # Identity
    domain_name="WTI crude oil",
    analyst_persona="WTI crude oil market analyst",
    analyst_forecasting_focus=(
        "calibrated probabilistic price forecasts for WTI crude oil futures, grounded in "
        "supply/demand fundamentals, geopolitical risk, and historical price dynamics"
    ),
    analyst_agent_name_prefix="wti_analyst",
    adaptive_agent_name_prefix="wti_adaptive_analyst",
    target_short_name="WTI",
    starter_fluency_areas=(
        "supply/demand fundamentals, OPEC+ policy, geopolitical and shipping-lane risk, and price dynamics"
    ),
    # Data / target
    target_series_id=WTI_SERIES_ID,
    target_units="USD/bbl",
    target_history_description=("WTI daily close history (recent 6 months daily, older history as weekly averages)"),
    data_ticker="CL=F",
    data_source_name="Yahoo Finance",
    data_fetch_example=(
        "```python\nraw = ticker.history(start='2004-01-01', end='2026-02-16', auto_adjust=False)\n```"
    ),
    code_exec_preinstalled="numpy, pandas, sklearn, yfinance, statsmodels, properscoring",
    multitask_origin_price_field="origin_price_usd_bbl",
    # Context retrieval
    context_retrieval_instruction=_WTI_CONTEXT_RETRIEVAL_INSTRUCTION,
    recommended_search_queries=(
        "WTI crude oil price trend and OPEC+ supply decisions",
        "Persian Gulf geopolitical risk shipping lane disruptions",
        "US Strategic Petroleum Reserve policy and global demand outlook",
    ),
    key_assumptions_hint="OPEC+ policy, shipping lane risk, inventory levels, macro demand",
    # Strategy skill
    strategy_skill_title="WTI Forecasting Strategy",
    strategy_skill_name="wti-strategy",
    adaptive_calibration_example=(
        "substituting a flat-trend model in elevated/extreme vol regimes if your strategy calls for it"
    ),
    # Skill directories
    pipeline_skill_dirs=(
        _ADAPTIVE_SKILLS_ROOT / "fetch-yfinance",
        _ADAPTIVE_SKILLS_ROOT / "vol-regime",
        _ADAPTIVE_SKILLS_ROOT / "trend-projection",
    ),
    meta_learning_skill_dir=_ADAPTIVE_SKILLS_ROOT / "meta-learning",
    seed_strategy_dir=_ADAPTIVE_SKILLS_ROOT / "wti-strategy",
    trained_strategy_dir=_ADAPTIVE_SKILLS_ROOT / "wti-strategy-trained",
    # Volatility regime
    vol_regime_bands=((15.0, "low"), (30.0, "medium"), (50.0, "elevated")),
    # Tool bounds
    frequency="B",
    horizons=(5, 10, 21),
    shock_threshold=SHOCK_THRESHOLD,
    shock_horizon=SHOCK_HORIZON,
    num_samples=200,
)
