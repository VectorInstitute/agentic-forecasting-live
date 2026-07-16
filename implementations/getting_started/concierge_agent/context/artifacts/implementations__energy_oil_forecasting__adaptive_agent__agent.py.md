# Source: implementations/energy_oil_forecasting/adaptive_agent/agent.py

kind: python

```python
"""Adaptive WTI crude oil analyst agent.

Unlike :mod:`energy_oil_forecasting.analyst_agent`, this agent is designed as
a persistent entity: it maintains a living forecasting strategy through mutable
skill files on the filesystem and handles multiple message types through a
single chat interface.

Provides:

- :func:`build_wti_adaptive_config`: full adaptive agent — E2B code execution,
  bounded web search, and five pipeline-component skills.
- :class:`WtiAdaptiveForecastPromptBuilder`: prompt builder for prediction-request
  messages, compatible with the existing backtest/eval harness.
- :func:`build_wti_adaptive_predictor`: convenience factory wiring the adaptive
  agent into an :class:`~aieng.forecasting.methods.agentic.predictor.AgentPredictor`
  for comparison against stateless baselines in backtests.

Skills
------
Skills are **pipeline components**, not end-to-end recipes. The agent composes
them as needed, loading multiple skills before writing a single complete code block.

``fetch-yfinance``
    One-shot patterns for downloading market data from Yahoo Finance.

``vol-regime``
    Volatility regime classification and anomaly detection.

``trend-projection``
    Linear trend fitting, projection, and interval calibration.

``wti-strategy``
    The agent's current forecasting strategy (mutable).

``meta-learning``
    Governs when and how ``wti-strategy`` is updated.

Code execution
--------------
Uses E2B (real sandbox). Each ``run_code`` call is a **fresh Python process** —
no state, variables, or files carry over between calls. All imports, data
fetching, and analysis must be in a single self-contained block.

Skill mutability
----------------
The ``wti-strategy`` skill is backed by a :class:`~energy_oil_forecasting.adaptive_agent.skill_state.WtiStrategyState`
Pydantic model persisted in ``skills/wti-strategy/skill_state.yaml``.
``SKILL.md`` is rendered from that model on every mutation and is never
hand-edited.  Five typed mutation tools (from :mod:`skill_tools`) are
registered via ``AgentConfig(extra_tools=build_skill_tools(strategy_dir))`` and
run in the host process — not inside E2B.  See :mod:`skill_tools` for the full
tool signatures and evidence governance rules.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import (
    AgentPredictor,
    ContinuousAgentForecastOutput,
    build_adk_agent,
)
from aieng.forecasting.methods.agentic.agent_factory import AgentConfig
from aieng.forecasting.methods.agentic.domain import build_adaptive_config
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from energy_oil_forecasting.adaptive_agent.skill_state import WtiStrategyState
from energy_oil_forecasting.analyst_agent import compress_history
from energy_oil_forecasting.domain import OIL_DOMAIN
from pydantic import BaseModel


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
#
# The adaptive analyst instruction and the web-search sub-agent instruction are
# rendered from the shared templates over the WTI ``OIL_DOMAIN`` fragments (see
# :mod:`energy_oil_forecasting.domain` and
# :func:`aieng.forecasting.methods.agentic.domain.build_adaptive_config`).  A new
# target series is configured by supplying a different ``DomainConfig``.


# ---------------------------------------------------------------------------
# Prompt builder — prediction requests
# ---------------------------------------------------------------------------


class WtiAdaptiveForecastPromptBuilder(BaseModel):
    """Prompt builder for prediction-request messages to the adaptive agent.

    Produces a structured JSON payload containing the compressed price history
    and key summary statistics.  The agent runs its own full statistical
    pipeline (fetch-yfinance → vol-regime → trend-projection) inside the E2B
    sandbox, applies calibration corrections from its ``wti-strategy`` skill,
    and incorporates news context from web search before returning its forecast.

    For resolution, self-review, and user-question invocations, construct
    plain-text messages directly and send them via the ADK runner.
    """

    model_config = {"extra": "forbid"}

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        df = context.get_series(task.target_series_id)
        compressed = compress_history(df)

        last_row = df.iloc[-1]
        last_close = float(last_row["value"])
        last_date = str(pd.Timestamp(last_row["timestamp"]).date())
        trailing_252 = df["value"].tail(252)

        payload: dict[str, Any] = {
            "task": task.task_id,
            "as_of": str(context.as_of)[:10],
            "horizons": list(task.horizons),
            "standard_quantiles": list(STANDARD_QUANTILES),
            "target_summary": {
                "last_close_usd_bbl": last_close,
                "last_date": last_date,
                "n_trading_days": int(len(df)),
                "52w_high": float(trailing_252.max()),
                "52w_low": float(trailing_252.min()),
            },
            "target_history_csv": compressed,
        }

        return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# AgentConfig factory
# ---------------------------------------------------------------------------


def build_wti_adaptive_config(
    model: str = ADVANCED_MODEL,
    search_model: str = LITE_MODEL,
    max_output_tokens: int = 16_384,
    strategy_dir: Path | None = None,
) -> AgentConfig:
    """Build the full adaptive WTI analyst :class:`AgentConfig`.

    Combines E2B code execution, bounded Google Search with temporal cutoff
    enforcement, and five skills: ``fetch-yfinance``, ``vol-regime``,
    ``trend-projection``, the selected strategy skill, and ``meta-learning``.

    Parameters
    ----------
    model : str
        Model for the top-level analyst agent.
    search_model : str
        Model for the context-retrieval (web-search) sub-tool. Defaults to the
        lite model (``gemini-3.1-flash-lite-preview``) independently of ``model`` (the
        advanced model) so web search stays cheap while the analyst reasons
        with more capability.
    max_output_tokens : int, default=16_384
        Maximum tokens per model response. Set above LiteLLM's OpenAI-compatible
        default of 4096 so the agent can write a complete ``run_code`` Python
        script in a single function call without truncation.
    strategy_dir : Path or None, default=None
        Directory containing the strategy skill (``skill_state.yaml``,
        ``SKILL.md``).  Defaults to ``skills/wti-strategy`` (the base variant).
        Pass an alternative path (e.g. ``skills/wti-strategy-trained``) to
        instantiate the trained variant after a self-directed study session.
        The same directory is used for both the ADK skill load and the mutation
        tool bindings, ensuring the tools always write to the skill the agent
        is reading.

    Returns
    -------
    AgentConfig
    """
    # Delegates to the shared, domain-agnostic builder with the WTI domain and
    # ``WtiStrategyState`` so the rendered SKILL.md keeps its oil branding.  The
    # strategy dir name is baked into the agent name (cache key is derived from
    # predictor_id, which is derived from agent name) so per-variant prediction
    # caches stay separate.
    return build_adaptive_config(
        OIL_DOMAIN,
        state_type=WtiStrategyState,
        model=model,
        search_model=search_model,
        max_output_tokens=max_output_tokens,
        strategy_dir=strategy_dir,
        confirmation_threshold=2,
    )


# ---------------------------------------------------------------------------
# Predictor convenience factory
# ---------------------------------------------------------------------------


def build_wti_adaptive_predictor(
    config: AgentConfig | None = None,
    strategy_dir: Path | None = None,
    model: str = ADVANCED_MODEL,
) -> AgentPredictor:
    """Wrap the adaptive agent in an :class:`AgentPredictor` for eval harness use.

    At each forecast origin the predictor sends a prediction-request payload to
    the agent.  The agent runs its full statistical pipeline (fetch-yfinance →
    vol-regime → trend-projection) in the E2B sandbox, applies calibration
    corrections from its ``wti-strategy`` skill, incorporates news context, and
    returns a probabilistic forecast.

    For resolution delivery and self-review invocations — the interactions
    through which the agent actually learns — use the ADK runner directly
    rather than this predictor interface.

    Parameters
    ----------
    config : AgentConfig, optional
        Agent config to use.  When provided, ``strategy_dir`` is ignored.
        Defaults to ``build_wti_adaptive_config(strategy_dir=strategy_dir)``.
    strategy_dir : Path or None, optional
        Strategy directory passed to :func:`build_wti_adaptive_config` when
        ``config`` is not provided.  Defaults to ``skills/wti-strategy``.
    model : str, optional
        Model identifier passed to :func:`build_wti_adaptive_config` when
        ``config`` is not provided.

    Returns
    -------
    AgentPredictor
    """
    if config is None:
        config = build_wti_adaptive_config(model=model, strategy_dir=strategy_dir)
    return AgentPredictor(
        agent_config=config,
        prompt_builder=WtiAdaptiveForecastPromptBuilder(),
        output_schema=ContinuousAgentForecastOutput,
    )


# ---------------------------------------------------------------------------
# Lazy root_agent for `adk web` interactive use
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    r"""Expose ``root_agent`` lazily for schema-free interactive use via ``adk web``.

    By default the agent loads the seed strategy (``wti-strategy``).  To load
    a different strategy — e.g. after a training session — set the
    ``WTI_STRATEGY_DIR`` environment variable to an absolute or repo-relative
    path before launching::

        WTI_STRATEGY_DIR=adaptive_agent/skills/wti-strategy-trained \
            uv run adk web adaptive_agent/
    """
    if name == "root_agent":
        import os  # noqa: PLC0415

        strategy_env = os.environ.get("WTI_STRATEGY_DIR")
        strategy_dir = Path(strategy_env) if strategy_env else None
        return build_adk_agent(build_wti_adaptive_config(strategy_dir=strategy_dir))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```
