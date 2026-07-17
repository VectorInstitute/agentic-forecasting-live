"""Track-2 qualitative scenario-analysis prompt + agent config for the TSX.

Track 2 asks the TSX news-analyst agent for a free-text scenario write-up
instead of a structured quantile forecast: the 2-3 scenarios Canadian
equity-market analysts are debating for the S&P/TSX Composite over the next
~60 business days, each with a name, a calibrated probability, concrete
key drivers, and a qualitative outlook — no ``output_schema``, no
``set_model_response`` tool. This is the free-text sibling of the oil
case-study's schema-*full* Task C
(``energy_oil_forecasting.tasks.TASK_SCENARIOS_SPEC`` /
``ScenarioAgentForecastOutput`` in
:mod:`aieng.forecasting.methods.agentic.outputs`): same qualitative shape
(scenario name, probability, key drivers, horizon outlook, base case), no JSON.

No change to :func:`~aieng.forecasting.methods.agentic.domain.build_analyst_config`
was needed to support this. ``output_schema`` is not one of its parameters —
nor of :class:`~aieng.forecasting.methods.agentic.agent_factory.AgentConfig` at
all. It is supplied independently by the *caller* of
:func:`~aieng.forecasting.methods.agentic.agent_factory.build_adk_agent`
(``AgentPredictor`` passes one for Track 1; the ``ws-scenario`` CLI passes
none — the default — for Track 2), so the same ``build_analyst_config`` /
``AgentConfig`` machinery already supports a schema-less agent identity
unmodified: only the *instruction* text differs (no JSON schema block, no
"call `set_model_response`" rule).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.methods.agentic.agent_factory import AgentConfig, ContextRetrievalConfig
from aieng.forecasting.methods.agentic.domain import build_analyst_config
from aieng.forecasting.methods.agentic.history import compress_history
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL

from workshop_experiments.domain_tsx import TSX_DOMAIN


# ---------------------------------------------------------------------------
# Scenario horizon
# ---------------------------------------------------------------------------

#: Scenario horizon advertised to the agent (business days) — matches the
#: judge's longest realized-outcome window (see workshop_experiments.scenario_outcome).
SCENARIO_HORIZON_BDAYS = 60


# ---------------------------------------------------------------------------
# Prompt — schema-less scenario instruction (oil tasks.py TASK_SCENARIOS_SPEC
# heritage, rewritten as free-text prose instead of a JSON contract)
# ---------------------------------------------------------------------------

_TSX_SCENARIO_INSTRUCTION = (
    "## Role\n\n"
    f"You are an expert {TSX_DOMAIN.analyst_persona}. You produce qualitative "
    f"scenario analysis grounded in {TSX_DOMAIN.analyst_forecasting_focus}.\n\n"
    "## Task\n\n"
    "You will receive a JSON payload containing:\n"
    "- `as_of`: the analysis origin date in YYYY-MM-DD format\n"
    "- `horizon_bdays`: the scenario horizon in business days (roughly 3 "
    "calendar months)\n"
    "- `target_summary`: the last close-to-close log return, its date, "
    "observation count, and trailing realised volatility\n"
    f"- `target_history_csv`: {TSX_DOMAIN.target_history_description}\n\n"
    "Identify the **2 or 3 scenarios** Canadian equity-market analysts are "
    "actually debating for the S&P/TSX Composite over the next "
    f"{SCENARIO_HORIZON_BDAYS} business days. This is qualitative scenario "
    "analysis, not a quantile forecast: there is **no output schema** and "
    "**no `set_model_response` tool** for this task — write your answer "
    "directly as markdown prose, not JSON.\n\n"
    "For each scenario, cover:\n"
    '1. **Name** — a short, memorable label (e.g. "BoC cutting cycle '
    'extends", "commodity-led breakout", "tariff-driven stall").\n'
    "2. **Probability** — your calibrated estimate that this scenario is the "
    "one that plays out; the probabilities across your scenarios should sum "
    "to roughly 1.0.\n"
    "3. **Key drivers** — the specific Bank of Canada policy path, Canadian "
    "inflation/jobs data, commodity (oil/gold) moves, the loonie and GoC "
    "yields, US policy spillovers, or sector-earnings catalysts that would "
    "produce this path. Be concrete: name the actual data points and dates "
    "you found via search, not generic categories.\n"
    f"4. **~{SCENARIO_HORIZON_BDAYS}-day outlook** — the direction and rough "
    "magnitude of the cumulative S&P/TSX Composite return you would expect "
    'under this scenario (e.g. "roughly flat to +3%", "a drawdown of '
    '5-8%").\n\n'
    "Then state which scenario is your **base case** and why.\n\n"
    "## Output format\n\n"
    "Markdown only, structured as:\n\n"
    "```markdown\n"
    "## Scenario: <name> (~<probability, e.g. 0.45>)\n"
    "**Key drivers:** <2-4 sentences, concrete and dated>\n"
    f"**~{SCENARIO_HORIZON_BDAYS}-day outlook:** <direction + rough magnitude>\n\n"
    "## Scenario: <name> (~<probability>)\n"
    "...\n\n"
    "## Base case\n"
    "<which scenario, and a short justification>\n"
    "```\n\n"
    "## Analysis discipline\n\n"
    "Call `search_web` to gather current Canadian macro and market "
    "intelligence BEFORE writing your scenarios. Call `search_web` with "
    "`query` and `cutoff_date` (set to the `as_of` date from the payload) — "
    "this is the temporal fence that keeps the analysis honest in a "
    "backtest. If `search_web` returns a result beginning with "
    "`[SEARCH_VERIFICATION_FAILED]`, treat it as no verified news context "
    "for that query — do not fill the gap from your own background "
    "knowledge, and note the gap in your write-up.\n\n"
    "Recommended queries (call `search_web` once per topic):\n"
    + "".join(f'- `search_web(query="{q}", cutoff_date=<as_of>)`\n' for q in TSX_DOMAIN.recommended_search_queries)
    + "\n"
    f"Ground each scenario's drivers in what you actually found "
    f"({TSX_DOMAIN.key_assumptions_hint}) — do not present a scenario whose "
    "drivers are pure speculation."
)


# ---------------------------------------------------------------------------
# Prompt builder — serialises the target series into the user-turn JSON payload
# ---------------------------------------------------------------------------


def build_scenario_prompt(*, as_of: str, context: ForecastContext) -> str:
    """Build the Track-2 scenario-analysis user prompt for one origin.

    Mirrors :class:`workshop_experiments.domain_tsx.TsxReturnForecastPromptBuilder`
    (same target-series JSON payload shape: ``target_summary`` +
    ``target_history_csv``), but carries no ``task`` — Track 2 is a standalone
    scenario write-up, not a per-task backtest prediction — and advertises
    ``horizon_bdays`` instead of the quantile-forecast ``horizons`` /
    ``standard_quantiles`` fields.

    Parameters
    ----------
    as_of : str
        The forecast origin date, ``YYYY-MM-DD``.
    context : ForecastContext
        A cutoff-scoped view of the TSX data service at ``as_of``.
    """
    df = context.get_series(TSX_DOMAIN.target_series_id)
    compressed = compress_history(df)

    values = df["value"].astype(float)
    last_value = float(values.iloc[-1])
    last_date = str(pd.Timestamp(df["timestamp"].iloc[-1]).date())
    trailing_63 = values.tail(63)

    payload: dict[str, Any] = {
        "as_of": as_of,
        "horizon_bdays": SCENARIO_HORIZON_BDAYS,
        "target_summary": {
            "last_log_return": last_value,
            "last_date": last_date,
            "n_obs": int(len(df)),
            "trailing_63d_std": float(trailing_63.std()) if len(trailing_63) > 1 else float("nan"),
        },
        "target_history_csv": compressed,
    }
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# AgentConfig factory
# ---------------------------------------------------------------------------


def build_tsx_scenario_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Build the Track-2 scenario-analyst :class:`AgentConfig` for the TSX.

    Same news-grounded wiring as
    :func:`workshop_experiments.domain_tsx.build_tsx_news_config` (cutoff-bounded
    ``search_web`` + an independent temporal-leakage verifier, no code
    execution) but with the schema-less scenario instruction in place of the
    quantile-forecast one. The agent name is ``tsx_analyst_scenario``.

    Callers must build the ADK agent with
    ``build_adk_agent(config)`` — i.e. passing **no** ``output_schema`` — so
    the agent has no ``set_model_response`` tool and returns its free-text
    write-up as plain model text (see
    :func:`aieng.forecasting.methods.agentic.agent_factory.build_adk_agent`,
    whose ``output_schema`` defaults to ``None``).
    """
    return build_analyst_config(
        TSX_DOMAIN,
        name_suffix="scenario",
        instruction=_TSX_SCENARIO_INSTRUCTION,
        model=model,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=TSX_DOMAIN.context_retrieval_instruction,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
    )


__all__ = [
    "SCENARIO_HORIZON_BDAYS",
    "build_scenario_prompt",
    "build_tsx_scenario_config",
]
