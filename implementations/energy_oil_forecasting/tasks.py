"""Task specifications and agent predictor wiring for the WTI experiment.

Implements the "one agent, three tasks" pattern: a single :class:`AgentConfig`
identity with task-specific prompt builders and output schemas supplied via
:class:`~aieng.forecasting.methods.agentic.predictor.AgentPredictor`.

The scenario output classes are thin oil-branded subclasses of the shared,
domain-agnostic :class:`~aieng.forecasting.methods.agentic.outputs.ScenarioCard`
and :class:`~aieng.forecasting.methods.agentic.outputs.ScenarioAgentForecastOutput`.
They pin the numeric scenario-card fields to the exact key names
(``wti_range_60d``, ``point_estimate_60d``) the notebook caches key on.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, Literal

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import (
    AgentPredictor,
    ContinuousAgentForecastOutput,
    DiscreteAgentForecastOutput,
)
from aieng.forecasting.methods.agentic.agent_factory import AgentConfig
from aieng.forecasting.methods.agentic.outputs import (
    AgentForecastOutput,
)
from aieng.forecasting.methods.agentic.outputs import (
    ScenarioAgentForecastOutput as _BaseScenarioAgentForecastOutput,
)
from aieng.forecasting.methods.agentic.outputs import (
    ScenarioCard as _BaseScenarioCard,
)
from aieng.forecasting.models import LITE_MODEL
from energy_oil_forecasting.analyst_agent import (
    WtiPriceForecastPromptBuilder,
    build_wti_multitask_news_config,
    build_wti_news_config,
    compress_history,
)
from energy_oil_forecasting.paths import SHOCK_HORIZON, SHOCK_THRESHOLD
from pydantic import BaseModel


# ── Task specification strings (embedded in user prompts for NB3) ───────────
# Each spec uses the corresponding output class's prompt_schema_json() so the
# required JSON format in the prompt is always in sync with the Pydantic schema.

TASK_TRAJECTORY_SPEC = (
    "Forecast the WTI crude oil price at the horizons listed in the payload.\n\n"
    "If a `set_model_response` tool is available, call it with your complete "
    "JSON as `json_response`. Otherwise return the JSON directly as plain text.\n\n"
    "Required JSON format:\n" + ContinuousAgentForecastOutput.prompt_schema_json()
)

TaskKind = Literal["trajectory", "shock", "scenario"]


class WtiMultitaskPromptBuilder(BaseModel):
    """Prompt builder for task-spec-driven agent calls (NB3)."""

    task_spec: str

    model_config = {"extra": "forbid"}

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        df = context.get_series(task.target_series_id)
        last_row = df.iloc[-1]
        payload: dict[str, Any] = {
            "task": task.task_id,
            "task_spec": self.task_spec,
            "as_of": str(context.as_of)[:10],
            "origin_price_usd_bbl": float(last_row["value"]),
            "target_history_csv": compress_history(df),
        }
        return json.dumps(payload, indent=2)


class ScenarioCard(_BaseScenarioCard):
    """One scenario card from Task C agent output.

    Adds the WTI-specific 60-day price range and point estimate to the generic
    scenario card, under the exact field names the notebook caches expect.
    """

    wti_range_60d: list[float]
    point_estimate_60d: float


class ScenarioAgentForecastOutput(_BaseScenarioAgentForecastOutput):
    """Track 2 scenario analysis output for the energy case study.

    Narrows the scenario-card type to :class:`ScenarioCard` and advertises the
    WTI numeric fields in the prompt schema template.  Rendering and
    :meth:`to_predictions` are inherited unchanged from the shared base.
    """

    scenarios: list[ScenarioCard]

    scenario_card_template_extra: ClassVar[dict[str, object]] = {
        "wti_range_60d": ["<float_low>", "<float_high>"],
        "point_estimate_60d": "<float>",
    }


# Task specification strings embedded in user prompts for NB3.
# Defined after the output classes so each spec can reference the
# corresponding prompt_schema_json() classmethod — single source of truth.

TASK_SHOCK_SPEC = (
    f"Estimate P(up) — the probability that WTI will close MORE THAN\n"
    f"${int(SHOCK_THRESHOLD)}/bbl HIGHER than today's price at the end of\n"
    f"{SHOCK_HORIZON} trading days.\n\n"
    "If a `set_model_response` tool is available, call it with your complete "
    "JSON as `json_response`. Otherwise return the JSON directly as plain text.\n\n"
    "Required JSON format:\n" + DiscreteAgentForecastOutput.prompt_schema_json()
)

TASK_SCENARIOS_SPEC = (
    "Identify the three scenarios oil market analysts are debating for WTI "
    "over the next 60 days.\n\n"
    "If a `set_model_response` tool is available, call it with your complete "
    "JSON as `json_response`. Otherwise return the JSON directly as plain text.\n\n"
    "Required JSON format:\n" + ScenarioAgentForecastOutput.prompt_schema_json()
)

TASK_SPECS: dict[TaskKind, str] = {
    "trajectory": TASK_TRAJECTORY_SPEC,
    "shock": TASK_SHOCK_SPEC,
    "scenario": TASK_SCENARIOS_SPEC,
}


TASK_OUTPUT_SCHEMAS: dict[TaskKind, type[AgentForecastOutput]] = {
    "trajectory": ContinuousAgentForecastOutput,
    "shock": DiscreteAgentForecastOutput,
    "scenario": ScenarioAgentForecastOutput,
}


def build_wti_news_predictor(
    task: TaskKind,
    model: str = LITE_MODEL,
) -> AgentPredictor:
    """Build a news-grounded agent predictor for the given task kind.

    Parameters
    ----------
    task : TaskKind
        One of ``"trajectory"``, ``"shock"``, or ``"scenario"``.
    model : str
        Model identifier passed through to the underlying
        :class:`~aieng.forecasting.methods.agentic.agent_factory.AgentConfig`.
        Defaults to the lite model (``"gemini-3.1-flash-lite-preview"``); pass the
        advanced model (``"gemini-3.5-flash"``) when more capability is needed.
    """
    if task == "trajectory":
        return AgentPredictor(
            agent_config=build_wti_news_config(model=model),
            prompt_builder=WtiPriceForecastPromptBuilder(),
            output_schema=ContinuousAgentForecastOutput,
        )
    return AgentPredictor(
        agent_config=build_wti_multitask_news_config(model=model),
        prompt_builder=WtiMultitaskPromptBuilder(task_spec=TASK_SPECS[task]),
        output_schema=TASK_OUTPUT_SCHEMAS[task],
    )


def build_wti_agent_predictor_for_task(config: AgentConfig, task: TaskKind) -> AgentPredictor:
    """Wire any WTI agent config to a task-specific predictor."""
    if task == "trajectory":
        return AgentPredictor(
            agent_config=config,
            prompt_builder=WtiPriceForecastPromptBuilder(),
            output_schema=ContinuousAgentForecastOutput,
        )
    return AgentPredictor(
        agent_config=config,
        prompt_builder=WtiMultitaskPromptBuilder(task_spec=TASK_SPECS[task]),
        output_schema=TASK_OUTPUT_SCHEMAS[task],
    )


__all__ = [
    "TASK_SCENARIOS_SPEC",
    "TASK_SHOCK_SPEC",
    "TASK_SPECS",
    "TASK_TRAJECTORY_SPEC",
    "ScenarioAgentForecastOutput",
    "ScenarioCard",
    "TaskKind",
    "WtiMultitaskPromptBuilder",
    "build_wti_agent_predictor_for_task",
    "build_wti_news_predictor",
]
