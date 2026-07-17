"""Offline tests for the Track-2 TSX scenario prompt + agent config.

No LLM/API calls: the config-building and prompt-building functions are pure
(or operate on a fake ``ForecastContext``), so these exercise the real code
paths ``ws-scenario`` uses without spending anything.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from workshop_experiments.scenario_domain_tsx import (
    SCENARIO_HORIZON_BDAYS,
    build_scenario_prompt,
    build_tsx_scenario_config,
)


class _FakeContext:
    """Minimal stand-in for ``ForecastContext``: only ``get_series`` is used."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Store the series to hand back for any series id."""
        self._df = df

    def get_series(self, series_id: str) -> pd.DataFrame:  # noqa: ARG002 - protocol match
        """Return the stored series regardless of the requested id."""
        return self._df


def _synthetic_return_series(n: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=n)
    values = [0.001 * (i % 5 - 2) for i in range(n)]
    return pd.DataFrame({"timestamp": dates, "value": values})


def test_scenario_config_builds_with_scenario_agent_name() -> None:
    """The Track-2 config builds; the agent name is tsx_analyst_scenario."""
    config = build_tsx_scenario_config()
    assert config.name == "tsx_analyst_scenario"
    assert config.context_retrieval.enabled
    assert not config.code_execution.enabled


def test_scenario_instruction_has_no_json_output_schema() -> None:
    """The scenario instruction is schema-less: no set_model_response *contract*.

    It does mention "no `set_model_response` tool" (explaining Track 2's
    schema-less shape), but never *directs* the agent to call it, and never
    embeds a JSON schema block the way the Track-1 news/code instructions do.
    """
    config = build_tsx_scenario_config()
    assert "call `set_model_response`" not in config.instruction
    assert "```json" not in config.instruction
    assert "no output schema" in config.instruction.lower()
    assert "no `set_model_response` tool" in config.instruction.lower()
    assert "markdown" in config.instruction.lower()


def test_scenario_instruction_names_the_scenario_shape() -> None:
    """The instruction asks for 2-3 named, probability-weighted scenarios."""
    config = build_tsx_scenario_config()
    text = config.instruction
    assert "2 or 3 scenarios" in text
    assert "Probability" in text
    assert "Key drivers" in text
    assert "Base case" in text
    assert f"{SCENARIO_HORIZON_BDAYS}-day outlook" in text


def test_scenario_instruction_no_sp500_or_oil_persona_bleed() -> None:
    """No cross-domain persona bleed into the scenario instruction."""
    text = build_tsx_scenario_config().instruction
    lowered = text.lower()
    for forbidden in ("s&p 500 equity-index analyst", "wti crude oil market analyst", "oil market analyst"):
        assert forbidden not in lowered


def test_build_scenario_prompt_is_valid_json_with_expected_fields() -> None:
    """The prompt payload is valid JSON with as_of, horizon_bdays, and history."""
    context = _FakeContext(_synthetic_return_series())
    prompt = build_scenario_prompt(as_of="2025-04-01", context=context)
    payload = json.loads(prompt)

    assert payload["as_of"] == "2025-04-01"
    assert payload["horizon_bdays"] == SCENARIO_HORIZON_BDAYS
    assert "target_summary" in payload
    assert payload["target_summary"]["n_obs"] == 80
    assert "target_history_csv" in payload
    assert payload["target_history_csv"].startswith("date,close")
    # No quantile-forecast fields — this is not the Track-1 payload shape.
    assert "horizons" not in payload
    assert "standard_quantiles" not in payload


def test_build_scenario_prompt_rejects_empty_series() -> None:
    """An empty series raises rather than silently emitting a bogus payload."""
    context = _FakeContext(pd.DataFrame({"timestamp": [], "value": []}))
    with pytest.raises(IndexError):
        build_scenario_prompt(as_of="2025-04-01", context=context)
