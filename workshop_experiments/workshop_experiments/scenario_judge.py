"""``ws-scenario-judge`` — LLM-judge grading pass over persisted Track-2 write-ups.

A **side-channel** evaluator, deliberately separate from ``ws-scenario``: for
each persisted scenario write-up (see :mod:`workshop_experiments.scenario` /
:mod:`workshop_experiments.scenario_store`) it computes the realized S&P/TSX
Composite outcome over the following 5 / 21 / 60 business days
(:mod:`workshop_experiments.scenario_outcome`) and asks an LLM judge to score
the write-up against what actually happened, on three rubric axes:

- **drivers** — were the key drivers identified correctly, i.e. do they match
  the realized narrative for the following weeks?
- **calibration** — how well do the stated scenario probabilities match the
  realized direction (a scenario that priced the realized outcome highly
  scores well; one that priced it as a tail case does not)?
- **specificity** — concrete, dated, checkable claims vs. generic hedging.

Each axis is rated 1-5 with a short justification. The judge call reuses the
LLM-process judge-call shape from
``implementations/boc_rate_decisions/rationale_eval.py``
(:func:`~aieng.forecasting.methods.llm_processes._client.make_json_schema_response_format`
+ ``run_async``/``sample_n_async``), so proxy routing, retries, and
strict-schema enforcement are shared with the forecasters.

**Dry-run by default**; pass ``--run`` to actually call the judge model. An
origin already judged is skipped on the next run (resume semantics) unless
``--force`` is given. An origin with no persisted write-up is reported as
missing and skipped either way.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd
from pydantic import BaseModel, Field

from workshop_experiments.scenario import DEFAULT_ORIGINS
from workshop_experiments.scenario_outcome import (
    JUDGE_HORIZONS,
    RealizedOutcomeSummary,
    compute_realized_outcome_summary,
)
from workshop_experiments.scenario_store import (
    DEFAULT_SCENARIO_STORE_DIR,
    has_judge_verdict,
    has_writeup,
    load_scenario_writeup,
    write_judge_verdict,
)


#: Default judge model — Track 1's forecaster defaults (LITE_MODEL /
#: ADVANCED_MODEL) are Gemini; the judge is deliberately pinned to a different
#: model family so grading is not the same model marking its own homework.
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Judge verdict schema + rubric prompt
# ---------------------------------------------------------------------------


class ScenarioJudgeVerdict(BaseModel):
    """The judge's structured rubric assessment of one scenario write-up.

    Attributes
    ----------
    drivers_score : int
        1-5. Were the key drivers identified correctly — do they match the
        realized narrative for the weeks following the origin?
    calibration_score : int
        1-5. How well the stated scenario probabilities match the realized
        direction (a scenario that priced the realized outcome highly scores
        well; one that treated it as a tail case does not).
    specificity_score : int
        1-5. Concrete, dated, checkable claims vs. generic hedging.
    """

    drivers_score: int = Field(ge=1, le=5)
    drivers_justification: str = ""
    calibration_score: int = Field(ge=1, le=5)
    calibration_justification: str = ""
    specificity_score: int = Field(ge=1, le=5)
    specificity_justification: str = ""
    overall_justification: str = ""


_JUDGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "drivers_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "drivers_justification": {"type": "string"},
        "calibration_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "calibration_justification": {"type": "string"},
        "specificity_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "specificity_justification": {"type": "string"},
        "overall_justification": {"type": "string"},
    },
    "required": [
        "drivers_score",
        "drivers_justification",
        "calibration_score",
        "calibration_justification",
        "specificity_score",
        "specificity_justification",
        "overall_justification",
    ],
    "additionalProperties": False,
}

_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator of qualitative equity-market scenario analysis. "
    "You are given a forecaster's scenario write-up for the S&P/TSX Composite "
    "index (2-3 named, probability-weighted scenarios written BEFORE the outcome "
    "was known) and a summary of what the index ACTUALLY did over the following "
    "5/21/60 business days. Score the write-up on three axes.\n"
    "\n"
    "Rules:\n"
    "- Return ONLY a JSON object matching the provided schema. No prose, no markdown.\n"
    "- 'drivers_score' in [1, 5] rates whether the write-up's key drivers match "
    "the realized narrative for the weeks following the origin: 5 = the cited "
    "drivers are exactly what moved the index; 1 = the cited drivers are "
    "unrelated to what actually happened. Judge this from the realized-outcome "
    "summary alone (you were not given the news that broke); if the summary is "
    "silent on cause, judge plausibility and specificity of the causal claims "
    "instead.\n"
    "- 'calibration_score' in [1, 5] rates whether the STATED PROBABILITIES "
    "matched the REALIZED DIRECTION: 5 = the scenario that best matches the "
    "realized 5/21/60-day returns was assigned the highest (or a clearly "
    "dominant) probability; 1 = the realized outcome was assigned the lowest "
    "probability or excluded entirely. This is about probability weighting, not "
    "about whether any one scenario's numeric range was hit exactly.\n"
    "- 'specificity_score' in [1, 5] rates how concrete and checkable the "
    "write-up is: 5 = specific, dated data points and named catalysts; 1 = "
    "generic hedging that could apply to almost any origin.\n"
    "- Each '*_justification' is 1-3 sentences citing specifics from the "
    "write-up and the realized-outcome summary.\n"
    "- 'overall_justification' is a 1-2 sentence summary of the overall grade."
)


def _build_judge_user_prompt(
    *,
    origin: date,
    writeup_markdown: str,
    realized_outcome_markdown: str,
) -> str:
    """Assemble the judge's user message."""
    return (
        f"Scenario write-up origin: {origin.isoformat()}\n"
        "\n"
        "Forecaster's scenario write-up (written before the outcome was known):\n"
        f"{writeup_markdown}\n"
        "\n"
        f"{realized_outcome_markdown}\n"
        "\n"
        "Return the JSON rubric verdict."
    )


def judge_scenario_writeup(
    *,
    origin: date,
    writeup_markdown: str,
    realized_outcome_summary: RealizedOutcomeSummary,
    model: str = DEFAULT_JUDGE_MODEL,
    reasoning_effort: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    timeout_s: float = 120.0,
) -> ScenarioJudgeVerdict:
    """Run one LLM-as-judge call scoring a scenario write-up against reality.

    Reuses the shared LLM-process completion seam (proxy routing +
    strict-schema enforcement) — the same judge-call shape as
    ``implementations/boc_rate_decisions/rationale_eval.judge_rationale_alignment``.

    Returns
    -------
    ScenarioJudgeVerdict

    Raises
    ------
    RuntimeError
        If the judge returns no schema-valid verdict.
    """
    from aieng.forecasting.methods.llm_processes._client import (  # noqa: PLC0415
        make_json_schema_response_format,
        run_async,
        sample_n_async,
    )

    base_messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_judge_user_prompt(
                origin=origin,
                writeup_markdown=writeup_markdown,
                realized_outcome_markdown=realized_outcome_summary.to_markdown(),
            ),
        },
    ]
    response_format = make_json_schema_response_format("ScenarioJudgeVerdict", _JUDGE_JSON_SCHEMA, model=model)
    parsed, _cost, _in, _out, _fails = run_async(
        sample_n_async(
            schema_cls=ScenarioJudgeVerdict,
            model=model,
            base_messages=base_messages,
            response_format=response_format,
            n_samples=1,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            reasoning_effort=reasoning_effort,
            api_base=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
    )
    if not parsed:
        raise RuntimeError(f"Scenario judge returned no schema-valid verdict for origin {origin.isoformat()}.")
    return parsed[0]


# ---------------------------------------------------------------------------
# Plan / run wiring
# ---------------------------------------------------------------------------

#: Injection seam for the realized-series lookup: ``horizon -> tsx_logret_{h}b frame``.
GetSeries = Callable[[int], pd.DataFrame]

#: Injection seam for the judge call itself (offline tests supply a fake).
Judge = Callable[..., ScenarioJudgeVerdict]


@dataclass(frozen=True)
class ScenarioJudgePlan:
    """The resolved plan for a ``ws-scenario-judge`` run (the dry-run default)."""

    origins: tuple[date, ...]
    model: str
    store_dir: Path

    def describe(self) -> str:
        """Render the plan as a human-readable block for the dry-run default."""
        lines = [
            "ws-scenario-judge plan (dry-run)",
            f"  judge model : {self.model}",
            f"  store dir   : {self.store_dir}",
            f"  horizons    : {list(JUDGE_HORIZONS)} business days",
            "  origins     :",
        ]
        for origin in self.origins:
            wu = "write-up present" if has_writeup(origin, self.store_dir) else "MISSING write-up"
            jv = "already judged" if has_judge_verdict(origin, self.store_dir) else "not yet judged"
            lines.append(f"    - {origin.isoformat()}: {wu}, {jv}")
        lines.append("  NOTE: --run makes a judge-model call per un-judged, write-up-present origin.")
        return "\n".join(lines)


def plan_scenario_judge(
    *,
    origins: Sequence[str] | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    store_dir: Path = DEFAULT_SCENARIO_STORE_DIR,
) -> ScenarioJudgePlan:
    """Resolve a scenario-judge run plan without running anything."""
    resolved = tuple(date.fromisoformat(o) for o in (origins or DEFAULT_ORIGINS))
    return ScenarioJudgePlan(origins=resolved, model=model, store_dir=store_dir)


def judge_scenario_for_origin(
    origin: date,
    *,
    store_dir: Path,
    get_series: GetSeries,
    judge: Judge,
    model: str,
    force_refresh: bool = False,
) -> Path | None:
    """Judge (or skip) one origin's persisted write-up.

    Returns the judge-verdict file path, or ``None`` when the origin was
    skipped: no persisted write-up, or already judged and ``force_refresh`` is
    ``False``.
    """
    if not force_refresh and has_judge_verdict(origin, store_dir):
        return None
    writeup = load_scenario_writeup(origin, store_dir)
    if writeup is None:
        return None

    summary = compute_realized_outcome_summary(get_series, origin=origin)
    verdict = judge(
        origin=origin,
        writeup_markdown=writeup.markdown,
        realized_outcome_summary=summary,
        model=model,
    )

    payload = {
        "origin": origin.isoformat(),
        "judge_model": model,
        "judged_at": datetime.now(tz=timezone.utc).isoformat(),
        "realized_outcome": {
            "summary_markdown": summary.to_markdown(),
            "horizons": [
                {
                    "horizon": outcome.horizon,
                    "forecast_date": outcome.forecast_date.isoformat(),
                    "log_return": outcome.log_return,
                    "pct_return": outcome.pct_return,
                    "direction": outcome.direction,
                }
                for outcome in summary.outcomes
            ],
        },
        "verdict": verdict.model_dump(),
    }
    return write_judge_verdict(origin, payload, store_dir)


def run_scenario_judge_plan(
    plan: ScenarioJudgePlan,
    *,
    get_series: GetSeries,
    judge: Judge = judge_scenario_writeup,
    force_refresh: bool = False,
) -> list[Path]:
    """Judge every pending origin in *plan*, persisting the verdict per origin.

    ``get_series`` and ``judge`` are injectable for offline tests; the CLI
    wires ``get_series`` to a live ``DataService`` lookup and leaves ``judge``
    at its real default.
    """
    written: list[Path] = []
    for origin in plan.origins:
        path = judge_scenario_for_origin(
            origin,
            store_dir=plan.store_dir,
            get_series=get_series,
            judge=judge,
            model=plan.model,
            force_refresh=force_refresh,
        )
        if path is not None:
            written.append(path)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-judge grading pass over persisted Track-2 scenario write-ups.")
    parser.add_argument(
        "--origin",
        action="append",
        default=None,
        metavar="YYYY-MM-DD",
        help="Scenario origin (repeatable; default: the landmark set).",
    )
    parser.add_argument("--model", default=DEFAULT_JUDGE_MODEL, help="Judge model (default: claude-sonnet-4-6).")
    parser.add_argument(
        "--store-dir", default=str(DEFAULT_SCENARIO_STORE_DIR), help="Scenario write-up / verdict store root."
    )
    parser.add_argument("--run", action="store_true", help="Actually call the judge model.")
    parser.add_argument("--force", action="store_true", help="Re-judge and overwrite an already-judged origin.")
    return parser


def run(argv: list[str] | None = None) -> int:
    """Entry point for the ``ws-scenario-judge`` console script."""
    args = _build_arg_parser().parse_args(argv)
    plan = plan_scenario_judge(origins=args.origin, model=args.model, store_dir=Path(args.store_dir))

    if not args.run:
        print(plan.describe())
        return 0

    from workshop_experiments.data_tsx import build_tsx_log_return_service, tsx_logret_series_id  # noqa: PLC0415

    data_service = build_tsx_log_return_service(windows=JUDGE_HORIZONS)
    now = datetime.now(tz=timezone.utc).replace(tzinfo=None)

    def get_series(horizon: int) -> pd.DataFrame:
        return data_service.get_series(tsx_logret_series_id(horizon), as_of=now)

    written = run_scenario_judge_plan(plan, get_series=get_series, force_refresh=args.force)
    print(f"wrote {len(written)} judge verdict(s).")
    for path in written:
        print(f"  - {path}")
    return 0


def main() -> None:
    """Console-script wrapper that exits with the run's status code."""
    sys.exit(run())


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "GetSeries",
    "Judge",
    "ScenarioJudgePlan",
    "ScenarioJudgeVerdict",
    "judge_scenario_for_origin",
    "judge_scenario_writeup",
    "main",
    "plan_scenario_judge",
    "run",
    "run_scenario_judge_plan",
]
