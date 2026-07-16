"""The live reflection step: post-resolution learning for the learning twin.

After resolutions land for the learning twin, the live pipeline runs a bounded
reflection session. Inputs: the twin's own committed forecast + rationale, the
realized outcome and CRPS, date-bounded news for the (now-past) window, and the
current strategy. The session proposes updates through the **gated** mutation
layer (:func:`build_gated_skill_tools`), so every proposal is tiered and every
adoption/rejection writes a ``mutation_event``.

Spend-bearing and pluggable: the driver runs whatever
:class:`~workshop_experiments.adaptive.session.StudySession` it is given, so the
prompt construction, turn bounding, and accounting test offline; the real
(model + E2B) session is built only under the live pipeline's run flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from aieng.forecasting.methods.agentic.agent_factory import AgentConfig

from workshop_experiments.adaptive.domain import build_sp500_adaptive_config
from workshop_experiments.adaptive.gates.gated_tools import build_gated_skill_tools
from workshop_experiments.adaptive.gates.policy import GatePolicy
from workshop_experiments.adaptive.session import StudySession, TurnResult
from workshop_experiments.adaptive.study import StudyAccounting


@dataclass
class ReflectionInputs:
    """Everything the reflection session is handed for one resolved origin."""

    origin: date
    committed_forecast: str
    rationale: str
    realized: str
    crps: str
    strategy_summary: str = ""


def build_reflection_prompt(inputs: ReflectionInputs, *, window_days: int = 21) -> str:
    """Build the bounded reflection prompt for one resolved origin."""
    strat = f"\nYour current strategy (summary):\n{inputs.strategy_summary}\n" if inputs.strategy_summary else ""
    return f"""\
Live reflection for your forecast committed at origin {inputs.origin.isoformat()},
now resolved. You are a live forecaster whose proposed updates must
survive the gates, so propose thoughtfully rather than overzealously.

Your committed forecast:
{inputs.committed_forecast}

Your committed rationale:
{inputs.rationale}

Realized outcome:
{inputs.realized}

Score (CRPS, lower is better):
{inputs.crps}
{strat}
Reflect:
1. Retrieve date-bounded news for the window around {inputs.origin.isoformat()}
   (cutoff_date = {inputs.origin.isoformat()}; nothing post-origin). On
   [SEARCH_VERIFICATION_FAILED], proceed on price history and note the gap.
2. Was the outcome knowable from a narrative signal you underweighted, or was it
   genuine noise? Distinguish a durable forecasting flaw from a one-off shock.
3. Propose updates ONLY where they clear your meta-learning evidence bar. Record
   observations freely; open or confirm hypotheses where the live record supports
   them; graduate a calibration correction only when the confirmation threshold is
   met. An approach-narrative rewrite will enter a forward shadow test, not take
   effect immediately — reserve it for a structural insight.
"""


def build_reflection_config(
    policy: GatePolicy,
    *,
    origin_date: date,
    model: str,
    now: date | None = None,
) -> AgentConfig:
    """Build the learning twin's reflection :class:`AgentConfig` (gated tools).

    Same adaptive agent as the twin predictor, but with the **gated** mutation
    tools attached so every proposal routes through *policy*. The strategy dir is
    the learning twin's evolving dir (the policy's ``strategy_dir``).
    """
    base = build_sp500_adaptive_config(
        model=model,
        strategy_dir=policy.strategy_dir,
        confirmation_threshold=policy.config.confirmation_threshold,
        attach_mutation_tools=False,  # replace with gated tools below
    )
    gated = build_gated_skill_tools(policy, origin_date=origin_date, now=now)
    return base.model_copy(update={"extra_tools": tuple(gated)})


@dataclass
class ReflectionResult:
    """Outcome of one reflection session."""

    origin: date
    turns: list[TurnResult]
    accounting: StudyAccounting


def run_reflection(
    session: StudySession,
    inputs: ReflectionInputs,
    *,
    max_turns: int = 2,
    window_days: int = 21,
    transcript_dir: Path | None = None,
) -> ReflectionResult:
    """Drive a bounded reflection over *session*; return turns + accounting.

    The agent's gated tools do the gating and event-writing as a side effect;
    this driver only bounds the turns and accounts for tokens.
    """
    from workshop_experiments.adaptive.study import CONTINUE_PROMPT  # noqa: PLC0415

    acc = StudyAccounting()
    turns: list[TurnResult] = []
    first = build_reflection_prompt(inputs, window_days=window_days)
    for turn in range(1, max(1, max_turns) + 1):
        prompt = first if turn == 1 else CONTINUE_PROMPT
        result = session.run_turn(prompt)
        turns.append(result)
        acc.add(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            wall_time_s=result.wall_time_s,
        )
    if transcript_dir is not None:
        transcript_dir.mkdir(parents=True, exist_ok=True)
        import json  # noqa: PLC0415

        path = transcript_dir / f"reflection_{inputs.origin.isoformat()}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for i, t in enumerate(turns, start=1):
                handle.write(
                    json.dumps(
                        {
                            "turn": i,
                            "response": t.text,
                            "input_tokens": t.input_tokens,
                            "output_tokens": t.output_tokens,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    return ReflectionResult(origin=inputs.origin, turns=turns, accounting=acc)


__all__ = [
    "ReflectionInputs",
    "ReflectionResult",
    "build_reflection_config",
    "build_reflection_prompt",
    "run_reflection",
]
