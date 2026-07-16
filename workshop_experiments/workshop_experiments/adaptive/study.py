"""The Study Hall + Residency driver (net-new, non-notebook, resumable).

Replaces the build_nb05 notebook cells with a proper driver over a sticky
:class:`~workshop_experiments.adaptive.session.StudySession`:

- **Phase A — Study Hall** (pre-2025 history): one self-guided multi-turn session.
  The §4 agenda (stylized facts, conditional playbooks, event studies, indicator
  validation, self-calibration) is offered as *suggested directions*, not a
  prescriptive curriculum. Every ``checkpoint_every`` turns the driver issues an
  explicit "distill what you've learned into the strategy now" turn — the mutation
  tools have already persisted anything the agent recorded, so a checkpoint just
  forces consolidation. Transcript + token accounting are persisted per turn, so
  an interrupted run resumes from the next scheduled turn (the trained strategy
  on disk already carries every mutation made before the interruption).
- **Phase B — Residency** (2025 origins): bounded postmortems over the selected
  origins (worst-N by CRPS + controls). Each postmortem hands the agent its own
  forecast, the realized outcome and score, and instructs it to retrieve
  date-bounded news for the origin window and update context-override rules
  through the mutation tools.

Nothing here calls a model API directly; it drives whatever
:class:`StudySession` it is given. The CLI supplies the real (spend-bearing)
session only under ``--run``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from workshop_experiments.adaptive.session import StudySession


# ---------------------------------------------------------------------------
# Prompts (agenda = suggested directions, not a curriculum)
# ---------------------------------------------------------------------------

STUDY_HALL_PROMPT = """\
You are in a self-directed Study Hall. You have code execution over the full
pre-2025 S&P 500 (^GSPC) history and your pipeline + strategy + meta-learning
skills. Nothing here is scored — the point is skill formation. Work the loop:
explore -> hypothesise -> test with code -> distill into your strategy through
the mutation tools, governed by your meta-learning skill.

These are SUGGESTED directions, not a checklist — follow what the data makes
interesting, in whatever order:
- Stylized facts: fat tails, volatility clustering, the leverage effect,
  autocorrelation of returns vs |returns|, drawdown/recovery profiles, seasonality.
- Conditional playbooks: empirical forward-return distributions conditional on
  regime (VIX bands, curve slope, trend state) — "when VIX > 30, the 5-day
  forward distribution looks like X".
- Event studies: 2008, the 2011 downgrade, 2015 China, 2018 Q4, COVID 2020, the
  2022 hiking cycle — shock-decay and vol-spike profiles around known breaks.
- Indicator validation: build realized-vol estimators, momentum/mean-reversion by
  horizon, moving-average states — keep what has distributional forecast value,
  discard folklore.
- Self-calibration: backtest your own quantile bands on history and learn
  coverage corrections (e.g. "my 90% bands ran at 78% in high-VIX regimes").

Always end a substantive finding by recording it through the appropriate mutation
tool if — and only if — it clears your meta-learning evidence bar. Begin now with
whatever you find most promising.
"""

CONTINUE_PROMPT = """\
Continue your self-directed study. Build on what you have found so far — go
deeper on a promising thread, test a hypothesis you opened, or turn to a
direction you have not yet explored. Use code execution; record durable findings
through the mutation tools per your meta-learning skill.
"""

DISTILL_PROMPT = """\
Checkpoint: distill what you have learned so far into your strategy NOW. Review
your observations and open hypotheses; where the evidence clears the bar, record
observations, open or update hypotheses, and — only where the confirmation
threshold is met — graduate calibration corrections. If your approach narrative no
longer captures how you actually forecast, update it with a rationale. Be
conservative: consolidate what the data supports, do not invent corrections.
"""


def build_postmortem_prompt(
    *,
    origin: date,
    committed_forecast: str,
    realized: str,
    crps: str,
    window_days: int = 21,
) -> str:
    """Build a Residency postmortem prompt for one origin.

    Hands the agent its own committed forecast, the realized outcome + CRPS, and
    the origin window, and asks it to retrieve date-bounded news and update
    context-override rules through the mutation tools.
    """
    return f"""\
Postmortem for your forecast made at origin {origin.isoformat()}.

Your committed forecast:
{committed_forecast}

Realized outcome:
{realized}

Score (CRPS, lower is better):
{crps}

Diagnose this forecast:
1. Retrieve date-bounded market news for the window around {origin.isoformat()}
   (roughly the {window_days} business days after the origin). Set
   cutoff_date = {origin.isoformat()} so nothing post-origin leaks. If a search
   returns [SEARCH_VERIFICATION_FAILED], proceed on price history and note the gap.
2. Was the miss (or the good call) knowable from narrative signals you ignored,
   or was it genuinely not forecastable from information available at the origin?
3. If it points to a durable context-override rule — when and by how much
   retrieved context should move or widen your statistical prior — record it
   through the mutation tools per your meta-learning skill. If it does not clear
   the bar, say what additional evidence you would need.
"""


# ---------------------------------------------------------------------------
# Accounting + run state
# ---------------------------------------------------------------------------


@dataclass
class StudyAccounting:
    """Turn + token accounting for a study run (persisted with the state)."""

    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time_s: float = 0.0

    def add(self, *, input_tokens: int, output_tokens: int, wall_time_s: float) -> None:
        """Fold one turn's token + wall-time usage into the running totals."""
        self.turns += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.wall_time_s += wall_time_s


@dataclass
class StudyResult:
    """Outcome of a phase run."""

    phase: str
    turns_run: int
    accounting: StudyAccounting
    transcript_path: Path
    state_path: Path
    checkpoints: list[int] = field(default_factory=list)


def _state_path(run_dir: Path) -> Path:
    return run_dir / "study_state.json"


def _transcript_path(run_dir: Path) -> Path:
    return run_dir / "transcript.jsonl"


def _load_state(run_dir: Path) -> dict[str, Any]:
    path = _state_path(run_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_state(run_dir: Path, state: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _state_path(run_dir).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_transcript(run_dir: Path, entry: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with _transcript_path(run_dir).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")


@dataclass(frozen=True)
class StudyHallPrompts:
    """The three Study Hall turn prompts (domain-specific text).

    Defaults are the S&P 500 prompts; the TSX bootcamp study supplies its own via
    :data:`workshop_experiments.adaptive.domain_tsx` so the single-session driver
    reuses the exact scheduling / checkpoint / accounting machinery.
    """

    study_hall: str = STUDY_HALL_PROMPT
    cont: str = CONTINUE_PROMPT
    distill: str = DISTILL_PROMPT


def _prompt_for_turn(
    turn_number: int, checkpoint_every: int, prompts: StudyHallPrompts | None = None
) -> tuple[str, str]:
    """Return ``(kind, prompt)`` for a 1-indexed Study Hall *turn_number*."""
    prompts = prompts or StudyHallPrompts()
    if turn_number == 1:
        return "study_hall", prompts.study_hall
    if checkpoint_every > 0 and turn_number % checkpoint_every == 0:
        return "distill", prompts.distill
    return "continue", prompts.cont


def run_study_hall(
    session: StudySession,
    run_dir: Path,
    *,
    turn_budget: int = 50,
    checkpoint_every: int = 10,
    prompts: StudyHallPrompts | None = None,
) -> StudyResult:
    """Drive Phase A over *session*, persisting after every turn (resumable).

    Resume: if ``run_dir`` already holds state, the driver continues from the
    next scheduled turn. The trained strategy on disk already carries every
    mutation the agent made before the interruption (the mutation tools persist).

    ``prompts`` overrides the three turn prompts (S&P 500 by default). The TSX
    bootcamp single-session study passes its own :class:`StudyHallPrompts`.
    """
    prompts = prompts or StudyHallPrompts()
    state = _load_state(run_dir)
    acc = StudyAccounting(**state.get("accounting", {}))
    turns_completed = int(state.get("turns_completed", 0))
    checkpoints = list(state.get("checkpoints", []))

    for turn_number in range(turns_completed + 1, turn_budget + 1):
        kind, prompt = _prompt_for_turn(turn_number, checkpoint_every, prompts)
        result = session.run_turn(prompt)
        acc.add(
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            wall_time_s=result.wall_time_s,
        )
        _append_transcript(
            run_dir,
            {
                "turn": turn_number,
                "kind": kind,
                "response": result.text,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "wall_time_s": result.wall_time_s,
            },
        )
        if kind == "distill":
            checkpoints.append(turn_number)
        _save_state(
            run_dir,
            {
                "phase": "study_hall",
                "turns_completed": turn_number,
                "turn_budget": turn_budget,
                "checkpoint_every": checkpoint_every,
                "checkpoints": checkpoints,
                "accounting": asdict(acc),
            },
        )

    return StudyResult(
        phase="study_hall",
        turns_run=max(0, turn_budget - turns_completed),
        accounting=acc,
        transcript_path=_transcript_path(run_dir),
        state_path=_state_path(run_dir),
        checkpoints=checkpoints,
    )


@dataclass
class Postmortem:
    """One Residency postmortem's inputs."""

    origin: date
    committed_forecast: str
    realized: str
    crps: str


def run_residency(
    session: StudySession,
    run_dir: Path,
    postmortems: list[Postmortem],
    *,
    turns_per_postmortem: int = 3,
    window_days: int = 21,
) -> StudyResult:
    """Drive Phase B: a bounded postmortem over each selected origin (resumable).

    Resumes at postmortem granularity: origins already completed (recorded in the
    state) are skipped.
    """
    state = _load_state(run_dir)
    acc = StudyAccounting(**state.get("accounting", {}))
    done: list[str] = list(state.get("origins_done", []))

    for pm in postmortems:
        key = pm.origin.isoformat()
        if key in done:
            continue
        first = build_postmortem_prompt(
            origin=pm.origin,
            committed_forecast=pm.committed_forecast,
            realized=pm.realized,
            crps=pm.crps,
            window_days=window_days,
        )
        for turn in range(1, max(1, turns_per_postmortem) + 1):
            prompt = first if turn == 1 else CONTINUE_PROMPT
            result = session.run_turn(prompt)
            acc.add(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                wall_time_s=result.wall_time_s,
            )
            _append_transcript(
                run_dir,
                {
                    "origin": key,
                    "turn": turn,
                    "kind": "postmortem" if turn == 1 else "continue",
                    "response": result.text,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "wall_time_s": result.wall_time_s,
                },
            )
        done.append(key)
        _save_state(
            run_dir,
            {
                "phase": "residency",
                "origins_done": done,
                "turns_per_postmortem": turns_per_postmortem,
                "accounting": asdict(acc),
            },
        )

    return StudyResult(
        phase="residency",
        turns_run=acc.turns,
        accounting=acc,
        transcript_path=_transcript_path(run_dir),
        state_path=_state_path(run_dir),
    )


__all__ = [
    "CONTINUE_PROMPT",
    "DISTILL_PROMPT",
    "STUDY_HALL_PROMPT",
    "Postmortem",
    "StudyAccounting",
    "StudyHallPrompts",
    "StudyResult",
    "build_postmortem_prompt",
    "run_residency",
    "run_study_hall",
]
