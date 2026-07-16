"""Gated mutation tools — the policy layer the learning twin actually calls.

The frozen twin gets no mutation tools. The learning twin gets *these* — five
callables with the same signatures as the raw
:func:`build_skill_tools` set, but each routed through a :class:`GatePolicy` so
every write is gated and emits a ``mutation_event``:

- ``record_observation`` → tier 1 (appended).
- ``open_hypothesis`` → tier 2 (proposed), capped by ``max_open_hypotheses``.
- ``record_hypothesis_outcome`` → tier 2 (confirmed/refuted).
- ``graduate_hypothesis`` → tier 2 graduation (k-confirmation + weekly rate limit).
- ``update_approach_narrative`` → tier 3: opens a **shadow challenger** rather
  than applying immediately; the rewrite is adopted only if it does not
  underperform over the shadow window.

Attach the returned list via ``AgentConfig(extra_tools=...)`` on the learning
twin. Because the tools raise :class:`FrozenMutationError` when the policy is
read-only, wiring them onto a frozen policy is a hard error, not a silent write.
"""

from __future__ import annotations

from datetime import date
from typing import Callable

from workshop_experiments.adaptive.gates.policy import GatePolicy


def build_gated_skill_tools(
    policy: GatePolicy,
    *,
    origin_date: date | None = None,
    now: date | None = None,
) -> list[Callable[..., str]]:
    """Build the five gated mutation tools bound to *policy*.

    Parameters
    ----------
    policy : GatePolicy
        The learning twin's gate. ``origin_date`` (the triggering resolution) and
        ``now`` (for the weekly rate-limit reference) are folded into each call.
    origin_date : date or None
        Stamped on the emitted events as the triggering origin.
    now : date or None
        Reference date for the weekly graduation rate limit (defaults to
        ``origin_date`` then today).
    """
    ref_now = now or origin_date or date.today()

    def record_observation(finding: str, linked_hypothesis: str = "") -> str:
        result = policy.observe(
            finding,
            rationale=finding,
            linked_hypothesis=linked_hypothesis,
            origin_date=origin_date,
        )
        return result.message

    def open_hypothesis(claim: str, initial_evidence: str) -> str:
        result = policy.propose_hypothesis(claim, initial_evidence, rationale=initial_evidence, origin_date=origin_date)
        return result.message

    def record_hypothesis_outcome(hypothesis_id: str, outcome: str) -> str:
        result = policy.record_outcome(
            hypothesis_id,
            outcome,
            rationale=f"live resolution outcome: {outcome}",
            origin_date=origin_date,
        )
        return result.message

    def graduate_hypothesis(hypothesis_id: str, condition: str, adjustment: str, horizon_scope: str) -> str:
        result = policy.graduate(
            hypothesis_id,
            condition=condition,
            adjustment=adjustment,
            horizon_scope=horizon_scope,
            rationale=f"{condition} -> {adjustment} ({horizon_scope})",
            now=ref_now,
            origin_date=origin_date,
        )
        return result.message

    def update_approach_narrative(new_text: str, rationale: str) -> str:
        candidate_id = f"cand-{policy.twin_id}-{ref_now.isoformat()}"
        result = policy.propose_behavioral(
            candidate_id=candidate_id,
            kind="approach_narrative",
            description=new_text,
            rationale=rationale,
            opened_on=ref_now,
            origin_date=origin_date,
        )
        if not result.ok:
            return result.message
        return (
            f"Approach-narrative rewrite entered the shadow gate as {candidate_id}. "
            "It runs in shadow and is adopted only if it does not underperform the "
            "current strategy over the shadow window."
        )

    return [
        record_observation,
        open_hypothesis,
        record_hypothesis_outcome,
        graduate_hypothesis,
        update_approach_narrative,
    ]


__all__ = ["build_gated_skill_tools"]
