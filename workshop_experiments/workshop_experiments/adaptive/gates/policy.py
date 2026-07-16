"""Tiered gate policy — the layer over the strategy mutation tools.

The learning twin proposes freely but adopts through a gate. This policy wraps
the five generic mutation tools (:func:`build_skill_tools`) and the shadow gate,
enforces the §4 guardrails, and writes one ``mutation_event`` per action:

- **Tier 1 — observations** pass straight through (``appended``).
- **Tier 2 — hypotheses** use the tools' confirmation-threshold machinery: open
  (``proposed``), accumulate outcomes (``confirmed`` / ``refuted``), and graduate
  a calibration correction once ``k`` confirmations land — subject to the weekly
  rate limit on graduated behavioral changes (``graduated``).
- **Tier 3 — behavioral changes** (approach-narrative rewrites and other direct
  forecast-altering candidates) enter the shadow champion/challenger gate
  (``shadowing`` → ``adopted`` / ``rejected``).
- **Circuit breaker** — when the learner's trailing CRPS degrades past the frozen
  twin's, adaptation freezes (``frozen_circuit_breaker``) and the run continues.

The **frozen twin** uses a ``read_only`` policy: any mutation attempt is logged
and raises :class:`FrozenMutationError`. Its predictor is also built without the
mutation tools, so the read path is the only path it has.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from aieng.forecasting.methods.agentic.adaptive_skill import AdaptiveSkillStore
from aieng.forecasting.methods.agentic.adaptive_skill_tools import build_skill_tools
from aieng.forecasting.methods.agentic.strategy_state import StrategyState

from workshop_experiments.adaptive.domain import Sp500StrategyState
from workshop_experiments.adaptive.gates.circuit_breaker import BreakerReading, evaluate_breaker
from workshop_experiments.adaptive.gates.config import GateConfig
from workshop_experiments.adaptive.gates.events import MutationEventStore
from workshop_experiments.adaptive.gates.shadow import ShadowDecision, ShadowGate


logger = logging.getLogger(__name__)


class FrozenMutationError(RuntimeError):
    """Raised when a read-only (frozen twin) policy is asked to mutate."""


class GateResult:
    """The outcome of one gated action (message + optional event/id)."""

    def __init__(self, *, ok: bool, message: str, event: dict[str, Any] | None = None, ref: str | None = None) -> None:
        self.ok = ok
        self.message = message
        self.event = event
        self.ref = ref  # hypothesis id / candidate id where relevant

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        """Return a compact debugging representation."""
        return f"GateResult(ok={self.ok}, ref={self.ref!r}, message={self.message!r})"


class GatePolicy:
    """Tiered adaptation gate over a single strategy variant.

    Parameters
    ----------
    strategy_dir : Path
        The (trained) strategy directory this policy mutates.
    log_dir : Path
        Root of the live append-only log (mutation events land under it).
    twin_id : str
        The learning twin id (e.g. ``adaptive_learning``).
    config : GateConfig
        Tunable gate parameters.
    state_type : type[StrategyState]
        Strategy-state subclass (default :class:`Sp500StrategyState`).
    shadow_dir : Path or None
        Where shadow-candidate state lives; defaults to a sibling
        ``<strategy_dir>-shadow`` directory.
    read_only : bool
        Build the frozen-twin policy: every mutation attempt raises
        :class:`FrozenMutationError`.
    """

    def __init__(
        self,
        *,
        strategy_dir: Path,
        log_dir: Path,
        twin_id: str,
        config: GateConfig,
        state_type: type[StrategyState] = Sp500StrategyState,
        shadow_dir: Path | None = None,
        read_only: bool = False,
    ) -> None:
        self.strategy_dir = strategy_dir
        self.twin_id = twin_id
        self.config = config
        self.read_only = read_only
        self.frozen = False  # circuit-breaker latch
        self.store: AdaptiveSkillStore[StrategyState] = AdaptiveSkillStore(
            skill_dir=strategy_dir,
            state_type=state_type,
            confirmation_threshold=config.confirmation_threshold,
        )
        self._tools: dict[str, Callable[..., str]] = {
            t.__name__: t
            for t in build_skill_tools(strategy_dir, state_type, confirmation_threshold=config.confirmation_threshold)
        }
        self.events = MutationEventStore(log_dir, twin_id)
        self.shadow = ShadowGate(
            shadow_dir or strategy_dir.parent / f"{strategy_dir.name}-shadow",
            window=config.shadow_window_days,
        )

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------
    def _guard_writable(self, action: str) -> None:
        """Raise if the policy is read-only (frozen twin)."""
        if self.read_only:
            logger.warning("frozen twin %s: dropping mutation attempt %r", self.twin_id, action)
            raise FrozenMutationError(
                f"twin {self.twin_id!r} is read-only (frozen); mutation {action!r} is not permitted."
            )

    def _is_frozen_by_breaker(self) -> bool:
        return self.frozen

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def open_hypotheses(self) -> list[Any]:
        """Return the currently-open hypotheses."""
        return [h for h in self.store.load().hypotheses if h.status == "open"]

    def _graduations_last_week(self, now: date) -> int:
        """Count graduated behavioral changes in the trailing 7 days."""
        cutoff = now.toordinal() - 7
        count = 0
        for event in self.events.read_all():
            if event["gate_outcome"] != "graduated":
                continue
            occurred = datetime.strptime(event["occurred_at"][:10], "%Y-%m-%d").date()
            if occurred.toordinal() >= cutoff:
                count += 1
        return count

    # ------------------------------------------------------------------
    # Tier 1 — observations
    # ------------------------------------------------------------------
    def observe(
        self,
        finding: str,
        *,
        rationale: str,
        linked_hypothesis: str = "",
        origin_date: date | None = None,
        occurred_on: date | None = None,
    ) -> GateResult:
        """Record a pattern-level observation (ungated)."""
        self._guard_writable("record_observation")
        if self._is_frozen_by_breaker():
            return GateResult(ok=False, message="adaptation frozen by circuit breaker")
        message = self._tools["record_observation"](finding, linked_hypothesis)
        event = self.events.write(
            tier="observation",
            gate_outcome="appended",
            rationale=rationale,
            origin_date=origin_date,
            occurred_on=occurred_on,
        )
        return GateResult(ok=True, message=message, event=event)

    # ------------------------------------------------------------------
    # Tier 2 — hypotheses
    # ------------------------------------------------------------------
    def propose_hypothesis(
        self,
        claim: str,
        initial_evidence: str,
        *,
        rationale: str,
        origin_date: date | None = None,
        occurred_on: date | None = None,
    ) -> GateResult:
        """Open a hypothesis (tier 2), capped by ``max_open_hypotheses``."""
        self._guard_writable("open_hypothesis")
        if self._is_frozen_by_breaker():
            return GateResult(ok=False, message="adaptation frozen by circuit breaker")
        n_open = len(self.open_hypotheses())
        if n_open >= self.config.max_open_hypotheses:
            return GateResult(
                ok=False,
                message=(
                    f"open-hypothesis cap reached ({n_open}/{self.config.max_open_hypotheses}); "
                    "resolve or refute an existing hypothesis first."
                ),
            )
        message = self._tools["open_hypothesis"](claim, initial_evidence)
        hyp_id = self.store.load().hypotheses[-1].id
        event = self.events.write(
            tier="hypothesis",
            gate_outcome="proposed",
            rationale=rationale,
            origin_date=origin_date,
            occurred_on=occurred_on,
        )
        return GateResult(ok=True, message=message, event=event, ref=hyp_id)

    def record_outcome(
        self,
        hypothesis_id: str,
        outcome: str,
        *,
        rationale: str,
        origin_date: date | None = None,
        occurred_on: date | None = None,
    ) -> GateResult:
        """Record a confirming/refuting live resolution for a hypothesis (tier 2)."""
        self._guard_writable("record_hypothesis_outcome")
        if self._is_frozen_by_breaker():
            return GateResult(ok=False, message="adaptation frozen by circuit breaker")
        if outcome not in ("confirmed", "refuted"):
            return GateResult(ok=False, message=f"invalid outcome {outcome!r}")
        message = self._tools["record_hypothesis_outcome"](hypothesis_id, outcome)
        hyp = next((h for h in self.store.load().hypotheses if h.id == hypothesis_id), None)
        confirmations = hyp.confirmations if hyp is not None else 0
        event = self.events.write(
            tier="hypothesis",
            gate_outcome="confirmed" if outcome == "confirmed" else "refuted",
            rationale=rationale,
            origin_date=origin_date,
            occurred_on=occurred_on,
            confirmations=confirmations,
        )
        return GateResult(ok=True, message=message, event=event, ref=hypothesis_id)

    def graduate(
        self,
        hypothesis_id: str,
        *,
        condition: str,
        adjustment: str,
        horizon_scope: str,
        rationale: str,
        now: date,
        origin_date: date | None = None,
        occurred_on: date | None = None,
    ) -> GateResult:
        """Graduate a confirmed hypothesis into a calibration correction (tier 2).

        Enforces the confirmation threshold ``k`` (via the underlying tool) and
        the weekly rate limit on graduated behavioral changes. A rejected
        attempt writes **no** event; a successful graduation writes a
        ``graduated`` behavioral event.
        """
        self._guard_writable("graduate_hypothesis")
        if self._is_frozen_by_breaker():
            return GateResult(ok=False, message="adaptation frozen by circuit breaker")

        recent = self._graduations_last_week(now)
        if recent >= self.config.max_graduations_per_week:
            return GateResult(
                ok=False,
                message=(
                    f"weekly graduation rate limit reached ({recent}/"
                    f"{self.config.max_graduations_per_week} in the trailing 7 days); "
                    "defer this graduation."
                ),
            )

        message = self._tools["graduate_hypothesis"](hypothesis_id, condition, adjustment, horizon_scope)
        if message.startswith("Cannot graduate"):
            # k not met — the tool rejected it; surface, no event.
            return GateResult(ok=False, message=message, ref=hypothesis_id)

        event = self.events.write(
            tier="behavioral",
            gate_outcome="graduated",
            rationale=rationale,
            origin_date=origin_date,
            occurred_on=occurred_on,
        )
        return GateResult(ok=True, message=message, event=event, ref=hypothesis_id)

    # ------------------------------------------------------------------
    # Tier 3 — behavioral shadow gate
    # ------------------------------------------------------------------
    def propose_behavioral(
        self,
        *,
        candidate_id: str,
        kind: str,
        description: str,
        rationale: str,
        opened_on: date,
        origin_date: date | None = None,
    ) -> GateResult:
        """Open a tier-3 shadow challenger (one candidate at a time)."""
        self._guard_writable("propose_behavioral")
        if self._is_frozen_by_breaker():
            return GateResult(ok=False, message="adaptation frozen by circuit breaker")
        if self.shadow.has_open_candidate():
            return GateResult(ok=False, message="a shadow candidate is already open (bounded to one at a time)")
        self.shadow.open_challenger(candidate_id=candidate_id, kind=kind, description=description, opened_on=opened_on)
        event = self.events.write(
            tier="behavioral",
            gate_outcome="shadowing",
            rationale=rationale,
            origin_date=origin_date,
            occurred_on=opened_on,
        )
        return GateResult(ok=True, message=f"shadow candidate {candidate_id} opened", event=event, ref=candidate_id)

    def record_shadow_pair(self, *, origin: date, champion_crps: float, challenger_crps: float) -> None:
        """Append one resolved origin's champion/challenger CRPS pair to the shadow."""
        self._guard_writable("record_shadow_pair")
        self.shadow.record_pair(origin=origin, champion_crps=champion_crps, challenger_crps=challenger_crps)

    def resolve_shadow(
        self,
        *,
        decided_on: date,
        rationale: str,
        apply: Callable[[], str] | None = None,
        origin_date: date | None = None,
    ) -> GateResult:
        """Decide a completed shadow window; adopt or reject and write the event.

        When adopted, *apply* (if given) performs the actual strategy mutation
        (e.g. ``update_approach_narrative``); its message is folded into the
        result. When rejected, the candidate is archived with the comparison
        stats and nothing is applied.
        """
        self._guard_writable("resolve_shadow")
        if self._is_frozen_by_breaker():
            return GateResult(ok=False, message="adaptation frozen by circuit breaker")
        decision: ShadowDecision = self.shadow.evaluate(decided_on=decided_on)
        applied_msg = ""
        if decision.adopted and apply is not None:
            applied_msg = " " + apply()
        event = self.events.write(
            tier="behavioral",
            gate_outcome="adopted" if decision.adopted else "rejected",
            rationale=f"{rationale} — {decision.summary}",
            origin_date=origin_date,
            occurred_on=decided_on,
        )
        return GateResult(
            ok=decision.adopted,
            message=decision.summary + applied_msg,
            event=event,
            ref=decision.candidate_id,
        )

    def update_narrative(self, new_text: str, rationale: str) -> str:
        """Apply an approach-narrative rewrite through the underlying tool.

        Intended to be passed as the ``apply`` callback to
        :meth:`resolve_shadow` after a challenger is adopted.
        """
        self._guard_writable("update_approach_narrative")
        return self._tools["update_approach_narrative"](new_text, rationale)

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------
    def check_circuit_breaker(
        self,
        learner_crps: list[float],
        frozen_crps: list[float],
        *,
        rationale: str = "circuit-breaker check",
        origin_date: date | None = None,
        occurred_on: date | None = None,
    ) -> BreakerReading:
        """Evaluate the breaker; freeze adaptation and write an event if it trips.

        Idempotent: once frozen, a subsequent tripping check writes no further
        event. Never raises for the learning twin — freezing is a normal outcome
        that lets the run continue with the learner frozen.
        """
        reading = evaluate_breaker(
            learner_crps,
            frozen_crps,
            window=self.config.circuit_breaker_window,
            ratio=self.config.circuit_breaker_ratio,
        )
        if reading.tripped and not self.frozen:
            self.frozen = True
            self.events.write(
                tier="behavioral",
                gate_outcome="frozen_circuit_breaker",
                rationale=f"{rationale}: {reading.summary}",
                origin_date=origin_date,
                occurred_on=occurred_on,
            )
        return reading


__all__ = ["FrozenMutationError", "GatePolicy", "GateResult"]
