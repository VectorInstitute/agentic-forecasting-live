"""Offline tests for the tiered adaptation gate (workshop stage 2c)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from workshop_experiments.adaptive.gates import (
    FrozenMutationError,
    GateConfig,
    GatePolicy,
    ShadowError,
    ShadowGate,
    evaluate_breaker,
    trailing_mean,
)
from workshop_experiments.adaptive.gates.events import MutationEventStore
from workshop_experiments.adaptive.gates.gated_tools import build_gated_skill_tools
from workshop_experiments.adaptive.reseed import reseed_strategy
from workshop_experiments.live.schema_validation import check


D = dt.date(2026, 3, 2)


def _policy(tmp_path: Path, *, twin_id: str = "adaptive_learning", read_only: bool = False, **cfg) -> GatePolicy:
    """Build a learning (or read-only) gate over a freshly reseeded strategy."""
    trained = tmp_path / "strategy"
    reseed_strategy(trained_dir=trained)
    return GatePolicy(
        strategy_dir=trained,
        log_dir=tmp_path / "log",
        twin_id=twin_id,
        config=GateConfig(**cfg),
        read_only=read_only,
    )


# ---------------------------------------------------------------------------
# Tier 2 — k-confirmation graduation
# ---------------------------------------------------------------------------


def test_graduation_requires_k_confirmations(tmp_path: Path) -> None:
    """Graduation requires k confirmations."""
    pol = _policy(tmp_path, confirmation_threshold=3)
    hyp = pol.propose_hypothesis("bands too narrow in high VIX", "3 origins", rationale="durable", occurred_on=D).ref

    early = pol.graduate(hyp, condition="VIX high", adjustment="widen 12%", horizon_scope="5bd", rationale="x", now=D)
    assert not early.ok
    assert "Cannot graduate" in early.message

    for i in range(3):
        pol.record_outcome(hyp, "confirmed", rationale=f"res {i}", occurred_on=D)
    graduated = pol.graduate(
        hyp, condition="VIX high", adjustment="widen 12%", horizon_scope="5bd", rationale="k met", now=D
    )
    assert graduated.ok
    assert graduated.event["tier"] == "behavioral"
    assert graduated.event["gate_outcome"] == "graduated"
    # The correction actually landed in the strategy state.
    assert len(pol.store.load().calibration_corrections) == 1


def test_open_hypothesis_cap(tmp_path: Path) -> None:
    """Open hypothesis cap."""
    pol = _policy(tmp_path, max_open_hypotheses=2)
    assert pol.propose_hypothesis("c1", "e1", rationale="r", occurred_on=D).ok
    assert pol.propose_hypothesis("c2", "e2", rationale="r", occurred_on=D).ok
    capped = pol.propose_hypothesis("c3", "e3", rationale="r", occurred_on=D)
    assert not capped.ok
    assert "cap reached" in capped.message


# ---------------------------------------------------------------------------
# Weekly rate limit
# ---------------------------------------------------------------------------


def test_weekly_graduation_rate_limit(tmp_path: Path) -> None:
    """Weekly graduation rate limit."""
    pol = _policy(tmp_path, confirmation_threshold=1, max_graduations_per_week=1)
    ids = []
    for n in range(2):
        hyp = pol.propose_hypothesis(f"claim {n}", "ev", rationale="r", occurred_on=D).ref
        pol.record_outcome(hyp, "confirmed", rationale="r", occurred_on=D)
        ids.append(hyp)

    first = pol.graduate(
        ids[0], condition="c", adjustment="a", horizon_scope="all", rationale="r", now=D, occurred_on=D
    )
    assert first.ok
    # Second graduation the same week is rate-limited (no event written).
    second = pol.graduate(
        ids[1], condition="c", adjustment="a", horizon_scope="all", rationale="r", now=D, occurred_on=D
    )
    assert not second.ok
    assert "rate limit" in second.message
    # A week later it is allowed.
    later_day = D + dt.timedelta(days=8)
    later = pol.graduate(
        ids[1], condition="c", adjustment="a", horizon_scope="all", rationale="r", now=later_day, occurred_on=later_day
    )
    assert later.ok


# ---------------------------------------------------------------------------
# Tier 3 — shadow adopt / reject on synthetic CRPS series
# ---------------------------------------------------------------------------


def test_shadow_adopts_when_challenger_not_worse(tmp_path: Path) -> None:
    """Shadow adopts when challenger not worse."""
    gate = ShadowGate(tmp_path / "shadow", window=3)
    gate.open_challenger(candidate_id="cand-1", kind="approach_narrative", description="rewrite", opened_on=D)
    for i, (champ, chall) in enumerate([(0.50, 0.40), (0.60, 0.55), (0.50, 0.45)]):
        gate.record_pair(origin=D + dt.timedelta(days=i), champion_crps=champ, challenger_crps=chall)
    assert gate.ready()
    decision = gate.evaluate(decided_on=D + dt.timedelta(days=4))
    assert decision.adopted
    assert decision.challenger_mean_crps < decision.champion_mean_crps
    # Archived + active slot cleared.
    assert not gate.has_open_candidate()
    assert (tmp_path / "shadow" / "archive" / "cand-1.json").exists()


def test_shadow_rejects_when_challenger_worse(tmp_path: Path) -> None:
    """Shadow rejects when challenger worse."""
    gate = ShadowGate(tmp_path / "shadow", window=2)
    gate.open_challenger(candidate_id="cand-2", kind="calibration_correction", description="widen", opened_on=D)
    gate.record_pair(origin=D, champion_crps=0.40, challenger_crps=0.60)
    gate.record_pair(origin=D + dt.timedelta(days=1), champion_crps=0.40, challenger_crps=0.55)
    decision = gate.evaluate(decided_on=D + dt.timedelta(days=2))
    assert not decision.adopted


def test_shadow_one_candidate_at_a_time(tmp_path: Path) -> None:
    """Shadow one candidate at a time."""
    gate = ShadowGate(tmp_path / "shadow", window=2)
    gate.open_challenger(candidate_id="a", kind="approach_narrative", description="x", opened_on=D)
    with pytest.raises(ShadowError):
        gate.open_challenger(candidate_id="b", kind="approach_narrative", description="y", opened_on=D)


def test_shadow_incomplete_window_cannot_evaluate(tmp_path: Path) -> None:
    """Shadow incomplete window cannot evaluate."""
    gate = ShadowGate(tmp_path / "shadow", window=3)
    gate.open_challenger(candidate_id="a", kind="approach_narrative", description="x", opened_on=D)
    gate.record_pair(origin=D, champion_crps=0.4, challenger_crps=0.3)
    with pytest.raises(ShadowError):
        gate.evaluate(decided_on=D)


def test_policy_shadow_adopt_applies_narrative(tmp_path: Path) -> None:
    """Policy shadow adopt applies narrative."""
    pol = _policy(tmp_path, shadow_window_days=2)
    pol.propose_behavioral(
        candidate_id="cand-x", kind="approach_narrative", description="new", rationale="r", opened_on=D
    )
    pol.record_shadow_pair(origin=D, champion_crps=0.5, challenger_crps=0.4)
    pol.record_shadow_pair(origin=D + dt.timedelta(days=1), champion_crps=0.5, challenger_crps=0.4)
    new_text = "A materially rewritten approach narrative that is long enough to be real."
    res = pol.resolve_shadow(
        decided_on=D + dt.timedelta(days=2),
        rationale="challenger won",
        apply=lambda: pol.update_narrative(new_text, "adopted via shadow"),
    )
    assert res.ok
    assert res.event["gate_outcome"] == "adopted"
    assert pol.store.load().approach_narrative == new_text


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


def test_trailing_mean_insufficient_history() -> None:
    """Trailing mean insufficient history."""
    assert trailing_mean([1.0, 2.0], 3) is None
    assert trailing_mean([1.0, 2.0, 3.0], 3) == 2.0


def test_breaker_trips_and_freezes(tmp_path: Path) -> None:
    """Breaker trips and freezes."""
    pol = _policy(tmp_path, circuit_breaker_window=21, circuit_breaker_ratio=1.15)
    reading = evaluate_breaker([1.2] * 21, [1.0] * 21, window=21, ratio=1.15)
    assert reading.tripped

    br = pol.check_circuit_breaker([1.2] * 21, [1.0] * 21, occurred_on=D)
    assert br.tripped
    assert pol.frozen
    # A frozen learner drops further mutations (run continues, no crash).
    assert not pol.observe("x", rationale="y", occurred_on=D).ok
    # The freeze wrote exactly one circuit-breaker event.
    events = pol.events.read_all()
    breaker_events = [e for e in events if e["gate_outcome"] == "frozen_circuit_breaker"]
    assert len(breaker_events) == 1
    # Idempotent: a second tripping check writes no further event.
    pol.check_circuit_breaker([1.2] * 21, [1.0] * 21, occurred_on=D)
    assert len([e for e in pol.events.read_all() if e["gate_outcome"] == "frozen_circuit_breaker"]) == 1


def test_breaker_does_not_trip_within_ratio(tmp_path: Path) -> None:
    """Breaker does not trip within ratio."""
    reading = evaluate_breaker([1.1] * 21, [1.0] * 21, window=21, ratio=1.15)
    assert not reading.tripped


def test_breaker_insufficient_history_never_trips() -> None:
    """Breaker insufficient history never trips."""
    reading = evaluate_breaker([5.0] * 5, [1.0] * 5, window=21, ratio=1.15)
    assert not reading.tripped
    assert reading.learner_mean is None


# ---------------------------------------------------------------------------
# Frozen twin genuinely cannot mutate
# ---------------------------------------------------------------------------


def test_frozen_twin_raises_on_every_mutation(tmp_path: Path) -> None:
    """Frozen twin raises on every mutation."""
    pol = _policy(tmp_path, twin_id="adaptive_frozen", read_only=True)
    with pytest.raises(FrozenMutationError):
        pol.observe("x", rationale="y")
    with pytest.raises(FrozenMutationError):
        pol.propose_hypothesis("c", "e", rationale="r")
    with pytest.raises(FrozenMutationError):
        pol.graduate("hyp-001", condition="c", adjustment="a", horizon_scope="all", rationale="r", now=D)
    with pytest.raises(FrozenMutationError):
        pol.update_narrative("text", "why")
    # Nothing was written to the strategy or the event log.
    assert pol.store.load().observations == []
    assert pol.events.read_all() == []


def test_frozen_twin_gated_tools_raise(tmp_path: Path) -> None:
    """Frozen twin gated tools raise."""
    pol = _policy(tmp_path, twin_id="adaptive_frozen", read_only=True)
    tools = {t.__name__: t for t in build_gated_skill_tools(pol, origin_date=D)}
    with pytest.raises(FrozenMutationError):
        tools["record_observation"]("a finding")


# ---------------------------------------------------------------------------
# mutation_event schema conformance
# ---------------------------------------------------------------------------


def test_events_conform_to_schema_and_are_persisted(tmp_path: Path) -> None:
    """Events conform to schema and are persisted."""
    pol = _policy(tmp_path, confirmation_threshold=1)
    pol.observe("pattern across origins", rationale="r", origin_date=D, occurred_on=D)
    hyp = pol.propose_hypothesis("claim", "ev", rationale="r", origin_date=D, occurred_on=D).ref
    pol.record_outcome(hyp, "confirmed", rationale="r", origin_date=D, occurred_on=D)

    events = pol.events.read_all()
    assert len(events) == 3
    for event in events:
        assert check("mutation_event", event) is None  # raises on any violation
        assert event["twin_id"] == "adaptive_learning"
    # Records live under the reserved per-day mutations/ dir.
    assert list((tmp_path / "log").glob("*/*/*/mutations/*.json"))


def test_confirmation_count_recorded_on_hypothesis_events(tmp_path: Path) -> None:
    """Confirmation count recorded on hypothesis events."""
    pol = _policy(tmp_path, confirmation_threshold=3)
    hyp = pol.propose_hypothesis("c", "e", rationale="r", occurred_on=D).ref
    out = pol.record_outcome(hyp, "confirmed", rationale="r", occurred_on=D)
    assert out.event["confirmations"] == 1


def test_event_store_read_filters_by_twin(tmp_path: Path) -> None:
    """Event store read filters by twin."""
    log = tmp_path / "log"
    learn = MutationEventStore(log, "adaptive_learning")
    frozen = MutationEventStore(log, "adaptive_frozen")
    learn.write(tier="observation", gate_outcome="appended", rationale="r", occurred_on=D)
    assert len(learn.read_all()) == 1
    assert frozen.read_all() == []
