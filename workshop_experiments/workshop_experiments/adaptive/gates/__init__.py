"""Tiered strategy-adaptation gate for the live learning twin (workshop 2c).

Public surface:

- :class:`GateConfig` — tunable §4 parameters (k, M, breaker window/ratio, rate
  limits).
- :class:`GatePolicy` — the state machine over the mutation tools + shadow gate.
- :class:`ShadowGate` — the tier-3 champion/challenger lifecycle.
- :func:`evaluate_breaker` — the circuit-breaker comparison.
- :class:`MutationEventStore` — schema-conforming mutation-event persistence.
"""

from __future__ import annotations

from workshop_experiments.adaptive.gates.circuit_breaker import (
    BreakerReading,
    evaluate_breaker,
    trailing_mean,
)
from workshop_experiments.adaptive.gates.config import GateConfig
from workshop_experiments.adaptive.gates.events import (
    MutationEventStore,
    iter_mutation_events,
)
from workshop_experiments.adaptive.gates.policy import (
    FrozenMutationError,
    GatePolicy,
    GateResult,
)
from workshop_experiments.adaptive.gates.shadow import (
    ShadowCandidate,
    ShadowDecision,
    ShadowError,
    ShadowGate,
)


__all__ = [
    "BreakerReading",
    "FrozenMutationError",
    "GateConfig",
    "GatePolicy",
    "GateResult",
    "MutationEventStore",
    "ShadowCandidate",
    "ShadowDecision",
    "ShadowError",
    "ShadowGate",
    "evaluate_breaker",
    "iter_mutation_events",
    "trailing_mean",
]
