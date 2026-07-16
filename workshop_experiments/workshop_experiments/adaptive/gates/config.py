"""Gate parameters — config, not constants (pending Ethan's sign-off).

Every threshold in the tiered gate is a field here so it can be tuned from
``live_config.yaml`` without touching code. Defaults are the provisional §4
values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GateConfig:
    """Tunable parameters for the tiered strategy-adaptation gate.

    Parameters
    ----------
    confirmation_threshold : int
        ``k`` — confirming live resolutions a hypothesis needs before
        ``graduate_hypothesis`` is permitted (tier 2).
    shadow_window_days : int
        ``M`` — trading days a tier-3 behavioral candidate runs in shadow before
        the champion/challenger decision.
    circuit_breaker_window : int
        Trailing window (trading days) for the learner-vs-frozen CRPS comparison.
    circuit_breaker_ratio : float
        The learner's trailing mean CRPS may not exceed this multiple of the
        frozen twin's before adaptation freezes.
    max_graduations_per_week : int
        Rate limit on graduated behavioral changes (rolling 7 days).
    max_open_hypotheses : int
        Cap on simultaneously-open hypotheses.
    """

    confirmation_threshold: int = 3
    shadow_window_days: int = 10
    circuit_breaker_window: int = 21
    circuit_breaker_ratio: float = 1.15
    max_graduations_per_week: int = 1
    max_open_hypotheses: int = 5

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> GateConfig:
        """Build a :class:`GateConfig` from a config mapping (unknown keys ignored)."""
        if not raw:
            return cls()
        known = {
            "confirmation_threshold",
            "shadow_window_days",
            "circuit_breaker_window",
            "circuit_breaker_ratio",
            "max_graduations_per_week",
            "max_open_hypotheses",
        }
        kwargs = {k: raw[k] for k in known if k in raw}
        return cls(**kwargs)


__all__ = ["GateConfig"]
