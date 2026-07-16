"""Adaptive S&P 500 agent: study driver, live twins, and the tiered gate (2c).

This package supplies the stage-2c pieces of the workshop:

- :mod:`.domain` — ``SP500_ADAPTIVE_DOMAIN`` (SP500_DOMAIN + adaptive skill dirs),
  the branded ``Sp500StrategyState``, and the adaptive config/predictor factories.
- :mod:`.reseed` — RESEED semantics (seed → trained by copy; the seed is never
  mutated).
- :mod:`.gates` — the tiered adaptation gate (observations / hypotheses /
  behavioral shadow gate / circuit breaker) and mutation-event persistence.
- :mod:`.origins` — postmortem origin selection (worst-N by CRPS + best-N controls).
- :mod:`.study` — the Study Hall + Residency driver (net-new, non-notebook,
  resumable) behind a pluggable session so it builds and tests offline.
- :mod:`.twins` — the frozen/learning twin runtime wired into the live harness.
- :mod:`.reflection` — the post-resolution reflection step feeding the gate.
- :mod:`.evaluate` — the retrospective before/after eval wiring.
"""

from __future__ import annotations

#: Schema/data-contract version the mutation events carry (mirrors the live one).
from workshop_experiments.live import SCHEMA_VERSION


__all__ = ["SCHEMA_VERSION"]
