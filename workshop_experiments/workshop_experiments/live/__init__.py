"""Live daily S&P 500 forecasting harness (workshop stage 2b).

Every trading day, shortly after the US market close, a single run predicts
S&P 500 cumulative log returns at h = 1 / 5 / 21 business days for every
configured method x model, writes schema-conforming prediction records into an
append-only log, resolves + CRPS-scores matured horizons, regenerates the
monitor aggregates, and commits + pushes. Missed days become logged gaps and are
never backfilled.

The public entry point is the ``ws-live-run`` CLI (:mod:`.cli`). The pieces are
split so each is independently testable offline:

- :mod:`.config` — load ``live_config.yaml`` and expand the ladder.
- :mod:`.records` — build/validate schema records from ``Prediction`` objects.
- :mod:`.log_store` — the append-only per-day log layout.
- :mod:`.predict` — the predict step (real + simulate prediction sources).
- :mod:`.resolve` — the resolve step (realized providers + CRPS).
- :mod:`.aggregate` — regenerate the deterministic monitor aggregates.
- :mod:`.schema_validation` — reuse the monitor schema validator.
- :mod:`.lockfile` — the single-run lock.
- :mod:`.gitops` — stage / commit / push.
"""

from __future__ import annotations


#: Data-contract schema version every written record + aggregate carries.
SCHEMA_VERSION = "1.1.0"

__all__ = ["SCHEMA_VERSION"]
