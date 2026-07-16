"""Workshop-paper and live-evaluation S&P 500 forecasting experiments.

This package is the foundation for the workshop paper's S&P 500 experiment and
the forthcoming live daily-forecasting harness. It is **not** bootcamp /
participant material — there are no notebooks. Everything runs as an importable
package plus resumable CLI runners.

Public surface:

- :data:`workshop_experiments.domain.SP500_DOMAIN` — the equity-index
  :class:`~aieng.forecasting.methods.agentic.domain.DomainConfig`.
- :mod:`workshop_experiments.registry` — named predictor factories
  (conventional, LLMP ± covariates, analyst / code-executing agents).
- :mod:`workshop_experiments.runner` — a per-origin persisting, resumable
  backtest runner with token/cost accounting.
- :mod:`workshop_experiments.scoring` — leaderboard/analysis frames from
  persisted predictions.
- :mod:`workshop_experiments.specs` (YAML files) — smoke, weekly backtest
  (2025), weekly protected eval (2026), and daily bonus-layer specs.
"""

from __future__ import annotations


__all__ = ["__version__"]

__version__ = "0.1.0"
