"""Retrospective before/after eval for the adaptive strategy (``ws-adaptive-eval``).

Compares three frozen strategy arms on the 2026 protected weekly grid, alongside
the stateless ladder, via the existing resumable runner + scorer:

- **untrained** — the seed strategy (domain priors only).
- **phase_a** — after Study Hall only.
- **phase_ab** — after Study Hall + Residency.

Each arm is a read-only adaptive predictor pointed at that arm's strategy dir
(no mutation tools — the eval is frozen). The command builds one predictor per
arm, runs it across the spec (per-origin persisted, resumable), and scores the
persisted predictions. The predictor factory is injectable so the wiring is
tested with a fake predictor; actual runs are spend-gated behind ``run=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.models import ADVANCED_MODEL

from workshop_experiments.adaptive.domain import (
    SEED_STRATEGY_DIR,
    build_sp500_adaptive_predictor,
)
from workshop_experiments.runner import DEFAULT_STORE_DIR, RunAccounting, run_predictor_on_spec
from workshop_experiments.specs import load_spec


#: Arm name -> default strategy dir. ``phase_a`` / ``phase_ab`` dirs are produced
#: by ``ws-study`` runs; callers override the mapping to point at their outputs.
DEFAULT_ARMS: dict[str, Path] = {
    "untrained": SEED_STRATEGY_DIR,
    "phase_a": SEED_STRATEGY_DIR.parent / "sp500-strategy-phase-a",
    "phase_ab": SEED_STRATEGY_DIR.parent / "sp500-strategy-trained",
}

#: Factory: ``(arm_name, strategy_dir, model) -> Predictor``.
ArmPredictorFactory = Callable[[str, Path, str], Predictor]


def _default_arm_predictor(arm: str, strategy_dir: Path, model: str) -> Predictor:
    """Build the read-only adaptive predictor for one arm (spend-bearing at run)."""
    return build_sp500_adaptive_predictor(
        strategy_dir=strategy_dir,
        model=model,
        attach_mutation_tools=False,
    )


@dataclass(frozen=True)
class AdaptiveEvalPlan:
    """The resolved plan for a before/after eval (printed by ``--dry-run``)."""

    spec_name: str
    arms: dict[str, Path]
    model: str
    store_dir: Path

    def describe(self) -> str:
        """Render the plan as a human-readable block for ``--dry-run``."""
        lines = [
            "ws-adaptive-eval plan (dry-run)",
            f"  spec      : {self.spec_name}",
            f"  model     : {self.model}",
            f"  store dir : {self.store_dir}",
            "  arms      :",
        ]
        for arm, path in self.arms.items():
            exists = "present" if (path / "skill_state.yaml").exists() else "MISSING"
            lines.append(f"    - {arm:10s} -> {path}  [{exists}]")
        lines.append("  NOTE: actual runs make model + E2B calls; pass run=True (CLI: --run).")
        return "\n".join(lines)


def plan_adaptive_eval(
    *,
    spec_name: str = "sp500_ws_eval_2026_weekly",
    arms: Mapping[str, Path] | None = None,
    model: str = ADVANCED_MODEL,
    store_dir: Path = DEFAULT_STORE_DIR,
) -> AdaptiveEvalPlan:
    """Resolve a before/after eval plan without running anything."""
    return AdaptiveEvalPlan(
        spec_name=spec_name,
        arms=dict(arms) if arms is not None else dict(DEFAULT_ARMS),
        model=model,
        store_dir=store_dir,
    )


def run_adaptive_eval(
    plan: AdaptiveEvalPlan,
    data_service: object,
    *,
    predictor_factory: ArmPredictorFactory = _default_arm_predictor,
    force_refresh: bool = False,
) -> dict[str, RunAccounting]:
    """Run every arm across the eval spec (per-origin persisted, resumable).

    Returns ``{arm_name: RunAccounting}``. Scoring the persisted predictions is a
    separate offline step (``ws-score --spec <spec_name>``), matching the rest of
    the workshop pipeline. ``predictor_factory`` is injectable for offline tests.
    """
    spec = load_spec(plan.spec_name)
    results: dict[str, RunAccounting] = {}
    for arm, strategy_dir in plan.arms.items():
        predictor = predictor_factory(arm, strategy_dir, plan.model)
        results[arm] = run_predictor_on_spec(
            predictor,
            spec,
            data_service,  # type: ignore[arg-type]
            store_dir=plan.store_dir,
            force_refresh=force_refresh,
        )
    return results


__all__ = [
    "DEFAULT_ARMS",
    "AdaptiveEvalPlan",
    "ArmPredictorFactory",
    "plan_adaptive_eval",
    "run_adaptive_eval",
]
