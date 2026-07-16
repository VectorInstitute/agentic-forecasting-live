"""Retrospective before/after eval for the adaptive strategy (``ws-adaptive-eval``).

Compares frozen strategy arms on the 2026 protected weekly grid via the existing
resumable runner + scorer. The arms differ by domain:

- **S&P 500** (the stage-2c three-arm shape): ``untrained`` (seed) /
  ``phase_a`` (after Study Hall) / ``phase_ab`` (after Study Hall + Residency).
- **S&P/TSX Composite** (the bootcamp pre/post shape): two arms — ``untrained``
  (the seed strategy, domain priors only) vs ``trained`` (after the
  single-session bootcamp study).

Each arm is a read-only adaptive predictor pointed at that arm's strategy dir
(no mutation tools — the eval is frozen). The command builds one predictor per
arm, runs it across the spec (per-origin persisted, resumable), and scores the
persisted predictions separately (``ws-score``). The per-arm predictor ids are
domain- and arm-distinct (via the ``{domain}_adaptive_analyst_{strategy_dir}``
agent name), so caches never collide. The predictor factory is injectable so the
wiring is tested with a fake predictor; actual runs are spend-gated behind
``run=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.models import ADVANCED_MODEL

from workshop_experiments.runner import DEFAULT_STORE_DIR, RunAccounting, run_predictor_on_spec
from workshop_experiments.specs import load_spec


#: Factory: ``(arm_name, strategy_dir, model) -> Predictor``.
ArmPredictorFactory = Callable[[str, Path, str], Predictor]


def _sp500_default_arms() -> dict[str, Path]:
    """S&P 500 arm name -> default strategy dir (three-arm 2c shape)."""
    from workshop_experiments.adaptive.domain import SEED_STRATEGY_DIR  # noqa: PLC0415

    return {
        "untrained": SEED_STRATEGY_DIR,
        "phase_a": SEED_STRATEGY_DIR.parent / "sp500-strategy-phase-a",
        "phase_ab": SEED_STRATEGY_DIR.parent / "sp500-strategy-trained",
    }


def _tsx_default_arms() -> dict[str, Path]:
    """TSX arm name -> default strategy dir (two-arm pre/post shape)."""
    from workshop_experiments.adaptive.domain_tsx import (  # noqa: PLC0415
        TSX_SEED_STRATEGY_DIR,
        TSX_TRAINED_STRATEGY_DIR,
    )

    return {"untrained": TSX_SEED_STRATEGY_DIR, "trained": TSX_TRAINED_STRATEGY_DIR}


def _sp500_arm_predictor(arm: str, strategy_dir: Path, model: str) -> Predictor:
    """Build the read-only S&P 500 adaptive predictor for one arm (spends at run)."""
    from workshop_experiments.adaptive.domain import build_sp500_adaptive_predictor  # noqa: PLC0415

    return build_sp500_adaptive_predictor(strategy_dir=strategy_dir, model=model, attach_mutation_tools=False)


def _tsx_arm_predictor(arm: str, strategy_dir: Path, model: str) -> Predictor:
    """Build the read-only TSX adaptive predictor for one arm (spend-bearing at run)."""
    from workshop_experiments.adaptive.domain_tsx import build_tsx_adaptive_predictor  # noqa: PLC0415

    return build_tsx_adaptive_predictor(strategy_dir=strategy_dir, model=model, attach_mutation_tools=False)


@dataclass(frozen=True)
class _DomainEval:
    """Per-domain eval defaults (arms, eval spec, predictor factory)."""

    default_spec: str
    arms_factory: Callable[[], dict[str, Path]]
    predictor_factory: ArmPredictorFactory


_DOMAINS: dict[str, _DomainEval] = {
    "sp500": _DomainEval("sp500_ws_eval_2026_weekly", _sp500_default_arms, _sp500_arm_predictor),
    "tsx": _DomainEval("tsx_ws_eval_2026_weekly", _tsx_default_arms, _tsx_arm_predictor),
}

#: Backward-compatible S&P 500 default arm map (the original three-arm shape).
DEFAULT_ARMS: dict[str, Path] = _sp500_default_arms()


@dataclass(frozen=True)
class AdaptiveEvalPlan:
    """The resolved plan for a before/after eval (printed by ``--dry-run``)."""

    spec_name: str
    arms: dict[str, Path]
    model: str
    store_dir: Path
    domain: str = "sp500"

    def describe(self) -> str:
        """Render the plan as a human-readable block for ``--dry-run``."""
        lines = [
            "ws-adaptive-eval plan (dry-run)",
            f"  domain    : {self.domain}",
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
    domain: str = "sp500",
    spec_name: str | None = None,
    arms: Mapping[str, Path] | None = None,
    model: str = ADVANCED_MODEL,
    store_dir: Path = DEFAULT_STORE_DIR,
) -> AdaptiveEvalPlan:
    """Resolve a before/after eval plan without running anything.

    ``domain`` selects the default arm set and eval spec (``sp500`` = the three-arm
    2c shape; ``tsx`` = the two-arm pre/post shape). ``spec_name`` / ``arms``
    override the domain defaults when given.
    """
    if domain not in _DOMAINS:
        raise SystemExit(f"unknown --domain {domain!r}; choose from {sorted(_DOMAINS)}")
    dom = _DOMAINS[domain]
    return AdaptiveEvalPlan(
        spec_name=spec_name or dom.default_spec,
        arms=dict(arms) if arms is not None else dom.arms_factory(),
        model=model,
        store_dir=store_dir,
        domain=domain,
    )


def run_adaptive_eval(
    plan: AdaptiveEvalPlan,
    data_service: object,
    *,
    predictor_factory: ArmPredictorFactory | None = None,
    force_refresh: bool = False,
) -> dict[str, RunAccounting]:
    """Run every arm across the eval spec (per-origin persisted, resumable).

    Returns ``{arm_name: RunAccounting}``. Scoring the persisted predictions is a
    separate offline step (``ws-score --spec <spec_name>``), matching the rest of
    the workshop pipeline. ``predictor_factory`` is injectable for offline tests;
    when ``None`` it defaults to the plan domain's real (spend-bearing) factory.
    """
    factory = predictor_factory or _DOMAINS[plan.domain].predictor_factory
    spec = load_spec(plan.spec_name)
    results: dict[str, RunAccounting] = {}
    for arm, strategy_dir in plan.arms.items():
        predictor = factory(arm, strategy_dir, plan.model)
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
