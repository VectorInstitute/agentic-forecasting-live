"""Live twins runtime: frozen + learning arms over one adaptive strategy.

Both twins are seeded — at twin-deployment day — from the *same* trained strategy
(the Study Hall + Residency output). The only degree of freedom between them is
experience:

- **frozen twin** (``adaptive_frozen``) — reads its own snapshot dir, which is
  never mutated. Its predictor and gate are both read-only. It is the control.
- **learning twin** (``adaptive_learning``) — reads an evolving dir that the
  reflection step mutates through the tiered :class:`GatePolicy`.

Prediction never mutates (both twin predictors are built without mutation tools);
adaptation happens only in the out-of-band reflection step, so a daily forecast
can never silently rewrite the strategy. :func:`deploy_twins` performs the
twin-deployment reseed (a first-class, dated event); ``build_twin_predictor``
turns a rung into a runnable predictor (the spend-bearing real-run path).
"""

from __future__ import annotations

from pathlib import Path

from aieng.forecasting.methods.agentic import AgentPredictor

from workshop_experiments.adaptive.domain import (
    SKILLS_ROOT,
    TRAINED_STRATEGY_DIR,
    build_sp500_adaptive_predictor,
)
from workshop_experiments.adaptive.gates import GateConfig, GatePolicy
from workshop_experiments.adaptive.reseed import reseed_strategy
from workshop_experiments.live.config import LiveConfig, LivePredictor


#: The frozen twin's read-only strategy snapshot (never mutated after deploy).
FROZEN_STRATEGY_DIR = SKILLS_ROOT / "sp500-strategy-frozen"
#: The learning twin's evolving strategy dir (mutated by reflection + gates).
LEARNING_STRATEGY_DIR = SKILLS_ROOT / "sp500-strategy-learning"


def strategy_dir_for_twin(twin_id: str) -> Path:
    """Return the strategy directory a twin reads."""
    if twin_id == "adaptive_frozen":
        return FROZEN_STRATEGY_DIR
    if twin_id == "adaptive_learning":
        return LEARNING_STRATEGY_DIR
    raise ValueError(f"unknown twin_id {twin_id!r}")


def deploy_twins(
    *,
    source: Path = TRAINED_STRATEGY_DIR,
    force: bool = False,
) -> tuple[Path, Path]:
    """Seed both twins from *source* (the trained study output); RESEED semantics.

    Returns ``(frozen_dir, learning_dir)``. Both start identical to the trained
    strategy; from here they diverge only through the learning twin's gated
    reflection. The source is never mutated. This is the twin-deployment event —
    record its date as a first-class fact.
    """
    frozen = reseed_strategy(seed_dir=source, trained_dir=FROZEN_STRATEGY_DIR, force=force)
    learning = reseed_strategy(seed_dir=source, trained_dir=LEARNING_STRATEGY_DIR, force=force)
    return frozen, learning


def build_twin_predictor(
    twin: LivePredictor,
    *,
    strategy_dir: Path | None = None,
    model: str | None = None,
) -> AgentPredictor:
    """Build the runnable (read-only) :class:`AgentPredictor` for a twin rung.

    Both twins predict without mutation tools — a daily forecast never rewrites
    the strategy. Adaptation for the learning twin happens only in the reflection
    step (see :mod:`workshop_experiments.adaptive.reflection`).

    This is the real-run path and constructs the ADK agent (it needs the strategy
    dir to exist). Offline tests exercise the gate, config expansion, and
    read-only enforcement without building the agent.
    """
    if twin.twin_id not in ("adaptive_frozen", "adaptive_learning"):
        raise ValueError(f"not a twin rung: {twin.predictor_id} (twin_id={twin.twin_id!r})")
    return build_sp500_adaptive_predictor(
        strategy_dir=strategy_dir or strategy_dir_for_twin(twin.twin_id),
        model=model or twin.model or twin.model_label or "gemini-3.5-flash",
        attach_mutation_tools=False,
    )


def build_learning_gate(
    config: LiveConfig,
    *,
    strategy_dir: Path | None = None,
    log_dir: Path | None = None,
    shadow_dir: Path | None = None,
) -> GatePolicy:
    """Build the learning twin's tiered gate from the config's ``gates:`` block."""
    return GatePolicy(
        strategy_dir=strategy_dir or LEARNING_STRATEGY_DIR,
        log_dir=log_dir or config.log_dir,
        twin_id="adaptive_learning",
        config=GateConfig.from_mapping(config.gate_params),
        shadow_dir=shadow_dir,
        read_only=False,
    )


def build_frozen_gate(
    config: LiveConfig,
    *,
    strategy_dir: Path | None = None,
    log_dir: Path | None = None,
) -> GatePolicy:
    """Build the frozen twin's read-only gate (any mutation attempt raises)."""
    return GatePolicy(
        strategy_dir=strategy_dir or FROZEN_STRATEGY_DIR,
        log_dir=log_dir or config.log_dir,
        twin_id="adaptive_frozen",
        config=GateConfig.from_mapping(config.gate_params),
        read_only=True,
    )


__all__ = [
    "FROZEN_STRATEGY_DIR",
    "LEARNING_STRATEGY_DIR",
    "build_frozen_gate",
    "build_learning_gate",
    "build_twin_predictor",
    "deploy_twins",
    "strategy_dir_for_twin",
]
