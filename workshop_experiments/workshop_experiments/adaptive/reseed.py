"""RESEED semantics: copy the read-only seed strategy into a trained variant.

Both study phases and the live learning twin write into a *trained* strategy
directory that is **seeded by copy** from the committed seed — the seed itself is
never mutated. :func:`reseed_strategy` performs that copy deterministically:

1. Create the destination directory if needed.
2. Copy the seed ``skill_state.yaml`` into it (the source of truth).
3. Re-render ``SKILL.md`` from that state through the store (so the destination
   is a valid, load-able skill dir with matching frontmatter ``name:``).

It never copies the seed's ``.history/`` backups, and it refuses to overwrite an
existing populated trained dir unless ``force=True`` — so a resumed run does not
silently discard accumulated learning.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from aieng.forecasting.methods.agentic.adaptive_skill import AdaptiveSkillStore

from workshop_experiments.adaptive.domain import (
    SEED_STRATEGY_DIR,
    TRAINED_STRATEGY_DIR,
    Sp500StrategyState,
)


_STATE_FILENAME = "skill_state.yaml"


class ReseedError(RuntimeError):
    """Raised when a reseed would clobber existing trained state without force."""


def reseed_strategy(
    *,
    seed_dir: Path = SEED_STRATEGY_DIR,
    trained_dir: Path = TRAINED_STRATEGY_DIR,
    force: bool = False,
) -> Path:
    """Seed *trained_dir* from *seed_dir* by copy; return the trained dir.

    Parameters
    ----------
    seed_dir : Path
        The read-only seed strategy directory (must contain ``skill_state.yaml``).
    trained_dir : Path
        Destination trained variant directory (created if missing).
    force : bool, default=False
        Overwrite an existing populated ``trained_dir``. Without it, a
        ``trained_dir`` that already holds ``skill_state.yaml`` raises
        :class:`ReseedError` (protecting accumulated learning on resume).

    Raises
    ------
    FileNotFoundError
        If the seed ``skill_state.yaml`` is missing.
    ReseedError
        If ``trained_dir`` is already populated and ``force`` is ``False``.
    """
    seed_state = seed_dir / _STATE_FILENAME
    if not seed_state.exists():
        raise FileNotFoundError(f"seed strategy has no {_STATE_FILENAME}: {seed_dir}")
    if seed_dir.resolve() == trained_dir.resolve():
        raise ReseedError("refusing to reseed a strategy onto itself (seed must never be mutated)")

    trained_dir.mkdir(parents=True, exist_ok=True)
    if (trained_dir / _STATE_FILENAME).exists() and not force:
        raise ReseedError(f"{trained_dir} already holds {_STATE_FILENAME}; pass force=True to reset it to the seed.")

    shutil.copy2(seed_state, trained_dir / _STATE_FILENAME)
    # Re-render SKILL.md from the copied state so the dir is a valid skill.
    store = AdaptiveSkillStore(skill_dir=trained_dir, state_type=Sp500StrategyState)
    store.save(store.load())
    return trained_dir


def is_seeded(trained_dir: Path = TRAINED_STRATEGY_DIR) -> bool:
    """Return whether *trained_dir* already holds a strategy state."""
    return (trained_dir / _STATE_FILENAME).exists()


__all__ = ["ReseedError", "is_seeded", "reseed_strategy"]
