# Source: implementations/energy_oil_forecasting/adaptive_agent/skill_tools.py

kind: python

```python
"""Mutation tools for the ``wti-strategy`` adaptive skill.

Thin oil-side wrapper over the shared, domain-agnostic tool factory
:func:`aieng.forecasting.methods.agentic.adaptive_skill_tools.build_skill_tools`.
The wrapper binds the factory to :class:`WtiStrategyState` so the rendered
``SKILL.md`` keeps its WTI branding.  The five mutation tools, their evidence
governance, and the scope guard are documented on the shared module.

The sub-model re-exports below keep existing imports of ``Observation``,
``Hypothesis``, ``CalibrationCorrection``, and ``VersionEntry`` from this module
working.  There are no import-time singletons: nothing touches the filesystem
until :func:`build_skill_tools` is called.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from aieng.forecasting.methods.agentic.adaptive_skill_tools import (
    build_skill_tools as _build_skill_tools,
)
from energy_oil_forecasting.adaptive_agent.skill_state import (
    CalibrationCorrection,
    Hypothesis,
    Observation,
    VersionEntry,
    WtiStrategyState,
)


def build_skill_tools(
    strategy_dir: Path,
    *,
    confirmation_threshold: int = 3,
) -> list[Callable[..., str]]:
    """Build the five WTI strategy mutation tools bound to *strategy_dir*.

    Delegates to the shared factory with :class:`WtiStrategyState` as the state
    type.  See
    :func:`aieng.forecasting.methods.agentic.adaptive_skill_tools.build_skill_tools`
    for the full tool signatures and evidence-governance rules.

    Parameters
    ----------
    strategy_dir : Path
        Directory containing the strategy skill (``skill_state.yaml``,
        ``SKILL.md``, ``.history/``).  Must exist and be a directory.
    confirmation_threshold : int, default=3
        Number of confirming hypothesis outcomes required before
        ``graduate_hypothesis`` is permitted.

    Returns
    -------
    list[Callable[..., str]]
        ``[record_observation, open_hypothesis, record_hypothesis_outcome,
        graduate_hypothesis, update_approach_narrative]``
    """
    return _build_skill_tools(
        strategy_dir,
        WtiStrategyState,
        confirmation_threshold=confirmation_threshold,
    )


__all__ = [
    "CalibrationCorrection",
    "Hypothesis",
    "Observation",
    "VersionEntry",
    "WtiStrategyState",
    "build_skill_tools",
]
```
