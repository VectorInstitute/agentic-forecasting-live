# Source: implementations/energy_oil_forecasting/adaptive_agent/skill_state.py

kind: python

```python
"""WTI forecasting strategy state model.

Defines the structured state backing the ``wti-strategy`` adaptive skill.
``WtiStrategyState`` is a thin oil-branded subclass of the domain-agnostic
:class:`~aieng.forecasting.methods.agentic.strategy_state.StrategyState`: it
pins the rendered ``SKILL.md`` heading, default skill name, and frontmatter
description to the WTI wording so committed artifacts round-trip byte-identically.
All fields, sub-models, and rendering logic live in the shared library.

The state is persisted to ``skills/wti-strategy/skill_state.yaml`` and rendered
to ``skills/wti-strategy/SKILL.md`` on every mutation so that the ADK
``SkillToolset`` always reads an up-to-date version.  See the shared
:mod:`aieng.forecasting.methods.agentic.strategy_state` module for the
learning-layer hierarchy and evidence burdens.
"""

from __future__ import annotations

from typing import ClassVar

# Re-exported so existing imports of the sub-models from this module keep working.
from aieng.forecasting.methods.agentic.strategy_state import (
    CalibrationCorrection,
    Hypothesis,
    Observation,
    StrategyState,
    VersionEntry,
)


class WtiStrategyState(StrategyState):
    """Oil-branded strategy state for the adaptive WTI crude oil analyst.

    Identical in structure to :class:`StrategyState`; only the presentation
    strings are pinned so the ``wti-strategy`` artifacts render exactly as
    committed.
    """

    markdown_title: ClassVar[str] = "WTI Forecasting Strategy"
    default_skill_name: ClassVar[str] = "wti-strategy"
    frontmatter_description_lines: ClassVar[tuple[str, ...]] = (
        "The adaptive WTI analyst's current forecasting strategy. Load this at the",
        "start of every prediction task. This file is generated — edit the state",
        "through the mutation tools, not by hand.",
    )


__all__ = [
    "CalibrationCorrection",
    "Hypothesis",
    "Observation",
    "VersionEntry",
    "WtiStrategyState",
]
```
