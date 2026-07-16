"""Generic adaptive forecasting-strategy state.

``StrategyState`` is a concrete ``AdaptiveSkillState`` that models a
*learnable forecasting strategy* — the living approach an adaptive
agent refines across invocations.  It captures four learning layers with
distinct evidence burdens:

``observations``
    Append-only log of pattern-level findings.  Lowest evidence bar — record
    any finding that is not a single-outlier surprise.

``hypotheses``
    Candidate systematic corrections under active testing.  Accumulate
    confirmation / refutation counts across resolutions.  A hypothesis
    graduates to a calibration correction when its confirmation count reaches
    the store's ``confirmation_threshold``.

``calibration_corrections``
    Confirmed systematic adjustments applied at prediction time.  Each entry is
    graduated from a confirmed hypothesis — never added directly.

``approach_narrative``
    Free-text description of the agent's overall forecasting philosophy.
    Highest evidence bar.

The rendered ``SKILL.md`` layout is domain-agnostic; the three domain-specific
strings — the markdown heading, the default skill name, and the frontmatter
description — are ``ClassVar`` parameters.  Subclass ``StrategyState`` and
override those class variables to brand the skill for a specific domain (see
:class:`~energy_oil_forecasting.adaptive_agent.skill_state.WtiStrategyState`).
"""

from __future__ import annotations

from typing import ClassVar, Literal

from aieng.forecasting.methods.agentic.adaptive_skill import AdaptiveSkillState
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """A single pattern-level finding from a resolution or self-review."""

    date: str
    finding: str
    linked_hypothesis: str | None = None


class Hypothesis(BaseModel):
    """A candidate systematic correction under active testing.

    ``status`` progresses through ``open`` → ``confirmed`` or ``open`` →
    ``refuted``.  Confirmed hypotheses are graduated to
    :class:`CalibrationCorrection` via the ``graduate_hypothesis`` tool.
    """

    id: str
    claim: str
    status: Literal["open", "confirmed", "refuted"] = "open"
    confirmations: int = 0
    refutations: int = 0
    opened_on: str


class CalibrationCorrection(BaseModel):
    """A confirmed systematic adjustment applied at prediction time.

    Every entry here was graduated from a confirmed hypothesis; the
    ``source_hypothesis`` field preserves that lineage.
    """

    condition: str
    adjustment: str
    horizon_scope: str
    source_hypothesis: str
    confirmed_on: str


class VersionEntry(BaseModel):
    """One row in the version history table."""

    date: str
    description: str


# ---------------------------------------------------------------------------
# Strategy state
# ---------------------------------------------------------------------------


class StrategyState(AdaptiveSkillState):
    """Domain-agnostic structured state for an adaptive forecasting strategy.

    See the module docstring for the learning-layer hierarchy and evidence
    burdens.  The three ``ClassVar`` parameters below carry the only
    domain-specific presentation strings; override them in a subclass to brand
    the rendered ``SKILL.md`` for a particular target series.
    """

    #: Heading rendered as ``# {markdown_title}`` at the top of the body.
    markdown_title: ClassVar[str] = "Forecasting Strategy"
    #: Frontmatter ``name:`` used when the store does not pass an explicit name.
    default_skill_name: ClassVar[str] = "strategy"
    #: Content lines for the frontmatter ``description: >-`` block (rendered
    #: with a two-space indent, one line per entry).
    frontmatter_description_lines: ClassVar[tuple[str, ...]] = (
        "The adaptive analyst's current forecasting strategy. Load this at the",
        "start of every prediction task. This file is generated — edit the state",
        "through the mutation tools, not by hand.",
    )

    approach_narrative: str
    calibration_corrections: list[CalibrationCorrection] = []
    hypotheses: list[Hypothesis] = []
    observations: list[Observation] = []
    version_history: list[VersionEntry] = []

    def build_markdown(self, skill_name: str | None = None) -> str:  # noqa: PLR0912
        """Render the full ``SKILL.md`` content from current state."""
        lines: list[str] = []

        # Frontmatter — skill_name must match the containing dir name (ADK requires)
        lines += [
            "---",
            f"name: {skill_name or self.default_skill_name}",
            "description: >-",
        ]
        lines += [f"  {line}" for line in self.frontmatter_description_lines]
        lines += [
            "---",
            "",
        ]

        lines += [
            f"# {self.markdown_title}",
            "",
            "## Approach",
            "",
            self.approach_narrative.strip(),
            "",
        ]

        # Active calibration corrections
        lines += [
            "## Active calibration corrections",
            "",
        ]
        if self.calibration_corrections:
            lines += [
                "| Condition | Adjustment | Horizon scope | Confirmed on |",
                "|-----------|-----------|---------------|--------------|",
            ]
            for c in self.calibration_corrections:
                lines.append(f"| {c.condition} | {c.adjustment} | {c.horizon_scope} | {c.confirmed_on} |")
        else:
            lines.append("*(No calibration corrections yet. Graduate a confirmed hypothesis to add one.)*")
        lines.append("")

        # Open hypotheses
        lines += [
            "## Open hypotheses",
            "",
        ]
        open_hyps = [h for h in self.hypotheses if h.status == "open"]
        if open_hyps:
            lines += [
                "| ID | Claim | Confirmations | Refutations |",
                "|----|-------|---------------|-------------|",
            ]
            for h in open_hyps:
                lines.append(f"| {h.id} | {h.claim} | {h.confirmations} | {h.refutations} |")
        else:
            lines.append("*(No open hypotheses.)*")
        lines.append("")

        # Closed hypotheses (confirmed / refuted) — collapsed for readability
        closed_hyps = [h for h in self.hypotheses if h.status != "open"]
        if closed_hyps:
            lines += [
                "## Closed hypotheses",
                "",
                "| ID | Claim | Status | Confirmations | Refutations |",
                "|----|-------|--------|---------------|-------------|",
            ]
            for h in closed_hyps:
                lines.append(f"| {h.id} | {h.claim} | {h.status} | {h.confirmations} | {h.refutations} |")
            lines.append("")

        # Observations
        lines += [
            "## Observations",
            "",
        ]
        if self.observations:
            lines += [
                "| Date | Finding | Linked hypothesis |",
                "|------|---------|-------------------|",
            ]
            for o in self.observations:
                linked = o.linked_hypothesis or "—"
                lines.append(f"| {o.date} | {o.finding} | {linked} |")
        else:
            lines.append("*(No observations yet. Record findings from resolutions and self-reviews.)*")
        lines.append("")

        # Version history
        lines += [
            "## Version history",
            "",
            "| Date | Change |",
            "|------|--------|",
        ]
        for v in self.version_history:
            lines.append(f"| {v.date} | {v.description} |")
        lines.append("")

        return "\n".join(lines)
