"""Pluggable multi-turn session abstraction for the study + reflection drivers.

The drivers are structured around a :class:`StudySession` protocol —
``run_turn(prompt) -> TurnResult`` over a *sticky* conversation — so the
scheduling, checkpointing, transcript, and token accounting are all exercised
offline with a fake session, while the real
:class:`AdkStudySession` (which makes model + E2B calls) is only constructed
behind the CLI's ``--run`` flag.

Token accounting: the ADK text runner surfaces the final text, not a usage
record, so :func:`approx_tokens` gives a deterministic char/4 estimate for the
persisted accounting. When a run is executed for real, the authoritative token
and cost figures come from the per-turn Langfuse traces (each turn is a tagged
trace); the persisted estimate is a lower-effort cross-check, never the billing
source of truth.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TurnResult:
    """One session turn's output plus its (estimated) token accounting."""

    text: str
    input_tokens: int
    output_tokens: int
    wall_time_s: float


def approx_tokens(text: str) -> int:
    """Deterministic char/4 token estimate (see the module note on accounting)."""
    return max(0, len(text) // 4)


class StudySession(Protocol):
    """A sticky multi-turn conversation: successive turns share context."""

    def run_turn(self, prompt: str) -> TurnResult:
        """Send one user turn and return its result."""
        ...

    def close(self) -> None:
        """Release any underlying resources."""
        ...


class AdkStudySession:
    """Real session backed by a sticky :class:`AdkTextRunner` (spend-bearing).

    Built only under ``--run``. Uses ``fresh_session_per_message=False`` so the
    agent carries context across turns (matching the build_nb05 sticky pattern),
    with Langfuse tracing tagged per turn. Constructed lazily so importing this
    module never requires the ``agentic`` extra to be usable.
    """

    def __init__(
        self,
        agent_config: object,
        *,
        app_name: str,
        langfuse_tags: list[str] | None = None,
        trace_name: str | None = None,
    ) -> None:
        from aieng.forecasting.methods.agentic import build_adk_agent  # noqa: PLC0415
        from aieng.forecasting.methods.agentic.adk_runner import (  # noqa: PLC0415
            AdkTextRunner,
            AdkTextRunnerConfig,
        )

        self._runner = AdkTextRunner(
            build_adk_agent(agent_config),  # type: ignore[arg-type]
            config=AdkTextRunnerConfig(
                app_name=app_name,
                fresh_session_per_message=False,  # sticky study conversation
                enable_langfuse_tracing=True,
                langfuse_tags=langfuse_tags,
                langfuse_trace_name=trace_name,
            ),
        )

    def run_turn(self, prompt: str) -> TurnResult:
        """Run one turn on the sticky session (blocking wrapper around async)."""
        import asyncio  # noqa: PLC0415

        start = time.perf_counter()
        text = asyncio.run(self._runner.run_text_async(prompt))
        wall = time.perf_counter() - start
        return TurnResult(
            text=text,
            input_tokens=approx_tokens(prompt),
            output_tokens=approx_tokens(text),
            wall_time_s=wall,
        )

    def close(self) -> None:
        """Close the underlying runner."""
        import asyncio  # noqa: PLC0415

        asyncio.run(self._runner.aclose())


__all__ = ["AdkStudySession", "StudySession", "TurnResult", "approx_tokens"]
