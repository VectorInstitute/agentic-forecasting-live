"""``ws-scenario`` — Track-2 qualitative scenario analysis for the TSX.

For each configured origin, runs the TSX news-analyst agent *without* an
output schema (see :mod:`workshop_experiments.scenario_domain_tsx`: Track 2 is
a free-text scenario write-up, not a quantile forecast) and persists the
write-up plus run metadata under
``data/scenarios/<origin>/`` (see :mod:`workshop_experiments.scenario_store`).

Grading the write-up against what actually happened is a separate, later step
(``ws-scenario-judge``) — this CLI only produces and persists the write-up.

**Dry-run by default**; pass ``--run`` to make model + web-search calls. An
origin whose write-up is already persisted is skipped on the next run (resume
semantics, matching :mod:`workshop_experiments.runner`) unless ``--force`` is
given.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from aieng.forecasting.methods.agentic.agent_factory import AgentConfig
from aieng.forecasting.models import LITE_MODEL

from workshop_experiments.scenario_domain_tsx import build_scenario_prompt, build_tsx_scenario_config
from workshop_experiments.scenario_store import (
    DEFAULT_SCENARIO_STORE_DIR,
    ScenarioWriteup,
    has_writeup,
    write_scenario_writeup,
)


#: Landmark origin set (config/CLI default) — two spring-2025 origins and two
#: early-2026 origins, spanning the live-eval window.
DEFAULT_ORIGINS: tuple[str, ...] = ("2025-04-01", "2025-04-08", "2026-02-25", "2026-03-31")


@dataclass(frozen=True)
class ScenarioRunResult:
    """One Track-2 agent run: the write-up text plus its provenance."""

    markdown: str
    agent_name: str
    model: str
    trace_id: str | None
    trace_url: str | None


#: Injection seam for the real model call: ``(config, prompt, as_of) -> result``.
Track2Call = Callable[[AgentConfig, str, str], ScenarioRunResult]


@dataclass(frozen=True)
class ScenarioPlan:
    """The resolved plan for a ``ws-scenario`` run (printed by the dry-run default)."""

    origins: tuple[date, ...]
    model: str
    search_model: str
    store_dir: Path

    def describe(self) -> str:
        """Render the plan as a human-readable block for the dry-run default."""
        lines = [
            "ws-scenario plan (dry-run)",
            "  agent        : tsx_analyst_scenario (Track 2, no output schema)",
            f"  model        : {self.model}",
            f"  search model : {self.search_model}",
            f"  store dir    : {self.store_dir}",
            "  origins      :",
        ]
        for origin in self.origins:
            status = "already persisted" if has_writeup(origin, self.store_dir) else "pending"
            lines.append(f"    - {origin.isoformat()}: {status}")
        lines.append("  NOTE: --run makes model + web-search calls; dry-run (default) makes none.")
        return "\n".join(lines)


def plan_scenario_run(
    *,
    origins: Sequence[str] | None = None,
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    store_dir: Path = DEFAULT_SCENARIO_STORE_DIR,
) -> ScenarioPlan:
    """Resolve a Track-2 scenario run plan without running anything."""
    resolved = tuple(date.fromisoformat(o) for o in (origins or DEFAULT_ORIGINS))
    return ScenarioPlan(origins=resolved, model=model, search_model=search_model, store_dir=store_dir)


def _run_agent_track2(config: AgentConfig, prompt: str, as_of: str) -> ScenarioRunResult:
    """Real Track-2 call: build the schema-less agent, run one turn, return the text.

    Imported lazily so the module (and the CLI's dry-run default) never pulls
    in the ``agentic`` extra or touches a model API unless ``--run`` is passed.
    """
    import asyncio  # noqa: PLC0415

    from aieng.forecasting.methods.agentic.adk_runner import AdkTextRunner, AdkTextRunnerConfig  # noqa: PLC0415
    from aieng.forecasting.methods.agentic.agent_factory import AS_OF_STATE_KEY, build_adk_agent  # noqa: PLC0415
    from aieng.forecasting.methods.llm_processes._client import trace_url_for  # noqa: PLC0415

    def _langfuse_available() -> bool:
        try:
            import langfuse  # noqa: F401, PLC0415

            return True
        except ModuleNotFoundError:
            return False

    # No output_schema -> Track-2 (free-text) mode: build_adk_agent registers no
    # set_model_response tool and returns whatever text the model produces.
    agent = build_adk_agent(config)
    runner = AdkTextRunner(
        agent=agent,
        config=AdkTextRunnerConfig(
            app_name="ws_scenario",
            default_user_id="scenario_agent",
            fresh_session_per_message=True,
            enable_langfuse_tracing=_langfuse_available(),
            langfuse_tags=["ws_scenario", "track2"],
            langfuse_trace_name=f"ws_scenario_{config.name}_{as_of}",
            langfuse_propagate_metadata={"agent_name": config.name, "model": str(config.model), "origin": as_of},
        ),
    )
    text = asyncio.run(runner.run_text_async(prompt, initial_state={AS_OF_STATE_KEY: as_of}))
    trace_id = runner.last_trace_id
    return ScenarioRunResult(
        markdown=text,
        agent_name=config.name,
        model=str(config.model),
        trace_id=trace_id,
        trace_url=trace_url_for(trace_id) if trace_id else None,
    )


def run_scenario_for_origin(
    origin: date,
    *,
    config: AgentConfig,
    data_service: Any,
    store_dir: Path,
    call: Track2Call,
    force_refresh: bool = False,
) -> Path | None:
    """Run (or skip, if already persisted) Track-2 scenario analysis for one origin.

    Returns the directory written, or ``None`` when the origin was skipped
    (already persisted and ``force_refresh`` is ``False``).
    """
    if not force_refresh and has_writeup(origin, store_dir):
        return None

    as_of = origin.isoformat()
    context = data_service.context(as_of=datetime.combine(origin, datetime.min.time()))
    prompt = build_scenario_prompt(as_of=as_of, context=context)
    result = call(config, prompt, as_of)

    writeup = ScenarioWriteup(
        origin=origin,
        markdown=result.markdown,
        meta={
            "origin": as_of,
            "model": result.model,
            "agent_name": result.agent_name,
            "trace_id": result.trace_id,
            "trace_url": result.trace_url,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        },
    )
    return write_scenario_writeup(writeup, store_dir)


def run_scenario_plan(
    plan: ScenarioPlan,
    data_service: Any,
    *,
    call: Track2Call | None = None,
    force_refresh: bool = False,
) -> list[Path]:
    """Run every pending origin in *plan*, persisting per origin.

    ``call`` is injectable for offline tests; when ``None`` it defaults to the
    real (spend-bearing) :func:`_run_agent_track2`.
    """
    resolved_call = call or _run_agent_track2
    config = build_tsx_scenario_config(model=plan.model, search_model=plan.search_model)
    written: list[Path] = []
    for origin in plan.origins:
        path = run_scenario_for_origin(
            origin,
            config=config,
            data_service=data_service,
            store_dir=plan.store_dir,
            call=resolved_call,
            force_refresh=force_refresh,
        )
        if path is not None:
            written.append(path)
    return written


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track-2 qualitative scenario analysis for the S&P/TSX Composite.")
    parser.add_argument(
        "--origin",
        action="append",
        default=None,
        metavar="YYYY-MM-DD",
        help="Forecast origin (repeatable; default: the landmark set).",
    )
    parser.add_argument("--model", default=LITE_MODEL, help="Scenario agent model.")
    parser.add_argument("--search-model", default=LITE_MODEL, help="search_web sub-agent model.")
    parser.add_argument("--store-dir", default=str(DEFAULT_SCENARIO_STORE_DIR), help="Scenario write-up store root.")
    parser.add_argument("--run", action="store_true", help="Actually run the agent (makes model + web-search calls).")
    parser.add_argument("--force", action="store_true", help="Re-run and overwrite an already-persisted origin.")
    return parser


def run(argv: list[str] | None = None) -> int:
    """Entry point for the ``ws-scenario`` console script."""
    args = _build_arg_parser().parse_args(argv)
    plan = plan_scenario_run(
        origins=args.origin, model=args.model, search_model=args.search_model, store_dir=Path(args.store_dir)
    )

    if not args.run:
        print(plan.describe())
        return 0

    from workshop_experiments.data_tsx import build_tsx_workshop_service  # noqa: PLC0415

    data_service = build_tsx_workshop_service(include_covariates=False)
    written = run_scenario_plan(plan, data_service, force_refresh=args.force)
    print(f"wrote {len(written)} scenario write-up(s).")
    for path in written:
        print(f"  - {path}")
    return 0


def main() -> None:
    """Console-script wrapper that exits with the run's status code."""
    sys.exit(run())


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_ORIGINS",
    "ScenarioPlan",
    "ScenarioRunResult",
    "Track2Call",
    "main",
    "plan_scenario_run",
    "run",
    "run_scenario_for_origin",
    "run_scenario_plan",
]
