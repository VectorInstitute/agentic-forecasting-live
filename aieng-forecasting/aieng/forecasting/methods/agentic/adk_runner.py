"""General-purpose ADK runner: text-in / text-out over ``InMemoryRunner``.

This module provides :class:`AdkTextRunner`, a thin wrapper around Google
ADK's :class:`~google.adk.runners.InMemoryRunner` that exposes a single
``run_text_async(prompt) -> str`` method, manages per-user session lifecycle,
and optionally propagates Langfuse trace attributes for each turn.

This module requires the ``agentic`` extra; importing it without the extra
raises :class:`ImportError`.
"""

from __future__ import annotations

import types as py_types
from typing import Any

from pydantic import BaseModel, Field


try:
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.agents.run_config import RunConfig
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types
except ModuleNotFoundError as exc:
    raise ImportError(
        "This module requires the 'agentic' extra. Install it with 'pip install aieng-forecasting[agentic]'."
    ) from exc


#: Defensive cap on the number of tool calls captured per run. The public
#: dashboard only needs a representative summary, so an unbounded transcript is
#: neither useful nor safe to surface.
_MAX_TOOL_CALLS = 50

#: Maximum length of a curated tool-call title (e.g. a search query).
_TOOL_TITLE_MAX_CHARS = 120

#: Tool names treated as code execution for title curation. The E2B-backed
#: ``CodeInterpreter`` registers its callable as ``run_code``; the others are
#: defensive aliases so a renamed code tool still gets a body-free label.
_CODE_EXEC_TOOLS = frozenset({"run_code", "execute_code", "code_execution"})


def _curate_tool_call_title(tool_name: str, args: dict[str, Any]) -> str:
    """Return a short, PUBLIC-safe title for one tool invocation.

    This is the curation policy, enforced at *capture* time so that no raw
    tool payload is ever retained beyond this point:

    * ``search_web`` -> the query string, truncated to
      :data:`_TOOL_TITLE_MAX_CHARS` characters.
    * code execution (see :data:`_CODE_EXEC_TOOLS`) -> a length label such as
      ``"python (N lines)"``; the code body is never retained.
    * any other tool -> the tool name (a stable, non-sensitive label).

    Retrieved article text is never emitted, and source URLs are omitted
    entirely.

    Parameters
    ----------
    tool_name : str
        The invoked tool's name.
    args : dict of str to Any
        The tool-call arguments as reported by the model.

    Returns
    -------
    str
        A curated, human-readable title safe to publish.
    """
    if tool_name == "search_web":
        query = args.get("query")
        return query[:_TOOL_TITLE_MAX_CHARS] if isinstance(query, str) else ""
    if tool_name in _CODE_EXEC_TOOLS:
        code = args.get("code")
        if isinstance(code, str) and code:
            return f"python ({code.count(chr(10)) + 1} lines)"
        return "python"
    return tool_name


def _collect_tool_calls(event: Any, sink: list[dict[str, str]]) -> None:
    """Append curated ``{"tool", "title"}`` entries for *event*'s tool calls.

    Inspects the event's content parts for ``function_call`` payloads and, for
    each one carrying a real string name, records a curated summary in *sink*.
    The scan is deliberately defensive: events without a proper ``parts`` list
    (e.g. intermediate/non-content events) are skipped, and *sink* is never
    grown past :data:`_MAX_TOOL_CALLS`.
    """
    if len(sink) >= _MAX_TOOL_CALLS:
        return
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None)
    if not isinstance(parts, (list, tuple)):
        return
    for part in parts:
        function_call = getattr(part, "function_call", None)
        name = getattr(function_call, "name", None)
        if not isinstance(name, str) or not name:
            continue
        args = getattr(function_call, "args", None)
        args = args if isinstance(args, dict) else {}
        sink.append({"tool": name, "title": _curate_tool_call_title(name, args)})
        if len(sink) >= _MAX_TOOL_CALLS:
            return


class AdkTextRunnerConfig(BaseModel):
    """Configuration for :class:`AdkTextRunner`.

    Attributes
    ----------
    app_name : str
        Application id shared by the session service and runner.
    default_user_id : str
        Fallback user id when :meth:`~AdkTextRunner.run_text_async` is called
        without an explicit ``user_id``.
    fresh_session_per_message : bool
        When ``True`` (default), each :meth:`~AdkTextRunner.run_text_async`
        call creates a fresh ADK session and any supplied ``session_id`` is
        ignored.  When ``False``, sessions are reused per ``user_id``
        (sticky conversation).
    enable_langfuse_tracing : bool
        When ``True``, initialise Langfuse at construction time and wrap every
        turn with ``propagate_attributes``.  Requires the ``agentic`` extra.
    langfuse_tags : list of str or None
        Tags forwarded to Langfuse ``propagate_attributes``.
    langfuse_propagate_metadata : dict of str to str, or None
        Extra key/value metadata merged with ``adk_app_name`` and forwarded
        to ``propagate_attributes``.
    langfuse_trace_name : str or None
        ``trace_name`` forwarded to Langfuse ``propagate_attributes``.
    langfuse_version : str or None
        ``version`` forwarded to Langfuse ``propagate_attributes``.

    Notes
    -----
    When ``enable_langfuse_tracing`` is ``True``, ``user_id``, ``session_id``,
    ``trace_name``, and every key/value in ``langfuse_propagate_metadata`` must
    be US-ASCII and ≤ 200 characters each; Langfuse silently drops
    non-conforming values.
    """

    app_name: str = Field(
        ...,
        description="Application id shared by session service and runner.",
    )
    default_user_id: str = Field(
        default="user",
        description=(
            "Used when ``run_text_async`` is called without ``user_id``. "
            "If Langfuse tracing is enabled, must be US-ASCII and ≤ 200 characters."
        ),
    )
    fresh_session_per_message: bool = Field(
        default=True,
        description=(
            "If True, each ``run_text_async`` creates a new session (``session_id`` is ignored). "
            "If False, turns for the same ``user_id`` reuse one session: the first call creates it, "
            "later calls omit ``session_id`` unless switching threads; optional explicit "
            "``session_id`` joins or replaces the sticky session for that user."
        ),
    )
    enable_langfuse_tracing: bool = Field(
        default=False,
        description=(
            "If True, call :func:`~aieng.forecasting.langfuse_tracing.init_langfuse_tracing` "
            "at runner construction and wrap each turn with Langfuse "
            "``propagate_attributes``. Forwards resolved ``user_id`` and ADK ``session_id`` "
            "plus optional fields below. Langfuse requires propagated identifiers to be "
            "US-ASCII and ≤ 200 characters; invalid values may be dropped with warnings. "
            "Requires the ``agentic`` extra (``langfuse``)."
        ),
    )
    langfuse_tags: list[str] | None = Field(
        default=None,
        description=("Optional tags for ``propagate_attributes`` to categorize observations in Langfuse."),
    )
    langfuse_propagate_metadata: dict[str, str] | None = Field(
        default=None,
        description=(
            "Extra metadata merged with ``adk_app_name`` for ``propagate_attributes``. "
            "Keys and values must be US-ASCII strings ≤ 200 characters each; avoid large "
            "payloads or sensitive data (non-conforming entries may be dropped with warnings)."
        ),
    )
    langfuse_trace_name: str | None = Field(
        default=None,
        description=("Optional ``trace_name`` for ``propagate_attributes``: US-ASCII, ≤ 200 characters."),
    )
    langfuse_version: str | None = Field(
        default=None,
        description=(
            "Optional ``version`` for independently versioned parts of the app (e.g. agent "
            "revision). Use short US-ASCII values suitable for span attributes."
        ),
    )

    model_config = {"extra": "forbid"}


class AdkTextRunner:
    """Wrap ``InMemoryRunner`` with session helpers.

    Parameters
    ----------
    agent : BaseAgent
        The ADK agent to run.
    config : AdkTextRunnerConfig
        The configuration for the runner.

    Examples
    --------
    Build a runner from an :class:`AgentConfig` and send one prompt:

    >>> from aieng.forecasting.methods.agentic import (
    ...     AgentConfig,
    ...     build_adk_agent,
    ... )
    >>> from aieng.forecasting.methods.agentic.adk_runner import (
    ...     AdkTextRunner,
    ...     AdkTextRunnerConfig,
    ... )
    >>> agent = build_adk_agent(AgentConfig(instruction="You are a helpful assistant."))
    >>> runner = AdkTextRunner(
    ...     agent,
    ...     config=AdkTextRunnerConfig(app_name="demo"),
    ... )
    >>> reply = await runner.run_text_async("Hello.")
    """

    def __init__(self, agent: BaseAgent, *, config: AdkTextRunnerConfig) -> None:
        """Construct the runner and optionally initialise Langfuse tracing."""
        self.config = config
        self.agent = agent
        self._runner = InMemoryRunner(agent=agent, app_name=config.app_name)
        # Sticky ADK session per user when ``fresh_session_per_message`` is False.
        self._conversation_session_by_user: dict[str, str] = {}
        # Trace id captured during the most recent traced run (see ``last_trace_id``).
        self._last_trace_id: str | None = None
        # Curated tool-call summaries captured during the most recent run
        # (see ``last_tool_calls``).
        self._last_tool_calls: list[dict[str, str]] = []
        if config.enable_langfuse_tracing:
            from aieng.forecasting.langfuse_tracing import init_langfuse_tracing  # noqa: PLC0415

            init_langfuse_tracing()

    @property
    def last_trace_id(self) -> str | None:
        """Langfuse trace id captured during the most recent traced run, if any.

        The agent runs on a worker event loop whose trace context the caller's
        thread cannot see; the runner captures the id here so a predictor can link
        and score the trace after the run. ``None`` when tracing is off or the last
        run produced no trace.
        """
        return self._last_trace_id

    @property
    def last_tool_calls(self) -> list[dict[str, str]]:
        """Curated tool-call summaries captured during the most recent run.

        Each entry is a ``{"tool": <name>, "title": <curated title>}`` dict, in
        invocation order, capped at :data:`_MAX_TOOL_CALLS`. Titles are curated
        at capture time (see :func:`_curate_tool_call_title`) so the list is
        safe to surface publicly — it never contains code bodies or retrieved
        article text. Empty when the last run made no tool calls (or captured
        none). A predictor threads this into prediction metadata so the live
        harness can publish a ``curated_trace_summary``.
        """
        return self._last_tool_calls

    @property
    def runner(self) -> InMemoryRunner:
        """Underlying ADK runner (session, artifact, memory services)."""
        return self._runner

    async def _resolve_session_id(
        self,
        user_id: str | None,
        session_id: str | None,
        *,
        initial_state: dict[str, Any] | None = None,
    ) -> str:
        """Return the ADK session id to use for a single turn.

        Parameters
        ----------
        user_id : str or None
            Resolved user id; falls back to ``default_user_id`` when ``None``.
        session_id : str or None
            Explicit session id from the caller.  ``None`` triggers sticky-session
            lookup or new-session creation depending on ``fresh_session_per_message``.
        initial_state : dict[str, Any] or None
            Seeded into a newly-created session's state. Only takes effect when
            this call actually creates a session — has no effect when an
            existing sticky session (``fresh_session_per_message=False``) is
            reused, since session state can only be seeded at creation.

        Returns
        -------
        str
            ADK session id for this turn.
        """
        if user_id is None:
            user_id = self.config.default_user_id

        if self.config.fresh_session_per_message:
            new_session = await self._runner.session_service.create_session(
                app_name=self.config.app_name,
                user_id=user_id,
                state=initial_state,
            )
            sid = new_session.id
        elif session_id is not None:
            sid = session_id
            self._conversation_session_by_user[user_id] = sid
        elif user_id in self._conversation_session_by_user:
            sid = self._conversation_session_by_user[user_id]
        else:
            new_session = await self._runner.session_service.create_session(
                app_name=self.config.app_name,
                user_id=user_id,
                state=initial_state,
            )
            sid = new_session.id
            self._conversation_session_by_user[user_id] = sid

        return sid

    async def run_text_async(
        self,
        prompt: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        run_config: RunConfig | None = None,
        initial_state: dict[str, Any] | None = None,
    ) -> str:
        """Run one user turn; return the first final model text or an empty string.

        Parameters
        ----------
        prompt : str
            The user prompt to run.
        user_id : str | None, optional
            The user id to use for the session. If not provided, the default
            user id is used. With Langfuse tracing, must be US-ASCII and ≤ 200
            characters for propagation.
        session_id : str | None, optional
            The session id to use for the session. If not provided, a new session
            is created. With Langfuse tracing, the ADK session id must remain
            US-ASCII and ≤ 200 characters for propagation.
        run_config : RunConfig | None, optional
            The run configuration to use for the run. If not provided, the default
            run configuration is used.
        initial_state : dict[str, Any] | None, optional
            Seeded into the session's state when this call creates a new
            session (see :meth:`_resolve_session_id`). Use this to pass
            harness-controlled values (e.g. a forecast's ``as_of`` date) that
            tools can read via ``ToolContext.state`` without the LLM being
            able to see or influence them.

        Returns
        -------
        str
            The first final model text or an empty string.

        Notes
        -----
        If ``fresh_session_per_message`` is True, each call uses a new ADK session and
        ``session_id`` is ignored.

        If it is False, the runner keeps a session per ``user_id``: omit ``session_id``
        after the first message to continue the same conversation. Pass ``session_id``
        to attach to an existing session or switch threads; that id is remembered for
        later calls with ``session_id`` omitted (same user).

        When ``enable_langfuse_tracing`` is True, each turn runs inside Langfuse
        ``propagate_attributes`` using the resolved ``user_id`` and ADK ``session_id``.
        """
        from aieng.forecasting.methods.agentic.agent_factory import SMR_STATE_KEY  # noqa: PLC0415

        user_id = user_id or self.config.default_user_id

        session_id = await self._resolve_session_id(user_id, session_id, initial_state=initial_state)

        content = genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])

        # Reset per-run capture and expose it live so callers see whatever was
        # recorded even if the run raises partway through.
        tool_calls: list[dict[str, str]] = []
        self._last_tool_calls = tool_calls

        async def drain_run() -> str:
            async for event in self._runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content,
                run_config=run_config,
            ):
                _collect_tool_calls(event, tool_calls)
                if event.is_final_response() and event.content and event.content.parts:
                    return event.content.parts[0].text or ""
            return ""

        async def run_and_resolve() -> str:
            """Run the agent and return the best available output string.

            When the agent uses our set_model_response shim (LiteLlm path with
            tools + output_schema), the structured JSON is stored in session
            state under SMR_STATE_KEY.  We prefer that over the model's
            subsequent "Task complete." text response.
            """
            text = await drain_run()
            session = await self._runner.session_service.get_session(
                app_name=self.config.app_name,
                user_id=user_id,
                session_id=session_id,
            )
            if session is not None and SMR_STATE_KEY in (session.state or {}):
                return str(session.state[SMR_STATE_KEY])
            return text

        if self.config.enable_langfuse_tracing:
            from langfuse import get_client, propagate_attributes  # noqa: PLC0415

            metadata: dict[str, str] = {"adk_app_name": self.config.app_name}
            if self.config.langfuse_propagate_metadata:
                metadata = {**metadata, **self.config.langfuse_propagate_metadata}

            pa_kw: dict[str, Any] = {
                k: v
                for k, v in {
                    "user_id": user_id,
                    "session_id": session_id,
                    "metadata": metadata,
                    "tags": self.config.langfuse_tags,
                    "trace_name": self.config.langfuse_trace_name,
                    "version": self.config.langfuse_version,
                }.items()
                if v is not None
            }
            # Wrap the run in an explicit Langfuse span so (a) the ADK spans nest
            # under one root trace and (b) we can capture the trace id while its
            # context is active — the caller's thread cannot see it otherwise.
            self._last_trace_id = None
            client = get_client()
            root_name = self.config.langfuse_trace_name or self.config.app_name
            with client.start_as_current_observation(name=root_name, as_type="agent"):
                with propagate_attributes(**pa_kw):
                    result = await run_and_resolve()
                self._last_trace_id = client.get_current_trace_id()
            return result

        return await run_and_resolve()

    def clear_conversation(self, *, user_id: str | None = None) -> None:
        """Drop sticky session id(s). Next ``run_text_async`` starts a new chat.

        With ``user_id``, clear only that user. With ``None``, clear every user.
        No effect when ``fresh_session_per_message`` is True.

        Parameters
        ----------
        user_id : str | None, optional
            The user id to clear the conversation for. If not provided, all users
            are cleared. No effect when ``fresh_session_per_message`` is True.
        """
        if user_id is None:
            self._conversation_session_by_user.clear()
        else:
            self._conversation_session_by_user.pop(user_id, None)

    async def aclose(self) -> None:
        """Close the underlying runner (plugins, toolsets)."""
        self._conversation_session_by_user.clear()
        await self._runner.close()  # type: ignore[no-untyped-call]

    async def __aenter__(self) -> AdkTextRunner:
        """Return self for use as an ``async with`` target."""
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: py_types.TracebackType | None
    ) -> None:
        """Close the runner when leaving the ``async with`` block."""
        await self.aclose()
