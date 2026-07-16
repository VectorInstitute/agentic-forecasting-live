"""Tests for the LiteLLM call seam in ``llm_processes._client``.

These tests target the proxy-migration-sensitive logic that has no coverage
elsewhere: model prefixing, ``reasoning_effort`` routing, and markdown fence
stripping.  All LLM I/O is mocked so tests run without network access.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aieng.forecasting.methods.llm_processes._client import (
    _one_completion_async,
    make_json_schema_response_format,
    run_async,
    sample_n_async,
    strip_markdown_fence,
)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# make_json_schema_response_format — proxy/Gemini compatibility
# ---------------------------------------------------------------------------


def test_response_format_strips_additional_properties_recursively() -> None:
    """``additionalProperties`` is dropped at every level (proxy Gemini rejects it)."""
    schema = {
        "type": "object",
        "properties": {
            "forecasts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"q": {"type": "number"}},
                    "additionalProperties": False,
                },
            },
        },
        "required": ["forecasts"],
        "additionalProperties": False,
    }
    rf = make_json_schema_response_format("T", schema)
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True

    def _no_additional(node: object) -> bool:
        if isinstance(node, dict):
            return "additionalProperties" not in node and all(_no_additional(v) for v in node.values())
        if isinstance(node, list):
            return all(_no_additional(v) for v in node)
        return True

    assert _no_additional(rf["json_schema"]["schema"])
    # Declared structure is otherwise preserved.
    assert rf["json_schema"]["schema"]["required"] == ["forecasts"]
    assert "properties" in rf["json_schema"]["schema"]["properties"]["forecasts"]["items"]


# ---------------------------------------------------------------------------
# strip_markdown_fence — pure function, no mock needed
# ---------------------------------------------------------------------------


def test_strip_markdown_fence_removes_json_fence() -> None:
    """JSON fenced with ```json ... ``` is unwrapped to the inner content."""
    fenced = '```json\n{"point_forecast": 100}\n```'
    assert strip_markdown_fence(fenced) == '{"point_forecast": 100}'


def test_strip_markdown_fence_removes_plain_fence() -> None:
    """JSON fenced with plain ``` ... ``` is also unwrapped."""
    fenced = '```\n{"point_forecast": 100}\n```'
    assert strip_markdown_fence(fenced) == '{"point_forecast": 100}'


def test_strip_markdown_fence_leaves_plain_json_unchanged() -> None:
    """Content that is already plain JSON passes through unchanged."""
    plain = '{"point_forecast": 100}'
    assert strip_markdown_fence(plain) == plain


def test_strip_markdown_fence_strips_surrounding_whitespace() -> None:
    """Leading/trailing whitespace is stripped regardless of fencing."""
    assert strip_markdown_fence("  hello  ") == "hello"


def test_strip_markdown_fence_trims_trailing_prose() -> None:
    """Prose appended after the JSON is discarded (e.g. Claude via the proxy)."""
    response = '{"point_forecast": 100}\n\n**Method:** linear extrapolation of trend.'
    assert strip_markdown_fence(response) == '{"point_forecast": 100}'


def test_strip_markdown_fence_trims_fence_and_trailing_prose() -> None:
    """A fenced JSON block followed by prose is reduced to the JSON payload."""
    response = '```json\n{"point_forecast": 100}\n```\n\n**Method:** trend.'
    assert strip_markdown_fence(response) == '{"point_forecast": 100}'


def test_strip_markdown_fence_ignores_braces_in_leading_prose() -> None:
    """A stray brace inside prose does not derail extraction of the real JSON."""
    response = 'Use {x} notation. Here is the forecast: {"point_forecast": 100}'
    assert strip_markdown_fence(response) == '{"point_forecast": 100}'


def test_strip_markdown_fence_leaves_non_json_unchanged() -> None:
    """Content with no JSON object/array passes through fence-stripped only."""
    assert strip_markdown_fence("no json here") == "no json here"


# ---------------------------------------------------------------------------
# _one_completion_async — proxy routing via mocked litellm.acompletion
# ---------------------------------------------------------------------------


def _mock_litellm_response(content: str) -> MagicMock:
    """Build a minimal litellm-shaped response object."""
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp._hidden_params = {}
    resp.usage = None
    return resp


_DUMMY_MESSAGES = [{"role": "user", "content": "forecast"}]
_DUMMY_FORMAT = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}, "strict": True}}


@pytest.mark.asyncio
async def test_proxy_path_prefixes_model_with_openai() -> None:
    """When api_base is set, the model is prefixed with 'openai/'.

    Ensures LiteLLM routes the call via the OpenAI-compatible proxy path.
    """
    captured: list[dict] = []

    async def fake_acompletion(**kwargs):  # type: ignore[override]
        captured.append(kwargs)
        return _mock_litellm_response("{}")

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        await _one_completion_async(
            model="gemini-3-flash-preview",
            messages=_DUMMY_MESSAGES,
            response_format=_DUMMY_FORMAT,
            temperature=1.0,
            max_tokens=512,
            timeout_s=30.0,
            reasoning_effort=None,
            api_base="https://proxy.example.com/v1",
        )

    assert captured[0]["model"] == "openai/gemini-3-flash-preview"
    assert captured[0]["api_base"] == "https://proxy.example.com/v1"


@pytest.mark.asyncio
async def test_proxy_path_does_not_double_prefix_already_prefixed_model() -> None:
    """A model already starting with 'openai/' is not prefixed again."""
    captured: list[dict] = []

    async def fake_acompletion(**kwargs):  # type: ignore[override]
        captured.append(kwargs)
        return _mock_litellm_response("{}")

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        await _one_completion_async(
            model="openai/gemini-3-flash-preview",
            messages=_DUMMY_MESSAGES,
            response_format=_DUMMY_FORMAT,
            temperature=1.0,
            max_tokens=512,
            timeout_s=30.0,
            reasoning_effort=None,
            api_base="https://proxy.example.com/v1",
        )

    assert captured[0]["model"] == "openai/gemini-3-flash-preview"


@pytest.mark.asyncio
async def test_non_proxy_path_does_not_prefix_model() -> None:
    """Without api_base, the model name is sent to litellm unchanged."""
    captured: list[dict] = []

    async def fake_acompletion(**kwargs):  # type: ignore[override]
        captured.append(kwargs)
        return _mock_litellm_response("{}")

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        await _one_completion_async(
            model="gemini-3-flash-preview",
            messages=_DUMMY_MESSAGES,
            response_format=_DUMMY_FORMAT,
            temperature=1.0,
            max_tokens=512,
            timeout_s=30.0,
            reasoning_effort=None,
        )

    assert captured[0]["model"] == "gemini-3-flash-preview"
    assert "api_base" not in captured[0]


@pytest.mark.asyncio
async def test_proxy_path_sends_reasoning_effort_via_extra_body() -> None:
    """On the proxy path, reasoning_effort is injected via extra_body (not top-level).

    LiteLLM silently strips reasoning_effort for non-o1/o3 models when routing
    via a generic OpenAI-compatible endpoint.  Using extra_body bypasses the
    param-filter step and passes the value directly to the proxy.
    """
    captured: list[dict] = []

    async def fake_acompletion(**kwargs):  # type: ignore[override]
        captured.append(kwargs)
        return _mock_litellm_response("{}")

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        await _one_completion_async(
            model="gemini-3.1-pro-preview",
            messages=_DUMMY_MESSAGES,
            response_format=_DUMMY_FORMAT,
            temperature=1.0,
            max_tokens=512,
            timeout_s=30.0,
            reasoning_effort="low",
            api_base="https://proxy.example.com/v1",
        )

    kw = captured[0]
    assert kw.get("extra_body", {}).get("reasoning_effort") == "low"
    assert "reasoning_effort" not in kw  # must not appear at the top level
    assert kw.get("drop_params") is True


@pytest.mark.asyncio
async def test_non_proxy_path_sends_reasoning_effort_at_top_level() -> None:
    """Without a proxy, reasoning_effort is a top-level litellm kwarg."""
    captured: list[dict] = []

    async def fake_acompletion(**kwargs):  # type: ignore[override]
        captured.append(kwargs)
        return _mock_litellm_response("{}")

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        await _one_completion_async(
            model="gemini-3.1-pro-preview",
            messages=_DUMMY_MESSAGES,
            response_format=_DUMMY_FORMAT,
            temperature=1.0,
            max_tokens=512,
            timeout_s=30.0,
            reasoning_effort="low",
        )

    kw = captured[0]
    assert kw["reasoning_effort"] == "low"
    assert "extra_body" not in kw or "reasoning_effort" not in kw.get("extra_body", {})
    assert kw.get("drop_params") is True


# ---------------------------------------------------------------------------
# OTEL parent-span linking — generations nest under the @observe predict span
# ---------------------------------------------------------------------------
#
# Regression coverage for the Langfuse tracing defect: LLMP generations were
# emitted as orphan root spans (traceName: null) instead of nesting under the
# named predict span. Root cause: LiteLLM runs its OTEL success handler on a
# ThreadPoolExecutor worker whose contextvars context has no active span, so
# ambient-context parent resolution fails. The seam now passes the predict span
# explicitly via metadata["litellm_parent_otel_span"] (LiteLLM priority 1).


def _in_memory_tracer() -> tuple[trace.Tracer, InMemorySpanExporter]:
    """Build a standalone OTEL tracer + in-memory exporter (no global provider)."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


class _Tiny(BaseModel):
    ok: int


@pytest.mark.asyncio
async def test_one_completion_attaches_active_span_as_litellm_parent() -> None:
    """The active OTEL span is passed to litellm as ``litellm_parent_otel_span``."""
    tracer, _ = _in_memory_tracer()
    captured: list[dict] = []

    async def fake_acompletion(**kwargs):  # type: ignore[override]
        captured.append(kwargs)
        return _mock_litellm_response("{}")

    with (
        patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)),
        tracer.start_as_current_span("predict") as predict_span,
    ):
        await _one_completion_async(
            model="gemini-3-flash-preview",
            messages=_DUMMY_MESSAGES,
            response_format=_DUMMY_FORMAT,
            temperature=1.0,
            max_tokens=512,
            timeout_s=30.0,
            reasoning_effort=None,
        )

    assert captured[0]["metadata"]["litellm_parent_otel_span"] is predict_span


@pytest.mark.asyncio
async def test_one_completion_omits_parent_metadata_when_no_span_active() -> None:
    """With no active span, no ``litellm_parent_otel_span`` metadata is attached."""
    captured: list[dict] = []

    async def fake_acompletion(**kwargs):  # type: ignore[override]
        captured.append(kwargs)
        return _mock_litellm_response("{}")

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        await _one_completion_async(
            model="gemini-3-flash-preview",
            messages=_DUMMY_MESSAGES,
            response_format=_DUMMY_FORMAT,
            temperature=1.0,
            max_tokens=512,
            timeout_s=30.0,
            reasoning_effort=None,
        )

    assert "litellm_parent_otel_span" not in captured[0].get("metadata", {})


def test_generation_nests_under_predict_span_across_executor_boundary() -> None:
    """End-to-end seam: the generation nests under predict despite the thread hop.

    Drives the real seam (``run_async`` -> ``sample_n_async`` -> the async
    single-completion path) inside an ``@observe``-style span, and simulates how
    LiteLLM's ``langfuse_otel`` callback fires: on a ``ThreadPoolExecutor``
    worker. The fake callback records the *ambient* span it sees on that worker
    (the fresh, span-less context that caused orphaning) and builds the
    generation span from the explicit ``litellm_parent_otel_span`` metadata
    (LiteLLM priority 1). We assert the ambient context is indeed empty on the
    worker, yet the generation still lands in the predict trace as its child.
    """
    tracer, exporter = _in_memory_tracer()
    ambient_valid_on_worker: list[bool] = []

    def _fake_litellm_otel_success(metadata: dict) -> None:
        # Priority 3 (ambient) — what LiteLLM would fall back to on this worker.
        ambient = trace.get_current_span()
        ambient_valid_on_worker.append(ambient.get_span_context().is_valid)
        # Priority 1 (explicit parent from metadata) — the fix.
        parent = metadata.get("litellm_parent_otel_span")
        ctx = trace.set_span_in_context(parent) if parent is not None else None
        tracer.start_span("litellm-generation", context=ctx).end()

    async def fake_acompletion(**kwargs):  # type: ignore[override]
        metadata = kwargs.get("metadata", {})
        # LiteLLM emits its OTEL span from a threadpool worker (fresh context).
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_fake_litellm_otel_success, metadata).result()
        return _mock_litellm_response('{"ok": 1}')

    with (
        patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)),
        tracer.start_as_current_span("predict") as predict_span,
    ):
        predict_trace_id = predict_span.get_span_context().trace_id
        predict_span_id = predict_span.get_span_context().span_id
        parsed, *_ = run_async(
            sample_n_async(
                schema_cls=_Tiny,
                model="gemini-3-flash-preview",
                base_messages=_DUMMY_MESSAGES,
                response_format=_DUMMY_FORMAT,
                n_samples=1,
                temperature=1.0,
                max_tokens=512,
                timeout_s=30.0,
                reasoning_effort=None,
            )
        )

    assert parsed == [_Tiny(ok=1)]
    # The worker thread's ambient OTEL context is span-less — the exact
    # condition that orphaned generations before the fix.
    assert ambient_valid_on_worker == [False]
    # Yet the generation nests under the predict span in the same trace.
    gen_spans = [s for s in exporter.get_finished_spans() if s.name == "litellm-generation"]
    assert len(gen_spans) == 1
    gen = gen_spans[0]
    assert gen.context.trace_id == predict_trace_id
    assert gen.parent is not None
    assert gen.parent.span_id == predict_span_id
