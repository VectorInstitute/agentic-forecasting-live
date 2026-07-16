"""Named predictor factories for the workshop S&P 500 experiments.

Each factory returns a fresh :class:`~aieng.forecasting.evaluation.predictor.Predictor`
from a short name. Model-parameterised methods (LLMP, agents) fold the model
into their ``predictor_id`` so persisted caches under
``data/predictions/<spec>/`` keep model variants separate; conventional methods
carry a fixed id (they consume no model).

Method groups
-------------
- ``conventional`` — ``naive``, ``ets``, ``kalman``, ``autoarima``,
  ``lightgbm``, ``lightgbm_cov`` (free, local, no API).
- ``llmp`` — ``llmp_qgrid``, ``llmp_qgrid_cov`` (one structured completion per
  origin; ``_cov`` serialises the covariate panel into the prompt).
- ``agent`` — ``agent_news`` (news-grounded analyst), ``agent_code``
  (code-executing analyst), both built via the PR-1 ``build_analyst_config``
  machinery over :data:`SP500_DOMAIN`.

``predictor_id`` examples
-------------------------
- ``last_value_naive``, ``darts_ets``, ``darts_lightgbm``, ``darts_lightgbm_cov``
- ``llmp_quantile_grid_sp500_ws[gemini-3.1-flash-lite-preview]``
- ``llmp_quantile_grid_sp500_ws_cov[gemini-3.1-flash-lite-preview]``
- ``agent_predictor_sp500_analyst_news_<model>_continuous``
- ``agent_predictor_sp500_analyst_code_<model>_continuous``
"""

from __future__ import annotations

from typing import Callable

from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.methods.baselines import LastValuePredictor
from aieng.forecasting.methods.llm_processes import (
    QuantileGridLLMPredictor,
    QuantileGridLLMPredictorConfig,
)
from aieng.forecasting.methods.numerical import (
    DartsAutoARIMAPredictor,
    DartsExponentialSmoothingPredictor,
    DartsKalmanForecasterPredictor,
    DartsLightGBMPredictor,
)
from aieng.forecasting.models import LITE_MODEL

from workshop_experiments.data import SP500_COVARIATE_PANEL
from workshop_experiments.data_tsx import TSX_COVARIATE_PANEL
from workshop_experiments.domain import (
    build_sp500_agent_predictor,
    build_sp500_code_config,
    build_sp500_news_config,
)
from workshop_experiments.domain_tsx import (
    build_tsx_agent_predictor,
    build_tsx_code_config,
    build_tsx_news_config,
)


#: Factory signature: ``(model) -> Predictor``. ``model`` is honoured by the
#: LLMP and agent factories and ignored by the conventional ones.
PredictorFactory = Callable[[str, list[str]], Predictor]


# ---------------------------------------------------------------------------
# LLMP recipe framing (shared by the quantile-grid variants). Mirrors the
# sp500_forecasting sampled-trajectory recipe so the elicitation strategies see
# identical equity framing.
# ---------------------------------------------------------------------------

_LLMP_HISTORY_WINDOW = 64

_LLMP_SERIES_DESCRIPTION = (
    "Series: S&P 500 (^GSPC) close-to-close cumulative log return over a fixed "
    "number of business days.\n"
    "Units: log-return (a value of 0.01 is roughly a +1% move).\n"
    "Frequency: business days (Mon-Fri)."
)

_LLMP_USER_PROMPT_SUFFIX = (
    "Notes for this series:\n"
    "- Daily index returns are close to a martingale: the *level* of the return "
    "is barely predictable, so point forecasts should sit near 0 and the value "
    "is in the *spread* (volatility and tail risk), not a confident direction.\n"
    "- Returns cluster in volatility — calm and turbulent stretches persist — so "
    "recent realised dispersion is the best guide to the width of your interval.\n"
    "- Keep the distribution roughly symmetric about ~0 unless the recent history "
    "or the covariate blocks give a clear reason to skew it; avoid extrapolating "
    "a short run of up or down days into a trend."
)


# ---------------------------------------------------------------------------
# Conventional factories (no model, no API)
# ---------------------------------------------------------------------------


def _make_naive(model: str, covariate_panel: list[str]) -> Predictor:
    return LastValuePredictor()


def _make_ets(model: str, covariate_panel: list[str]) -> Predictor:
    return DartsExponentialSmoothingPredictor()


def _make_kalman(model: str, covariate_panel: list[str]) -> Predictor:
    return DartsKalmanForecasterPredictor()


def _make_autoarima(model: str, covariate_panel: list[str]) -> Predictor:
    return DartsAutoARIMAPredictor()


def _make_lightgbm(model: str, covariate_panel: list[str]) -> Predictor:
    return DartsLightGBMPredictor()


def _make_lightgbm_cov(model: str, covariate_panel: list[str]) -> Predictor:
    if not covariate_panel:
        raise ValueError(
            "lightgbm_cov requires a non-empty covariate panel (is --no-covariates set, or are all covariates unavailable?)"
        )
    return DartsLightGBMPredictor(covariate_series_ids=list(covariate_panel))


# ---------------------------------------------------------------------------
# LLMP factories (model-parameterised)
# ---------------------------------------------------------------------------


def _make_llmp_qgrid(model: str, covariate_panel: list[str]) -> Predictor:
    return QuantileGridLLMPredictor(
        QuantileGridLLMPredictorConfig(
            model=model,
            history_window=_LLMP_HISTORY_WINDOW,
            series_description=_LLMP_SERIES_DESCRIPTION,
            user_prompt_suffix=_LLMP_USER_PROMPT_SUFFIX,
            variant_tag="sp500_ws",
        )
    )


def _make_llmp_qgrid_cov(model: str, covariate_panel: list[str]) -> Predictor:
    if not covariate_panel:
        raise ValueError(
            "llmp_qgrid_cov requires a non-empty covariate panel (is --no-covariates set, or are all covariates unavailable?)"
        )
    return QuantileGridLLMPredictor(
        QuantileGridLLMPredictorConfig(
            model=model,
            history_window=_LLMP_HISTORY_WINDOW,
            series_description=_LLMP_SERIES_DESCRIPTION,
            user_prompt_suffix=_LLMP_USER_PROMPT_SUFFIX,
            covariate_series_ids=list(covariate_panel),
            variant_tag="sp500_ws_cov",
        )
    )


# ---------------------------------------------------------------------------
# Agent factories (model-parameterised; construct on demand — may need proxy
# env at run time, never at import time)
# ---------------------------------------------------------------------------


def _make_agent_news(model: str, covariate_panel: list[str]) -> Predictor:
    return build_sp500_agent_predictor(build_sp500_news_config(model=model))


def _make_agent_code(model: str, covariate_panel: list[str]) -> Predictor:
    return build_sp500_agent_predictor(build_sp500_code_config(model=model))


# ---------------------------------------------------------------------------
# TSX variants (Canada-focused primary target). Distinct method names and, for
# the model-parameterised rungs, distinct predictor ids (via the ``tsx_ws``
# variant tag and the ``tsx_analyst_*`` agent names) so caches never collide with
# the S&P 500 rungs. Conventional rungs reuse the target-agnostic baseline/darts
# predictors; the runner keys their stores by the tsx_* spec id.
# ---------------------------------------------------------------------------

_TSX_LLMP_SERIES_DESCRIPTION = (
    "Series: S&P/TSX Composite (^GSPTSE) close-to-close cumulative log return over a fixed "
    "number of business days.\n"
    "Units: log-return (a value of 0.01 is roughly a +1% move).\n"
    "Frequency: business days (Mon-Fri)."
)

_TSX_LLMP_USER_PROMPT_SUFFIX = (
    "Notes for this series:\n"
    "- Daily index returns are close to a martingale: the *level* of the return "
    "is barely predictable, so point forecasts should sit near 0 and the value "
    "is in the *spread* (volatility and tail risk), not a confident direction.\n"
    "- The TSX is heavy in energy and materials, so oil, gold, and the Canadian "
    "dollar are primary risk drivers; US equity moves spill over strongly.\n"
    "- Returns cluster in volatility — calm and turbulent stretches persist — so "
    "recent realised dispersion is the best guide to the width of your interval.\n"
    "- Keep the distribution roughly symmetric about ~0 unless the recent history "
    "or the covariate blocks give a clear reason to skew it; avoid extrapolating "
    "a short run of up or down days into a trend."
)


def _make_tsx_naive(model: str, covariate_panel: list[str]) -> Predictor:
    return LastValuePredictor()


def _make_tsx_ets(model: str, covariate_panel: list[str]) -> Predictor:
    return DartsExponentialSmoothingPredictor()


def _make_tsx_kalman(model: str, covariate_panel: list[str]) -> Predictor:
    return DartsKalmanForecasterPredictor()


def _make_tsx_autoarima(model: str, covariate_panel: list[str]) -> Predictor:
    return DartsAutoARIMAPredictor()


def _make_tsx_lightgbm(model: str, covariate_panel: list[str]) -> Predictor:
    return DartsLightGBMPredictor()


def _make_tsx_lightgbm_cov(model: str, covariate_panel: list[str]) -> Predictor:
    if not covariate_panel:
        raise ValueError(
            "tsx_lightgbm_cov requires a non-empty covariate panel (is --no-covariates set, or are all covariates unavailable?)"
        )
    return DartsLightGBMPredictor(covariate_series_ids=list(covariate_panel))


def _make_tsx_llmp_qgrid(model: str, covariate_panel: list[str]) -> Predictor:
    return QuantileGridLLMPredictor(
        QuantileGridLLMPredictorConfig(
            model=model,
            history_window=_LLMP_HISTORY_WINDOW,
            series_description=_TSX_LLMP_SERIES_DESCRIPTION,
            user_prompt_suffix=_TSX_LLMP_USER_PROMPT_SUFFIX,
            variant_tag="tsx_ws",
        )
    )


def _make_tsx_llmp_qgrid_cov(model: str, covariate_panel: list[str]) -> Predictor:
    if not covariate_panel:
        raise ValueError(
            "tsx_llmp_qgrid_cov requires a non-empty covariate panel (is --no-covariates set, or are all covariates unavailable?)"
        )
    return QuantileGridLLMPredictor(
        QuantileGridLLMPredictorConfig(
            model=model,
            history_window=_LLMP_HISTORY_WINDOW,
            series_description=_TSX_LLMP_SERIES_DESCRIPTION,
            user_prompt_suffix=_TSX_LLMP_USER_PROMPT_SUFFIX,
            covariate_series_ids=list(covariate_panel),
            variant_tag="tsx_ws_cov",
        )
    )


def _make_tsx_agent_news(model: str, covariate_panel: list[str]) -> Predictor:
    return build_tsx_agent_predictor(build_tsx_news_config(model=model))


def _make_tsx_agent_code(model: str, covariate_panel: list[str]) -> Predictor:
    return build_tsx_agent_predictor(build_tsx_code_config(model=model))


# ---------------------------------------------------------------------------
# Registry + method groups
# ---------------------------------------------------------------------------

REGISTRY: dict[str, PredictorFactory] = {
    "naive": _make_naive,
    "ets": _make_ets,
    "kalman": _make_kalman,
    "autoarima": _make_autoarima,
    "lightgbm": _make_lightgbm,
    "lightgbm_cov": _make_lightgbm_cov,
    "llmp_qgrid": _make_llmp_qgrid,
    "llmp_qgrid_cov": _make_llmp_qgrid_cov,
    "agent_news": _make_agent_news,
    "agent_code": _make_agent_code,
    # TSX (Canada-focused primary target)
    "tsx_naive": _make_tsx_naive,
    "tsx_ets": _make_tsx_ets,
    "tsx_kalman": _make_tsx_kalman,
    "tsx_autoarima": _make_tsx_autoarima,
    "tsx_lightgbm": _make_tsx_lightgbm,
    "tsx_lightgbm_cov": _make_tsx_lightgbm_cov,
    "tsx_llmp_qgrid": _make_tsx_llmp_qgrid,
    "tsx_llmp_qgrid_cov": _make_tsx_llmp_qgrid_cov,
    "tsx_agent_news": _make_tsx_agent_news,
    "tsx_agent_code": _make_tsx_agent_code,
}

#: Free, local, API-free S&P 500 methods — safe to smoke-run without a budget gate.
CONVENTIONAL_METHODS: tuple[str, ...] = (
    "naive",
    "ets",
    "kalman",
    "autoarima",
    "lightgbm",
    "lightgbm_cov",
)
#: LLM-process methods (one structured completion per origin).
LLMP_METHODS: tuple[str, ...] = ("llmp_qgrid", "llmp_qgrid_cov")
#: News-grounded and code-executing analyst agents.
AGENT_METHODS: tuple[str, ...] = ("agent_news", "agent_code")

#: TSX (Canada-focused) counterparts of every rung.
TSX_CONVENTIONAL_METHODS: tuple[str, ...] = (
    "tsx_naive",
    "tsx_ets",
    "tsx_kalman",
    "tsx_autoarima",
    "tsx_lightgbm",
    "tsx_lightgbm_cov",
)
TSX_LLMP_METHODS: tuple[str, ...] = ("tsx_llmp_qgrid", "tsx_llmp_qgrid_cov")
TSX_AGENT_METHODS: tuple[str, ...] = ("tsx_agent_news", "tsx_agent_code")

#: Methods that call an LLM/agent API (spend-gated).
API_METHODS: tuple[str, ...] = LLMP_METHODS + AGENT_METHODS + TSX_LLMP_METHODS + TSX_AGENT_METHODS
#: Every registered method name.
ALL_METHODS: tuple[str, ...] = (
    CONVENTIONAL_METHODS
    + LLMP_METHODS
    + AGENT_METHODS
    + TSX_CONVENTIONAL_METHODS
    + TSX_LLMP_METHODS
    + TSX_AGENT_METHODS
)

#: Registry method names that operate on the TSX target (drive covariate-panel
#: and data-service selection for the caller).
TSX_METHODS: frozenset[str] = frozenset(TSX_CONVENTIONAL_METHODS + TSX_LLMP_METHODS + TSX_AGENT_METHODS)


def build_predictor(name: str, *, model: str = LITE_MODEL, covariate_panel: list[str] | None = None) -> Predictor:
    """Construct the predictor registered under *name*.

    Parameters
    ----------
    name : str
        A key in :data:`REGISTRY` (see :data:`ALL_METHODS`).
    model : str, default=LITE_MODEL
        Model id for LLMP / agent methods. Ignored by conventional methods.
    covariate_panel : list[str] | None, default=None
        Covariate series ids for the ``*_cov`` variants. ``None`` falls back
        to the full :data:`SP500_COVARIATE_PANEL`; callers that have built the
        data service should pass the panel filtered to actually-registered
        series (some covariates, e.g. gold, can be unavailable upstream).

    Returns
    -------
    Predictor
        A fresh predictor instance.

    Raises
    ------
    KeyError
        If *name* is not a registered method.
    """
    try:
        factory = REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown predictor {name!r}. Known methods: {', '.join(ALL_METHODS)}.") from None
    if covariate_panel is not None:
        panel = list(covariate_panel)
    else:
        panel = list(TSX_COVARIATE_PANEL) if name in TSX_METHODS else list(SP500_COVARIATE_PANEL)
    return factory(model, panel)


def resolve_methods(methods: list[str]) -> list[str]:
    """Expand method selectors into a concrete, de-duplicated method list.

    Accepts the group aliases ``all``, ``conventional``, ``llmp``, ``agent``,
    ``api`` alongside individual method names, preserving first-seen order.

    Raises
    ------
    KeyError
        If any token is neither a group alias nor a registered method.
    """
    groups = {
        "all": ALL_METHODS,
        "conventional": CONVENTIONAL_METHODS,
        "llmp": LLMP_METHODS,
        "agent": AGENT_METHODS,
        "api": API_METHODS,
        "tsx_conventional": TSX_CONVENTIONAL_METHODS,
        "tsx_llmp": TSX_LLMP_METHODS,
        "tsx_agent": TSX_AGENT_METHODS,
        "tsx_all": TSX_CONVENTIONAL_METHODS + TSX_LLMP_METHODS + TSX_AGENT_METHODS,
    }
    resolved: list[str] = []
    for token in methods:
        expanded = groups.get(token, (token,))
        for name in expanded:
            if name not in REGISTRY:
                raise KeyError(f"Unknown method or group {token!r}. Known: {', '.join([*groups, *ALL_METHODS])}.")
            if name not in resolved:
                resolved.append(name)
    return resolved


__all__ = [
    "AGENT_METHODS",
    "ALL_METHODS",
    "API_METHODS",
    "CONVENTIONAL_METHODS",
    "LLMP_METHODS",
    "REGISTRY",
    "TSX_AGENT_METHODS",
    "TSX_CONVENTIONAL_METHODS",
    "TSX_LLMP_METHODS",
    "TSX_METHODS",
    "PredictorFactory",
    "build_predictor",
    "resolve_methods",
]
