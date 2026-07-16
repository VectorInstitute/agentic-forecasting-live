"""Reuse the monitor's schema validator to gate every written record.

The versioned schemas and their ``jsonschema`` registry live in
``monitor/validate_fixtures.py`` (the executable half of the data contract). The
monitor tree is not an installed package, so this module loads that file by path
and reuses its ``build_registry`` / ``validate_instance`` machinery — the harness
validates its real output against exactly the same schemas CI runs on the
fixtures. No schema logic is reimplemented here.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from types import ModuleType
from typing import Any

from workshop_experiments.live.config import REPO_ROOT


#: The monitor's data-contract validator module.
_VALIDATE_FIXTURES = REPO_ROOT / "monitor" / "validate_fixtures.py"


@lru_cache(maxsize=1)
def _monitor_validator() -> ModuleType:
    """Import ``monitor/validate_fixtures.py`` by path (it is not a package)."""
    spec = importlib.util.spec_from_file_location("monitor_validate_fixtures", _VALIDATE_FIXTURES)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load monitor validator from {_VALIDATE_FIXTURES}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _registry() -> Any:
    """Return a cached schema registry built from ``monitor/schemas``."""
    return _monitor_validator().build_registry()


def validate(schema_name: str, instance: Any) -> list[str]:
    """Validate *instance* against a named monitor schema; return error strings.

    Parameters
    ----------
    schema_name : str
        One of ``prediction``, ``resolution``, ``leaderboard``, ``manifest``,
        ``gap_log``, ``forecast_bundle``, ``mutation_event``.
    instance : Any
        The decoded record/aggregate to validate.

    Returns
    -------
    list[str]
        Empty when valid; otherwise one message per schema violation.
    """
    module = _monitor_validator()
    return list(module.validate_instance(schema_name, instance, registry=_registry()))


def check(schema_name: str, instance: Any) -> None:
    """Validate *instance* and raise ``ValueError`` on any schema violation."""
    errors = validate(schema_name, instance)
    if errors:
        raise ValueError(f"{schema_name} record failed schema validation:\n" + "\n".join(errors))


__all__ = ["check", "validate"]
