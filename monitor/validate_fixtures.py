"""Validate every monitor fixture against the versioned JSON schemas.

This is the executable half of the data contract: it proves the mock fixtures
in ``monitor/site/data/`` conform exactly to ``monitor/schemas/*.schema.json``,
and it is the same check the live harness team should run against real output
before committing it.

The repo's unit-test workflow only collects tests under ``aieng-forecasting/tests``
and ``implementations/tests`` (see ``.github/workflows/unit_tests.yml``), so this
file is not picked up by that job without a CI change. It is therefore wired two
ways instead:

* as a **standalone script** — ``uv run python monitor/validate_fixtures.py``
  (exits non-zero on any failure), suitable for a pre-commit hook or the
  ``deploy-monitor`` workflow's pre-deploy gate;
* as a **pytest module** — ``uv run pytest monitor/validate_fixtures.py`` — for
  anyone who points pytest at it directly.

Requires ``jsonschema`` (already in the project's dev dependencies).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource


MONITOR_DIR = Path(__file__).resolve().parent
SCHEMA_DIR = MONITOR_DIR / "schemas"
DATA_DIR = MONITOR_DIR / "site" / "data"

# Short-name -> schema $id, so tests and the CLI can refer to schemas by name.
SCHEMA_IDS: dict[str, str] = {
    "prediction": "https://raw.githubusercontent.com/VectorInstitute/agentic-forecasting-live/main/monitor/schemas/prediction.schema.json",
    "resolution": "https://raw.githubusercontent.com/VectorInstitute/agentic-forecasting-live/main/monitor/schemas/resolution.schema.json",
    "leaderboard": "https://raw.githubusercontent.com/VectorInstitute/agentic-forecasting-live/main/monitor/schemas/leaderboard.schema.json",
    "gap_log": "https://raw.githubusercontent.com/VectorInstitute/agentic-forecasting-live/main/monitor/schemas/gap_log.schema.json",
    "mutation_event": "https://raw.githubusercontent.com/VectorInstitute/agentic-forecasting-live/main/monitor/schemas/mutation_event.schema.json",
    "manifest": "https://raw.githubusercontent.com/VectorInstitute/agentic-forecasting-live/main/monitor/schemas/manifest.schema.json",
    "forecast_bundle": "https://raw.githubusercontent.com/VectorInstitute/agentic-forecasting-live/main/monitor/schemas/forecast_bundle.schema.json",
}


def load_json(path: Path) -> object:
    """Parse a JSON file.

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    object
        The decoded JSON content.
    """
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_registry() -> Registry:
    """Build a referencing registry from every schema in ``monitor/schemas``.

    Cross-file ``$ref`` targets (the forecast bundle references the prediction
    and resolution schemas by ``$id``) resolve through this registry.

    Returns
    -------
    Registry
        A referencing registry containing all monitor schemas.
    """
    resources = []
    for schema_path in sorted(SCHEMA_DIR.glob("*.schema.json")):
        schema = load_json(schema_path)
        resources.append(Resource.from_contents(schema))
    return Registry().with_resources((resource.id(), resource) for resource in resources)


def validator_for(schema_name: str, registry: Registry) -> Draft202012Validator:
    """Return a format-checking validator for a named schema.

    Parameters
    ----------
    schema_name : str
        Key into :data:`SCHEMA_IDS`.
    registry : Registry
        Registry that can resolve the schema and its references.

    Returns
    -------
    Draft202012Validator
        A validator bound to the resolved schema.
    """
    schema = registry.get_or_retrieve(SCHEMA_IDS[schema_name]).value.contents
    return Draft202012Validator(schema, registry=registry, format_checker=Draft202012Validator.FORMAT_CHECKER)


def _errors(validator: Draft202012Validator, instance: object, label: str) -> list[str]:
    """Collect formatted validation errors for one instance.

    Parameters
    ----------
    validator : Draft202012Validator
        Validator to run.
    instance : object
        The JSON instance to check.
    label : str
        Human-readable label identifying the instance in messages.

    Returns
    -------
    list[str]
        One string per validation error (empty when valid).
    """
    messages = []
    for error in sorted(validator.iter_errors(instance), key=lambda e: list(e.path)):
        location = "/".join(str(part) for part in error.path) or "<root>"
        messages.append(f"{label} :: {location}: {error.message}")
    return messages


def collect_failures() -> list[str]:
    """Validate all fixtures and return a flat list of failure messages.

    Returns
    -------
    list[str]
        Empty when every fixture conforms to its schema.
    """
    registry = build_registry()
    failures: list[str] = []

    manifest_validator = validator_for("manifest", registry)
    failures += _errors(manifest_validator, load_json(DATA_DIR / "manifest.json"), "manifest.json")

    leaderboard_validator = validator_for("leaderboard", registry)
    failures += _errors(leaderboard_validator, load_json(DATA_DIR / "leaderboard.json"), "leaderboard.json")

    gap_validator = validator_for("gap_log", registry)
    gaps = load_json(DATA_DIR / "gaps.json")
    assert isinstance(gaps, dict)
    for i, entry in enumerate(gaps["gaps"]):
        failures += _errors(gap_validator, entry, f"gaps.json[{i}]")

    mutation_validator = validator_for("mutation_event", registry)
    mutations = load_json(DATA_DIR / "mutations.json")
    assert isinstance(mutations, dict)
    for i, entry in enumerate(mutations["mutations"]):
        failures += _errors(mutation_validator, entry, f"mutations.json[{i}]")

    bundle_validator = validator_for("forecast_bundle", registry)
    bundle_paths = sorted((DATA_DIR / "forecasts").glob("*.json"))
    if not bundle_paths:
        failures.append("forecasts/: no per-origin bundles found")
    for path in bundle_paths:
        failures += _errors(bundle_validator, load_json(path), f"forecasts/{path.name}")

    return failures


def test_fixtures_conform_to_schemas() -> None:
    """Pytest entry point: every fixture validates against its schema."""
    failures = collect_failures()
    assert not failures, "Fixture schema violations:\n" + "\n".join(failures)


def main() -> int:
    """Run validation as a script.

    Returns
    -------
    int
        ``0`` when all fixtures conform, ``1`` otherwise.
    """
    failures = collect_failures()
    if failures:
        print(f"FAIL: {len(failures)} fixture schema violation(s):")
        for message in failures:
            print(f"  - {message}")
        return 1
    bundle_count = len(list((DATA_DIR / "forecasts").glob("*.json")))
    print(f"OK: all fixtures valid (manifest, leaderboard, gaps, mutations, {bundle_count} forecast bundles).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
