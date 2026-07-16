#!/usr/bin/env bash
# Wrapper that launchd (or cron) invokes for one live harness run.
#
# Sources the repo `.env` so model/proxy credentials are present, then runs one
# `ws-live-run` cycle. Kept intentionally thin: all logic lives in the CLI.
#
# Usage: ops/run.sh [extra ws-live-run args...]
set -euo pipefail

# Repo root = three levels up from this script
# (workshop_experiments/workshop_experiments/live/ops/run.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

cd "${REPO_ROOT}"

# Load repo credentials/config if present (never fail if absent).
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

# `uv run` resolves the workspace venv; the console script is `ws-live-run`.
exec uv run --project "${REPO_ROOT}/aieng-forecasting" ws-live-run "$@"
