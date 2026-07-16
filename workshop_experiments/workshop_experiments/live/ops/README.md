# Live harness ops — scheduling `ws-live-run`

The live harness runs once per trading day, shortly after the US market close,
via the `ws-live-run` CLI. This directory holds a `launchd` template and a thin
wrapper so a macOS host can drive it on a schedule.

> These are **templates**. Nothing here installs or starts anything — follow the
> steps below deliberately on the deployment host.

## Files

| File | Purpose |
|---|---|
| `run.sh` | Sources the repo `.env`, then runs one `ws-live-run` cycle. |
| `ai.vectorinstitute.ws-live-run.plist` | `launchd` job template (Mon–Fri 17:30). |
| [`HONESTY.md`](HONESTY.md) | Submission-time trust model: why commit dates are not the anchor, and what an auditor checks. |

## What one run does

`trading-day check → predict → resolve → aggregate → commit → push`, guarded by a
single-run lockfile (`live/log/../.ws-live-run.lock`). A non-session weekday
(NYSE holiday) exits cleanly as a *non-session day* — **not** a gap. A method
that fails after its retries becomes a per-method gap-log entry and the run
continues.

The daily commit subject carries the UTC submission timestamp after `@`. On the
`live` fork, the push then triggers a server-timestamped attestation Release
(`.github/workflows/attest-live-log.yml`) — the actual proof of submission time.
See [`HONESTY.md`](HONESTY.md) for why commit dates are not the trust anchor.

Verify the plan first, without any writes/API/network:

```bash
uv run --project aieng-forecasting ws-live-run --dry-run
```

Exercise the whole pipeline offline (committed smoke data, no API/network):

```bash
uv run --project aieng-forecasting ws-live-run --simulate --no-push
```

## Timezone note (important)

`launchd`'s `StartCalendarInterval` fires against the **host's local wall-clock
time**, not a named zone. The submission target is **17:30 America/Toronto**, so
either:

- set the deployment host's timezone to `America/Toronto` (recommended), or
- adjust the `Hour`/`Minute` in the plist to whatever local time equals 17:30
  Toronto on that host, keeping DST in mind.

The config (`live_config.yaml`) records `17:30 America/Toronto` as the canonical
intent; the host schedule must match it.

## Install (macOS `launchd`)

```bash
REPO_ROOT="/absolute/path/to/agentic-forecasting"   # no trailing slash
PLIST_SRC="$REPO_ROOT/workshop_experiments/workshop_experiments/live/ops/ai.vectorinstitute.ws-live-run.plist"
PLIST_DST="$HOME/Library/LaunchAgents/ai.vectorinstitute.ws-live-run.plist"

# Fill in the repo path and install.
sed "s#__REPO_ROOT__#$REPO_ROOT#g" "$PLIST_SRC" > "$PLIST_DST"
chmod +x "$REPO_ROOT/workshop_experiments/workshop_experiments/live/ops/run.sh"

# Load and enable.
launchctl load  "$PLIST_DST"
launchctl enable "gui/$(id -u)/ai.vectorinstitute.ws-live-run"
```

Trigger a one-off run immediately (e.g. to validate the install):

```bash
launchctl start ai.vectorinstitute.ws-live-run
```

## Inspect

```bash
# Is the job registered and what was its last exit code?
launchctl list | grep ws-live-run

# Wrapper stdout / stderr:
tail -f "$REPO_ROOT/workshop_experiments/workshop_experiments/live/log/ws-live-run.out.log"
tail -f "$REPO_ROOT/workshop_experiments/workshop_experiments/live/log/ws-live-run.err.log"
```

## Uninstall

```bash
launchctl unload "$HOME/Library/LaunchAgents/ai.vectorinstitute.ws-live-run.plist"
rm "$HOME/Library/LaunchAgents/ai.vectorinstitute.ws-live-run.plist"
```

## Health check — seeing missed days

Two independent signals:

1. **The gap log is the record of truth.** Missed methods/days are committed as
   gap-log entries and surfaced on the monitor (and in
   `monitor/site/data/gaps.json`). Because gaps are committed, absence of a
   commit for a trading day is itself visible in `git log` of the `live` remote.
2. **launchd state.** `launchctl list | grep ws-live-run` shows the last exit
   status; a non-zero status with no fresh commit means the run failed before it
   could log gaps — check `ws-live-run.err.log`.

A *non-session day* is expected and silent (logged to stdout, no gap, no commit).
A *missed session* shows up as either a gap-log entry (partial failure) or a
missing daily commit (total failure) — never as a silent backfill.
