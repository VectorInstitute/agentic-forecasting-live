# Honesty & the submission-time trust model

The live harness commits each day's forecasts to a **public fork**
(`VectorInstitute/agentic-forecasting-live`). The workshop paper claims that this
commit history proves *when* each prediction was submitted — i.e. that a forecast
for horizon *h* existed **before** the outcome was known. This document states
exactly what is and is not proven, and what an auditor should check.

## Why commit dates are not the anchor

Git author and committer dates are **client-supplied and trivially forgeable**:

```bash
git commit --date="2020-01-01T00:00:00Z" ...          # author date
GIT_COMMITTER_DATE="2020-01-01T00:00:00Z" git commit ... # committer date
```

Both land in the object and survive a push unchanged. So a bare commit history —
however tidy — is **not** evidence of submission time. Anyone with push access
could have written any date. The `submission_timestamp` field *inside* each log
record is likewise just a string the harness wrote; on its own it proves nothing.

## The anchor: server-side Release `created_at`

GitHub makes the fork's owner **not** the timestamping authority. When a push to
`main` touches the live log path, `.github/workflows/attest-live-log.yml` cuts a
**GitHub Release** tagged `attest/<UTC date>-<short sha>`. A Release carries a
`created_at` field that **GitHub sets server-side at creation time** and that a
client cannot back-date. That server timestamp is the trust anchor.

Because the Release is created by CI *in reaction to* the push, its `created_at`
is a tight upper bound on when the committed forecast became public: the record
provably existed no later than that server time.

## What an auditor checks

For any forecast record in the log:

1. **Find its attestation.** Locate the Release whose head commit (in the body,
   and as the tag's target) contains that record. Read the Release `created_at`
   from the GitHub API or UI — *not* from any date in the git object.
2. **Compare against the record's `submission_timestamp`.** The record's
   `submission_timestamp` (UTC, written by the harness) should precede the
   Release `created_at` by only **minutes** — the predict → resolve → aggregate →
   commit → push → CI latency. A gap of minutes is expected; hours or days is a
   red flag worth investigating.
3. **Cross-check the commit subject.** The daily commit subject embeds the same
   UTC submission timestamp after `@` (e.g.
   `live: 2026-07-15 predictions (22 methods) / resolutions (12) @ 2026-07-15T21:32:04Z`).
   This is a convenience cross-check only — it is client-supplied like the commit
   dates, so it corroborates but does **not** anchor.
4. **Confirm the horizon math.** For the submission time to prove foreknowledge
   was impossible, the outcome date for horizon *h* must fall strictly after the
   attested `created_at`. That comparison — not the commit graph — is what rules
   out lookahead.

In short: **the Release `created_at` vs. the record `submission_timestamp` is the
check; commit dates are explicitly not the anchor.**

## Known limitations

- **Existence-by-time, not absence-of-earlier-knowledge.** The attestation proves
  a forecast existed *no later than* the server timestamp. It cannot prove the
  operator did not privately know or compute the answer *earlier* and merely
  delay publishing. It bounds the *late* side, not the *early* side.
- **Trust in GitHub as timestamping authority.** The anchor is only as trustworthy
  as GitHub's server clock and Release metadata. A compromised or colluding
  platform could in principle mis-set `created_at`. This moves trust off the
  repository owner and onto GitHub — a meaningful improvement, not an absolute.
- **Same-actor CI.** The workflow runs in the same repo it attests. It raises the
  cost of forgery (you would have to defeat the platform, not just edit a date)
  but is not a third-party notary.
- **Coverage gaps.** Only pushes that touch the log path trigger an attestation.
  A record committed without triggering CI (e.g. path filter drift) would be
  unattested; the `paths:` filter is kept in sync with `log_dir` in
  `live_config.yaml` precisely to avoid this.

## Possible future strengthening

For an independent (non-GitHub) anchor, the committed log — or a hash of it —
could additionally be stamped with [OpenTimestamps](https://opentimestamps.org/),
which anchors a hash into the Bitcoin blockchain and yields a verifiable proof
that is independent of GitHub and of the repository owner. That would upgrade the
anchor from "trust GitHub's clock" to "trust a public, append-only ledger." It is
noted here as a future option, not a current dependency.
