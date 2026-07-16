"""The commit + push step: stage exactly the log + aggregates, commit, push.

Stages only the append-only log and the regenerated aggregates, commits with the
sanctioned daily message, and pushes to the ``live`` remote's ``main`` — the
daily commit stream the monitor deploys from. Never force-pushes; on a push
rejection it rebase-pulls once and retries. ``--no-push`` (``push=False``) stops
after the local commit for tests.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def commit_message(origin: str, n_methods: int, n_resolutions: int, submission_timestamp: str) -> str:
    """Return the sanctioned daily commit subject line.

    The UTC ``submission_timestamp`` is folded into the subject as a cheap,
    human-readable cross-check: an auditor can eyeball it against the
    server-side attestation Release's ``created_at`` (see
    ``live/ops/HONESTY.md``). It is a *convenience marker only* — git
    author/committer dates and this string are all client-supplied and
    forgeable, so none of them is the trust anchor. The attestation Release's
    server-set ``created_at`` is.

    Example subject:
    ``live: 2026-07-15 predictions (22 methods) / resolutions (12) @ <ts>``
    where ``<ts>`` is the UTC ``submission_timestamp`` (e.g. ``2026-07-15T21:32:04Z``).
    """
    return f"live: {origin} predictions ({n_methods} methods) / resolutions ({n_resolutions}) @ {submission_timestamp}"


def _git(repo_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git command in *repo_root*, capturing output."""
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=check,
        capture_output=True,
        text=True,
    )


def stage_and_commit(repo_root: Path, paths: list[Path], message: str) -> bool:
    """Stage exactly *paths* and commit with *message*.

    Returns ``True`` if a commit was created, ``False`` if there was nothing to
    commit (all paths already clean). Only the given paths are staged — never
    ``git add -A``.
    """
    rel = [str(p.resolve().relative_to(repo_root.resolve())) for p in paths]
    if rel:
        _git(repo_root, "add", "--", *rel)
    status = _git(repo_root, "status", "--porcelain")
    if not status.stdout.strip():
        return False
    _git(repo_root, "commit", "-m", message)
    return True


def push(repo_root: Path, remote: str = "live", branch: str = "main") -> None:
    """Push to *remote*/*branch*; on rejection, rebase-pull once and retry.

    Never force-pushes. Raises ``subprocess.CalledProcessError`` if the push
    still fails after the single rebase-and-retry.
    """
    first = _git(repo_root, "push", remote, f"HEAD:{branch}", check=False)
    if first.returncode == 0:
        return
    # Rejected (likely non-fast-forward): rebase onto the remote tip, retry once.
    _git(repo_root, "pull", "--rebase", remote, branch)
    _git(repo_root, "push", remote, f"HEAD:{branch}")


__all__ = ["commit_message", "push", "stage_and_commit"]
