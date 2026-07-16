"""Commit-message formatting and local staging/commit behavior (offline).

Push is never exercised (no network / no remote); these tests cover the message
contract and that only the given paths are staged and committed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from workshop_experiments.live.gitops import commit_message, stage_and_commit


def test_commit_message_format() -> None:
    """The daily commit subject matches the sanctioned format.

    The UTC submission timestamp is appended after ``@`` as a human-readable
    cross-check against the attestation Release's server-set ``created_at``.
    """
    assert commit_message("2026-07-15", 22, 12, "2026-07-15T21:32:04Z") == (
        "live: 2026-07-15 predictions (22 methods) / resolutions (12) @ 2026-07-15T21:32:04Z"
    )


def test_commit_message_includes_submission_timestamp() -> None:
    """The exact UTC submission timestamp appears verbatim in the subject."""
    subject = commit_message("2026-01-02", 6, 0, "2026-01-02T22:05:59Z")
    assert subject.endswith(" @ 2026-01-02T22:05:59Z")


def _init_repo(root: Path) -> None:
    """Initialise a throwaway git repo with an identity."""
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)


def test_stage_and_commit_only_given_paths(tmp_path: Path) -> None:
    """Only the named paths are staged; an untracked sibling stays out."""
    _init_repo(tmp_path)
    tracked = tmp_path / "log"
    tracked.mkdir()
    (tracked / "a.json").write_text("{}\n")
    (tmp_path / "unrelated.txt").write_text("nope\n")

    created = stage_and_commit(tmp_path, [tracked], "live: test")
    assert created is True

    tree = subprocess.run(
        ["git", "show", "--name-only", "--pretty=format:", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert "log/a.json" in tree
    assert "unrelated.txt" not in tree


def test_stage_and_commit_noop_when_clean(tmp_path: Path) -> None:
    """Committing with nothing staged returns False (no empty commit)."""
    _init_repo(tmp_path)
    empty = tmp_path / "log"
    empty.mkdir()
    assert stage_and_commit(tmp_path, [empty], "live: test") is False
