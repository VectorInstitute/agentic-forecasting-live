"""A single-run lock so overlapping harness runs cannot interleave writes.

Uses ``fcntl.flock`` on a lock file: the advisory lock is held for the lifetime
of the open file handle and released automatically when the process exits, so a
crashed run never leaves a stale lock behind (unlike a PID file). Acquisition is
non-blocking — a second concurrent run fails fast with :class:`LockHeld` rather
than queueing.
"""

from __future__ import annotations

import errno
import fcntl
import os
from pathlib import Path
from types import TracebackType


class LockHeld(RuntimeError):  # noqa: N818 - a condition, not an error state
    """Raised when the run lock is already held by another process."""


class RunLock:
    """A non-blocking advisory file lock usable as a context manager.

    Examples
    --------
    >>> with RunLock(Path("/tmp/ws-live.lock")):  # doctest: +SKIP
    ...     ...  # exclusive section
    """

    def __init__(self, path: Path) -> None:
        """Create a lock bound to *path* (not yet acquired)."""
        self.path = path
        self._fd: int | None = None

    def acquire(self) -> None:
        """Acquire the lock, or raise :class:`LockHeld` if already held."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise LockHeld(f"run lock already held: {self.path}") from exc
            raise
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.fsync(fd)
        self._fd = fd

    def release(self) -> None:
        """Release the lock if held (idempotent)."""
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> RunLock:
        """Acquire on context entry."""
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Release on context exit."""
        self.release()


__all__ = ["LockHeld", "RunLock"]
