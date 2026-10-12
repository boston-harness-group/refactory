"""Settle an owned CLI through its shutdown lifecycle before returning failure."""
from __future__ import annotations

import os
import signal
import subprocess
import threading
from contextlib import contextmanager
from typing import Any, Iterator

# DSH's own root-disposal deadline is five seconds.
SHUTDOWN_GRACE_S = 10


@contextmanager
def _termination_as_exit() -> Iterator[None]:
    """Let a CLI's default SIGTERM path unwind its owned subprocess first."""
    main = threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGTERM) if main else None
    changed = main and previous == signal.SIG_DFL

    def terminate(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    if changed:
        signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        if changed:
            signal.signal(signal.SIGTERM, previous)


def shutdown(proc: subprocess.Popen, *, label: str = 'model runtime') -> None:
    """Request runtime-owned disposal; a forced exit reports unconfirmed cleanup.

    Only this direct process is signalled. The selected runtime owns its managed
    tool processes. No PID discovery or unrelated process groups are involved.
    """
    try:
        proc.terminate()
    except ProcessLookupError:
        pass
    try:
        proc.communicate(timeout=SHUTDOWN_GRACE_S)
    except subprocess.TimeoutExpired as exc:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        # Escaped descendants may retain pipes; never wait on them indefinitely.
        try:
            proc.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.stderr is not None:
                proc.stderr.close()
            proc.wait(timeout=1)
        raise RuntimeError(
            f'{label} did not finish shutdown within {SHUTDOWN_GRACE_S}s; '
            'forced exit; child cleanup unconfirmed; not retried'
        ) from exc


def run_owned(argv: list[str], *, timeout: float | None = None,
              label: str = 'DSH runtime', **kwargs: Any) -> subprocess.CompletedProcess:
    """Run one owned CLI, preserving inherited terminal I/O and failure truth.

    POSIX isolates its signals so a terminal interrupt reaches this owner once;
    the owner forwards a single shutdown request and awaits disposal. CLI calls
    on the main thread also unwind default SIGTERM. Existing custom/ignored
    handlers and worker-thread signal policy remain the caller's responsibility.
    """
    with _termination_as_exit():
        proc = subprocess.Popen(argv, start_new_session=os.name == 'posix', **kwargs)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except BaseException:
            shutdown(proc, label=label)
            raise
    return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)
