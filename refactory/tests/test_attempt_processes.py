"""Ending what a timed-out attempt left running (`attempt_processes`), on Linux and macOS.

These drive real processes this test starts; each test ends only the PIDs it started.
The retry paths on both runtimes are in test_node_retry
(`test_kept_workspace_retry_ends_the_timed_out_attempts_processes_first`).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from refactory_dsh import attempt_processes

SLEEP = "import time; time.sleep(300)"


def _kill(pids: list[int]) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + 30
    while not (path.exists() and path.read_text().strip()):
        assert time.monotonic() < deadline, f"{path} never written"
        time.sleep(0.05)


def test_a_process_working_in_the_workspace_is_found_and_ended(tmp_path):
    root = tmp_path / "workspace" / "repo" / "sub"
    root.mkdir(parents=True)
    outside = subprocess.Popen([sys.executable, "-c", SLEEP], cwd=tmp_path)
    inside = subprocess.Popen([sys.executable, "-c", SLEEP], cwd=root, start_new_session=True)
    try:
        time.sleep(0.3)
        found = attempt_processes.find(tmp_path / "workspace")
        assert inside.pid in found and outside.pid not in found
        assert attempt_processes.end(tmp_path / "workspace") == []
        assert attempt_processes._gone(inside.pid)
        assert outside.poll() is None
    finally:
        _kill([inside.pid, outside.pid])
        inside.wait(timeout=10)
        outside.wait(timeout=10)


def test_a_process_in_the_attempts_session_is_found_wherever_it_works(tmp_path):
    """The attempt's session identifies a process that works outside the workspace."""
    root = tmp_path / "workspace"
    root.mkdir()
    pid_file = tmp_path / "child.pid"
    leader = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess, sys, pathlib, time\n"
         f"child = subprocess.Popen([sys.executable, '-c', {SLEEP!r}], cwd={str(tmp_path)!r})\n"
         f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
         "time.sleep(300)\n"],
        cwd=tmp_path, start_new_session=True)
    try:
        _wait_for(pid_file)
        child = int(pid_file.read_text())
        assert child not in attempt_processes.find(root)
        assert child in attempt_processes.find(root, session=leader.pid)
        assert attempt_processes.end(root, session=leader.pid) == []
        assert attempt_processes._gone(child)
    finally:
        _kill([leader.pid] + ([int(pid_file.read_text())] if pid_file.exists() else []))
        leader.wait(timeout=10)


def test_a_zombie_counts_as_ended_and_the_wait_does_not_run_out(tmp_path):
    """A process whose parent never reaps it stays a zombie after SIGTERM: it runs nothing,
    so it counts as ended, at once rather than after the bounded wait."""
    root = tmp_path / "workspace"
    root.mkdir()
    pid_file = tmp_path / "child.pid"
    # The parent works outside the workspace and in another session, so only its child is
    # the attempt's; it never waits for the child, which becomes a zombie when it ends.
    parent = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess, sys, pathlib, time\n"
         f"child = subprocess.Popen([sys.executable, '-c', {SLEEP!r}], cwd={str(root)!r}, "
         "start_new_session=True)\n"
         f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
         "time.sleep(300)\n"],
        cwd=tmp_path)
    try:
        _wait_for(pid_file)
        child = int(pid_file.read_text())
        time.sleep(0.3)
        assert attempt_processes.find(root) == [child]
        started = time.monotonic()
        assert attempt_processes.end(root, term_grace_s=5, kill_wait_s=5) == []
        assert time.monotonic() - started < 4
        assert attempt_processes._state(child) == "Z"
    finally:
        _kill([parent.pid])
        parent.wait(timeout=10)


def test_a_process_in_a_sibling_workspace_with_a_lookalike_name_is_left_alone(tmp_path):
    """``emit2`` beside ``emit`` is another node's workspace: its process is not ``emit``'s."""
    root = tmp_path / "workspaces" / "emit"
    sibling = tmp_path / "workspaces" / "emit2"
    root.mkdir(parents=True)
    sibling.mkdir()
    other = subprocess.Popen([sys.executable, "-c", SLEEP], cwd=sibling, start_new_session=True)
    try:
        time.sleep(0.3)
        assert other.pid in attempt_processes.find(sibling)
        assert attempt_processes.find(root) == []
        assert attempt_processes.end(root) == []
        assert other.poll() is None
    finally:
        _kill([other.pid])
        other.wait(timeout=10)
