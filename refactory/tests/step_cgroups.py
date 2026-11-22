"""The step cgroups a test makes, released when the test ends.

Where a test runs in a delegated cgroup, each sealed step it prepares gets a cgroup of
its own (`seal_cgroup.make`). A step that is collected, or prepared through
`workspace.dispatched`, has it removed then; a test helper that only prepares a step
(`workspace.prepare`) and never collects it would leave it, empty, until the test unit
stops. :func:`track` records every step cgroup made while a test runs, and
:func:`release` ends what is still sealed in each workspace and removes its cgroup, as
the run's end does (`workspace.release_cgroup`). The conftest does both for every test.
"""

from __future__ import annotations

from typing import Any


def track(monkeypatch: Any) -> list[tuple[str, str]]:
    """Record each step cgroup made from now on: (its workspace, its path)."""
    from refactory_dsh import seal_cgroup

    made: list[tuple[str, str]] = []
    real = seal_cgroup.make

    def tracked(workspace: Any, *args: Any, **kwargs: Any) -> tuple[str, str]:
        path, held = real(workspace, *args, **kwargs)
        made.append((str(workspace), path))
        return path, held

    monkeypatch.setattr(seal_cgroup, "make", tracked)
    return made


def release(made: list[tuple[str, str]]) -> None:
    """End what is still sealed in each recorded workspace, then remove its cgroup."""
    from refactory_dsh import workspace

    while made:
        sealed, path = made.pop()
        workspace.release_cgroup(sealed, path)
