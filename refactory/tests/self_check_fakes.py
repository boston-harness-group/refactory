"""The self-check's chunks and clock, as `test_self_check._Chunks` answers them, in
whichever process runs the self-check: the test's, or the op child the walker runs it in
(`child_patches`). Their state -- the failing ids, the slow files, the clock, the commits
seen -- lives in the JSON file ``$SELF_CHECK_FAKES``, so every process shares it.

This module is also the self-check's ``time``: :func:`monotonic` is the fake clock.
"""
from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any

from child_patches import real

STATE_ENV = "SELF_CHECK_FAKES"


def load() -> dict[str, Any]:
    return json.loads(Path(os.environ[STATE_ENV]).read_text())


def save(state: dict[str, Any]) -> None:
    Path(os.environ[STATE_ENV]).write_text(json.dumps(state))


def monotonic() -> float:
    """The chunks' clock: it moves only when a chunk runs past its time."""
    return float(load()["now"])


def run_check(ctx: Any, command: str, timeout: float, **kwargs: Any) -> Any:
    """`repo_change.run_check`: a self-check's chunk (``prepared``) is answered; the whole
    check and the base's runs run for real."""
    from refactory_dsh import repo_change

    if not kwargs.get("prepared"):
        return real(repo_change, "run_check")(ctx, command, timeout, **kwargs)
    state = load()
    commit = str(ctx.facts.get("commit"))
    if commit not in state["commits"]:
        state["commits"].append(commit)
        if state["emissions"] is not None:
            state["fails"] = sorted(state["emissions"][len(state["commits"]) - 1])
        save(state)
    # In the test's own process the test's `_Chunks` answers, so a subclass can change how.
    return (INSTANCE or answer)(command, timeout)


#: The test's `_Chunks`, in the test's own process; an op child has none.
INSTANCE: Any = None


def answer(command: str, timeout: float) -> Any:
    """What pytest and the sandbox would answer for one chunk."""
    from refactory_dsh.sandbox import SandboxResult

    state = load()
    fails, slow = set(state["fails"]), set(state["slow"])
    named = [word for word in shlex.split(command) if word.endswith(".py") or ".py::" in word]
    # Full command (no individual .py files named): all failures apply.
    if not named:
        failed = sorted(fails)
        if failed:
            return SandboxResult(ok=False, returncode=1, elapsed_s=0.1, stdout="".join(
                f"FAILED {test_id} - AssertionError\n" for test_id in failed)
                + f"{len(failed)} failed in 0.10s")
        return SandboxResult(ok=True, returncode=0, stdout="all passed in 0.10s", elapsed_s=0.1)
    if any(name.split("::")[0] in slow for name in named):
        state["now"] = float(state["now"]) + timeout
        save(state)
        return SandboxResult(ok=False, returncode=124, timed_out=True, timeout_s=int(timeout),
                             elapsed_s=timeout, stderr=f"timed out after {int(timeout)}s")
    failed = sorted(test_id for test_id in fails if test_id in named or test_id.split("::")[0] in named)
    if failed:
        return SandboxResult(ok=False, returncode=1, elapsed_s=0.1, stdout="".join(
            f"FAILED {test_id} - AssertionError\n" for test_id in failed)
            + f"{len(failed)} failed in 0.10s")
    return SandboxResult(ok=True, returncode=0, stdout=f"{len(named)} passed in 0.10s", elapsed_s=0.1)


def record_prepass(prev_record: dict[str, Any], command: str, run_fn: Any, timeout: float,
                   **kw: Any) -> Any:
    """`failfast.targeted_prepass`, each call's last status and keywords recorded."""
    from refactory_dsh import failfast

    with Path(os.environ[STATE_ENV]).with_suffix(".prepass.jsonl").open("a") as handle:
        handle.write(json.dumps({"status": prev_record.get("status"), "kw": sorted(kw)}) + "\n")
    return real(failfast, "targeted_prepass")(prev_record, command, run_fn, timeout, **kw)


def prepass_calls() -> list[tuple[Any, list[str]]]:
    """The calls :func:`record_prepass` saw, as ``(last status, keyword names)``."""
    path = Path(os.environ[STATE_ENV]).with_suffix(".prepass.jsonl")
    lines = path.read_text().splitlines() if path.exists() else []
    return [(entry["status"], entry["kw"]) for entry in map(json.loads, lines)]
