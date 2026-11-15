"""Helpers and fixtures the design-worktree tests share (`test_design_worktree*.py`).

The tests were one file; they are split so CI can run them on separate
workers. Nothing here is collected.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import pytest

from refactory_dsh import (cli, console, design_ops, interactive, lowering, nn, projects, registry, repo_change,
                           worktree)
from refactory_dsh import run as node_runner
from refactory_dsh.create import CreateMode as CreateModeForGraph
from refactory_dsh.design import DesignMode
from refactory_dsh.fast import DesignFastMode
from refactory_dsh.nn import Verdict
from refactory_dsh.paths import REPO_ROOT
from refactory_dsh.projects import projects_dir
from tests_support import (DESIGN_DOC_ANSWER, PITFALLS_ANSWER,
                           RESEARCH_ANSWER, VERIFICATION_ANSWER,
                           REVIEW_ANSWERS, FakeModel,
                           StaleEmitModel, git, git_repo, render, stub)

NAME = "worktree_tool"


# ── graph edges for audit reloop and refused emission ──────────────────────


#: A design run plans its delivery by asking gh for its login, so it asks the fake (dm-271).
pytestmark = pytest.mark.usefixtures("fake_gh")


@pytest.fixture
def repo(tmp_path) -> Path:
    return git_repo(tmp_path / "repo")


def _answers(*emits: str) -> list[str]:
    """Every answer a design run needs, in the order its graph asks.

    pick_corrections is a gate (no model answer needed).  When there are no
    corrections, pick_corrections returns "skip" and apply_corrections does
    not run.
    """
    return [RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
            DESIGN_DOC_ANSWER, *emits,
            *REVIEW_ANSWERS[:3], VERIFICATION_ANSWER, REVIEW_ANSWERS[3]]


def _mode(**knobs) -> DesignMode:
    """Design mode with some of its knobs set, as a person sets them for one run."""
    mode = DesignMode()
    state = mode.state_dict()
    for name, value in knobs.items():
        state["knobs"][name]["value"] = value
    mode.load_state_dict(state)
    return mode


def _design(workdir: Path, model: FakeModel, mode: DesignMode | None = None, **task: str):
    return (mode or DesignMode()).run(
        workdir, model=model, verdicts={"gate_strategy": "proceed"},
        task={"name": NAME, "brief": "a tool", "check": "python tool.py", **task},
    )


def _report(workdir: Path) -> dict:
    return json.loads((workdir / "design.json").read_text())


def _edit(name: str, *pairs: tuple[str, str]) -> str:
    """An ``===EDIT:`` block: each pair is the lines to find and the lines put in their place."""
    body = "".join(f"{repo_change.SEARCH}\n{search}{repo_change.DIVIDER}\n{replacement}{repo_change.REPLACE}\n"
                   for search, replacement in pairs)
    return f"===EDIT: {name}===\n{body}"


def _commit(repo: Path, name: str, text: str) -> None:
    (repo / name).write_text(text)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", f"add {name}")


class _WorkspaceBuilder(FakeModel):
    """A builder at work in its workspace: ``first`` and ``retry`` are what it does there
    on its first and second dispatch, beside its scripted emission's changes."""

    def __init__(self, answers: list[str], first: Callable[[Path], object],
                 retry: Callable[[Path], object] = lambda _: None) -> None:
        super().__init__(answers)
        self.steps = [first, retry]

    def complete(self, prompt: str, *, system: str = "", workdir: object = None, cwd: object = None,
                 env: object = None, confine: bool = False) -> str:
        answer = super().complete(prompt, system=system, workdir=workdir, cwd=cwd, env=env, confine=confine)
        if cwd is not None and "Write the artifact `" in prompt and self.steps:
            self.steps.pop(0)(Path(str(cwd)))
        return answer


def _binary(clone: Path) -> None:
    (clone / "logo.png").write_bytes(b"\x89PNG\0\0")


def _symlink(clone: Path) -> None:
    (clone / "link.py").symlink_to("app.py")


def _executable(clone: Path) -> None:
    (clone / "app.py").chmod(0o755)


def _bidi(clone: Path) -> None:
    (clone / "tool\u202e.py").write_text("x = 1\n")


def _undo(*names: str) -> Callable[[Path], None]:
    def undo(clone: Path) -> None:
        for name in names:
            path = clone / name
            if path.is_symlink() or not path.exists():
                path.unlink(missing_ok=True)
            elif name == "app.py":
                path.chmod(0o644)
            else:
                path.unlink()
    return undo


def _applied(text: str, answer: str, name: str = "models.md") -> tuple[str, int]:
    """An emission's edits to one file, parsed as materialise parses them, then applied to ``text``."""
    _files, edits, _deletes = repo_change._emitted(answer, edits=True)
    return repo_change._edited(Path(name), text, edits[Path(name)], f"git show HEAD:{name}")


#: models.md keeps each paragraph on one line; b13b's refused edit searched mid-paragraph.
PARAGRAPH = ("the refused emission does not count against `check_attempts`; the gate's own iteration "
             "limit still bounds retries. An emission that changes nothing is refused back to emit.\n")
SENTENCE = "the gate's own iteration limit still bounds retries."
ADDED = " A CEO redo keeps the count."


# ── the boundaries a run keeps ─────────────────────────────────────────────
class _EditingModel(FakeModel):
    """The #319 worker: edits a checkout directly, then answers that it is already done."""

    def __init__(self, answers: list[str], edit: Callable[[], Path]) -> None:
        super().__init__(answers)
        self.edit = edit

    def complete(self, prompt: str, *, system: str = "", workdir: object = None, **workspace: object) -> str:
        answer = super().complete(prompt, system=system, workdir=workdir, **workspace)
        if "Write the artifact `" in prompt:
            (self.edit() / "app.py").write_text("print('edited in place')\n")
        return answer


class _HookChangingModel(FakeModel):
    """An agent that rewrites a shared hook while a planning step runs (#347)."""

    def __init__(self, answers: list[str], target_fn) -> None:
        super().__init__(answers)
        self._target_fn = target_fn

    def complete(self, prompt: str, *, system: str = "", workdir: object = None, **workspace: object) -> str:
        answer = super().complete(prompt, system=system, workdir=workdir, **workspace)
        if "You are the research director" in prompt:
            import subprocess
            target = self._target_fn()
            hooks = Path(subprocess.run(
                ['git', '-C', str(target), 'rev-parse', '--path-format=absolute',
                 '--git-common-dir'], capture_output=True, text=True,
                check=True).stdout.strip()) / 'hooks'
            hooks.mkdir(parents=True, exist_ok=True)
            (hooks / 'pre-commit').write_text('#!/bin/sh\nexit 0\n')
        return answer


# ── design_fast, on the same steps (`repo_change`) ─────────────────────────────
#: What design_fast's strategist and focused QA answer; the coder's answers are each test's own.
FAST_PLAN = "Change only the requested file."
FAST_QA = json.dumps({"status": "pass", "checks": [{"name": "runs", "command": "python tool.py",
                                                    "status": "pass", "evidence": "exit 0"}],
                      "findings": [], "not_run": []})


def _fast(workdir: Path, *emits: str, refused: int = 0, knobs: dict | None = None, **task: str):
    """A design_fast run: the strategist, then each emission and, once one applies, its focused QA.

    The first ``refused`` emissions are ones materialise cannot apply, so no QA follows them.
    """
    mode = DesignFastMode()
    state = mode.state_dict()
    for name, value in (knobs or {}).items():
        state["knobs"][name]["value"] = value
    mode.load_state_dict(state)
    answers = [FAST_PLAN] + [answer for index, emit in enumerate(emits)
                             for answer in ((emit,) if index < refused
                                            else (emit, FAST_QA, VERIFICATION_ANSWER))]
    return mode.run(workdir, model=FakeModel(answers),
                    task={"name": NAME, "brief": "a tool", "check": "python tool.py", **task})


def _self_checks(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "self-check-record.jsonl").read_text().splitlines()]


# ── dm-208: a control or format character in a path is refused by name ─────
#: One character of each class a path may not hold, and how a refusal names it.
UNPRINTABLE = [("\x7f", "U+007F (a control character)"), ("\x9b", "U+009B (a control character)"),
               ("\u202e", "U+202E RIGHT-TO-LEFT OVERRIDE"), ("\u200f", "U+200F RIGHT-TO-LEFT MARK"),
               ("\u2067", "U+2067 RIGHT-TO-LEFT ISOLATE"), ("\u061c", "U+061C ARABIC LETTER MARK"),
               ("\u200b", "U+200B ZERO WIDTH SPACE"), ("\u2060", "U+2060 WORD JOINER"),
               ("\ufeff", "U+FEFF ZERO WIDTH NO-BREAK SPACE")]
UNPRINTABLE_IDS = ["DEL", "C1-control", "bidi-override", "bidi-mark", "bidi-isolate", "arabic-mark",
                   "zero-width-space", "word-joiner", "byte-order-mark"]


# Every name above, underscored ones too, for the test files' star import.
__all__ = [name for name in dir() if not name.startswith("__") and name != "pytestmark"]
