"""A standard design run launches no more processes than it did when this budget was set.

Each process launch costs under a millisecond on Linux and 30 ms or more on a Mac whose
endpoint-security agent inspects every exec, so launches, not CPU, decide whether a run
and this suite take about as long on both. This test runs one passing design round (a
target checkout, the research and design nodes, one emission, the reviews, the check and
both judges) and counts the processes refactory's own code starts in it, by program. A
change that adds launches to the agent-step path -- a git command per snapshot, a
subprocess per file -- fails here, on Linux CI as well as on a Mac, before it makes the
Mac several times slower. Raise a budget only with a reason in the commit.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import pytest

from refactory_dsh import spawn_count
from test_qa_judge import _answers, _design, _round, _tool
from tests_support import FakeModel, git_repo, overwatch_judgment, qa_judgment

#: What one passing design round may launch from refactory's own code, by program, and in
#: all. Measured on 2026-11-13: 429 git launches with the target guard's seven git commands
#: a snapshot, 225 with three, and nothing else.
BUDGET = {"git": 240}
TOTAL = 250

#: The launches being counted while a test of this module runs; empty otherwise.
_COUNTING: list[Counter] = []


def _from_refactory() -> bool:
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_globals.get("__name__", "").startswith("refactory_dsh"):
            return True
        frame = frame.f_back
    return False


def _hook(event: str, args: tuple) -> None:
    if _COUNTING and event in spawn_count.EVENTS and _from_refactory():
        program, _argv = spawn_count._program(event, args)
        _COUNTING[-1][os.path.basename(program)] += 1


# An audit hook cannot be removed; it counts only while a test here holds `_COUNTING`.
sys.addaudithook(_hook)


@pytest.fixture
def launches():
    counted: Counter = Counter()
    _COUNTING.append(counted)
    try:
        yield counted
    finally:
        _COUNTING.remove(counted)


@pytest.mark.usefixtures("fake_gh")
def test_a_passing_design_round_stays_within_its_launch_budget(tmp_path, launches):
    repo = git_repo(tmp_path / "repo")
    model = FakeModel(_answers(_round(_tool("built"), judge=qa_judgment("proceed"),
                                      overwatch=overwatch_judgment("pass"))))
    result = _design(tmp_path / "run", model, repo)
    assert result.exit_ok, result.error

    found = dict(launches.most_common())
    over = {program: (found.get(program, 0), cap) for program, cap in BUDGET.items()
            if found.get(program, 0) > cap}
    assert not over and sum(found.values()) <= TOTAL, (
        f"a passing design round launched {sum(found.values())} processes (budget {TOTAL}): "
        f"{found}; over budget: {over}. Count where they come from with "
        f"REFACTORY_SPAWN_LOG=<file> (refactory_dsh.spawn_count).")
