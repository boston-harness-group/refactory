"""Nothing in the check runs twice on the same input.

Each test drives the real ops on a fixture repository with real git and the real pytest,
whose sessions and tests a ``conftest.py`` logs (`LOGGER`): which directory each session
ran in -- the run's worktree, its check copy, the base worktree -- with what arguments,
and which tests ran there. The counts are the proof: the base's suite runs once per run,
the attempt's chain once per check, a test the self-check passed on a commit not again
in the full check on that commit, an unchanged test file not again for guardedness, and
each preparation once.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from refactory_dsh import base_check, design_ops, failfast, nn, repo_change, sandbox, worktree


@pytest.fixture(autouse=True)
def _own_pytest_temproot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real pytest sessions these checks start keep their temporary directories under
    this test's own (`test_self_check._own_pytest_temproot` says why)."""
    tempfile.gettempdir()
    root = tmp_path / "check-tmp"
    root.mkdir()
    monkeypatch.setenv("TMPDIR", str(root))


PY = sys.executable
PYTEST = f"{PY} -m pytest -p no:cacheprovider"

#: Logs each pytest session (``S <cwd> <args>``) and each test's call (``T <cwd> <id>``).
LOGGER = """import os, sys
LOG = {log!r}


def pytest_sessionstart(session):
    with open(LOG, "a") as f:
        f.write("S " + os.getcwd() + " " + " ".join(sys.argv[1:]) + "\\n")


def pytest_runtest_logreport(report):
    if report.when == "call":
        with open(LOG, "a") as f:
            f.write("T " + os.getcwd() + " " + report.nodeid + " " + report.outcome + "\\n")
"""

#: A setup command that logs where it ran: each preparation is one line.
SETUP = PY + " -c \"import os; open({log!r}, 'a').write(os.getcwd() + chr(10))\""

TEST_OLD = "def test_old():\n    assert 1 == 2, 'OLD-DETAIL'\n"
OLD = "tests/test_old.py::test_old"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


class Fixture:
    """A target repository, a run with an attempt on it, and what its pytest sessions logged."""

    def __init__(self, tmp_path: Path, base_files: dict[str, str], attempt: dict[str, str], *,
                 check: str, setup: bool = False):
        self.log = tmp_path / "pytest.log"
        self.setup_log = tmp_path / "setup.log"
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-q", "-b", "main")
        _git(repo, "config", "user.email", "t@t")
        _git(repo, "config", "user.name", "T")
        files = {".gitignore": "__pycache__/\n", "conftest.py": LOGGER.format(log=str(self.log)),
                 "tests/__init__.py": "", **base_files}
        for name, text in files.items():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text(text)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "base")
        run_dir = tmp_path / "run"
        worktree.create(repo, _git(repo, "rev-parse", "HEAD"), "norerun0", run_dir)
        self.found = worktree.commit_attempt(
            run_dir, {Path(k): v for k, v in attempt.items()}, "attempt")
        facts = {"commit": self.found.head, "check": check, "brief": "", "name": "norerun"}
        if setup:
            facts["setup"] = SETUP.format(log=str(self.setup_log))
        self.ctx = nn.RunContext(run_dir, facts=facts)
        self.check = check

    def _side(self, cwd: str) -> str:
        name = Path(cwd).name
        return ("base" if name.startswith("base-") else "copy" if name.startswith("check-")
                else "ref" if name.startswith("ref-") else "run")

    def sessions(self, side: str) -> list[str]:
        """The arguments of each pytest session on ``side``."""
        lines = self.log.read_text().splitlines() if self.log.exists() else []
        return [line.split(" ", 2)[2] if line.count(" ") >= 2 else ""
                for line in lines if line.startswith("S ") and self._side(line.split()[1]) == side]

    def suites(self, side: str) -> list[str]:
        """The sessions on ``side`` that ran whole test files or directories, not ids."""
        return [args for args in self.sessions(side) if "::" not in args]

    def tests(self, side: str) -> list[str]:
        lines = self.log.read_text().splitlines() if self.log.exists() else []
        return [line.split()[2] for line in lines
                if line.startswith("T ") and self._side(line.split()[1]) == side]

    def setups(self, side: str) -> int:
        lines = self.setup_log.read_text().splitlines() if self.setup_log.exists() else []
        return sum(self._side(line) == side for line in lines)


CHAIN = f"{PYTEST} -q tests/test_old.py && {PYTEST} -q tests/test_more.py"
MORE = "def test_more():\n    pass\n"


# ── 1. the base's suite runs once per run ─────────────────────────────────────

def test_the_base_suite_runs_once_per_run(tmp_path):
    """A failing chain on the base: one run of its marker script gives the whole result
    and every step's record; later checks read the cache."""
    fx = Fixture(tmp_path, {"tests/test_old.py": TEST_OLD, "tests/test_more.py": MORE},
                 {"change.txt": "x\n"}, check=CHAIN)
    first = base_check.run_base_check(fx.ctx, CHAIN, 120)
    assert not first.ok and first.failures == {OLD}
    assert [step["exit_code"] for step in first._step_results] == [1, 0]
    # The chain's own status: the first step's failure, though the second step passed.
    assert first.exit_code == 1
    assert fx.suites("base") == ["-p no:cacheprovider -q tests/test_old.py",
                                 "-p no:cacheprovider -q tests/test_more.py"]
    again = base_check.run_base_check(fx.ctx, CHAIN, 120)
    assert again.failures == {OLD} and len(fx.suites("base")) == 2


def test_the_base_is_prepared_once_for_targeted_runs_and_the_whole_check(tmp_path):
    """`targeted_base_failures` prepares the base (its setup runs there); `run_base_check`
    afterwards reuses that preparation instead of running setup again."""
    fx = Fixture(tmp_path, {"tests/test_old.py": TEST_OLD, "tests/test_more.py": MORE},
                 {"change.txt": "x\n"}, check=CHAIN, setup=True)
    targeted, _ = base_check.targeted_base_failures(fx.ctx, CHAIN, [OLD], prepare_timeout=120,
                                                    timeout=120)
    assert targeted.failures == {OLD} and fx.setups("base") == 1
    whole = base_check.run_base_check(fx.ctx, CHAIN, 120)
    assert whole.failures == {OLD}
    assert fx.setups("base") == 1


def test_a_base_whose_files_were_cleaned_is_prepared_again(tmp_path):
    """The shared preparation is forgotten when the base's files are reset: an untracked
    file left in the base makes `_ensure_base_worktree` clean it, and setup runs again."""
    fx = Fixture(tmp_path, {"tests/test_old.py": TEST_OLD, "tests/test_more.py": MORE},
                 {"change.txt": "x\n"}, check=CHAIN, setup=True)
    base_check.targeted_base_failures(fx.ctx, CHAIN, [OLD], prepare_timeout=120, timeout=120)
    base_path = base_check._base_worktree_path(Path(fx.found.target), "norerun0")
    (base_path / "stray.txt").write_text("left behind\n")
    base_check._ensure_base_worktree(fx.ctx)
    base_check.run_base_check(fx.ctx, CHAIN, 120)
    assert fx.setups("base") == 2


# ── 2. the attempt's chain runs once per check ────────────────────────────────

def test_the_attempts_chain_runs_once_and_its_steps_come_from_that_run(tmp_path):
    """The attempt's check runs step by step from the start: the comparison with the base
    reads those records and runs nothing of the chain again. test_old is inherited, and
    the step after it ran, so it is excused."""
    fx = Fixture(tmp_path, {"tests/test_old.py": TEST_OLD, "tests/test_more.py": MORE},
                 {"change.txt": "x\n"}, check=CHAIN)
    result = repo_change.run_check(fx.ctx, CHAIN, 120, beside_reviews=True, per_step=True)
    assert not result.ok and result.returncode == 1
    assert [step["exit_code"] for step in result.step_results] == [1, 0]
    assert "___REFACTORY_STEP_" not in result.stdout
    filtered = base_check.filter_check(fx.ctx, result, CHAIN, 120, named_text="")
    assert filtered.all_excused and filtered.inherited == {OLD}
    assert len(fx.suites("copy")) == 2 and len(fx.suites("base")) == 2


def test_a_failing_step_still_gives_correct_per_step_results_and_decisions(tmp_path):
    """The change breaks the second step's test: its failure comes from the one run's
    second step, is charged, and the inherited first-step failure is excused."""
    broken = "def test_more():\n    assert False, 'MORE-DETAIL'\n"
    fx = Fixture(tmp_path, {"tests/test_old.py": TEST_OLD, "tests/test_more.py": MORE},
                 {"tests/test_more.py": broken}, check=CHAIN)
    result = repo_change.run_check(fx.ctx, CHAIN, 120, beside_reviews=True, per_step=True)
    assert [step["failures"] for step in result.step_results] == [
        [OLD], ["tests/test_more.py::test_more"]]
    filtered = base_check.filter_check(fx.ctx, result, CHAIN, 120, named_text="")
    assert filtered.new_failures == {"tests/test_more.py::test_more"}
    assert filtered.inherited == {OLD} and not filtered.all_excused
    assert "MORE-DETAIL" in filtered.new_failure_detail
    # Each step of the attempt's chain ran once in the check; what ran again is the
    # targeted confirmation of the new failure, by id.
    assert len(fx.suites("copy")) == 2


@pytest.mark.parametrize("command, status", [
    ("set -e; false; true", 1),
    ("set -e; false && true; true", 0),
    ("set -e; true || false; false", 1),
    ("false && true", 1),
    ("false; true", 0),
    ("set -e; set +e; false; true", 0),
])
def test_the_marker_script_exits_as_the_command_would(tmp_path, command, status):
    """errexit kept by the script itself: the chain's status is the command's own."""
    plain = subprocess.run(["sh", "-c", command], cwd=tmp_path).returncode
    result, steps = base_check.run_chain(command, lambda script: sandbox.run_command(
        script, workdir=tmp_path, timeout_s=30))
    assert plain == status and result.returncode == status
    assert steps is None or all(step["ran"] or step["operator"] == "||" for step in steps)


# ── 3. what the self-check passed is not run again by the full check ──────────

TWO = ("def test_one():\n    pass\n\n\ndef test_two():\n    pass\n\n\n"
       "def test_two_more():\n    pass\n")


def _self_check(fx: Fixture, selection: list[str], limit: float = 120):
    found = worktree.recorded(fx.ctx.workdir)
    here = base_check.residue(found)
    result = failfast._self_check(
        selection, {}, fx.check,
        lambda cmd, timeout: repo_change.run_check(fx.ctx, cmd, timeout, triage_prep=False,
                                                   prepared=True),
        limit, found.path)
    design_ops._record_self_check_passed(fx.ctx, fx.check, result.passed_ids, here)
    return result


def _merged_full_check(fx: Fixture, check: str):
    """The full check beside the reviews as `design_ops._full_check` runs it: the passes it
    may merge (`_mergeable_passes`), deselected, then one run in the check copy."""
    passed = design_ops._mergeable_passes(fx.ctx, check, 120)
    command, used = base_check.with_deselected(check, passed)
    result = repo_change.run_check(fx.ctx, command, 120, beside_reviews=True, per_step=True)
    if used:
        base_check.settle_deselected(result, command)
    return passed, used, result


def test_a_self_check_pass_is_not_run_again_by_the_full_check(tmp_path):
    """The self-check ran test_two.py whole on this commit: what passed there is deselected
    from the full check on the same commit and merged in. test_two is the exception:
    ``--deselect test_two`` would take test_two_more too, so it runs again."""
    check = f"{PYTEST} -q tests"
    fx = Fixture(tmp_path, {"tests/test_two.py": TWO, "tests/test_three.py": MORE},
                 {"tests/test_two.py": TWO + "\n\ndef test_new():\n    pass\n"}, check=check)
    repo_change.prepare_attempt(fx.ctx, 120)
    prepass = _self_check(fx, ["tests/test_two.py"])
    ids = {f"tests/test_two.py::{name}" for name in ("test_one", "test_two_more", "test_new")}
    assert set(prepass.passed_ids) == ids
    passed, used, result = _merged_full_check(fx, check)
    assert set(used) == ids and set(passed) == ids
    assert result.ok
    # The full check ran only what the self-check had not merged.
    assert sorted(fx.tests("copy")) == ["tests/test_three.py::test_more",
                                        "tests/test_two.py::test_two"]
    assert "3 deselected" in result.stdout


def test_an_id_whose_deselect_would_take_another_test_is_not_merged(tmp_path):
    """Selected by id, test_one and test_two pass. Their file's collection shows that
    ``--deselect test_two`` would also take test_two_more, which did not run: test_two
    runs again in the full check, test_one does not."""
    check = f"{PYTEST} -q tests"
    fx = Fixture(tmp_path, {"tests/test_two.py": TWO}, {"change.txt": "x\n"}, check=check)
    repo_change.prepare_attempt(fx.ctx, 120)
    prepass = _self_check(fx, ["tests/test_two.py::test_one", "tests/test_two.py::test_two"])
    assert prepass.passed_ids == ["tests/test_two.py::test_one"]
    _, used, result = _merged_full_check(fx, check)
    assert used == ["tests/test_two.py::test_one"] and result.ok
    assert sorted(fx.tests("copy")) == ["tests/test_two.py::test_two",
                                        "tests/test_two.py::test_two_more"]


def test_self_check_passes_hold_only_for_their_commit(tmp_path):
    check = f"{PYTEST} -q tests"
    fx = Fixture(tmp_path, {"tests/test_two.py": TWO}, {"change.txt": "x\n"}, check=check)
    repo_change.prepare_attempt(fx.ctx, 120)
    _self_check(fx, ["tests/test_two.py"])
    assert design_ops._mergeable_passes(fx.ctx, check, 120)
    assert design_ops._mergeable_passes(fx.ctx, check + " -x", 120) == []
    found = worktree.commit_attempt(fx.ctx.workdir, {Path("change.txt"): "y\n"}, "next attempt")
    fx.ctx.set_fact("commit", found.head)
    assert design_ops._mergeable_passes(fx.ctx, check, 120) == []


def test_a_chunk_stopped_at_its_limit_keeps_what_it_reported(tmp_path):
    """test_c hangs silently and the chunk is stopped: test_a and test_b had passed and
    are kept, as is the failure of test_f reported before the stop; test_c is not."""
    hangs = ("import time\n\n\ndef test_a():\n    pass\n\n\ndef test_f():\n    assert False\n\n\n"
             "def test_b():\n    pass\n\n\ndef test_c():\n    time.sleep(60)\n")
    check = f"{PYTEST} -q tests"
    fx = Fixture(tmp_path, {"tests/test_slow.py": hangs}, {"change.txt": "x\n"}, check=check)
    repo_change.prepare_attempt(fx.ctx, 120)
    prepass = _self_check(fx, ["tests/test_slow.py"], limit=6)
    assert prepass.failing_ids == ["tests/test_slow.py::test_f"]
    assert prepass.passed_ids == ["tests/test_slow.py::test_a", "tests/test_slow.py::test_b"]
    assert prepass.cut_ids == ["tests/test_slow.py"]
    assert any("kept" in note for note in prepass.notes), prepass.notes
    assert design_ops._mergeable_passes(fx.ctx, check, 120) == prepass.passed_ids


def test_a_partial_chunk_keeps_its_passes_for_the_full_check():
    """Without xdist, a pass counts once the next test's line began; parametrized ids
    are exact, a plain id from a file that did not finish is not."""
    out = ("tests/test_p.py::test_p[1] PASSED     [ 10%]\n"
           "tests/test_p.py::test_q PASSED     [ 20%]\n"
           "tests/test_p.py::test_p[2] PASSED     [ 30%]\n"
           "tests/test_p.py::test_hang ")
    passes = failfast.chunk_passes(out, finished=False)
    assert passes == ["tests/test_p.py::test_p[1]", "tests/test_p.py::test_q",
                      "tests/test_p.py::test_p[2]"]
    assert failfast.exact_passes(passes, out, ["tests/test_p.py"], finished=False) == [
        "tests/test_p.py::test_p[1]", "tests/test_p.py::test_p[2]"]
    # Stopped in test_p[2]'s teardown: nothing after it began, so it is not a pass.
    assert "tests/test_p.py::test_p[2]" not in failfast.chunk_passes(
        out.rsplit("tests/test_p.py::test_hang", 1)[0], finished=False)


def test_xdist_lines_are_read_per_worker():
    out = ("[gw0] [ 25%] PASSED tests/test_a.py::test_x\n"
           "[gw1] [ 50%] PASSED tests/test_a.py::test_y\n"
           "[gw0] [ 75%] FAILED tests/test_a.py::test_z\n")
    assert failfast.chunk_passes(out, finished=False) == ["tests/test_a.py::test_x"]
    assert failfast.chunk_passes(out, finished=True) == ["tests/test_a.py::test_x",
                                                          "tests/test_a.py::test_y"]


@pytest.mark.parametrize("command, added", [
    (f"{PYTEST} -q tests", " -o verbosity_test_cases=1"),
    (f"{PYTEST} tests", " -o verbosity_test_cases=1"),
    (f"{PYTEST} -v tests", ""),
])
def test_the_chunk_prints_a_line_per_test(command, added):
    assert failfast._with_test_lines(command) == command + added


def test_deselect_reaches_only_a_single_plain_pytest_step():
    ids = ["tests/test_a.py::test_x[a b]"]
    command, used = base_check.with_deselected("pnpm test && pytest -q tests", ids)
    assert command == "pnpm test && pytest -q tests '--deselect=tests/test_a.py::test_x[a b]'"
    assert used == ids
    for refused in ("pytest a && pytest b", "pytest tests | tee log", "pytest tests > out",
                    "pnpm test"):
        assert base_check.with_deselected(refused, ids) == (refused, [])


def test_deselect_stops_at_its_size_limit(monkeypatch):
    monkeypatch.setattr(base_check, "DESELECT_LIMIT_BYTES", 100)
    ids = [f"tests/test_a.py::test_{n:03}" for n in range(10)]
    _, used = base_check.with_deselected("pytest tests", ids)
    assert 0 < len(used) < len(ids) and used == ids[:len(used)]


# ── 4. guardedness runs only test files whose content changed ─────────────────

GUARD = "import tool\n\n\ndef test_v():\n    assert tool.V == 2\n"


def test_an_unchanged_test_file_is_not_run_again_for_guardedness(tmp_path):
    check = f"{PYTEST} -q tests"
    fx = Fixture(tmp_path, {"tool.py": "V = 1\n", "tests/test_g.py": "def test_g():\n    pass\n"},
                 {"tool.py": "V = 2\n", "tests/test_w.py": "def test_w():\n    pass\n"},
                 check=check)
    first = base_check.compute_test_guardedness(fx.ctx, check, 120)
    assert first["summary"] == {"guarding": 0, "not_guarding": 1, "cannot_run_on_base": 0,
                                "total_new_tests": 1}
    assert len(fx.sessions("base")) == 1
    again = base_check.compute_test_guardedness(fx.ctx, check, 120)
    assert again == first and len(fx.sessions("base")) == 1
    # A later round adds test_v and leaves test_w as it was: only test_v runs on the base.
    worktree.commit_attempt(fx.ctx.workdir, {Path("tests/test_v.py"): GUARD}, "round 2")
    third = base_check.compute_test_guardedness(fx.ctx, check, 120)
    assert fx.sessions("base")[-1].split()[-1] == "tests/test_v.py"
    assert len(fx.sessions("base")) == 2
    assert third["summary"] == {"guarding": 1, "not_guarding": 1, "cannot_run_on_base": 0,
                                "total_new_tests": 2}
    # And a changed test_w runs again.
    worktree.commit_attempt(fx.ctx.workdir, {Path("tests/test_w.py"):
                                             "def test_w():\n    assert True\n"}, "round 3")
    base_check.compute_test_guardedness(fx.ctx, check, 120)
    assert fx.sessions("base")[-1].split()[-1] == "tests/test_w.py"
    assert len(fx.sessions("base")) == 3


def test_a_timed_out_guardedness_run_is_not_kept(tmp_path):
    check = f"{PYTEST} -q tests"
    fx = Fixture(tmp_path, {"tests/test_g.py": "def test_g():\n    pass\n"},
                 {"tests/test_s.py": "import time\n\ndef test_s():\n    time.sleep(30)\n"},
                 check=check)
    base_check.compute_test_guardedness(fx.ctx, check, 3)
    assert not (fx.ctx.workdir / base_check.GUARDEDNESS_CACHE_FILE).exists()


def test_a_timed_out_run_that_named_failures_is_not_kept(tmp_path, monkeypatch):
    """A run stopped part way that had named a failure is still not kept."""
    check = f"{PYTEST} -q tests"
    fx = Fixture(tmp_path, {"tests/test_g.py": "def test_g():\n    pass\n"},
                 {"tests/test_s.py": "def test_s():\n    assert False\n"}, check=check)
    base_check._ensure_base_worktree(fx.ctx)
    stopped = sandbox.SandboxResult(ok=False, timed_out=True, returncode=124,
                                    stdout="FAILED tests/test_s.py::test_s - assert False\n")
    monkeypatch.setattr(base_check.sandbox, "run_command", lambda *a, **k: stopped)
    record = base_check.compute_test_guardedness(fx.ctx, check, 120)
    assert record["summary"]["guarding"] == 1
    assert not (fx.ctx.workdir / base_check.GUARDEDNESS_CACHE_FILE).exists()


# ── 5. the check copy is prepared once per commit ─────────────────────────────

def test_the_check_copy_is_prepared_once_for_the_check_and_its_targeted_reruns(tmp_path):
    broken = "def test_more():\n    assert False\n"
    fx = Fixture(tmp_path, {"tests/test_old.py": TEST_OLD, "tests/test_more.py": MORE},
                 {"tests/test_more.py": broken}, check=CHAIN, setup=True)
    result = repo_change.run_check(fx.ctx, CHAIN, 120, beside_reviews=True, per_step=True)
    assert fx.setups("copy") == 1
    base_check.targeted_attempt_failures(fx.ctx, CHAIN, ["tests/test_more.py::test_more"],
                                         timeout=120)
    base_check.targeted_attempt_failures(fx.ctx, CHAIN, ["tests/test_more.py::test_more"],
                                         timeout=120)
    assert fx.setups("copy") == 1
    assert not result.ok


def test_a_check_copy_whose_files_changed_is_prepared_again(tmp_path):
    fx = Fixture(tmp_path, {"tests/test_old.py": TEST_OLD, "tests/test_more.py": MORE},
                 {"change.txt": "x\n"}, check=CHAIN, setup=True)
    found = worktree.recorded(fx.ctx.workdir)
    copy, *_ = repo_change._check_copy(fx.ctx, found, 120)
    (Path(copy.path) / "change.txt").write_text("edited by a test\n")
    repo_change._check_copy(fx.ctx, found, 120)
    assert fx.setups("copy") == 2
    assert (Path(copy.path) / "change.txt").read_text() == "x\n"


# ── the graph, on each runtime: suite runs across two QA passes ───────────────
#
# Design mode's real graph on a target whose check is one pytest suite. Attempt 1 touches
# test_pin.py (so the self-check runs its two tests, which pass) and breaks data.txt
# (test_data, which only the full check finds; test_env fails on the base too). The full
# check sends it back; attempt 2 fixes data.txt and its check passes, test_env excused.

DATA_TEST = ("from pathlib import Path\n\n\ndef test_data():\n"
             "    assert Path('da' + 'ta.txt').read_text() == 'one\\n', 'DATA-DETAIL'\n")
PIN = "tests/test_pin.py::test_pin"


def _graph_target(tmp_path: Path) -> tuple[Path, Path]:
    import test_self_check as T

    log = tmp_path / "pytest.log"
    repo = T._target(tmp_path / "repo", {
        **T.PINNED, "tests/test_env.py": T.ALWAYS_FAILS, "data.txt": "one\n",
        "tests/test_data.py": DATA_TEST, "conftest.py": LOGGER.format(log=str(log)),
        ".gitignore": "__pycache__/\n"})
    return repo, log


def _emissions() -> tuple[str, str]:
    from tests_support import render

    first = render({"tests/test_pin.py": "import tool\n\n\ndef test_pin():\n"
                                         "    assert tool.VERSION == 1  # pinned\n\n\n"
                                         "def test_version_set():\n    assert tool.VERSION\n",
                    "data.txt": "two\n"})
    return first, render({"data.txt": "one\n", "extra.txt": "x\n"})


def _assert_counts(log: Path) -> None:
    """The base's suite ran once; the attempt's full check once per QA pass, in the check
    copy; test_pin, which the self-check passed on attempt 1's commit, not again in that
    commit's full check."""
    lines = log.read_text().splitlines()

    def side(cwd: str) -> str:
        name = Path(cwd).name
        return "base" if name.startswith("base-") else "copy" if name.startswith("check-") else "run"

    # A run of the whole suite names the ``tests`` directory itself.
    suites = [(side(line.split()[1]), line.split(" ", 2)[2]) for line in lines
              if line.startswith("S ")]
    full = [(where, args) for where, args in suites if "tests" in args.split()]
    assert sum(where == "base" for where, _ in full) == 1, suites
    assert sum(where == "copy" for where, _ in full) == 2, suites
    pin_runs = [side(line.split()[1]) for line in lines
                if line.startswith("T ") and line.split()[2] == PIN]
    # In the run's worktree by the self-checks, and in the copy only on attempt 2's check.
    assert pin_runs.count("copy") == 1, lines
    assert pin_runs.count("run") >= 1, lines


def _assert_delivery_owed(run: Path) -> None:
    """Attempt 1's check deselected the self-check's passes, so it is noted for delivery's
    whole run (`delivery.whole_suite_owed`), and no such check is recorded as a whole
    passing run (`delivery.record_full_check`)."""
    from refactory_dsh import delivery

    owed = json.loads((run / delivery.DESELECTED_CHECKS).read_text())["commits"]
    ledger = run / delivery.PASSED_CHECKS
    whole = json.loads(ledger.read_text()) if ledger.exists() else {}
    assert owed and not set(owed) & {entry["commit"] for entry in whole.values()}


@pytest.mark.usefixtures("fake_gh")
def test_suite_runs_across_two_passes_on_the_walker(tmp_path):
    import test_self_check as T
    from tests_support import FakeModel

    repo, log = _graph_target(tmp_path)
    first, fixed = _emissions()
    reviews = [*T.REVIEW_ANSWERS[:3], T.VERIFICATION_ANSWER, T.REVIEW_ANSWERS[3]]
    model = FakeModel([*T._answers(first), fixed, *reviews])
    result = T._design(tmp_path / "run", model, repo)
    assert result.exit_ok, result.error
    assert sum(T.EMIT in prompt for prompt in model.prompts) >= 2
    _assert_counts(log)
    _assert_delivery_owed(tmp_path / "run")


@pytest.mark.usefixtures("fake_gh")
def test_suite_runs_across_two_passes_on_the_spine(tmp_path, monkeypatch):
    import test_self_check as T
    from test_first_self_check import _check_node

    repo, log = _graph_target(tmp_path)
    first, fixed = _emissions()
    run = T._spine_run(repo, tmp_path, monkeypatch, "abcdef748001")
    gate = f"{design_ops.__name__}:gate_check"
    T._emit(run, first, monkeypatch)
    code, out, err = T._prepare(run, monkeypatch)
    assert code == 0 and T._label(out) == "proceed", (run / design_ops.SELF_CHECK_RECORD).read_text()
    verdict = _check_node(run, monkeypatch)
    assert verdict["verdict"] == "reloop" and "DATA-DETAIL" in verdict["reason"]
    code, out, err = T._node(run, gate, "gate_check", "gate", monkeypatch)
    assert code == 0 and T._label(out) == "reloop", err
    T._emit(run, fixed, monkeypatch)
    code, out, err = T._prepare(run, monkeypatch)
    assert code == 0 and T._label(out) == "proceed", err
    verdict = _check_node(run, monkeypatch)
    assert verdict["verdict"] == "proceed", verdict
    _assert_counts(log)
    _assert_delivery_owed(run)


# ── delivery's merged check: one run, its steps held to the excusal rule ──────

def test_the_merged_check_runs_once_and_blocks_a_failing_non_test_step(tmp_path):
    """``pytest old && false``: the reference fails test_old too, but the step after it
    failed without naming a test, so the merged result is not excused; the chain ran once."""
    from test_judge_paths import _merged

    from refactory_dsh import delivery

    log = tmp_path / "pytest.log"
    command = f"{PYTEST} tests/test_old.py && false"
    ctx, found, merge, tip = _merged(
        tmp_path, {"tests/test_old.py": TEST_OLD, "conftest.py": LOGGER.format(log=str(log))},
        {"feature.py": "x = 1\n"}, {"other.txt": "main moved\n"}, check=command)
    reason, infra = delivery._check_merged(ctx, found, merge, found.head, 120, target_tip=tip)
    assert not infra and "step `false` failed (exit 1)" in reason.split("\n", 1)[0], reason
    merged_runs = [line for line in log.read_text().splitlines()
                   if line.startswith("S ") and Path(line.split()[1]).name.startswith("run-")
                   and "::" not in line]
    assert len(merged_runs) == 1, merged_runs


def test_the_merged_check_excuses_an_inherited_failure_when_every_step_ran(tmp_path):
    """``pytest old && pytest more``: test_old fails on the reference too and test_more
    passes in the same run, so the merged result is delivered."""
    from test_judge_paths import _merged

    from refactory_dsh import delivery

    ctx, found, merge, tip = _merged(
        tmp_path, {"tests/test_old.py": TEST_OLD, "tests/test_more.py": MORE},
        {"feature.py": "x = 1\n"}, {"other.txt": "main moved\n"}, check=CHAIN)
    assert delivery._check_merged(ctx, found, merge, found.head, 120, target_tip=tip) == ("", False)


def test_unreadable_steps_leave_a_chain_incomplete():
    assert base_check.chain_incomplete("pytest a && pytest b", None)
    assert base_check.chain_incomplete("pytest a", None) is None


# ── the review's findings: the check stays binding ────────────────────────────

CFG = ("from pathlib import Path\n\n\ndef test_cfg():\n    assert Path('local.cfg').exists()\n\n\n"
       "def test_other():\n    pass\n")


def test_a_file_only_the_run_worktree_has_stops_the_passes_being_shared(tmp_path):
    """H1: a gitignored ``local.cfg`` the builder left in the run's worktree makes test_cfg
    pass in the self-check. The clean check copy lacks it, so nothing is shared and the
    full check runs test_cfg, which fails there."""
    check = f"{PYTEST} -q tests"
    fx = Fixture(tmp_path, {".gitignore": "__pycache__/\nlocal.cfg\n", "tests/test_cfg.py": CFG},
                 {"change.txt": "x\n"}, check=check)
    repo_change.prepare_attempt(fx.ctx, 120)
    (Path(worktree.recorded(fx.ctx.workdir).path) / "local.cfg").write_text("x")
    prepass = _self_check(fx, ["tests/test_cfg.py"])
    assert "tests/test_cfg.py::test_cfg" in prepass.passed_ids
    passed, used, result = _merged_full_check(fx, check)
    assert passed == [] and used == []
    assert not result.ok and "tests/test_cfg.py::test_cfg" in fx.tests("copy")


WRITE = "from pathlib import Path\n\n\ndef test_write():\n    Path('data.txt').write_text('bad\\n')\n"
DATA = ("from pathlib import Path\n\n\ndef test_data():\n"
        "    assert Path('data.txt').read_text() == 'good\\n'\n")


def test_a_targeted_base_run_that_dirties_the_base_does_not_reach_the_base_check(tmp_path):
    """H2: a targeted base run rewrites the tracked data.txt. The base is put back after it
    and the whole base check finds it changed from its preparation anyway, so test_data
    does not fail on the base."""
    check = f"{PYTEST} -q tests"
    fx = Fixture(tmp_path, {"data.txt": "good\n", "tests/test_d.py": DATA, "tests/test_w.py": WRITE},
                 {"change.txt": "x\n"}, check=check)
    base_check.targeted_base_failures(fx.ctx, check, ["tests/test_w.py::test_write"],
                                      prepare_timeout=120, timeout=120)
    base_path = base_check._base_worktree_path(Path(fx.found.target), "norerun0")
    assert (base_path / "data.txt").read_text() == "good\n"
    whole = base_check.run_base_check(fx.ctx, check, 120)
    assert whole.ok and whole.failures == set()


def test_a_base_changed_behind_the_marker_is_not_taken_for_prepared(tmp_path):
    """H2: something else rewrites a tracked file in the prepared base; the base check
    sees its state differ from the preparation's, puts it back and prepares again."""
    check = f"{PYTEST} -q tests/test_d.py"
    fx = Fixture(tmp_path, {"data.txt": "good\n", "tests/test_d.py": DATA},
                 {"change.txt": "x\n"}, check=check, setup=True)
    base_check.targeted_base_failures(fx.ctx, check, ["tests/test_d.py::test_data"],
                                      prepare_timeout=120, timeout=120)
    base_path = base_check._base_worktree_path(Path(fx.found.target), "norerun0")
    (base_path / "data.txt").write_text("bad\n")
    whole = base_check.run_base_check(fx.ctx, check, 120)
    assert whole.ok and fx.setups("base") == 2


@pytest.mark.parametrize("command", [
    "sh -c 'exit 5' || [ $? -eq 5 ]",
    "sh -c 'exit 5' || test $? -eq 3",
    "false; [ $? -eq 1 ] && true",
    "sh -c 'exit 3' && true || [ $? -eq 3 ]",
    "set -e; sh -c 'exit 2' || [ $? -eq 2 ]; true",
    "set -e; sh -c 'exit 3'; [ $? -eq 3 ]",
])
@pytest.mark.parametrize("shell", ["/bin/sh", "/bin/bash", "/bin/dash"])
def test_each_step_sees_the_status_the_shell_would_give_it(tmp_path, command, shell):
    """M1: ``$?`` in a step is the chain's status so far, as in the shell itself, under each
    shell the check may run in (`/bin/sh` is dash on Debian and Ubuntu)."""
    if not Path(shell).exists():
        pytest.skip(f"{shell} is not installed")
    plain = subprocess.run([shell, "-c", command], cwd=tmp_path).returncode
    plan = base_check.chain_plan(command)
    script = subprocess.run([shell, "-c", plan.script], cwd=tmp_path, capture_output=True,
                            text=True)
    assert script.returncode == plain
    steps = base_check._parse_step_results(script.stdout, plan.marker_id, plan.steps)
    records = [{**sr, "step": text} for sr, (text, _) in zip(steps, plan.steps)]
    assert base_check.chain_status(records) == plain


def test_a_step_whose_every_test_was_deselected_passes_without_running_one(tmp_path):
    """Every test of the check passed in the self-check on this commit: all are deselected,
    none runs again, and pytest's exit 5 for the empty step is read as the pass it is --
    for a single step and as a step of a chain."""
    for check in (f"{PYTEST} -q tests", f"{PYTEST} -q tests && echo lint ok",
                  f"{PYTEST} -q -n 2 tests"):
        root = tmp_path / str(len(check))
        root.mkdir()
        fx = Fixture(root, {"tests/test_three.py": MORE}, {"change.txt": "x\n"}, check=check)
        repo_change.prepare_attempt(fx.ctx, 120)
        _self_check(fx, ["tests/test_three.py"])
        _, used, result = _merged_full_check(fx, check)
        assert used == ["tests/test_three.py::test_more"]
        assert result.ok and result.returncode == 0, result.stdout
        assert fx.tests("copy") == []


def test_a_step_that_collected_nothing_still_fails():
    """A step that collected nothing without deselecting anything stays failing; so does
    one whose exit 5 pytest did not explain."""
    result = sandbox.SandboxResult(ok=False, returncode=5, stdout="no tests ran in 0.01s\n")
    base_check.settle_deselected(result, "pytest -q tests")
    assert not result.ok and result.returncode == 5
    result = sandbox.SandboxResult(ok=False, returncode=5, stdout="crashed\n")
    base_check.settle_deselected(result, "pytest -q tests --deselect=tests/test_a.py::test_x")
    assert not result.ok and result.returncode == 5
    result = sandbox.SandboxResult(ok=False, returncode=5, stdout="no tests ran in 0.70s\n")
    base_check.settle_deselected(result, "pytest -q -n 4 tests --deselect=tests/test_a.py::test_x")
    assert result.ok and result.returncode == 0  # xdist's word for all deselected


ORDER_BASE = ("STATE = []\n\n\ndef test_a_sets():\n    pass\n\n\n"
              "def test_b_needs_empty():\n    assert STATE == []\n\n\ndef test_c():\n    pass\n")
ORDER_LEAKS = ORDER_BASE.replace("def test_a_sets():\n    pass", "def test_a_sets():\n    STATE.append(1)")


def test_an_order_dependent_failure_hidden_by_deselection_is_caught_at_delivery(tmp_path):
    """M2: the change makes test_a leave state that breaks test_b only in its file's order.
    The self-check ran test_b alone and passed it, so the full check deselected it and
    passed. That pass never stands for delivery: delivery runs the whole suite on the
    reviewed commit, finds test_b failing where the base passes it, and does not deliver
    it as passing (a draft, or no push)."""
    from refactory_dsh import delivery

    check = f"{PYTEST} -q tests"
    fx = Fixture(tmp_path, {"tests/test_o.py": ORDER_BASE},
                 {"tests/test_o.py": ORDER_LEAKS}, check=check)
    repo_change.prepare_attempt(fx.ctx, 120)
    _self_check(fx, ["tests/test_o.py::test_b_needs_empty", "tests/test_o.py::test_c"])
    _, used, result = _merged_full_check(fx, check)
    assert "tests/test_o.py::test_b_needs_empty" in used and result.ok
    found = worktree.recorded(fx.ctx.workdir)
    path = Path(found.path)
    # What `_full_check` does with such a pass: noted as deselected, never recorded whole.
    delivery.record_deselected_check(fx.ctx, found.head)
    assert delivery.reusable_check(fx.ctx, path, found.head) == ""
    assert delivery.whole_suite_owed(fx.ctx, path, found.head)
    reason, infra = delivery._check_merged(fx.ctx, found, found.head, found.head, 120,
                                           target_tip=found.base)
    assert not infra and "tests/test_o.py::test_b_needs_empty" in reason.split("\n", 1)[0], reason
    # A whole run that passes stands for the tree again.
    delivery.record_full_check(fx.ctx, path, check)
    assert not delivery.whole_suite_owed(fx.ctx, path, found.head)


def test_a_crashed_chunk_keeps_nothing(monkeypatch):
    """A chunk stopped at the limit keeps what it reported, but not when a signal killed it:
    nothing a crashed run printed is kept."""
    out = ("tests/test_a.py::test_x PASSED  [ 50%]\ntests/test_a.py::test_y FAILED  [ 60%]\n"
           "tests/test_a.py::test_z ")
    for crashed, kept in ((False, True), (True, False)):
        monkeypatch.setattr(failfast, "sandbox_crashed", lambda result, crashed=crashed: crashed)
        result = sandbox.SandboxResult(ok=False, timed_out=True, returncode=124, stdout=out)
        prepass = failfast._self_check(["tests/test_a.py"], {}, "pytest -q tests",
                                       lambda cmd, timeout: result, 30, None)
        assert bool(prepass.failing_ids) is kept, prepass
        assert (prepass.failing_ids == ["tests/test_a.py::test_y"]) is kept


def test_a_pytest_without_test_lines_is_said_in_the_record():
    """pytest before 8.1 warns of the unknown option and prints no line per test: the
    record says so, and no pass is shared."""
    warned = ("PytestConfigWarning: Unknown config option: verbosity_test_cases\n"
              "1 passed in 0.01s\n")
    prepass = failfast._self_check(
        ["tests/test_a.py"], {}, "pytest -q tests",
        lambda cmd, timeout: sandbox.SandboxResult(ok=True, returncode=0, stdout=warned), 30, None)
    assert failfast.NO_TEST_LINES in prepass.notes and prepass.passed_ids == []


def test_the_guardedness_cache_is_kept_per_check_command(tmp_path):
    check = f"{PYTEST} -q tests"
    fx = Fixture(tmp_path, {"tests/test_g.py": "def test_g():\n    pass\n"},
                 {"tests/test_w.py": "def test_w():\n    pass\n"}, check=check)
    base_check.compute_test_guardedness(fx.ctx, check, 120)
    base_check.compute_test_guardedness(fx.ctx, check, 120)
    assert len(fx.sessions("base")) == 1
    base_check.compute_test_guardedness(fx.ctx, check + " -p no:randomly", 120)
    assert len(fx.sessions("base")) == 2


@pytest.mark.usefixtures("fake_gh")
def test_a_check_that_passed_with_deselection_is_run_whole_at_delivery_on_the_walker(tmp_path):
    """One QA pass whose full check deselects what the self-check passed and passes: that
    pass is not recorded as a whole run of its commit, and the whole suite runs once more,
    in the run's worktree, at delivery."""
    import test_self_check as T
    from tests_support import FakeModel

    from refactory_dsh import delivery

    log = tmp_path / "pytest.log"
    repo = T._target(tmp_path / "repo", {**T.PINNED, "conftest.py": LOGGER.format(log=str(log)),
                                         ".gitignore": "__pycache__/\n"})
    first, _ = _emissions()
    model = FakeModel(T._answers(first.replace("===FILE: data.txt===\ntwo\n", "")))
    result = T._design(tmp_path / "run", model, repo)
    assert result.exit_ok, result.error
    sessions = [line.split() for line in log.read_text().splitlines() if line.startswith("S ")]
    whole = [Path(words[1]).name for words in sessions if "tests" in words[2:]]
    copy_runs = [name for name in whole if name.startswith("check-")]
    run_runs = [name for name in whole if name.startswith("run-")]
    assert len(copy_runs) == 1 and len(run_runs) == 1, whole
    run = tmp_path / "run"
    owed = json.loads((run / delivery.DESELECTED_CHECKS).read_text())["commits"]
    assert len(owed) == 1
    # The deselected pass is never recorded as a whole run of its commit.
    ledger = run / delivery.PASSED_CHECKS
    whole_runs = json.loads(ledger.read_text()) if ledger.exists() else {}
    assert owed[0] not in {entry["commit"] for entry in whole_runs.values()}, whole_runs
