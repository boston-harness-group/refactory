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
    result = failfast._self_check(
        selection, {}, fx.check,
        lambda cmd, timeout: repo_change.run_check(fx.ctx, cmd, timeout, triage_prep=False,
                                                   prepared=True),
        limit, found.path)
    design_ops._record_self_check_passed(fx.ctx, fx.check, result.passed_ids)
    return result


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
    passed = design_ops.self_check_passed(fx.ctx, check)
    command, used = base_check.with_deselected(check, passed)
    # The last always runs, so a step never deselects all it collects (pytest exit 5).
    assert used == passed[:-1] and passed[-1] == "tests/test_two.py::test_new"
    result = repo_change.run_check(fx.ctx, command, 120, beside_reviews=True, per_step=True)
    assert result.ok
    # The full check ran only what the self-check had not merged.
    assert sorted(fx.tests("copy")) == ["tests/test_three.py::test_more",
                                        "tests/test_two.py::test_new",
                                        "tests/test_two.py::test_two"]
    assert "2 deselected" in result.stdout


def test_an_id_whose_deselect_would_take_another_test_is_not_merged(tmp_path):
    """Selected by id, test_one and test_two pass. Their file's collection shows that
    ``--deselect test_two`` would also take test_two_more, which did not run: test_two
    runs again in the full check, test_one does not."""
    check = f"{PYTEST} -q tests"
    fx = Fixture(tmp_path, {"tests/test_two.py": TWO}, {"change.txt": "x\n"}, check=check)
    repo_change.prepare_attempt(fx.ctx, 120)
    prepass = _self_check(fx, ["tests/test_two.py::test_one", "tests/test_two.py::test_two"])
    assert prepass.passed_ids == ["tests/test_two.py::test_one"]
    command, used = base_check.with_deselected(check, design_ops.self_check_passed(fx.ctx, check))
    result = repo_change.run_check(fx.ctx, command, 120, beside_reviews=True, per_step=True)
    # test_one, the only pass, is the one that always runs (`with_deselected`).
    assert used == [] and result.ok and sorted(fx.tests("copy")) == [
        "tests/test_two.py::test_one", "tests/test_two.py::test_two",
        "tests/test_two.py::test_two_more"]


def test_self_check_passes_hold_only_for_their_commit(tmp_path):
    check = f"{PYTEST} -q tests"
    fx = Fixture(tmp_path, {"tests/test_two.py": TWO}, {"change.txt": "x\n"}, check=check)
    repo_change.prepare_attempt(fx.ctx, 120)
    _self_check(fx, ["tests/test_two.py"])
    assert design_ops.self_check_passed(fx.ctx, check)
    assert design_ops.self_check_passed(fx.ctx, check + " -x") == []
    worktree.commit_attempt(fx.ctx.workdir, {Path("change.txt"): "y\n"}, "next attempt")
    assert design_ops.self_check_passed(fx.ctx, check) == []


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
    assert design_ops.self_check_passed(fx.ctx, check) == prepass.passed_ids


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
    assert base_check.with_deselected("pnpm test && pytest -q tests", ids) == (
        "pnpm test && pytest -q tests", [])  # the last id always runs
    ids = [*ids, "tests/test_a.py::test_y"]
    command, used = base_check.with_deselected("pnpm test && pytest -q tests", ids)
    assert command == "pnpm test && pytest -q tests '--deselect=tests/test_a.py::test_x[a b]'"
    assert used == ids[:1]
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
