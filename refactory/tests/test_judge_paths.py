"""The shared failure judge on each check path, on real worktrees with real pytest.

Every test here runs the real ops -- `base_check.filter_check`, `targeted_attempt_failures`,
`targeted_base_failures`, `run_base_check`, `design_ops._the_changes_failures` and
`delivery._check_merged` -- on a fixture repository whose tests run with the real pytest
(``python -m pytest``) or, where a runner's exact output matters, a recorded one. Nothing
of git or the targeted runs is mocked.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from refactory_dsh import base_check, delivery, design_ops, nn, repo_change, worktree
from refactory_dsh.base_check import (
    _build_positive_detail,
    filter_check,
    judge_failures,
    parse_test_failures,
)



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

OLD = "tests/test_old.py::test_old"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def _commit(repo: Path, files: dict[str, str], message: str = "change") -> str:
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if name.startswith("node_modules/.bin/"):
            path.chmod(0o755)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _target(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "T")
    (repo / ".gitignore").write_text("__pycache__/\ngen_mod.py\n")
    return repo, _commit(repo, files, "base")


def _run(tmp_path: Path, repo: Path, base: str, attempt: dict[str, str], *,
         check: str, **facts: str) -> tuple[nn.RunContext, worktree.Worktree]:
    run_dir = tmp_path / "run"
    worktree.create(repo, base, "judge0paths", run_dir)
    found = worktree.commit_attempt(run_dir, {Path(k): v for k, v in attempt.items()}, "attempt")
    ctx = nn.RunContext(run_dir, facts={"commit": found.head, "check": check, "brief": "",
                                        "name": "judged", **facts})
    return ctx, found


#: A test that fails on the base and on every attempt: the pre-existing failure.
TEST_OLD = "def test_old():\n    assert 1 == 2, 'OLD-DETAIL'\n"


# ── order-dependent failures are charged, not excused as not-reproduced ────────

APP_BASE = "REGISTRY = []\n\ndef register():\n    pass\n"
APP_LEAKS = "REGISTRY = []\n\ndef register():\n    REGISTRY.append(1)\n"
TEST_ORDER = ("import app\n\n"
              "def test_a():\n    app.register()\n\n"
              "def test_b():\n    assert app.REGISTRY == [], 'ORDER-DETAIL'\n")
ORDER_B = "tests/test_order.py::test_b"


def test_an_order_dependent_failure_is_charged(tmp_path):
    """test_b fails only after test_a, in its file's order: alone it passes. The change
    makes test_a leave state behind, so the failure is the change's."""
    repo, base = _target(tmp_path, {"app.py": APP_BASE, "tests/__init__.py": "",
                                    "tests/test_order.py": TEST_ORDER, "conftest.py": ""})
    command = f"{PYTEST} tests"
    ctx, _ = _run(tmp_path, repo, base, {"app.py": APP_LEAKS}, check=command)
    result = repo_change.run_check(ctx, command, 120)
    assert not result.ok and ORDER_B in parse_test_failures(result.stdout)
    # Alone, on the attempt, it passes; in its file's order it fails.
    assert base_check.targeted_attempt_failures(ctx, command, [ORDER_B], timeout=120) == {ORDER_B}
    filtered = filter_check(ctx, result, command, 120, named_text="")
    assert ORDER_B in filtered.new_failures and not filtered.all_excused
    assert filtered.reasons[ORDER_B] == "new-failure"


# ── the targeted attempt re-run runs in the check copy ─────────────────────────

def test_the_targeted_attempt_rerun_runs_in_the_check_copy(tmp_path):
    """The re-run writes nothing into the run's worktree, where reviewers work."""
    writes = ("import os\n\ndef test_w():\n"
              "    open('ran-here.txt', 'w').write(os.getcwd())\n    assert False\n")
    repo, base = _target(tmp_path, {"tests/test_w.py": "def test_w():\n    pass\n"})
    command = f"{PYTEST} tests"
    ctx, found = _run(tmp_path, repo, base, {"tests/test_w.py": writes}, check=command)
    failed = base_check.targeted_attempt_failures(ctx, command, ["tests/test_w.py::test_w"],
                                                  timeout=120)
    assert failed == {"tests/test_w.py::test_w"}
    assert not (Path(found.path) / "ran-here.txt").exists()
    copy = repo_change._check_copy_path(Path(found.target), "judge0paths")
    assert (copy / "ran-here.txt").exists()


# ── a timed-out base is never cached; a crashed one falls back to targeted runs ─

#: A pytest stand-in that prints recorded output: ``out/<first arg>.txt`` for the
#: whole run, else the real pytest. It runs the real pytest for targeted runs (ids).
FAKE_PYTEST = """#!/bin/sh
case "$(basename "$PWD")" in base-*) side=base;; *) side=attempt;; esac
if [ "$1" = "tests" ] && [ -f "rec/$side.sh" ]; then . "rec/$side.sh"; fi
exec "{py}" -m pytest -p no:cacheprovider "$@"
"""


def _recorded(tmp_path: Path, base_rec: str, attempt_rec: str = "",
              extra: dict[str, str] | None = None) -> tuple[nn.RunContext, worktree.Worktree]:
    """A target whose whole-run check, ``pytest tests``, does what ``rec/<side>.sh`` says on
    each side (the base worktree, or the attempt's), and runs the real pytest otherwise."""
    files = {"node_modules/.bin/pytest": FAKE_PYTEST.format(py=PY),
             "tests/test_old.py": TEST_OLD, "rec/base.sh": base_rec, **(extra or {})}
    repo, base = _target(tmp_path, files)
    attempt = {"change.txt": "x\n", **({"rec/attempt.sh": attempt_rec} if attempt_rec else {})}
    return _run(tmp_path, repo, base, attempt, check="pytest tests")


def test_a_timed_out_base_that_named_failures_is_not_cached(tmp_path):
    ctx, _ = _recorded(tmp_path, f"echo 'FAILED {OLD} - x'; sleep 30\n")
    result = base_check.run_base_check(ctx, "pytest tests", 2)
    assert result.environment_failure and "time limit" in result.environment_failure
    assert not (ctx.workdir / base_check.CACHE_FILE).exists()


def test_a_crashed_base_falls_back_to_targeted_runs(tmp_path):
    """The whole base run is killed by a signal, twice; the targeted base run of the
    attempt's failure still answers, so the pre-existing failure is excused."""
    ctx, _ = _recorded(tmp_path, "kill -SEGV $$\n")
    command = "pytest tests"
    result = repo_change.run_check(ctx, command, 120)
    assert not result.ok
    filtered = filter_check(ctx, result, command, 120, named_text="")
    assert filtered.base_environment_failure is None, filtered.inherited_detail
    assert filtered.all_excused and OLD in filtered.inherited


# ── the per-step comparison confirms per-step-only failures ────────────────────

FLAKY = ("import os, pathlib\n\n"
         "def test_flaky():\n"
         "    if not pathlib.Path('flaky_on').exists():\n        return\n"
         "    seen = pathlib.Path(os.environ['HOME']) / ('seen-' + os.getcwd().replace('/', '_'))\n"
         "    first = not seen.exists()\n    seen.write_text('x')\n"
         "    assert not first, 'FLAKY-DETAIL'\n")
FLAKY_ID = "tests/test_flaky.py::test_flaky"


def test_a_per_step_only_failure_that_does_not_reproduce_is_excused(tmp_path):
    """``pytest old && pytest flaky``: the pre-existing failure would stop the chain; run
    step by step from the start, the check also reaches test_flaky, which fails once and
    then passes. It goes through targeted confirmation and is excused as not reproduced."""
    repo, base = _target(tmp_path, {"tests/test_old.py": TEST_OLD,
                                    "tests/test_flaky.py": FLAKY})
    command = f"{PYTEST} tests/test_old.py && {PYTEST} tests/test_flaky.py"
    ctx, _ = _run(tmp_path, repo, base, {"flaky_on": "1\n"}, check=command)
    result = repo_change.run_check(ctx, command, 120, beside_reviews=True, per_step=True)
    assert not result.ok and parse_test_failures(result.stdout) == {OLD, FLAKY_ID}
    assert [step["exit_code"] for step in result.step_results] == [1, 1]
    filtered = filter_check(ctx, result, command, 120, named_text="")
    assert filtered.reasons[FLAKY_ID] == "not-reproduced", filtered.reasons
    assert filtered.all_excused and {OLD, FLAKY_ID} <= filtered.inherited


M_ID = "tests/test_integ.py::test_M"
N_ID = "tests/test_integ.py::test_N"


def test_the_per_step_detail_shows_no_excused_failure(tmp_path):
    """The integ step fails test_M (the base does too) and test_N (the change broke it):
    the builder's detail shows test_N and never test_M's line or assertion."""
    integ = ("import app\n\n\ndef test_M():\n    assert False, 'M-DETAIL'\n\n\n"
             "def test_N():\n    assert app.VALUE == 1, 'N-DETAIL'\n")
    repo, base = _target(tmp_path, {"tests/test_old.py": TEST_OLD, "conftest.py": "",
                                    "tests/test_integ.py": integ, "app.py": "VALUE = 1\n"})
    command = f"{PYTEST} tests/test_old.py && {PYTEST} tests/test_integ.py"
    ctx, _ = _run(tmp_path, repo, base, {"app.py": "VALUE = 2\n"}, check=command)
    result = repo_change.run_check(ctx, command, 120, per_step=True)
    filtered = filter_check(ctx, result, command, 120, named_text="")
    assert N_ID in filtered.new_failures and M_ID in filtered.inherited
    detail = filtered.new_failure_detail
    assert "N-DETAIL" in detail
    assert f"FAILED {M_ID}" not in detail and "M-DETAIL" not in detail, detail


# ── vitest send-back: summary lines only, never the whole output ───────────────

VITEST_OUT = """
 RUN  v1.6.0 /repo

 ❯ tests/a.spec.ts (2)
 ❯ tests/b.spec.ts (9)

⎯⎯⎯⎯⎯⎯⎯ Failed Tests 2 ⎯⎯⎯⎯⎯⎯⎯

 FAIL  tests/a.spec.ts > suite > charged
AssertionError: charged boom
 ❯ tests/a.spec.ts:4:12
⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯[1/2]⎯
 FAIL  tests/b.spec.ts > suite > excused
AssertionError: EXCUSED-DETAIL
 ❯ tests/b.spec.ts:9:3
⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯[2/2]⎯

 Test Files  2 failed (2)
      Tests  2 failed | 9 passed (11)
   Start at  10:00:00
   Duration  1.20s
"""


def test_vitest_send_back_has_its_summary_and_no_excused_detail():
    detail = _build_positive_detail(VITEST_OUT, {"tests/a.spec.ts > suite > charged"},
                                    {"tests/b.spec.ts > suite > excused"},
                                    {"tests/b.spec.ts > suite > excused": "base-whole-run"})
    assert "charged boom" in detail
    assert detail.count("FAIL  tests/a.spec.ts > suite > charged") == 1, detail
    assert "Tests  2 failed | 9 passed (11)" in detail and "Test Files  2 failed (2)" in detail
    assert "EXCUSED-DETAIL" not in detail and "RUN  v1.6.0" not in detail, detail


def test_send_back_without_a_summary_line_keeps_only_the_charged_lines():
    """No runner summary to find: the detail is the charged failure's own line, not the
    output it was looked for in."""
    out = ("collecting NOISE-LINE\nFAILED tests/test_a.py::test_x - boom\n"
           "FAILED tests/test_b.py::test_y - EXCUSED-DETAIL\n")
    detail = _build_positive_detail(out, {"tests/test_a.py::test_x"},
                                    {"tests/test_b.py::test_y"}, {})
    assert detail.splitlines()[0] == "FAILED tests/test_a.py::test_x - boom"
    assert "NOISE-LINE" not in detail and "EXCUSED-DETAIL" not in detail, detail


# ── a log line naming conftest.py is not a failure ─────────────────────────────

def test_a_log_line_naming_a_conftest_line_is_not_a_failure():
    from refactory_dsh.check_digest import _parse_pytest

    log = "ERROR    tests/conftest.py:12 setup could not reach the cache\n"
    assert parse_test_failures(log) == set()
    assert _parse_pytest(log) is None
    real = "ERROR tests/conftest.py - ImportError: no module\n"
    assert parse_test_failures(real) == {"tests/conftest.py"}


# ── touched test files: both signatures, or charged ────────────────────────────

def test_a_touched_file_failure_the_targeted_base_has_is_charged_without_signatures():
    tid = "tests/test_a.py::test_x"
    jr = judge_failures({tid}, set(), named_text="", changed_files=["tests/test_a.py"],
                        targeted_base_failures_set={tid}, targeted_attempt_failures={tid})
    assert tid in jr.charged and jr.reasons[tid] == "signature-mismatch"
    same = {tid: "AssertionError:tests/test_a.py:2"}
    jr = judge_failures({tid}, set(), named_text="", changed_files=["tests/test_a.py"],
                        targeted_base_failures_set={tid}, attempt_signatures=same,
                        base_signatures=same)
    assert tid in jr.excused


@pytest.mark.parametrize("touched, charged", [
    # The same failure at the same place: inherited.
    (TEST_OLD + "\n\ndef test_more():\n    pass\n", False),
    # The edit moved where it fails: its signature differs, so it is the change's.
    ("import os\n\n" + TEST_OLD, True),
])
def test_the_self_check_compares_a_touched_files_failure_by_signature(tmp_path, touched,
                                                                       charged):
    """The attempt edits the pre-existing failure's file and it still fails: inherited
    only when the targeted runs on both sides fail it with one signature."""
    repo, base = _target(tmp_path, {"tests/test_old.py": TEST_OLD})
    command = f"{PYTEST} tests"
    ctx, _ = _run(tmp_path, repo, base, {"tests/test_old.py": touched}, check=command)
    changes, inherited, said = design_ops._the_changes_failures(ctx, command, [OLD], 120, 120)
    assert (changes, inherited) == (([OLD], []) if charged else ([], [OLD])), said


def test_the_self_check_excuses_a_failure_that_does_not_reproduce(tmp_path):
    """Correction 1 on the self-check: the base passes the test, and on the attempt it
    fails once and then passes, alone and in its file's order. Not charged."""
    repo, base = _target(tmp_path, {"tests/test_flaky.py": FLAKY})
    command = f"{PYTEST} tests"
    ctx, found = _run(tmp_path, repo, base, {"flaky_on": "1\n"}, check=command)
    home = repo_change._check_home(ctx)
    first = subprocess.run([PY, "-m", "pytest", "-p", "no:cacheprovider", FLAKY_ID],
                           cwd=found.path, capture_output=True, text=True,
                           env={**worktree.check_environment(Path(found.path), home),
                                "HOME": str(home)})
    assert first.returncode == 1, first.stdout
    changes, inherited, _ = design_ops._the_changes_failures(ctx, command, [FLAKY_ID], 120, 120)
    assert changes == [] and inherited == [FLAKY_ID]


# ── delivery's merged-result check, on real worktrees ──────────────────────────

def _merged(tmp_path: Path, base_files: dict[str, str], attempt: dict[str, str],
            tip_files: dict[str, str], *, check: str, setup: str = ""
            ) -> tuple[nn.RunContext, worktree.Worktree, str, str]:
    """A run whose attempt is merged with main, which moved to ``tip_files`` after the
    run began. Returns the context, the run's worktree, the merge commit and the tip."""
    repo, base = _target(tmp_path, base_files)
    ctx, found = _run(tmp_path, repo, base, attempt, check=check,
                      **({"setup": setup} if setup else {}))
    tip = _commit(repo, tip_files, "main moved")
    path = Path(found.path)
    _git(path, "fetch", "-q", str(repo), "main")
    _git(path, "-c", "user.email=t@t", "-c", "user.name=T", "merge", "-q", "--no-edit",
         "FETCH_HEAD")
    return ctx, found, _git(path, "rev-parse", "HEAD"), tip


#: The reference needs its setup: the test imports what setup generates.
SETUP = f"{PY} -c \"open('gen_mod.py', 'w').write('X = 1\\n')\""
TEST_GEN_OLD = "import gen_mod\n\ndef test_old():\n    assert gen_mod.X == 2, 'OLD-DETAIL'\n"


def test_the_merged_check_excuses_what_the_prepared_reference_fails(tmp_path):
    """Main moved; the merged result fails only test_old, which the reference at the
    tip fails too -- once its setup has generated the module it imports."""
    ctx, found, merge, tip = _merged(
        tmp_path, {"tests/test_old.py": TEST_GEN_OLD}, {"feature.py": "x = 1\n"},
        {"other.txt": "main moved\n"}, check=f"{PYTEST} tests", setup=SETUP)
    reason, infra = delivery._check_merged(ctx, found, merge, found.head, 120, target_tip=tip)
    assert (reason, infra) == ("", False)
    reference = base_check._reference_worktree_path(Path(found.target), "judge0paths")
    assert (reference / "gen_mod.py").exists() and _git(reference, "rev-parse", "HEAD") == tip


def test_the_merged_check_charges_a_skipped_step(tmp_path):
    """``pytest old && pytest new``: step one's failure is the reference's too; the step
    the shell would skip after it runs anyway, in the check's one step-by-step run, and
    its new failure is charged. Not delivered as passing."""
    new_broken = "def test_new():\n    assert False, 'NEW-DETAIL'\n"
    ctx, found, merge, tip = _merged(
        tmp_path, {"tests/test_old.py": TEST_OLD, "tests/test_new.py": "def test_new():\n    pass\n"},
        {"tests/test_new.py": new_broken}, {"other.txt": "main moved\n"},
        check=f"{PYTEST} tests/test_old.py && {PYTEST} tests/test_new.py")
    reason, infra = delivery._check_merged(ctx, found, merge, found.head, 120, target_tip=tip)
    assert not infra and reason
    first = reason.split("\n", 1)[0]
    assert "charged: tests/test_new.py::test_new" in first and "NEW-DETAIL" in reason, reason
    assert _git(Path(found.path), "rev-parse", "HEAD") == found.head, "reset to the reviewed commit"


def test_the_merged_check_charges_a_failure_its_summary_does_not_name(tmp_path):
    """The summary counts an error no line names: nothing is excused, whatever the
    reference says of the named one."""
    rec = (f"echo 'FAILED {OLD} - assert 1 == 2'; "
           "echo '= 1 failed, 3 passed, 1 error in 0.1s ='; exit 1\n")
    files = {"node_modules/.bin/pytest": FAKE_PYTEST.format(py=PY),
             "tests/test_old.py": TEST_OLD, "rec/attempt.sh": rec}
    ctx, found, merge, tip = _merged(tmp_path, files, {"change.txt": "x\n"},
                                     {"other.txt": "main moved\n"}, check="pytest tests")
    reason, _ = delivery._check_merged(ctx, found, merge, found.head, 120, target_tip=tip)
    assert "not identified by name" in reason.split("\n", 1)[0], reason


def test_the_merged_check_charges_a_new_failure_and_keeps_its_reason(tmp_path):
    broken = "def test_new():\n    assert False, 'NEW-DETAIL'\n"
    ctx, found, merge, tip = _merged(
        tmp_path, {"tests/test_old.py": TEST_OLD, "tests/test_new.py": "def test_new():\n    pass\n"},
        {"tests/test_new.py": broken}, {"other.txt": "main moved\n"}, check=f"{PYTEST} tests")
    reason, _ = delivery._check_merged(ctx, found, merge, found.head, 120, target_tip=tip)
    first = reason.split("\n", 1)[0]
    assert "charged: tests/test_new.py::test_new" in first, first
    record = json.dumps(reason)
    assert "NEW-DETAIL" in record


# ── the check-to-gate path, on each runtime ────────────────────────────────────
#
# Design mode's real graph: the pre-existing failure (test_env, which the base fails too)
# is excused, and the one the change makes (test_data, which reads the data file the
# change rewrites) is charged and sent back to the builder, with test_env's detail left
# out. The self-check does not select test_data -- it imports nothing the change touched --
# so the whole check beside the reviews is what finds it.

DATA_ID = "tests/test_data.py::test_data"
TEST_DATA = ("from pathlib import Path\n\n\ndef test_data():\n"
             "    assert Path('data.txt').read_text() == 'one\\n', 'DATA-DETAIL'\n")


def _graph_target(tmp_path: Path) -> Path:
    import test_self_check as T

    return T._target(tmp_path / "repo", {**T.PINNED, "tests/test_env.py": T.ALWAYS_FAILS,
                                         "data.txt": "one\n", "tests/test_data.py": TEST_DATA})


def _breaks_data() -> str:
    from tests_support import render

    return render({"data.txt": "two\n"})


def _assert_sent_back(text: str) -> None:
    import test_self_check as T

    assert DATA_ID in text and "DATA-DETAIL" in text, text
    assert f"{T.ENV_FAILURE} (base-whole-run)" in text, text
    assert f"FAILED {T.ENV_FAILURE}" not in text and "no network" not in text, text


@pytest.mark.usefixtures("fake_gh")
def test_check_to_gate_on_the_walker(tmp_path):
    """The executor walks the graph: the first attempt's check sends test_data back to
    the builder and excuses test_env, whose detail the builder never sees."""
    import test_self_check as T
    from tests_support import FakeModel, render

    repo = _graph_target(tmp_path)
    fixed = render({"data.txt": "one\n", "extra.txt": "x\n"})
    reviews = [*T.REVIEW_ANSWERS[:3], T.VERIFICATION_ANSWER, T.REVIEW_ANSWERS[3]]
    model = FakeModel([*T._answers(_breaks_data()), fixed, *reviews])
    result = T._design(tmp_path / "run", model, repo)
    assert result.exit_ok, result.error
    emits = [prompt for prompt in model.prompts if T.EMIT in prompt]
    assert len(emits) >= 2
    _assert_sent_back(emits[1])


def test_check_to_gate_on_the_spine(tmp_path, monkeypatch):
    """Each node as the spine runs it, from the lowered graph's declarations: the check
    beside the reviews records a reloop that charges test_data and excuses test_env, and
    gate_check routes on it."""
    import test_self_check as T
    from test_first_self_check import _check_node

    run = T._spine_run(_graph_target(tmp_path), tmp_path, monkeypatch, "abcdef744001")
    T._emit(run, _breaks_data(), monkeypatch)
    code, out, err = T._prepare(run, monkeypatch)
    assert code == 0 and T._label(out) == "proceed", err
    verdict = _check_node(run, monkeypatch)
    assert verdict["verdict"] == "reloop"
    _assert_sent_back(verdict["reason"])
    code, out, err = T._node(run, f"{design_ops.__name__}:gate_check", "gate_check", "gate",
                             monkeypatch)
    assert code == 0 and T._label(out) == "reloop", err


def test_the_merged_check_compares_a_touched_files_failure_by_signature(tmp_path):
    """The attempt edited the pre-existing failure's file and moved where it fails: the
    reference fails it too, but not the same way, so it is the change's."""
    ctx, found, merge, tip = _merged(
        tmp_path, {"tests/test_old.py": TEST_OLD}, {"tests/test_old.py": "import os\n\n" + TEST_OLD},
        {"other.txt": "main moved\n"}, check=f"{PYTEST} tests")
    reason, _ = delivery._check_merged(ctx, found, merge, found.head, 120, target_tip=tip)
    assert f"charged: {OLD} (signature-mismatch)" in reason.split("\n", 1)[0], reason
