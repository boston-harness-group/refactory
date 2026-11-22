"""Tests for per-step chain excusal in base_check.

Covers `_chain_steps`, `_build_marker_script`, `_parse_step_results`, `_run_chain_markers`
and `_per_step_comparison` through `filter_check`. The scenario tests feed step records
to the comparison; the real tests run the chain in a shell, and the end-to-end ones run
`run_base_check` and `filter_check` on real worktrees with a pytest stand-in on PATH.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from refactory_dsh import base_check, repo_change, sandbox, worktree
from refactory_dsh.base_check import (
    BaseCheckResult,
    _build_marker_script,
    _chain_steps,
    _parse_step_results,
    _run_chain_markers,
    _runner,
    _runner_from_output,
    _step_record,
    _without_errexit,
    filter_check,
)


# ── Unit tests ───────────────────────────────────────────────────────────


class TestChainSteps:
    """Unit tests for _chain_steps."""

    def test_splits_only_on_chain_operators(self):
        """Pipelines stay whole; only &&, ||, ; split."""
        result = _chain_steps("A | B && C")
        assert result == [("A | B", ""), ("C", "&&")]

    def test_heredoc_refuses_to_split(self):
        """A heredoc command is not split."""
        assert _chain_steps("cat <<EOF\na && b\nEOF && pytest") is None

    def test_compound_commands_refuse_to_split(self):
        """if/for/while/case make a command unsplittable."""
        assert _chain_steps("if true; then pytest; fi && lint") is None
        assert _chain_steps("for x in a b; do echo $x; done && test") is None
        assert _chain_steps("while true; do break; done && lint") is None
        assert _chain_steps("case $x in a) echo;; esac && test") is None

    def test_quoted_heredoc_marker_does_split(self):
        """<< inside quotes does not trigger the heredoc guard."""
        result = _chain_steps('echo "<<EOF" && pytest')
        assert result is not None
        assert len(result) == 2

    def test_backtick_spanning_operator(self):
        """An open backtick at a split point means the substitution spans an operator."""
        assert _chain_steps("cmd `a && b` && c") is None

    def test_records_operators(self):
        """Operators are recorded correctly."""
        result = _chain_steps("A && B || C ; D")
        assert result == [("A", ""), ("B", "&&"), ("C", "||"), ("D", ";")]

    def test_single_step_returns_one_element(self):
        """A single command with no operators returns a single-element list."""
        result = _chain_steps("pytest tests/")
        assert result == [("pytest tests/", "")]

    def test_subshell_stays_whole(self):
        """Content inside parentheses is not split."""
        result = _chain_steps("(a && b) && c")
        assert result == [("(a && b)", ""), ("c", "&&")]

    def test_dollar_paren_stays_whole(self):
        """$(...) with operators inside is not split."""
        result = _chain_steps("echo $(a && b) && c")
        assert result == [("echo $(a && b)", ""), ("c", "&&")]

    def test_quoted_operators_stay_whole(self):
        """Operators inside quotes are not split points."""
        result = _chain_steps("echo 'a && b' && c")
        assert result == [("echo 'a && b'", ""), ("c", "&&")]
        result2 = _chain_steps('echo "a && b" && c')
        assert result2 == [('echo "a && b"', ""), ("c", "&&")]

    def test_reason_recorded_heredoc(self):
        """When a heredoc prevents splitting, the reason is recorded."""
        reason: list[str] = []
        result = _chain_steps("cat <<EOF\ndata\nEOF && pytest", reason=reason)
        assert result is None
        assert len(reason) == 1
        assert "heredoc" in reason[0]

    def test_reason_recorded_keyword(self):
        """When a shell keyword prevents splitting, the reason is recorded."""
        reason: list[str] = []
        result = _chain_steps("if true; then pytest; fi && lint", reason=reason)
        assert result is None
        assert len(reason) == 1
        assert "keyword" in reason[0]
        assert "'if'" in reason[0]

    def test_reason_recorded_backtick(self):
        """When a backtick substitution prevents splitting, the reason is recorded."""
        reason: list[str] = []
        result = _chain_steps("cmd `a && b` && c", reason=reason)
        assert result is None
        assert len(reason) == 1
        assert "backtick" in reason[0]

    def test_reason_empty_on_success(self):
        """When splitting succeeds, no reason is recorded."""
        reason: list[str] = []
        result = _chain_steps("A && B", reason=reason)
        assert result is not None
        assert len(reason) == 0


class TestRunnerFromOutput:
    """Test _runner_from_output: detect runner from output when step text doesn't name it."""

    def test_detects_pytest_from_summary(self):
        """Pytest summary line in output detects pytest."""
        output = "FAILED tests/test_a.py::test_x\n= 1 failed, 3 passed in 0.12s =\n"
        assert _runner_from_output(output) == "pytest"

    def test_detects_vitest_from_summary(self):
        """Vitest summary line in output detects vitest."""
        output = " FAIL  src/app.spec.ts > suite > test T\n Tests  1 failed | 9 passed (10)\n"
        assert _runner_from_output(output) == "vitest"

    def test_returns_none_for_non_runner_output(self):
        """Non-runner output returns None."""
        output = "build failed\nerror: some error\n"
        assert _runner_from_output(output) is None

    def test_runner_step_text_takes_priority(self):
        """_runner from step text is used first; _runner_from_output is fallback."""
        # Step text explicitly names pytest
        assert _runner("pytest tests/") == "pytest"
        # Step text doesn't name a runner, but output has pytest summary
        assert _runner("./run_tests.sh") is None
        assert _runner_from_output("FAILED tests/test_a.py::test_x\n= 1 failed in 0.12s =\n") == "pytest"




class TestChainStepsComments:
    """An unquoted ``#`` starting a word starts a comment; nothing in it is split."""

    def test_operators_in_a_comment_are_not_split(self):
        assert _chain_steps("pytest a && pytest b # then x && y || z") == [
            ("pytest a", ""), ("pytest b", "&&")]

    def test_comment_ends_at_its_line(self):
        assert _chain_steps("pytest a # unit && lint\n&& pytest b") == [
            ("pytest a", ""), ("pytest b", "&&")]

    def test_hash_inside_a_word_or_quotes_is_not_a_comment(self):
        assert _chain_steps("echo a#b && echo '#c && d' && echo ${#X}") == [
            ("echo a#b", ""), ("echo '#c && d'", "&&"), ("echo ${#X}", "&&")]

    def test_runner_named_only_in_a_comment_is_not_a_runner(self):
        steps = _chain_steps("make check # runs pytest && more")
        assert steps == [("make check", "")]
        assert _runner(steps[0][0]) is None


# ── marker script and parsing ────────────────────────────────────────────


def _shell(tmp_path: Path):
    """Run a script as the base and the check copy do: `sandbox.run_command` in a directory."""
    return lambda script: sandbox.run_command(script, workdir=tmp_path, timeout_s=30,
                                              home=tmp_path)


class TestMarkerScriptAndParsing:
    """`_build_marker_script` and `_parse_step_results`."""

    def test_script_merges_stderr_first(self):
        assert _build_marker_script([("a", ""), ("b", "&&")], "abc").startswith("exec 2>&1\n")

    def test_parsing_round_trip(self):
        steps = [("echo hello", ""), ("pytest tests/", "&&"), ("echo c", "||")]
        mark = "___REFACTORY_STEP_abcdef012345_"
        stdout = (f"hello\n{mark}0_EXIT_0\nFAILED tests/test_a.py::test_x - assert False\n"
                  f"{mark}1_EXIT_1\n{mark}2_SKIPPED\n")
        results = _parse_step_results(stdout, "abcdef012345", steps)
        assert [(r["ran"], r["exit_code"]) for r in results] == [(True, 0), (True, 1), (False, None)]
        assert results[0]["output"] == "hello"
        assert "FAILED" in results[1]["output"]

    def test_missing_marker_or_other_id_is_unreadable(self):
        steps = [("echo hello", ""), ("pytest", "&&")]
        assert _parse_step_results("hello\n___REFACTORY_STEP_abc_0_EXIT_0\n", "abc", steps) is None
        assert _parse_step_results("___REFACTORY_STEP_zzz_0_EXIT_0\n___REFACTORY_STEP_zzz_1_EXIT_0",
                                   "abc", steps) is None


class TestOneShell:
    """The steps run in one shell through `_run_chain_markers`, so their effects carry."""

    def test_cd_and_export_carry_to_later_steps(self, tmp_path):
        steps = _chain_steps("mkdir -p sub && cd sub && pwd && export X=1 && sh -c 'echo $X'")
        results = _run_chain_markers(steps, _shell(tmp_path))
        assert results is not None
        assert [r["exit_code"] for r in results] == [0, 0, 0, 0, 0]
        assert results[2]["output"].strip() == str((tmp_path / "sub").resolve())
        assert results[4]["output"].strip() == "1"

    def test_stderr_is_in_the_step_output(self, tmp_path):
        results = _run_chain_markers(_chain_steps("echo out && echo err >&2 && true"),
                                     _shell(tmp_path))
        assert results[1]["output"] == "err"


class TestShellOperatorSemantics:
    """``&&`` steps run whatever came before; ``||`` steps only when the shell would run them."""

    def _ran(self, tmp_path: Path, command: str) -> list[tuple[bool, int | None, str]]:
        results = _run_chain_markers(_chain_steps(command), _shell(tmp_path))
        assert results is not None
        return [(r["ran"], r["exit_code"], r["output"]) for r in results]

    def test_and_step_runs_after_a_failure(self, tmp_path):
        assert self._ran(tmp_path, "false && echo ran") == [(True, 1, ""), (True, 0, "ran")]

    def test_or_chain_after_success_runs_nothing(self, tmp_path):
        assert self._ran(tmp_path, "true || echo b || echo c") == [
            (True, 0, ""), (False, None, ""), (False, None, "")]

    def test_or_chain_runs_until_one_succeeds(self, tmp_path):
        assert self._ran(tmp_path, "false || echo b || echo c") == [
            (True, 1, ""), (True, 0, "b"), (False, None, "")]

    def test_or_follows_the_chain_status_not_the_previous_step(self, tmp_path):
        # The shell reads `false && echo x || echo y` as `(false && echo x) || echo y`: x is
        # skipped by the shell (run here anyway) and the chain is still failing, so y runs.
        assert self._ran(tmp_path, "false && echo x || echo y") == [
            (True, 1, ""), (True, 0, "x"), (True, 0, "y")]
        assert self._ran(tmp_path, "true && true || echo y") == [
            (True, 0, ""), (True, 0, ""), (False, None, "")]

    def test_install_guard(self, tmp_path):
        command = "test -d node_modules || mkdir node_modules && echo built"
        assert self._ran(tmp_path, command) == [(True, 1, ""), (True, 0, ""), (True, 0, "built")]
        # Installed now: the shell skips the install, so the marker run does too.
        assert self._ran(tmp_path, command) == [(True, 0, ""), (False, None, ""), (True, 0, "built")]

    def test_unreadable_markers_warn_and_return_none(self, tmp_path, capsys):
        # `_chain_steps` refuses an `exit` step; one that ends the shell anyway loses its markers.
        steps = [("echo a", ""), ("exit 3", "&&"), ("echo c", "&&")]
        assert _run_chain_markers(steps, _shell(tmp_path)) is None
        assert "could not be read: 1 of 3 step markers" in capsys.readouterr().err


# ── scenario tests: step records through filter_check ───────────────────


def _pytest_out(failures: list[str], passed: int = 3) -> str:
    lines = [f"FAILED {f} - assert False" for f in failures]
    lines.append(f"========== {len(failures)} failed, {passed} passed in 0.12s ==========")
    return "\n".join(lines) + "\n"


def _vitest_out(failures: list[str], passed: int = 9) -> str:
    lines = [f" FAIL  {f}" for f in failures]
    lines.append(f"      Tests  {len(failures)} failed | {passed} passed ({len(failures) + passed})")
    return "\n".join(lines) + "\n"


def _recs(*steps: tuple[str, str, int, str]) -> list[dict]:
    return [_step_record({"step": s, "operator": op, "ran": True, "output": out, "exit_code": code})
            for s, op, code, out in steps]


def _scenario(tmp_path: Path, command: str, base_steps: list[dict], attempt_steps: list[dict],
              attempt_out: str, *, base_complete: bool = True) -> Any:
    """`filter_check` on a failed base with ``base_steps`` and an attempt whose re-run gives ``attempt_steps``."""
    base = BaseCheckResult(ok=False, failures={f for sr in base_steps for f in sr["failures"]},
                           output="", exit_code=1)
    base._base_complete = base_complete  # type: ignore[attr-defined]
    base._step_results = base_steps  # type: ignore[attr-defined]
    ctx = SimpleNamespace(workdir=tmp_path, facts={}, exists=lambda name: False,
                          read_text=lambda name: "", budget=None)
    attempt = sandbox.SandboxResult(ok=False, stdout=attempt_out, stderr="", returncode=1)
    found = SimpleNamespace(path=str(tmp_path / "run-x"), base="abc", head="def",
                            target=str(tmp_path / "target"))
    with patch.object(base_check, "run_base_check", return_value=base), \
         patch.object(base_check, "_run_chain_markers", return_value=attempt_steps), \
         patch.object(base_check.worktree, "recorded", return_value=found), \
         patch.object(base_check, "_all_changed_files", return_value=[]):
        return filter_check(ctx, attempt, command, 60.0)


S = "tests/test_a.py::test_S"
N = "tests/test_a.py::test_N"


def test_tok_sh_pytest_grep(tmp_path):
    """S inherited; N and the grep step stay the attempt's."""
    filtered = _scenario(
        tmp_path, "tok.sh && pytest X && grep -q Y f",
        _recs(("tok.sh", "", 1, "tok failed"), ("pytest X", "&&", 1, _pytest_out([S])),
              ("grep -q Y f", "&&", 1, "")),
        _recs(("tok.sh", "", 1, "tok failed"), ("pytest X", "&&", 1, _pytest_out([S, N], passed=2)),
              ("grep -q Y f", "&&", 1, "")),
        "tok failed\n")
    assert filtered.inherited == {S}
    assert N in filtered.new_failures
    assert not filtered.all_excused


def test_earlier_step_ending_differently_charges_the_failure(tmp_path):
    """pnpm build && vitest: build failed on the base and passed on the attempt, so T is the attempt's."""
    t = "src/app.spec.ts > suite > test T"
    filtered = _scenario(
        tmp_path, "pnpm build && vitest",
        _recs(("pnpm build", "", 1, "build failed"), ("vitest", "&&", 1, _vitest_out([t]))),
        _recs(("pnpm build", "", 0, "build ok"), ("vitest", "&&", 1, _vitest_out([t]))),
        _vitest_out([t]))
    assert t in filtered.new_failures
    assert not filtered.all_excused


def test_same_outcomes_excuse(tmp_path):
    """cd pkg && pytest: cd ended alike, pytest accounted on both sides, so x is excused."""
    x = "pkg/tests/test_a.py::test_x"
    out = _pytest_out([x])
    filtered = _scenario(tmp_path, "cd pkg && pytest",
                         _recs(("cd pkg", "", 0, ""), ("pytest", "&&", 1, out)),
                         _recs(("cd pkg", "", 0, ""), ("pytest", "&&", 1, out)), out)
    assert filtered.inherited == {x}
    assert filtered.all_excused


def test_attempt_rerun_finds_failures_its_check_never_ran(tmp_path):
    """pytest unit && ruff check . && pytest integ: the attempt's check stopped at unit; its re-run finds N."""
    unit, integ = "tests/test_unit.py::test_S", "tests/test_integ.py::test_N"
    filtered = _scenario(
        tmp_path, "pytest unit && ruff check . && pytest integ",
        _recs(("pytest unit", "", 1, _pytest_out([unit])), ("ruff check .", "&&", 0, "ok"),
              ("pytest integ", "&&", 0, "========== 3 passed in 0.1s ==========")),
        _recs(("pytest unit", "", 1, _pytest_out([unit])), ("ruff check .", "&&", 0, "ok"),
              ("pytest integ", "&&", 1, _pytest_out([integ]))),
        _pytest_out([unit]))
    assert filtered.inherited == {unit}
    assert integ in filtered.new_failures
    assert not filtered.all_excused
    assert "pytest integ" in filtered.new_failure_detail


def test_failing_and_step_does_not_stop_the_comparison(tmp_path):
    """tok.sh failing on both sides does not drop the pytest step behind it: S is inherited."""
    filtered = _scenario(
        tmp_path, "tok.sh && pytest X",
        _recs(("tok.sh", "", 1, "tok failed"), ("pytest X", "&&", 1, _pytest_out([S]))),
        _recs(("tok.sh", "", 1, "tok failed"), ("pytest X", "&&", 1, _pytest_out([S]))),
        "tok failed\n")
    assert filtered.inherited == {S}
    assert filtered.all_excused


def test_unaccounted_base_step_blocks_the_excusal(tmp_path):
    """The base's pytest named S but counted two failures: S is not excused."""
    filtered = _scenario(
        tmp_path, "tok.sh && pytest tests/",
        _recs(("tok.sh", "", 1, "tok failed"),
              ("pytest tests/", "&&", 1, f"FAILED {S} - x\n========== 2 failed in 0.1s ==========")),
        _recs(("tok.sh", "", 1, "tok failed"), ("pytest tests/", "&&", 1, _pytest_out([S]))),
        _pytest_out([S]), base_complete=False)
    assert S in filtered.new_failures
    assert not filtered.all_excused


def test_non_test_step_regression_is_charged(tmp_path):
    """pytest X && check-output.sh: the second step passed on the base and failed on the attempt."""
    out = _pytest_out([S])
    filtered = _scenario(tmp_path, "pytest X && check-output.sh",
                         _recs(("pytest X", "", 1, out), ("check-output.sh", "&&", 0, "all good")),
                         _recs(("pytest X", "", 1, out), ("check-output.sh", "&&", 1, "bad")), out)
    assert filtered.inherited == {S}
    assert not filtered.all_excused
    assert "check-output.sh" in filtered.new_failure_detail


def test_passing_base_needs_no_step_run(tmp_path):
    """A passing base excuses nothing, so the attempt's steps are not run again."""
    base = BaseCheckResult(ok=True, failures=set(), output="", exit_code=0)
    ctx = SimpleNamespace(workdir=tmp_path, facts={}, exists=lambda name: False,
                          read_text=lambda name: "", budget=None)
    attempt = sandbox.SandboxResult(ok=False, stdout=_pytest_out([S]), stderr="", returncode=1)
    with patch.object(base_check, "run_base_check", return_value=base), \
         patch.object(base_check, "_run_chain_markers") as markers, \
         patch.object(base_check.worktree, "recorded",
                      return_value=SimpleNamespace(path=str(tmp_path), base="a", head="b")), \
         patch.object(base_check, "_all_changed_files", return_value=[]):
        filtered = filter_check(ctx, attempt, "tok.sh && pytest tests/", 60.0)
    assert S in filtered.new_failures and not filtered.all_excused
    markers.assert_not_called()


def test_heredoc_is_not_split():
    reason: list[str] = []
    assert _chain_steps("cat <<EOF\ndata\nEOF && pytest tests/", reason=reason) is None
    assert "heredoc" in reason[0]


@pytest.mark.parametrize("step, rewritten", [
    ("set -e", ":"),
    ("set -euo pipefail", "set -u -o pipefail"),
    ("set -o errexit -o nounset", "set -o nounset"),
    ("set -ex", "set -x"),
    ("set +e", "set +e"),
    ("set -- a -e", "set -- a -e"),
    ("pytest -e", "pytest -e"),
])
def test_set_step_drops_errexit_only(step, rewritten):
    assert _without_errexit(step) == rewritten


@pytest.mark.parametrize("command, why", [
    ("pytest unit; exit 1", "exit"),
    ("trap 'rm -f x' EXIT; pytest unit", "trap"),
    ("set $OPTS && pytest unit", "set"),
    ("shopt -s -o errexit; pytest unit", "errexit"),
    ("eval set -e; pytest unit", "errexit"),
])
def test_steps_the_shared_shell_cannot_run_are_not_split(command, why):
    reason: list[str] = []
    assert _chain_steps(command, reason=reason) is None
    assert why in reason[0]


def test_set_euo_pipefail_keeps_pipefail(tmp_path):
    """errexit goes, so the later steps run; pipefail stays, so a later pipeline's failure shows."""
    steps = _chain_steps("set -euo pipefail; false | true && echo after")
    assert "set -u -o pipefail" in _build_marker_script(steps, "x")
    results = _run_chain_markers(steps, _shell(tmp_path))
    assert [(r["exit_code"], r["output"]) for r in results] == [(0, ""), (1, ""), (0, "after")]


def test_base_with_unreadable_steps_charges(tmp_path):
    """The base's per-step run could not be read: nothing is excused."""
    out = _pytest_out([S])
    base = BaseCheckResult(ok=False, failures={S}, output="", exit_code=1)
    base._steps_unreadable = True  # type: ignore[attr-defined]
    ctx = SimpleNamespace(workdir=tmp_path, facts={}, exists=lambda name: False,
                          read_text=lambda name: "", budget=None)
    attempt = sandbox.SandboxResult(ok=False, stdout=out, stderr="", returncode=1)
    with patch.object(base_check, "run_base_check", return_value=base), \
         patch.object(base_check, "_run_chain_markers") as markers, \
         patch.object(base_check.worktree, "recorded",
                      return_value=SimpleNamespace(path=str(tmp_path), base="a", head="b")), \
         patch.object(base_check, "_all_changed_files", return_value=[]):
        filtered = filter_check(ctx, attempt, "tok.sh && pytest tests/", 60.0)
    assert not filtered.all_excused and not filtered.inherited
    markers.assert_not_called()


# ── end to end: run_base_check and filter_check on real worktrees ───────


#: A pytest stand-in, on the check's PATH through the worktree's ``node_modules/.bin``
#: (`worktree.check_environment`). ``results/<suite>.txt`` names the tests that fail;
#: ``@notests`` collects nothing (exit 5), ``@unaccounted`` counts one failure more than
#: it names, ``@spam N`` prints N lines of noise. Every call is logged with its directory.
_FAKE_PYTEST = """#!/bin/sh
echo "$PWD $1" >> "{log}"
f="results/$1.txt"
names=""; mode=""
if [ -f "$f" ]; then names=$(grep -v '^@' "$f"); mode=$(grep '^@' "$f" | head -n 1); fi
n=0
for t in $names; do echo "FAILED $t - assert False"; n=$((n + 1)); done
case "$mode" in
  "@notests") echo "no tests ran in 0.01s"; exit 5 ;;
  "@unaccounted") n=$((n + 1)) ;;
  "@spam "*) i=0; while [ "$i" -lt "${{mode#@spam }}" ]; do
      echo "noise $i ......................................................"; i=$((i + 1)); done ;;
esac
if [ "$n" -gt 0 ]; then echo "========== $n failed, 3 passed in 0.10s =========="; exit 1; fi
echo "========== 3 passed in 0.10s =========="
"""

#: A lint or build stand-in: ``results/<tool>.txt`` lists its errors, each printed with
#: the absolute paths a real tool prints (the worktree's and HOME's), which differ by side.
_FAKE_TOOL = """#!/bin/sh
f="results/$(basename "$0").txt"
[ -s "$f" ] || { echo "$(basename "$0"): ok"; exit 0; }
while IFS= read -r e; do echo "$PWD/src/app.py:1: $e (cache $HOME/.cache)"; done < "$f"
exit 1
"""

UNIT_S = "tests/test_unit.py::test_S"
INTEG_N = "tests/test_integ.py::test_N"


class _Ctx:
    """The parts of a run's context `run_check` and `filter_check` use."""

    def __init__(self, workdir: Path, facts: dict[str, Any]):
        self.workdir = workdir
        self.facts = dict(facts)
        self.budget = None

    def knob(self, name: str, default: Any = None) -> Any:
        return default

    def set_fact(self, name: str, value: Any) -> None:
        self.facts[name] = value

    def exists(self, name: str) -> bool:
        return (self.workdir / name).exists()

    def read_text(self, name: str) -> str:
        return (self.workdir / name).read_text()

    def write_text(self, name: str, text: str) -> Path:
        (self.workdir / name).write_text(text)
        return self.workdir / name


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def _commit_files(repo: Path, files: dict[str, str]) -> str:
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        if name.startswith("node_modules/.bin/"):
            path.chmod(0o755)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "change")
    return _git(repo, "rev-parse", "HEAD")


class _Run(SimpleNamespace):
    ctx: _Ctx
    found: worktree.Worktree
    log: Path

    def base_calls(self) -> list[str]:
        if not self.log.exists():
            return []
        run_path = Path(self.found.path).resolve()
        base = run_path.parent / f"base-{run_path.name.removeprefix('run-')}"
        return [line for line in self.log.read_text().splitlines()
                if line.startswith(str(base))]


def _run(tmp_path: Path, base: dict[str, str], attempt: dict[str, str]) -> _Run:
    """A target whose base fails as ``base`` says and a run whose attempt fails as ``attempt`` says."""
    log = tmp_path / "pytest-calls.log"
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "T")
    base_commit = _commit_files(repo, {
        "node_modules/.bin/pytest": _FAKE_PYTEST.format(log=log),
        "node_modules/.bin/lint": _FAKE_TOOL,
        "node_modules/.bin/build": _FAKE_TOOL,
        **{f"results/{suite}.txt": text for suite, text in base.items()},
    })
    run_dir = tmp_path / "run"
    worktree.create(repo, base_commit, "chain0excuse", run_dir)
    found = worktree.commit_attempt(
        run_dir, {Path("change.txt"): "the attempt\n",
                  **{Path(f"results/{suite}.txt"): text for suite, text in attempt.items()}},
        "attempt")
    worktree.write_wrapper(found, run_dir)
    return _Run(ctx=_Ctx(run_dir, {"commit": found.head}), found=found, log=log)


def _check(run: _Run, command: str) -> Any:
    """The attempt's own check, then `filter_check`, as design mode runs them."""
    result = repo_change.run_check(run.ctx, command, 60)
    assert not result.ok, result.stdout
    return filter_check(run.ctx, result, command, 60, named_text="")


def test_base_stopping_at_its_first_step_still_records_its_steps(tmp_path):
    """H1: the base stops at unit, which the whole-command blocker calls complete; the
    attempt fails S too and breaks integ, which only its re-run reaches."""
    run = _run(tmp_path, base={"unit": UNIT_S}, attempt={"unit": UNIT_S, "integ": INTEG_N})
    filtered = _check(run, "pytest unit && pytest integ")
    assert not filtered.all_excused
    assert INTEG_N in filtered.new_failures
    assert filtered.inherited == {UNIT_S}
    cached = json.loads((run.ctx.workdir / base_check.CACHE_FILE).read_text())
    assert [(s["step"], s["exit_code"]) for s in cached["steps"]] == [
        ("pytest unit", 1), ("pytest integ", 0)]


def test_shared_failure_in_one_shell_is_excused(tmp_path):
    """The same failure in the same step, every step ending alike: excused."""
    run = _run(tmp_path, base={"unit": UNIT_S}, attempt={"unit": UNIT_S, "integ": ""})
    filtered = _check(run, "pytest unit && pytest integ")
    assert filtered.all_excused
    assert filtered.inherited == {UNIT_S}
    assert filtered.base_complete


@pytest.mark.parametrize("base_integ, excused", [("", False), ("@notests", True)])
def test_runner_step_failing_without_naming_a_test(tmp_path, base_integ, excused):
    """H2: integ collects nothing on the attempt (exit 5). It is charged unless the base's
    integ ended the same way."""
    run = _run(tmp_path, base={"unit": UNIT_S, "integ": base_integ},
               attempt={"unit": UNIT_S, "integ": "@notests"})
    filtered = _check(run, "pytest unit && pytest integ")
    assert filtered.all_excused is excused
    if not excused:
        assert "pytest integ` exited 5" in filtered.new_failure_detail


def test_marker_run_leaves_no_files_in_the_run_worktree(tmp_path):
    """H3: the attempt's steps run in its check copy, as its check beside the reviews does."""
    run = _run(tmp_path, base={"unit": UNIT_S}, attempt={"unit": UNIT_S})
    filtered = _check(run, "pytest unit && touch made_by_marker")
    assert filtered.all_excused
    assert not (Path(run.found.path) / "made_by_marker").exists()
    run_id = Path(run.found.path).name.removeprefix("run-")
    copy = repo_change._check_copy_path(Path(run.found.target), run_id)
    assert (copy / "made_by_marker").exists()
    assert _git(Path(run.found.path), "status", "--porcelain") == ""


def test_cache_hit_never_reruns_the_base(tmp_path):
    """The base's steps are cached under its key; a second check reads them and runs nothing on the base."""
    run = _run(tmp_path, base={"unit": UNIT_S}, attempt={"unit": UNIT_S})
    command = "pytest unit && pytest integ"
    first = _check(run, command)
    calls = len(run.base_calls())
    assert calls == 3  # the check stops at unit; the steps run unit and integ
    second = _check(run, command)
    assert len(run.base_calls()) == calls
    assert first.all_excused and second.all_excused


def test_base_complete_needs_its_runner_accounted(tmp_path):
    """The base's unit names S but counts two failures: incomplete, and S is not excused."""
    run = _run(tmp_path, base={"unit": f"{UNIT_S}\n@unaccounted"},
               attempt={"unit": UNIT_S})
    filtered = _check(run, "pytest unit && pytest integ")
    assert filtered.base_complete is False
    assert not filtered.all_excused


def test_cut_output_charges(tmp_path, monkeypatch, capsys):
    """M1: the attempt's per-step output is cut between its markers, so nothing is excused."""
    monkeypatch.setattr(sandbox, "OUTPUT_KEEP_BYTES", 4096)
    monkeypatch.setattr(sandbox, "OUTPUT_TAIL_BYTES", 512)
    # The steps pass alike on both sides; only the attempt's print enough to be cut.
    run = _run(tmp_path, base={"unit": UNIT_S},
               attempt={"unit": UNIT_S, "integ": "@spam 200", "e2e": "@spam 200"})
    filtered = _check(run, "pytest unit && pytest integ && pytest e2e")
    assert not filtered.all_excused
    assert not filtered.inherited
    assert "could not be read" in filtered.inherited_detail
    assert "step markers in its kept output" in capsys.readouterr().err


@pytest.mark.parametrize("tool", ["lint", "build"])
@pytest.mark.parametrize("attempt_errors, excused", [("E1", True), ("E1\nE2", False)])
def test_new_error_in_a_step_failing_on_the_base(tmp_path, tool, attempt_errors, excused):
    """The base's lint (or build) fails with E1 and its tests with S. The same E1 -- printed
    with each side's own paths -- leaves S excused; an added E2 is the attempt's."""
    run = _run(tmp_path, base={tool: "E1", "unit": UNIT_S},
               attempt={tool: attempt_errors, "unit": UNIT_S})
    filtered = _check(run, f"{tool} && pytest unit")
    assert filtered.all_excused is excused
    assert filtered.inherited == ({UNIT_S} if excused else set())
    if not excused:
        assert f"`{tool}` exited 1" in filtered.new_failure_detail


@pytest.mark.parametrize("command", ["set -e; pytest unit",
                                     "set -euo pipefail; pytest unit && pytest integ"])
def test_errexit_chain_excuses_the_base_failure(tmp_path, command):
    """errexit would end the marker shell at S; the marker script drops it, so S is compared."""
    run = _run(tmp_path, base={"unit": UNIT_S}, attempt={"unit": UNIT_S})
    filtered = _check(run, command)
    assert filtered.all_excused
    assert filtered.inherited == {UNIT_S}


def test_cache_without_steps_takes_the_whole_command_path(tmp_path):
    """A base cache written before per-step results existed is read as before (correction 2):
    the whole-command blocker decides, and the attempt's steps are not run again."""
    run = _run(tmp_path, base={"unit": UNIT_S}, attempt={"unit": UNIT_S})
    command = "pytest unit && pytest integ"
    cache = run.ctx.workdir / base_check.CACHE_FILE
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "cache_key": base_check.cache_key(run.found.base, "", "", command),
        "ok": False, "failures": [UNIT_S], "output": "", "exit_code": 1,
        "environment_failure": None, "base_complete": True, "base_clean": True,
    }))
    filtered = _check(run, command)
    assert filtered.all_excused  # what main decides for this cache
    run_id = Path(run.found.path).name.removeprefix("run-")
    assert not repo_change._check_copy_path(Path(run.found.target), run_id).exists()
    assert run.base_calls() == []


def test_cut_base_output_charges(tmp_path, monkeypatch):
    """The base's per-step output is cut between its markers: recorded as unreadable, so
    nothing is excused -- unlike a cache from before per-step results, which is not."""
    monkeypatch.setattr(sandbox, "OUTPUT_KEEP_BYTES", 4096)
    monkeypatch.setattr(sandbox, "OUTPUT_TAIL_BYTES", 512)
    run = _run(tmp_path, base={"unit": UNIT_S, "integ": "@spam 200", "e2e": "@spam 200"},
               attempt={"unit": UNIT_S, "integ": "", "e2e": ""})
    filtered = _check(run, "pytest unit && pytest integ && pytest e2e")
    assert not filtered.all_excused
    assert "base's per-step results could not be read" in filtered.inherited_detail
    cached = json.loads((run.ctx.workdir / base_check.CACHE_FILE).read_text())
    assert cached["steps_unreadable"] is True


def test_signature_ignores_each_sides_own_paths():
    """The same error printed under each side's worktree, HOME and temporary files matches;
    another error does not. (The end-to-end tests' worktrees all sit under /tmp, so they
    cannot tell the root and HOME replacement from the temporary-directory one.)"""
    from refactory_dsh.base_check import _roots, _signature

    base = _roots((Path("/work/wt/base-1"), "<root>"), (Path("/work/run/.base-home"), "<home>"))
    attempt = _roots((Path("/work/wt/check-1"), "<root>"), (Path("/work/run/.copy-home"), "<home>"))

    def lint(root: str, home: str, error: str) -> str:
        return f"{root}/src/app.py:1: {error} (cache {home}/.cache, scratch /tmp/tmpq8x1/out)"

    same = _signature(lint("/work/wt/base-1", "/work/run/.base-home", "E1"), base)
    assert same == _signature(lint("/work/wt/check-1", "/work/run/.copy-home", "E1"), attempt)
    assert same != _signature(lint("/work/wt/check-1", "/work/run/.copy-home", "E2"), attempt)
