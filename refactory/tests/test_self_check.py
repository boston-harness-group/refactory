"""Design mode's self-check: the tests a change can break run before the reviews (b41a).

Every attempt's self-check, first or fix-up, runs in `prepare_check` on the tests the
change touches (`test_selection`), in chunks within a time limit
(`failfast.targeted_prepass`), and sends the change's failures back to emit as a fix-up
that spends no attempt and runs no review (`test_first_self_check.py`). The whole check
runs beside the reviews, not here. Each test here drives the real runner, the real
worktree and, for routes and loops, the real design graph.
"""

from __future__ import annotations

import io
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from refactory_dsh import design_ops, failfast, lowering, repo_change, test_selection, worktree
from refactory_dsh.check_digest import build_probe_command
from refactory_dsh.design import DesignMode
from refactory_dsh.fast import DesignFastMode
from refactory_dsh.invocations import CHECK_PREP, SELF_CHECK_RECORD
from refactory_dsh.paths import REPO_ROOT
from refactory_dsh.run import main as node_main
from tests_support import (DESIGN_DOC_ANSWER, PITFALLS_ANSWER,
                           RESEARCH_ANSWER, REVIEW_ANSWERS, VERIFICATION_ANSWER, FakeModel,
                           early_asks_judgment, git, git_repo, render)

NAME = "selfcheck_tool"
CHECK = "python -m pytest -q tests"
HEALTH = "You are the health checker."
EMIT = "Write the artifact `"
PIN = "tests/test_pin.py::test_pin"


# ── helpers ───────────────────────────────────────────────────────────────────


#: A design run plans its delivery by asking gh for its login, so it asks the fake (dm-271).
pytestmark = pytest.mark.usefixtures("fake_gh")


def _target(root: Path, files: dict[str, str]) -> Path:
    """A repository on main with ``files`` committed over `git_repo`'s one file."""
    repo = git_repo(root)
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "tests")
    return repo


#: A tool and a test that pins its version, as a test pins behaviour a change replaces.
PINNED = {
    "tool.py": "VERSION = 1\n",
    "tests/test_pin.py": "import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 1\n",
}


def _answers(*emits: str) -> list[str]:
    return [RESEARCH_ANSWER, PITFALLS_ANSWER, "perspectives", "one plan",
            DESIGN_DOC_ANSWER, *emits,
            *REVIEW_ANSWERS[:3], VERIFICATION_ANSWER, REVIEW_ANSWERS[3]]


def _mode(**knobs) -> DesignMode:
    mode = DesignMode()
    state = mode.state_dict()
    for key, value in knobs.items():
        state["knobs"][key]["value"] = value
    mode.load_state_dict(state)
    return mode


def _design(run: Path, model: FakeModel, repo: Path, mode: DesignMode | None = None, **task: str):
    return (mode or DesignMode()).run(
        run, model=model, verdicts={"gate_strategy": "proceed"},
        task={"name": NAME, "brief": "a tool", "check": CHECK, "target": str(repo), **task})


def _records(run: Path) -> list[dict]:
    """Every self-check the run recorded; a run that recorded none fails the test."""
    return [json.loads(line) for line in (run / SELF_CHECK_RECORD).read_text().splitlines()]


def _count(model: FakeModel, marker: str) -> int:
    return sum(marker in prompt for prompt in model.prompts)


def _commits(run: Path) -> list[str]:
    found = worktree.recorded(run)
    return git(Path(found.path), "log", "--format=%s", f"{found.base}..HEAD").splitlines()


def _spine_run(repo: Path, tmp_path: Path, monkeypatch, run_id: str, **knobs) -> Path:
    """A run directory the spine would run: a worktree, declarations, facts and knobs."""
    run = tmp_path / "run"
    worktree.create(repo, "HEAD", run_id, run)
    lowering.write_declarations(_mode(**knobs).state_dict(), run)
    (run / "facts.json").write_text(json.dumps({"name": NAME, "brief": "a tool", "check": CHECK}))
    monkeypatch.setenv("REFACTORY_KNOBS", json.dumps({"check_attempts": 3, "refusal_attempts": 3,
                                                      "self_check_fixups": 2,
                                                      "self_check_timeout_s": 300,
                                                      "check_timeout_s": 180, **knobs}))
    return run


def _node(run: Path, target: str, node: str, kind: str, monkeypatch) -> tuple[int, str, str]:
    """One node visit as the spine makes it: `refactory_dsh.run` in its own process."""
    monkeypatch.setenv("REFACTORY_WORKDIR", str(run))
    monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": node, "type": kind}))
    out, err = io.StringIO(), io.StringIO()
    saved = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        code = node_main([target, "--node", node])
    finally:
        sys.stdout, sys.stderr = saved
    return code, out.getvalue(), err.getvalue()


def _emit(run: Path, emission: str, monkeypatch) -> None:
    (run / "emit.txt").write_text(emission)
    code, _, err = _node(run, f"{repo_change.OPS}:materialise", "materialise", "op", monkeypatch)
    assert code == 0, err


def _label(out: str) -> str:
    """The verdict a gate printed: its last line."""
    lines = out.strip().splitlines()
    return lines[-1].strip() if lines else ""


def _prepare(run: Path, monkeypatch, judge: str | None = None) -> tuple[int, str, str]:
    """The self-check as the spine runs it: prepare_check decides on the self-check's
    touched tests. The judge's verdict is written where the agent node writes it when
    ``judge`` is given; `test_the_judge_runs_through_the_worker_on_the_spine` runs the
    agent node itself."""
    if judge is not None:
        (run / design_ops.EARLY_ASKS_VERDICT).write_text(judge)
    return _node(run, f"{design_ops.__name__}:prepare_check", "prepare_check", "gate",
                 monkeypatch)


def _as_a_fixup(run: Path) -> None:
    """Make the attempt's next self-check a fix-up's, as a first self-check that sent the
    attempt back leaves it (``self_check_fixup_count``). A fix-up's self-check runs in
    chunks within the self-check's limit: the tests of the chunks, the cut, the bounds and
    the files that run alone start here. Call it after the emission is materialised."""
    facts = _facts(run)
    (run / "facts.json").write_text(json.dumps({**facts, "self_check_fixup_count": 1}))


def _facts(run: Path) -> dict:
    return json.loads((run / "facts.json").read_text())


class _Chunks:
    """The self-check's chunks (`repo_change.run_check` with ``prepared``), answered at once as
    pytest and the sandbox would answer them, on a clock of their own (``failfast.time``).

    A chunk that names a file of ``slow`` runs past the time it is given and is stopped
    there, as the sandbox stops it: the clock moves by that time and the result is the
    sandbox's timeout. Any other chunk fails the ids of ``fails`` that it names, or whose
    file it names, and passes the rest. With ``emissions``, ``fails`` follows the emission
    under test instead: the n-th commit the self-check runs on fails ``emissions[n]``. The
    whole check and the base's runs still run for real.
    `test_a_self_check_that_says_nothing_goes_on_with_no_fixup` stops a real chunk at a real
    limit; the tests of pytest's own output (junit seconds, a not-found id, a module that
    does not import, a parametrized id) run real chunks.
    """

    def __init__(self, monkeypatch, *, fails=(), slow=(), emissions=None):
        self.fails, self.slow, self.now = set(fails), set(slow), 0.0
        self.commits: list[str] = []
        monkeypatch.setattr(failfast, "time", SimpleNamespace(monotonic=lambda: self.now))
        real = repo_change.run_check

        def run_check(ctx, command, timeout, **kwargs):
            if not kwargs.get("prepared"):
                return real(ctx, command, timeout, **kwargs)
            commit = str(ctx.facts.get("commit"))
            if commit not in self.commits:
                self.commits.append(commit)
                if emissions is not None:
                    self.fails = set(emissions[len(self.commits) - 1])
            return self(command, timeout)

        monkeypatch.setattr(repo_change, "run_check", run_check)

    def __call__(self, command: str, timeout: float):
        from refactory_dsh.sandbox import SandboxResult

        named = [word for word in shlex.split(command) if word.endswith(".py") or ".py::" in word]
        # Full command (no individual .py files named): all failures apply.
        if not named:
            failed = sorted(self.fails)
            if failed:
                return SandboxResult(ok=False, returncode=1, elapsed_s=0.1, stdout="".join(
                    f"FAILED {test_id} - AssertionError\n" for test_id in failed)
                    + f"{len(failed)} failed in 0.10s")
            return SandboxResult(ok=True, returncode=0, stdout="all passed in 0.10s",
                                 elapsed_s=0.1)
        if any(name.split("::")[0] in self.slow for name in named):
            self.now += timeout
            return SandboxResult(ok=False, returncode=124, timed_out=True, timeout_s=int(timeout),
                                 elapsed_s=timeout, stderr=f"timed out after {int(timeout)}s")
        failed = sorted(test_id for test_id in self.fails
                        if test_id in named or test_id.split("::")[0] in named)
        if failed:
            return SandboxResult(ok=False, returncode=1, elapsed_s=0.1, stdout="".join(
                f"FAILED {test_id} - AssertionError\n" for test_id in failed)
                + f"{len(failed)} failed in 0.10s")
        return SandboxResult(ok=True, returncode=0, stdout=f"{len(named)} passed in 0.10s",
                             elapsed_s=0.1)


def _unwatched(monkeypatch) -> None:
    """No snapshot of the target around each agent visit (`target_guard.watch`). In a loop of
    many emissions the snapshots' git calls take most of the time. The loops that use this
    never reach the reviews, so nothing runs beside an agent for the guard to catch;
    test_target_guard and test_design_worktree walk runs with it."""
    from refactory_dsh import target_guard

    monkeypatch.setattr(target_guard, "watch", lambda workdir, node: None)


# ── the loop, on the walker, through the real design graph ────────────────────


def test_a_stale_generated_file_and_a_pinned_test_are_fixed_inside_the_attempt(tmp_path):
    """The first emission changes the tool, which leaves the generated file stale and
    breaks the pinned test. Generate rewrites the file into the attempt; the self-check
    sends the pinned test back; the second emission fixes it; one review fork; no
    attempt spent; one commit with both changes."""
    repo = _target(tmp_path / "repo", {
        **PINNED,
        "gen.py": "import tool\nopen('generated.txt', 'w').write(f'version {tool.VERSION}\\n')\n",
        "generated.txt": "version 1\n",
        "tests/test_generated.py": ("import tool\n\n\ndef test_generated():\n"
                                    "    assert open('generated.txt').read() == f'version {tool.VERSION}\\n'\n"),
    })
    breaks = render({"tool.py": "VERSION = 2\n"})
    fixes = render({"tests/test_pin.py": "import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 2\n"})
    model = FakeModel(_answers(breaks, fixes))
    run = tmp_path / "run"
    result = _design(run, model, repo, generate="python gen.py")
    assert result.exit_ok, result.error

    first, second = _records(run)
    assert first["outcome"] == "fixup" and first["failing_ids"] == [PIN]
    assert set(first["ran"]) == {"tests/test_generated.py", "tests/test_pin.py"} and not first["cut"]
    assert second["outcome"] == "pass" and second["fixups_before"] == 1
    # The fix-up reached emit, led by its reason.
    second_emit = [prompt for prompt in model.prompts if EMIT in prompt][1]
    assert "self-check fix-up 1 of 2" in second_emit and PIN in second_emit
    # Nothing was spent, and the reviews ran once, on the fixed attempt.
    assert not result.facts.get("attempts_used") and not result.facts.get("check_attempts_used")
    assert _count(model, HEALTH) == 1 and _count(model, EMIT) == 2
    # One attempt, one commit, holding the change, the regenerated file and the fix.
    assert _commits(run) == [f"{NAME}: attempt 1"]
    head = Path(worktree.recorded(run).path)
    assert git(head, "show", "HEAD:generated.txt") == "version 2"
    assert "VERSION == 2" in git(head, "show", "HEAD:tests/test_pin.py")
    assert git(head, "show", "HEAD:tool.py") == "VERSION = 2"


def test_a_clean_emission_goes_straight_on(tmp_path):
    repo = _target(tmp_path / "repo", PINNED)
    model = FakeModel(_answers(render({"tool.py": "VERSION = 1\nNAME = 'tool'\n"})))
    run = tmp_path / "run"
    result = _design(run, model, repo)
    assert result.exit_ok, result.error
    [record] = _records(run)
    assert record["outcome"] == "pass" and record["ran"] == record["selection"] == ["tests/test_pin.py"]
    # The first attempt now runs only touched tests (not the whole check)
    assert "tests/test_pin.py" in record["commands"][0]
    assert _count(model, EMIT) == 1 and _count(model, HEALTH) == 1
    assert not result.facts.get("self_check_fixup_count")
    assert not result.facts.get("total_self_check_fixups_used")


def test_the_bound_then_the_early_stop_spends_one_attempt(tmp_path, monkeypatch):
    """A pinned test that keeps failing gets two fix-ups; the third failure is the early
    stop, which spends the attempt. No review runs until an attempt passes. The self-checks
    are `_Chunks`' answers, failing the pinned test until the fourth emission."""
    _Chunks(monkeypatch, emissions=[{PIN}, {PIN}, {PIN}, set()])
    repo = _target(tmp_path / "repo", PINNED)
    still_broken = [render({"tool.py": f"VERSION = {n}\n"}) for n in (2, 3, 4)]
    fixed = render({"tool.py": "VERSION = 5\n",
                    "tests/test_pin.py": "import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 5\n"})
    model = FakeModel(_answers(*still_broken, fixed))
    run = tmp_path / "run"
    result = _design(run, model, repo, _mode(check_attempts=2))
    assert result.exit_ok, result.error
    outcomes = [(entry["outcome"], entry["attempt"]) for entry in _records(run)]
    assert outcomes == [("fixup", 1), ("fixup", 1), ("exhausted", 1), ("pass", 2)]
    assert result.facts["attempts_used"] == 1 and result.facts["total_self_check_fixups_used"] == 2
    # The reviews ran once, after the fourth emission; none ran during the fix-ups.
    first_review = next(i for i, prompt in enumerate(model.prompts) if HEALTH in prompt)
    emits = [i for i, prompt in enumerate(model.prompts) if EMIT in prompt]
    assert len(emits) == 4 and emits[-1] < first_review and _count(model, HEALTH) == 1
    # The fix-ups were amended into attempt 1; the early stop started attempt 2.
    assert _commits(run) == [f"{NAME}: attempt 2", f"{NAME}: attempt 1"]


def test_the_worst_case_ends_in_the_build_loop_stop_not_the_iteration_limit(tmp_path, monkeypatch):
    """Two refusals before every applied emission, and every applied emission fails its
    self-check: fix-ups, early stops and refusals all reloop prepare_check, and the build
    still ends in b39b's stop, past the old limit of 8 and under the new one. The pinned
    test's failure is `_Chunks`' answer, and the target is not watched (`_unwatched`): the
    loop is what is under test."""
    _Chunks(monkeypatch, fails={PIN})
    _unwatched(monkeypatch)
    repo = _target(tmp_path / "repo", PINNED)
    refused = "===EDIT: tool.py===\n<<<<<<< SEARCH\nnot in the file\n=======\nx\n>>>>>>> REPLACE\n"
    emits = [emission for n in range(9) for emission in (refused, refused,
                                                          render({"tool.py": f"VERSION = {10 + n}\n"}))]
    model = FakeModel(_answers(*emits))
    run = tmp_path / "run"
    result = _design(run, model, repo, _mode(check_attempts=3, self_check_fixups=2))
    # The check limit stops the build; its attempts applied, so it ends as a draft marked
    # check failing (b45a), not at a gate's iteration limit.
    assert result.exit_ok, result.error
    assert result.facts["draft_delivery"]["reason"] == "check attempts exhausted after 3 check failure(s)"
    assert result.facts["draft_delivery"]["check_failing"] is True
    assert [entry["outcome"] for entry in _records(run)] == ["fixup", "fixup", "exhausted"] * 3
    # Every emission was a prepare_check reloop but the last: 26 of them, against 45.
    assert _count(model, EMIT) == 27 and DesignMode().build().nodes["prepare_check"].max_iterations == 45


# ── the base: only the change's failures go back ──────────────────────────────


ENV_FAILURE = "tests/test_env.py::test_env"
ALWAYS_FAILS = "import tool\n\n\ndef test_env():\n    assert False, 'no network'\n"
LEFT_DESIGN = ("# A tool\n\n## Summary\n\nBump the tool.\n\n## Rule and sites\n\n"
               "- `tool.py:1` — **[left: the version stays an int]**\n\n## Docs\n\n- none\n\n"
               "## Changes\n\n- tool.py\n")


def test_round_one_sends_back_the_changes_failure_and_not_the_bases(tmp_path, monkeypatch):
    """On round 1 the failing tests run on the base worktree base_check prepares: the one
    the change broke goes back, the one the base fails too does not."""
    repo = _target(tmp_path / "repo", {
        **PINNED, "tests/test_env.py": ALWAYS_FAILS})
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500001")
    (run / "design.md").write_text(LEFT_DESIGN)
    _emit(run, render({"tool.py": "VERSION = 2\n"}), monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0, err
    assert out.strip().endswith("reloop") and "self-check fix-up 1 of 2" in out
    [record] = _records(run)
    assert record["failing_ids"] == [PIN] and record["inherited_filtered"] == [ENV_FAILURE]
    assert record["base"] == "compared with a run of 2 test(s) on the base"
    validation = _facts(run)["validation"]
    assert validation.startswith("self-check fix-up 1 of 2")
    assert f"- {PIN}" in validation and f"- {ENV_FAILURE}" not in validation
    assert "not this change's: " + ENV_FAILURE in validation
    # The design's constraints go back with it (R5), so a pinning test is not "fixed"
    # by reverting what the design changed.
    assert "[left: the version stays an int] — do not change this" in validation
    # When the base fails it too, the attempt goes on to the reviews. The fix-up changes
    # tool.py too, so its self-check runs test_env.py again, as an importer of what it changed.
    _emit(run, render({"tests/test_pin.py": "import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 2\n",
                       "tool.py": "VERSION = 2\nNAME = 'tool'\n"}), monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and out.strip() == "proceed", err
    assert _records(run)[-1]["outcome"] == "inherited"
    # What the base said was kept: the second comparison ran nothing on the base.
    assert _records(run)[-1]["base"] == "compared with earlier runs of these tests on the base"


def test_the_checks_own_base_result_answers_first(tmp_path, monkeypatch):
    """When the check has compared with the base already, its cached result answers, and
    the base is not prepared again. The chunks are `_Chunks`' answers."""
    from refactory_dsh import base_check

    _Chunks(monkeypatch, fails={PIN, ENV_FAILURE})
    repo = _target(tmp_path / "repo", {**PINNED, "tests/test_env.py": ALWAYS_FAILS})
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500011")
    _emit(run, render({"tool.py": "VERSION = 2\n"}), monkeypatch)
    found = worktree.recorded(run)
    (run / base_check.CACHE_FILE).write_text(json.dumps({
        "cache_key": base_check.cache_key(found.base, "", "", CHECK), "ok": False,
        "failures": [ENV_FAILURE], "output": "", "exit_code": 1, "environment_failure": None,
        "base_complete": True, "base_clean": True}))

    def prepared(ctx):
        raise AssertionError("the base was prepared again")

    monkeypatch.setattr(base_check, "_ensure_base_worktree", prepared)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and "self-check fix-up 1 of 2" in out, err
    [record] = _records(run)
    assert record["failing_ids"] == [PIN] and record["inherited_filtered"] == [ENV_FAILURE]
    assert record["base"] == "compared with the check's base result"


def test_a_failure_the_last_check_inherited_is_not_sent_back():
    from refactory_dsh.sandbox import SandboxResult

    output = f"FAILED {ENV_FAILURE} - AssertionError: no network\n1 failed in 0.10s"
    failed = SandboxResult(ok=False, returncode=1, stdout=output, stderr="", timed_out=False,
                           elapsed_s=0.1)
    prepass = failfast.targeted_prepass(
        {"status": "fail", "inherited": [ENV_FAILURE]}, CHECK, lambda command, timeout: failed,
        180, selection=["tests/test_env.py"], self_check_timeout_s=300)
    assert prepass.ran and prepass.ran_ids == ["tests/test_env.py"]
    assert not prepass.early_stop and prepass.failing_ids == []


def test_a_base_that_runs_out_of_time_lets_nothing_be_inherited(tmp_path, monkeypatch):
    """A base run stopped at its limit is unavailable: its failure goes back to the builder.
    The base run is given the self-check's limit and answered as the sandbox answers a run
    it stopped there (`test_a_self_check_that_says_nothing_goes_on_with_no_fixup` stops a
    real run at a real limit)."""
    from refactory_dsh.sandbox import SandboxResult

    repo = _target(tmp_path / "repo", {**PINNED, "tests/test_env.py": ALWAYS_FAILS})
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500002", self_check_timeout_s=10)
    _emit(run, render({"tool.py": "VERSION = 1\nMARK = 1\n"}), monkeypatch)
    limits: list[float] = []
    _base_runs(monkeypatch, SandboxResult(ok=False, returncode=124, timed_out=True, timeout_s=10,
                                          elapsed_s=10.0, stderr="timed out after 10s"), limits)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0, err
    assert "self-check fix-up 1 of 2" in out
    [record] = _records(run)
    assert record["failing_ids"] == [ENV_FAILURE] and record["inherited_filtered"] == []
    assert record["base"].startswith("unavailable, so nothing is inherited: the base run of 1 test(s) "
                                     "was stopped at its 10 s limit")
    assert limits == [10.0]


# ── the cut, and what does not count ──────────────────────────────────────────


def test_the_cut_keeps_a_finished_chunks_failure(tmp_path, monkeypatch):
    """A selection larger than the limit: the first chunk fails fast, the second is
    still running at the limit. The second is stopped and the rest never starts, and the
    first chunk's failure still goes back as a fix-up. The chunks are `_Chunks`' answers."""
    _Chunks(monkeypatch, fails={"tests/test_a.py::test_a"}, slow={"tests/test_b.py"})
    repo = _target(tmp_path / "repo", {
        "tool.py": "VERSION = 1\n",
        "tests/test_a.py": "import tool\n\n\ndef test_a():\n    assert tool.VERSION == 1\n",
        "tests/test_b.py": "import time\nimport tool\n\n\ndef test_b():\n    time.sleep(60)\n",
        "tests/test_c.py": "import tool\n\n\ndef test_c():\n    assert tool\n",
    })
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500003", self_check_timeout_s=10)
    _emit(run, render({"tool.py": "VERSION = 2\n"}), monkeypatch)
    _as_a_fixup(run)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0, err
    assert out.strip().endswith("reloop") and "self-check fix-up 2 of 2" in out
    [record] = _records(run)
    assert record["selection"] == ["tests/test_a.py", "tests/test_b.py", "tests/test_c.py"]
    assert record["ran"] == ["tests/test_a.py"]
    assert record["cut"] == ["tests/test_b.py", "tests/test_c.py"]
    assert record["failing_ids"] == ["tests/test_a.py::test_a"]
    assert record["commands"] == ["python -m pytest -q tests/test_a.py"]
    assert "chunk 2 of 3 was stopped at the 10 s limit" in " ".join(record["notes"])
    assert record["duration_s"] < 30


def test_a_chunk_runs_one_runner_so_a_failing_spec_cannot_hide_a_python_failure(tmp_path):
    """The targeted command keeps the check's ``&&``: a chunk holding a spec and a Python
    file would never run the Python file once vitest failed. Chunks hold one runner."""
    from refactory_dsh.sandbox import SandboxResult

    (tmp_path / "package.json").write_text('{"scripts": {"test": "vitest run"}}')
    selection = [*(f"tests/test_p{n}.py" for n in (1, 2, 3)), "web/tests/s.spec.ts",
                 *(f"tests/test_p{n}.py" for n in (4, 5, 6))]

    def run(command: str, timeout: float) -> SandboxResult:
        if command.startswith("pnpm test"):  # vitest fails before naming a test; && stops there
            return SandboxResult(ok=False, returncode=1, stdout="vitest could not start", stderr="",
                                 timed_out=False, elapsed_s=0.1)
        failing = [name for name in ("tests/test_p3.py", "tests/test_p5.py") if name in command]
        if failing:
            return SandboxResult(ok=False, returncode=1, stderr="", timed_out=False, elapsed_s=0.1,
                                 stdout="".join(f"FAILED {name}::test_x - AssertionError\n"
                                                for name in failing) + "1 failed in 0.10s")
        return SandboxResult(ok=True, returncode=0, stdout="2 passed in 0.10s", stderr="",
                             timed_out=False, elapsed_s=0.1)

    prepass = failfast.targeted_prepass({}, "pnpm test && python -m pytest -q tests", run, 180,
                                        root=str(tmp_path), selection=selection, self_check_timeout_s=300)
    # The failures before and after the spec's chunk both come back.
    assert prepass.failing_ids == ["tests/test_p3.py::test_x", "tests/test_p5.py::test_x"]
    assert all(("pnpm test" in command) != ("pytest" in command) for command in prepass.commands)
    assert "chunk 3 did not count" in " ".join(prepass.notes)
    # The spec's chunk reported nothing, so it is not counted as run.
    assert "web/tests/s.spec.ts" in prepass.cut_ids and "web/tests/s.spec.ts" not in prepass.ran_ids


def test_a_chunk_holds_at_most_four_tests():
    """About six chunks, but never more than four tests in one, so a chunk the limit
    stops loses little."""
    chunks = failfast._chunks([f"tests/test_{n}.py" for n in range(60)])
    assert [len(chunk) for chunk in chunks] == [4] * 15
    assert [len(chunk) for chunk in failfast._chunks([f"tests/test_{n}.py" for n in range(7)])] == [2, 2, 2, 1]


def test_a_self_check_that_says_nothing_goes_on_with_no_fixup(tmp_path, monkeypatch):
    """Its only chunk runs past the limit: inconclusive, so the attempt goes on to the
    reviews and the whole check, with nothing spent and nothing sent back.

    The file's one real limit: the sandbox stops a real pytest chunk at 3 s, the shortest
    limit the knobs allow (a third of check_timeout_s at its floor of 10). The other tests
    of the limit take `_Chunks`' answers."""
    repo = _target(tmp_path / "repo", {
        "tool.py": "VERSION = 1\n",
        "tests/test_slow.py": "import time\nimport tool\n\n\ndef test_slow():\n    time.sleep(60)\n",
    })
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500004", self_check_timeout_s=10,
                     check_timeout_s=10)
    _emit(run, render({"tool.py": "VERSION = 2\n"}), monkeypatch)
    _as_a_fixup(run)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and out.strip() == "proceed", err
    [record] = _records(run)
    assert record["outcome"] == "inconclusive" and record["ran"] == []
    assert record["cut"] == ["tests/test_slow.py"]
    # Stopped at the self-check's limit, not at the check's 10 s.
    assert record["limit_s"] == 3 and record["duration_s"] < 2 * record["limit_s"]
    assert (run / CHECK_PREP).exists()
    facts = _facts(run)
    assert not facts.get("self_check_fixup_count") and not facts.get("attempts_used")


def test_a_docs_only_emission_selects_nothing_and_runs_nothing(tmp_path, monkeypatch):
    repo = _target(tmp_path / "repo", {**PINNED, "README.md": "a tool\n"})
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500005")
    _emit(run, render({"README.md": "a tool, documented\n", "docs/guide.md": "use it\n"}), monkeypatch)
    ran = []
    monkeypatch.setattr(repo_change, "run_check", lambda *a, **k: ran.append(a) or None)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and out.strip() == "proceed", err
    [record] = _records(run)
    assert record["outcome"] == "nothing selected" and record["selection"] == []
    assert ran == []


# ── both runtimes: the spine runs each visit in its own process ───────────────


def test_on_the_spine_the_fixups_carry_across_processes_and_the_bound_spends_the_attempt(
        tmp_path, monkeypatch):
    """materialise and prepare_check as separate processes sharing facts.json: each
    failing self-check reloops to emit spending nothing, the count survives each process,
    each fix-up is amended into the attempt, and at the bound the attempt is spent. The
    pinned test's failure is `_Chunks`' answer."""
    _Chunks(monkeypatch, fails={PIN})
    repo = _target(tmp_path / "repo", PINNED)
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500006", check_attempts=2)
    for number in (1, 2):
        _emit(run, render({"tool.py": f"VERSION = {number + 1}\n"}), monkeypatch)
        code, out, err = _prepare(run, monkeypatch)
        assert code == 0, err
        reason, verdict = out.strip().rsplit("\n", 1)
        assert verdict == "reloop" and reason.startswith(f"self-check fix-up {number} of 2")
        facts = _facts(run)
        assert facts["self_check_fixup_count"] == number
        assert facts["total_self_check_fixups_used"] == number
        assert not facts.get("attempts_used") and not facts.get("check_attempts_used")
        assert _commits(run) == [f"{NAME}: attempt 1"]
    _emit(run, render({"tool.py": "VERSION = 9\n"}), monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0, err
    assert out.strip().endswith("reloop") and "self-check fix-up" not in out
    facts = _facts(run)
    assert facts["attempts_used"] == 1 and facts["check_attempts_used"] == 1
    assert facts["self_check_fixup_count"] == 0
    assert json.loads((run / "check-record.json").read_text())["check_mode"] == "targeted"
    # The attempt is spent, so the next emission is a commit of its own.
    _emit(run, render({"tool.py": "VERSION = 1\n"}), monkeypatch)
    assert _commits(run) == [f"{NAME}: attempt 2", f"{NAME}: attempt 1"]
    # The lowered graph sends prepare_check's reloop to the builder (through pick_corrections),
    # with room for the worst case.
    workflow = lowering.to_workflow(DesignMode().state_dict())
    assert ("prepare_check", "pick_corrections", "reloop") in {
        (e.source, e.target, e.condition) for e in workflow.edges}
    assert workflow.nodes["prepare_check"].max_iterations == 45


def test_the_run_cap_on_fixups_ends_them_for_the_run(tmp_path, monkeypatch):
    """Fix-ups are bounded per run too: self_check_fixups x check_attempts."""
    _Chunks(monkeypatch, fails={PIN})
    repo = _target(tmp_path / "repo", PINNED)
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500007", check_attempts=3)
    _emit(run, render({"tool.py": "VERSION = 2\n"}), monkeypatch)
    facts = _facts(run)
    (run / "facts.json").write_text(json.dumps({**facts, "total_self_check_fixups_used": 6}))
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0, err
    assert "self-check fix-up" not in out and _facts(run)["attempts_used"] == 1
    [record] = _records(run)
    assert record["outcome"] == "exhausted" and record["why"] == "6 of 6 fix-ups used on this run"


# ── the fix-up's commit: one amendment path ───────────────────────────────────


def test_a_fixup_is_amended_through_the_one_amend_path_with_the_attempts_guards(tmp_path, monkeypatch):
    repo = _target(tmp_path / "repo", PINNED)
    run = tmp_path / "run"
    worktree.create(repo, "HEAD", "abcdef500008", run)
    worktree.commit_attempt(run, {Path("tool.py"): "VERSION = 2\n"}, f"{NAME}: attempt 1")
    amended = []
    real = worktree.amend
    monkeypatch.setattr(worktree, "amend", lambda *a, **k: amended.append(a[2]) or real(*a, **k))
    worktree.commit_attempt(run, {Path("tests/test_pin.py"): "x = 1\n"}, "unused",
                            (Path("app.py"),), amending=True)
    assert amended == [["tests/test_pin.py"]]
    assert _commits(run) == [f"{NAME}: attempt 1"]
    found = worktree.recorded(run)
    assert git(Path(found.path), "rev-parse", "HEAD") == found.head
    assert git(Path(found.path), "ls-tree", "--name-only", "HEAD", "app.py") == ""
    # The attempt's guards hold for an amendment: no git state, no change-nothing amend.
    with pytest.raises(worktree.AttemptRefused, match="git's own state"):
        worktree.commit_attempt(run, {Path(".git/config"): "x"}, "unused", amending=True)
    with pytest.raises(worktree.AttemptRefused, match="changed nothing"):
        worktree.commit_attempt(run, {Path("tool.py"): "VERSION = 2\n"}, "unused", amending=True)
    assert amended == [["tests/test_pin.py"]]


def test_an_amendment_that_died_before_its_record_is_dropped(tmp_path):
    repo = _target(tmp_path / "repo", PINNED)
    run = tmp_path / "run"
    worktree.create(repo, "HEAD", "abcdef500009", run)
    found = worktree.commit_attempt(run, {Path("tool.py"): "VERSION = 2\n"}, f"{NAME}: attempt 1")
    path = Path(found.path)
    # An amendment that committed and died before recording the new head.
    (path / "tool.py").write_text("VERSION = 3\n")
    git(path, "commit", "-q", "-a", "--amend", "--no-edit")
    (run / worktree.PENDING).write_text(json.dumps({"files": ["tool.py"], "parent": found.head,
                                                    "amending": True}))
    worktree.commit_attempt(run, {Path("tool.py"): "VERSION = 4\n"}, "unused", amending=True)
    assert git(path, "show", "HEAD:tool.py") == "VERSION = 4"
    assert _commits(run) == [f"{NAME}: attempt 1"]


# ── design_fast and the sequential check keep the dm-128 pre-pass ─────────────


def test_design_fast_and_the_sequential_check_keep_the_dm128_prepass(tmp_path, monkeypatch):
    """design_fast's focused_check and check_artifact pass no selection, so they run the
    last check's failures only, from round 2, as before; neither writes a self-check."""
    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    calls = []
    real = failfast.targeted_prepass

    def spy(prev_record, command, run_fn, timeout, **kw):
        calls.append((prev_record.get("status"), kw))
        return real(prev_record, command, run_fn, timeout, **kw)

    monkeypatch.setattr(failfast, "targeted_prepass", spy)
    plan = "Change only the requested file and review the required check result."
    qa = json.dumps({"status": "pass", "checks": [{"name": "behaviour", "command": "inspection",
                                                   "status": "pass", "evidence": "it holds"}],
                     "findings": [], "not_run": []})
    bad = "===FILE: answer.py===\nassert False, 'test_bar'\n"
    good = "===FILE: answer.py===\nassert True\n"
    fast = DesignFastMode()
    state = fast.state_dict()
    state["knobs"]["repair_attempts"]["value"] = 2
    fast.load_state_dict(state)
    assert "self_check_fixups" not in state["knobs"] and "self_check_timeout_s" not in state["knobs"]
    fast.run(tmp_path / "fast", model=FakeModel([plan, bad, good, qa, VERIFICATION_ANSWER]),
             task={"name": "fast_tool", "brief": "assert true", "check": f"{sys.executable} answer.py"})
    assert calls and all("selection" not in kw for _, kw in calls)
    # Round 1 has no last failure to rerun; round 2 reruns round 1's.
    statuses = [status for status, _ in calls]
    assert statuses[0] != "fail" and statuses[1] == "fail"
    assert not (tmp_path / "fast" / SELF_CHECK_RECORD).exists()

    calls.clear()
    repo = _target(tmp_path / "repo", PINNED)
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500010")
    _emit(run, render({"tool.py": "VERSION = 2\n"}), monkeypatch)
    _node(run, f"{design_ops.__name__}:check_artifact", "check", "op", monkeypatch)
    assert calls and all("selection" not in kw for _, kw in calls)
    assert not (run / SELF_CHECK_RECORD).exists()


# ── review of #502: an undone attempt, collection errors, the cut's order ─────


UNDOES = render({"tool.py": "VERSION = 1\n"})
BREAKS = render({"tool.py": "VERSION = 2\n"})
PIN_FIXED = render({"tests/test_pin.py": "import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 2\n"})
GENERATED = {
    "gen.py": "import tool\nopen('generated.txt', 'w').write(f'version {tool.VERSION}\\n')\n",
    "generated.txt": "version 1\n",
}


def test_a_fixup_that_undoes_the_whole_attempt_is_refused_not_a_crash(tmp_path, monkeypatch):
    """The builder answers the pinned test by reverting its change: the amend would empty
    the attempt, which git refuses. It is a refused emission instead, and the run goes on.
    The self-checks are `_Chunks`' answers: the first applied emission fails the pinned
    test, the second passes."""
    _Chunks(monkeypatch, emissions=[{PIN}, set()])
    repo = _target(tmp_path / "repo", PINNED)
    # The builder works in a clone of the attempt, which keeps the refused round's revert
    # (`clones.prepare`: a head that did not move keeps the clone as the agent left it), so
    # its next round puts the change back and fixes the test, as the refusal asks.
    restored = render({"tool.py": "VERSION = 2\n",
                       "tests/test_pin.py": "import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 2\n"})
    model = FakeModel(_answers(BREAKS, UNDOES, restored))
    run = tmp_path / "run"
    result = _design(run, model, repo)
    assert result.exit_ok, result.error
    emits = [prompt for prompt in model.prompts if EMIT in prompt]
    assert len(emits) == 3 and "undoes the whole attempt" in emits[2]
    assert [entry["outcome"] for entry in _records(run)] == ["fixup", "pass"]
    assert _commits(run) == [f"{NAME}: attempt 1"]
    assert git(Path(worktree.recorded(run).path), "show", "HEAD:tool.py") == "VERSION = 2"


def test_an_empty_diff_attempt_is_sent_back_as_a_fixup(tmp_path, monkeypatch):
    """The fix-up reverts the source but not the file generate wrote from it; generate then
    writes that file back too, so the attempt's tree equals its parent's.  It is detected
    as an empty-diff and sent back as a fix-up, not forwarded to the reviews. The
    self-checks are `_Chunks`' answers: the first emission fails the pinned test, the last
    passes (the emptied one runs none)."""
    _Chunks(monkeypatch, emissions=[{PIN}, set()])
    repo = _target(tmp_path / "repo", {**PINNED, **GENERATED})
    # Three emissions: BREAKS triggers self-check fixup, UNDOES is the fix-up that
    # generate empties, FIXES_BOTH actually fixes both tool.py and the test.
    fixes_both = render({"tool.py": "VERSION = 2\n",
                         "tests/test_pin.py": "import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 2\n"})
    model = FakeModel(_answers(BREAKS, UNDOES, fixes_both))
    run = tmp_path / "run"
    result = _design(run, model, repo, _mode(self_check_fixups=3), generate="python gen.py")
    assert result.exit_ok, result.error
    outcomes = [entry["outcome"] for entry in _records(run)]
    assert outcomes == ["fixup", "empty_diff_fixup", "pass"]
    # The empty-diff fix-up routed back to emit (a third emission prompt exists).
    emits = [prompt for prompt in model.prompts if EMIT in prompt]
    assert len(emits) == 3 and "generate put back everything" in emits[2]
    # Fix-up count incremented.
    found = worktree.recorded(run)
    assert _commits(run) == [f"{NAME}: attempt 1"]


def test_an_empty_diff_on_second_attempt_is_detected(tmp_path, monkeypatch):
    """An emptied second attempt (parent = attempt 1) is also detected as an empty-diff
    fix-up, verifying the parent-tree comparison (<head>^{tree} vs <head>~1^{tree}). The
    self-checks are `_Chunks`' answers: attempt 1 passes, the break fails the pinned test,
    and the emptied fix-up would pass, were it run."""
    _Chunks(monkeypatch, emissions=[set(), {PIN}, set()])
    repo = _target(tmp_path / "repo", {**PINNED, **GENERATED})
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500050")
    facts = _facts(run)
    (run / "facts.json").write_text(json.dumps({**facts, "generate": "python gen.py"}))
    # First attempt: a clean emission that passes (adds NAME to tool.py).
    attempt1_tool = "VERSION = 1\nNAME = 'tool'\n"
    _emit(run, render({"tool.py": attempt1_tool}), monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and out.strip() == "proceed", err
    # Second attempt: emit breaks, self-check sends it back.
    _emit(run, BREAKS, monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and "self-check fix-up 1 of 2" in out, err
    # Fix-up: revert to attempt 1's tool.py, which generate then empties too.
    _emit(run, render({"tool.py": attempt1_tool}), monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and out.strip().endswith("reloop"), err
    assert "generate put back everything" in out
    records = _records(run)
    assert records[-1]["outcome"] == "empty_diff_fixup"


def test_on_the_spine_a_fixup_that_undoes_the_attempt_is_a_counted_refusal(tmp_path, monkeypatch):
    _Chunks(monkeypatch, fails={PIN})
    repo = _target(tmp_path / "repo", PINNED)
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500012")
    _emit(run, BREAKS, monkeypatch)
    assert "self-check fix-up 1 of 2" in _prepare(run, monkeypatch)[1]
    _emit(run, UNDOES, monkeypatch)
    facts = _facts(run)
    assert facts["consecutive_refusals"] == 1 and "undoes the whole attempt" in facts["emission_refused"]
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and out.strip().endswith("reloop") and "materialise could not apply" in out, err
    assert _facts(run)["self_check_fixup_count"] == 1
    found = worktree.recorded(run)
    assert _commits(run) == [f"{NAME}: attempt 1"] and worktree.tracked_edits(found) == []
    assert git(Path(found.path), "show", "HEAD:tool.py") == "VERSION = 2"


def test_a_module_that_does_not_import_goes_back_with_the_chunks_other_failures(tmp_path, monkeypatch):
    """pytest stops a chunk when a module does not import, so its other tests never ran:
    the module goes back as its file's failure, the rest of the chunk runs again, and a
    real failure there goes back too."""
    plain = "import tool\n\n\ndef test_{n}():\n    assert tool\n"
    files = {f"tests/test_{n}.py": plain.format(n=n) for n in range(1, 8)}
    files["tests/test_3.py"] = "import tool\nfrom tool import OLD\n\n\ndef test_3():\n    assert OLD\n"
    files["tests/test_4.py"] = "import tool\n\n\ndef test_4():\n    assert tool.VERSION == 1\n"
    repo = _target(tmp_path / "repo", {"tool.py": "VERSION = 1\nOLD = 1\n", **files})
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500013")
    _emit(run, BREAKS, monkeypatch)
    _as_a_fixup(run)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and "self-check fix-up 2 of 2" in out, err
    [record] = _records(run)
    assert record["failing_ids"] == ["tests/test_3.py", "tests/test_4.py::test_4"]
    assert sorted(record["ran"]) == sorted(files) and record["cut"] == []
    assert "tests/test_3.py did not import" in " ".join(record["notes"])


def test_a_slow_file_at_the_cut_does_not_hide_cheap_failures_after_it(tmp_path, monkeypatch):
    """The rest of the selection runs cheapest first, so the slow (big) file is the one the
    limit cuts, and the cheap failing file behind it in name order still reports. The chunks
    are `_Chunks`' answers."""
    _Chunks(monkeypatch, fails={"tests/test_e_fails.py::test_e"}, slow={"tests/test_a_slow.py"})
    slow = "import time\nimport m1\n" + "# padding\n" * 400 + "\n\ndef test_a():\n    time.sleep(60)\n"
    files = {"tests/test_a_slow.py": slow,
             **{f"tests/test_{c}.py": f"import m{n}\n\n\ndef test_{c}():\n    assert m{n}\n"
                for n, c in ((2, "b"), (3, "c"), (4, "d"))},
             "tests/test_e_fails.py": "import m5\n\n\ndef test_e():\n    assert m5.X == 1\n",
             **{f"m{n}.py": "X = 1\n" for n in range(1, 6)}}
    repo = _target(tmp_path / "repo", files)
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500014", self_check_timeout_s=10)
    _emit(run, render({f"m{n}.py": "X = 2\n" for n in range(1, 6)}), monkeypatch)
    _as_a_fixup(run)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and "self-check fix-up 2 of 2" in out, err
    [record] = _records(run)
    assert record["failing_ids"] == ["tests/test_e_fails.py::test_e"]
    assert record["selection"][-1] == "tests/test_a_slow.py" and record["cut"] == ["tests/test_a_slow.py"]


def test_the_builder_is_told_to_run_the_self_checks_tests_and_how_long_they_took(tmp_path):
    """In design runs A1 and A2 the builder ran the whole suite under its own 5-minute
    `timeout`, which killed it with no output, up to nine times a session; and a builder that
    stopped without running the tests its change broke was sent back to a cold session by the
    self-check. Its briefing, the first dispatch's too, names ``refactory-targeted-tests``,
    which runs the self-check's selection with the check's options kept (so `-n 4` stays);
    a fix-up's adds the seconds the self-check measured, and the first has no time to name."""
    repo = _target(tmp_path / "repo", PINNED)
    check = "python -m pytest -q -n 4 tests"
    breaks = render({"tool.py": "VERSION = 2\n"})
    fixes = render({"tests/test_pin.py": "import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 2\n"})
    model = FakeModel(_answers(breaks, fixes))
    run = tmp_path / "run"
    result = _design(run, model, repo, check=check)
    assert result.exit_ok, result.error

    first, _ = _records(run)
    [ran] = first["commands"]
    assert "-n 4" in ran and "tests/test_pin.py" in ran
    first_emit, second_emit = [prompt for prompt in model.prompts if EMIT in prompt]
    told = ("Before you stop, run `refactory-targeted-tests` and fix what fails. It runs exactly "
            "what the self-check will run for your change. ")
    assert f"{told}{design_ops.NO_BUILDER_TESTS}\n" in first_emit
    assert "none yet" not in first_emit and "tests took about" not in first_emit
    seconds = max(1, round(first["duration_s"]))
    assert f"{told}The last self-check's tests took about {seconds} s.\n" in second_emit
    assert "{{builder_tests}}" not in first_emit + second_emit


def test_the_builder_is_told_a_time_only_when_the_run_measured_one(tmp_path):
    """A self-check that ran everything states its elapsed time; one cut at its limit
    states the sum of its files' measured times, or that none was timed when a file has none
    or only a lower bound. It never guesses."""
    repo = _target(tmp_path / "repo", {**PINNED, "tests/test_other.py": "def test_other():\n    pass\n"})
    check = "python -m pytest -q -n 4 tests"
    names = ["tests/test_pin.py", "tests/test_other.py"]

    def prepass(**fields):
        return SimpleNamespace(**{"selection": names, "ran_ids": names, "cut_ids": [], "notes": [],
                                  "elapsed_s": 41.4, "timings": [], **fields})

    none = design_ops.NO_BUILDER_TESTS
    assert design_ops.builder_tests(check, prepass(), [], repo) == (
        "The last self-check's tests took about 41 s.")
    cut = prepass(ran_ids=names[:1], cut_ids=names[1:])
    assert design_ops.builder_tests(check, cut, [], repo) == none
    timed = [{"timings": [{"tests": [name], "s": 7.0, "counted": True}]} for name in names]
    assert "took about 14 s" in design_ops.builder_tests(check, cut, timed, repo)
    bounded = [{"timings": [{"tests": [names[0]], "s": 7.0, "counted": True},
                            {"tests": [names[1]], "s": 9.0, "counted": True, "timed_out": True}]}]
    assert design_ops.builder_tests(check, cut, bounded, repo) == none
    assert design_ops.builder_tests(check, prepass(selection=[]), [], repo) == none


def test_a_fixup_runs_the_last_self_checks_failures_first(tmp_path, monkeypatch):
    _Chunks(monkeypatch, fails={PIN})
    repo = _target(tmp_path / "repo", PINNED)
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500015")
    _emit(run, BREAKS, monkeypatch)
    _prepare(run, monkeypatch)
    _emit(run, render({"tool.py": "VERSION = 3\n"}), monkeypatch)
    _prepare(run, monkeypatch)
    first, second = _records(run)
    assert first["selection"][0] == "tests/test_pin.py" and first["how_selected"]["first"] == []
    assert second["selection"][0] == PIN and second["how_selected"]["first"] == [PIN]


def test_empty_diff_fixup_exhausts_the_budget(tmp_path, monkeypatch):
    """When the fix-up budget is already spent, an empty-diff attempt is exhausted. The
    self-checks are `_Chunks`' answers: the pinned test fails on the first emission and
    would pass on the emptied one, were it run."""
    _Chunks(monkeypatch, emissions=[{PIN}, set()])
    repo = _target(tmp_path / "repo", {**PINNED, **GENERATED})
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500051", self_check_fixups=1)
    facts = _facts(run)
    (run / "facts.json").write_text(json.dumps({**facts, "generate": "python gen.py"}))
    # First emission breaks the pin.
    _emit(run, BREAKS, monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and "self-check fix-up 1 of 1" in out, err
    # Fix-up: UNDOES, which generate empties. Budget is 1, already used 1.
    _emit(run, UNDOES, monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0, err
    records = _records(run)
    assert records[-1]["outcome"] == "exhausted"
    assert _facts(run)["attempts_used"] == 1


def test_a_timed_out_chunk_records_its_elapsed_time_as_a_lower_bound(tmp_path, monkeypatch):
    """A chunk stopped at the time limit records a timing entry with timed_out: True,
    its files appear in cut_ids, and the entry has counted: True. The chunks are
    `_Chunks`' answers, so the stopped chunk took the 10 s it was given on their clock."""
    _Chunks(monkeypatch, slow={"tests/test_b.py"})
    repo = _target(tmp_path / "repo", {
        "tool.py": "VERSION = 1\n",
        "tests/test_a.py": "import tool\n\n\ndef test_a():\n    assert tool\n",
        "tests/test_b.py": "import time\nimport tool\n\n\ndef test_b():\n    time.sleep(60)\n",
    })
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500052", self_check_timeout_s=10)
    _emit(run, render({"tool.py": "VERSION = 2\n"}), monkeypatch)
    _as_a_fixup(run)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and out.strip() == "proceed", err
    [record] = _records(run)
    assert record["outcome"] == "inconclusive"
    # Find the timed_out timing entry.
    timed_out_entries = [t for t in record["timings"] if t.get("timed_out")]
    assert len(timed_out_entries) >= 1
    entry = timed_out_entries[0]
    assert entry["s"] > 0 and entry["counted"] is True
    # The chunk's files are in cut.
    for test in entry["tests"]:
        assert test in record["cut"]


def test_costs_from_record_does_not_split_by_size_and_handles_timed_out(tmp_path):
    """costs_from_record uses per-file durations when available, and a non-timed_out
    entry always wins over a timed_out entry."""
    repo = _target(tmp_path / "repo", {
        "tests/test_big.py": "# padding\n" * 500 + "\ndef test_big():\n    pass\n",
        "tests/test_small.py": "\ndef test_small():\n    pass\n",
    })
    # (a) Two-file counted chunk with per-test durations: each gets its own time.
    entries_a = [{"timings": [{"tests": ["tests/test_big.py", "tests/test_small.py"],
                                "s": 10.0, "counted": True,
                                "durations": [
                                    {"test_id": "tests/test_big.py::test_big", "seconds": 8.0},
                                    {"test_id": "tests/test_small.py::test_small", "seconds": 2.0},
                                ]}]}]
    costs_a = test_selection.costs_from_record(entries_a, repo)
    assert costs_a["tests/test_big.py"] == 8.0
    assert costs_a["tests/test_small.py"] == 2.0

    # (b) A timed_out chunk at 120 s: its files get 120.0 when no other entry exists.
    entries_b = [{"timings": [{"tests": ["tests/test_big.py"], "s": 120.0,
                                "counted": True, "timed_out": True}]}]
    costs_b = test_selection.costs_from_record(entries_b, repo)
    assert costs_b["tests/test_big.py"] == 120.0

    # (c) A timed_out entry at 120 s followed by a non-timed_out entry at 5.0 with
    # durations: the non-timed_out wins (stays 5.0).
    entries_c = [{"timings": [{"tests": ["tests/test_big.py"], "s": 120.0,
                                "counted": True, "timed_out": True},
                               {"tests": ["tests/test_big.py"], "s": 5.0,
                                "counted": True,
                                "durations": [
                                    {"test_id": "tests/test_big.py::test_big", "seconds": 5.0},
                                ]}]}]
    costs_c = test_selection.costs_from_record(entries_c, repo)
    assert costs_c["tests/test_big.py"] == 5.0


def test_a_missing_test_id_is_dropped_and_the_rest_of_the_chunk_runs(tmp_path, monkeypatch):
    """pytest exits 4 on an id that is not there and runs nothing else in the command. The
    chunk's missing id is dropped and the rest of the chunk runs again: the rerun's failure
    is reported, its tests count as run, and a note says so. pytest names the missing id
    from the path's ``tests/…`` and without its ``[…]``, so the chunk's
    ``refactory/tests/…::test_gone[2.5]`` is matched by path suffix and bracket prefix."""
    repo = _target(tmp_path / "repo", {
        "tool.py": "VERSION = 1\n",
        "refactory/__init__.py": "",
        "refactory/tests/__init__.py": "",
        "refactory/tests/test_other.py": "def test_here():\n    pass\n",
        "refactory/tests/test_real.py": "import tool\n\n\ndef test_real():\n    assert tool.VERSION == 1\n",
        **{f"refactory/tests/test_f{n}.py": f"def test_f{n}():\n    pass\n" for n in range(5)},
    })
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500053")
    _emit(run, render({"tool.py": "VERSION = 2\n"}), monkeypatch)
    _as_a_fixup(run)
    gone, real = "refactory/tests/test_other.py::test_gone[2.5]", "refactory/tests/test_real.py"
    others = [f"refactory/tests/test_f{n}.py" for n in range(5)]
    real_select = test_selection.select

    def seven_tests(*args, **kwargs):
        selection = real_select(*args, **kwargs)
        selection.tests = [gone, real, *others]
        return selection

    monkeypatch.setattr(test_selection, "select", seven_tests)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0, err
    [record] = _records(run)
    # Seven tests make chunks of two: the missing id and test_real.py share the first, which
    # runs again without the id.
    assert record["commands"][:2] == [f"python -m pytest -q {real} {shlex.quote(gone)}",
                                      f"python -m pytest -q {real}"]
    assert record["failing_ids"] == [f"{real}::test_real"]
    assert real in record["ran"] and gone not in record["ran"] and set(others) <= set(record["ran"])
    assert [note for note in record["notes"] if "not-found" in note] == [
        "chunk 1: dropped not-found ids ['tests/test_other.py::test_gone'], reran the rest"]


def test_fixup_first_ids_do_not_duplicate_with_engine_or_touched_tier(tmp_path):
    """BUG-3: A parametrized first-id does not block its file from running whole later;
    a whole-file first-id deduplicates against the touched tier."""
    repo = _target(tmp_path / "repo", {
        "m.py": "X = 1\n",
        "tests/test_a.py": "import m\n\n\ndef test_x():\n    pass\n\ndef test_y():\n    pass\n",
    })
    files = test_selection.repository_files(repo)

    # Case (a): first=["tests/test_a.py::test_x"] with test_a.py touched.
    sel_a = test_selection.select(["m.py", "tests/test_a.py"], repo, {},
                                  check_command=CHECK, files=files,
                                  first=["tests/test_a.py::test_x"])
    # test_x runs first AND test_a.py still runs whole later.
    assert sel_a.tests[0] == "tests/test_a.py::test_x"
    assert "tests/test_a.py" in sel_a.tests
    # Both present: the parametrized id does not block the whole file.
    assert sel_a.tests.count("tests/test_a.py::test_x") == 1
    assert sel_a.tests.count("tests/test_a.py") == 1

    # Case (b): first=["tests/test_a.py"] with test_a.py touched.
    sel_b = test_selection.select(["m.py", "tests/test_a.py"], repo, {},
                                  check_command=CHECK, files=files,
                                  first=["tests/test_a.py"])
    # test_a.py appears exactly once (deduped).
    assert sel_b.tests.count("tests/test_a.py") == 1


def test_N10_timings_from_earlier_self_checks_flow_to_selection_ordering(tmp_path, monkeypatch):
    """The second self-check's selection orders files by their measured time, not by size.
    Kills mutant N10 (costs=test_selection.costs_from_record -> costs={}).

    Deterministic: after the first self-check, synthetic timings with durations are
    injected into the record, so the ordering does not depend on real elapsed time. The
    chunks are `_Chunks`' answers."""
    _Chunks(monkeypatch)
    # big_fast: large file (lots of padding) that runs instantly
    big_fast = "import m1\n" + "# pad\n" * 200 + "\ndef test_fast():\n    pass\n"
    # small_slow: tiny file that runs instantly too (no real sleep)
    small_slow = "import m2\n\ndef test_slow():\n    pass\n"
    repo = _target(tmp_path / "repo", {
        "m1.py": "X = 1\n", "m2.py": "X = 1\n",
        "tests/test_fast.py": big_fast,
        "tests/test_slow.py": small_slow,
    })
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500054")
    # First emission: both tests run and are timed.
    _emit(run, render({"m1.py": "X = 2\n", "m2.py": "X = 2\n"}), monkeypatch)
    _as_a_fixup(run)
    _prepare(run, monkeypatch)
    first_record = _records(run)[0]
    assert first_record["timings"]
    # Inject synthetic durations into the record: test_fast.py = 0.1 s, test_slow.py = 10.0 s
    record_path = run / SELF_CHECK_RECORD
    modified = dict(first_record)
    modified["timings"] = [
        {"tests": ["tests/test_fast.py", "tests/test_slow.py"],
         "s": 10.1, "counted": True,
         "durations": [
             {"test_id": "tests/test_fast.py::test_fast", "seconds": 0.1},
             {"test_id": "tests/test_slow.py::test_slow", "seconds": 10.0},
         ]}
    ]
    record_path.write_text(json.dumps(modified) + "\n")
    # Second emission: the selection should use the synthetic timings.
    _emit(run, render({"m1.py": "X = 3\n", "m2.py": "X = 3\n"}), monkeypatch)
    _as_a_fixup(run)
    _prepare(run, monkeypatch)
    second_record = _records(run)[-1]
    importers = [t for t in second_record["selection"]
                 if t in ("tests/test_fast.py", "tests/test_slow.py")]
    assert len(importers) == 2, f"expected both files in selection, got {importers}"
    assert importers == ["tests/test_fast.py", "tests/test_slow.py"], \
        f"expected timing-based order [fast, slow], got {importers}"


def test_N12_base_prepared_flag_after_base_worktree_removed_triggers_repreparation(
        tmp_path, monkeypatch):
    """When the base worktree directory is removed after preparation, the base is
    re-prepared on the next comparison.  Kills mutant N12: dropping ``_at(...)`` at
    base_check.py skips re-preparation, runs the test in an empty directory, and the
    base is unavailable, so the base's own failure would go back as the change's.

    The second emission introduces a new module and test so ``missing`` is non-empty:
    the ``if missing:`` block at base_check.py is entered and ``_at`` is checked. The
    chunks are `_Chunks`' answers; the base runs are real."""
    from refactory_dsh import base_check

    chunks = _Chunks(monkeypatch, fails={PIN, ENV_FAILURE})
    repo = _target(tmp_path / "repo", {
        **PINNED, "tests/test_env.py": ALWAYS_FAILS,
        "extra.py": "E = 1\n",
        "tests/test_extra.py": "import extra\n\n\ndef test_extra():\n    assert extra.E\n",
    })
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500055")
    # First emission: only changes tool.py; test_extra.py is not selected.
    _emit(run, BREAKS, monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and "self-check fix-up 1 of 2" in out, err
    # Remove the base worktree, as the check's base run does after a crash; the targeted
    # cache still says the base was prepared.
    found = worktree.recorded(run)
    run_id = Path(found.path).name.removeprefix("run-")
    base_path = base_check._base_worktree_path(Path(found.target), run_id)
    assert base_check._at(base_path, found.base)
    base_check.remove_base_worktree(Path(found.target), run_id)
    assert not base_path.exists()
    # Second emission: changes both tool.py and extra.py, bringing test_extra into
    # the selection; its test id has not been learned on the base, so ``missing`` is
    # non-empty and the ``if missing:`` block is entered.
    chunks.fails.add("tests/test_extra.py::test_extra")
    _emit(run, render({"tool.py": "VERSION = 2\nMARK = 1\n", "extra.py": "E = 0\n"}),
          monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0, err
    records = _records(run)
    assert len(records) == 2
    # The base was prepared again at its commit, the new test ran there, and the base's
    # failure is recognised as inherited, not sent back.
    assert base_check._at(base_path, found.base)
    assert records[1]["base"] == "compared with a run of 1 test(s) on the base"
    assert records[1]["inherited_filtered"] == ["tests/test_env.py::test_env"]
    assert records[1]["outcome"] == "fixup"
    assert "tests/test_extra.py::test_extra" in records[1]["failing_ids"]


def test_N17_collection_error_file_ids_in_base_guardedness_classification(tmp_path, monkeypatch):
    """A new test file that imports a module only the attempt introduces is classified
    as cannot_run_on_base by compute_test_guardedness via _parse_collection_errors.
    Driven through the whole check, as a node visit (check_artifact → _full_check →
    compute_test_guardedness → _parse_collection_errors).
    Kills mutant N17: replacing _parse_collection_errors with lambda _: (set(), False)
    changes the classification from cannot_run_on_base to fail."""
    repo = _target(tmp_path / "repo", {
        "tool.py": "VERSION = 1\n",
        "tests/test_pin.py": "import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 1\n",
    })
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500056")
    # Emit code that introduces a new module and a test that imports it.
    # The test passes on the attempt (module exists) but cannot be collected
    # on the base (module does not exist → ModuleNotFoundError).
    _emit(run, render({
        "new_module.py": "FEATURE = True\n",
        "tests/test_new_module.py": (
            "import new_module\n\n"
            "def test_feature():\n"
            "    assert new_module.FEATURE\n"
        ),
    }), monkeypatch)
    code, _, err = _node(run, f"{design_ops.__name__}:check_artifact", "check", "op", monkeypatch)
    assert code == 0, err
    guardedness = json.loads((run / "test-guardedness.json").read_text())
    # The new test file should be classified as cannot_run_on_base because
    # new_module does not exist on the base → collection error on import.
    found_results = [r for r in guardedness.get("results", [])
                     if r["file"] == "tests/test_new_module.py"]
    assert found_results, (
        f"tests/test_new_module.py not in guardedness results: {guardedness}")
    assert found_results[0].get("file_level") == "cannot_run_on_base", (
        f"expected cannot_run_on_base, got {found_results[0]}")


def test_timed_costs_order_the_rest(tmp_path):
    """A file an earlier self-check timed costs what it took; untimed files cost their size;
    a chunk that did not count times nothing."""
    repo = _target(tmp_path / "repo", {
        "m1.py": "X = 1\n", "m2.py": "X = 1\n",
        "tests/test_big_fast.py": "import m1\n" + "# padding\n" * 300 + "\n\ndef test_a():\n    pass\n",
        "tests/test_small_slow.py": "import m2\n\n\ndef test_b():\n    pass\n",
    })
    files = test_selection.repository_files(repo)
    by_size = test_selection.select(["m1.py", "m2.py"], repo, {}, check_command=CHECK, files=files)
    assert by_size.tests == ["tests/test_small_slow.py", "tests/test_big_fast.py"]
    earlier = [{"timings": [{"tests": ["tests/test_small_slow.py"], "s": 30.0, "counted": True,
                             "durations": [{"test_id": "tests/test_small_slow.py::test_b",
                                            "seconds": 30.0}]},
                            {"tests": ["tests/test_big_fast.py"], "s": 0.5, "counted": True},
                            {"tests": ["tests/test_big_fast.py"], "s": 99.0, "counted": False}]}]
    costs = test_selection.costs_from_record(earlier, repo)
    assert costs == {"tests/test_small_slow.py": 30.0, "tests/test_big_fast.py": 0.5}
    timed = test_selection.select(["m1.py", "m2.py"], repo, {}, check_command=CHECK, files=files,
                                  costs=costs)
    assert timed.tests == ["tests/test_big_fast.py", "tests/test_small_slow.py"]


def test_a_chunk_that_ran_only_some_ids_of_a_file_is_a_lower_bound_for_it(tmp_path):
    """A not-found rerun or a check of several steps writes no per-test seconds: a chunk of
    one file that ran only some of its tests by id says how long those took, not the file,
    so it is a lower bound, as it is with per-test seconds."""
    some = {"tests": ["tests/test_b.py::test_x"], "s": 0.4, "counted": True}
    stopped = {"tests": ["tests/test_b.py"], "s": 50.0, "counted": True, "timed_out": True}
    timed = {"tests": ["tests/test_b.py::test_x"], "s": 0.4, "counted": True,
             "durations": [{"test_id": "tests/test_b.py::test_x", "seconds": 0.3}]}
    for chunk in (some, timed):
        assert test_selection.lower_bounds([{"timings": [chunk]}]) == ["tests/test_b.py"]
        # A later, higher bound raises it; a finished time would not be.
        assert test_selection.costs_from_record([{"timings": [chunk, stopped]}], tmp_path) == {
            "tests/test_b.py": 50.0}


def test_the_full_check_still_runs_and_binds_after_a_self_check_cut_at_the_limit(
        tmp_path, monkeypatch):
    """A fix-up's self-check is stopped by its limit before the failing test: it says
    nothing, and the whole check beside the reviews runs for real and still finds the
    failure and sends the attempt back. Nothing a fix-up's partial run found is kept for
    the whole check to reuse, so it stays binding. The first self-check's whole check and
    the chunks are `_Chunks`' answers; the whole check beside the reviews is real."""
    _Chunks(monkeypatch, emissions=[{"tests/test_z_pin.py::test_z"}, set(), set()],
            slow={"tests/test_z_pin.py"})
    repo = _target(tmp_path / "repo", {
        "tool.py": "VERSION = 1\n",
        "tests/test_a_slow.py": "import tool\n\n\ndef test_a():\n    assert tool\n",
        "tests/test_z_pin.py": "import tool\n\n\ndef test_z():\n    assert tool.VERSION == 1\n",
    })
    # The fix-up changes tool.py again and leaves the pinned test broken.
    still_broken = render({"tool.py": "VERSION = 3\n"})
    fixed = render({"tests/test_z_pin.py": "import tool\n\n\ndef test_z():\n    assert tool.VERSION == 3\n"})
    model = FakeModel(_answers(BREAKS, still_broken, fixed))
    run = tmp_path / "run"
    result = _design(run, model, repo, _mode(self_check_timeout_s=10, check_timeout_s=60))
    assert result.exit_ok, result.error
    records = _records(run)
    # The first attempt now runs in chunks (like a fix-up): the slow file gets cut.
    # The self-check is inconclusive, so it proceeds to fork_qa where the full check
    # finds the failure and sends back.
    assert records[0]["outcome"] == "inconclusive"
    # The full check beside the reviews found the failure and sent back; the run
    # needed three attempts to succeed.  The self-check records are all
    # inconclusive (the slow file is always cut), but the run exits ok because
    # the third emission fixes the pinned test.  Three emissions prove the full
    # path: inconclusive self-check → fork_qa full check finds failure → send-back
    # → builder retries → eventual success.
    assert len(records) == 3, f"expected 3 self-check records, got {len(records)}: {[r['outcome'] for r in records]}"
    for i, rec in enumerate(records):
        assert rec["outcome"] == "inconclusive", (
            f"record {i} should be inconclusive (slow test always cut), got {rec['outcome']}")
    assert len([p for p in model.prompts if EMIT in p]) == 3, (
        "the run needed 3 emissions: inconclusive self-check → full check failure → send-back, twice, then success")


def _base_runs(monkeypatch, result=None, limits=None):
    """Count the base runs the self-check makes, and answer them with ``result`` if given.
    ``limits``, if given, collects the time limit each base run is given."""
    from refactory_dsh import base_check

    runs = []
    real = base_check.sandbox.run_command

    def run_command(command, *, workdir, **kw):
        if Path(workdir).name.startswith(base_check.BASE_WORKTREE_PREFIX):
            runs.append((command, kw.get("home")))
            if limits is not None:
                limits.append(kw.get("timeout_s"))
            if result is not None:
                return result
        return real(command, workdir=workdir, **kw)

    monkeypatch.setattr(base_check.sandbox, "run_command", run_command)
    return runs


def test_a_base_run_that_names_no_failure_is_unavailable(tmp_path, monkeypatch):
    from refactory_dsh import base_check
    from refactory_dsh.sandbox import SandboxResult

    _Chunks(monkeypatch, fails={ENV_FAILURE})
    repo = _target(tmp_path / "repo", {**PINNED, "tests/test_env.py": ALWAYS_FAILS})
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500016")
    _emit(run, render({"tool.py": "VERSION = 1\nMARK = 1\n"}), monkeypatch)
    _base_runs(monkeypatch, SandboxResult(ok=False, returncode=1, stdout="the base broke",
                                          stderr="", timed_out=False, elapsed_s=0.1))
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and "self-check fix-up 1 of 2" in out, err
    [record] = _records(run)
    assert record["failing_ids"] == [ENV_FAILURE] and record["inherited_filtered"] == []
    assert record["base"] == ("unavailable, so nothing is inherited: the base run failed (exit 1) "
                              "without naming a failing test")
    # Nothing is learned from it.
    assert json.loads((run / base_check.TARGETED_CACHE_FILE).read_text())["tests"] == {}


def test_a_parametrized_failure_the_parser_cuts_is_still_compared_with_the_base(tmp_path, monkeypatch):
    """A parameter with a space is cut from the failure's id (``test_param[a``), which pytest
    cannot find; passed back as it is, it stops the whole base run and nothing is
    inherited. Its file runs on the base instead, and its failure there is inherited."""
    param = ("import pytest\nimport tool\n\n\n@pytest.mark.parametrize('value', ['a b'])\n"
             "def test_param(value):\n    assert False\n")
    repo = _target(tmp_path / "repo", {**PINNED, "tests/test_param.py": param})
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500024")
    _emit(run, BREAKS, monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and "self-check fix-up 1 of 2" in out, err
    [record] = _records(run)
    assert record["failing_ids"] == [PIN]
    assert record["inherited_filtered"] == ["tests/test_param.py::test_param[a"]
    assert record["base"] == "compared with a run of 2 test(s) on the base"


def test_a_base_whose_generate_fails_is_unavailable(tmp_path, monkeypatch):
    _Chunks(monkeypatch, fails={ENV_FAILURE})
    repo = _target(tmp_path / "repo", {**PINNED, "tests/test_env.py": ALWAYS_FAILS})
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500017")
    facts = _facts(run)
    (run / "facts.json").write_text(json.dumps({**facts, "generate": "test -f attempt.flag"}))
    _emit(run, render({"tool.py": "VERSION = 1\nMARK = 1\n", "attempt.flag": "x\n"}), monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and "self-check fix-up 1 of 2" in out, err
    [record] = _records(run)
    assert record["failing_ids"] == [ENV_FAILURE] and record["inherited_filtered"] == []
    assert record["base"].startswith("unavailable, so nothing is inherited: the base's generate "
                                     "or setup failed")


def test_a_preparation_failure_during_a_fixup_ends_the_fixups(tmp_path, monkeypatch):
    """A fix-up whose emission breaks the generate step is judged by the whole check, which
    spends the attempt: the next emission is a new attempt, not an amendment."""
    _Chunks(monkeypatch, fails={PIN})
    repo = _target(tmp_path / "repo", {**PINNED, **GENERATED})
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500018")
    facts = _facts(run)
    (run / "facts.json").write_text(json.dumps({**facts, "generate": "python gen.py"}))
    _emit(run, BREAKS, monkeypatch)
    assert "self-check fix-up 1 of 2" in _prepare(run, monkeypatch)[1]
    _emit(run, render({"gen.py": "raise SystemExit('generate broke')\n"}), monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and out.strip().endswith("reloop"), err
    facts = _facts(run)
    assert facts["self_check_fixup_count"] == 0 and facts["attempts_used"] == 1
    _emit(run, render({"gen.py": GENERATED["gen.py"], "tool.py": "VERSION = 1\n"}), monkeypatch)
    assert _commits(run) == [f"{NAME}: attempt 2", f"{NAME}: attempt 1"]


def test_the_exhausted_early_stop_keeps_the_bases_failures_out(tmp_path, monkeypatch):
    _Chunks(monkeypatch, fails={PIN, ENV_FAILURE})
    repo = _target(tmp_path / "repo", {**PINNED, "tests/test_env.py": ALWAYS_FAILS})
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500019", self_check_fixups=0)
    _emit(run, BREAKS, monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and "self-check fix-up" not in out, err
    assert _records(run)[0]["outcome"] == "exhausted"
    record = json.loads((run / "check-record.json").read_text())
    assert record["inherited"] == [ENV_FAILURE]
    assert [f["test_id"] for runner in record["runners"] for f in runner["failures"]] == [PIN]


def test_what_was_learned_on_another_base_is_not_reused(tmp_path, monkeypatch):
    from refactory_dsh import base_check

    _Chunks(monkeypatch, fails={PIN})
    repo = _target(tmp_path / "repo", PINNED)
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500020")
    _emit(run, BREAKS, monkeypatch)
    stale = base_check.cache_key("0" * 40, "", "", CHECK)
    (run / base_check.TARGETED_CACHE_FILE).write_text(json.dumps(
        {"cache_key": stale, "prepared": True, "tests": {PIN: True}}))
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and "self-check fix-up 1 of 2" in out, err
    [record] = _records(run)
    assert record["failing_ids"] == [PIN] and record["inherited_filtered"] == []
    assert record["base"] == "compared with a run of 1 test(s) on the base"


def test_the_base_runs_with_its_own_home(tmp_path, monkeypatch):
    home_test = ("import os\nimport tool\n\n\ndef test_home():\n"
                 "    assert os.environ['HOME'].endswith('base-check-home')\n")
    repo = _target(tmp_path / "repo", {**PINNED, "tests/test_home.py": home_test})
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500021")
    _emit(run, render({"tool.py": "VERSION = 1\nMARK = 1\n"}), monkeypatch)
    runs = _base_runs(monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and "self-check fix-up 1 of 2" in out, err
    [record] = _records(run)
    # It fails in the attempt's HOME and passes in the base's: the change's, not inherited.
    assert record["failing_ids"] == ["tests/test_home.py::test_home"]
    assert record["inherited_filtered"] == []
    assert [Path(home).name for _, home in runs] == ["base-check-home"]


def test_the_base_is_prepared_once_for_its_key(tmp_path, monkeypatch):
    """A fix-up whose self-check needs the base again runs the new tests there without
    preparing the base a second time. The chunks are `_Chunks`' answers; the base runs are
    real."""
    from refactory_dsh import base_check

    chunks = _Chunks(monkeypatch, fails={PIN, ENV_FAILURE})
    other = "import tool\n\n\ndef test_other():\n    assert not hasattr(tool, 'BROKEN')\n"
    repo = _target(tmp_path / "repo", {**PINNED, "tests/test_env.py": ALWAYS_FAILS,
                                       "tests/test_other.py": other})
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500022")
    prepared = []
    real = base_check._ensure_base_worktree
    monkeypatch.setattr(base_check, "_ensure_base_worktree",
                        lambda ctx: prepared.append(1) or real(ctx))
    runs = _base_runs(monkeypatch)
    _emit(run, BREAKS, monkeypatch)
    assert "self-check fix-up 1 of 2" in _prepare(run, monkeypatch)[1]
    chunks.fails.add("tests/test_other.py::test_other")
    _emit(run, render({"tool.py": "VERSION = 2\nBROKEN = 1\n"}), monkeypatch)
    assert "self-check fix-up 2 of 2" in _prepare(run, monkeypatch)[1]
    first, second = _records(run)
    assert first["inherited_filtered"] == [ENV_FAILURE] and first["failing_ids"] == [PIN]
    assert second["failing_ids"] == [PIN, "tests/test_other.py::test_other"]
    assert len(runs) == 2 and prepared == [1]


def test_an_amendment_puts_the_worktree_back_first(tmp_path):
    """What a check or reviewer changed in the worktree is put back before a fix-up is
    amended in, so the amendment holds only the emission."""
    repo = _target(tmp_path / "repo", PINNED)
    run = tmp_path / "run"
    worktree.create(repo, "HEAD", "abcdef500023", run)
    found = worktree.commit_attempt(run, {Path("tool.py"): "VERSION = 2\n"}, f"{NAME}: attempt 1")
    path = Path(found.path)
    (path / "app.py").write_text("print('left by a reviewer')\n")
    found = worktree.commit_attempt(run, {Path("tests/test_pin.py"): "x = 2\n"}, "unused", amending=True)
    assert worktree.tracked_edits(found) == []
    assert (path / "app.py").read_text() == "print('app')\n"
    assert git(path, "show", "HEAD:app.py") == "print('app')"


def test_prepare_checks_limit_follows_the_knobs_to_their_bounds():
    from refactory_dsh.design import prepare_check_iterations

    assert prepare_check_iterations({}) == 45 == DesignMode().graph().nodes["prepare_check"].max_iterations
    widest = _mode(check_attempts=5, self_check_fixups=5)
    # (8 + 25 + 1) x 2 + 25 + 8 reloops at most, so 102.
    assert widest.graph().nodes["prepare_check"].max_iterations == 102
    state = widest.state_dict()
    assert state["graph"]["nodes"]["prepare_check"]["max_iterations"] == 102
    assert lowering.to_workflow(state).nodes["prepare_check"].max_iterations == 102
    capped = _mode(check_attempts=5, self_check_fixups=5, check_attempt_cap=20)
    assert capped.graph().nodes["prepare_check"].max_iterations == (20 + 25 + 1) * 2 + 25 + 20 + 1


def test_the_worst_case_at_the_knobs_bounds_still_ends_in_the_stop(tmp_path, monkeypatch):
    """Five attempts of five fix-ups each, two refusals before every applied emission: 89
    reloops of prepare_check, under its limit of 102 and twice the old fixed 45. The pinned
    test's failure is `_Chunks`' answer, and the target is not watched (`_unwatched`): the
    loop is what is under test."""
    _Chunks(monkeypatch, fails={PIN})
    _unwatched(monkeypatch)
    repo = _target(tmp_path / "repo", PINNED)
    refused = "===EDIT: tool.py===\n<<<<<<< SEARCH\nnot in the file\n=======\nx\n>>>>>>> REPLACE\n"
    emits = [emission for n in range(30) for emission in (refused, refused,
                                                          render({"tool.py": f"VERSION = {10 + n}\n"}))]
    model = FakeModel(_answers(*emits))
    run = tmp_path / "run"
    result = _design(run, model, repo, _mode(check_attempts=5, self_check_fixups=5))
    # The check limit stops the build; its attempts applied, so it ends as a draft marked
    # check failing (b45a), not at a gate's iteration limit.
    assert result.exit_ok, result.error
    assert result.facts["draft_delivery"]["reason"] == "check attempts exhausted after 5 check failure(s)"
    assert result.facts["draft_delivery"]["check_failing"] is True
    assert [entry["outcome"] for entry in _records(run)] == (["fixup"] * 5 + ["exhausted"]) * 5
    assert _count(model, EMIT) == 90


# ── the selection ─────────────────────────────────────────────────────────────


def _synthetic(root: Path) -> Path:
    """A repository whose tests mostly import two engine modules, with features on top, a
    structural test, a helper, a generated file, a spec, docs, and a test outside the
    check's scope."""
    def test(name: str, imports: str, pad: int = 0) -> tuple[str, str]:
        # ``pad`` makes a file bigger, so costlier to the selection's estimate.
        padding = f"# {'x' * pad}\n" if pad else ""
        return f"tests/test_{name}.py", f"{padding}{imports}\n\n\ndef test_{name}():\n    pass\n"

    files = dict([
        test("layout", "from pathlib import Path", 300),
        *(test(f"whole_{n}", "from pkg import core, engine") for n in (1, 2, 3)),
        test("engine_a", "from pkg import engine", 200),
        test("engine_b", "from pkg import engine"),
        test("core_a", "from pkg import core"),
        test("feature", "from pkg import engine, feature", 50),
        test("leaf", "from pkg import core, engine, leaf"),
        test("leaf_and_feature", "from pkg import engine, feature, leaf", 100),
        test("gone", "from pkg.gone import Z"),
        test("helped", "import helpers"),
    ])
    files.update({
        "pkg/__init__.py": "", "pkg/engine.py": "X = 1\n", "pkg/core.py": "C = 1\n",
        "pkg/feature.py": "F = 1\n", "pkg/leaf.py": "Y = 1\n", "pkg/gone.py": "Z = 1\n",
        "pkg/generated.py": "# Generated by gen.py. Do not edit.\nW = 1\n",
        "tests/helpers.py": "from pkg import leaf\n",
        "legacy/tests/test_old.py": "from pkg.gone import Z\n\n\ndef test_old():\n    assert Z\n",
        "web/src/a.ts": "export const a = 1\n",
        "web/tests/a.spec.ts": "import { a } from '../src/a'\n",
        "docs/guide.md": "a guide\n",
    })
    return _target(root, files)


def test_the_selection_order(tmp_path):
    repo = _synthetic(tmp_path / "repo")
    files = test_selection.repository_files(repo)
    check = "pnpm test && python -m pytest -q tests"
    last = {"status": "fail", "inherited": ["tests/test_feature.py::test_inherited"],
            "runners": [{"failures": [{"test_id": "tests/test_feature.py::test_feature"},
                                      {"test_id": "tests/test_feature.py::test_inherited"}]}]}

    def select(touched: list[str], prev: dict | None = None, listing: list[str] = files):
        return test_selection.select(touched, repo, prev or {}, check_command=check, files=listing)

    # A feature change: the last failures, then the importers and the structural test,
    # cheapest (smallest) first; nothing that cannot reach the change.
    leaf = select(["pkg/leaf.py", "pkg/feature.py"], last)
    assert leaf.hubs_changed == [] and leaf.engine == []
    assert leaf.tests == ["tests/test_feature.py::test_feature", "tests/test_leaf.py",
                          "tests/test_feature.py", "tests/test_leaf_and_feature.py",
                          "tests/test_layout.py"]

    # An engine change: the structural test and the tests on the whole engine (every hub)
    # first, then the tests the emission touched, then the rest of the importers, cheapest
    # first.
    engine = select(["pkg/engine.py", "tests/test_feature.py"])
    assert engine.hubs_changed == ["pkg/engine.py"]
    assert engine.engine == ["tests/test_layout.py", "tests/test_whole_1.py", "tests/test_whole_2.py",
                             "tests/test_whole_3.py", "tests/test_leaf.py"]
    assert engine.tests == [*engine.engine, "tests/test_feature.py", "tests/test_engine_b.py",
                            "tests/test_leaf_and_feature.py", "tests/test_engine_a.py"]

    # On a fix-up, what the last self-check found failing runs first of all.
    fixup = test_selection.select(["pkg/engine.py"], repo, {}, check_command=check, files=files,
                                  first=["tests/test_engine_a.py::test_engine_a"])
    assert fixup.tests[0] == "tests/test_engine_a.py::test_engine_a"

    # A deleted module's importers are found, and a helper's; a test the check does not
    # collect is not.
    gone = select(["pkg/gone.py", "tests/helpers.py"],
                  listing=[name for name in files if name != "pkg/gone.py"])
    assert gone.tests == ["tests/test_helped.py", "tests/test_gone.py", "tests/test_layout.py"]

    # A spec runs when it is touched, never because a source it imports is.
    assert select(["web/src/a.ts"]).tests == ["tests/test_layout.py"]
    assert select(["web/tests/a.spec.ts"]).tests == ["web/tests/a.spec.ts"]

    # Docs and generated files select nothing.
    assert select(["docs/guide.md", "pkg/generated.py"]).tests == []


#: The files b39a's and b39b's attempt 1 changed (their commits e48abd40a9 and 21b7aabe4).
B39A_TOUCHED = """packages/refactory/tool-refactory/src/generated-help.ts
packages/workflow/workflow-spine/tests/fixtures/design-graph.json refactory/docs/api/capabilities.md
refactory/docs/architecture.md refactory/docs/hardening.md refactory/docs/models.md
refactory/docs/tutorial.html refactory/refactory_dsh/create_ops.py refactory/refactory_dsh/data/coverage.json
refactory/refactory_dsh/data/documentation.json refactory/refactory_dsh/design.py
refactory/refactory_dsh/design_ops.py refactory/refactory_dsh/fast.py refactory/tests/test_design_scale.py
refactory/tests/test_design_short_path.py refactory/tests/test_interactive.py
refactory/tests/tests_support.py""".split()
B39B_TOUCHED = """packages/refactory/tool-refactory/src/generated-help.ts
packages/workflow/workflow-spine/src/execute.ts packages/workflow/workflow-spine/src/index.ts
packages/workflow/workflow-spine/tests/correction.spec.ts refactory/docs/api/capabilities.md
refactory/docs/hardening.md refactory/docs/tutorial.html refactory/refactory_dsh/create.py
refactory/refactory_dsh/create_ops.py refactory/refactory_dsh/data/coverage.json
refactory/refactory_dsh/data/documentation.json refactory/refactory_dsh/design.py
refactory/refactory_dsh/fast.py refactory/refactory_dsh/fast_ops.py refactory/refactory_dsh/nn/executor.py
refactory/refactory_dsh/repo_change.py refactory/refactory_dsh/run.py
refactory/tests/test_build_loop_stop.py""".split()
REPOSITORY_CHECK = "pnpm test && .venv/bin/python -m pytest -q refactory/tests"


def _tests(names: str) -> list[str]:
    return [f"refactory/tests/{name}.py" for name in names.split()]


#: The test suite's imports as b39's base had them (dad4af31), frozen: what each of the
#: check's test files imported. A hub is a share of the suite, so the live tree, which
#: grows, would move the fixture.
B39_IMPORTS = Path(__file__).parent / "fixtures" / "b39-base-imports.json"


def test_the_selection_catches_b39s_round_one_failures_with_the_engine_tests_first():
    """On b39a's and b39b's attempt-1 changes the selection holds a failing test for every
    round-1 failure, and the engine tests (the boundary, lowering smoke and mode
    validation) run first, found by import fan-in, not by name."""
    imports = {test: set(modules) for test, modules in json.loads(B39_IMPORTS.read_text()).items()}
    assert len(imports) == 131

    def select(touched: list[str]) -> test_selection.Selection:
        return test_selection.select(touched, REPO_ROOT, {}, check_command=REPOSITORY_CHECK,
                                     imports=imports)

    b39a, b39b = select(B39A_TOUCHED), select(B39B_TOUCHED)
    assert test_selection.hubs(imports) == {"refactory/refactory_dsh/nn/__init__.py",
                                            "refactory/refactory_dsh/registry.py",
                                            "refactory/refactory_dsh/design.py"}
    assert set(_tests("test_lowering_smoke test_edit_design test_pitfalls test_check_beside_reviews")) \
        <= set(b39a.tests)
    # Eight of b39b's nine; the package-lifecycle tests import no module it changed, and
    # failed only on the V7 its default-package and registry tests fail on too.
    assert set(_tests("test_boundary test_default_package test_design_probe test_edit_design "
                      "test_fast_modes test_registry test_lowering_smoke test_design_worktree")) \
        <= set(b39b.tests)
    for selection in (b39a, b39b):
        assert selection.hubs_changed == ["refactory/refactory_dsh/design.py"]
        engine = selection.engine
        assert selection.tests[:len(engine)] == engine
        # The structural tests (they import nothing of the repository), then the tests
        # on every hub: a few, not the 33 that import the design graph.
        assert engine == _tests("test_boundary test_boundary_refs test_docs_english_only "
                                "test_stale_install_finish test_work_complete test_lowering_smoke "
                                "test_walker_gate_core test_node_timeout test_verification "
                                "test_lowering test_edit_design test_design_worktree")
        # The tests the change touched come right after them.
        touched = [test for test in selection.touched_tests if test not in engine]
        assert selection.tests[len(engine):len(engine) + len(touched)] == touched


# ── the targeted command ──────────────────────────────────────────────────────


def test_whole_test_files_go_to_their_own_runner_only():
    py = "refactory/tests/test_x.py"
    spec = "packages/workflow/workflow-spine/tests/execute.spec.ts"
    # A Python-only selection never starts the vitest step.
    assert build_probe_command(REPOSITORY_CHECK, [], root=REPO_ROOT, test_files=[py]) == (
        f".venv/bin/python -m pytest -q {py}", None)
    # A spec goes to pnpm test as vitest's own file filter, with no "--" and no --related.
    command, refusal = build_probe_command(REPOSITORY_CHECK, [], root=REPO_ROOT, test_files=[spec])
    assert refusal is None and command == f"pnpm test {spec}"
    # An id inside a file that runs whole is not named twice; a vitest id's file runs whole.
    command, _ = build_probe_command(
        REPOSITORY_CHECK, [f"{py}::test_a", "refactory/tests/test_y.py::test_b",
                           f"{spec} > suite > case"], root=REPO_ROOT, test_files=[py, spec])
    assert command == (f"pnpm test {spec} && .venv/bin/python -m pytest -q {py} "
                       "refactory/tests/test_y.py::test_b")
    # Ids alone build what they always did.
    assert build_probe_command(REPOSITORY_CHECK, [f"{spec} > suite > case"], root=REPO_ROOT) == (
        f"pnpm test {spec} -t case", None)


def test_replay_measure5_structural_cuts_no_more_than_main(tmp_path, monkeypatch):
    """measure5-structural's self-checks 1-3 (dm-247), replayed through this branch's own
    selection, chunking and record: main cut 3, 2 and 4 files there; here self-check 2
    cuts at most 2 and self-check 3 at most 4.

    The fixture holds the run's records and what `test_selection.select` reads of the tree
    at each commit: select runs on a stand-in tree (each test file at its size) with the
    fixture's imports, and its tiers must be the run's. Each chunk takes a + b times its
    tests' junit seconds, fitted on self-check 1's recorded chunks, on `_Clocked`'s clock.
    Self-check 1 has no earlier timing, so it must come out as the run had it: the same
    selection, cut at the same chunk."""
    fixture = json.loads((Path(__file__).parent / "fixtures" / "measure5_replay.json").read_text())
    records, check = fixture["records"], fixture["check"]
    seconds = {**fixture["file_durations"], **fixture["id_durations"]}
    # What select reads of a test file's imports: its hubs and changed modules, and how
    # many it imports in all (the others stand in as modules no other file imports).
    imports = {test: {*modules, *(f"{test}#{n}" for n in range(count - len(modules)))}
               for test, (count, modules) in fixture["imports"].items()}
    xs = [sum(seconds[name] for name in chunk["tests"]) for chunk in records[0]["timings"]]
    ys = [chunk["s"] for chunk in records[0]["timings"]]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    b = (sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
         / sum((x - mean_x) ** 2 for x in xs))
    runner = _Clocked(seconds.__getitem__, monkeypatch, startup=mean_y - b * mean_x, scale=b)
    earlier: list[dict] = []
    selections, cuts = [], []
    for number, record in enumerate(records):
        root = tmp_path / f"tree{number}"
        for test, size in {**fixture["sizes"], **record["sizes"]}.items():
            (root / test).parent.mkdir(parents=True, exist_ok=True)
            (root / test).write_text("#" * (size - 1) + "\n")
        last = record["how_selected"]["last_failures"]
        prev = ({"status": "fail", "runners": [{"failures": [{"test_id": test_id} for test_id in last]}]}
                if last else {})
        selection = test_selection.select(
            record["touched"], root, prev, check_command=check, imports=imports,
            first=record["how_selected"]["first"],
            costs=test_selection.costs_from_record(earlier, root),
            bounds=test_selection.lower_bounds(earlier), alone=test_selection.run_alone(earlier))
        for tier in ("first", "engine", "hubs_changed", "touched_tests", "last_failures"):
            assert getattr(selection, tier) == record["how_selected"][tier], tier
        prepass = failfast.targeted_prepass(
            prev, check, runner, 1800, root=str(root), selection=selection.tests,
            self_check_timeout_s=fixture["limit_s"], alone=selection.alone)
        earlier.append({"timings": prepass.timings})
        selections.append(selection.tests)
        cuts.append(prepass.cut_ids)
    assert selections[0] == records[0]["selection"] and cuts[0] == records[0]["cut"]
    assert len(cuts[1]) <= 2, cuts[1]
    assert len(cuts[2]) <= 4, cuts[2]


class _Clocked:
    """A runner whose chunks take scripted seconds on a fake clock, as the sandbox runs them.

    A chunk takes ``startup`` plus ``scale`` times its tests' seconds. One that needs longer
    than it is given is stopped there and, its runner killed, writes no junit; one that
    finishes writes ``scale`` times each test's seconds to the self-check's ``--junitxml``.
    """

    def __init__(self, seconds, monkeypatch, *, startup: float = 0.0, scale: float = 1.0):
        self.seconds, self.startup, self.scale = seconds, startup, scale
        self.now = 0.0
        monkeypatch.setattr(failfast, "time", SimpleNamespace(monotonic=lambda: self.now))

    def __call__(self, command: str, timeout: float):
        from refactory_dsh.sandbox import SandboxResult

        words = shlex.split(command)
        names = [word for word in words if word.endswith(".py") or ".py::" in word]
        junit = next((word.split("=", 1)[1] for word in words if word.startswith("--junitxml=")),
                     None)
        took = self.startup + self.scale * sum(self.seconds(name) for name in names)
        if took > timeout:
            self.now += timeout
            return SandboxResult(ok=False, returncode=-9, timed_out=True, elapsed_s=timeout)
        self.now += took
        if junit:
            Path(junit).write_text("<testsuite>" + "".join(
                f'<testcase classname="{name.split("::")[0][:-3].replace("/", ".")}" '
                f'name="{name.split("::")[1] if "::" in name else "test_all"}" '
                f'time="{self.scale * self.seconds(name):.3f}"/>' for name in names) + "</testsuite>")
        return SandboxResult(ok=True, returncode=0, stdout=f"{len(names)} passed", elapsed_s=took)


def test_a_small_slow_file_stopped_beside_another_runs_alone_then_after_the_cheap_files(
        tmp_path, monkeypatch):
    """dm-247: a chunk stopped at the limit writes no junit, so a small but slow file stopped
    beside another has no time of its own and would run first again, cutting everything
    behind it. The stopped chunk's files run one per chunk on the next self-check, where the
    slow file gets a lower bound of its own; the self-check after that runs it after the
    cheap files, so it is the only file cut. It still runs alone there: it has only a bound.

    test_a_slow.py takes 60 s against a 20 s limit; every other file takes 0.5 s."""
    # Each test imports a module of its own, so no module is a hub; the change touches them all.
    sizes = {"a_slow": 0, "t1": 300, "t2": 300, **{c: 10 * n for n, c in enumerate("bcdef", start=4)}}
    files = {**{f"m_{name}.py": "X = 1\n" for name in sizes},
             **{f"tests/test_{name}.py": f"import m_{name}\n{'#' * size}\n\n\ndef test_{name}():\n"
                                         f"    assert m_{name}\n" for name, size in sizes.items()}}
    repo = _target(tmp_path / "repo", files)
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500070", self_check_timeout_s=20)
    runner = _Clocked(lambda name: 60.0 if "test_a_slow" in name else 0.5, monkeypatch)
    real = repo_change.run_check
    monkeypatch.setattr(repo_change, "run_check", lambda ctx, command, timeout, **kwargs: (
        runner(command, timeout) if "pytest" in command else real(ctx, command, timeout, **kwargs)))
    asked = []
    real_select = test_selection.select

    def select(*args, **kwargs):
        asked.append(kwargs)
        return real_select(*args, **kwargs)

    monkeypatch.setattr(test_selection, "select", select)
    for n in (2, 3, 4):  # three attempts, each touching two cheap tests and changing every module
        _emit(run, render({**{f"m_{name}.py": f"X = {n}\n" for name in sizes},
                           **{f"tests/test_{t}.py": files[f"tests/test_{t}.py"] + f"# {n}\n"
                              for t in ("t1", "t2")}}), monkeypatch)
        _as_a_fixup(run)
        code, out, err = _prepare(run, monkeypatch)
        assert code == 0 and out.strip() == "proceed", err
    first, second, third = _records(run)
    slow, beside = "tests/test_a_slow.py", "tests/test_b.py"
    cheap = [f"tests/test_{c}.py" for c in "cdef"]
    # 1: the slow file is the smallest, so it runs first of the rest, beside test_b.py. Their
    # chunk is stopped: no junit, so no time for either file.
    assert first["selection"] == ["tests/test_t1.py", "tests/test_t2.py", slow, beside, *cheap]
    assert [chunk["tests"] for chunk in first["timings"] if chunk.get("timed_out")] == [[slow, beside]]
    assert first["cut"] == [slow, beside, *cheap]
    assert test_selection.lower_bounds([first]) == []
    # 2: the two run one per chunk. The slow one, still first, is stopped alone: a bound.
    assert second["how_selected"]["alone"] == [slow, beside] and second["selection"] == first["selection"]
    assert [chunk["tests"] for chunk in second["timings"] if chunk.get("timed_out")] == [[slow]]
    assert test_selection.lower_bounds([first, second]) == [slow]
    # 3: it runs after the cheap files, alone, and only it is cut.
    assert third["selection"] == ["tests/test_t1.py", "tests/test_t2.py", beside, *cheap, slow]
    assert third["how_selected"]["alone"] == [slow, beside]
    assert third["cut"] == [slow] and third["ran"] == third["selection"][:-1]
    # prepare_check hands select what the run's record says: its bounds and its lone files.
    assert [(kwargs["bounds"], kwargs["alone"]) for kwargs in asked] == [
        ([], []), ([], [slow, beside]), ([slow], [slow, beside])]


def test_a_file_with_only_a_bound_runs_last_while_nothing_in_its_tier_has_finished(
        tmp_path, monkeypatch):
    """dm-247 when no file of the cheapest-first tier has finished: with no measured rate
    the others cost their size in bytes, which a bound in seconds cannot be weighed
    against, so the file that has only a bound runs after them instead of first.

    Ten files, each importing a module of its own (no hub), none touched, a 10 s limit:
    the smallest takes 60 s, the others 0.5 s. Self-check 1 stops it beside test_f1.py,
    self-check 2 stops it alone (a 10 s bound), and self-check 3 cuts only it."""
    names = ["a_slow", *(f"f{n}" for n in range(1, 10))]
    files = {**{f"m_{name}.py": "X = 1\n" for name in names},
             **{f"tests/test_{name}.py": f"import m_{name}\n{'#' * 20 * n}\n\n\ndef test_{name}():\n"
                                         f"    assert m_{name}\n" for n, name in enumerate(names)}}
    repo = _target(tmp_path / "repo", files)
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500071", self_check_timeout_s=10)
    runner = _Clocked(lambda name: 60.0 if "test_a_slow" in name else 0.5, monkeypatch)
    real = repo_change.run_check
    monkeypatch.setattr(repo_change, "run_check", lambda ctx, command, timeout, **kwargs: (
        runner(command, timeout) if "pytest" in command else real(ctx, command, timeout, **kwargs)))
    for n in (2, 3, 4):
        _emit(run, render({f"m_{name}.py": f"X = {n}\n" for name in names}), monkeypatch)
        _as_a_fixup(run)
        code, out, err = _prepare(run, monkeypatch)
        assert code == 0 and out.strip() == "proceed", err
    first, second, third = _records(run)
    slow, others = "tests/test_a_slow.py", [f"tests/test_{name}.py" for name in names[1:]]
    assert first["selection"] == second["selection"] == [slow, *others]
    assert first["cut"] == second["cut"] == [slow, *others]
    assert test_selection.lower_bounds([first, second]) == [slow]
    assert not test_selection.costs_from_record([first, second], tmp_path).keys() - {slow}
    assert third["selection"] == [*others, slow] and third["cut"] == [slow]


def test_run_alone_holds_a_stopped_chunks_files_until_a_chunk_finishes_each():
    x, y, z = "tests/test_x.py", "tests/test_y.py", "tests/test_z.py"
    stopped = {"timings": [{"tests": [x, y], "s": 9.0, "timed_out": True, "counted": True}]}
    assert test_selection.run_alone([stopped]) == [x, y]
    # A finished time frees y; x, stopped again alone, has only a bound and stays.
    second = {"timings": [{"tests": [y], "s": 1.0, "counted": True},
                          {"tests": [x], "s": 5.0, "timed_out": True, "counted": True}]}
    assert test_selection.run_alone([stopped, second]) == [x]
    # Some of x's tests by id are no time for x, and a chunk that did not count says nothing.
    by_id = {"timings": [{"tests": [f"{x}::test_a"], "s": 0.5, "counted": True},
                         {"tests": [x], "s": 0.5, "counted": False}]}
    assert test_selection.run_alone([stopped, second, by_id]) == [x]
    # Per-test seconds for x, run whole beside another file, are.
    together = {"timings": [{"tests": [x, z], "s": 2.0, "counted": True, "durations": [
        {"test_id": f"{x}::test_a", "seconds": 1.0}, {"test_id": f"{z}::test_b", "seconds": 0.5}]}]}
    assert test_selection.run_alone([stopped, second, by_id, together]) == []
    # A file stopped alone runs alone next time too.
    assert test_selection.run_alone([{"timings": [
        {"tests": [z], "s": 3.0, "timed_out": True, "counted": True}]}]) == [z]


def test_a_file_that_runs_alone_shares_its_chunk_with_no_other_file():
    names = [f"tests/test_{n}.py" for n in range(10)]
    selection = [*names[:3], f"{names[3]}::test_a", f"{names[3]}::test_b", *names[4:]]
    chunks = failfast._chunks(selection, alone=[names[3], names[6]])
    assert chunks == [names[:2], [names[2]], [f"{names[3]}::test_a", f"{names[3]}::test_b"],
                      names[4:6], [names[6]], names[7:9], [names[9]]]
    assert failfast._chunks(selection) == [selection[n:n + 2] for n in range(0, 11, 2)]


def test_a_lower_bound_never_lowers_a_size_estimate_and_sets_no_rate(tmp_path):
    """A file stopped near the limit has a small bound: it costs its size estimate when that
    is higher. A bound is no rate a file ran at, so a small file's 30 s bound does not make
    every untimed file dear."""
    repo = _target(tmp_path / "repo", {
        "tests/test_done.py": "#" * 999 + "\n", "tests/test_big.py": "#" * 999 + "\n",
        "tests/test_tiny_slow.py": "#" * 9 + "\n", "tests/test_untimed.py": "#" * 999 + "\n"})
    tests = ["tests/test_done.py", "tests/test_big.py", "tests/test_tiny_slow.py",
             "tests/test_untimed.py"]
    known = {"tests/test_done.py": 1.0, "tests/test_big.py": 0.2, "tests/test_tiny_slow.py": 30.0}
    costs = test_selection.estimated_costs(tests, repo, known,
                                           ["tests/test_big.py", "tests/test_tiny_slow.py"])
    assert costs == pytest.approx({"tests/test_done.py": 1.0, "tests/test_big.py": 1.0,
                                   "tests/test_tiny_slow.py": 30.0, "tests/test_untimed.py": 1.0})
    # With no finished file there is no rate: sizes stay in bytes, and the bounds go last.
    unmeasured = test_selection.estimated_costs(tests[1:], repo, known,
                                                ["tests/test_big.py", "tests/test_tiny_slow.py"])
    assert sorted(unmeasured, key=unmeasured.get) == [
        "tests/test_untimed.py", "tests/test_big.py", "tests/test_tiny_slow.py"]


def test_costs_from_record_multi_file_chunk_without_durations_keeps_estimates(tmp_path):
    """A finished multi-file chunk without ``durations`` (e.g. a vitest chunk) leaves
    each file's existing estimate unchanged.  A single-file chunk without durations
    assigns its elapsed time to that file."""
    repo = _target(tmp_path / "repo", {
        "tests/test_a.py": "\ndef test_a():\n    pass\n",
        "tests/test_b.py": "\ndef test_b():\n    pass\n",
    })
    # Pre-existing size estimate for test_b.
    initial = [{"timings": [{"tests": ["tests/test_b.py"], "s": 0.1, "counted": True}]}]
    costs_before = test_selection.costs_from_record(initial, repo)
    assert costs_before["tests/test_b.py"] == 0.1

    # A two-file chunk at 60 s without durations: both files keep their estimates.
    entries = [{"timings": [{"tests": ["tests/test_b.py"], "s": 0.1, "counted": True},
                            {"tests": ["tests/test_a.py", "tests/test_b.py"],
                             "s": 60.0, "counted": True}]}]
    costs = test_selection.costs_from_record(entries, repo)
    assert costs["tests/test_b.py"] == 0.1, (
        f"multi-file chunk without durations changed test_b to {costs['tests/test_b.py']}")
    assert "tests/test_a.py" not in costs, (
        f"test_a should not have a cost from a multi-file chunk without durations")

    # A single-file chunk at 60 s without durations: the file gets 60.0.
    entries_single = [{"timings": [{"tests": ["tests/test_a.py"], "s": 60.0, "counted": True}]}]
    costs_single = test_selection.costs_from_record(entries_single, repo)
    assert costs_single["tests/test_a.py"] == 60.0


def test_costs_from_record_lower_bound_never_overwrites_finished_time(tmp_path):
    """A lower bound never lowers a finished time; max is always used."""
    repo = _target(tmp_path / "repo", {
        "tests/test_x.py": "\ndef test_x():\n    pass\n",
    })
    # (a) Non-timed-out at 5.0 s, then timed-out at 120 s: stays 5.0.
    entries_a = [{"timings": [
        {"tests": ["tests/test_x.py"], "s": 5.0, "counted": True,
         "durations": [{"test_id": "tests/test_x.py::test_x", "seconds": 5.0}]},
        {"tests": ["tests/test_x.py"], "s": 120.0, "counted": True, "timed_out": True},
    ]}]
    costs_a = test_selection.costs_from_record(entries_a, repo)
    assert costs_a["tests/test_x.py"] == 5.0

    # (b) Two timed-out entries: 120 s then 80 s → max = 120.
    entries_b = [{"timings": [
        {"tests": ["tests/test_x.py"], "s": 120.0, "counted": True, "timed_out": True},
        {"tests": ["tests/test_x.py"], "s": 80.0, "counted": True, "timed_out": True},
    ]}]
    costs_b = test_selection.costs_from_record(entries_b, repo)
    assert costs_b["tests/test_x.py"] == 120.0

    # (c) Two non-timed-out entries: 10 s then 5 s → max = 10.
    entries_c = [{"timings": [
        {"tests": ["tests/test_x.py"], "s": 10.0, "counted": True,
         "durations": [{"test_id": "tests/test_x.py::test_x", "seconds": 10.0}]},
        {"tests": ["tests/test_x.py"], "s": 5.0, "counted": True,
         "durations": [{"test_id": "tests/test_x.py::test_x", "seconds": 5.0}]},
    ]}]
    costs_c = test_selection.costs_from_record(entries_c, repo)
    assert costs_c["tests/test_x.py"] == 10.0


def test_a_real_pytest_chunk_records_its_tests_seconds_from_junit_outside_the_worktree(
        tmp_path, monkeypatch):
    """The self-check adds ``--junitxml`` to a chunk's pytest command, so the record holds
    each test's seconds. The file is written outside the worktree, never inside it even for
    a moment, and removed afterwards. A check of several steps gets no ``--junitxml``: it
    would reach the last step, which need not be pytest."""
    from refactory_dsh.sandbox import SandboxResult

    repo = _target(tmp_path / "repo", PINNED)
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500060")
    sent = []
    real = repo_change.run_check

    def recording(ctx, command, timeout, **kwargs):
        sent.append(command)
        return real(ctx, command, timeout, **kwargs)

    monkeypatch.setattr(repo_change, "run_check", recording)
    _emit(run, render({"tool.py": "VERSION = 1\nNAME = 'tool'\n"}), monkeypatch)
    _as_a_fixup(run)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and out.strip() == "proceed", err
    [record] = _records(run)
    [timing] = record["timings"]
    assert [entry["test_id"] for entry in timing["durations"]] == [PIN]
    assert timing["durations"][0]["seconds"] >= 0
    [command] = [command for command in sent if "pytest" in command]
    assert command.startswith(record["commands"][0] + " --junitxml=")
    junit = Path(command.split("--junitxml=", 1)[1])
    found = worktree.recorded(run)
    assert junit.is_absolute() and not junit.exists()
    assert not junit.resolve().is_relative_to(Path(found.path).resolve())
    assert not git(Path(found.path), "status", "--porcelain").strip()

    several = []

    def run_fn(command, timeout):
        several.append(command)
        return SandboxResult(ok=True, returncode=0, stdout="1 passed", elapsed_s=0.1)

    failfast.targeted_prepass({}, "python -m pytest -q tests && python tools/lint.py", run_fn, 180,
                              root=str(repo), selection=["tests/test_pin.py"], self_check_timeout_s=60)
    assert several == ["python -m pytest -q tests/test_pin.py && python tools/lint.py"]


def test_two_consecutive_empty_diff_fixups_with_fixups_1_exhausts(tmp_path, monkeypatch):
    """Two emptied attempts in a row with ``self_check_fixups=1`` must end exhausted.

    The first emission changes ``generated.txt``; generate puts it back →
    ``empty_diff_fixup`` (count goes 0→1).  The second emission does the same;
    generate restores it again, but now count=1 ≥ allowed=1 → ``exhausted``.

    Remove line 1293 (``ctx.set_fact("self_check_fixup_count", used + 1)``)
    and the second empty-diff sees used=0 < allowed=1 → fixups again instead
    of exhausting → test fails."""
    repo = _target(tmp_path / "repo", {**PINNED, **GENERATED})
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500061", self_check_fixups=1)
    facts = _facts(run)
    (run / "facts.json").write_text(json.dumps({**facts, "generate": "python gen.py"}))

    # First emission: changes generated.txt; generate puts it back → empty_diff_fixup.
    _emit(run, render({"generated.txt": "version 99\n"}), monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0, err
    records = _records(run)
    assert records[-1]["outcome"] == "empty_diff_fixup"
    assert _facts(run)["self_check_fixup_count"] == 1
    assert _facts(run)["total_self_check_fixups_used"] == 1

    # Second emission: same change; generate restores again → exhausted.
    _emit(run, render({"generated.txt": "version 99\n"}), monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0, err
    records = _records(run)
    assert records[-1]["outcome"] == "exhausted"


def test_parse_junitxml_handles_class_based_tests(tmp_path):
    """``_parse_junitxml`` correctly maps class-based test classnames to file paths.

    For ``tests.test_failfast.TestNotFoundIds::test_method``, the classname is
    ``tests.test_failfast.TestNotFoundIds``; the file path must be
    ``tests/test_failfast.py``, not ``tests/test_failfast/TestNotFoundIds.py``."""
    xml_content = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest">
  <testcase classname="tests.test_failfast.TestNotFoundIds" name="test_dropped" time="0.123"/>
  <testcase classname="tests.test_failfast.TestNotFoundIds" name="test_suffix" time="0.456"/>
  <testcase classname="tests.test_failfast" name="test_module_level" time="0.789"/>
</testsuite>
"""
    xml_path = tmp_path / "junit.xml"
    xml_path.write_text(xml_content)
    durations = failfast._parse_junitxml(str(xml_path))
    assert durations is not None
    by_id = {d["test_id"]: d["seconds"] for d in durations}
    # Class-based tests: classname "tests.test_failfast.TestNotFoundIds" → file "tests/test_failfast.py"
    assert "tests/test_failfast.py::test_dropped" in by_id
    assert "tests/test_failfast.py::test_suffix" in by_id
    assert by_id["tests/test_failfast.py::test_dropped"] == pytest.approx(0.123)
    assert by_id["tests/test_failfast.py::test_suffix"] == pytest.approx(0.456)
    # Module-level test: classname "tests.test_failfast" → file "tests/test_failfast.py"
    assert "tests/test_failfast.py::test_module_level" in by_id
    assert by_id["tests/test_failfast.py::test_module_level"] == pytest.approx(0.789)
    # Verify NO wrong path like "tests/test_failfast/TestNotFoundIds.py"
    wrong_paths = [d["test_id"] for d in durations if "TestNotFoundIds" in d["test_id"]]
    assert not wrong_paths, f"Class name leaked into test_id: {wrong_paths}"


def test_parse_junitxml_keeps_a_capitalised_package_directory(tmp_path):
    """Only trailing class names leave the path, so a capitalised package directory stays
    (``src/MyPkg/tests/test_x.py``, not ``src.py``)."""
    xml_path = tmp_path / "junit.xml"
    xml_path.write_text(
        '<testsuite>'
        '<testcase classname="src.MyPkg.tests.test_x" name="test_a" time="1.5"/>'
        '<testcase classname="src.MyPkg.tests.test_x.TestK.TestInner" name="test_b" time="2"/>'
        '<testcase classname="tests.test_y" name="test_c" time="3"/>'
        '</testsuite>')
    ids = [entry["test_id"] for entry in failfast._parse_junitxml(str(xml_path))]
    assert ids == ["src/MyPkg/tests/test_x.py::test_a", "src/MyPkg/tests/test_x.py::test_b",
                   "tests/test_y.py::test_c"]


def test_empty_diff_check_skipped_when_head_equals_base(tmp_path, monkeypatch):
    """When head == base (no emission committed), the empty-diff check is skipped.

    The base has a parent with the same tree (via ``--allow-empty``).  Without
    the ``found.head != found.base`` guard, ``tree(head) == tree(head~1)``
    would falsely trigger ``empty_diff_fixup``.  With the guard, head == base
    so the comparison is skipped and the normal flow (``nothing selected``)
    proceeds.  Removing the guard fails this test."""
    repo = _target(tmp_path / "repo", PINNED)
    # Add an empty commit so tree(HEAD) == tree(HEAD~1).
    git(repo, "commit", "--allow-empty", "-m", "empty")
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500063")
    found = worktree.recorded(run)
    assert found.head == found.base  # no emission committed
    # Set "commit" fact so prepare_attempt's _restored_attempt does not crash.
    facts = _facts(run)
    facts["commit"] = found.head
    (run / "facts.json").write_text(json.dumps(facts))
    # Call prepare_check without emitting: prepare_attempt returns []
    # (nothing to prepare), code falls through to the empty-diff check.
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0, err
    records = _records(run)
    assert records[-1]["outcome"] not in ("empty_diff_fixup", "exhausted")


def test_a_not_found_chunk_with_no_time_to_rerun_is_recorded(tmp_path, monkeypatch):
    """A chunk whose ids a rename removed collects nothing and cannot rerun: it is recorded
    as having verified nothing (not a pass), and no second run is attempted."""
    import time as _time

    from refactory_dsh.sandbox import SandboxResult

    exit4 = SandboxResult(ok=False, returncode=4,
                          stdout="ERROR: not found: /abs/path/tests/test_x.py::test_gone\n",
                          stderr="", timed_out=False, elapsed_s=9.5)
    monkeypatch.setattr(failfast, "SELF_CHECK_CHUNKS", 1)

    calls = []
    elapsed = [0.0]

    def mock_monotonic():
        return 1000.0 + elapsed[0]

    monkeypatch.setattr(_time, "monotonic", mock_monotonic)

    def run_fn(cmd, timeout):
        calls.append(cmd)
        elapsed[0] = 9.5
        return exit4

    result = failfast.targeted_prepass(
        {"status": "fail"}, "pytest -q tests", run_fn, 10,
        selection=["tests/test_x.py::test_gone", "tests/test_x.py::test_here"],
        self_check_timeout_s=10)
    assert len(calls) == 1, "no rerun is attempted, so the chunk's tests never run"
    assert result.ran_nothing, "the self-check says it verified nothing"
    assert not result.broken_command, "a renamed id is not a broken command"

def test_rerun_handles_module_import_error(tmp_path, monkeypatch):
    """When the rerun's output contains ``_INTERRUPTED``, broken modules are in ran
    and remaining tests are re-queued.  Exercises the rerun-specific _INTERRUPTED
    check at failfast.py lines 321-332 (not the normal chunk path at line 350)."""
    from refactory_dsh.sandbox import SandboxResult

    # Force all items into one chunk so the exit-4 chunk has remaining items
    # for a rerun (with SELF_CHECK_CHUNKS=6 and 3 items, chunk_size=1 puts each
    # item in its own chunk, and the exit-4 chunk has no remaining items).
    monkeypatch.setattr(failfast, "SELF_CHECK_CHUNKS", 1)

    exit4_output = "ERROR: not found: /abs/path/tests/test_x.py::test_gone\n"
    exit4 = SandboxResult(ok=False, returncode=4, stdout=exit4_output, stderr="",
                          timed_out=False, elapsed_s=0.1)
    interrupted_output = (
        "ERROR tests/test_y.py - ModuleNotFoundError: No module named 'missing'\n"
        "Interrupted: 1 error during collection\n"
        "1 error in 0.10s"
    )
    interrupted = SandboxResult(ok=False, returncode=2, stdout=interrupted_output, stderr="",
                                timed_out=False, elapsed_s=0.1)
    pass_result = SandboxResult(ok=True, returncode=0, stdout="1 passed in 0.10s", stderr="",
                                timed_out=False, elapsed_s=0.1)

    call_count = [0]
    def run_fn(cmd, timeout):
        call_count[0] += 1
        if call_count[0] == 1:
            return exit4
        if call_count[0] == 2:
            return interrupted
        return pass_result

    prepass = failfast.targeted_prepass(
        {}, CHECK, run_fn, 180,
        selection=["tests/test_x.py::test_gone", "tests/test_y.py", "tests/test_z.py"],
        self_check_timeout_s=300,
    )
    # test_y.py did not import; it should be in ran (from the rerun _INTERRUPTED path).
    assert "tests/test_y.py" in prepass.ran_ids
    # test_z.py should have been re-queued and run separately (not via ran.extend).
    assert "tests/test_z.py" in prepass.ran_ids
    # Verify the re-queue path was taken: test_z.py ran in a third call.
    assert call_count[0] >= 3, "test_z.py must be re-queued and run separately"
    assert any("did not import" in n for n in prepass.notes)


def test_exhausted_empty_diff_does_not_call_targeted_prepass(tmp_path, monkeypatch):
    """When the empty-diff fixup budget is exhausted, ``targeted_prepass`` is not called:
    a ``PrepassResult`` is built directly. The self-checks are `_Chunks`' answers: the
    pinned test fails on the first emission and would pass on the emptied one, were it run."""
    _Chunks(monkeypatch, emissions=[{PIN}, set()])
    repo = _target(tmp_path / "repo", {**PINNED, **GENERATED})
    run = _spine_run(repo, tmp_path, monkeypatch, "abcdef500064", self_check_fixups=1)
    facts = _facts(run)
    (run / "facts.json").write_text(json.dumps({**facts, "generate": "python gen.py"}))

    # Monkeypatch targeted_prepass to raise if called with selection=[].
    from refactory_dsh import failfast as ff
    real_tp = ff.targeted_prepass
    called_empty = []

    def guarded_tp(*args, **kwargs):
        sel = kwargs.get("selection")
        if sel is not None and sel == []:
            called_empty.append(True)
            raise AssertionError("targeted_prepass called with selection=[]")
        return real_tp(*args, **kwargs)

    monkeypatch.setattr(ff, "targeted_prepass", guarded_tp)

    # First emission: BREAKS → self-check fixup.
    _emit(run, BREAKS, monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0 and "self-check fix-up 1 of 1" in out, err

    # Second emission: UNDOES → generate empties → empty_diff detected.
    # Budget: used=1 >= allowed=1 → exhausted path.
    # The exhausted path must NOT call targeted_prepass(selection=[]).
    _emit(run, UNDOES, monkeypatch)
    code, out, err = _prepare(run, monkeypatch)
    assert code == 0, err
    assert not called_empty, "targeted_prepass was called with selection=[]"
    records = _records(run)
    assert records[-1]["outcome"] == "exhausted"
    assert _facts(run)["attempts_used"] == 1


def test_the_limit_is_the_knob_and_never_more_than_a_third_of_the_check():
    assert failfast.self_check_limit(300, 1800) == 300
    assert failfast.self_check_limit(300, 180) == 60
    knob = DesignMode.declare()["self_check_timeout_s"]
    assert (knob.default, knob.lo, knob.hi, knob.frozen) == (300, 10, 600, False)


def test_a_renamed_id_with_time_left_is_recorded_not_a_broken_command(tmp_path, monkeypatch):
    """Exit 4 whose not-found ids were dropped leaves nothing to run: the selection is
    stale, not the command, so it is recorded and the whole check still runs."""
    import time as _time

    from refactory_dsh.sandbox import SandboxResult

    exit4 = SandboxResult(ok=False, returncode=4,
                          stdout="ERROR: not found: /abs/tests/test_x.py::test_gone\n",
                          stderr="", timed_out=False, elapsed_s=0.5)
    monkeypatch.setattr(failfast, "SELF_CHECK_CHUNKS", 1)

    def run_fn(cmd, timeout):
        return exit4

    result = failfast.targeted_prepass(
        {"status": "fail"}, "pytest -q tests", run_fn, 60,
        selection=["tests/test_x.py::test_gone"], self_check_timeout_s=60)
    assert result.ran_nothing, "it verified nothing"
    assert not result.broken_command, "a stale id is not a command pytest cannot parse"


def test_a_rerun_that_collects_nothing_is_recorded(tmp_path, monkeypatch):
    """The not-found rerun can also collect nothing: that must not pass quietly either."""
    from refactory_dsh.sandbox import SandboxResult

    first = SandboxResult(ok=False, returncode=4,
                          stdout="ERROR: not found: /abs/tests/test_y.py::test_gone\n",
                          stderr="", timed_out=False, elapsed_s=0.5)
    second = SandboxResult(ok=False, returncode=5, stdout="no tests ran\n",
                           stderr="", timed_out=False, elapsed_s=0.2)
    monkeypatch.setattr(failfast, "SELF_CHECK_CHUNKS", 1)
    calls = []

    def run_fn(cmd, timeout):
        calls.append(cmd)
        return first if len(calls) == 1 else second

    result = failfast.targeted_prepass(
        {"status": "fail"}, "pytest -q tests", run_fn, 60,
        selection=["tests/test_y.py::test_gone", "tests/test_x.py::test_here"],
        self_check_timeout_s=60)
    assert len(calls) == 2, "the not-found rerun happens"
    assert result.ran_nothing, "the rerun verified nothing, and the record says so"
    assert not result.broken_command
