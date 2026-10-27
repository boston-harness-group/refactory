"""An attempt's first self-check runs the whole check, beside the early asks judge (b48a).

`prepare_check` sends an attempt's first emission to a fork: `first_self_check` runs the
brief's whole check once, so the builder gets every failure in one pass, and the early
asks judge, an agent node, reads the diff against the brief's asks and the approved
corrections. `gate_self_check` sends the attempt's own failures and every item the judge
found not built back as one fix-up, or lets the attempt go on to the reviews, where the
check beside them reuses the whole check's result while the attempt is unchanged. Both
the self-check and the whole check inherit a failure the base has too, unless the brief
or a correction names it. Each test drives the real graph on the walker, or each node in
its own process as the spine runs it; the judge's agent node runs through the worker.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

import test_self_check as T
from refactory_dsh import design_ops, failfast, judges, lowering, repo_change, test_selection
from refactory_dsh.design import (DesignMode, gate_self_check_iterations,
                                  prepare_check_iterations)
from refactory_dsh.invocations import CHECK_PREP, CHECK_VERDICT
from refactory_dsh.sandbox import SandboxResult
from tests_support import FakeModel, early_asks_judgment, not_built, render

pytestmark = pytest.mark.usefixtures("fake_gh")

PIN, PINNED, CHECK, EMIT = T.PIN, T.PINNED, T.CHECK, T.EMIT
BREAKS = render({"tool.py": "VERSION = 2\n"})
FIXES_PIN = render({"tests/test_pin.py": "import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 2\n"})
CLEAN = render({"tool.py": "VERSION = 1\nNAME = 'tool'\n"})
JUDGE = design_ops.EARLY_ASKS_JUDGE_MARK
TWO_ASKS = "1. Change tool.py to VERSION 2.\n2. Add helper.py with a helper()."
HELPER = "Add helper.py with a helper()."


def _emits(model: FakeModel) -> list[str]:
    return [prompt for prompt in model.prompts if EMIT in prompt]


def _judged(model: FakeModel) -> list[str]:
    return [prompt for prompt in model.prompts if JUDGE in prompt]


def _whole_runs(monkeypatch, command: str = CHECK) -> list[dict]:
    """Every whole-check run, wherever it runs: the first self-check's, in the attempt's
    worktree, and the check beside the reviews', in its check copy."""
    calls: list[dict] = []
    real = repo_change.run_check

    def spy(ctx, cmd, timeout, **kwargs):
        if cmd == command:
            calls.append({"commit": str(ctx.facts.get("commit")), **kwargs})
        return real(ctx, cmd, timeout, **kwargs)

    monkeypatch.setattr(repo_change, "run_check", spy)
    return calls


def _check_node(run: Path, monkeypatch) -> dict:
    """The check beside the reviews, as the spine runs it, and the decision it recorded."""
    code, _, err = T._node(run, f"{design_ops.__name__}:check_beside_reviews", "check", "op",
                           monkeypatch)
    assert code == 0, err
    return json.loads((run / CHECK_VERDICT).read_text())


# ── the graph, on both runtimes ───────────────────────────────────────────────


def _edges(edges) -> set[tuple]:
    return {(edge.source, edge.target, edge.condition) for edge in edges}


def test_the_judge_is_an_agent_node_beside_the_first_self_check_only():
    graph = DesignMode().build()
    edges = _edges(graph.edges)
    assert graph.nodes["early_asks_check"].kind == "agent"
    assert graph.nodes["early_asks_check"].writes == (design_ops.EARLY_ASKS_VERDICT,)
    fork = graph.nodes["fork_self_check"]
    assert fork.kind == "fork" and set(fork.targets) == {"first_self_check", "early_asks_check"}
    assert fork.optional_targets == ("early_asks_check",)
    assert graph.nodes["join_self_check"].sources == ("first_self_check",)
    # The first emission goes to the fork; a fix-up's self-check stays in prepare_check.
    assert ("materialise", "prepare_check", None) in edges
    assert ("prepare_check", "fork_self_check", "first") in edges
    assert ("prepare_check", "pick_corrections", "reloop") in edges
    assert ("prepare_check", "fork_qa", "proceed") in edges
    assert not [edge for edge in edges if edge[1] == "early_asks_check"]
    assert {edge for edge in edges if edge[0] == "gate_self_check"} == {
        ("gate_self_check", "pick_corrections", "reloop"), ("gate_self_check", "fork_qa", "proceed"),
        ("gate_self_check", "report", "draft"), ("gate_self_check", "fork_self_check", "reread")}
    # The spine gets the same shape, and dispatches the judge as it does every agent node.
    workflow = lowering.to_workflow(DesignMode().state_dict())
    assert edges <= _edges(workflow.edges) | {e for e in edges if e[0] in ("fork_self_check",)}
    assert type(workflow.nodes["early_asks_check"]).__name__ == "AgentNode"
    assert workflow.nodes["first_self_check"].command.endswith("design_ops:first_self_check")


def test_the_judges_limits_follow_the_knobs_and_never_come_before_the_build_loops_stop():
    """gate_self_check counts its reloops and its rereads; it runs once an attempt's first
    self-check, which prepare_check routes, so its limit never ends a run."""
    defaults = {name: knob.default for name, knob in DesignMode.declare().items()}
    assert gate_self_check_iterations(defaults) == 2 * prepare_check_iterations(defaults) + 1
    mode = T._mode(check_attempts=5, check_attempt_cap=20, self_check_fixups=5)
    knobs = {name: value for name, value in mode.values().items()}
    assert mode.graph().nodes["gate_self_check"].max_iterations == gate_self_check_iterations(knobs)
    assert mode.state_dict()["graph"]["nodes"]["gate_self_check"]["max_iterations"] == \
        gate_self_check_iterations(knobs) > prepare_check_iterations(knobs)


# ── the judge's verdict ───────────────────────────────────────────────────────


@pytest.mark.parametrize("text, problem", [
    ("", "is empty"),
    ("all built", "holds 0 JSON objects"),
    ('{"items": []}', "lists no items"),
    ('{"items": [{"kind": "ask", "text": "a", "built": true}]}', "cites no path:line"),
    ('{"items": [{"kind": "ask", "text": "a", "built": "maybe"}]}', "whether it is built"),
    ('{"items": [{"kind": "wish", "text": "a", "built": false}]}', "expected ask or correction"),
    ('{"items": [{"kind": "correction", "text": "a", "built": false}]}', "judges no ask"),
    ('{"items": [{"kind": "ask", "text": "a", "built": false}]} {"items": []}', "holds 2 JSON"),
])
def test_a_verdict_that_judged_nothing_is_unreadable_never_everything_built(text, problem):
    with pytest.raises(judges.Unreadable, match=problem):
        design_ops.parse_early_asks(text, corrections_given=False)


def test_a_verdict_lists_the_corrections_when_the_person_gave_some():
    only_asks = early_asks_judgment()
    assert design_ops.parse_early_asks(only_asks, corrections_given=False)[0]["built"] is True
    with pytest.raises(judges.Unreadable, match="judges no correction"):
        design_ops.parse_early_asks(only_asks, corrections_given=True)
    fenced = "Here it is:\n```json\n" + early_asks_judgment(
        {"kind": "ask", "text": "x", "built": True, "evidence": "tool.py:3-9"},
        not_built("y", kind="correction")) + "\n```\n"
    items = design_ops.parse_early_asks(fenced, corrections_given=True)
    assert [(item["kind"], item["built"]) for item in items] == [("ask", True), ("correction", False)]
    assert design_ops._not_built_lines(items) == ["- (correction) y — missing: nothing in the diff does 'y'"]


# ── ask 1: every failure in round 1 ──────────────────────────────────────────

#: Eight tests that pin the version; five sort ahead of a slow file a chunked run is cut at.
MANY = {"tool.py": "VERSION = 1\n",
        **{f"tests/test_f{n}.py": f"import tool\n\n\ndef test_f{n}():\n    assert tool.VERSION == 1\n"
           for n in range(1, 9)},
        "tests/test_e_slow.py": "import tool\n" + "# padding\n" * 300 + "\n\ndef test_e():\n    assert tool\n"}
MANY_IDS = sorted(f"tests/test_f{n}.py::test_f{n}" for n in range(1, 9))
#: The slow file is the largest, so a chunked run takes it last: the cut hides nothing behind
#: it. These three are larger still, so a chunked run would reach them after the cut.
MANY.update({f"tests/test_f{n}.py": MANY[f"tests/test_f{n}.py"] + "# padding\n" * 400
             for n in (6, 7, 8)})


def test_the_first_self_check_lists_every_failure_and_converges_in_one_fixup(tmp_path, monkeypatch):
    """Eight pinned tests break; a chunked run would be cut at the slow file before three of
    them. The first self-check runs the whole check, so the first fix-up names all eight,
    and one fix-up is enough. On the walker, then on the spine node by node."""
    T._Chunks(monkeypatch, emissions=[set(MANY_IDS), set()], slow={"tests/test_e_slow.py"})
    repo = T._target(tmp_path / "repo", MANY)
    fixes = render({f"tests/test_f{n}.py": f"import tool\n\n\ndef test_f{n}():\n    assert tool.VERSION == 2\n"
                    for n in range(1, 9)})
    model = FakeModel(T._answers(BREAKS, fixes))
    result = T._design(tmp_path / "run", model, repo, T._mode(self_check_timeout_s=10, check_timeout_s=30))
    assert result.exit_ok, result.error
    first, second = T._records(tmp_path / "run")
    assert first["outcome"] == "fixup" and first["failing_ids"] == MANY_IDS and first["cut"] == []
    assert first["commands"] == [CHECK]
    fixup = _emits(model)[1]
    assert all(f"- {test_id}" in fixup for test_id in MANY_IDS)
    assert second["outcome"] == "pass" and second["fixups_before"] == 1
    assert len(_emits(model)) == 2 and T._count(model, T.HEALTH) == 1

    T._Chunks(monkeypatch, emissions=[set(MANY_IDS)], slow={"tests/test_e_slow.py"})
    run = T._spine_run(T._target(tmp_path / "repo2", MANY), tmp_path / "spine", monkeypatch,
                       "abcdef700001", self_check_timeout_s=10, check_timeout_s=30)
    T._emit(run, BREAKS, monkeypatch)
    code, out, err = T._prepare(run, monkeypatch)
    assert code == 0 and "self-check fix-up 1 of 2: 8 test(s)" in out, err
    [record] = T._records(run)
    assert record["failing_ids"] == MANY_IDS and record["cut"] == []


# ── ask 1 and correction 5: the check beside the reviews reuses the first run ────


def _passing_first_self_check(tmp_path, monkeypatch, run_id: str) -> Path:
    run = T._spine_run(T._target(tmp_path / "repo", PINNED), tmp_path, monkeypatch, run_id)
    T._emit(run, CLEAN, monkeypatch)
    code, out, err = T._prepare(run, monkeypatch)
    assert code == 0 and T._label(out) == "proceed", err
    return run


def test_the_check_beside_the_reviews_reuses_the_first_whole_check_and_says_so(tmp_path, monkeypatch):
    """A passing attempt runs its whole check once: the first self-check's, kept and reused
    by the check beside the reviews, whose evidence names the commit it was reused from.
    On the walker the whole run makes one whole-check run."""
    calls = _whole_runs(monkeypatch)
    run = _passing_first_self_check(tmp_path, monkeypatch, "abcdef700010")
    assert [call.get("prepared") for call in calls] == [True]
    verdict = _check_node(run, monkeypatch)
    assert verdict["verdict"] == "proceed" and len(calls) == 1
    commit = T._facts(run)["commit"]
    reused = f"reused from the first self-check, commit {commit[:12]}"
    assert f"({reused})" in (run / "check.txt").read_text()
    assert json.loads((run / "authoring-evidence.json").read_text())["validation"]["reused"] == reused
    assert not (run / design_ops.FIRST_CHECK_CACHE).exists()  # used once

    calls.clear()
    model = FakeModel(T._answers(CLEAN))
    result = T._design(tmp_path / "walk", model, T._target(tmp_path / "repo2", PINNED))
    assert result.exit_ok, result.error
    assert len(calls) == 1 and calls[0].get("prepared")


@pytest.mark.parametrize("key, value", [("commit", "0" * 40), ("command", CHECK + " -x"),
                                        ("setup", "make"), ("generate", "python gen.py")])
def test_the_first_whole_check_is_reused_only_while_the_attempt_is_unchanged(
        tmp_path, monkeypatch, key, value):
    """The kept result stands only for the commit, the check command, and the setup and
    generate commands it ran with: any one of them different and the check runs again."""
    run = _passing_first_self_check(tmp_path, monkeypatch, "abcdef700011")
    cache = run / design_ops.FIRST_CHECK_CACHE
    cache.write_text(json.dumps({**json.loads(cache.read_text()), key: value}))
    calls = _whole_runs(monkeypatch)
    assert _check_node(run, monkeypatch)["verdict"] == "proceed"
    assert [call.get("beside_reviews") for call in calls] == [True]
    assert "reused from the first self-check" not in (run / "check.txt").read_text()


def test_a_fixups_partial_run_is_never_reused_so_the_whole_check_stays_binding(tmp_path, monkeypatch):
    """The first self-check sends the pinned test back; the fix-up's self-check runs only
    what the fix-up can break, and passes. test_other.py, which the fix-up's run never
    selected, still fails: the check beside the reviews runs for real and finds it. Its
    chunks are `_Chunks`' answers; the check beside the reviews is real."""
    other = "import tool\n\n\ndef test_other():\n    assert tool.VERSION == 1\n"
    T._Chunks(monkeypatch, emissions=[{PIN}, set()])
    repo = T._target(tmp_path / "repo", {**PINNED, "tests/test_other.py": other})
    run = T._spine_run(repo, tmp_path, monkeypatch, "abcdef700012")
    T._emit(run, BREAKS, monkeypatch)
    code, out, err = T._prepare(run, monkeypatch)
    assert code == 0 and "self-check fix-up 1 of 2" in out, err
    T._emit(run, FIXES_PIN, monkeypatch)
    code, out, err = T._prepare(run, monkeypatch)
    assert code == 0 and T._label(out) == "proceed", err
    second = T._records(run)[-1]
    assert second["outcome"] == "pass" and "tests/test_other.py" not in second["selection"]
    assert not (run / design_ops.FIRST_CHECK_CACHE).exists()
    calls = _whole_runs(monkeypatch)
    verdict = _check_node(run, monkeypatch)
    assert verdict["verdict"] == "reloop" and "test_other" in verdict["reason"]
    assert [call.get("beside_reviews") for call in calls] == [True]


# ── item 6: a failing whole check that names no test ──────────────────────────


def test_a_failing_first_check_that_names_no_test_goes_back_and_is_never_a_pass(tmp_path, monkeypatch):
    """A build error fails the whole check before any test runs. It is the attempt's
    failure, sent back with what the check printed, never recorded as a pass nor kept for
    the check beside the reviews; and the fix-up's self-check, with no test to target,
    runs the whole check again."""
    broken = SandboxResult(ok=False, returncode=2, elapsed_s=0.1,
                           stdout="error TS2304: Cannot find name 'x'.\nbuild failed\n")
    whole = []
    real = repo_change.run_check

    def run_check(ctx, command, timeout, **kwargs):
        if kwargs.get("prepared") and command == CHECK:
            whole.append(str(ctx.facts.get("commit")))
            return broken if len(whole) == 1 else SandboxResult(ok=True, returncode=0,
                                                                stdout="2 passed", elapsed_s=0.1)
        return real(ctx, command, timeout, **kwargs)

    monkeypatch.setattr(repo_change, "run_check", run_check)
    run = T._spine_run(T._target(tmp_path / "repo", PINNED), tmp_path, monkeypatch, "abcdef700013")
    T._emit(run, BREAKS, monkeypatch)
    code, out, err = T._prepare(run, monkeypatch)
    assert code == 0 and T._label(out) == "reloop", err
    assert "the check failed (exit 2) without naming a failing test" in out
    assert "Cannot find name 'x'" in T._facts(run)["validation"]
    [first] = T._records(run)
    assert first["outcome"] == "fixup" and first["unparsed"] and first["failing_ids"] == []
    assert not (run / design_ops.FIRST_CHECK_CACHE).exists()
    T._emit(run, render({"tool.py": "VERSION = 1\nX = 1\n"}), monkeypatch)
    code, out, err = T._prepare(run, monkeypatch)
    assert code == 0 and T._label(out) == "proceed", err
    second = T._records(run)[-1]
    assert second["outcome"] == "pass" and second["commands"] == [CHECK] and len(whole) == 2


# ── ask 2 and correction 2: one inheritance rule, in the self-check and the whole check ──

ENV, ALWAYS = T.ENV_FAILURE, T.ALWAYS_FAILS


def _inherited_both_ways(run: Path, monkeypatch) -> None:
    """The self-check inherited ENV and went on; the check beside the reviews, reusing the
    first run and then running for real, inherits it too."""
    code, out, err = T._prepare(run, monkeypatch)
    assert code == 0 and T._label(out) == "proceed", err
    [record] = T._records(run)
    assert record["outcome"] == "inherited" and record["inherited_filtered"] == [ENV]
    cache = (run / design_ops.FIRST_CHECK_CACHE).read_text()
    assert _check_node(run, monkeypatch)["verdict"] == "proceed"
    assert "reused from the first self-check" in (run / "check.txt").read_text()
    (run / design_ops.FIRST_CHECK_CACHE).write_text(cache.replace('"command": "', '"command": "x'))
    (run / CHECK_VERDICT).unlink()
    calls = _whole_runs(monkeypatch)
    assert _check_node(run, monkeypatch)["verdict"] == "proceed"
    assert calls and ENV in T._facts(run)["inherited_failures"]


def test_a_base_failure_in_a_test_file_the_change_touched_is_inherited_by_both_checks(
        tmp_path, monkeypatch):
    repo = T._target(tmp_path / "repo", {**PINNED, "tests/test_env.py": ALWAYS})
    run = T._spine_run(repo, tmp_path, monkeypatch, "abcdef700020")
    T._emit(run, render({"tool.py": "VERSION = 1\nNAME = 'tool'\n",
                         "tests/test_env.py": ALWAYS + "# touched\n"}), monkeypatch)
    _inherited_both_ways(run, monkeypatch)


def test_a_base_failure_named_only_in_the_design_is_inherited_by_both_checks(tmp_path, monkeypatch):
    repo = T._target(tmp_path / "repo", {**PINNED, "tests/test_env.py": ALWAYS})
    run = T._spine_run(repo, tmp_path, monkeypatch, "abcdef700021")
    (run / "design.md").write_text(T.LEFT_DESIGN + f"\n## Notes\n\n- {ENV} needs a network\n")
    T._emit(run, CLEAN, monkeypatch)
    _inherited_both_ways(run, monkeypatch)


def test_on_the_walker_a_touched_base_failure_costs_no_round(tmp_path, monkeypatch):
    """End to end: one emission, no attempt spent, no draft."""
    repo = T._target(tmp_path / "repo", {**PINNED, "tests/test_env.py": ALWAYS})
    emission = render({"tool.py": "VERSION = 1\nNAME = 'tool'\n",
                       "tests/test_env.py": ALWAYS + "# touched\n"})
    model = FakeModel(T._answers(emission, emission, emission))
    result = T._design(tmp_path / "run", model, repo)
    assert result.exit_ok, result.error
    assert len(_emits(model)) == 1 and not result.facts.get("attempts_used")
    assert not result.facts.get("draft_delivery")
    assert result.facts["inherited_failures"] == [ENV]


@pytest.mark.parametrize("brief", [f"a tool; fix {ENV}", "a tool; tests/test_env.py must pass",
                                   "a tool; test_env must pass"])
def test_a_base_failure_the_brief_names_is_sent_back_by_both_checks(tmp_path, monkeypatch, brief):
    """By its id, its file or its name: the run exists to fix it, so it is never inherited."""
    repo = T._target(tmp_path / "repo", {**PINNED, "tests/test_env.py": ALWAYS})
    run = T._spine_run(repo, tmp_path, monkeypatch, "abcdef700022")
    (run / "facts.json").write_text(json.dumps({**T._facts(run), "brief": brief}))
    T._emit(run, CLEAN, monkeypatch)
    code, out, err = T._prepare(run, monkeypatch)
    assert code == 0 and T._label(out) == "reloop" and f"- {ENV}" in out, err
    [record] = T._records(run)
    assert record["failing_ids"] == [ENV] and record["inherited_filtered"] == []
    # The whole check, run for real on the same attempt, sends it back too.
    (run / CHECK_PREP).write_text(json.dumps({"attempt": "a1", "commit": T._facts(run)["commit"],
                                              "root": T._facts(run)["artifact"]}))
    verdict = _check_node(run, monkeypatch)
    assert verdict["verdict"] == "reloop" and ENV in verdict["reason"]


def test_a_correction_that_names_the_test_keeps_it_from_being_inherited(tmp_path, monkeypatch):
    repo = T._target(tmp_path / "repo", {**PINNED, "tests/test_env.py": ALWAYS})
    run = T._spine_run(repo, tmp_path, monkeypatch, "abcdef700023")
    (run / "facts.json").write_text(json.dumps({**T._facts(run),
                                                "gate_corrections": "- make test_env.py pass"}))
    T._emit(run, CLEAN, monkeypatch)
    judge = early_asks_judgment(prompt="## Corrections the person gave when approving the "
                                       "design\n- x\n\n## What to read")
    code, out, err = T._prepare(run, monkeypatch, judge=judge)
    assert code == 0 and T._label(out) == "reloop", err
    assert T._records(run)[0]["failing_ids"] == [ENV]


def test_a_name_that_only_starts_a_word_of_the_brief_does_not_keep_a_test(tmp_path, monkeypatch):
    """``test_env`` is not named by ``test_environment_checks``: whole words only."""
    repo = T._target(tmp_path / "repo", {**PINNED, "tests/test_env.py": ALWAYS})
    run = T._spine_run(repo, tmp_path, monkeypatch, "abcdef700024")
    (run / "facts.json").write_text(json.dumps({**T._facts(run),
                                                "brief": "a tool; keep test_environment_checks"}))
    T._emit(run, CLEAN, monkeypatch)
    code, out, err = T._prepare(run, monkeypatch)
    assert code == 0 and T._label(out) == "proceed", err
    assert T._records(run)[0]["inherited_filtered"] == [ENV]


# ── correction 4: b47a's first two self-checks, replayed through the new rules ──

B47A = json.loads((Path(__file__).parent / "fixtures" / "b47a_self_checks.json").read_text())
B47A_CHECK = "python -m pytest -q refactory/tests"


def _b47a_repo(root: Path) -> Path:
    """b47a's test files, each failing id a real test: on the base, the lab-only ones fail
    and the rest pass (the attempt's runs are `_Chunks`' answers)."""
    round1, round2 = B47A["rounds"]
    lab_only = set(B47A["lab_only"])
    functions: dict[tuple[str, str], list[str | None]] = {}
    for test_id in sorted({*round1["failing_ids"], *round2["failing_ids"]}):
        path, name = test_id.split("::", 1)
        function, _, param = name.partition("[")
        functions.setdefault((path, function), []).append(param[:-1] if param else None)
    files = {"tool.py": "VERSION = 1\n",
             **{name.split("::")[0]: "import pytest\n\n\ndef test_ok():\n    pass\n"
                for name in (*round1["selection"], *round2["selection"])}}
    for (path, function), params in functions.items():
        failing = any(test_id.startswith(f"{path}::{function}") for test_id in lab_only)
        body = "    assert False, 'fails in this environment'\n" if failing else "    pass\n"
        if params == [None]:
            files[path] += f"\n\ndef {function}():\n{body}"
        else:
            files[path] += (f"\n\n@pytest.mark.parametrize('p', {params!r}, ids={params!r})\n"
                            f"def {function}(p):\n{body}")
    return T._target(root, files)


class _SlowWholeFiles(T._Chunks):
    """`_Chunks`, with a slow file slow only when a chunk runs it whole: b47a's round 2 ran
    the ids round 1 found in test_gui_loop.py in its first chunks, and cut the whole file."""

    def __call__(self, command: str, timeout: float):
        named = [word for word in shlex.split(command) if word.endswith(".py") or ".py::" in word]
        if any(name in self.slow for name in named):
            self.now += timeout
            return SandboxResult(ok=False, returncode=124, timed_out=True, timeout_s=int(timeout),
                                 elapsed_s=timeout, stderr=f"timed out after {int(timeout)}s")
        slow, self.slow = self.slow, set()
        try:
            return super().__call__(command, timeout)
        finally:
            self.slow = slow


def test_b47a_replayed_round_one_lists_what_its_cut_hid_and_inherits_the_lab_only_failures(
        tmp_path, monkeypatch):
    """b47a's round 1 ran its 82 files in chunks, found 25 failures and cut 10 files at the
    limit; round 2 found 18, five of them in test_workers.py, which round 1 had cut. (Its
    other 13 were in files round 1 ran and passed: the fix-up made them.) The replay runs
    the real self-check on b47a's selections, ids and cuts, the cut files slow to a chunk:

    - the chunked run of round 1's selection reproduces b47a's round 1, so the fake is
      faithful;
    - the new first self-check lists round 1's failures and the five its cut hid, with the
      eight lab-only failures inherited, as the base fails them too;
    - the fix-up's self-check runs round 2's selection in chunks and cuts b47a's 4 files.
    """
    round1, round2 = B47A["rounds"]
    hidden = sorted(test_id for test_id in round2["failing_ids"]
                    if test_id.split("::")[0] in round1["cut"])
    assert len(round1["failing_ids"]) == 25 and len(round1["cut"]) == 10 and len(hidden) == 5
    chunks = _SlowWholeFiles(monkeypatch, emissions=[{*round1["failing_ids"], *hidden},
                                               set(round2["failing_ids"])],
                       slow=set(round1["cut"]))
    # The fake answers b47a's round-1 chunks as the lab did: 25 found, the same 10 files cut.
    chunks.fails = {*round1["failing_ids"], *hidden}
    replayed = failfast.targeted_prepass(
        {}, B47A_CHECK, chunks, 1800, selection=round1["selection"], self_check_timeout_s=300)
    assert sorted(replayed.failing_ids) == sorted(round1["failing_ids"])
    assert replayed.cut_ids == round1["cut"] and replayed.notes == round1["notes"]
    chunks.now, chunks.fails = 0.0, set()

    rounds = iter(B47A["rounds"])
    real_select = test_selection.select

    def recorded(*args, **kwargs):
        selection = real_select(*args, **kwargs)
        this = next(rounds)
        selection.tests, selection.alone = list(this["selection"]), list(this["alone"])
        return selection

    monkeypatch.setattr(test_selection, "select", recorded)
    run = T._spine_run(_b47a_repo(tmp_path / "repo"), tmp_path, monkeypatch, "abcdef7b47a1",
                       check_timeout_s=1800)
    (run / "facts.json").write_text(json.dumps({**T._facts(run), "check": B47A_CHECK,
                                                "brief": "one model choice for every run"}))
    T._emit(run, BREAKS, monkeypatch)
    code, out, err = T._prepare(run, monkeypatch)
    assert code == 0 and T._label(out) == "reloop", err
    [first] = T._records(run)
    lab_only = sorted(B47A["lab_only"])
    assert first["commands"] == [B47A_CHECK] and first["cut"] == []
    assert first["inherited_filtered"] == lab_only
    assert sorted(first["failing_ids"]) == sorted({*round1["failing_ids"], *hidden} - set(lab_only))
    assert set(hidden) <= set(first["failing_ids"])
    validation = T._facts(run)["validation"]
    assert all(f"- {test_id}" in validation for test_id in hidden)
    assert not any(f"- {test_id}\n" in validation for test_id in lab_only)

    chunks.slow = set(round2["cut"])
    T._emit(run, render({"tool.py": "VERSION = 3\n"}), monkeypatch)
    code, out, err = T._prepare(run, monkeypatch)
    assert code == 0 and "self-check fix-up 2 of 2" in out, err
    second = T._records(run)[-1]
    assert second["fixups_before"] == 1 and second["cut"] == round2["cut"]
    assert second["notes"] == round2["notes"]
    assert sorted(second["failing_ids"]) == sorted(round2["failing_ids"])
    assert CHECK not in second["commands"] and B47A_CHECK not in second["commands"]


# ── correction 1: the early asks judge ────────────────────────────────────────

BUILT = {"id": "ask-1", "kind": "ask", "text": "Change tool.py to VERSION 2", "built": True,
         "evidence": "tool.py:1"}
HELPER_ADDED = render({"helper.py": "def helper():\n    return 1\n"})


def test_an_ask_the_judge_finds_not_built_joins_the_first_fixup_with_the_failures(tmp_path, monkeypatch):
    T._Chunks(monkeypatch, emissions=[{PIN}, set()])
    repo = T._target(tmp_path / "repo", PINNED)
    fixes = render({"tests/test_pin.py": "import tool\n\n\ndef test_pin():\n    assert tool.VERSION == 2\n",
                    "helper.py": "def helper():\n    return 1\n"})
    model = FakeModel(T._answers(BREAKS, fixes),
                      early_asks=[early_asks_judgment(BUILT, not_built(HELPER))])
    result = T._design(tmp_path / "run", model, repo, brief=TWO_ASKS)
    assert result.exit_ok, result.error
    first, second = T._records(tmp_path / "run")
    assert first["outcome"] == "fixup" and first["failing_ids"] == [PIN]
    assert first["not_built"] == [f"- (ask) {HELPER} — missing: nothing in the diff does {HELPER!r}"]
    assert first["judge"]["status"] == "judged"
    fixup = _emits(model)[1]
    assert fixup.count("self-check fix-up 1 of 2: 1 test(s) this change can break fail on this "
                       "attempt; the early asks judge found 1 ask(s) or correction(s) not built") == 1
    assert f"- {PIN}" in fixup and "## Asks and corrections not built" in fixup and HELPER in fixup
    # The judge runs beside the first self-check only: the fix-up's ran no judge.
    assert len(_judged(model)) == 1 and TWO_ASKS in _judged(model)[0]
    assert second["outcome"] == "pass" and "judge" not in second


def test_an_ask_not_built_goes_back_even_when_every_test_passes(tmp_path, monkeypatch):
    """b47a's case: the function was there and its tests passed, but nothing called it."""
    T._Chunks(monkeypatch, fails=set())
    repo = T._target(tmp_path / "repo", PINNED)
    model = FakeModel(T._answers(CLEAN, HELPER_ADDED),
                      early_asks=[early_asks_judgment(BUILT, not_built(HELPER))])
    result = T._design(tmp_path / "run", model, repo, brief=TWO_ASKS)
    assert result.exit_ok, result.error
    emits = _emits(model)
    assert len(emits) == 2 and HELPER in emits[1]
    assert "self-check fix-up 1 of 2: the early asks judge found 1 ask(s) or correction(s) not built" in emits[1]
    first, second = T._records(tmp_path / "run")
    assert first["outcome"] == "fixup" and first["failing_ids"] == []
    # helper.py is new and nothing imports it: the fix-up's self-check selects nothing.
    assert second["outcome"] == "nothing selected" and T._count(model, T.HEALTH) == 1
    assert not result.facts.get("attempts_used")


def test_a_fully_built_attempt_goes_straight_on(tmp_path, monkeypatch):
    T._Chunks(monkeypatch, fails=set())
    model = FakeModel(T._answers(CLEAN))
    result = T._design(tmp_path / "run", model, T._target(tmp_path / "repo", PINNED))
    assert result.exit_ok, result.error
    [record] = T._records(tmp_path / "run")
    assert record["outcome"] == "pass" and "not_built" not in record
    assert record["judge"] == {"status": "judged", "not_built": [], "items": 1}
    assert len(_emits(model)) == 1 and len(_judged(model)) == 1
    assert not any("not built" in prompt for prompt in _emits(model))


def test_an_unreadable_verdict_is_asked_again_beside_the_same_self_check_result(tmp_path, monkeypatch):
    """The fork runs again: the judge is asked a second time, the whole check is not."""
    T._Chunks(monkeypatch, fails=set())
    calls = _whole_runs(monkeypatch)
    model = FakeModel(T._answers(CLEAN, HELPER_ADDED),
                      early_asks=["all built, trust me", early_asks_judgment(BUILT, not_built(HELPER))])
    result = T._design(tmp_path / "run", model, T._target(tmp_path / "repo", PINNED), brief=TWO_ASKS)
    assert result.exit_ok, result.error
    judged = _judged(model)
    assert len(judged) == 2 and "## Your last verdict could not be read" in judged[1]
    assert "early-asks.json holds 0 JSON objects" in judged[1]
    assert len([call for call in calls if call.get("prepared")]) == 1
    assert HELPER in _emits(model)[1]


def test_a_judge_that_still_fails_is_a_warning_on_record_never_everything_built(
        tmp_path, monkeypatch, capfd):
    """A judge whose model call fails, then whose answer cannot be read: the optional
    branch is recorded missing, asked again once, and the attempt goes on without a
    verdict, the record and a run warning saying the judge failed."""
    T._Chunks(monkeypatch, fails=set())

    class Failing(FakeModel):
        def complete(self, prompt, *, system="", workdir=None, **workspace):
            if JUDGE in prompt and not any(JUDGE in seen for seen in self.prompts):
                self.prompts.append(prompt)
                raise RuntimeError("the judge's session ended")
            return super().complete(prompt, system=system, workdir=workdir, **workspace)

    model = Failing(T._answers(CLEAN), early_asks=["{}"])
    result = T._design(tmp_path / "run", model, T._target(tmp_path / "repo", PINNED))
    assert result.exit_ok, result.error
    [record] = T._records(tmp_path / "run")
    assert record["outcome"] == "pass" and record["judge"]["status"] == "failed"
    assert "early-asks.json lists no items" in record["judge"]["problem"]
    assert "also when it was asked again" in record["judge"]["problem"]
    assert "refactory: run warning: the early asks judge failed" in capfd.readouterr().err
    assert len(_judged(model)) == 2 and len(_emits(model)) == 1
    # The walker records the raising judge's error as the spine's worker does, so the
    # judge asked again is told why (#517 follow-up).
    assert "(the judge failed: RuntimeError: the judge's session ended)" in _judged(model)[1]


def test_the_judges_limits_never_end_the_run_however_often_the_pair_is_used(tmp_path, monkeypatch):
    """Every attempt's first self-check fails and its judge is read only the second time:
    the self-check gate reloops and rereads again and again, and the build still ends in the
    build loop's stop (a draft), never at a gate's iteration limit."""
    T._Chunks(monkeypatch, fails={PIN})
    T._unwatched(monkeypatch)
    emits = [render({"tool.py": f"VERSION = {10 + n}\n"}) for n in range(12)]
    model = FakeModel(T._answers(*emits), early_asks=["?", early_asks_judgment()] * 6)
    result = T._design(tmp_path / "run", model, T._target(tmp_path / "repo", PINNED),
                       T._mode(check_attempts=3, self_check_fixups=1))
    assert result.exit_ok, result.error
    assert result.facts["draft_delivery"]["reason"] == "check attempts exhausted after 3 check failure(s)"
    outcomes = [entry["outcome"] for entry in T._records(tmp_path / "run")]
    assert outcomes == ["fixup", "exhausted"] * 3
    assert len(_judged(model)) == 6  # asked twice an attempt: the pair was used again and again



# ── dm-276: past the fix-up budget, not-built items alone go on to the reviews ──

#: #517's probe P18: a judge that keeps finding the helper not built.
NEVER_BUILT = [early_asks_judgment(BUILT, not_built(HELPER))] * 6


def test_past_the_fixup_budget_unbuilt_asks_alone_go_on_to_the_reviews(tmp_path, monkeypatch):
    """P18: no fix-up budget, a judge that keeps answering not built, every test passing.
    The attempt goes on to the reviews with the not-built items listed for the QA judge and
    the overwatch (both read the synthesis), instead of spending attempts unreviewed and
    delivering a draft marked check failing."""
    T._Chunks(monkeypatch, fails=set())
    model = FakeModel(T._answers(CLEAN), early_asks=list(NEVER_BUILT))
    run = tmp_path / "run"
    result = T._design(run, model, T._target(tmp_path / "repo", PINNED),
                       T._mode(self_check_fixups=0), brief=TWO_ASKS)
    assert result.exit_ok, result.error
    [record] = T._records(run)
    assert record["outcome"] == "pass" and record["failing_ids"] == []
    assert record["not_built_to_reviews"] == "0 of 0 fix-ups used on this attempt"
    assert len(_emits(model)) == 1 and T._count(model, T.HEALTH) == 1
    assert not result.facts.get("draft_delivery") and not result.facts.get("attempts_used")
    synthesis = (run / "reviews/qa-synthesized.md").read_text()
    section = synthesis.split("## Asks and corrections not built", 1)[1]
    assert f"- (ask) {HELPER}" in section and "judge each one" in section
    for judge in ("You are the QA judge.", "You are the overwatch."):
        assert any(judge in prompt and "reviews/qa-synthesized.md" in prompt for prompt in model.prompts)


def test_on_the_spine_past_the_budget_unbuilt_asks_go_on_and_failing_tests_still_stop(
        tmp_path, monkeypatch):
    """The same on the spine, node by node: gate_self_check proceeds and synthesize_qa lists
    the items. Failing tests past the budget keep the early stop, the items with them."""
    T._Chunks(monkeypatch, fails=set())
    run = T._spine_run(T._target(tmp_path / "repo", PINNED), tmp_path, monkeypatch, "abcdef700060",
                       self_check_fixups=0)
    T._emit(run, CLEAN, monkeypatch)
    code, out, err = T._prepare(run, monkeypatch, judge=NEVER_BUILT[0])
    assert code == 0 and T._label(out) == "proceed", err
    [record] = T._records(run)
    assert record["outcome"] == "pass" and record["not_built_to_reviews"]
    code, _, err = T._node(run, f"{design_ops.__name__}:synthesize_qa", "synthesize_qa", "op",
                           monkeypatch)
    assert code == 0, err
    assert f"- (ask) {HELPER}" in (run / "reviews/qa-synthesized.md").read_text().split(
        "## Asks and corrections not built", 1)[1]

    T._Chunks(monkeypatch, fails={PIN})
    failing = T._spine_run(T._target(tmp_path / "repo2", PINNED), tmp_path / "f", monkeypatch,
                           "abcdef700061", self_check_fixups=0)
    T._emit(failing, BREAKS, monkeypatch)
    code, out, err = T._prepare(failing, monkeypatch, judge=NEVER_BUILT[0])
    assert code == 0 and T._label(out) != "proceed", err
    [record] = T._records(failing)
    assert record["outcome"] == "exhausted" and "not_built_to_reviews" not in record
    assert HELPER in T._facts(failing)["validation"]


def test_a_new_attempts_first_self_check_clears_the_last_attempts_judge_verdict(tmp_path, monkeypatch):
    """#517's M33: an early-asks.json an earlier attempt's judge left is set aside before
    the fork, so a judge that writes nothing this time is never read as the old verdict."""
    T._Chunks(monkeypatch, fails=set())
    run = T._spine_run(T._target(tmp_path / "repo", PINNED), tmp_path, monkeypatch, "abcdef700062")
    T._emit(run, CLEAN, monkeypatch)
    (run / design_ops.EARLY_ASKS_VERDICT).write_text(early_asks_judgment(BUILT))
    code, out, err = T._node(run, f"{design_ops.__name__}:prepare_check", "prepare_check", "gate",
                             monkeypatch)
    assert code == 0 and T._label(out) == "first", err
    assert not (run / design_ops.EARLY_ASKS_VERDICT).exists()


@pytest.mark.parametrize("said, named", [
    ("**Updated tests:** `{line}`", "{line}"), ("**Updated tests**: {line}", "{line}"),
    ("> _Updated tests:_ **{line}**", "{line}"),
    ("* Updated tests: **tests/test_pin.py::test_pin** (follows ask 1)", "{line}"),
    # A path that starts or ends with `_` is a name, not emphasis.
    ("Updated tests: _private/test_pin.py::test_pin (follows ask 1)",
     "_private/test_pin.py::test_pin (follows ask 1)"),
    ("Updated tests: tests/test_pin.py::test_pin_ (follows ask 1)",
     "tests/test_pin.py::test_pin_ (follows ask 1)"),
    ("Updated tests: tests/test_x.py::test__init__", "tests/test_x.py::test__init__")])
def test_markdown_around_the_updated_tests_line_is_not_part_of_the_test_id(tmp_path, monkeypatch,
                                                                            said, named):
    T._Chunks(monkeypatch, fails=set())
    line = "tests/test_pin.py::test_pin (follows ask 1)"
    model = _Builder(T._answers(CLEAN), [f"Done.\n{said.format(line=line)}\n"])
    result = T._design(tmp_path / "run", model, T._target(tmp_path / "repo", PINNED))
    assert result.exit_ok, result.error
    assert result.facts["updated_tests"] == f"- {named.format(line=line)}"


def test_past_the_budget_a_failure_that_names_no_test_still_stops_with_the_unbuilt_items(
        tmp_path, monkeypatch):
    """A whole check that fails naming no test (a build error) is the attempt's failure, not
    a pass: past the budget it takes the early stop with the judge's not-built items, and
    the attempt never reaches the reviews."""
    broken = SandboxResult(ok=False, returncode=2, elapsed_s=0.1,
                           stdout="error TS2304: Cannot find name 'x'.\nbuild failed\n")
    real = repo_change.run_check

    def run_check(ctx, command, timeout, **kwargs):
        if kwargs.get("prepared") and command == CHECK:
            return broken
        return real(ctx, command, timeout, **kwargs)

    monkeypatch.setattr(repo_change, "run_check", run_check)
    run = T._spine_run(T._target(tmp_path / "repo", PINNED), tmp_path, monkeypatch, "abcdef700063",
                       self_check_fixups=0)
    T._emit(run, BREAKS, monkeypatch)
    code, out, err = T._prepare(run, monkeypatch, judge=NEVER_BUILT[0])
    assert code == 0 and T._label(out) not in ("proceed", "first"), err
    [record] = T._records(run)
    assert record["outcome"] == "exhausted" and record["unparsed"]
    assert "not_built_to_reviews" not in record
    assert HELPER in T._facts(run)["validation"]


def test_an_earlier_attempts_unbuilt_items_are_not_listed_for_a_later_attempt(tmp_path, monkeypatch):
    """Attempt 1 goes to the reviews past the budget with an item not built, and the QA judge
    sends it back; attempt 2's judge finds everything built. The synthesis the QA judge and
    the overwatch read for attempt 2 lists nothing from attempt 1."""
    from tests_support import qa_judgment

    T._Chunks(monkeypatch, fails=set())
    held = {"source": "qa", "where": "helper.py:1", "what": "the helper is not there",
            "why": "the brief's second ask", "severity": "high"}
    second = render({"tool.py": "VERSION = 1\nNAME = 'tool'\n", "helper.py": "def helper():\n    return 1\n"})
    answers = T._answers(CLEAN)
    model = FakeModel([*answers[:-1], qa_judgment("redo", must_fix=[held]), second,
                       *answers[-5:]], early_asks=[NEVER_BUILT[0]])
    run = tmp_path / "run"
    result = T._design(run, model, T._target(tmp_path / "repo", PINNED),
                       T._mode(self_check_fixups=0), brief=TWO_ASKS)
    assert result.exit_ok, result.error
    first, later = T._records(run)
    assert first["not_built_to_reviews"] and "not_built_to_reviews" not in later
    assert sum("You are the QA judge." in prompt for prompt in model.prompts) == 2
    assert "## Asks and corrections not built" not in (run / "reviews/qa-synthesized.md").read_text()


# ── the judge on the spine: an agent node the worker dispatches ───────────────


def _judge_spec(run: Path) -> dict:
    """The judge's worker spec, as the spine lowers the design graph for this run."""
    from refactory_dsh import workers
    from refactory_dsh.model import DshModelClient

    state = T._mode().state_dict()
    lowered = workers.lower_workflow(lowering.to_workflow(state), run,
                                     DshModelClient(python_bin=Path(sys.executable).parent).context(),
                                     lowering.node_declarations(state))
    return json.loads(Path(shlex.split(lowered.nodes["early_asks_check"].dispatch_command)[-1]).read_text())


def _to_the_judge(tmp_path, monkeypatch, run_id: str) -> Path:
    """A spine run at its first self-check's fork, the whole check run and passed."""
    T._Chunks(monkeypatch, fails=set())
    run = T._spine_run(T._target(tmp_path / "repo", PINNED), tmp_path, monkeypatch, run_id)
    (run / "facts.json").write_text(json.dumps({**T._facts(run), "brief": TWO_ASKS}))
    (run / "brief.json").write_text(json.dumps({"brief": TWO_ASKS}))
    T._emit(run, CLEAN, monkeypatch)
    code, out, err = T._node(run, f"{design_ops.__name__}:prepare_check", "prepare_check", "gate",
                             monkeypatch)
    assert code == 0 and T._label(out) == "first", err
    code, _, err = T._node(run, f"{design_ops.__name__}:first_self_check", "first_self_check",
                           "op", monkeypatch)
    assert code == 0, err
    return run


def _gate(run: Path, monkeypatch) -> tuple[int, str, str]:
    return T._node(run, f"{design_ops.__name__}:gate_self_check", "gate_self_check", "gate",
                   monkeypatch)


def test_on_the_spine_the_judge_runs_through_the_worker_and_its_verdict_decides(tmp_path, monkeypatch):
    """The judge is dispatched as every agent node is: its worker renders the prompt from the
    run's facts, calls the model, and its answer becomes early-asks.json, which
    gate_self_check, its own process, reads."""
    from refactory_dsh.model import DshModelClient
    from refactory_dsh.worker_node import execute

    run = _to_the_judge(tmp_path, monkeypatch, "abcdef700040")
    prompts: list[str] = []

    def complete(self, prompt, *, system="", workdir=None):
        prompts.append(prompt)
        return early_asks_judgment(BUILT, not_built(HELPER))

    monkeypatch.setattr(DshModelClient, "complete", complete)
    monkeypatch.setenv("REFACTORY_WORKER_BACKEND", "dsh")
    spec = _judge_spec(run)
    assert spec["writes"] == [design_ops.EARLY_ASKS_VERDICT] and spec["optional"] is True
    monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": "early_asks_check", "type": "AgentNode"}))
    receipt = execute(spec, run)
    assert receipt["status"] == "delivered" and receipt["captured_final"] == design_ops.EARLY_ASKS_VERDICT
    [prompt] = prompts
    assert JUDGE in prompt and TWO_ASKS in prompt and "changes.diff" in prompt
    code, out, err = _gate(run, monkeypatch)
    assert code == 0 and T._label(out) == "reloop", err
    assert "the early asks judge found 1 ask(s) or correction(s) not built" in out and HELPER in out
    [record] = T._records(run)
    assert record["outcome"] == "fixup" and record["judge"]["status"] == "judged"


def test_on_the_spine_a_judge_that_fails_is_recorded_missing_and_asked_again(tmp_path, monkeypatch):
    """The worker records the optional judge missing, with its error; the gate asks it
    again, with the reason in its prompt; a judge still failing goes on with the warning."""
    from refactory_dsh import worker_node
    from refactory_dsh.model import DshModelClient

    run = _to_the_judge(tmp_path, monkeypatch, "abcdef700041")

    def complete(self, prompt, *, system="", workdir=None):
        raise RuntimeError("the model is unavailable")

    monkeypatch.setattr(DshModelClient, "complete", complete)
    monkeypatch.setenv("REFACTORY_WORKER_BACKEND", "dsh")
    spec = _judge_spec(run)
    monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": "early_asks_check", "type": "AgentNode"}))
    path = run / "judge-spec.json"
    path.write_text(json.dumps(spec))
    monkeypatch.chdir(run)
    monkeypatch.setattr(sys, "argv", ["worker_node", str(path)])
    assert worker_node.main() == 0
    assert "the model is unavailable" in (run / "reviews/early_asks_check-missing.txt").read_text()
    code, out, err = _gate(run, monkeypatch)
    assert code == 0 and T._label(out) == "reread", err
    assert "early-asks.json was not written (the judge failed: RuntimeError: the model is unavailable)" in out
    assert "## Your last verdict could not be read" in T._facts(run)["early_asks_reread"]
    # Asked again: the fork runs again and the self-check keeps its result.
    code, _, err = T._node(run, f"{design_ops.__name__}:first_self_check", "first_self_check",
                           "op", monkeypatch)
    assert code == 0 and "self-check on attempt" not in err
    assert worker_node.main() == 0
    code, out, err = _gate(run, monkeypatch)
    assert code == 0 and T._label(out) == "proceed", err
    assert "refactory: run warning: the early asks judge failed" in err
    [record] = T._records(run)
    assert record["judge"]["status"] == "failed" and record["outcome"] == "pass"


# ── correction 3 and dm-270: what the builder is told ─────────────────────────


def test_the_fixup_carries_the_corrections_whole_and_the_brief_to_6000_characters(tmp_path, monkeypatch):
    """Corrections of about 3,400 characters arrive whole, a 5,904-character brief whole, and
    each once; a longer brief is cut at 6,000 with a note."""
    T._Chunks(monkeypatch, fails={PIN})
    run = T._spine_run(T._target(tmp_path / "repo", PINNED), tmp_path, monkeypatch, "abcdef700050")
    corrections = "\n".join(f"- correction line {n} " + "x" * 90 for n in range(1, 31))
    brief = "B" * 5900 + " END"
    (run / "facts.json").write_text(json.dumps({**T._facts(run), "brief": brief,
                                                "gate_corrections": corrections}))
    T._emit(run, BREAKS, monkeypatch)
    judge = early_asks_judgment(BUILT, {**BUILT, "id": "correction-1", "kind": "correction"})
    code, out, err = T._prepare(run, monkeypatch, judge=judge)
    assert code == 0 and "self-check fix-up 1 of 2" in out, err
    validation = T._facts(run)["validation"]
    assert len(corrections) > 3000 and f"## Approved corrections\n{corrections}" in validation
    assert f"## Brief\n{brief}" in validation and "(truncated)" not in validation
    assert validation.count("## Approved corrections") == 1 and validation.count("## Brief") == 1
    assert "name it in your completion message as `Updated tests:" in validation

    facts = T._facts(run)
    long_brief = "A" * 6000 + "TAIL"
    ctx = type("Ctx", (), {"facts": {**facts, "brief": long_brief}})()
    block = design_ops._brief_and_corrections_block(ctx)
    assert "A" * 6000 + "\n(truncated)" in block and "TAIL" not in block


class _Builder(FakeModel):
    """A builder that makes its scripted emission's changes in its workspace, as every
    `FakeModel` does (the runtime derives emit.txt from them), and answers with a
    completion message of its own: ``said``, one per emission."""

    def __init__(self, answers: list[str], said: list[str]) -> None:
        super().__init__(answers)
        self.said = list(said)

    def complete(self, prompt, *, system="", workdir=None, **workspace):
        answer = super().complete(prompt, system=system, workdir=workdir, **workspace)
        if EMIT in prompt:
            assert workspace.get("cwd") is not None, "design's builder works in its workspace"
            return self.said.pop(0) if self.said else "Done."
        return answer


def test_the_reviewers_and_the_qa_judge_see_the_tests_the_builder_says_it_updated(tmp_path, monkeypatch):
    """dm-270: the builder names, in its completion message, each test it updated and the
    change it follows. The code reviewer and the QA director read that list as they start,
    and the QA judge reads it, beside the test files that failed and changed since, under
    the rule that a test weakened to pass is a must-fix."""
    T._Chunks(monkeypatch, emissions=[{PIN}, set()])
    line = "tests/test_pin.py::test_pin (follows ask 1: the version is 2)"
    model = _Builder(T._answers(BREAKS, FIXES_PIN),
                     ["Changed the version.", f"Fixed the pinned test.\nUpdated tests: {line}\n"])
    run = tmp_path / "run"
    result = T._design(run, model, T._target(tmp_path / "repo", PINNED))
    assert result.exit_ok, result.error
    assert result.facts["updated_tests"] == f"- {line}"
    for reviewer in ("You are the code reviewer.", "You are the QA director."):
        [prompt] = [prompt for prompt in model.prompts if reviewer in prompt]
        assert f"## Tests the builder says it updated\n- {line}\n" in prompt
    synthesis = (run / "reviews/qa-synthesized.md").read_text().split("## Updated tests", 1)[1]
    assert f"- {line}" in synthesis
    assert "failed in a self-check and were changed after it:\n- tests/test_pin.py" in synthesis
    assert "A test weakened to pass" in synthesis
    [judge] = [prompt for prompt in model.prompts if "You are the QA judge." in prompt]
    assert "A test that was updated to pass after failing" in judge
    # The emit prompt asks for the line, in the completion message the builder already gives.
    assert "end the message with one line: `Updated tests: <test id>" in _emits(model)[0]


def test_a_builder_that_names_no_test_leaves_the_reviewers_told_so(tmp_path, monkeypatch):
    T._Chunks(monkeypatch, fails=set())
    model = _Builder(T._answers(CLEAN), ["Done. Updated tests: none"])
    result = T._design(tmp_path / "run", model, T._target(tmp_path / "repo", PINNED))
    assert result.exit_ok, result.error
    assert result.facts["updated_tests"] == "(the builder named no test it updated)"
    assert "## Updated tests" not in (tmp_path / "run/reviews/qa-synthesized.md").read_text()


# ── ask 1: a fix-up runs the last failures and what the fix-up changed ────────


def test_a_fixups_self_check_runs_the_last_failures_and_what_the_fixup_changed(tmp_path, monkeypatch):
    """Not the whole attempt's diff: the first self-check ran everything already. The
    fix-up changed only test_pin.py, so test_other.py, which imports the tool the attempt
    changed, is not selected again (the attempt-wide selection held it)."""
    other = "import tool\n\n\ndef test_other():\n    assert tool.VERSION\n"
    T._Chunks(monkeypatch, emissions=[{PIN}, set()])
    model = FakeModel(T._answers(BREAKS, FIXES_PIN))
    run = tmp_path / "run"
    result = T._design(run, model, T._target(tmp_path / "repo", {**PINNED, "tests/test_other.py": other}))
    assert result.exit_ok, result.error
    first, second = T._records(run)
    assert sorted(first["selection"]) == ["tests/test_other.py", "tests/test_pin.py"]
    assert second["selection"] == [PIN, "tests/test_pin.py"]
    assert second["how_selected"]["fixup_changed"] == ["tests/test_pin.py"]
    assert second["how_selected"]["fixup_changed_since"] == first["commit"][:12]
