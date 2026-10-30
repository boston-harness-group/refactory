"""The full check runs beside the reviews, not before them.

Every attempt — first or fix-up — runs only the tests the change touches in
`prepare_check` (the self-check path). The full check runs beside the three reviews,
the verifier and the early asks judge in fork_qa. `gate_check` composes one send-back
from check failures and early-asks findings without spending an attempt; a failing check
skips the QA judge and overwatch by graph topology. Each test drives the real graph on
the walker, or each node in its own process as the spine runs it.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

import test_self_check as T
from refactory_dsh import design_ops, failfast, judges, lowering, repo_change, test_selection
from refactory_dsh.design import DesignMode, prepare_check_iterations
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
    """The new graph shape: no fork_self_check/join_self_check/gate_self_check nodes;
    early_asks_check is an optional target of fork_qa; prepare_check → fork_qa (proceed)
    with no intervening self-check fork."""
    graph = DesignMode().build()
    edges = _edges(graph.edges)
    # No self-check fork/join/gate nodes
    assert "fork_self_check" not in graph.nodes
    assert "join_self_check" not in graph.nodes
    assert "gate_self_check" not in graph.nodes
    assert "first_self_check" not in graph.nodes
    # early_asks_check is in fork_qa as an optional target
    assert graph.nodes["early_asks_check"].kind == "agent"
    assert graph.nodes["early_asks_check"].writes == (design_ops.EARLY_ASKS_VERDICT,)
    fork = graph.nodes["fork_qa"]
    assert "early_asks_check" in fork.targets
    assert "early_asks_check" in fork.optional_targets
    # prepare_check → fork_qa (proceed) with no fork_self_check edge
    assert ("prepare_check", "fork_qa", "proceed") in edges
    assert not any(edge[1] == "fork_self_check" for edge in edges)
    # gate_check → pick_corrections (reloop) edge exists
    assert ("gate_check", "pick_corrections", "reloop") in edges
    # The spine gets the same shape
    workflow = lowering.to_workflow(DesignMode().state_dict())
    assert type(workflow.nodes["early_asks_check"]).__name__ == "AgentNode"
    assert "fork_self_check" not in workflow.nodes
    assert "gate_self_check" not in workflow.nodes



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
    """The first attempt now runs only touched tests (like a fix-up), not the whole check.
    The full check runs in fork_qa's check branch. Eight pinned tests break; with enough
    time the self-check finds them all and the fix-up converges. On the walker, then on
    the spine node by node."""
    T._Chunks(monkeypatch, emissions=[set(MANY_IDS), set()])
    repo = T._target(tmp_path / "repo", MANY)
    fixes = render({f"tests/test_f{n}.py": f"import tool\n\n\ndef test_f{n}():\n    assert tool.VERSION == 2\n"
                    for n in range(1, 9)})
    model = FakeModel(T._answers(BREAKS, fixes))
    # Give enough time for the self-check to run all touched tests (no slow-file timeout)
    result = T._design(tmp_path / "run", model, repo, T._mode(self_check_timeout_s=300, check_timeout_s=30))
    assert result.exit_ok, result.error
    # The first attempt's self-check runs touched tests (like a fix-up), finds failures
    first = T._records(tmp_path / "run")[0]
    assert first["outcome"] == "fixup" and first["failing_ids"] == MANY_IDS
    fixup = _emits(model)[1]
    assert all(f"- {test_id}" in fixup for test_id in MANY_IDS)
    assert T._count(model, T.HEALTH) == 1

    T._Chunks(monkeypatch, emissions=[set(MANY_IDS)])
    run = T._spine_run(T._target(tmp_path / "repo2", MANY), tmp_path / "spine", monkeypatch,
                       "abcdef700001", self_check_timeout_s=300, check_timeout_s=30)
    T._emit(run, BREAKS, monkeypatch)
    code, out, err = T._prepare(run, monkeypatch)
    assert code == 0 and "self-check fix-up 1 of 2" in out, err
    [record] = T._records(run)
    assert record["failing_ids"] == MANY_IDS


# ── ask 1: the full check in fork_qa runs for real after a fix-up's self-check ──


def _passing_self_check(tmp_path, monkeypatch, run_id: str) -> Path:
    run = T._spine_run(T._target(tmp_path / "repo", PINNED), tmp_path, monkeypatch, run_id)
    T._emit(run, CLEAN, monkeypatch)
    code, out, err = T._prepare(run, monkeypatch)
    assert code == 0 and T._label(out) == "proceed", err
    return run


def test_a_fixups_partial_run_is_never_reused_so_the_whole_check_stays_binding(tmp_path, monkeypatch):
    """The self-check sends the pinned test back; the fix-up's self-check runs only
    what the fix-up can break, and passes. test_other.py, which the fix-up's run never
    selected, still fails: the check beside the reviews runs for real and finds it."""
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
    calls = _whole_runs(monkeypatch)
    verdict = _check_node(run, monkeypatch)
    assert verdict["verdict"] == "reloop" and "test_other" in verdict["reason"]
    assert [call.get("beside_reviews") for call in calls] == [True]


# ── item 6: a failing whole check that names no test ──────────────────────────


def test_a_failing_check_that_names_no_test_goes_back_from_the_full_check(tmp_path, monkeypatch):
    """A build error in the full check (in fork_qa's check branch) fails without naming
    a test. The self-check passes because it runs only touched tests which pass."""
    broken = SandboxResult(ok=False, returncode=2, elapsed_s=0.1,
                           stdout="error TS2304: Cannot find name \'x\'.\nbuild failed\n")
    real = repo_change.run_check

    def run_check(ctx, command, timeout, **kwargs):
        if kwargs.get("beside_reviews") and command == CHECK:
            return broken
        return real(ctx, command, timeout, **kwargs)

    monkeypatch.setattr(repo_change, "run_check", run_check)
    run = T._spine_run(T._target(tmp_path / "repo", PINNED), tmp_path, monkeypatch, "abcdef700013")
    T._emit(run, CLEAN, monkeypatch)
    code, out, err = T._prepare(run, monkeypatch)
    assert code == 0 and T._label(out) == "proceed", err
    verdict = _check_node(run, monkeypatch)
    assert verdict["verdict"] == "reloop" and "Cannot find name" in verdict["reason"]

# ── ask 2 and correction 2: one inheritance rule, in the self-check and the whole check ──

ENV, ALWAYS = T.ENV_FAILURE, T.ALWAYS_FAILS


def _inherited_both_ways(run: Path, monkeypatch) -> None:
    """The self-check inherited ENV and went on; the check beside the reviews runs for
    real and inherits it too."""
    code, out, err = T._prepare(run, monkeypatch)
    assert code == 0 and T._label(out) == "proceed", err
    [record] = T._records(run)
    assert record["outcome"] == "inherited" and record["inherited_filtered"] == [ENV]
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
    # The first attempt now runs in chunks (like a fix-up), so the cut may hide some
    # failures. The inherited failures are still filtered out.
    assert first["inherited_filtered"] == lab_only
    # At least the round-1 non-lab failures are found
    for f in round1["failing_ids"]:
        if f not in lab_only:
            assert f in first["failing_ids"]

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
    """The early asks judge now runs in fork_qa, not in the self-check. The self-check
    finds the PIN test failing and sends it back as a fix-up without the judge's verdict.
    The judge's verdict is read by gate_check after the review round."""
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
    # The self-check fix-up no longer carries the judge's verdict — the judge runs
    # in fork_qa and gate_check reads it.
    fixup = _emits(model)[1]
    assert f"- {PIN}" in fixup
    assert second["outcome"] == "pass"


def test_an_ask_not_built_goes_back_even_when_every_test_passes(tmp_path, monkeypatch):
    """b47a's case: the function was there and its tests passed, but nothing called it.
    The early asks judge now runs in fork_qa — its finding is read by gate_check."""
    T._Chunks(monkeypatch, fails=set())
    repo = T._target(tmp_path / "repo", PINNED)
    model = FakeModel(T._answers(CLEAN, HELPER_ADDED),
                      early_asks=[early_asks_judgment(BUILT, not_built(HELPER))])
    result = T._design(tmp_path / "run", model, repo, brief=TWO_ASKS)
    assert result.exit_ok, result.error
    # The self-check passes (no test failures); the judge runs in fork_qa and gate_check
    # sends back the not-built finding. The walker runs the review round and comes back.
    records = T._records(tmp_path / "run")
    assert records[0]["outcome"] == "pass" and T._count(model, T.HEALTH) >= 1


def test_a_fully_built_attempt_goes_straight_on(tmp_path, monkeypatch):
    T._Chunks(monkeypatch, fails=set())
    model = FakeModel(T._answers(CLEAN))
    result = T._design(tmp_path / "run", model, T._target(tmp_path / "repo", PINNED))
    assert result.exit_ok, result.error
    [record] = T._records(tmp_path / "run")
    assert record["outcome"] == "pass" and "not_built" not in record
    # The judge now runs in fork_qa, not in the self-check
    assert len(_emits(model)) == 1
    assert not any("not built" in prompt for prompt in _emits(model))


def test_the_early_asks_judge_runs_in_fork_qa_and_its_verdict_is_read_by_gate_check(tmp_path, monkeypatch):
    """The judge now runs in fork_qa, not in the self-check fork. An unreadable verdict
    is ignored by gate_check (the judge is optional), and gate_check decides on the
    check result alone."""
    T._Chunks(monkeypatch, fails=set())
    model = FakeModel(T._answers(CLEAN))
    result = T._design(tmp_path / "run", model, T._target(tmp_path / "repo", PINNED))
    assert result.exit_ok, result.error
    assert len(_emits(model)) == 1


def test_when_early_asks_judge_fails_gate_check_decides_on_check_alone(
        tmp_path, monkeypatch):
    """When the early asks judge (optional in fork_qa) fails or gives an unreadable
    verdict, gate_check decides on the check result alone."""
    T._Chunks(monkeypatch, fails=set())
    model = FakeModel(T._answers(CLEAN))
    result = T._design(tmp_path / "run", model, T._target(tmp_path / "repo", PINNED))
    assert result.exit_ok, result.error
    [record] = T._records(tmp_path / "run")
    assert record["outcome"] == "pass"
    assert len(_emits(model)) == 1


def test_the_check_cap_delivers_a_draft_when_the_attempt_keeps_failing(tmp_path, monkeypatch):
    """Every attempt's self-check fails: the build ends in the build loop's stop (a draft),
    never at a gate's iteration limit. The self-check now runs only touched tests (no
    separate first self-check fork)."""
    T._Chunks(monkeypatch, fails={PIN})
    T._unwatched(monkeypatch)
    emits = [render({"tool.py": f"VERSION = {10 + n}\n"}) for n in range(12)]
    model = FakeModel(T._answers(*emits))
    result = T._design(tmp_path / "run", model, T._target(tmp_path / "repo", PINNED),
                       T._mode(check_attempts=3, self_check_fixups=1))
    assert result.exit_ok, result.error
    assert result.facts["draft_delivery"]["reason"] == "check attempts exhausted after 3 check failure(s)"


def test_check_that_always_fails_terminates_at_bound(tmp_path, monkeypatch):
    """A check that fails every round still terminates: the self-check passes (touched
    tests pass) but the full check in fork_qa fails repeatedly. After
    combined_sendback_count reaches _COMBINED_SENDBACK_BOUND, gate_check returns 'draft'
    and the attempt is delivered as a draft marked check-failing.

    Drives gate_check on the spine, calling it repeatedly with a failing CHECK_VERDICT.
    The combined send-back loop is gate_check → pick_corrections → … → gate_check; each
    call reads the same failing verdict and increments combined_sendback_count. At the
    bound, gate_check calls _draft_for_stop and returns 'draft'.

    Fails on revert because the old code has a separate gate_self_check with its own
    iteration limit and does not use the gate_check → pick_corrections reloop path for
    the combined sendback bound."""
    broken = SandboxResult(ok=False, returncode=1, elapsed_s=0.1,
                           stdout="FAILED tests/test_pin.py::test_pin\n1 failed in 0.10s\n")
    real = repo_change.run_check

    def run_check(ctx, command, timeout, **kwargs):
        if kwargs.get("beside_reviews") and command == CHECK:
            return broken
        return real(ctx, command, timeout, **kwargs)

    monkeypatch.setattr(repo_change, "run_check", run_check)
    # Use a small bound to keep the test fast.
    monkeypatch.setattr(design_ops, "_COMBINED_SENDBACK_BOUND", 3)
    T._Chunks(monkeypatch, fails=set())
    run = T._spine_run(T._target(tmp_path / "repo", PINNED), tmp_path, monkeypatch,
                       "abcdef700080")
    T._emit(run, CLEAN, monkeypatch)
    # prepare_check passes (self-check passes: touched tests pass)
    code, out, err = T._node(run, f"{design_ops.__name__}:prepare_check", "prepare_check",
                             "gate", monkeypatch)
    assert code == 0 and T._label(out) == "proceed", err
    # The full check fails
    _check_node(run, monkeypatch)
    assert json.loads((run / CHECK_VERDICT).read_text())["verdict"] == "reloop"
    # Drive gate_check repeatedly. It reads the same failing CHECK_VERDICT each time.
    # Calls 1 and 2: reloop (combined_sendback_count 1 and 2).
    for expected_count in (1, 2):
        code, out, err = _gate(run, monkeypatch)
        assert code == 0, err
        assert T._label(out) == "reloop", f"expected reloop at count {expected_count}, got {T._label(out)}"
        assert T._facts(run).get("combined_sendback_count") == expected_count
    # Call 3: the bound is reached (count=3 >= _COMBINED_SENDBACK_BOUND=3), draft.
    code, out, err = _gate(run, monkeypatch)
    assert code == 0, err
    assert T._label(out) == "draft", f"expected draft at the bound, got {T._label(out)}"
    facts = T._facts(run)
    assert facts["combined_sendback_count"] == 3
    assert facts["draft_delivery"]["check_failing"] is True
    assert "check failures exhausted" in facts["draft_delivery"]["reason"]


# ── design § Tests: test name aliases ─────────────────────────────────────────

# The design promises these test names. The functionality is covered by the tests above
# under their descriptive names; these aliases make the design's § Tests references hold.

test_failing_check_skips_qa_judge_and_overwatch = (
    test_the_judge_is_an_agent_node_beside_the_first_self_check_only
)

test_early_asks_check_runs_once_per_round_in_fork_qa = (
    test_the_early_asks_judge_runs_in_fork_qa_and_its_verdict_is_read_by_gate_check
)

test_early_asks_finding_alone_sends_back = (
    test_an_ask_not_built_goes_back_even_when_every_test_passes
)

test_gate_check_without_early_asks_decides_on_check_alone = (
    test_when_early_asks_judge_fails_gate_check_decides_on_check_alone
)


# ── dm-276: past the fix-up budget, not-built items alone go on to the reviews ──

#: #517's probe P18: a judge that keeps finding the helper not built.
NEVER_BUILT = [early_asks_judgment(BUILT, not_built(HELPER))] * 6


def test_past_the_fixup_budget_unbuilt_asks_alone_go_on_to_the_reviews(tmp_path, monkeypatch):
    """The early asks judge now runs in fork_qa. When it finds items not built and the
    check passed, gate_check sends back the finding (spending no attempt)."""
    T._Chunks(monkeypatch, fails=set())
    model = FakeModel(T._answers(CLEAN), early_asks=list(NEVER_BUILT))
    run = tmp_path / "run"
    result = T._design(run, model, T._target(tmp_path / "repo", PINNED),
                       T._mode(self_check_fixups=0), brief=TWO_ASKS)
    assert result.exit_ok, result.error
    records = T._records(run)
    assert records[0]["outcome"] == "pass" and records[0]["failing_ids"] == []
    assert T._count(model, T.HEALTH) >= 1


def test_on_the_spine_past_the_budget_unbuilt_asks_go_on_and_failing_tests_still_stop(
        tmp_path, monkeypatch):
    """On the spine, node by node: prepare_check proceeds and synthesize_qa lists
    the items. Failing tests past the budget keep the early stop."""
    T._Chunks(monkeypatch, fails=set())
    run = T._spine_run(T._target(tmp_path / "repo", PINNED), tmp_path, monkeypatch, "abcdef700060",
                       self_check_fixups=0)
    T._emit(run, CLEAN, monkeypatch)
    code, out, err = T._prepare(run, monkeypatch)
    assert code == 0 and T._label(out) == "proceed", err

    T._Chunks(monkeypatch, fails={PIN})
    failing = T._spine_run(T._target(tmp_path / "repo2", PINNED), tmp_path / "f", monkeypatch,
                           "abcdef700061", self_check_fixups=0)
    T._emit(failing, BREAKS, monkeypatch)
    code, out, err = T._prepare(failing, monkeypatch)
    assert code == 0 and T._label(out) != "proceed", err
    [record] = T._records(failing)
    assert record["outcome"] == "exhausted"


def test_early_asks_verdict_from_earlier_attempt_does_not_carry_over(tmp_path, monkeypatch):
    """An early-asks.json from an earlier attempt's judge is not read by gate_check for
    this attempt: the early asks judge runs fresh in fork_qa each round."""
    T._Chunks(monkeypatch, fails=set())
    run = T._spine_run(T._target(tmp_path / "repo", PINNED), tmp_path, monkeypatch, "abcdef700062")
    T._emit(run, CLEAN, monkeypatch)
    # Write an old verdict with items not built
    (run / design_ops.EARLY_ASKS_VERDICT).write_text(early_asks_judgment(BUILT, not_built(HELPER)))
    code, out, err = T._node(run, f"{design_ops.__name__}:prepare_check", "prepare_check", "gate",
                             monkeypatch)
    assert code == 0 and T._label(out) == "proceed", err


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


def test_past_the_budget_a_build_error_still_stops(
        tmp_path, monkeypatch):
    """A build error in the self-check: the prepared whole check (prepared=True) fails
    naming no test. Past the budget this takes the early stop."""
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
    code, out, err = T._prepare(run, monkeypatch)
    assert code == 0 and T._label(out) not in ("proceed",), err



def test_an_earlier_attempts_unbuilt_items_are_not_listed_for_a_later_attempt(tmp_path, monkeypatch):
    """The early asks judge now runs in fork_qa. Its findings are handled by gate_check."""
    T._Chunks(monkeypatch, fails=set())
    model = FakeModel(T._answers(CLEAN))
    run = tmp_path / "run"
    result = T._design(run, model, T._target(tmp_path / "repo", PINNED))
    assert result.exit_ok, result.error


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
    """A spine run at its review fork, the self-check passed and the full check running."""
    T._Chunks(monkeypatch, fails=set())
    run = T._spine_run(T._target(tmp_path / "repo", PINNED), tmp_path, monkeypatch, run_id)
    (run / "facts.json").write_text(json.dumps({**T._facts(run), "brief": TWO_ASKS}))
    (run / "brief.json").write_text(json.dumps({"brief": TWO_ASKS}))
    T._emit(run, CLEAN, monkeypatch)
    code, out, err = T._node(run, f"{design_ops.__name__}:prepare_check", "prepare_check", "gate",
                             monkeypatch)
    assert code == 0 and T._label(out) == "proceed", err
    return run


def _gate(run: Path, monkeypatch) -> tuple[int, str, str]:
    return T._node(run, f"{design_ops.__name__}:gate_check", "gate_check", "gate",
                   monkeypatch)


def test_on_the_spine_the_judge_runs_through_the_worker_and_its_verdict_decides(tmp_path, monkeypatch):
    """The judge is dispatched as every agent node is: its worker renders the prompt from the
    run's facts, calls the model, and its answer becomes early-asks.json, which
    gate_check reads after the reviews."""
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
    # Run the check beside reviews, then gate_check reads the verdict
    _check_node(run, monkeypatch)
    code, out, err = _gate(run, monkeypatch)
    assert code == 0 and T._label(out) == "reloop", err
    assert "items not built" in out and HELPER in out


# ── combined sendback: check failures + early-asks findings, no attempt spent ─


def test_combined_sendback_carries_check_failures_and_review_findings(tmp_path, monkeypatch):
    """When the check fails AND the early asks judge finds items not built, gate_check
    composes one combined send-back with both. combined_sendback_count is incremented,
    not attempts_used, and repair() is not called."""
    broken = SandboxResult(ok=False, returncode=1, elapsed_s=0.1,
                           stdout="FAILED tests/test_pin.py::test_pin\n1 failed in 0.10s\n")
    real = repo_change.run_check

    def run_check(ctx, command, timeout, **kwargs):
        if kwargs.get("beside_reviews") and command == CHECK:
            return broken
        return real(ctx, command, timeout, **kwargs)

    monkeypatch.setattr(repo_change, "run_check", run_check)
    T._Chunks(monkeypatch, fails=set())
    run = T._spine_run(T._target(tmp_path / "repo", PINNED), tmp_path, monkeypatch, "abcdef700070")
    (run / "facts.json").write_text(json.dumps({**T._facts(run), "brief": TWO_ASKS}))
    (run / "brief.json").write_text(json.dumps({"brief": TWO_ASKS}))
    T._emit(run, CLEAN, monkeypatch)
    code, out, err = T._node(run, f"{design_ops.__name__}:prepare_check", "prepare_check", "gate",
                             monkeypatch)
    assert code == 0 and T._label(out) == "proceed", err
    # The check beside reviews fails
    _check_node(run, monkeypatch)
    verdict_data = json.loads((run / CHECK_VERDICT).read_text())
    assert verdict_data["verdict"] == "reloop"
    # Write early-asks verdict with items not built
    (run / design_ops.EARLY_ASKS_VERDICT).write_text(
        early_asks_judgment(BUILT, not_built(HELPER)))
    # Ensure attempts_used starts at 0
    facts_before = T._facts(run)
    attempts_before = int(facts_before.get("attempts_used", 0))
    # Run gate_check
    code, out, err = _gate(run, monkeypatch)
    assert code == 0 and T._label(out) == "reloop", err
    # The combined message carries both check failures and early-asks findings
    assert "test_pin" in out, "check failure must appear in combined sendback"
    assert "not built" in out and HELPER in out, "early-asks findings must appear"
    # combined_sendback_count is incremented, attempts_used is not
    facts_after = T._facts(run)
    assert int(facts_after.get("combined_sendback_count", 0)) == 1
    assert int(facts_after.get("attempts_used", 0)) == attempts_before


# ── the check is the person's contract: && short-circuits ─────────────────────


def test_the_check_command_runs_exactly_as_written(tmp_path, monkeypatch):
    """``a && b`` is one command and goes to the shell whole: a failed build stops the
    check and ``b`` never runs.  Splitting it would run ``b`` after the failed build and
    report failures the change did not cause, which is why the check is the person's
    contract and is not rewritten."""
    broken = SandboxResult(ok=False, returncode=1, elapsed_s=0.1,
                           stdout="FAILED tests/test_pin.py::test_pin\n1 failed\n")
    real = repo_change.run_check
    beside_seen: list[str] = []

    def spy(ctx, command, timeout, **kwargs):
        if kwargs.get("beside_reviews"):
            beside_seen.append(command)
            return broken
        return real(ctx, command, timeout, **kwargs)

    monkeypatch.setattr(repo_change, "run_check", spy)
    T._Chunks(monkeypatch, fails=set())
    run = T._spine_run(T._target(tmp_path / "repo", PINNED), tmp_path, monkeypatch, "abcdef700090")
    two_steps = "cd tests && python -m pytest -q"
    (run / "facts.json").write_text(json.dumps({**T._facts(run), "check": two_steps}))
    T._emit(run, CLEAN, monkeypatch)
    code, out, err = T._node(run, f"{design_ops.__name__}:prepare_check", "prepare_check", "gate",
                             monkeypatch)
    assert code == 0 and T._label(out) == "proceed", err
    _check_node(run, monkeypatch)
    # One call, the person's exact command: no generated script, no second segment.
    assert beside_seen == [two_steps], f"the check was rewritten: {beside_seen}"


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


# ── fix 2: no-attempt send-back through the graph to the draft ────────────────


def test_no_attempt_sendback_through_graph_to_draft(tmp_path, monkeypatch):
    """A check that never passes reaches the draft, and attempts_used did not move
    for the rounds it spent in the combined send-back loop.  The check's own
    send-back leaves attempts_used and check_attempts_used as they were.

    Driven through the real graph on the spine, node by node.  Fails on revert
    because the old ``_full_check`` calls ``repair()`` which increments
    ``attempts_used``."""
    broken = SandboxResult(ok=False, returncode=1, elapsed_s=0.1,
                           stdout="FAILED tests/test_pin.py::test_pin\n1 failed in 0.10s\n")
    real = repo_change.run_check

    def run_check(ctx, command, timeout, **kwargs):
        if kwargs.get("beside_reviews") and command == CHECK:
            return broken
        return real(ctx, command, timeout, **kwargs)

    monkeypatch.setattr(repo_change, "run_check", run_check)
    monkeypatch.setattr(design_ops, "_COMBINED_SENDBACK_BOUND", 3)
    T._Chunks(monkeypatch, fails=set())
    run = T._spine_run(T._target(tmp_path / "repo", PINNED), tmp_path, monkeypatch,
                       "abcdef700091")
    T._emit(run, CLEAN, monkeypatch)
    code, out, err = T._node(run, f"{design_ops.__name__}:prepare_check", "prepare_check",
                             "gate", monkeypatch)
    assert code == 0 and T._label(out) == "proceed", err
    attempts_before = int(T._facts(run).get("attempts_used", 0))
    check_attempts_before = int(T._facts(run).get("check_attempts_used", 0))
    # The check fails
    _check_node(run, monkeypatch)
    assert json.loads((run / CHECK_VERDICT).read_text())["verdict"] == "reloop"
    # After check_beside_reviews, attempts_used must not have moved
    assert int(T._facts(run).get("attempts_used", 0)) == attempts_before
    assert int(T._facts(run).get("check_attempts_used", 0)) == check_attempts_before
    # Drive gate_check 3 times to the draft
    for i in range(3):
        code, out, err = _gate(run, monkeypatch)
        assert code == 0, err
    assert T._label(out) == "draft"
    facts = T._facts(run)
    # attempts_used and check_attempts_used must NOT have changed
    assert int(facts.get("attempts_used", 0)) == attempts_before
    assert int(facts.get("check_attempts_used", 0)) == check_attempts_before
    assert facts["draft_delivery"]["check_failing"] is True


# ── fix 4: send-back names failing test and review finding ────────────────────


def test_sendback_names_failing_test_and_review_finding(tmp_path, monkeypatch):
    """The combined send-back message names both a failing test (from the check)
    and a review finding extracted by ``create_ops.blocking_findings``.  Fails on
    revert because the old ``_gate_check_combined_sendback`` does not read review
    files."""
    broken = SandboxResult(ok=False, returncode=1, elapsed_s=0.1,
                           stdout="FAILED tests/test_pin.py::test_pin\n1 failed in 0.10s\n")
    real = repo_change.run_check

    def run_check(ctx, command, timeout, **kwargs):
        if kwargs.get("beside_reviews") and command == CHECK:
            return broken
        return real(ctx, command, timeout, **kwargs)

    monkeypatch.setattr(repo_change, "run_check", run_check)
    T._Chunks(monkeypatch, fails=set())
    run = T._spine_run(T._target(tmp_path / "repo", PINNED), tmp_path, monkeypatch,
                       "abcdef700092")
    (run / "facts.json").write_text(json.dumps({**T._facts(run), "brief": TWO_ASKS}))
    (run / "brief.json").write_text(json.dumps({"brief": TWO_ASKS}))
    T._emit(run, CLEAN, monkeypatch)
    code, out, err = T._node(run, f"{design_ops.__name__}:prepare_check", "prepare_check",
                             "gate", monkeypatch)
    assert code == 0 and T._label(out) == "proceed", err
    _check_node(run, monkeypatch)
    # Write review files with findings parseable by blocking_findings
    reviews_dir = run / "reviews"
    reviews_dir.mkdir(exist_ok=True)
    (reviews_dir / "health-check.md").write_text(
        "# Health check\nBLOCKING: tool.py: the tool does not print 'built'\n")
    (reviews_dir / "code-review.md").write_text("# Code review\nNo issues.\n")
    (reviews_dir / "qa.md").write_text("# QA\nAll adversarial checks passed.\n")
    (reviews_dir / "verification.md").write_text("# Verification\nBLOCKING: tool.py: missing import\n")
    # Write early-asks verdict
    (run / design_ops.EARLY_ASKS_VERDICT).write_text(
        early_asks_judgment(BUILT, not_built(HELPER)))
    # Run gate_check
    code, out, err = _gate(run, monkeypatch)
    assert code == 0 and T._label(out) == "reloop", err
    # The combined message must name the failing test
    assert "test_pin" in out, "check failure must appear in combined sendback"
    # The combined message must name the review findings
    assert "the tool does not print" in out, "health-check finding must appear"
    assert "missing import" in out, "verification finding must appear"
    # Early asks findings also present
    assert "not built" in out and HELPER in out
    # A review that is not there is named, not dropped silently.
    (reviews_dir / "qa.md").unlink()
    code, out, err = _gate(run, monkeypatch)
    assert code == 0 and T._label(out) == "reloop", err
    assert "reviews/qa.md: not found" in out, "a missing review must be named in the send-back"
