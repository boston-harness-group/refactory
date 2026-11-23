"""Finishing a pull request: its Open items, and a truncated refine that works on them.

The format (`open_items`) round-trips and takes a person's edits; a finish (`finish`)
selects its items, skips the approval gate only when every item the builder gets is
concrete, never shows an ``environment`` item to design_doc or the builder, gets one fix
round across every gate that sends a change back, and writes the items back onto the
same pull request. The runs here are the walker's, against the fake GitHub of
`test_delivery`; `test_finish_runtimes.py` runs a finish through both runtimes' real
entry points.
"""
from __future__ import annotations

import json

import pytest

from refactory_dsh import delivery, design_ops, finish, open_items, worktree
from refactory_dsh.nn.executor import RunContext
from refactory_dsh.open_items import OpenItem
from refactory_dsh.run import BuildLoopStop
from refactory_dsh.steering import Ledger
import delivery_fakes
from _delivery_support import (  # noqa: F401  (_stage, _opened_golden and opened_pull are the fake GitHub's fixtures)
    BUILT,
    QA_HELD,
    QA_JUDGE_MARK,
    _attempt,
    _child,
    _child_patches,
    _describe,
    _mode,
    _opened_golden,
    _pulls,
    _record,
    _remote_head,
    _stage,
    opened_pull,
)
from tests_support import DESIGN_DOC_ANSWER, FakeModel, qa_judgment, render

QA = OpenItem("qa-1", "qa_finding", "tool.py prints one where the brief asks for two",
              ("tool.py:1", "tests/test_tool.py::test_prints_two"))
ENV = OpenItem("env-1", "environment", "ENV-ONLY-SUMMARY: the lab runner timed out",
               ("log: tests/test_slow.py::test_slow timed out after 600 s",))
JUDGMENT = OpenItem("jc-1", "judgment_call", "whether more.py belongs in this pull request",
                    ("the excuse rule used: out of scope for the brief",))
UNFINISHED = OpenItem("ask-1", "unfinished_ask", "the --two flag is not built", ("brief: 'add --two'",),
                      status="resolved")

BRIEF = "Finish the pull request's open items."
WRITE_ARTIFACT = "Write the artifact `"
DESIGN_DOC = "Write the design document"


# ── the format ────────────────────────────────────────────────────────────────
def test_render_and_parse_round_trip():
    items = [QA, ENV, JUDGMENT, UNFINISHED]
    text = open_items.render(items)
    assert text.startswith("## Open items\n\n- [ ] **qa-1** (qa_finding): tool.py prints one")
    assert "\n- [x] **ask-1** (unfinished_ask): the --two flag is not built\n  - brief: 'add --two'\n" in text
    assert open_items.parse(text) == items
    assert open_items.render(open_items.parse(text)) == text
    # Inside a description, between other sections and delivery's markers.
    body = f"{delivery.BODY_START}\n## Summary\n\nWhy.\n\n{text}{delivery.BODY_END}\n\n## Notes\n\n- [ ] **x** (qa_finding): not an item\n"
    assert open_items.parse(body) == items
    assert open_items.parse("## Summary\n\nno items here\n") == []


def test_parse_takes_a_persons_edits():
    """A ticked box (either case), a comment between items, an extra evidence line, an HTML
    comment, Windows line ends and an unknown kind are all read as a person meant them."""
    text = open_items.render([QA, ENV, JUDGMENT])
    edited = (text.replace("- [ ] **qa-1**", "- [X] **qa-1**")
              .replace("- [ ] **env-1**", "I think this one is flaky, not ours.\n\n- [ ] **env-1**")
              .replace("  - tests/test_tool.py::test_prints_two\n",
                       "  - tests/test_tool.py::test_prints_two\n  <!-- checked by hand -->\n"
                       "    fixed locally, see my comment\n")
              .replace("(judgment_call)", "(policy_question)")
              .replace("\n", "\r\n"))
    found = open_items.parse(edited)
    assert [item.id for item in found] == ["qa-1", "env-1", "jc-1"]
    assert found[0].status == "resolved"
    assert found[0].evidence == (*QA.evidence, "fixed locally, see my comment")
    assert found[1] == ENV
    assert found[2].kind == "judgment_call"
    # A person's comment written right under an item ends it; it is never evidence.
    commented = text.replace(
        "  - tests/test_tool.py::test_prints_two\n", "  - tests/test_tool.py::test_prints_two\nnot evidence\n")
    assert open_items.parse(commented)[0].evidence == QA.evidence
    # An indented line after that comment belongs to no item.
    stray = commented.replace("not evidence\n", "not evidence\n  - stray line under the comment\n")
    assert open_items.parse(stray)[0].evidence == QA.evidence


def test_replace_keeps_everything_else_and_one_section():
    section = open_items.render([QA])
    body = f"Intro.\n\n{section}\n## Later\n\nKept.\n\n{open_items.render([ENV])}"
    replaced = open_items.replace(body, [ENV])
    assert replaced.startswith("Intro.\n\n## Open items\n\n- [ ] **env-1**")
    assert replaced.count("## Open items") == 1 and "## Later\n\nKept.\n" in replaced
    assert open_items.replace("No items yet.\n", [QA]) == f"No items yet.\n\n{section}"
    assert "## Open items" not in open_items.without(body) and "Kept." in open_items.without(body)


def test_select_defaults_to_open_items_that_are_not_judgment_calls():
    items = [QA, ENV, JUDGMENT, UNFINISHED]
    assert open_items.select(items) == [QA, ENV]
    assert open_items.select(items, "jc-1, qa-1") == [QA, JUDGMENT]
    with pytest.raises(ValueError, match="no open item nope"):
        open_items.select(items, "qa-1,nope")
    with pytest.raises(ValueError, match="ask-1 is resolved already"):
        open_items.select(items, "ask-1")


# ── the gate ──────────────────────────────────────────────────────────────────
def _fact(*items: OpenItem) -> dict:
    return {"pull_request": "https://github.com/acme/app/pull/1", "selected": [i.to_dict() for i in items],
            "builder": [i.id for i in items if i.kind != "environment"],
            "environment": [i.id for i in items if i.kind == "environment"]}


def _taken(run) -> str:
    ledger = Ledger(run)
    return ledger.take_verdict("gate_strategy", ledger.review("gate_strategy", []), ["proceed", "reloop"])


@pytest.mark.parametrize("items, skipped", [
    ((QA, ENV), True),
    ((QA, JUDGMENT), False),
    ((QA, OpenItem("cf-1", "change_failure", "the check fails on tool.py")), False),
])
def test_the_gate_is_skipped_only_when_every_item_the_builder_gets_is_concrete(tmp_path, items, skipped):
    """Concrete items, an environment item beside them included, answer the gate up front
    in the ledger, where both runtimes take it; a judgment call, or a concrete kind with no
    evidence, leaves the gate to the person."""
    Ledger.seed(tmp_path, BRIEF)
    decided = finish.decide_gate(RunContext(tmp_path, knobs={}, facts={}), _fact(*items))
    assert decided["skipped"] is skipped
    assert _taken(tmp_path) == ("proceed" if skipped else "")
    if skipped:
        assert decided["reason"].startswith("the approval gate was skipped: every item")
    else:
        assert "asked" in decided["reason"] and items[1].id in decided["reason"]


# ── one fix round ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("spent, allowed", [({}, True), ({"finish_rounds": 1}, False)])
def test_a_finish_allows_one_round_across_every_gate(tmp_path, spent, allowed):
    on = RunContext(tmp_path, knobs={}, facts={"finish": _fact(QA), **spent})
    off = RunContext(tmp_path, knobs={}, facts={**spent})
    assert finish.allows_round(on) is allowed
    assert finish.allows_round(off) is True
    assert str(finish.sent_back(on, "reloop")) == "reloop" and on.facts["finish_rounds"] == len(spent) + 1
    assert finish.sent_back(off, "reloop") == "reloop" and off.facts.get("finish_rounds") == spent.get(
        "finish_rounds")


def test_gate_check_sends_back_once_on_a_finish(tmp_path):
    """gate_check's combined send-back on a finish: the first goes back to the builder and
    is counted, one after another gate's round is a stop (a draft when the run has a commit),
    not a second round."""
    first = RunContext(tmp_path / "a", knobs={"check_attempts": 2}, facts={"finish": _fact(QA)})
    assert str(design_ops._gate_check_combined_sendback(first, "the check failed")) == "reloop"
    assert first.facts["finish_rounds"] == 1
    after = RunContext(tmp_path / "b", knobs={"check_attempts": 2},
                       facts={"finish": _fact(QA), "finish_rounds": 1})
    with pytest.raises(BuildLoopStop, match=finish.ROUND_SPENT):
        design_ops._gate_check_combined_sendback(after, "the check failed")
    plain = RunContext(tmp_path / "c", knobs={"check_attempts": 2}, facts={"review_attempts_used": 1})
    assert str(design_ops._gate_check_combined_sendback(plain, "the check failed")) == "reloop"


def test_the_early_asks_send_back_once_on_a_finish(tmp_path):
    """gate_check's send-back for an ask the early asks judge found not built, on a finish."""
    not_built = json.dumps({"items": [{"id": "ask-1", "kind": "ask", "text": "the brief", "built": False,
                                       "missing": "the --two flag"}]})

    def ctx(name, facts):
        made = RunContext(tmp_path / name, knobs={"check_attempts": 2}, facts=facts)
        made.write_text(design_ops.EARLY_ASKS_VERDICT, not_built)
        return made

    first = ctx("a", {"finish": _fact(QA)})
    assert str(design_ops._gate_check_early_asks(first)) == "reloop" and first.facts["finish_rounds"] == 1
    with pytest.raises(BuildLoopStop, match=finish.ROUND_SPENT):
        design_ops._gate_check_early_asks(ctx("b", {"finish": _fact(QA), "finish_rounds": 1}))
    assert str(design_ops._gate_check_early_asks(ctx("c", {"finish_rounds": 1}))) == "reloop"


def test_the_review_gates_send_back_once_on_a_finish(tmp_path):
    first = RunContext(tmp_path / "a", knobs={"check_attempts": 2}, facts={"finish": _fact(QA)})
    assert str(design_ops._send_back(first, "reviews/qa-gate.txt", "qa_gate", "reloop: fix it")) == "reloop"
    assert first.facts["finish_rounds"] == 1
    after = RunContext(tmp_path / "b", knobs={"check_attempts": 2},
                       facts={"finish": _fact(QA), "finish_rounds": 1})
    with pytest.raises(BuildLoopStop, match=finish.ROUND_SPENT):
        design_ops._send_back(after, "reviews/qa-gate.txt", "qa_gate", "reloop: fix it")
    plain = RunContext(tmp_path / "c", knobs={"check_attempts": 2}, facts={"combined_sendback_count": 1})
    assert str(design_ops._send_back(plain, "reviews/qa-gate.txt", "qa_gate", "reloop: fix it")) == "reloop"


def test_a_conflict_goes_back_to_the_builder_only_while_the_round_is_unspent(tmp_path):
    fresh = RunContext(tmp_path, knobs={"check_attempts": 2}, facts={"finish": _fact(QA)})
    spent = RunContext(tmp_path, knobs={"check_attempts": 2}, facts={"finish": _fact(QA), "finish_rounds": 1})
    plain = RunContext(tmp_path, knobs={"check_attempts": 2}, facts={"finish_rounds": 1})
    assert [delivery._conflict_reloops_allowed(ctx) for ctx in (fresh, spent, plain)] == [1, 0, 2]
    refused = delivery.NotDelivered("conflict", merge=delivery.MergeResult(conflicts=["tool.py"]))
    assert delivery._conflict_budget_spent(fresh, refused, None) is None
    assert delivery._conflict_budget_spent(spent, refused, None) is not None
    assert delivery._conflict_budget_spent(plain, refused, None) is None


# ── prepare ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("knobs, pull, refused", [
    ({"finish": True}, None, "finish goes with refine"),
    ({"finish_items": "qa-1"}, None, "finish_items goes with finish"),
    ({"finish": True, "check_attempts": 3}, {"body": open_items.render([QA])}, "check_attempts=3"),
    ({"finish": True, "self_check_fixups": 2}, {"body": open_items.render([QA])}, "self_check_fixups=2"),
    ({"finish": True}, {"body": "no items"}, "no ## Open items section"),
    ({"finish": True}, {"body": open_items.render([ENV])}, "env-1 is an environment item"),
    ({"finish": True}, {"body": open_items.render([JUDGMENT])}, "judgment calls need --items"),
])
def test_prepare_refuses_what_a_finish_cannot_do(tmp_path, knobs, pull, refused):
    Ledger.seed(tmp_path, BRIEF)
    with pytest.raises(ValueError, match=refused):
        finish.prepare(RunContext(tmp_path, knobs=knobs, facts={}), pull)


def test_prepare_tells_the_ledger_the_builders_items_only(tmp_path):
    Ledger.seed(tmp_path, BRIEF)
    found = finish.prepare(RunContext(tmp_path, knobs={"finish": True}, facts={}),
                           {"url": "U", "body": open_items.render([QA, ENV, JUDGMENT])})["finish"]
    assert (found["builder"], found["environment"]) == (["qa-1"], ["env-1"])
    said = "\n".join(entry["text"] for entry in Ledger(tmp_path).entries())
    assert QA.summary in said and ENV.summary not in said and JUDGMENT.summary not in said


# ── results ───────────────────────────────────────────────────────────────────
def _checked(tmp_path, status: str, output: str = "") -> RunContext:
    ctx = RunContext(tmp_path, knobs={}, facts={"check": "pytest -q", "commit": "a" * 40})
    ctx.write_json("authoring-evidence.json", {"validation": {"status": status, "stdout": output}})
    return ctx


@pytest.mark.parametrize("status, output, state, said", [
    ("pass", "", "resolved", "passed, so the failure did not recur"),
    ("code_failure", "FAILED tests/test_other.py::test_x", "resolved", "but not on tests/test_slow.py::test_slow"),
    ("code_failure", "FAILED tests/test_slow.py::test_slow", "open",
     "the failure recurred (tests/test_slow.py::test_slow)"),
    ("check_timeout", "", "open", "did not finish cleanly (check_timeout)"),
    ("skipped", "", "open", "the check was skipped"),
    ("", "", "open", "the check did not run"),
])
def test_the_runs_check_re_evaluates_an_environment_item(tmp_path, status, output, state, said):
    ctx = _checked(tmp_path, status, output) if status else RunContext(tmp_path, knobs={}, facts={})
    result = finish.results(ctx, _fact(QA, ENV), draft=None)
    assert result["env-1"][0] == state and said in result["env-1"][1]
    assert result["qa-1"][0] == "resolved"
    assert finish.results(ctx, _fact(QA, ENV), draft={"reason": "limit"})["qa-1"][0] == "open"


# ── whole runs, on the walker ─────────────────────────────────────────────────
def _with_items(opened, *items: OpenItem) -> str:
    """Give the opened pull request an Open items section after delivery's own text."""
    body = _pulls(opened.github, "acme/app")[0]["body"] + "\n" + open_items.render(list(items))
    _describe(opened.github, "acme/app", body)
    return body


def _finish(run, opened, answers: list[str], **knobs) -> tuple:
    model = FakeModel([DESIGN_DOC_ANSWER, *answers])
    mode = _mode(finish=True, **finish.ROUND_KNOBS, **knobs)
    return mode.run(run, model=model, task={"name": "delivered_tool", "brief": BRIEF,
                                            "check": "python tool.py", "refine": opened.url}), model


def test_a_finish_fixes_its_items_and_updates_the_same_pull_request(opened_pull, tmp_path):
    """Two open items, a QA finding and an environment item: the gate is skipped (no verdict
    is given), design_doc and the builder see the finding and never the environment item,
    the work is pushed onto the pull request's own branch, no second pull request is opened,
    and both items are ticked with their evidence in its description."""
    _with_items(opened_pull, QA, ENV)
    run = tmp_path / "finish"
    result, model = _finish(run, opened_pull, _attempt(render({"tool.py": "print('two')\n"})))
    assert result.exit_ok, (result.error, result.gate_stop)
    assert result.facts["finish_gate"]["skipped"] is True
    [design] = [p for p in model.prompts if DESIGN_DOC in p]
    builds = [p for p in model.prompts if WRITE_ARTIFACT in p]
    assert len(builds) == 1
    for prompt in (design, *builds):
        assert ENV.summary not in prompt and "test_slow" not in prompt
    assert QA.summary in design and finish.FIX_PLAN in design
    found = worktree.recorded(run)
    assert _remote_head(opened_pull.github, "acme/app", opened_pull.pull["head"]) == found.head
    [pull] = _pulls(opened_pull.github, "acme/app")
    items = {item.id: item for item in open_items.parse(pull["body"])}
    assert items["qa-1"].status == "resolved" and items["qa-1"].evidence[-1].startswith(
        f"finish run at `{found.head[:12]}`: fixed")
    assert items["env-1"].status == "resolved" and "did not recur" in items["env-1"].evidence[-1]
    assert pull["body"].count("## Open items") == 1
    assert "## Finish run\n\n- The approval gate was skipped: every item" in pull["body"]
    assert _record(run)["status"] == "updated" and result.facts["pull_request"] == opened_pull.url


def test_a_judgment_call_leaves_the_gate_to_the_person(opened_pull, tmp_path):
    _with_items(opened_pull, QA, JUDGMENT)
    run = tmp_path / "finish"
    result, _model = _finish(run, opened_pull, [], finish_items="qa-1,jc-1")
    assert result.gate_stop == "gate_strategy", (result.error, result.gate_stop)
    assert result.facts["finish_gate"]["skipped"] is False
    assert _taken(run) == ""


def test_a_finish_gets_one_fix_round_across_its_gates(opened_pull, tmp_path):
    """The QA judge sends the first attempt back (the one round); the second attempt's check
    fails, and gate_check, which on its own would send it back once more, makes it a draft.
    The builder is asked twice, never three times, and the items stay open on the draft."""
    _with_items(opened_pull, QA, ENV)
    run = tmp_path / "finish"
    redo = qa_judgment("redo", must_fix=[QA_HELD])
    failing = render({"tool.py": "raise SystemExit(3)\n"})
    result, model = _finish(run, opened_pull, [*_attempt(BUILT, judge=redo), *_attempt(failing)])
    assert result.exit_ok, (result.error, result.gate_stop)
    assert sum(WRITE_ARTIFACT in p for p in model.prompts) == 2
    assert sum(QA_JUDGE_MARK in p for p in model.prompts) == 1
    assert finish.ROUND_SPENT in result.facts["draft_delivery"]["reason"]
    assert result.facts["finish_rounds"] == 1
    [pull] = _pulls(opened_pull.github, "acme/app")
    items = {item.id: item for item in open_items.parse(pull["body"])}
    assert items["qa-1"].status == "open" and "still open, the run delivered a draft" in items["qa-1"].evidence[-1]
    assert items["env-1"].status == "resolved" and "but not on tests/test_slow.py::test_slow" in \
        items["env-1"].evidence[-1]


def test_the_finish_section_and_items_are_kept_when_the_branch_cannot_be_pushed(opened_pull, tmp_path):
    """No push rights: the run says so and keeps the work local, its pull request text
    carrying the finish's results; the pull request is left as it was."""
    before = _with_items(opened_pull, QA, ENV)
    _child().setattr(delivery, "_push", delivery_fakes.push_fails, error="push refused by GitHub")
    run = tmp_path / "finish"
    result, _model = _finish(run, opened_pull, _attempt(render({"tool.py": "print('two')\n"})))
    assert result.exit_ok, (result.error, result.gate_stop)
    assert _record(run)["status"] == "kept local", _record(run)
    assert result.facts["delivery_result"].startswith("not delivered: push refused by GitHub")
    [pull] = _pulls(opened_pull.github, "acme/app")
    assert pull["body"] == before
    text = (run / delivery.BODY).read_text()
    assert "## Finish run" in text and "**qa-1** (qa_finding, worked on): resolved" in text


# ── the flags ─────────────────────────────────────────────────────────────────
def _args(*flags: str):
    import argparse

    from refactory_dsh import run_flags

    parser = argparse.ArgumentParser()
    run_flags.add_arguments(parser)
    return parser.parse_args(["--name", "t", "--check", "true", *flags])


@pytest.mark.parametrize("flags, refused", [
    (("--finish",), "--finish goes with --refine"),
    (("--refine", "acme/app#1", "--items", "qa-1"), "--items goes with --finish"),
    (("--refine", "acme/app#1", "--finish", "--attempts", "3"), "takes no --attempts"),
])
def test_both_runtimes_refuse_a_finish_they_cannot_honour(flags, refused):
    from refactory_dsh import run_flags

    for runtime in (run_flags.SPINE, run_flags.WALKER):
        assert refused in (run_flags.refusal(_args(*flags), "default_pack/design", runtime=runtime) or "")
    fast = run_flags.refusal(_args("--refine", "acme/app#1", "--finish"), "default_pack/design_fast",
                             runtime=run_flags.SPINE)
    assert "does not take --refine, --finish" in (fast or "")


def test_a_finish_is_the_finish_knobs_with_its_round_bounds():
    from refactory_dsh import run_flags

    knobs = run_flags.knobs(_args("--refine", "acme/app#1", "--finish", "--items", "qa-1,qa-2"),
                            "default_pack/design")
    assert {"refine=acme/app#1", "finish=1", "finish_items=qa-1,qa-2", "check_attempts=2",
            "check_attempt_cap=0", "self_check_fixups=0"} <= set(knobs)


def test_the_console_design_verb_forwards_a_finish(monkeypatch):
    from refactory_dsh import console

    seen: list[list[str]] = []
    monkeypatch.setattr(console, "_forward", lambda module, argv: seen.append(argv) or {})
    assert console.main(["design", "--name", "t", "--brief", "b", "--check", "true", "--refine", "acme/app#3",
                         "--finish", "--items", "qa-1"]) == 0
    assert "--finish" in seen[-1] and seen[-1][seen[-1].index("--items") + 1] == "qa-1"
    assert "--attempts" not in seen[-1]


def test_the_console_create_verb_still_forwards_its_attempts(monkeypatch):
    from refactory_dsh import console

    seen: list[list[str]] = []
    monkeypatch.setattr(console, "_forward", lambda module, argv: seen.append(argv) or {})
    console.main(["create", "--name", "t", "--brief", "b", "--attempts", "2"])
    assert seen and seen[-1][seen[-1].index("--attempts") + 1] == "2"
