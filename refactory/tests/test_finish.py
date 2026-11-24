"""Finishing a pull request: its Open items, and a truncated refine that works on them.

The format (`open_items`) round-trips, and an update changes only the box of an item it
has a result for and adds that result: every other line, a person's text, malformed items,
logs and duplicate sections included, stays byte for byte. A finish (`finish`) selects
its items, skips the approval gate only for its next visit and only when every item the
builder gets is concrete and the design review passed, never shows an ``environment``
item to design_doc or the builder, gets one fix round across every gate that sends a
change back, and resolves each item only on evidence. The runs here are the walker's,
against the fake GitHub of `test_delivery`; `test_finish_runtimes.py` runs a finish
through both runtimes' real entry points.
"""
from __future__ import annotations

import difflib
import json

import pytest

from refactory_dsh import delivery, design_ops, finish, open_items, worktree
from refactory_dsh.nn.executor import RunContext
from refactory_dsh.open_items import OpenItem
from refactory_dsh.run import BuildLoopStop
from refactory_dsh.steering import Ledger
import delivery_fakes
from _delivery_support import (  # noqa: F401  (_stage, _opened_golden and opened_pull are the fake GitHub's fixtures)
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
from tests_support import DESIGN_DOC_ANSWER, FakeModel, git, qa_judgment, render

QA = OpenItem("qa-1", "qa_finding", "tool.py prints one where the brief asks for two",
              ("tool.py:1", "tests/test_tool.py::test_prints_two"))
ENV = OpenItem("env-1", "environment", "ENV-ONLY-SUMMARY: the lab runner timed out",
               ("log: tests/test_slow.py::test_slow timed out after 600 s",))
JUDGMENT = OpenItem("jc-1", "judgment_call", "whether more.py belongs in this pull request",
                    ("the excuse rule used: out of scope for the brief",))
UNFINISHED = OpenItem("ask-1", "unfinished_ask", "the --two flag is not built", ("brief: 'add --two'",),
                      status="resolved")
CHANGE = OpenItem("cf-2", "change_failure", "tests/test_two.py::test_two fails on the change",
                  ("tests/test_two.py::test_two",))

BRIEF = "Finish the pull request's open items."
WRITE_ARTIFACT = "Write the artifact `"
DESIGN_DOC = "Write the design document"


def _changed(before: str, after: str) -> tuple[list[str], list[str]]:
    """The lines ``after`` removed from ``before``, and the lines it added."""
    diff = list(difflib.ndiff(before.split("\n"), after.split("\n")))
    return ([line[2:] for line in diff if line.startswith("- ")],
            [line[2:] for line in diff if line.startswith("+ ")])


# ── the format ────────────────────────────────────────────────────────────────
def test_render_and_parse_round_trip():
    done = OpenItem("cf-1", "change_failure", "it fails", ("t.py::t",), "resolved", ("finish run: passed",))
    items = [QA, ENV, JUDGMENT, UNFINISHED, done]
    text = open_items.render(items)
    assert text.startswith(f"{open_items.START}\n## Open items\n\n- [ ] **qa-1** (qa_finding): tool.py prints one")
    assert text.endswith(f"{open_items.END}\n")
    assert "\n- [x] **ask-1** (unfinished_ask): the --two flag is not built\n  - evidence: brief: 'add --two'\n" in text
    assert "  - evidence: t.py::t\n  - result: finish run: passed\n" in text
    assert open_items.parse(text) == items
    assert open_items.render(open_items.parse(text)) == text
    # Outside delivery's markers, with other sections around it.
    body = (f"{delivery.BODY_START}\n## Summary\n\nWhy.\n{delivery.BODY_END}\n\n{text}\n"
            "## Notes\n\n- [ ] **x** (qa_finding): not an item\n")
    assert open_items.parse(body) == items
    assert open_items.parse("## Summary\n\nno items here\n") == []


def test_only_an_evidence_line_is_evidence():
    """A note under an item, "(no evidence yet)" among them, is kept but is not evidence, so
    the item is not concrete; an ``evidence:`` line is, inside a fence or not, only outside one."""
    body = (f"{open_items.START}\n- [ ] **a** (qa_finding): one\n  (no evidence yet)\n  - see the chat\n"
            "- [ ] **b** (qa_finding): two\n  - evidence: b.py:3\n"
            "- [ ] **c** (qa_finding): three\n  ```\n  - evidence: inside a fence\n  ```\n"
            f"{open_items.END}\n")
    a, b, c = open_items.parse(body)
    assert (a.evidence, a.concrete) == ((), False)
    assert (b.evidence, b.concrete) == (("b.py:3",), True)
    assert (c.evidence, c.concrete) == ((), False)


def test_parse_takes_a_persons_edits():
    """A ticked box (either case), a comment between items, a note under an item, an HTML
    comment, Windows line ends and an unknown kind are read as a person meant them; the
    unknown kind is kept as written and acted on as a judgment call."""
    text = open_items.render([QA, ENV, OpenItem("u-1", "bug", "a kind nobody knows", ("u.py:1",))])
    edited = (text.replace("- [ ] **qa-1**", "- [X] **qa-1**")
              .replace("- [ ] **env-1**", "I think this one is flaky, not ours.\n\n- [ ] **env-1**")
              .replace("  - evidence: tests/test_tool.py::test_prints_two\n",
                       "  - evidence: tests/test_tool.py::test_prints_two\n  <!-- checked by hand -->\n"
                       "    fixed locally, see my comment\n")
              .replace("\n", "\r\n"))
    qa, env, unknown = open_items.parse(edited)
    assert qa.status == "resolved" and qa.evidence == QA.evidence
    assert env == ENV
    assert (unknown.kind, unknown.effective_kind) == ("bug", "judgment_call")
    assert open_items.select([qa, env, unknown]) == [env]


#: The reviewer's probe: a description with a person's text after a legacy Open items list.
PROBE = ("Intro\n\n## Open items\n\n- [ ] **qa-1** (qa_finding): x\n  - evidence: f.py:1\n\n---\n"
         "Person's note after the list: please also check docs.\n\n\U0001f916 Generated with Claude Code\n")


def test_an_update_of_a_legacy_section_changes_only_its_item():
    after = open_items.update(PROBE, {"qa-1": ("resolved", "finish run: confirmed")})
    removed, added = _changed(PROBE, after)
    assert removed == ["- [ ] **qa-1** (qa_finding): x"]
    assert added == ["- [x] **qa-1** (qa_finding): x", "  - result: finish run: confirmed"]
    assert after.endswith("---\nPerson's note after the list: please also check docs.\n\n"
                          "\U0001f916 Generated with Claude Code\n")


#: Inside a block: every malformed form the review listed, a fenced log, a note, then a
#: duplicate legacy section after the block. All of it must survive an update.
MALFORMED = [
    "- [ ] **a b** (qa_finding) one",
    "- [ ] a (qa_finding): x",
    "### group",
    "1. [ ] **n** (qa_finding): numbered",
    "- [] **e** (qa_finding): no space in the box",
    "- [ ] **`t`** (qa_finding): backticked id",
    "```",
    "log line, unindented",
    "```",
    "- [ ] **k** (QA_Finding): capitalised kind",
    "  - evidence: k.py:1",
    "- [ ] **u** (bug): unknown kind",
    "  - evidence: u.py:1",
    "<!-- a person's comment -->",
    "A person's note inside the block.",
]


def test_an_update_inside_the_block_keeps_every_other_line_byte_for_byte():
    body = ("Intro\n\n" + f"{delivery.BODY_START}\n## Summary\n\nWhy.\n{delivery.BODY_END}\n\n"
            + open_items.START + "\n## Open items\n\n- [ ] **qa-1** (qa_finding): x\n  - evidence: f.py:1\n"
            + "\n".join(MALFORMED) + "\n" + open_items.END + "\n\n## Open items\n\n- [ ] **qa-1** "
            "(qa_finding): a duplicate, outside the block\n\n\U0001f916 footer\n")
    found = {item.id: item for item in open_items.parse(body)}
    assert set(found) == {"qa-1", "k", "u"} and found["u"].kind == "bug"
    after = open_items.update(body, {"qa-1": ("resolved", "R1"), "k": ("open", "R2"), "u": ("open", "R3")})
    removed, added = _changed(body, after)
    assert removed == ["- [ ] **qa-1** (qa_finding): x"]
    assert added == ["- [x] **qa-1** (qa_finding): x", "  - result: R1", "  - result: R2", "  - result: R3"]
    for line in MALFORMED:
        assert after.count(line) == body.count(line), line
    assert after.endswith("- [ ] **qa-1** (qa_finding): a duplicate, outside the block\n\n\U0001f916 footer\n")
    assert "- [ ] **u** (bug): unknown kind\n  - evidence: u.py:1\n  - result: R3\n" in after


def test_an_update_leaves_unselected_items_and_a_persons_tick_alone():
    """Items with no result, a judgment call and one a person ticked meanwhile, are left as
    they are; an open result leaves a box a person ticked."""
    body = open_items.render([QA, JUDGMENT, CHANGE]).replace("- [ ] **cf-2**", "- [x] **cf-2**")
    after = open_items.update(body, {"qa-1": ("open", "R")})
    removed, added = _changed(body, after)
    assert removed == [] and added == ["  - result: R"]
    ticked = open_items.update(body, {"cf-2": ("open", "R")})
    assert "- [x] **cf-2**" in ticked


def test_an_update_adds_what_the_description_lost():
    after = open_items.update("Just prose.\n", {"qa-1": ("resolved", "R")}, [QA])
    assert after.startswith("Just prose.\n\n" + open_items.START)
    assert open_items.parse(after)[0].results == ("R",) and open_items.parse(after)[0].status == "resolved"
    assert open_items.update("Just prose.\n", {}, [QA]) == "Just prose.\n"


def test_without_takes_out_the_block_and_a_legacy_section():
    body = f"A\n{open_items.render([QA])}B\n## Open items\n- [ ] **x** (qa_finding): y\n## C\nc\n"
    assert open_items.without(body) == "A\nB\n## C\nc\n"


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


def _visit(run) -> str:
    """What the approval gate takes at a visit, as both runtimes take it."""
    ledger = Ledger(run)
    return ledger.take_verdict("gate_strategy", ledger.visit("gate_strategy", list(finish.GATE_READS)),
                               ["proceed", "reloop"])


@pytest.mark.parametrize("items, passed, skipped", [
    ((QA, ENV), True, True),
    ((QA, ENV), False, False),
    ((QA, JUDGMENT), True, False),
    ((QA, OpenItem("cf-1", "change_failure", "the check fails on tool.py")), True, False),
])
def test_the_gate_is_answered_once_only_for_concrete_items_and_a_passed_design_review(
        tmp_path, items, passed, skipped):
    """Concrete items (an environment item beside them included) and a design review that
    passed answer the gate's next visit, once, with the reason in the ledger; a judgment
    call, a concrete kind with no evidence, or a design review out of rewrites leaves the
    gate to the person."""
    Ledger.seed(tmp_path, BRIEF)
    for name in finish.GATE_READS:
        (tmp_path / name).write_text(f"{name}\n")
    ctx = RunContext(tmp_path, knobs={}, facts={"finish": _fact(*items)})
    ctx.set_fact("finish_gate", finish.decide_gate(ctx, _fact(*items)))
    finish.after_design_review(ctx, passed=passed)
    assert ctx.facts["finish_gate"]["skipped"] is skipped
    assert _visit(tmp_path) == ("proceed" if skipped else "")
    assert _visit(tmp_path) == ""  # one use
    entries = [entry["text"] for entry in Ledger(tmp_path).entries() if entry["kind"] == "verdict"]
    if skipped:
        [entry] = entries
        assert "reason: the approval gate was skipped: every item" in entry and "design review passed" in entry
        assert "source: upfront" not in entry and "scope: run" not in entry
    else:
        assert entries == []
        assert "asked" in ctx.facts["finish_gate"]["reason"]


def test_a_design_that_changed_after_the_answer_is_not_answered(tmp_path):
    Ledger.seed(tmp_path, BRIEF)
    for name in finish.GATE_READS:
        (tmp_path / name).write_text(f"{name}\n")
    ctx = RunContext(tmp_path, knobs={}, facts={"finish": _fact(QA)})
    ctx.set_fact("finish_gate", finish.decide_gate(ctx, _fact(QA)))
    finish.after_design_review(ctx, passed=True)
    (tmp_path / "design.md").write_text("a different design\n")
    assert _visit(tmp_path) == ""


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
    ({"finish": True}, {"body": "no items"}, "no Open items block"),
    ({"finish": True, "pr": False}, {"body": open_items.render([QA])}, "takes no pr=0"),
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
def _checked(tmp_path, status: str, output: str = "", *, confirmed: tuple[str, ...] = (),
             inherited: tuple[str, ...] = ()) -> RunContext:
    ctx = RunContext(tmp_path, knobs={}, facts={"check": "pytest -q", "commit": "a" * 40,
                                                "inherited_failures": list(inherited)})
    if status:
        ctx.write_json("authoring-evidence.json", {"validation": {"status": status, "stdout": output}})
    ctx.write_text("qa-judge.json", json.dumps({
        "verdict": "proceed", "must_fix": [], "follow_ups": [], "dismissed": [],
        "finish_items": [{"id": name, "met": True, "evidence": f"{name}.py:1"} for name in confirmed]
        + [{"id": "ask-2", "met": False, "evidence": "ask-2.py:1"}]}))
    return ctx


@pytest.mark.parametrize("status, output, state, said", [
    ("pass", "", "resolved", "passed, so the failure did not recur"),
    ("code_failure", "FAILED tests/test_other.py::test_x", "resolved", "but not on tests/test_slow.py::test_slow"),
    ("code_failure", "FAILED tests/test_slow.py::test_slow", "open",
     "the failure recurred (tests/test_slow.py::test_slow)"),
    ("code_failure", "FAILED tests/test_other.py::test_x\n[... 10 bytes of output not kept ...]\n", "open",
     "its output was cut"),
    ("check_timeout", "", "open", "did not finish cleanly (check_timeout)"),
    ("skipped", "", "open", "the check was skipped"),
    ("", "", "open", "the check did not run"),
])
def test_the_runs_check_re_evaluates_an_environment_item(tmp_path, status, output, state, said):
    result = finish.results(_checked(tmp_path, status, output), _fact(ENV), draft=None)
    assert result["env-1"][0] == state and said in result["env-1"][1]


def test_an_environment_item_that_names_no_test_is_not_resolved_by_a_failing_check(tmp_path):
    unnamed = OpenItem("env-2", "environment", "the runner ran out of disk", ("log: no space left",))
    result = finish.results(_checked(tmp_path, "code_failure", "FAILED a.py::b"), _fact(unnamed), draft=None)
    assert result["env-2"][0] == "open" and "names no test" in result["env-2"][1]


def test_each_item_is_resolved_only_on_its_own_evidence(tmp_path):
    """The reviewer's case: qa-1 fixed and confirmed by the QA judge, cf-2 left alone (its
    test still fails, excused as a failure the base has too). Only qa-1 is resolved; nothing
    is called fixed that the run did not confirm."""
    ctx = _checked(tmp_path, "pass", confirmed=("qa-1",), inherited=("tests/test_two.py::test_two",))
    ask = OpenItem("ask-2", "unfinished_ask", "the --two flag", ("brief: 'add --two'",))
    result = finish.results(ctx, _fact(QA, CHANGE, ask), draft=None)
    assert result["qa-1"] == ("resolved", "finish run at `aaaaaaaaaaaa`: the QA judge confirmed it met (qa-1.py:1)")
    assert result["cf-2"][0] == "open" and "attempted, not confirmed" in result["cf-2"][1]
    assert "tests/test_two.py::test_two still failed, excused" in result["cf-2"][1]
    # The judge answered for ask-2, with evidence, but not met.
    assert result["ask-2"] == ("open", "finish run at `aaaaaaaaaaaa`: attempted, not confirmed: the QA judge "
                                       "did not confirm it met")
    assert not any("fixed" in said for _state, said in result.values())


@pytest.mark.parametrize("status, output, inherited, state", [
    ("pass", "", (), "resolved"),
    ("pass", "FAILED tests/test_two.py::test_two (excused)", (), "open"),
    ("pass", "", ("tests/test_two.py::test_two",), "open"),
    ("code_failure", "", (), "open"),
])
def test_a_change_failure_is_resolved_only_when_its_tests_passed(tmp_path, status, output, inherited, state):
    result = finish.results(_checked(tmp_path, status, output, inherited=inherited), _fact(CHANGE), draft=None)
    assert result["cf-2"][0] == state
    nameless = OpenItem("cf-3", "change_failure", "something fails", ("the CI log",))
    assert finish.results(_checked(tmp_path, "pass"), _fact(nameless), draft=None)["cf-3"][0] == "open"


def test_a_draft_resolves_no_item_the_builder_got(tmp_path):
    ctx = _checked(tmp_path, "pass", confirmed=("qa-1",))
    assert finish.results(ctx, _fact(QA), draft={"reason": "limit"})["qa-1"][0] == "open"


# ── whole runs, on the walker ─────────────────────────────────────────────────
def _with_items(opened, *items: OpenItem) -> str:
    """Give the opened pull request an Open items block after delivery's own text."""
    body = _pulls(opened.github, "acme/app")[0]["body"] + "\n" + open_items.render(list(items))
    _describe(opened.github, "acme/app", body)
    return body


def _confirms(*ids: str, verdict: str = "proceed", must_fix: list | None = None) -> str:
    """A QA judge's answer that confirms ``ids`` met."""
    return json.dumps({**json.loads(qa_judgment(verdict, must_fix=must_fix)),
                       "finish_items": [{"id": name, "met": True, "evidence": "tool.py:1"} for name in ids]})


def _finish(run, opened, answers: list[str], *, model: FakeModel | None = None, check: str = "python tool.py",
            **knobs) -> tuple:
    model = model or FakeModel([DESIGN_DOC_ANSWER, *answers])
    mode = _mode(finish=True, **finish.ROUND_KNOBS, **knobs)
    return mode.run(run, model=model, task={"name": "delivered_tool", "brief": BRIEF,
                                            "check": check, "refine": opened.url}), model


CF9 = OpenItem("cf-9", "change_failure", "a test nobody selected", ("tests/test_nine.py::test_nine",))


class _TicksDuringTheRun(FakeModel):
    """A builder; while it works, the person ticks cf-9 in the pull request's description."""

    def __init__(self, answers: list[str], github) -> None:
        super().__init__(answers)
        self.github = github

    def complete(self, prompt: str, *, system: str = "", workdir: object = None, cwd: object = None,
                 env: object = None, confine: bool = False) -> str:
        if WRITE_ARTIFACT in prompt:
            body = _pulls(self.github, "acme/app")[0]["body"]
            _describe(self.github, "acme/app", body.replace("- [ ] **cf-9**", "- [x] **cf-9**"))
        return super().complete(prompt, system=system, workdir=workdir, cwd=cwd, env=env, confine=confine)


def test_a_finish_fixes_its_items_and_updates_the_same_pull_request(opened_pull, tmp_path):
    """A QA finding and an environment item selected, a judgment call and a change failure
    not: the gate is skipped (no verdict is given), design_doc and the builder never see the
    environment item, the work is pushed onto the pull request's own branch, no second pull
    request is opened, the two selected items get their results, and the description
    changes nowhere else, not even where the person ticked cf-9 during the run."""
    _with_items(opened_pull, QA, ENV, JUDGMENT, CF9)
    run = tmp_path / "finish"
    model = _TicksDuringTheRun(
        [DESIGN_DOC_ANSWER, *_attempt(render({"tool.py": "print('two')\n"}), judge=_confirms("qa-1"))],
        opened_pull.github)
    result, model = _finish(run, opened_pull, [], model=model, finish_items="qa-1,env-1")
    assert result.exit_ok, (result.error, result.gate_stop)
    assert result.facts["finish_gate"]["skipped"] is True
    [design] = [p for p in model.prompts if DESIGN_DOC in p]
    builds = [p for p in model.prompts if WRITE_ARTIFACT in p]
    assert len(builds) == 1
    for prompt in (design, *builds):
        assert ENV.summary not in prompt and "test_slow" not in prompt
    assert QA.summary in design and finish.FIX_PLAN in design
    # The QA judge is asked about each item the builder got, and only those.
    [judge] = [p for p in model.prompts if QA_JUDGE_MARK in p]
    assert finish.JUDGE_ASK in judge and QA.summary in judge and ENV.summary not in judge
    found = worktree.recorded(run)
    assert _remote_head(opened_pull.github, "acme/app", opened_pull.pull["head"]) == found.head
    [pull] = _pulls(opened_pull.github, "acme/app")
    items = {item.id: item for item in open_items.parse(pull["body"])}
    assert items["qa-1"].status == "resolved" and items["qa-1"].results == (
        f"finish run at `{found.head[:12]}`: the QA judge confirmed it met (tool.py:1)",)
    assert items["env-1"].status == "resolved" and "did not recur" in items["env-1"].results[-1]
    block = pull["body"][pull["body"].index(open_items.START):]
    ticked = open_items.render([QA, ENV, JUDGMENT, CF9]).replace("- [ ] **cf-9**", "- [x] **cf-9**")
    removed, added = _changed(ticked, block)
    assert removed == ["- [ ] **qa-1** (qa_finding): tool.py prints one where the brief asks for two",
                       "- [ ] **env-1** (environment): ENV-ONLY-SUMMARY: the lab runner timed out"], removed
    assert [line for line in added if not line.startswith("  - result: ")] == [
        "- [x] **qa-1** (qa_finding): tool.py prints one where the brief asks for two",
        "- [x] **env-1** (environment): ENV-ONLY-SUMMARY: the lab runner timed out"]
    assert "## Finish run\n\n- The approval gate was skipped: every item" in pull["body"]
    assert _record(run)["status"] == "updated" and result.facts["pull_request"] == opened_pull.url


def test_an_unconfirmed_item_is_attempted_not_fixed(opened_pull, tmp_path):
    """The QA judge proceeds but confirms nothing: the item stays open, attempted."""
    _with_items(opened_pull, QA)
    result, _model = _finish(tmp_path / "finish", opened_pull, _attempt(render({"tool.py": "print('two')\n"})))
    assert result.exit_ok, (result.error, result.gate_stop)
    [pull] = _pulls(opened_pull.github, "acme/app")
    [item] = open_items.parse(pull["body"])
    assert item.status == "open" and "attempted, not confirmed" in item.results[-1]


def test_a_judgment_call_leaves_the_gate_to_the_person(opened_pull, tmp_path):
    _with_items(opened_pull, QA, JUDGMENT)
    run = tmp_path / "finish"
    result, _model = _finish(run, opened_pull, [], finish_items="qa-1,jc-1")
    assert result.gate_stop == "gate_strategy", (result.error, result.gate_stop)
    assert result.facts["finish_gate"]["skipped"] is False
    assert not [entry for entry in Ledger(run).entries() if entry["kind"] == "verdict"]


def test_a_finish_gets_one_fix_round_across_its_gates(opened_pull, tmp_path):
    """The QA judge sends the first attempt back (the one round); the second attempt's check
    fails, and gate_check, which on its own would send it back once more, makes it a draft.
    The builder is asked twice, never three times, and the items stay open on the draft."""
    _with_items(opened_pull, QA, ENV)
    run = tmp_path / "finish"
    redo = qa_judgment("redo", must_fix=[QA_HELD])
    failing = render({"tool.py": "raise SystemExit(3)\n"})
    result, model = _finish(run, opened_pull, [*_attempt(render({"tool.py": "print('built')\n"}), judge=redo),
                                               *_attempt(failing)])
    assert result.exit_ok, (result.error, result.gate_stop)
    assert sum(WRITE_ARTIFACT in p for p in model.prompts) == 2
    assert sum(QA_JUDGE_MARK in p for p in model.prompts) == 1
    assert finish.ROUND_SPENT in result.facts["draft_delivery"]["reason"]
    assert result.facts["finish_rounds"] == 1
    [pull] = _pulls(opened_pull.github, "acme/app")
    items = {item.id: item for item in open_items.parse(pull["body"])}
    assert items["qa-1"].status == "open" and "the run delivered a draft" in items["qa-1"].results[-1]
    assert items["env-1"].status == "resolved" and "but not on tests/test_slow.py::test_slow" in \
        items["env-1"].results[-1]


def _push_to_pull(opened, files: dict[str, str], tmp_path) -> None:
    """A commit on the pull request's branch on GitHub, as its author would push it."""
    clone = tmp_path / "author"
    git(tmp_path, "clone", "-q", str(opened.github / "acme/app.git"), str(clone))
    git(clone, "checkout", "-q", opened.pull["head"])
    for name, text in files.items():
        (clone / name).parent.mkdir(parents=True, exist_ok=True)
        (clone / name).write_text(text)
    git(clone, "add", "-A")
    git(clone, "-c", "user.name=a", "-c", "user.email=a@example.com", "commit", "-q", "-m", "tests")
    git(clone, "push", "-q", "origin", opened.pull["head"])


def test_a_self_check_round_then_a_review_send_back_is_one_round(opened_pull, tmp_path):
    """The first build breaks a test the self-check runs: its send-back is the round. The
    QA judge then asks for a redo of the second build: no second round, a draft, for that
    reason. One builder round after the first build."""
    _push_to_pull(opened_pull, {"tests/test_tool.py": "def test_ok():\n    assert True\n"}, tmp_path)
    _with_items(opened_pull, QA)
    run = tmp_path / "finish"
    broken = render({"tests/test_tool.py": "def test_ok():\n    assert False\n", "tool.py": "print('two')\n"})
    fixed = render({"tests/test_tool.py": "def test_ok():\n    assert True\n"})
    redo = qa_judgment("redo", must_fix=[QA_HELD])
    result, model = _finish(run, opened_pull, [broken, *_attempt(fixed, judge=redo)],
                            check="python -m pytest -q tests")
    assert result.exit_ok, (result.error, result.gate_stop)
    assert sum(WRITE_ARTIFACT in p for p in model.prompts) == 2
    assert result.facts["finish_rounds"] == 1
    assert finish.ROUND_SPENT in result.facts["draft_delivery"]["reason"], result.facts["draft_delivery"]


def test_a_legacy_description_gets_only_its_results(opened_pull, tmp_path):
    """A description with no delivery markers and a plain ``## Open items`` list (the
    reviewer's probe): the finish adds its result and ticks the item, changes nothing else,
    and says so accurately."""
    legacy = PROBE.replace("- evidence: f.py:1", "- evidence: tool.py:1")
    _describe(opened_pull.github, "acme/app", legacy)
    run = tmp_path / "finish"
    result, _model = _finish(run, opened_pull, _attempt(render({"tool.py": "print('two')\n"}),
                                                        judge=_confirms("qa-1")))
    assert result.exit_ok, (result.error, result.gate_stop)
    [pull] = _pulls(opened_pull.github, "acme/app")
    removed, added = _changed(legacy, pull["body"])
    assert removed == ["- [ ] **qa-1** (qa_finding): x"]
    assert added[0] == "- [x] **qa-1** (qa_finding): x" and added[1].startswith("  - result: ") and len(added) == 2
    said = (f"the sections delivery writes in the description of {opened_pull.url} were not updated: it has "
            "no delivery markers, so its text is kept as written, but for the results this finish added "
            "to its Open items")
    assert said in _record(run)["note"]


def test_the_finish_section_and_items_are_kept_when_the_branch_cannot_be_pushed(opened_pull, tmp_path):
    """No push rights: the run says so and keeps the work local, its pull request text
    carrying the finish's results; the pull request is left as it was."""
    before = _with_items(opened_pull, QA, ENV)
    _child().setattr(delivery, "_push", delivery_fakes.push_fails, error="push refused by GitHub")
    run = tmp_path / "finish"
    result, _model = _finish(run, opened_pull, _attempt(render({"tool.py": "print('two')\n"}),
                                                        judge=_confirms("qa-1")))
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
    (("--refine", "acme/app#1", "--finish", "--no-pr"), "takes no --no-pr"),
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
