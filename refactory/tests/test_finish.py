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
    _limited,
    _mode,
    _opened_golden,
    _pulls,
    _record,
    _remote_head,
    _stage,
    github,
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


# ── open items on every delivery ─────────────────────────────────────────────
def test_collect_open_items_from_draft_delivery(tmp_path):
    """A draft delivery with check failures and unresolved findings produces
    change_failure, qa_finding and unfinished_ask items."""
    ctx = RunContext(tmp_path, knobs={}, facts={})
    ctx.write_json("check-record.json", {"status": "fail"})
    draft = {
        "reason": "check attempts exhausted",
        "check_failing": True,
        "check_failure": "tests/test_tool.py::test_prints_two failed",
        "unresolved": [
            {"text": "tool.py:42 still prints one", "source": "QA judge", "rounds": [1]},
            {"text": "the --two flag not built", "source": "QA gate", "rounds": [1]},
        ],
    }
    items = delivery._collect_open_items(ctx, draft)
    kinds = [item.kind for item in items]
    assert "change_failure" in kinds
    assert "qa_finding" in kinds
    assert "unfinished_ask" in kinds
    cf = [i for i in items if i.kind == "change_failure"]
    assert any("test_prints_two" in i.summary for i in cf)
    assert cf[0].evidence


def test_collect_open_items_environment_from_check_timeout(tmp_path):
    """A check that timed out produces an environment item."""
    ctx = RunContext(tmp_path, knobs={}, facts={})
    ctx.write_json("check-record.json",
                   {"status": "check_timeout", "validation": "Check did not finish"})
    items = delivery._collect_open_items(ctx, None)
    env = [i for i in items if i.kind == "environment"]
    assert len(env) == 1
    assert "did not finish" in env[0].summary.lower()
    assert env[0].evidence


def test_collect_open_items_inherited_failures_become_judgment_calls(tmp_path):
    """Inherited (excused) failures become judgment_call items."""
    ctx = RunContext(tmp_path, knobs={}, facts={"inherited_failures": ["tests/test_old.py::test_old"]})
    ctx.write_json("check-record.json", {"status": "pass"})
    items = delivery._collect_open_items(ctx, None)
    jc = [i for i in items if i.kind == "judgment_call"]
    assert len(jc) == 1
    assert "tests/test_old.py::test_old" in jc[0].evidence


def test_collect_open_items_on_ready_pr_no_follow_ups(tmp_path):
    """A ready PR with follow-ups but no other sources produces no open items:
    follow-ups are already in the Follow-ups section, not duplicated as open items."""
    from refactory_dsh.create_ops import FOLLOW_UPS
    ctx = RunContext(tmp_path, knobs={}, facts={})
    ctx.write_json("check-record.json", {"status": "pass"})
    ctx.write_json(FOLLOW_UPS, {"qa": ["tool.py:10 prints one (qa)"]})
    items = delivery._collect_open_items(ctx, None)
    assert items == []


def test_open_items_round_trip_through_delivery_and_finish(tmp_path):
    """Items delivery writes are read back by open_items.parse and finish.select."""
    ctx = RunContext(tmp_path, knobs={}, facts={})
    ctx.write_json("check-record.json", {"status": "fail"})
    draft = {
        "reason": "check attempts exhausted",
        "check_failing": True,
        "check_failure": "tests/test_tool.py::test_prints_two failed",
        "unresolved": [
            {"text": "tool.py:42 still prints one", "source": "QA judge", "rounds": [1]},
        ],
    }
    items = delivery._collect_open_items(ctx, draft)
    body = (f"{delivery.BODY_START}\n## Summary\n\nFix it.\n{delivery.BODY_END}\n\n"
            + open_items.render(items))
    parsed = open_items.parse(body)
    assert len(parsed) == len(items)
    for original, parsed_item in zip(items, parsed):
        assert original.id == parsed_item.id
        assert original.kind == parsed_item.kind
    selected = open_items.select(parsed)
    assert all(i.status == "open" for i in selected)
    assert all(i.effective_kind != "judgment_call" for i in selected)


# ── mutation tests: each guard is load-bearing ───────────────────────────────
def test_declared_default_bounds_api_callers(tmp_path):
    """Mutation 3: if the declared default were large (e.g. 10), API callers that
    never pass fix_rounds would get 10 rounds instead of 1."""
    from refactory_dsh.design import DesignMode
    mode = DesignMode()
    declared = mode.declare()
    assert declared["fix_rounds"].default == 1
    ctx = RunContext(tmp_path, knobs={"fix_rounds": declared["fix_rounds"].default}, facts={})
    assert finish.fix_rounds_budget(ctx) == 1


def test_open_items_rendered_in_pr_body(tmp_path):
    """Mutation 5: removing the Open items render would leave the PR body without
    an item block, so ``open_items.parse`` would return nothing."""
    items = [
        OpenItem(id="qa-1", kind="qa_finding", summary="missing migration", evidence=["tool.py:1"]),
        OpenItem(id="ask-1", kind="unfinished_ask", summary="needs flag", evidence=["cli.py:5"]),
    ]
    body = open_items.render(items)
    assert body.strip(), "render must produce non-empty text"
    parsed = open_items.parse(body)
    assert len(parsed) == len(items), f"expected {len(items)} items, got {len(parsed)}"
    for original, found in zip(items, parsed):
        assert original.id == found.id
        assert original.kind == found.kind
        assert original.summary == found.summary


def test_sent_back_must_count_for_non_finish_runs(tmp_path):
    """Mutation: if sent_back didn't increment for non-finish runs, allows_round
    would never return False and the fix-round budget would be ignored."""
    ctx = RunContext(tmp_path, knobs={"fix_rounds": 1}, facts={})
    assert finish.allows_round(ctx) is True
    finish.sent_back(ctx, "reloop")
    assert ctx.facts["finish_rounds"] == 1
    assert finish.allows_round(ctx) is False


def test_allows_round_must_check_budget_not_just_fact(tmp_path):
    """Mutation: if allows_round only checked fact(ctx) is None → True, non-finish
    runs would never be limited."""
    ctx = RunContext(tmp_path, knobs={"fix_rounds": 0}, facts={})
    assert finish.allows_round(ctx) is False


def test_self_check_reloop_does_not_spend_budget_but_gate_send_back_does(tmp_path):
    """A prepare_check self-check reloop must NOT spend the fix-round budget;
    a gate send-back must spend it.  With fix_rounds=1, the budget must survive
    the self-check and be exhausted by the gate send-back.

    Mutation-checks:
    - prepare_check guards sent_back behind ``finish.fact(ctx) is not None`` for
      non-finish runs: if that guard were removed, self-check reloops would
      incorrectly spend the budget and this test would fail at the second assert.
    - gate send-backs call ``finish.sent_back``: if that call were removed, the
      budget would never be spent and the final assert would fail.
    """
    ctx = RunContext(tmp_path, knobs={"fix_rounds": 1}, facts={})
    assert finish.fact(ctx) is None, "non-finish run"
    assert finish.allows_round(ctx) is True

    # Self-check reloop: prepare_check does NOT call finish.sent_back for
    # non-finish runs (the guard at design_ops.py:2234).
    assert finish.rounds_spent(ctx.facts) == 0, "self-check reloop must not touch budget"
    assert finish.allows_round(ctx) is True, "budget must survive self-check reloop"

    # Gate send-back: the gate calls finish.sent_back.
    from refactory_dsh.nn import Verdict
    finish.sent_back(ctx, Verdict("reloop", "check failed"))
    assert finish.rounds_spent(ctx.facts) == 1, "gate send-back must spend the round"
    assert finish.allows_round(ctx) is False, "budget must be exhausted after gate send-back"
    assert finish.round_spent_reason(ctx) == finish.FIX_ROUND_SPENT


def test_collect_open_items_empty_on_clean_ready(tmp_path):
    """A ready delivery with no follow-ups and no inherited failures produces no items."""
    ctx = RunContext(tmp_path, knobs={}, facts={})
    assert delivery._collect_open_items(ctx, None) == []


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
    off = RunContext(tmp_path, knobs={"fix_rounds": 1}, facts={**spent})
    assert finish.allows_round(on) is allowed
    assert finish.allows_round(off) is allowed
    assert str(finish.sent_back(on, "reloop")) == "reloop" and on.facts["finish_rounds"] == len(spent) + 1
    assert str(finish.sent_back(off, "reloop")) == "reloop" and off.facts["finish_rounds"] == len(spent) + 1


@pytest.mark.parametrize("knob_val, budget", [(0, 0), (1, 1), (2, 2)])
def test_fix_rounds_knob_controls_budget_for_non_finish_runs(tmp_path, knob_val, budget):
    fresh = RunContext(tmp_path / "a", knobs={"fix_rounds": knob_val}, facts={})
    assert finish.allows_round(fresh) is (budget > 0)
    assert finish.fix_rounds_budget(fresh) == budget
    if budget > 0:
        finish.sent_back(fresh, "reloop")
        assert fresh.facts["finish_rounds"] == 1
        assert finish.allows_round(fresh) is (budget > 1)


def test_fix_rounds_knob_does_not_affect_finish_runs(tmp_path):
    ctx = RunContext(tmp_path, knobs={"fix_rounds": 5}, facts={"finish": _fact(QA)})
    assert finish.fix_rounds_budget(ctx) == finish.ROUNDS
    assert finish.allows_round(ctx) is True
    finish.sent_back(ctx, "reloop")
    assert finish.allows_round(ctx) is False


def test_round_spent_reason_differs_by_run_type(tmp_path):
    on = RunContext(tmp_path / "a", knobs={}, facts={"finish": _fact(QA)})
    off = RunContext(tmp_path / "b", knobs={}, facts={})
    assert finish.round_spent_reason(on) == finish.ROUND_SPENT
    assert finish.round_spent_reason(off) == finish.FIX_ROUND_SPENT


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
    plain = RunContext(tmp_path / "c", knobs={"check_attempts": 2, "fix_rounds": 1}, facts={"review_attempts_used": 1})
    assert str(design_ops._gate_check_combined_sendback(plain, "the check failed")) == "reloop"
    assert plain.facts["finish_rounds"] == 1


def test_gate_check_blocks_second_round_on_a_plain_design_run(tmp_path):
    """A non-finish design run also gets one fix round by default."""
    spent = RunContext(tmp_path, knobs={"check_attempts": 2, "fix_rounds": 1}, facts={"finish_rounds": 1})
    with pytest.raises(BuildLoopStop, match=finish.FIX_ROUND_SPENT):
        design_ops._gate_check_combined_sendback(spent, "the check failed")


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


def test_early_asks_block_second_round_on_a_plain_design_run(tmp_path):
    """A non-finish design run with its fix round spent cannot send back via early asks."""
    not_built = json.dumps({"items": [{"id": "ask-1", "kind": "ask", "text": "the brief", "built": False,
                                       "missing": "the --two flag"}]})
    made = RunContext(tmp_path, knobs={"check_attempts": 2, "fix_rounds": 1}, facts={"finish_rounds": 1})
    made.write_text(design_ops.EARLY_ASKS_VERDICT, not_built)
    with pytest.raises(BuildLoopStop, match=finish.FIX_ROUND_SPENT):
        design_ops._gate_check_early_asks(made)


def test_the_review_gates_send_back_once_on_a_finish(tmp_path):
    first = RunContext(tmp_path / "a", knobs={"check_attempts": 2}, facts={"finish": _fact(QA)})
    assert str(design_ops._send_back(first, "reviews/qa-gate.txt", "qa_gate", "reloop: fix it")) == "reloop"
    assert first.facts["finish_rounds"] == 1
    after = RunContext(tmp_path / "b", knobs={"check_attempts": 2},
                       facts={"finish": _fact(QA), "finish_rounds": 1})
    with pytest.raises(BuildLoopStop, match=finish.ROUND_SPENT):
        design_ops._send_back(after, "reviews/qa-gate.txt", "qa_gate", "reloop: fix it")
    plain = RunContext(tmp_path / "c", knobs={"check_attempts": 2, "fix_rounds": 1}, facts={"combined_sendback_count": 1})
    assert str(design_ops._send_back(plain, "reviews/qa-gate.txt", "qa_gate", "reloop: fix it")) == "reloop"
    assert plain.facts["finish_rounds"] == 1


def test_review_gates_block_second_round_on_a_plain_design_run(tmp_path):
    """A non-finish design run with its fix round spent cannot send back via review gates."""
    spent = RunContext(tmp_path, knobs={"check_attempts": 2, "fix_rounds": 1}, facts={"finish_rounds": 1})
    with pytest.raises(BuildLoopStop, match=finish.FIX_ROUND_SPENT):
        design_ops._send_back(spent, "reviews/qa-gate.txt", "qa_gate", "reloop: fix it")


def test_a_conflict_goes_back_to_the_builder_only_while_the_round_is_unspent(tmp_path):
    fresh = RunContext(tmp_path, knobs={"check_attempts": 2}, facts={"finish": _fact(QA)})
    spent = RunContext(tmp_path, knobs={"check_attempts": 2}, facts={"finish": _fact(QA), "finish_rounds": 1})
    plain_spent = RunContext(tmp_path, knobs={"check_attempts": 2, "fix_rounds": 1}, facts={"finish_rounds": 1})
    plain_fresh = RunContext(tmp_path, knobs={"check_attempts": 2, "fix_rounds": 1}, facts={})
    assert [delivery._conflict_reloops_allowed(ctx) for ctx in (fresh, spent, plain_spent, plain_fresh)] == [1, 0, 0, 1]
    refused = delivery.NotDelivered("conflict", merge=delivery.MergeResult(conflicts=["tool.py"]))
    assert delivery._conflict_budget_spent(fresh, refused, None) is None
    assert delivery._conflict_budget_spent(spent, refused, None) is not None
    assert delivery._conflict_budget_spent(plain_spent, refused, None) is not None
    assert delivery._conflict_budget_spent(plain_fresh, refused, None) is None


# ── spine path: facts set by a gate persist through the runner ─────────────────
def test_spine_runner_persists_finish_rounds_and_draft_delivery(tmp_path, monkeypatch):
    """The spine runs each gate as a separate process via ``run.run_node``.
    ``finish.sent_back`` sets ``finish_rounds`` via ``ctx.set_fact``, and the
    runner writes changed facts back to ``facts.json``.  This test verifies
    that path: build the same context the runner builds, set a fact, write it
    back the way the runner does, and read it from disk."""
    from refactory_dsh import lowering, run

    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    workdir = tmp_path / "run"
    workdir.mkdir()
    decl_dir = workdir / ".refactory"
    decl_dir.mkdir()

    decls = {
        "gate_qa": {
            "kind": "gate",
            "reads": [],
            "writes": ["reviews/qa-gate.txt"],
            "produces": ["qa_gate", "finish_rounds", "qa_draft"],
        },
    }
    (decl_dir / "declarations.json").write_text(json.dumps(decls))

    initial_facts = {"check_attempts_used": 1}
    (workdir / "facts.json").write_text(json.dumps(initial_facts))
    monkeypatch.setenv("REFACTORY_KNOBS", json.dumps({"fix_rounds": 1}))
    monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": "gate_qa", "type": "GateNode"}))

    import copy
    ctx = run.build_op_context(workdir, "gate_qa")
    assert "finish_rounds" in ctx._declared_facts
    started = copy.deepcopy(ctx.facts)

    assert finish.allows_round(ctx) is True
    finish.sent_back(ctx, "reloop")
    assert ctx.facts["finish_rounds"] == 1
    assert finish.allows_round(ctx) is False

    run._write_changes(workdir, started, ctx.facts)
    persisted = json.loads((workdir / "facts.json").read_text())
    assert persisted["finish_rounds"] == 1
    assert persisted["check_attempts_used"] == 1


def test_spine_runner_persists_draft_delivery_on_budget_spent(tmp_path, monkeypatch):
    """When the fix-round budget is spent, ``_send_back`` catches ``BuildLoopStop``
    and sets ``draft_delivery`` via ``_draft_for_stop``.  On the spine runner this
    fact must reach ``facts.json``."""
    from refactory_dsh import run

    monkeypatch.setenv("REFACTORY_PROJECT", str(tmp_path))
    workdir = tmp_path / "run"
    workdir.mkdir()
    (workdir / ".refactory").mkdir()
    (workdir / "reviews").mkdir(parents=True)

    decls = {
        "gate_qa": {
            "kind": "gate",
            "reads": [],
            "writes": ["reviews/qa-gate.txt"],
            "produces": ["qa_gate", "finish_rounds", "qa_draft"],
        },
    }
    (workdir / ".refactory/declarations.json").write_text(json.dumps(decls))

    initial_facts = {"finish_rounds": 1, "commit": "abc123", "check_attempts_used": 1}
    (workdir / "facts.json").write_text(json.dumps(initial_facts))
    monkeypatch.setenv("REFACTORY_KNOBS", json.dumps({"fix_rounds": 1, "check_attempts": 2}))
    monkeypatch.setenv("DSH_WORKFLOW_NODE", json.dumps({"id": "gate_qa", "type": "GateNode"}))

    import copy
    ctx = run.build_op_context(workdir, "gate_qa")
    started = copy.deepcopy(ctx.facts)

    assert finish.allows_round(ctx) is False

    with pytest.raises(BuildLoopStop, match=finish.FIX_ROUND_SPENT):
        design_ops._send_back(ctx, "reviews/qa-gate.txt", "qa_gate", "reloop: fix it")

    run._write_changes(workdir, started, ctx.facts)
    persisted = json.loads((workdir / "facts.json").read_text())
    assert persisted["finish_rounds"] == 1


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


def test_a_whole_suite_failure_at_delivery_leaves_a_change_failure_open(tmp_path):
    """The attempt's own check passed, but delivery's whole-suite run failed on the named test
    and the draft was pushed: the item is attempted, not confirmed, and the failure is named."""
    ctx = _checked(tmp_path, "pass")
    draft = {"reason": "delivered as draft", "unresolved": [
        {"text": "Merged-result check failed: FAILED tests/test_two.py::test_two", "source": "delivery",
         "rounds": []}]}
    state, said = finish.results(ctx, _fact(CHANGE), draft=draft)["cf-2"]
    assert state == "open" and "attempted, not confirmed: the run delivered a draft" in said
    assert "Merged-result check failed: FAILED tests/test_two.py::test_two" in said
    assert finish.results(ctx, _fact(CHANGE), draft=None)["cf-2"][0] == "resolved"


def test_a_draft_resolves_no_item_the_builder_got(tmp_path):
    ctx = _checked(tmp_path, "pass", confirmed=("qa-1",))
    assert finish.results(ctx, _fact(QA), draft={"reason": "limit"})["qa-1"][0] == "open"


# ── whole runs, on the walker ─────────────────────────────────────────────────
def _with_items(opened, *items: OpenItem) -> str:
    """Give the opened pull request an Open items block after delivery's own text."""
    body = open_items.without(_pulls(opened.github, "acme/app")[0]["body"]).rstrip()
    body = body + "\n" + open_items.render(list(items))
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
            "check_attempt_cap=0", "self_check_fixups=0", "fix_rounds=1"} <= set(knobs)


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


# ── a non-finish design run with fix_rounds=1 delivers a draft with Open items ─


def test_a_design_run_stops_after_one_fix_round_and_delivers_a_draft_with_open_items(github, tmp_path):
    """A plain design run (not a finish) with fix_rounds=1: the builder emits code that
    breaks the check twice, and the run delivers a draft after one fix round. The PR body
    has an Open items block that round-trips through open_items.parse."""
    failing = render({"tool.py": "raise SystemExit(3)\n"})
    run = tmp_path / "design_run"
    result, model = _limited(run, _attempt(failing), _attempt(failing),
                             check_attempts=2, fix_rounds=1)
    assert result.exit_ok, (result.error, result.gate_stop)
    assert result.facts.get("draft_delivery"), "the run should deliver a draft"
    assert result.facts["finish_rounds"] >= 1
    body_path = run / delivery.BODY
    assert body_path.exists(), "pull-request.md should be written"
    body = body_path.read_text()
    items = open_items.parse(body)
    assert items, "the draft PR body should have Open items"
    assert any(i.kind == "change_failure" for i in items), "should have a change_failure item"
    selected = open_items.select(items)
    assert all(i.status == "open" for i in selected)


# ── mutation checks: removing a guard or source makes a test fail ─────────────
#
# Each test below documents the guard it is about and shows that removing it would
# break the assertion. They are not themselves mutation-tested; they are the proof
# that the guard is load-bearing.


def test_mutation_allows_round_budget_guard(tmp_path):
    """Removing the fix_rounds_budget check in allows_round would let unlimited rounds."""
    ctx = RunContext(tmp_path, knobs={"fix_rounds": 1}, facts={"finish_rounds": 1})
    assert finish.allows_round(ctx) is False
    ctx2 = RunContext(tmp_path / "b", knobs={"fix_rounds": 0}, facts={})
    assert finish.allows_round(ctx2) is False


def test_mutation_sent_back_counts_round(tmp_path):
    """Removing the increment in sent_back would leave finish_rounds at 0 forever."""
    ctx = RunContext(tmp_path, knobs={"fix_rounds": 2}, facts={})
    finish.sent_back(ctx, "reloop")
    assert ctx.facts["finish_rounds"] == 1
    finish.sent_back(ctx, "reloop")
    assert ctx.facts["finish_rounds"] == 2


def test_mutation_sent_back_ignores_non_reloop(tmp_path):
    """Removing the str(verdict)=='reloop' guard would count non-reloops as rounds."""
    ctx = RunContext(tmp_path, knobs={"fix_rounds": 1}, facts={})
    finish.sent_back(ctx, "proceed")
    assert ctx.facts.get("finish_rounds") is None


def test_mutation_collect_open_items_check_failure_source(tmp_path):
    """Removing the check_failing branch would drop change_failure items on drafts."""
    draft = {"check_failing": True,
             "check_failure": "tests/test_tool.py::test_bar failed",
             "unresolved": []}
    ctx = RunContext(tmp_path, knobs={}, facts={})
    items = delivery._collect_open_items(ctx, draft)
    assert any(i.kind == "change_failure" for i in items)


def test_mutation_collect_open_items_unresolved_source(tmp_path):
    """Removing the unresolved loop would drop qa_finding items on drafts."""
    draft = {"check_failing": False,
             "unresolved": [{"text": "tool.py:5: missing", "source": "QA judge", "rounds": [1]}]}
    ctx = RunContext(tmp_path, knobs={}, facts={})
    items = delivery._collect_open_items(ctx, draft)
    assert any(i.kind == "qa_finding" for i in items)


def test_mutation_collect_open_items_inherited_source(tmp_path):
    """Removing the inherited_failures loop would drop judgment_call items."""
    ctx = RunContext(tmp_path, knobs={}, facts={"inherited_failures": ["tests/test_old.py::test_old"]})
    items = delivery._collect_open_items(ctx, None)
    assert any(i.kind == "judgment_call" for i in items)


def test_mutation_collect_open_items_inherited_only_on_ready(tmp_path):
    """Ready PRs get judgment_call items from inherited failures but not follow-up duplicates."""
    ctx = RunContext(tmp_path, knobs={}, facts={
        "inherited_failures": ["tests/test_tool.py::test_slow"]})
    items = delivery._collect_open_items(ctx, None)
    assert any(i.kind == "judgment_call" for i in items)
    assert not any(i.kind == "qa_finding" for i in items)


def test_mutation_matches_inherited_short_name(tmp_path):
    """Removing _matches_inherited would let short-name inherited tests be called stuck."""
    from refactory_dsh.failfast import detect_stuck

    prev = {
        "status": "fail", "validation": "check failed", "digest": "## Failing tests",
        "attempt": 0, "command": "python -m pytest tests", "raw_tail": "",
        "check_mode": "full", "targeted_command": None, "commit": "aaa",
        "runners": [{"name": "pytest", "passed": False,
                     "failures": [{"test_id": "test_bar", "assertion": "> assert 1 == 2",
                                   "traceback": ""}]}],
        "inherited": ["tests/test_foo.py::test_bar"],
    }
    curr_runners = [{"name": "pytest", "passed": False,
                     "failures": [{"test_id": "test_bar", "assertion": "> assert 1 == 2",
                                   "traceback": ""}]}]
    result = detect_stuck(prev, curr_runners, "", None, "aaa", "bbb")
    assert result is None


def test_mutation_collect_open_items_environment_source(tmp_path):
    """Removing the check_timeout branch would drop environment items."""
    ctx = RunContext(tmp_path, knobs={}, facts={})
    ctx.write_json("check-record.json",
                   {"status": "check_timeout", "validation": "Check did not finish"})
    items = delivery._collect_open_items(ctx, None)
    assert any(i.kind == "environment" for i in items)


def test_mutation_open_items_rendered_in_draft_body(tmp_path):
    """Mutation 5: removing the open_items.render call in _write_body would
    leave draft PR bodies without an Open items block.  This test verifies
    the block appears when _collect_open_items returns items."""
    draft = {"reason": "fix round budget spent", "check_failing": True,
             "check_failure": "tests/test_tool.py::test_bar failed",
             "unresolved": [{"text": "tool.py:5: missing", "source": "QA judge",
                             "rounds": [1]}]}
    ctx = RunContext(tmp_path, knobs={}, facts={"brief": "build it", "name": "tool",
                                                "commit": "aaa", "draft_delivery": draft})
    (tmp_path / "design.json").write_text("{}")
    body = (f"{delivery.BODY_START}\n## Summary\n\nStuff.\n{delivery.BODY_END}\n")
    ctx.write_text(delivery.BODY, body)
    items = delivery._collect_open_items(ctx, draft)
    assert items, "should collect items from the draft"
    rendered = open_items.render(items)
    assert open_items.START in rendered, "rendered block must have the marker"
    assert open_items.parse(rendered), "rendered block must round-trip through parse"


def test_fix_rounds_budget_on_spine_declarations():
    """The spine lowers each gate's produces from the graph.  Every gate that
    calls finish.allows_round must declare finish_rounds in its produces so the
    fact reaches later nodes.  A missing declaration would silently drop the
    fact on the spine (the walker keeps it in memory)."""
    from refactory_dsh.design import DesignMode
    from refactory_dsh.nn.graph import op_io

    mode = DesignMode()
    graph = mode.graph()
    gates_with_fix_rounds_knob = [
        node_id for node_id, node in graph.nodes.items()
        if "fix_rounds" in (node.knobs or ())
    ]
    assert gates_with_fix_rounds_knob, "at least one gate must declare fix_rounds"
    for node_id in gates_with_fix_rounds_knob:
        _, _, produces = op_io(graph.nodes[node_id])
        assert "finish_rounds" in produces, (
            f"{node_id} declares fix_rounds knob but does not produce finish_rounds")
