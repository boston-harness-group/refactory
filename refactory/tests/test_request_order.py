"""The proposal request, organised: the right things in the right order, without bloat.

The score, then the pipeline once (the view), what this run has tried, what the analysts
found, the edits it may make, the items, and the reference last, before the answer's form.
Item evidence is chosen rather than cut: a few items in full -- the failures the findings
cite and passing items beside them, marked as successes -- each step's output fenced with
its headings demoted, so nothing in an item reads as a section. A text that still has to be
cut says so in place ("[… N chars omitted]", never a path: the proposer has no tools), and
each cut is in the request's own record (``ProposalContext.omissions``) for the run's trace.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.optim import Candidate, ProposalContext, base
from refactory_dsh.optim.memory import state_view

ORDER = ["## The pipeline you may edit", "## Recent results", "## What this run has already tried",
         "## What the analysts found, ranked", "## Where the pipeline fails, step by step",
         "## The operations you may use", "## Structural operators", "## Each step's settings",
         "## Item results of the candidate being edited",
         "## Trajectories of the candidate being edited", base.REFERENCE_HEADING,
         "## Each step in full, and the node schema", "## The rules the validator checks",
         "## Answer"]


def _record(item: str, passed: bool, kind: str = "K1", size: int = 300) -> dict[str, Any]:
    body = "\n".join(["## Option A", "This looks right.", "# Summary", "x" * size])
    return {"id": item, "passed": passed, "score": float(passed), "prediction": "B",
            "label": "A", "fail_reason": None if passed else "wrong option",
            "metadata": {"question": f"Question {item}?", "task_type": kind},
            "exchanges": [{"node": "draft", "prompt": "P", "response": body},
                          {"node": "final", "prompt": "Q", "response": f"## Answer\n{body}"}]}


def _ctx(records, *, findings=None, **fields) -> ProposalContext:
    from pathlib import Path as _P

    from refactory_dsh import registry

    _package, mode = registry.resolve(_P("."), "livemath/solver")
    view = state_view(mode().state_dict())
    analysis = None if findings is None else {
        "findings": findings, "group_size": 8,
        "batches": [{"kind": "failure", "items": [r["id"] for r in records if not r["passed"]]},
                    {"kind": "success", "items": [r["id"] for r in records if r["passed"]]}]}
    return ProposalContext(mode_name="m", nodes=view.nodes, edges=view.edges,
                           node_states=view.node_states, knobs=view.knobs,
                           catalogue=nn_edits.catalogue("broad"), item_records=records,
                           analysis=analysis, **fields)


def _many(n_fail: int = 12, n_pass: int = 8, size: int = 9000) -> list[dict[str, Any]]:
    return ([_record(f"f{i}", False, "K1" if i % 2 else "K2", size) for i in range(n_fail)]
            + [_record(f"p{i}", True, "K1" if i % 2 else "K3", size) for i in range(n_pass)])


FINDINGS = [{"source": "failure", "finding": "F-ONE", "items": ["f3", "f1"], "impact": "high",
             "where": {"nodes": ["solve"]}, "steps": {"first_wrong": "solve"}},
            {"source": "failure", "finding": "F-TWO", "items": ["f2"], "impact": "medium"},
            {"source": "success", "finding": "S-ONE", "items": ["p4"], "impact": "medium"}]


def _outside_fences(text: str) -> list[tuple[int, str]]:
    """Each line outside a code fence, with its line number: what reads as the request's
    own text (a fenced knob text or step output does not)."""
    out, fence = [], None
    for number, line in enumerate(text.splitlines()):
        found = re.match(r"^(`{3,})", line)
        if found and (fence is None or found.group(1) == fence):
            fence = None if fence else found.group(1)
            continue
        if fence is None:
            out.append((number, line))
    return out


def _sections(text: str) -> list[str]:
    return [line for _n, line in _outside_fences(text) if re.match(r"#{1,3} ", line)]


def test_the_sections_come_in_order():
    text = _ctx(_many(), findings=FINDINGS).render()
    heads = _sections(text)
    at = [next(i for i, h in enumerate(heads) if h.startswith(heading)) for heading in ORDER]
    assert at == sorted(at), [h for h, _ in sorted(zip(ORDER, at), key=lambda p: p[1])]
    # The pipeline once at the top, its detail only in the reference.
    pipeline = text.split("## The pipeline you may edit", 1)[1].split("\n## ", 1)[0]
    assert base.REFERENCE_POINTER in pipeline
    assert "Nodes:" not in text and "Edges:" not in text, "the view is the graph's one table"


def test_no_heading_from_an_item_reads_as_a_section():
    text = _ctx(_many(), findings=FINDINGS).render()
    headings = _sections(text)
    assert not any(h.startswith(("# Summary", "## Option A")) for h in headings), headings
    assert sum(1 for h in headings if h == "## Answer") == 1
    # Each step output sits in a fence, its headings demoted below the request's levels.
    item = text.split("### Item f3", 1)[1].split("### Item", 1)[0]
    assert "\n```\n##### Option A\n" in item and "\n#### Summary\n" in item


def test_a_few_items_are_shown_in_full_failures_the_findings_cite_first_then_successes():
    text = _ctx(_many(), findings=FINDINGS).render()
    shown = re.findall(r"^### Item (\S+) — (\S+ ?\S*)", text, re.M)
    ids = [item for item, _ in shown]
    assert ids == ["f3", "f2", "p4"], ids  # one per finding of each kind, in rank order
    passing = [i for i in ids if i.startswith("p")]
    assert text.count("(A success, for contrast.)") == len(passing)
    assert "### Every item" in text  # every item still in the table


def test_successes_are_shown_whenever_there_are_any():
    text = _ctx(_many(n_fail=10, n_pass=1), findings=FINDINGS[:2]).render()
    assert "(A success, for contrast.)" in text and "### Item p0" in text
    none = _ctx(_many(n_fail=10, n_pass=0), findings=FINDINGS[:2]).render()
    assert "(A success, for contrast.)" not in none


def test_a_cut_is_marked_in_place_without_a_path_and_recorded():
    huge = [_record("f0", False, size=200000), *_many(n_fail=5, n_pass=2, size=2000)]
    ctx = _ctx(huge, findings=[{"source": "failure", "finding": "F", "items": ["f0"]}])
    text = ctx.render()
    marks = re.findall(r"\[… ([\d,]+) chars omitted\]", text)
    assert marks and len(marks) == len(ctx.omissions), (marks, ctx.omissions)
    assert sorted(int(m.replace(",", "")) for m in marks) == sorted(
        o["chars"] for o in ctx.omissions)
    assert all(o["item"] == "f0" and o["step"] for o in ctx.omissions), ctx.omissions
    assert not re.search(r"(optimizer_trace|\.jsonl|/runs/|item_records)", text)
    # Rendered again, the record says what this rendering cut, not the last one's too.
    ctx.render()
    assert len(ctx.omissions) == len(marks)


def test_many_long_items_are_chosen_not_cut():
    """Size is no reason to cut: the items the findings choose are each shown whole, and
    the rest are in the table."""
    ctx = _ctx(_many(n_fail=20, n_pass=15, size=9000), findings=FINDINGS)
    text = ctx.render()
    items_part = text.split("## Trajectories of the candidate being edited", 1)[1].split(
        base.REFERENCE_HEADING, 1)[0]
    assert re.findall(r"^### Item (\S+)", items_part, re.M) == ["f3", "f2", "p4"]
    assert "chars omitted" not in items_part and ctx.omissions == []


def test_the_optimizer_traces_what_its_request_left_out(tmp_path):
    from refactory_dsh.optim.random_optimizer import AutoresearchOptimizer
    import json

    class Silent:
        def complete(self, prompt, **kw):
            return json.dumps({"diagnosis": "none", "proposals": []})

    optimizer = AutoresearchOptimizer(Silent())
    optimizer.trace_path = tmp_path / "optimizer_trace.jsonl"
    huge = [_record("f0", False, size=200000), *_many(n_fail=5, n_pass=2, size=2000)]
    ctx = _ctx(huge, findings=[{"source": "failure", "finding": "F", "items": ["f0"]}])
    optimizer.propose(ctx, 1, [Candidate.from_state({"graph": {"nodes": {}, "edges": []},
                                                     "knobs": {}})])
    calls = [json.loads(line) for line in optimizer.trace_path.read_text().splitlines()]
    proposal = next(c for c in calls if c.get("kind") == "proposal")
    assert proposal["omissions"] and all(o["item"] == "f0" for o in proposal["omissions"])
    assert len(re.findall(r"\[… [\d,]+ chars omitted\]", proposal["prompt"])) == len(
        proposal["omissions"])


def test_one_contrasting_success_is_shown_per_success_finding():
    """Two success findings, each citing its own passing item: both are shown beside the
    failures, one per finding in rank order."""
    findings = [*FINDINGS, {"source": "success", "finding": "S-TWO", "items": ["p6"],
                            "impact": "low"}]
    text = _ctx(_many(), findings=findings).render()
    ids = re.findall(r"^### Item (\S+) — ", text, re.M)
    assert ids == ["f3", "f2", "p4", "p6"], ids
    assert text.count("(A success, for contrast.)") == 2


def test_the_leaderboard_does_not_repeat_the_edits_the_attempt_lines_list():
    """"Recent results" keeps id, parent, score, gate; each candidate's edits are said once,
    in "What this run has already tried" -- unless that list left the candidate out."""
    edit = {"op": "set_knob", "name": "level", "value": 7}
    rows = [{"id": "c1", "parent": "seed", "score": 0.5, "structural": False, "exit_ok": True,
             "edits": [edit], "own": 1, "epoch": 1}]
    text = ProposalContext(leaderboard=rows, history=rows).render()
    recent = text.split("## Recent results", 1)[1].split("\n## ", 1)[0]
    assert "| edits |" not in recent and "level" not in recent
    tried = text.split("## What this run has already tried", 1)[1].split("\n## ", 1)[0]
    assert "level" in tried
    other = [{**rows[0], "id": "c0"}]
    text = ProposalContext(leaderboard=rows + other, history=rows).render()
    recent = text.split("## Recent results", 1)[1].split("\n## ", 1)[0]
    assert "| edits |" in recent and "(below)" in recent


def test_earlier_reasoning_shows_only_the_latest_round_in_full():
    """Older rounds are one line each in "What this run has already tried" and in memory;
    only the latest round's reasoning is a block, however few entries there are."""
    entries = [{"id": f"c{e}", "epoch": e, "score": 0.5, "edit": [], "rationale": f"WHY-{e}"}
               for e in (1, 2, 2)]
    text = "\n".join(base._reasoning_lines(entries))
    assert "### c1 " not in text and "WHY-1" not in text
    assert text.count("### c2 ") == 2 and text.count("WHY-2") == 2


def test_an_attempt_line_shows_only_its_own_edits_not_its_ancestors():
    """A lineage of three edits, two of them the candidate's own: its line shows those two
    and not its parent's."""
    lineage = [{"op": "set_knob", "name": f"knob_{k}", "value": k} for k in ("a", "b", "c")]
    rows = [{"id": "c1", "parent": "c0", "score": 0.5, "structural": False, "exit_ok": True,
             "edits": lineage, "own": 2, "epoch": 2}]
    (line,) = base._attempt_lines(rows, {})
    assert "knob_b" in line and "knob_c" in line and "knob_a" not in line


def test_an_attempt_the_last_edit_s_section_shows_points_there_for_what_it_built():
    rows = [{"id": "c1", "parent": "c0", "score": 0.5, "structural": True, "exit_ok": True,
             "edits": [{"op": "loop", "step": "solve"}], "own": 1, "epoch": 1,
             "built": ["+ loop(solve, judge j, max 2 -> 3 runs) BUILT-LINE"]},
            {"id": "c2", "parent": "c0", "score": 0.4, "structural": True, "exit_ok": True,
             "edits": [{"op": "loop", "step": "solve"}], "own": 1, "epoch": 1,
             "built": ["+ loop(solve, judge k, max 1 -> 2 runs) OTHER-LINE"]}]
    first, second = base._attempt_lines(rows, {}, ["c1"])
    assert 'built: under "What the last edit changed"' in first and "BUILT-LINE" not in first
    assert "OTHER-LINE" in second


def test_each_fact_of_the_request_is_said_once():
    """The validation gate's rule, the item counts and a score: each said once, in one
    form; an empty column is not printed; the blocks note only when there are blocks."""
    score = 0.5714285714285714
    rows = [{"id": "c1", "parent": "seed", "score": score, "structural": False,
             "exit_ok": True, "edits": [], "own": 0, "epoch": 1, "gate": "current"}]
    records = _many()  # not all shown whole: the "Every item" table is there
    ctx = _ctx(records, findings=FINDINGS, parsimony=0.0, selection="single",
               leaderboard=rows, history=rows, best_score=score, mean_score=score, mean_over=1)
    text = ctx.render()
    assert text.count("a tie with the current candidate is rejected") == 1
    assert "its rule is under \"What the score measures\"" in text
    counts = re.findall(r"\d+ answered wrong", text)
    assert len(counts) == 1, counts
    assert "0.5714285714285714" not in text and "0.571429" not in text
    assert "0.571 " not in text and "0.5714" in text
    header = text.split("### Every item\n", 1)[1].split("\n", 1)[0]
    assert header.startswith("| item | outcome |") and "| error |" not in header, header
    assert "score_effect: none" not in text and "declares no ready-made blocks" in text
    ungated = _ctx(records, findings=FINDINGS, parsimony=0.0, leaderboard=rows, history=rows,
                   best_score=score, mean_score=score).render()
    assert "0.5714285714285714" not in ungated and "Best score so far: 0.5714." in ungated


def test_a_judge_reason_said_the_same_on_two_items_reaches_the_rerun_in_both():
    """A line is settled only when it comes from no item (the settings, the templates, the
    ops' framing), never because it recurs: a judge that sends both items back with the
    same words has that reason shown at each item's re-run, and a choice both items share
    is still shown, with each item's input."""
    instructions = "Solve the question step by step and give one choice label."
    reason = "Sent back because: the answer ignores the boundary case entirely"
    choice = "C. Converges"
    knobs = [{"name": "solve__task", "kind": "str", "value": instructions, "frozen": False}]

    def record(item: str) -> dict:
        first = (f"History:\n- User: QUESTION {item}: does the series converge?\n{choice}\n"
                 f"{instructions}\nAnswer the latest user request.")
        return {"id": item, "passed": False, "score": 0.0, "prediction": "A",
                "metadata": {"question": f"QUESTION {item}: does the series converge?",
                             "choices": [{"label": "C", "text": "Converges"}]},
                "exchanges": [{"node": "solve", "prompt": first, "response": f"FIRST {item}"},
                              {"node": "judge", "prompt": f"FIRST {item}\n{instructions}",
                               "response": "reloop: ignores the boundary case"},
                              {"node": "solve", "prompt": f"{first}\n{reason}",
                               "response": f"SECOND {item}"}]}

    text = ProposalContext(item_records=[record("r1"), record("r2")], knobs=knobs).render()
    for item in ("r1", "r2"):
        block = text.split(f"### Item {item}", 1)[1].split("### Item ", 1)[0]
        rerun = block.split("#### Step ")[3]
        assert "Also in its prompt (not in its settings):" in rerun, rerun
        assert reason in rerun, rerun
        assert "Converges" in block  # the choice, said with the item's input
        assert instructions not in block  # the settings still hold the instructions
        # The ops' framing, in every one of the step's prompts and from no item, is not.
        assert "Answer the latest user request." not in block and "History:" not in block


def test_a_line_carrying_an_items_reply_is_shown_even_in_every_prompt_of_a_step():
    """Framing holds nothing from any item: a line quoting a step's reply is the item's
    evidence even when it is the same in every one of the step's prompts."""
    verdict = "the answer ignores the boundary case entirely"
    knobs = [{"name": "report__task", "kind": "str", "value": "Write the report.",
              "frozen": False}]

    def record(item: str) -> dict:
        return {"id": item, "passed": False, "score": 0.0, "prediction": "A",
                "metadata": {"question": f"QUESTION {item}"},
                "exchanges": [{"node": "judge", "prompt": f"QUESTION {item}",
                               "response": f"reloop\n{verdict}"},
                              {"node": "report", "prompt": f"Write the report.\n"
                                                           f"Prior verdict: {verdict}",
                               "response": f"REPORT {item}"}]}

    text = ProposalContext(item_records=[record("r1"), record("r2")], knobs=knobs).render()
    for item in ("r1", "r2"):
        block = text.split(f"### Item {item}", 1)[1].split("### Item ", 1)[0]
        assert f"Prior verdict: {verdict}" in block.split("#### Step ")[2], block


def _fix_only_records(source: str, reason: str = "misses the boundary case at zero"
                      ) -> list[dict]:
    """Two items where a fix step runs only after a judge, so every one of its prompts
    carries the judge's reason: from a code gate (the record's ``steps``, as
    `items.gate_visit` writes it) or from a model judge's JSON reply."""
    visit = ({"node": "check", "kind": "gate", "verdict": "reloop",
              "reason": " ".join(reason.split())}
             if source == "code gate" else None)

    def record(item: str) -> dict:
        exchanges = [{"node": "solve", "prompt": f"Q {item}", "response": f"ANS {item}"}]
        if source == "model judge JSON":
            exchanges.append({"node": "judge", "prompt": f"ANS {item}",
                              "response": json.dumps({"verdict": "reloop", "reason": reason})})
        exchanges.append({"node": "fix", "prompt": f"ANS {item}\nFeedback: {reason}",
                          "response": f"FIX {item}"})
        return {"id": item, "passed": False, "score": 0.0, "prediction": "A",
                "metadata": {"question": f"Q {item}"}, "exchanges": exchanges,
                **({"steps": [visit]} if visit else {})}

    return [record("r1"), record("r2")]


@pytest.mark.parametrize("source", ["code gate", "model judge JSON"])
def test_a_judge_reason_reaching_a_step_that_only_runs_after_it_is_shown(source):
    """The reason is in every one of the fix step's prompts, and is still each item's
    evidence: a code gate's reason (under the record's ``steps``) and the ``reason`` of a
    model judge's JSON reply both count as item data."""
    knobs = [{"name": "fix__task", "kind": "str", "value": "Fix it.", "frozen": False}]
    text = ProposalContext(item_records=_fix_only_records(source), knobs=knobs).render()
    for item in ("r1", "r2"):
        block = text.split(f"### Item {item}", 1)[1].split("### Item ", 1)[0]
        assert "Feedback: misses the boundary case at zero" in block, block


@pytest.mark.parametrize("source", ["code gate", "model judge JSON"])
def test_a_reason_over_two_lines_is_shown_line_by_line(source):
    """A code gate's reason is recorded as one line (`items.gate_visit`) but reaches the
    fix step over two: each prompt line holding a piece of it is the item's evidence."""
    reason = "misses the boundary case at zero\nand the residual check is skipped"
    knobs = [{"name": "fix__task", "kind": "str", "value": "Fix it.", "frozen": False}]
    text = ProposalContext(item_records=_fix_only_records(source, reason),
                           knobs=knobs).render()
    for item in ("r1", "r2"):
        block = text.split(f"### Item {item}", 1)[1].split("### Item ", 1)[0]
        assert "Feedback: misses the boundary case at zero" in block, block
        assert "and the residual check is skipped" in block, block


def test_a_piece_of_a_reply_quoted_in_every_prompt_is_shown():
    """A step quoting part of an earlier reply (a judge's reason, clipped) in every one of
    its prompts carries that item's evidence, not framing."""
    reply = "reloop: the answer ignores the boundary case and the zero input entirely"
    knobs = [{"name": "fix__task", "kind": "str", "value": "Fix it.", "frozen": False}]

    def record(item: str) -> dict:
        return {"id": item, "passed": False, "score": 0.0, "prediction": "A",
                "metadata": {"question": f"Q {item}"},
                "exchanges": [{"node": "judge", "prompt": f"Q {item}", "response": reply},
                              {"node": "fix", "prompt": f"Q {item}\nFeedback: the answer "
                                                        "ignores the boundary case",
                               "response": f"FIX {item}"}]}

    text = ProposalContext(item_records=[record("r1"), record("r2")], knobs=knobs).render()
    for item in ("r1", "r2"):
        block = text.split(f"### Item {item}", 1)[1].split("### Item ", 1)[0]
        assert "Feedback: the answer ignores the boundary case" in block, block


def test_framing_is_not_taken_for_a_piece_of_a_reply_that_uses_its_words():
    """The framing lines the real S4 runs put in every prompt stay framing although the
    replies use their words: a tail counts only after a short label and only when long
    enough, a datum only as whole words, and of a record's steps only a code gate's
    reason is item data (not a visit's node id). A real label still counts."""
    framing = ["- User: ## Question", "## Question", "## Independent",
               "## Option Comparison Notes", "## Files from earlier steps (independent_inputs)",
               "Before answering, read the notes and then weigh the hypotheses"]
    reason = "misses the boundary case at zero"

    def record(item: str) -> dict:
        prompt = "\n".join(["History:", *framing, f"Q {item}", f"Feedback: {reason}",
                             "Answer the latest user request."])
        return {"id": item, "passed": False, "score": 0.0, "prediction": "A",
                "metadata": {"question": f"Q {item}"},
                "steps": [{"node": "independent_inputs", "kind": "op", "output": "x"},
                          {"node": "check", "kind": "gate", "verdict": "reloop",
                           "reason": reason}],
                "exchanges": [{"node": "compare", "prompt": f"Q {item}",
                               "response": f"## Option C\nThe Question is subtle. {item}\n"
                                           "Independent of the bound, B holds; then weigh "
                                           "the hypotheses."},
                              {"node": "decide", "prompt": prompt, "response": f"D {item}"}]}

    text = ProposalContext(item_records=[record("r1"), record("r2")]).render()
    for item in ("r1", "r2"):
        decide = text.split(f"### Item {item}", 1)[1].split("### Item ", 1)[0] \
            .split("#### Step ")[2]
        assert f"Feedback: {reason}" in decide, decide
        for line in framing:
            assert line not in decide, (line, decide)


@pytest.mark.parametrize("reply_line", ["## Question", "Question", "Independent",
                                        "## Independent"])
def test_a_reply_line_that_is_a_bare_heading_or_word_is_no_item_data(reply_line):
    """A reply with a line that is only a heading or one word says nothing a framing line
    holding that word carries from the item: the framing stays out of "Also"."""
    framing = ["- User: ## Question", "## Question", "## Independent"]

    def record(item: str) -> dict:
        return {"id": item, "passed": False, "score": 0.0, "prediction": "A",
                "metadata": {"question": f"Q {item}"},
                "exchanges": [{"node": "compare", "prompt": f"Q {item}",
                               "response": f"{reply_line}\nB, because {item} is bounded."},
                              {"node": "decide",
                               "prompt": "\n".join([*framing, f"Q {item}"]),
                               "response": f"D {item}"}]}

    text = ProposalContext(item_records=[record("r1"), record("r2")]).render()
    for item in ("r1", "r2"):
        decide = text.split(f"### Item {item}", 1)[1].split("### Item ", 1)[0] \
            .split("#### Step ")[2]
        for line in framing:
            assert line not in decide, (reply_line, line, decide)


@pytest.mark.parametrize("source,said,quoted", [
    ("model judge", "## Issue: misses the boundary case at zero",
     "The judge sent it back with this note: ## Issue: misses the boundary case at zero"),
    ("code gate", "incomplete", "Feedback: incomplete"),
])
def test_a_reason_written_as_a_heading_or_one_word_is_still_shown(source, said, quoted):
    """Only a model's free-text line that is a bare label is left out of item data: a
    judge's reason written as a long heading (quoted inside a sentence, so not shown by
    name), and a code gate's one-word reason, are the item's evidence wherever they
    repeat."""
    knobs = [{"name": "fix__task", "kind": "str", "value": "Fix it.", "frozen": False}]

    def record(item: str) -> dict:
        exchanges = [{"node": "solve", "prompt": f"Q {item}", "response": f"ANS {item}"}]
        if source == "model judge":
            exchanges.append({"node": "judge", "prompt": f"ANS {item}",
                              "response": f"{said}\nThe rest of the working is fine."})
        exchanges.append({"node": "fix", "prompt": f"ANS {item}\n{quoted}",
                          "response": f"FIX {item}"})
        return {"id": item, "passed": False, "score": 0.0, "prediction": "A",
                "metadata": {"question": f"Q {item}"}, "exchanges": exchanges,
                **({"steps": [{"node": "check", "kind": "gate", "verdict": "reloop",
                               "reason": said}]} if source == "code gate" else {})}

    text = ProposalContext(item_records=[record("r1"), record("r2")], knobs=knobs).render()
    for item in ("r1", "r2"):
        block = text.split(f"### Item {item}", 1)[1].split("### Item ", 1)[0]
        assert quoted in block.split("#### Step ")[-1], block


def test_with_one_item_no_line_is_taken_for_framing():
    """Framing is told from an item's own text only across two or more items; with one,
    every line not in the settings or templates is shown."""
    hint = "Hint from the run: try the ratio test first"
    record = {"id": "r1", "passed": False, "score": 0.0, "prediction": "A",
              "metadata": {"question": "QUESTION r1"},
              "exchanges": [{"node": "solve", "prompt": f"QUESTION r1\n{hint}",
                             "response": "ANSWER r1"}]}
    assert hint in ProposalContext(item_records=[record]).render()


def test_the_items_legend_says_what_each_item_shows_and_a_reloop_reason_reaches_it():
    """The legend beside the items (`base.ITEM_FORM`) is what is rendered: each step's
    `Sees:` line, its reply, and only the part of its prompt its settings do not hold --
    so a reason a judge sent a step back with is shown at the re-run, and the step's own
    instructions are left to "Each step's settings"."""
    instructions = "Solve the question step by step and give one choice label."
    knobs = [{"name": "solve__task", "kind": "str", "value": instructions, "frozen": False}]

    def record(item: str) -> dict:
        question = f"QUESTION {item}: which statement holds for the stated hypotheses?"
        first = f"{question}\n{instructions}"
        judge = f"reloop: the answer to {item} ignores the boundary case entirely"
        return {"id": item, "passed": False, "score": 0.0, "prediction": "A",
                "metadata": {"question": question},
                "exchanges": [{"node": "solve", "prompt": first, "response": f"FIRST answer for {item}, with its working"},
                              {"node": "judge", "prompt": f"FIRST answer for {item}, with its working\n{instructions}",
                               "response": judge},
                              {"node": "solve", "prompt": f"{first}\nSent back because: "
                                                          f"the answer to {item} ignores the "
                                                          "boundary case entirely",
                               "response": f"SECOND {item}"}]}

    text = ProposalContext(item_records=[record("r1"), record("r2")], knobs=knobs).render()
    assert base.ITEM_FORM in text and "each whole" not in text
    item = text.split("### Item r1", 1)[1].split("### Item r2", 1)[0]
    steps = item.split("#### Step ")[1:]
    assert len(steps) == 3 and "Sees: the input" in steps[0] and "Sees: the input" in steps[2]
    assert "Sees: solve's reply" in steps[1]
    assert all(instructions not in s for s in steps), "the instructions are in the settings"
    assert "Rest of its prompt" not in text
    rerun = steps[2]
    assert "Also in its prompt (not in its settings):" in rerun
    assert "Sent back because: the answer to r1 ignores the boundary case entirely" in rerun
