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

import re
from pathlib import Path
from typing import Any

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
    assert "Nodes:" not in pipeline and base.REFERENCE_POINTER in pipeline
    assert text.index("Nodes:") > text.index(base.REFERENCE_HEADING)


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
    assert ids[:3] == ["f3", "f2", "f1"], ids  # one per finding in rank order, then more
    assert len([i for i in ids if i.startswith("f")]) == base.ITEMS_FAILING_SHOWN
    passing = [i for i in ids if i.startswith("p")]
    assert passing and passing[0] == "p4" and len(passing) <= base.ITEMS_PASSING_SHOWN
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


def test_the_request_stays_within_bounds_on_many_long_items():
    text = _ctx(_many(n_fail=20, n_pass=15, size=9000), findings=FINDINGS).render()
    assert len(text) < 120_000, len(text)
    items_part = text.split("## Trajectories of the candidate being edited", 1)[1].split(
        base.REFERENCE_HEADING, 1)[0]
    assert len(items_part) < base.ITEMS_SHOWN_BUDGET + 12_000, len(items_part)


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
