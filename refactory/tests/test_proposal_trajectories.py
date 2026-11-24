"""What the optimizer sees of the edited candidate's train items, and what it said.

``ProposalContext.render`` shows a per-epoch sample of train items in full (input,
each step's output in graph order, expected vs predicted), within a budget; and
``AutoresearchOptimizer`` records every model call to ``optimizer_trace.jsonl``.
Each test fails when the behaviour it names is reverted.
"""

from __future__ import annotations

import re

import json
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh.loop import Loop, LoopConfig
from refactory_dsh.optim import ProposalContext, base
from refactory_dsh.optim.random_optimizer import AutoresearchOptimizer
import structure_fuzz as sf

NODES = [{"id": "compare", "kind": "agent"}, {"id": "precision", "kind": "agent"},
         {"id": "decide", "kind": "agent"}]


def _record(item: str, *, passed: bool = False, error: str | None = None,
            outputs: dict[str, str] | None = None) -> dict[str, Any]:
    """A LiveMath-like train record: a question, four choices, three steps."""
    outputs = outputs or {"compare": f"{item}: options A and C both look plausible",
                          "precision": f"{item}: C requires compactness, A does not",
                          "decide": "Answer: C" if passed else "Answer: A"}
    exchanges = [{"node": node, "prompt": f"solve {item} ({node})", "response": outputs[node]}
                 for node in ("compare", "precision", "decide") if node in outputs]
    return {
        "id": item, "passed": passed, "score": 1.0 if passed else 0.0,
        "error_text": error, "error_kind": "RuntimeError" if error else None,
        "exchanges": [] if error else exchanges,
        "prediction": None if error else ("C" if passed else "A"),
        "label": "C", "label_text": "Every bounded sequence has a convergent subsequence",
        "fail_reason": None if passed or error else "MCQ=0: predicted 'A' but expected 'C'",
        "metadata": {"question": f"Which statement holds in {item}?",
                     "choices": [{"label": "A", "text": "Every sequence converges"},
                                 {"label": "B", "text": "No sequence converges"},
                                 {"label": "C", "text": "Every bounded sequence has a convergent subsequence"},
                                 {"label": "D", "text": "None of the above"}]},
    }


def _render(records: list[dict[str, Any]], **fields: Any) -> str:
    return ProposalContext(nodes=NODES, item_records=records, **fields).render()


def _shown(text: str) -> list[str]:
    return [line.split()[2] for line in text.splitlines() if line.startswith("### Item ")]


def test_a_failed_item_is_shown_in_full():
    """The question and choices, each step's output in the record's order, the
    expected label (its choice line says its text, once), the prediction and the
    reason."""
    text = _render([_record("i1")])
    section = text[text.index("## Trajectories of the candidate being edited"):]
    for expected in ("Input (question):", "Which statement holds in i1?",
                     "C. Every bounded sequence has a convergent subsequence",
                     "Expected: C; Given: A", "Grader: MCQ=0: predicted 'A' but expected 'C'"):
        assert expected in section, expected
    assert section.count("Every bounded sequence has a convergent subsequence") == 1
    steps = [section.index(f"Step `{node}`") for node in ("compare", "precision", "decide")]
    assert steps == sorted(steps)


def test_steps_keep_the_order_a_forked_graph_ran_them(tmp_path, monkeypatch):
    """A real livemath/skill_graph pass (compare, then precision and hypotheses in a
    fork, then decide), rendered with the mode's own describe(), whose nodes are
    sorted by name: the steps show in the order the pass ran them."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_livemath_blocks import _NodeScriptedModel, _patch_task

    from refactory_packages.livemath.skill_graph import SkillGraph

    _patch_task(monkeypatch)
    mode = SkillGraph()
    model = _NodeScriptedModel({"compare": "COMPARE notes", "precision": "PRECISION notes",
                                "hypotheses": "HYPOTHESES notes", "decide": "<answer>A</answer>"})
    (tmp_path / "pass").mkdir()
    result = mode.run(tmp_path / "pass", model=model)
    assert result.exit_ok, result.error
    record = next(r for r in result.item_records if r.get("exchanges"))
    ran = [e["node"] for e in record["exchanges"]]
    assert ran == ["compare", "precision", "hypotheses", "decide"]
    nodes = mode.describe()["nodes"]
    assert [n["id"] for n in nodes] == sorted(n["id"] for n in nodes), "setup: describe sorts"
    text = ProposalContext(nodes=nodes, item_records=[record]).render()
    section = text[text.index("## Trajectories"):]
    order = [section.index(f"Step `{node}`") for node in ran]
    assert order == sorted(order), "the order the pass ran, not alphabetical"


def _section(text: str) -> str:
    from refactory_dsh.optim.base import REFERENCE_HEADING

    start = text.index("## Trajectories of the candidate being edited")
    return text[start:text.index("\n" + REFERENCE_HEADING, start)]


def _livemath_sized(item: str, *, passed: bool = False) -> dict[str, Any]:
    """A record the size of a real LiveMath skill_graph item: a 1.5k question, five
    400-character choices, four steps of 2.8k each."""
    record = _record(item, passed=passed)
    record["metadata"] = {"question": f"Q{item} " + "q" * 1500,
                          "choices": [{"label": label, "text": label * 400} for label in "ABCDE"]}
    record["exchanges"] = [{"node": node, "prompt": "p", "response": f"{node} START " + "r" * 2800 + " END"}
                           for node in ("compare", "precision", "hypotheses", "decide")]
    return record


def test_a_set_that_fits_is_shown_whole_in_the_proposal_call():
    """Within the budget: every item, whole (a 9000-character step output is not
    clipped), wrong first, then failed with an error (with or without recorded
    steps), then passed, each by id."""
    records = ([_record(f"w{i:02d}") for i in range(12)]
               + [_record(f"p{i}", passed=True) for i in range(5)]
               + [_record("e1", error="RuntimeError: worker exited with code 1")])
    stepped = {**_record("e2"), "error_text": "RuntimeError: decide crashed",
               "error_kind": "RuntimeError"}
    long = "HEAD " + "x" * 9000 + " TAIL"
    records[0] = _record("w00", outputs={"compare": long, "precision": "p", "decide": "d"})
    text = _render(records + [stepped])
    assert _shown(text) == [f"w{i:02d}" for i in range(12)] + ["e1", "e2"] + [f"p{i}" for i in range(5)]
    assert "19 train item(s): 5 passed, 12 answered wrong, 2 failed with an error" in text
    assert text.count("12 answered wrong") == 1, "the counts are said once"
    assert "### Item e1 — failed with an error" in text
    # Its error is said once, under "Errors, by cause"; the item points there.
    item = text.split("### Item e1 — failed with an error", 1)[1].split("### Item ", 1)[0]
    assert 'Error: cause 1 under "Errors, by cause" above' in item, item
    assert text.count("RuntimeError: worker exited with code 1") == 1
    assert f"Every item. {base.ITEM_FORM}" in text and long in text and "chars omitted" not in text
    assert "### Item e2 — failed with an error" in text
    assert text.count("RuntimeError: decide crashed") == 1 and "Error: cause 2 under" in text


def test_an_item_is_shown_whole_and_only_a_text_too_large_to_send_is_cut():
    from refactory_dsh.optim.base import ITEM_TEXT_MAX

    long = "HEAD " + "x" * 9000 + " TAIL"
    text = _render([_record("w1", outputs={"compare": long, "precision": "p", "decide": "d"})],
                   evidence_budget=4000)
    assert long in _section(text), "longer than the budget, still whole"
    huge = "HEAD " + "x" * ITEM_TEXT_MAX + " TAIL"
    section = _section(_render([_record("w1", outputs={"compare": huge, "precision": "p",
                                                       "decide": "d"})], evidence_budget=4000))
    # Cut in place with the count only (no path); nothing else said about it in the request.
    assert "HEAD" in section and "TAIL" in section and re.search(r"\[… [\d,]+ chars omitted\]",
                                                                section)
    from refactory_dsh.optim.base import trajectory_groups
    (group,) = trajectory_groups([_record("w1", outputs={"compare": long, "precision": "p",
                                                          "decide": "d"})], 4000)
    assert len(group[0][1]) <= 4000


class _ReaderClient:
    """Readers answer with one failure pattern naming their group's items; the
    proposal call is recorded. ``fail_group`` makes that reader call fail."""

    def __init__(self, fail_group: int | None = None) -> None:
        self.fail_group = fail_group
        self.readers: list[str] = []
        self.proposals: list[str] = []
        self.lock = __import__("threading").Lock()

    def complete(self, prompt: str, *, system: str = "") -> str:
        from refactory_dsh.optim.random_optimizer import READER_SYSTEM

        if system == READER_SYSTEM:
            group = int(prompt.split("group ", 1)[1].split(" ", 1)[0])
            with self.lock:
                self.readers.append(prompt)
            if group == self.fail_group:
                raise RuntimeError("503 overloaded")
            items = [line.split()[2] for line in prompt.splitlines() if line.startswith("### Item ")]
            return json.dumps({"failure_patterns": [{
                "pattern": f"PATTERN-{group}", "step": "decide", "items": items,
                "evidence": [f"QUOTE-{group}"]}], "success_patterns": []})
        self.proposals.append(prompt)
        return '{"diagnosis": "d", "proposals": []}'


def _sized_records() -> list[dict[str, Any]]:
    return ([_livemath_sized(f"w{i:02d}") for i in range(25)]
            + [_livemath_sized(f"p{i:02d}", passed=True) for i in range(10)])


def test_a_large_set_is_read_in_groups_and_the_findings_reach_the_proposal(tmp_path):
    """35 LiveMath-sized items: readers go through whole-item groups, each within the
    budget, every item exactly once, concurrently; their findings, a table of every
    item and some items in full reach the proposal call; the trace records each
    reader call; a retry reads nothing again."""
    from refactory_dsh.optim.base import TRAJECTORY_BUDGET, trajectory_groups

    records = _sized_records()
    groups = trajectory_groups(records)
    assert len(groups) > 1
    assert all(len("\n\n".join(block for _i, block in g)) <= TRAJECTORY_BUDGET for g in groups)
    covered = [item for g in groups for item, _block in g]
    assert sorted(covered) == sorted(r["id"] for r in records) and len(covered) == len(set(covered))

    client = _ReaderClient()
    optimizer = AutoresearchOptimizer(client)
    optimizer.trace_path = tmp_path / "optimizer_trace.jsonl"
    ctx = ProposalContext(nodes=NODES, item_records=records, epoch=1, generation=2)
    optimizer.propose(ctx, 1, [])
    assert len(client.readers) == len(groups)
    read = sorted(line.split()[2] for p in client.readers for line in p.splitlines()
                  if line.startswith("### Item "))
    assert read == sorted(covered), "every item read exactly once"
    prompt = client.proposals[0]
    for index in range(1, len(groups) + 1):
        assert f"PATTERN-{index}" in prompt and f'"QUOTE-{index}"' in prompt
    table = prompt.split("### Every item", 1)[1].split("### Some items in full", 1)[0]
    assert all(f"| {r['id']} |" in table for r in records)
    assert "### Item " in prompt.split("### Some items in full", 1)[1]
    assert len(_section(prompt)) <= TRAJECTORY_BUDGET
    calls = [json.loads(line) for line in optimizer.trace_path.read_text().splitlines()]
    assert [c["kind"] for c in calls].count("reader") == len(groups)
    assert calls[-1]["kind"] == "proposal"
    optimizer.propose(ctx, 1, [])
    assert len(client.readers) == len(groups), "a retry reads no group again"


def test_a_failed_reader_leaves_the_proposal_with_what_it_has(tmp_path):
    records = _sized_records()
    client = _ReaderClient(fail_group=2)
    optimizer = AutoresearchOptimizer(client)
    optimizer.trace_path = tmp_path / "optimizer_trace.jsonl"
    optimizer.propose(ProposalContext(nodes=NODES, item_records=records), 1, [])
    prompt = client.proposals[0]
    assert "PATTERN-1" in prompt and "PATTERN-2" not in prompt
    assert "(Group 2 -- items" in prompt and "could not be read: RuntimeError: 503 overloaded" in prompt
    failed = [json.loads(line) for line in optimizer.trace_path.read_text().splitlines()
              if json.loads(line).get("group") == 2]
    assert failed and "503 overloaded" in failed[0]["error"]


def test_a_large_set_without_readers_still_lists_every_item():
    """An optimizer that calls no reader still gets every item in the table."""
    records = _sized_records()
    text = _render(records)
    assert "No reader went through these items" in text
    table = text.split("### Every item", 1)[1]
    assert all(f"| {r['id']} |" in table for r in records)


def test_every_proposal_sees_the_same_complete_set():
    records = [_record(f"w{i:02d}") for i in range(30)] + [_record("p1", passed=True)]
    shown = _shown(_render(records, epoch=1))
    assert shown == [f"w{i:02d}" for i in range(30)] + ["p1"]
    assert shown == _shown(_render(records, epoch=2, generation=3, attempt=2))


def test_repeats_show_each_item_once_and_a_zero_prediction_shows():
    """With repeats each item is shown once, from its first repeat, with no repeat
    label; a prediction of 0 is rendered, not dropped."""
    one = {**_record("i1"), "repeat": 1}
    two = {**_record("i1", outputs={"compare": "SECOND", "precision": "p", "decide": "d"}), "repeat": 2}
    for records in ([one, two], [two, one]):
        text = _render(records)
        assert _shown(text) == ["i1"]
        assert "SECOND" not in _section(text) and "repeat" not in _section(text)
    zero = {**_record("i2"), "prediction": 0}
    assert "Given: 0" in _render([zero])


def test_the_loop_hands_the_optimizer_train_records_only(tmp_path, monkeypatch):
    """After a run with val and test parts, the context's records and the items it
    shows in full are all train items."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_skillopt import TEST_IDS, TRAIN_IDS, VAL_IDS, _patch_mode, _write_parts
    from test_val_gate import LevelProbe

    from refactory_dsh.outerloop import ValidationGate

    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(lambda run, cand: 0.5))
    with _patch_mode(LevelProbe):
        loop = Loop("skillopt_probe/probe", LoopConfig(
            epochs=1, batch_size=1, optimizer="random", seed=7, runs_root=tmp_path / "run",
            allowed_ops=("set_knob",), holdout=0.0, val_gate=True,
            parts_manifest=str(_write_parts(tmp_path, test=True))), project=Path("."))
        loop.run()
        context = loop._context()
    ids = {str(record["id"]) for record in context.item_records}
    assert ids and ids <= set(TRAIN_IDS)
    shown = set(_shown(context.render()))
    assert shown and shown <= set(TRAIN_IDS)
    assert not shown & (set(VAL_IDS) | set(TEST_IDS))


class _Client:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)

    def complete(self, prompt: str, *, system: str = "") -> str:
        reply = self.replies.pop(0) if self.replies else '{"diagnosis": "nothing", "proposals": []}'
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_every_autoresearch_call_is_recorded(tmp_path):
    """System, prompt, reply, epoch, generation, parsed proposals and error, one line
    per call, written as it happens, including a call that failed (each of its three
    attempts: `AutoresearchOptimizer._ask_proposal`)."""
    reply = json.dumps({"diagnosis": "the decide step trusts compare",
                        "proposals": [{"edit": {"op": "set_knob", "name": "k", "value": 1}}]})
    optimizer = AutoresearchOptimizer(_Client([reply, *[RuntimeError("503 overloaded")] * 3]))
    optimizer.trace_path = tmp_path / "optimizer_trace.jsonl"
    ctx = ProposalContext(epoch=2, generation=5, nodes=NODES, item_records=[_record("i1")])
    assert len(optimizer.propose(ctx, 2, [])) == 1
    with pytest.raises(RuntimeError):
        optimizer.propose(ctx, 2, [])
    calls = [json.loads(line) for line in optimizer.trace_path.read_text().splitlines()]
    assert len(calls) == 4
    first, *failed_attempts = calls
    failed = failed_attempts[-1]
    assert [call["attempt"] for call in calls] == [1, 1, 2, 3]
    assert (first["epoch"], first["generation"], first["proposals"], first["error"]) == (2, 5, 1, None)
    assert first["reply"] == reply and first["system"] == optimizer._system
    assert "Which statement holds in i1?" in first["prompt"]
    assert first["diagnosis"] == "the decide step trusts compare"
    assert failed["reply"] is None and "503 overloaded" in failed["error"]


def test_a_trace_that_cannot_be_written_keeps_the_proposals(tmp_path):
    reply = json.dumps({"diagnosis": "d",
                        "proposals": [{"edit": {"op": "set_knob", "name": "k", "value": 1}}]})
    optimizer = AutoresearchOptimizer(_Client([reply]))
    optimizer.trace_path = tmp_path  # a directory: appending to it fails
    with pytest.warns(RuntimeWarning, match="could not record"):
        proposals = optimizer.propose(ProposalContext(), 1, [])
    assert len(proposals) == 1


def test_the_trace_is_written_by_a_run_and_survives_a_resume(tmp_path):
    root = tmp_path / "run"

    def run(epochs: int, resume: bool) -> Loop:
        loop = Loop("circle_packing/adaptive", LoopConfig(
            epochs=epochs, batch_size=2, seed=5, optimizer="autoresearch", runs_root=root,
            resume=resume), project=Path("."), model=_Client([]))
        loop.run()
        return loop

    run(1, False)
    trace = root / "optimizer_trace.jsonl"
    before = trace.read_text().splitlines()
    assert before and all(json.loads(line)["epoch"] == 1 for line in before)
    run(2, True)
    after = trace.read_text().splitlines()
    assert after[:len(before)] == before, "a resume keeps what was recorded"
    assert any(json.loads(line)["epoch"] == 2 for line in after[len(before):])
    # A fresh run reusing the directory (as one sharing an enclosing budget may, with
    # no ledger of its own) starts the trace over, as it does the journal.
    from refactory_dsh.budgets import LEDGER
    (root / LEDGER).unlink()
    run(1, False)
    assert all(json.loads(line)["epoch"] == 1 for line in trace.read_text().splitlines()), (
        "a fresh run in the same directory starts the trace over")


def test_each_proposal_attempt_renders_with_its_own_attempt_number():
    """A retry after a refusal is handed the attempt number."""
    from refactory_dsh.optim import Candidate, Optimizer, Proposal

    seen: list[int] = []

    class Refused(Optimizer):
        name = "refused"

        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            seen.append(ctx.attempt)
            return [Proposal(edit={"op": "set_knob", "name": "missing", "value": 1},
                             parent=parents[0].id)]

    seed = Candidate.from_state({"graph": {"nodes": {}, "edges": []}, "knobs": {}})
    Refused(k=3).propose_valid(ProposalContext(), 1, parents=[seed])
    assert seen == [1, 2, 3]


def test_the_loop_shows_every_train_item_in_every_request(tmp_path):
    """Through the real Loop, every proposal request (a retry after a refusal and a
    later generation included) shows every train item of the candidate edited."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_skillopt import TRAIN_IDS, _patch_mode, _write_parts
    from test_val_gate import LevelProbe

    from refactory_dsh.optim import Optimizer, Proposal

    class RefusedOnce(Optimizer):
        name = "refused_once"

        def __init__(self) -> None:
            super().__init__(k=1)
            self.seen: list[list[str]] = []

        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            self.seen.append(_shown(ctx.render()))
            if len(self.seen) == 1:
                return [Proposal(edit={"op": "no_such_op"}, parent=parents[0].id)]
            return [Proposal(edit={"op": "set_knob", "name": "level", "value": len(self.seen)},
                             parent=parents[0].id)]

    proposer = RefusedOnce()
    with _patch_mode(LevelProbe):
        loop = Loop("skillopt_probe/probe", LoopConfig(
            epochs=2, batch_size=1, optimizer="random", seed=7, runs_root=tmp_path / "run",
            allowed_ops=("set_knob",), holdout=0.0, attempts=3,
            parts_manifest=str(_write_parts(tmp_path))), project=Path("."))
        loop.optimizer = proposer
        loop.run()
    assert len(proposer.seen) >= 3
    for shown in proposer.seen:
        assert sorted(shown) == sorted(TRAIN_IDS), shown


def _traces_section(text: str) -> str:
    start = text.index("## Evidence from the traces")
    return text[start:text.index("\n## ", start + 5)]


def test_the_traces_carry_no_worker_receipts_or_paths(tmp_path, monkeypatch):
    """Through the loop: the evidence from the traces has the measurement summary
    and no receipts, run directories or leaderboard copy, and stays small; a
    livemath/skill_graph-like gradient with 140 receipts per candidate shrinks to a
    few lines."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_skillopt import _patch_mode, _write_parts
    from test_val_gate import LevelProbe

    with _patch_mode(LevelProbe):
        loop = Loop("skillopt_probe/probe", LoopConfig(
            epochs=2, batch_size=2, optimizer="random", seed=7, runs_root=tmp_path / "run",
            allowed_ops=("set_knob",), holdout=0.0, parts_manifest=str(_write_parts(tmp_path))),
            project=Path("."))
        loop.run()
        section = _traces_section(loop._context().render())
    assert str(tmp_path) not in section and "workdir" not in section
    assert "measurement_summary" in section and '"trial_runs":' in section
    assert len(section) < 4000

    receipts = [{"worker_id": f"w-{i:04d}", "node_id": "compare", "role": "solver",
                 "state": "done", "actual_route": {"model": "m"},
                 "runtime": {"dir": f"/runs/abc/runs/xyz/.refactory/workers/w-{i:04d}"},
                 "usage": {"tokens": 10}} for i in range(140)]
    row = {"id": "c1", "score": 0.4, "measurements": {"trials": [
        {"trial": 1, "workdir": "/runs/abc/runs/c1", "workers": receipts}]}}
    gradient = {"epoch": 2, "leaderboard": [row] * 8,
                # A strategy's own trace key holding rows: cleaned the same way.
                "recent": [row] * 2,
                "knob_attribution": {"compare__system_prompt": {"tried": {
                    "You are an expert. " + "x" * 3000: {"n": 1, "mean": 0.4, "max": 0.4}}}},
                "measurement_summary": {"trial_runs": 8, "failed": 1, "timeouts": 1,
                                        "failures_by_cause": {"CompletionTimeout": 1}}}
    section = _traces_section(ProposalContext(traces=[gradient]).render())
    assert "w-0001" not in section and "/runs/" not in section and "workers" not in section
    assert "3019 chars" in section and "CompletionTimeout" in section
    assert len(section) < 1500


def test_groups_never_exceed_the_budget_with_uneven_items():
    from refactory_dsh.optim.base import trajectory_groups

    records = [_record(f"w{i}", outputs={"compare": "y" * size, "precision": "p", "decide": "d"})
               for i, size in enumerate((6000, 6000, 6000, 1000))]
    groups = trajectory_groups(records, 10000)
    assert len(groups) >= 2
    assert all(len("\n\n".join(block for _i, block in g)) <= 10000 for g in groups)
    assert [item for g in groups for item, _b in g] == ["w0", "w1", "w2", "w3"]



class _WordyReaderClient(_ReaderClient):
    """Readers that ramble: 50 patterns each, long texts and many long quotes."""

    def complete(self, prompt: str, *, system: str = "") -> str:
        from refactory_dsh.optim.random_optimizer import READER_SYSTEM

        if system != READER_SYSTEM:
            return super().complete(prompt, system=system)
        with self.lock:
            self.readers.append(prompt)
        items = [line.split()[2] for line in prompt.splitlines() if line.startswith("### Item ")]
        return json.dumps({"failure_patterns": [{
            "pattern": f"WORDY-{k} " + "§" * 5000, "step": "decide" * 50, "items": items * 10,
            "evidence": ["¤" * 3000] * 8} for k in range(50)], "success_patterns": []})


def test_wordy_readers_are_kept_within_the_budget(tmp_path):
    from refactory_dsh.optim.base import READER_PATTERNS, TRAJECTORY_BUDGET

    client = _WordyReaderClient()
    optimizer = AutoresearchOptimizer(client)
    optimizer.propose(ProposalContext(nodes=NODES, item_records=_sized_records()), 1, [])
    section = _section(client.proposals[0])
    findings = section.split("### Every item", 1)[0]
    assert len(findings) <= TRAJECTORY_BUDGET
    assert section.count("WORDY-") <= READER_PATTERNS * len(client.readers)
    assert "§" * 500 not in section and "¤" * 300 not in section
    assert "more pattern(s) not shown: the findings are kept within" in section
    table = section.split("### Every item", 1)[1]
    assert all(f"| {r['id']} |" in table for r in _sized_records()), "the table survives"


def test_reader_findings_are_kept_and_reused_by_a_resumed_run(tmp_path):
    records = _sized_records()
    first = _ReaderClient()
    optimizer = AutoresearchOptimizer(first)
    optimizer.trace_path = tmp_path / "optimizer_trace.jsonl"
    optimizer.propose(ProposalContext(nodes=NODES, item_records=records), 1, [])
    assert list((tmp_path / "reader_findings").glob("*.json"))
    again = _ReaderClient()
    resumed = AutoresearchOptimizer(again)  # a fresh process: no in-memory cache
    resumed.trace_path = tmp_path / "optimizer_trace.jsonl"
    resumed.propose(ProposalContext(nodes=NODES, item_records=records), 1, [])
    assert again.readers == [], "a resumed run pays for no group again"
    assert "PATTERN-1" in again.proposals[0]


def test_optimizer_and_reader_calls_run_without_tools(tmp_path, monkeypatch):
    """Every optimizer call goes out with tool_policy none: the worker receipt
    records it for the proposal and for each reader. The run's own client keeps its
    tools."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_livemath_blocks import _make_route_client

    from refactory_dsh.model import DshModelClient

    def fake_complete(self, prompt, *, system="", workdir=None, target_workdir=None):  # type: ignore[no-untyped-def]
        from refactory_dsh.optim.random_optimizer import READER_SYSTEM
        if system == READER_SYSTEM:
            return json.dumps({"failure_patterns": [], "success_patterns": []})
        return '{"diagnosis": "d", "proposals": []}'

    monkeypatch.setattr(DshModelClient, "_complete", fake_complete)
    client = _make_route_client()
    client._dsh_home = tmp_path / "home"
    client._budget = None
    optimizer = AutoresearchOptimizer(client)
    optimizer.bind_model(client, workdir=tmp_path / "proposer")
    assert optimizer._client._tool_policy == "none"
    assert getattr(client, "_tool_policy", "") == ""
    optimizer.propose(ProposalContext(nodes=NODES, item_records=_sized_records()), 1, [])
    receipts = [json.loads(p.read_text()) for p in (tmp_path / "proposer").glob("**/.refactory/workers/*.json")]
    assert len(receipts) >= 2, "the readers and the proposal each made a call"
    assert all(r.get("requested_tool_policy") == "none" for r in receipts), receipts


class _ManyShortPatterns(_ReaderClient):
    def complete(self, prompt: str, *, system: str = "") -> str:
        from refactory_dsh.optim.random_optimizer import READER_SYSTEM

        if system != READER_SYSTEM:
            return super().complete(prompt, system=system)
        with self.lock:
            self.readers.append(prompt)
        return json.dumps({"failure_patterns": [
            {"pattern": f"SHORT-{k}", "step": "decide", "items": ["w00"], "evidence": ["x"]}
            for k in range(50)], "success_patterns": []})


def test_each_reader_contributes_a_bounded_number_of_patterns():
    from refactory_dsh.optim.base import READER_PATTERNS

    client = _ManyShortPatterns()
    AutoresearchOptimizer(client).propose(
        ProposalContext(nodes=NODES, item_records=_sized_records()), 1, [])
    section = _section(client.proposals[0])
    assert section.count("SHORT-") == READER_PATTERNS * len(client.readers)


def test_a_retry_reuses_readers_even_without_a_run_directory():
    client = _ReaderClient()
    optimizer = AutoresearchOptimizer(client)  # no trace path, so nothing on disk
    ctx = ProposalContext(nodes=NODES, item_records=_sized_records())
    optimizer.propose(ctx, 1, [])
    read = len(client.readers)
    optimizer.propose(ctx, 1, [])
    assert len(client.readers) == read


class _Routed:
    """A model client whose optimizer role is routed to ``backend``."""

    def __init__(self, backend: str) -> None:
        self.backend = backend

    def for_node(self, node_id: str, role: str = "") -> "_Routed":
        return self

    def route(self) -> dict:
        return {"worker_backend": self.backend}

    def complete(self, prompt: str, *, system: str = "") -> str:
        return '{"diagnosis": "d", "proposals": []}'


@pytest.mark.parametrize("backend", ["codex", "openshell"])
def test_an_optimizer_routed_where_tools_cannot_be_removed_is_refused_at_launch(backend):
    """The autoresearch optimizer's calls run without tools; a route that cannot
    enforce that is refused before anything runs, saying how to route it."""
    from refactory_dsh.loop import LoopError

    loop = Loop("circle_packing/adaptive", LoopConfig(optimizer="autoresearch"),
                project=Path("."), model=_Routed(backend))
    with pytest.raises(LoopError, match=rf"dsh, claude_code backends.*routed to {backend}.*--role-routes"):
        loop._check_optimizer_route()
    for allowed in ("dsh", "claude_code"):
        Loop("circle_packing/adaptive", LoopConfig(optimizer="autoresearch"), project=Path("."),
             model=_Routed(allowed))._check_optimizer_route()


def test_the_launch_refusal_comes_before_any_rollout(tmp_path):
    from refactory_dsh.loop import LoopError

    loop = Loop("circle_packing/adaptive", LoopConfig(optimizer="autoresearch",
                runs_root=tmp_path / "run", epochs=1), project=Path("."), model=_Routed("codex"))
    with pytest.raises(LoopError, match="routed to codex"):
        loop.run()
    assert loop.rollouts == 0 and not (tmp_path / "run" / "runs").exists()


def test_text_only_keeps_the_budget_guard():
    """A token-limited run wraps a client without usage telemetry in a guard that
    refuses its completions; taking the no-tools copy keeps the guard."""
    from refactory_dsh.budgets import BudgetExhausted, _UnmeteredModel
    from refactory_dsh.optim.random_optimizer import text_only

    class Inner:
        policy = ""

        def with_tool_policy(self, policy: str) -> "Inner":
            copy = Inner()
            copy.policy = policy
            return copy

    class Budget:
        def require_usage_capability(self, available: bool) -> None:
            pass

    guarded = text_only(_UnmeteredModel(Inner(), Budget()))
    assert isinstance(guarded, _UnmeteredModel) and guarded._model.policy == "none"
    with pytest.raises(BudgetExhausted):
        guarded.complete("p")


def test_a_failed_readers_error_is_clipped():
    from refactory_dsh.optim.base import _trajectories, trajectory_groups

    records = _sized_records()
    groups = trajectory_groups(records)
    failure = {"group": 1, "items": [i for i, _b in groups[0]], "error": "E" * 5000}
    text = "\n".join(_trajectories(records, findings=[], failures=[failure]))
    line = next(l for l in text.splitlines() if l.startswith("(Group 1"))
    assert "E" * 400 not in line and "could not be read" in line


def _full(text: str) -> list[str]:
    return _shown(text.split("### Some items in full", 1)[1])


def test_items_the_readers_cited_are_shown_in_full_first_one_per_pattern():
    """The items behind the readers' patterns are the ones shown in full, not the
    first by id: one per pattern (failure patterns first, an item already shown is not
    shown twice), and as many as there are patterns, each whole."""
    records = _sized_records()
    findings = [
        {"group": 1, "items": [],
         "failure_patterns": [
             {"pattern": "A", "step": "s", "items": ["w20", "w21"], "evidence": ["qa"]},
             {"pattern": "B", "step": "s", "items": ["w23", "nope"], "evidence": ["qb"]}],
         "success_patterns": [{"pattern": "S", "step": "s", "items": ["p09"], "evidence": ["qs"]}]},
        {"group": 2, "items": [],
         "failure_patterns": [{"pattern": "C", "step": "s", "items": ["w20", "w24"],
                               "evidence": ["qc"]}],
         "success_patterns": []},
    ]
    text = _render(records, reader_findings=findings)
    full = _full(text)
    # One failure per failure pattern (an item already shown is not shown twice), then
    # one passing item per success pattern for contrast; each whole.
    expected = ["w20", "w23", "w24", "p09"]
    assert full == expected, full
    assert "the readers cited, one per finding in rank order" in text
    assert "chars omitted" not in _section(text)


_CITING = [
    {"group": 1, "items": [],
     "failure_patterns": [
         {"pattern": "A", "step": "s", "items": ["w20"], "evidence": ["qa"]},
         {"pattern": "B", "step": "s", "items": ["w23"], "evidence": ["qb"]},
         {"pattern": "C", "step": "s", "items": ["w24"], "evidence": ["qc"]}],
     "success_patterns": [{"pattern": "S", "step": "s", "items": ["p09"], "evidence": ["qs"]}]},
]


def _whole_size(records: list[dict[str, Any]], *ids: str, after: tuple[str, ...] = ()) -> int:
    """What items ``ids`` take shown whole after the items ``after``: a step's ``Ran as:``
    line an earlier item said is not said again (`evidence.said_once`)."""
    by_id = {r["id"]: r for r in records}
    how: set[tuple[str, str]] = set()
    for i in after:
        base._shown_size(by_id[i], None, None, how)
    return sum(base._shown_size(by_id[i], None, None, how) for i in ids)


def test_without_a_room_every_item_the_rule_picks_is_shown_whatever_their_size():
    """A request within its room is the rule unchanged: one item per finding, nothing
    named as left out."""
    records = _sized_records()
    text = _render(records, reader_findings=_CITING)
    assert len(text) < base.PROPOSAL_ROOM
    assert _full(text) == ["w20", "w23", "w24", "p09"]
    assert "left out so this request fits" not in text


def test_a_room_keeps_the_picked_items_that_fit_in_turn_and_names_the_rest():
    """With a room, the picked items are taken failure, success, failure, ... in rank
    order until the next would not fit; the rest are named by id (each is still in the
    table), and the section's own count says how many are shown."""
    records = _sized_records()
    room = _whole_size(records, "w20", "p09") + 10
    text = _render(records, reader_findings=_CITING, shown_items_room=room)
    assert _full(text) == ["w20", "p09"]
    section = text.split("### Some items in full", 1)[1]
    assert section.startswith("\n1 failing item(s)")
    note = next(l for l in section.splitlines() if "left out so this request fits" in l)
    assert f"({room} characters were left for items in full)" in note
    assert "w23 (failed), w24 (failed)" in note
    table = text.split("### Every item", 1)[1].split("### Some items in full", 1)[0]
    assert all(f"| {i} |" in table for i in ("w23", "w24"))


def test_a_room_smaller_than_the_first_item_cuts_its_longest_texts_to_fit():
    """The first item no longer stays whatever its size: it gives way, its longest texts
    (the step replies) cut to one length, the input (shorter) whole; with less room than
    it takes even at `SHOWN_TEXT_FLOOR`, it is named with the rest."""
    records = _sized_records()
    room = _whole_size(records, "w20") - 4000
    text = _render(records, reader_findings=_CITING, shown_items_room=room)
    assert _full(text) == ["w20"]
    block = text.split("### Item w20", 1)[1]
    assert len("### Item w20" + block.split("\n## ", 1)[0]) <= room
    assert block.count("chars omitted]") == 4 and "Qw20 " + "q" * 1500 in block
    assert _full(_render(records, reader_findings=_CITING, shown_items_room=1)) == []


def test_the_room_is_not_the_reader_findings_room():
    """A room larger than the reader findings' share of the budget is the room used (the
    findings block's own room once shadowed it)."""
    records = _sized_records()
    room = _whole_size(records, "w20", "p09", "w23", "w24") + 10
    assert room > int(base.TRAJECTORY_BUDGET * base.READER_FINDINGS_SHARE)
    text = _render(records, reader_findings=_CITING, shown_items_room=room)
    assert _full(text) == ["w20", "w23", "w24", "p09"]


def test_the_proposal_room_is_the_analysts_room():
    """Both are the whole request's size for the same optimizer model's context."""
    from refactory_dsh.optim import analysts

    assert base.PROPOSAL_ROOM == analysts.ANALYST_ROOM


def _before(text: str, heading: str = "### Every item") -> str:
    return text.split(heading, 1)[0]


def test_a_request_over_its_room_shows_fewer_items_whole_and_nothing_else_changes():
    """Over ``request_room``, the request is made again with only the items that fit what
    the rest of it leaves, whole, in turn, the rest named: it then fits. What comes
    before (the readers' findings, at their own budget) is the same text."""
    records = _sized_records()
    unbounded = _render(records, reader_findings=_CITING, request_room=0)
    assert _full(unbounded) == ["w20", "w23", "w24", "p09"]
    # Room for about two of the four.
    room = len(unbounded) - _whole_size(records, "w23", "w24", after=("w20",)) + 200
    fitted = _render(records, reader_findings=_CITING, request_room=room)
    assert len(fitted) <= room < len(unbounded)
    assert _full(fitted) == ["w20", "p09"]
    # Measured as shown: the next the rule picked would not have fit.
    shown = ProposalContext(nodes=NODES, item_records=records)
    w23 = next(r for r in records if r["id"] == "w23")
    assert len(fitted) + base._shown_size(w23, shown._settled(), None) > room
    assert "left out so this request fits the optimizer model's context" in fitted
    assert "w23 (failed), w24 (failed)" in fitted
    assert _before(fitted) == _before(unbounded)
    assert ProposalContext(nodes=NODES, item_records=records,
                           request_room=room).evidence_budget == base.TRAJECTORY_BUDGET


def test_a_request_within_its_room_is_the_unbounded_request():
    records = _sized_records()
    unbounded = _render(records, reader_findings=_CITING, request_room=0)
    assert _render(records, reader_findings=_CITING,
                   request_room=len(unbounded)) == unbounded


def test_a_tiny_room_names_every_item_rather_than_keep_one_past_it():
    records = _sized_records()
    text = _render(records, reader_findings=_CITING, request_room=1000)
    assert _full(text) == []
    assert "w20 (failed), p09 (passed), w23 (failed), w24 (failed)." in text.split(
        "### Some items in full", 1)[1]


class _TooLongOnce(_ReaderClient):
    """The first proposal call is refused as longer than the model's context."""

    def __init__(self) -> None:
        super().__init__()
        self.systems: list[str] = []
        self.refused = False

    def complete(self, prompt: str, *, system: str = "") -> str:
        from refactory_dsh.optim.random_optimizer import READER_SYSTEM

        if system != READER_SYSTEM:
            self.systems.append(system)
            if not self.refused:
                self.refused = True
                self.proposals.append(prompt)
                raise RuntimeError("invalid_request_error: prompt is too long: 230000 tokens")
        return super().complete(prompt, system=system)


def test_the_proposer_fits_the_whole_call_in_the_room_system_prompt_included():
    client = _ReaderClient()
    records = _sized_records()
    optimizer = AutoresearchOptimizer(client)
    room = 60000  # under the request the readers' findings make, system prompt aside
    optimizer.propose(ProposalContext(nodes=NODES, item_records=records,
                                      request_room=room), 1, [])
    assert len(client.proposals[-1]) + len(optimizer._system) <= room  # noqa: SLF001
    assert _full(client.proposals[-1]) == ["w00", "p00"]  # of 5 picked unbounded


def test_a_proposal_refused_as_too_long_is_asked_once_more_at_half_its_length(tmp_path):
    """As the analysts' calls are: only the items shown whole give way, the retry is
    traced as one, and its answer is read."""
    client = _TooLongOnce()
    records = _sized_records()
    optimizer = AutoresearchOptimizer(client)
    optimizer.trace_path = tmp_path / "optimizer_trace.jsonl"
    optimizer.propose(ProposalContext(nodes=NODES, item_records=records), 1, [])
    first, second = client.proposals[-2:]
    assert len(second) <= len(first) // 2 < len(first)
    assert len(_full(second)) < len(_full(first))
    rows = [json.loads(line) for line in optimizer.trace_path.read_text().splitlines()]
    proposals = [r for r in rows if r["kind"] == "proposal"]
    assert "prompt is too long" in proposals[0]["error"]
    assert proposals[-1].get("context_retry") is True and not proposals[-1].get("error")


def test_a_proposal_refused_as_too_long_with_nothing_to_give_way_is_not_asked_again():
    client = _TooLongOnce()
    with pytest.raises(RuntimeError, match="prompt is too long"):
        AutoresearchOptimizer(client).propose(
            ProposalContext(nodes=NODES, item_records=[_record("w1")]), 1, [])
    assert len(client.systems) == 1


def test_without_cited_items_failures_by_id_then_passes_for_contrast():
    """No findings, or findings that cite no known item: one failure per kind of failure
    (its outcome and declared kind), then a passing item of each kind for contrast."""
    records = _sized_records()
    stray = [{"group": 1, "items": [], "success_patterns": [],
              "failure_patterns": [{"pattern": "X", "items": ["nope"], "evidence": []}]}]
    for text in (_render(records), _render(records, reader_findings=stray)):
        full = _full(text)
        assert full == ["w00", "p00"], full  # one kind of failure here: one of it, one pass
        assert "one per kind of failure" in text
    # Two declared kinds: one failure of each, and a pass of the same kind beside each.
    kinds = [{**r, "item_meta": {"type": "algebra" if int(r["id"][1:]) % 2 else "topology"}}
             for r in records]
    assert _full(_render(kinds)) == ["w00", "w01", "p00", "p01"]


def test_a_reader_reply_that_is_not_valid_json_is_a_failed_group_not_an_empty_one():
    """The S1-s42 reply with an unescaped quote, as a reader's reply: the group is said to
    have failed, with the JSON error -- not read as a group with no patterns."""
    from pathlib import Path

    from refactory_dsh.optim.random_optimizer import READER_SYSTEM

    bad = (Path(__file__).parent / "fixtures" / "analyst_reply_unescaped_quote.txt").read_text()

    class Broken(_ReaderClient):
        def complete(self, prompt: str, *, system: str = "") -> str:
            if system == READER_SYSTEM:
                with self.lock:
                    self.readers.append(prompt)
                return bad
            return super().complete(prompt, system=system)

    client = Broken()
    AutoresearchOptimizer(client).propose(
        ProposalContext(nodes=NODES, item_records=_sized_records()), 1, [])
    section = _section(client.proposals[0])
    assert "could not be read: ValueError: the answer's JSON object is not valid JSON" in section


LONG_CHOICE = ("Every bounded sequence in a complete metric space has a subsequence that is "
               "Cauchy, and it converges whenever the space is also totally bounded")


def _long_choice(record: dict[str, Any]) -> dict[str, Any]:
    """``record`` with its expected choice's text long (a formula, as LiveMath's are)."""
    record = {**record, "label_text": LONG_CHOICE, "metadata": dict(record["metadata"])}
    record["metadata"]["choices"] = [
        {**c, "text": LONG_CHOICE} if c["label"] == "C" else c
        for c in record["metadata"]["choices"]]
    return record


def test_an_items_expected_choice_text_is_said_once_with_its_choices():
    """With its choices shown, an item's expected answer is its label (its choice line says
    the text); the table of every item gives only the label for an item shown in full, and
    the whole text for one that is not."""
    records = [_long_choice(_record(f"w{i:02d}")) for i in range(30)] + [
        _long_choice(_record(f"p{i}", passed=True)) for i in range(10)]
    text = "\n".join(base._trajectories(records, budget=20000))
    table = text.split("### Every item", 1)[1].split("### Some items in full", 1)[0]
    shown = [line.split()[2] for line in text.splitlines() if line.startswith("### Item ")]
    assert shown and "Expected: C; Given: A" in text
    for item in shown:
        block = text.split(f"### Item {item} ", 1)[1].split("\n### Item ", 1)[0]
        assert block.count(LONG_CHOICE) == 1, item  # its choice line only
        row = next(r for r in table.splitlines() if r.startswith(f"| {item} |"))
        assert LONG_CHOICE not in row and "| C |" in row, row
    hidden = next(r["id"] for r in records if r["id"] not in shown)
    row = next(r for r in table.splitlines() if r.startswith(f"| {hidden} |"))
    assert LONG_CHOICE[:60] in row, row


def test_an_item_s_cause_number_is_its_row_in_errors_by_cause():
    """Tied counts and repeats: the table and the items number causes over the same
    records, so item a points at the row holding its own error, never at b's."""
    records = [_record("b", error="X broke"), _record("a", error="Y broke"),
               {**_record("a", error="Y broke"), "repeat": 2},
               {**_record("b", error="X broke"), "repeat": 2}]
    text = _render(records)
    table = text.split("Errors, by cause:", 1)[1].split("## ", 1)[0]
    rows = [line for line in table.splitlines() if "broke" in line]
    for item, error in (("a", "Y broke"), ("b", "X broke")):
        block = text.split(f"### Item {item} — failed with an error", 1)[1].split("### Item ", 1)[0]
        number = int(re.search(r"cause (\d+) under", block).group(1))
        assert error in rows[number - 1], (item, number, rows)


@pytest.mark.parametrize("label,label_text,whole", [
    ("C", "Every bounded sequence has a convergent subsequence", False),
    ("yes", "Every bounded sequence has a convergent subsequence", True),
    ("C", "A different statement", True),
])
def test_expected_is_a_label_only_when_a_shown_choice_says_its_text(label, label_text, whole):
    """A label whose choice line is shown with that text says the rest; any other label
    (no such choice, or a choice with other text) keeps the expected text."""
    record = {**_record("w1"), "label": label, "label_text": label_text}
    text = _render([record])
    item = text.split("### Item w1", 1)[1].split("### Item ", 1)[0]
    expected = f"Expected: {label} — {label_text}" if whole else f"Expected: {label};"
    assert expected in item, item
    # The table of every item reads the same rule.
    assert base._expected(record, label_only=True) == (f"{label} — {label_text}" if whole else label)


# ── a harness timeout is not a target ───────────────────────────────────────


def _timed_out(item: str, how: str = "error") -> dict[str, Any]:
    if how == "error":
        return {**_record(item, error="solve: CompletionTimeout: model completion timed out "
                                      "after 600s of inactivity"), "error_kind": "timeout"}
    return {**_record(item), "missing_steps": {"decide": {"cause": "timeout", "sessions": 1}}}


@pytest.mark.parametrize("how", ["error", "missing step"])
def test_a_harness_timeout_is_listed_once_and_read_as_nothing_else(how):
    """The item is listed once, by id, as the harness's; it is in no error row, no item
    shown in full, no count of the pipeline's failures and no time; it still scores 0."""
    # The timed-out item first: were it numbered with the errors, e1 would be cause 2.
    records = [_timed_out("t1", how), _record("w1"),
               _record("e1", error="RuntimeError: worker exited with code 1"),
               _record("p1", passed=True)]
    text = _render(records)
    results = text.split("## Item results", 1)[1].split("\n## ", 1)[0]
    assert f"1 more (t1), each a {base.HARNESS_TIMEOUT}; mean score over all 4: 0.25." in results
    assert "3 train item(s): 1 passed, 1 answered wrong, 1 failed with an error." in results
    assert "| t1" not in results and results.count("t1") == 1
    assert "timed out after 600s" not in text and "No answer --" not in text
    assert "### Item t1" not in text and _shown(text) == ["w1", "e1", "p1"]
    assert 'Error: cause 1 under "Errors, by cause" above' in text


def test_the_analysts_read_no_harness_timeout():
    from refactory_dsh.optim import analysts

    failures, successes = analysts.split([_record("w1"), _timed_out("t1"),
                                          _record("p1", passed=True)], seed=1)
    assert [r["id"] for r in failures] == ["w1"] and [r["id"] for r in successes] == ["p1"]


def test_how_nodes_compose_says_a_timeout_in_one_sentence():
    from refactory_dsh.nn import edits as edits_mod

    state = sf.seeds()["fuzz_draft_review"].state()
    graph = state["graph"]
    text = "\n".join(base.composition_section(edits_mod.catalogue(),
                                               list(graph["nodes"].values()), graph["edges"]))
    compose = text.split("- **crashes**", 1)[1].split("\n- **", 1)[0]
    assert f"An item that times out is a {base.HARNESS_TIMEOUT}." in compose
    assert "not retried" not in compose and "count toward" not in compose


def test_what_the_last_edit_changed_reads_no_harness_timeout_as_the_edit_s():
    """A child that timed out on an item its parent passed broke nothing: the pair is
    left out of the counts and the pairs, said once, and its error is no item error."""
    parent = [_record("w1"), _record("p1", passed=True), _record("p2", passed=True)]
    child = [_record("w1"), _record("p1", passed=True), _timed_out("p2")]
    change = base.edit_change(parent, child)
    assert change["broken"] == [] and change["timed_out"] == ["p2"] and change["shared"] == 2
    lines = "\n".join(base._edit_change_lines([{
        "child": "c", "parent": "p", "edit": [], "child_records": child,
        "parent_records": parent}]))
    assert "0 broken" in lines and "Left out: 1 item(s) the harness timed out" in lines
    assert "Item errors:" not in lines and "### Item p2" not in lines
    assert base.pipeline_item_errors(child) == []


def test_a_repeated_harness_timeout_is_one_item():
    records = [_record("w1"), _timed_out("t1"), {**_timed_out("t1"), "repeat": 2}]
    results = _render(records).split("## Item results", 1)[1].split("\n## ", 1)[0]
    assert "1 more (t1), each a" in results and "2 more" not in results
