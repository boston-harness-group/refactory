"""What the optimizer sees of the edited candidate's train items, and what it said.

``ProposalContext.render`` shows a per-epoch sample of train items in full (input,
each step's output in graph order, expected vs predicted), within a budget; and
``AutoresearchOptimizer`` records every model call to ``optimizer_trace.jsonl``.
Each test fails when the behaviour it names is reverted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh.loop import Loop, LoopConfig
from refactory_dsh.optim import ProposalContext
from refactory_dsh.optim.random_optimizer import AutoresearchOptimizer

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
    expected label and text, the prediction and the reason."""
    text = _render([_record("i1")])
    section = text[text.index("## Trajectories of the candidate being edited"):]
    for expected in ("Input (question):", "Which statement holds in i1?",
                     "C. Every bounded sequence has a convergent subsequence",
                     "Expected: C — Every bounded sequence has a convergent subsequence",
                     "Predicted: A", "Grader: MCQ=0: predicted 'A' but expected 'C'"):
        assert expected in section, expected
    steps = [section.index(f"Step `{node}` output:") for node in ("compare", "precision", "decide")]
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
    order = [section.index(f"Step `{node}` output:") for node in ran]
    assert order == sorted(order), "the order the pass ran, not alphabetical"


def _section(text: str) -> str:
    start = text.index("## Trajectories of the candidate being edited")
    return text[start:text.index("\n## What this run has already tried", start)]


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
               + [_record("e1", error="CompletionTimeout: timed out")])
    stepped = {**_record("e2"), "error_text": "RuntimeError: decide crashed",
               "error_kind": "RuntimeError"}
    long = "HEAD " + "x" * 9000 + " TAIL"
    records[0] = _record("w00", outputs={"compare": long, "precision": "p", "decide": "d"})
    text = _render(records + [stepped])
    assert _shown(text) == [f"w{i:02d}" for i in range(12)] + ["e1", "e2"] + [f"p{i}" for i in range(5)]
    assert "All 19 train item(s): 12 answered wrong, 2 failed with an error, 5 passed." in text
    assert "### Item e1 — failed with an error" in text and "Error: CompletionTimeout: timed out" in text
    assert "Every item in full" in text and long in text and "characters omitted" not in text
    assert "### Item e2 — failed with an error" in text and "Error: RuntimeError: decide crashed" in text


def test_an_item_longer_than_the_budget_alone_is_clipped_with_a_note():
    long = "HEAD " + "x" * 9000 + " TAIL"
    text = _render([_record("w1", outputs={"compare": long, "precision": "p", "decide": "d"})],
                   evidence_budget=4000)
    section = _section(text)
    assert "HEAD" in section and "TAIL" in section and "characters omitted" in section
    assert "longer than one call's evidence budget" in section
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
    assert "Predicted: 0" in _render([zero])


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
    per call, written as it happens, including a call that failed."""
    reply = json.dumps({"diagnosis": "the decide step trusts compare",
                        "proposals": [{"edit": {"op": "set_knob", "name": "k", "value": 1}}]})
    optimizer = AutoresearchOptimizer(_Client([reply, RuntimeError("503 overloaded")]))
    optimizer.trace_path = tmp_path / "optimizer_trace.jsonl"
    ctx = ProposalContext(epoch=2, generation=5, nodes=NODES, item_records=[_record("i1")])
    assert len(optimizer.propose(ctx, 2, [])) == 1
    with pytest.raises(RuntimeError):
        optimizer.propose(ctx, 2, [])
    calls = [json.loads(line) for line in optimizer.trace_path.read_text().splitlines()]
    assert len(calls) == 2
    first, failed = calls
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
    assert "(3019 characters)" in section and "CompletionTimeout" in section
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
    assert len(section) <= TRAJECTORY_BUDGET
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
