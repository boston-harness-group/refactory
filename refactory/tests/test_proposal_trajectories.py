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


def test_the_sample_respects_its_budget_and_clips_long_outputs():
    """Up to 8 wrong and 2 passed items; a long step output keeps its head and tail;
    when the section is too small even at the floor, wrong items are left out (the
    passed ones stay), the cap is exact and the note says how many; an errored item
    stays in the cause table only."""
    records = ([_record(f"w{i:02d}") for i in range(12)]
               + [_record(f"p{i}", passed=True) for i in range(5)]
               + [_record("e1", error="CompletionTimeout: timed out")])
    shown = _shown(_render(records))
    assert sum(item.startswith("w") for item in shown) == 8
    assert sum(item.startswith("p") for item in shown) == 2
    assert "e1" not in shown
    errored = [_record(f"e{i}", error="CompletionTimeout") for i in range(5)]
    assert _shown(_render(errored + [_record("w1")])) == ["w1"], "errored items stay in the table"
    assert "CompletionTimeout" in _render(records), "the cause table still lists it"

    long = "HEAD " + "x" * 9000 + " TAIL"
    text = _render([_record("i1", outputs={"compare": long, "precision": "p", "decide": "d"})],
                   trajectory_text_chars=400)
    assert "HEAD" in text and "TAIL" in text and "characters omitted" in text
    assert "x" * 300 not in text

    for cap in range(2000, 6500, 37):  # the cap is exact, separators and note included
        assert len(_section(_render(records, trajectory_section_chars=cap))) <= cap, cap
    for cap in (2500, 4000):
        capped = _render(records, trajectory_section_chars=cap)
        shown = _shown(capped)
        assert len(_section(capped)) <= cap, cap
        assert sum(item.startswith("p") for item in shown) == 2, "the passed items stay"
        left = 10 - len(shown)
        assert left > 0 and f"({left} more chosen item(s) left out to stay within {cap} characters)" in capped


def _livemath_sized(item: str, *, passed: bool = False) -> dict[str, Any]:
    """A record the size of a real LiveMath skill_graph item: a 1.5k question, five
    400-character choices, four steps of 2.8k each."""
    record = _record(item, passed=passed)
    record["metadata"] = {"question": f"Q{item} " + "q" * 1500,
                          "choices": [{"label": label, "text": label * 400} for label in "ABCDE"]}
    record["exchanges"] = [{"node": node, "prompt": "p", "response": f"{node} START " + "r" * 2800 + " END"}
                           for node in ("compare", "precision", "hypotheses", "decide")]
    return record


def test_livemath_sized_items_all_fit_at_the_defaults():
    """Eight wrong and two passed LiveMath-sized items (~15k characters each) are all
    shown at the default 120k cap: each text is clipped further instead of an item
    being dropped, and the cap holds."""
    records = ([_livemath_sized(f"w{i:02d}") for i in range(12)]
               + [_livemath_sized(f"p{i}", passed=True) for i in range(4)])
    text = _render(records)
    shown = _shown(text)
    assert sum(item.startswith("w") for item in shown) == 8
    assert sum(item.startswith("p") for item in shown) == 2
    section = _section(text)
    assert 100000 < len(section) <= 120000, "the texts use the room, within the cap"
    assert "characters omitted" in section and "left out" not in section
    assert section.count("START") == 40 and section.count("END") == 40, "every step keeps head and tail"


def test_the_sample_varies_by_epoch_and_attempt_and_is_reproducible():
    records = [_record(f"w{i:02d}") for i in range(30)]
    first = _shown(_render(records, epoch=1))
    assert first == _shown(_render(records, epoch=1))
    assert set(first) != set(_shown(_render(records, epoch=2)))
    assert set(first) != set(_shown(_render(records, epoch=1, attempt=2))), "a retry reads others"
    assert set(first) != set(_shown(_render(records, epoch=1, generation=3))), "so does a generation"


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
    """A retry after a refusal is handed the attempt number, so its sample differs."""
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


def test_the_loop_gives_each_retry_and_generation_its_own_sample(tmp_path):
    """Through the real Loop: a proposal refused once is asked for again with the
    next attempt number and reads another sample; a later generation reads another
    one too."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_skillopt import _patch_mode, _write_parts
    from test_val_gate import LevelProbe

    from refactory_dsh.optim import Optimizer, Proposal

    class RefusedOnce(Optimizer):
        name = "refused_once"

        def __init__(self) -> None:
            super().__init__(k=1)
            self.seen: list[tuple[int, int, list[str]]] = []

        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            self.seen.append((ctx.generation, ctx.attempt, _shown(ctx.render())))
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
    (gen1, attempt1, first), (gen1b, attempt2, retry), (gen2, attempt3, later) = proposer.seen[:3]
    assert (gen1, attempt1) == (gen1b - 0, 1) and attempt2 == 2, proposer.seen
    assert gen2 > gen1 and attempt3 == 1
    assert first and retry and later
    assert first != retry, "the retry reads another sample"
    assert later not in (first, retry), "a later generation reads another sample"


def test_items_too_large_to_fit_still_leave_a_note():
    """Items that do not fit even with every text at the floor: the section still
    says how many it left out, instead of disappearing."""
    huge = _record("w1")
    huge["exchanges"] = [{"node": f"s{i}", "prompt": "p", "response": "x" * 800} for i in range(250)]
    text = _render([huge], trajectory_section_chars=20000)
    section = _section(text)
    assert _shown(text) == []
    assert "0 of 1 train item(s) answered wrong" in section
    assert "(1 more chosen item(s) left out to stay within 20000 characters)" in section
