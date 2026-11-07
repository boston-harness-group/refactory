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
    # Recorded out of graph order (parallel steps finish in any order).
    exchanges = [{"node": node, "prompt": f"solve {item} ({node})", "response": outputs[node]}
                 for node in ("decide", "compare", "precision") if node in outputs]
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
    """The question and choices, each step's output in graph order (not the order
    the calls finished), the expected label and text, the prediction and the reason."""
    text = _render([_record("i1")])
    section = text[text.index("## Trajectories of the candidate being edited"):]
    for expected in ("Input (question):", "Which statement holds in i1?",
                     "C. Every bounded sequence has a convergent subsequence",
                     "Expected: C — Every bounded sequence has a convergent subsequence",
                     "Predicted: A", "Grader: MCQ=0: predicted 'A' but expected 'C'"):
        assert expected in section, expected
    steps = [section.index(f"Step `{node}` output:") for node in ("compare", "precision", "decide")]
    assert steps == sorted(steps), "steps follow the graph's order"
    assert section.index("i1: options A and C") < section.index("Answer: A")


def test_the_sample_respects_its_budget_and_clips_long_outputs():
    """Up to 8 wrong and 2 passed items; a long step output keeps its head and tail;
    the section stops at its character cap and says how many it left out; an
    errored item stays in the cause table only."""
    records = ([_record(f"w{i:02d}") for i in range(12)]
               + [_record(f"p{i}", passed=True) for i in range(5)]
               + [_record("e1", error="CompletionTimeout: timed out")])
    shown = _shown(_render(records))
    assert sum(item.startswith("w") for item in shown) == 8
    assert sum(item.startswith("p") for item in shown) == 2
    assert "e1" not in shown
    assert "CompletionTimeout" in _render(records), "the cause table still lists it"

    long = "HEAD " + "x" * 9000 + " TAIL"
    text = _render([_record("i1", outputs={"compare": long, "precision": "p", "decide": "d"})],
                   trajectory_text_chars=400)
    assert "HEAD" in text and "TAIL" in text and "characters omitted" in text
    assert "x" * 300 not in text

    capped = _render(records, trajectory_section_chars=2500)
    assert 0 < len(_shown(capped)) < 10
    assert "not shown: the section is capped at 2500 characters" in capped


def test_the_sample_varies_by_epoch_and_is_reproducible():
    records = [_record(f"w{i:02d}") for i in range(30)]
    first = _shown(_render(records, epoch=1))
    assert first == _shown(_render(records, epoch=1))
    assert set(first) != set(_shown(_render(records, epoch=2)))


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
