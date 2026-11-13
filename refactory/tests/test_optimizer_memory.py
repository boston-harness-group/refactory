"""The memory pass (`optim.memory`), SkillOpt-style retries, item metadata, edits counted
by logical change, and the follow-ups of #645's review.

- The memory pass: once per epoch from epoch 2, one optimizer-model call compares the
  candidate current when the previous epoch proposed with the one current now (or,
  unchanged, that candidate's previous-epoch rollout with this epoch's) on 20 train items
  sampled as SkillOpt samples them, reusing the epoch's own rollouts; it writes ONE note
  every analyst, merge and proposal request reads, journaled and replayed; a failed pass
  keeps the earlier note, marked not updated.
- Every analyst, merge, memory and step-review call is attempted up to three times with
  SkillOpt's waits; a spent budget, a cancelled run or a missing model choice stop at once.
- Analysts see the item fields a Task declares (``Task.item_metadata``), never a grader
  field.
- A structural edit and the edits that only connect it count as one toward the cap.

Each test fails when the behaviour it names is reverted.
"""

from __future__ import annotations

import json
from concurrent.futures import CancelledError
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh.optim import ProposalContext, analysts, memory, random_optimizer
from refactory_dsh.optim.base import Candidate, Proposal, _trajectory, proposal_refusal
from refactory_dsh.optim.random_optimizer import AutoresearchOptimizer, answer_json
from refactory_dsh.outerloop import ValidationGate

from test_gate_aware_context import Recording, _passes_by_level, _val
from test_skillopt import TRAIN_IDS, _patch_mode
from test_val_gate import LevelProbe, _level, _loop


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:] if end == -1 else text[start:end]


def _rec(item: str, passed: bool, out: str = "", **extra: Any) -> dict[str, Any]:
    return {"id": item, "passed": passed, "score": float(passed), "prediction": f"P-{item}",
            "fail_reason": "" if passed else f"WHY-{item}",
            "exchanges": [{"node": "solve", "prompt": f"IN-{item}", "response": out or f"OUT-{item}"}],
            **extra}


# ── sampling, categories, the request ───────────────────────────────────────


def test_the_sample_is_skillopts_twenty_shuffled_with_seed_plus_epoch_times_2000():
    import random

    ids = [f"i{n:02d}" for n in range(50)]
    expected = sorted(ids)
    random.Random(7 + 3 * 2000).shuffle(expected)
    assert memory.sample_ids(list(reversed(ids)), 7, 3) == expected[:20]
    assert memory.MEMORY_SAMPLES == 20
    assert memory.sample_ids(ids, 7, 3) != memory.sample_ids(ids, 7, 4), "per epoch"
    assert sorted(memory.sample_ids(ids[:8], 7, 3)) == sorted(ids[:8]), "fewer: all"


def test_items_fall_in_skillopts_four_categories_and_missing_sides_are_said():
    prev = [_rec("a", False), _rec("b", True), _rec("c", False), _rec("d", True)]
    curr = [_rec("a", True), _rec("b", False), _rec("c", False), _rec("d", True), _rec("e", True)]
    entries, missing = memory.pairs(prev, curr, ["a", "b", "c", "d", "e"])
    assert [(e["id"], e["category"]) for e in entries] == [
        ("a", "improved"), ("b", "regressed"), ("c", "persistent_fail"), ("d", "stable_success")]
    assert missing == ["e"]
    assert memory.counts(entries) == {"regressed": 1, "persistent_fail": 1, "improved": 1,
                                      "stable_success": 1}


def _built(entries: list[dict[str, Any]], *, same: bool = False, room: int = memory.MEMORY_ROOM,
           note: str = "") -> tuple[str, dict[str, int]]:
    return memory.build_prompt(
        epoch=3, mode_name="m",
        prev={"id": "c1" if not same else "c2", "pipeline": "PIPE-START", "rollout": "R-PREV"},
        curr={"id": "c2", "pipeline": "PIPE-NOW", "rollout": "R-NOW"},
        changes=["- `c2` (from `c1`): EDITS"], previous_note=note, entries=entries, room=room)


def test_the_request_lists_regressions_first_with_both_traces_and_stable_ones_without():
    prev = [_rec("a", False, "A-BEFORE"), _rec("b", True, "B-BEFORE"), _rec("d", True, "D-BEFORE")]
    curr = [_rec("a", True, "A-NOW"), _rec("b", False, "B-NOW"), _rec("d", True, "D-NOW")]
    entries, _ = memory.pairs(prev, curr, ["a", "b", "d"])
    text, clip = _built(entries, note="EARLIER-NOTE")
    assert clip == {"items": 0, "chars": 0}
    order = [text.index(h) for h in ("### Regressions", "### Persistent failures",
                                     "### Improvements", "### Stable successes")]
    assert order == sorted(order), "SkillOpt's order: regressions first"
    assert "HIGHEST PRIORITY" in text and "- Regressed (right -> wrong): 1" in text
    regressed = _section(text, "### Regressions")
    assert "B-BEFORE" in text and "B-NOW" in regressed and "Reason now: WHY-b" in regressed
    assert "A-BEFORE" in text and "A-NOW" in text
    stable = text[text.index("### Stable successes"):]
    assert "#### Task d: d" in stable and "D-NOW" not in text and "D-BEFORE" not in text
    assert "PIPE-START" in text and "PIPE-NOW" in text and "EDITS" in text
    assert "EARLIER-NOTE" in text and "Reflect on whether it improved or harmed" in text
    assert '"memory": "<compact optimizer-side guidance' in text
    first, _ = _built(entries)
    assert "(No previous optimizer memory -- this is the first update.)" in first


def test_an_unchanged_candidate_is_compared_with_its_own_earlier_rollout():
    entries, _ = memory.pairs([_rec("a", True)], [_rec("a", False)], ["a"])
    text, _ = _built(entries, same=True)
    assert "two rollouts of this one pipeline (R-PREV, and R-NOW)" in text
    assert "PIPE-START" not in text and "## What changed between them" not in text
    assert "First rollout: PASS" in text and "Second rollout: FAIL" in text


def test_the_traces_share_the_room_each_cut_said_and_the_system_prompt_counted():
    prev = [_rec(f"i{n}", False, "x" * 60000) for n in range(4)]
    curr = [_rec(f"i{n}", n % 2 == 0, "y" * 60000) for n in range(4)]
    entries, _ = memory.pairs(prev, curr, [f"i{n}" for n in range(4)])
    text, clip = _built(entries, room=200000)
    assert len(text) + len(memory.MEMORY_SYSTEM) <= 200000
    assert clip["items"] == 8 and clip["chars"] > 200000
    assert "characters omitted" in text and "clipped head and tail" in text
    small, _ = memory.pairs([_rec("i0", False, "x" * 9000)], [_rec("i0", True, "y" * 9000)], ["i0"])
    whole, none = _built(small)
    assert none == {"items": 0, "chars": 0} and "characters omitted" not in whole
    assert "x" * 9000 in whole and "y" * 9000 in whole


def test_a_note_is_kept_within_its_cap_and_the_cut_said():
    assert memory.parse_note({"memory": "  "}) is None and memory.parse_note([]) is None
    kept = memory.parse_note({"reasoning": "R", "memory": "N" * (memory.MEMORY_CHARS + 50)})
    assert kept["text"].startswith("N" * memory.MEMORY_CHARS)
    assert kept["text"].endswith("… [cut: 50 chars]") and kept["reasoning"] == "R"
    assert memory.parse_note({"meta_skill_content": "M"})["text"] == "M", "SkillOpt's key too"


# ── the optimizer's call ────────────────────────────────────────────────────


class Client:
    """Answers the memory call with ``note`` (or raises it); anything else proposes
    nothing."""

    def __init__(self, note: Any = "THE-NOTE") -> None:
        self.note = note
        self.calls: list[tuple[str, str]] = []

    def complete(self, prompt: str, *, system: str = "", **_: Any) -> str:
        self.calls.append((system, prompt))
        if system == memory.MEMORY_SYSTEM:
            if isinstance(self.note, BaseException):
                raise self.note
            if callable(self.note):
                return self.note(prompt)
            return json.dumps({"reasoning": "R", "memory": self.note})
        return json.dumps({"diagnosis": "d", "proposals": []})

    def memory_calls(self) -> list[str]:
        return [p for s, p in self.calls if s == memory.MEMORY_SYSTEM]


def _build(room: int) -> tuple[str, dict[str, int]]:
    text = "R" * min(room, 5000)
    return text, {"items": 1 if room < 5000 else 0, "chars": max(0, 5000 - room)}


def test_the_memory_call_is_traced_as_memory_and_cached_for_a_resume(tmp_path):
    optimizer = AutoresearchOptimizer(Client())
    optimizer.trace_path = tmp_path / "optimizer_trace.jsonl"
    made = optimizer.write_memory(epoch=2, generation=1, build=_build)
    assert made["note"]["text"] == "THE-NOTE" and made["error"] is None and made["calls"] == 1
    (trace,) = [json.loads(line) for line in optimizer.trace_path.read_text().splitlines()]
    assert trace["kind"] == "memory" and trace["epoch"] == 2 and trace["attempt"] == 1
    again = Client()
    resumed = AutoresearchOptimizer(again)
    resumed.trace_path = tmp_path / "optimizer_trace.jsonl"
    assert resumed.write_memory(epoch=2, generation=1, build=_build)["cached"] is True
    assert again.memory_calls() == [], "a resumed run pays for none"


def test_a_failed_memory_call_is_attempted_three_times_with_skillopts_waits(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(random_optimizer, "_sleep", waits.append)
    client = Client(RuntimeError("503 overloaded"))
    made = AutoresearchOptimizer(client).write_memory(epoch=2, generation=1, build=_build)
    assert len(client.memory_calls()) == 3 and waits == [1, 2, 4]
    assert made["note"] is None and "503 overloaded" in made["error"] and made["calls"] == 3
    unreadable = Client(lambda prompt: "no json here")
    made = AutoresearchOptimizer(unreadable).write_memory(epoch=2, generation=1, build=_build)
    assert len(unreadable.memory_calls()) == 1, "a reply is never retried"
    assert made["note"] is None


@pytest.mark.parametrize("fatal", [CancelledError(), "budget", "choice"])
def test_a_spent_budget_a_cancelled_run_or_a_missing_model_stop_at_once(fatal):
    from refactory_dsh.budgets import BudgetExhausted
    from refactory_dsh.model import ModelChoiceError

    error = {"budget": BudgetExhausted("spent"), "choice": ModelChoiceError("no model")}.get(
        fatal, fatal) if isinstance(fatal, str) else fatal
    client = Client(error)
    with pytest.raises(type(error)):
        AutoresearchOptimizer(client).write_memory(epoch=2, generation=1, build=_build)
    assert len(client.memory_calls()) == 1


def test_a_memory_call_too_long_is_retried_shorter_once_and_never_at_the_same_size():
    def too_long(prompt: str) -> str:
        if len(prompt) > 3000:
            raise RuntimeError("prompt is too long: 250000 tokens > 200000 maximum")
        return json.dumps({"memory": "SHORT-NOTE"})

    client = Client(too_long)
    made = AutoresearchOptimizer(client).write_memory(epoch=2, generation=1, build=_build)
    first, second = client.memory_calls()
    assert len(second) <= len(first) // 2 and made["context_retry"] is True
    assert made["note"]["text"] == "SHORT-NOTE" and made["clipped_items"] == 1

    def fixed(room: int) -> tuple[str, dict[str, int]]:
        return "R" * 5000, {"items": 0, "chars": 0}

    stuck = Client(too_long)
    made = AutoresearchOptimizer(stuck).write_memory(epoch=2, generation=1, build=fixed)
    assert len(stuck.memory_calls()) == 1, "a request that cannot shrink is not sent again"
    assert "could not be made shorter" in made["error"]


# ── in the loop ─────────────────────────────────────────────────────────────


class Remembering(Recording):
    """The scripted proposer, with a memory call that records what it was asked and
    writes ``NOTE-<epoch>``; it fails in the epochs listed in ``fail``."""

    def __init__(self, plan: list[list[tuple[int, int]]], fail: tuple[int, ...] = ()) -> None:
        super().__init__(plan)
        self.fail = fail
        self.asked: list[dict[str, Any]] = []

    def write_memory(self, *, epoch, generation, build):  # type: ignore[no-untyped-def]
        text, clip = build(memory.MEMORY_ROOM)
        self.asked.append({"epoch": epoch, "prompt": text})
        if epoch in self.fail:
            return {"note": None, "error": f"RuntimeError: DOWN-{epoch}", "calls": 3}
        return {"note": {"text": f"NOTE-{epoch}", "reasoning": "R"}, "error": None, "calls": 1,
                "clipped_items": clip["items"], "clipped_chars": clip["chars"]}


def _run(tmp_path: Path, monkeypatch: Any, plan: list, *, proposer: Any = None,
         **overrides: Any):
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(_val))
    _passes_by_level(monkeypatch)
    proposer = proposer or Remembering(plan)
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, **{"epochs": len(plan), "val_gate": True, **overrides})
        loop.optimizer = proposer
        loop.run()
    return loop, proposer


def _events(loop: Any, event: str) -> list[dict[str, Any]]:
    lines = (loop.root / "journal.jsonl").read_text().splitlines()
    return [e for e in map(json.loads, lines) if e.get("event") == event]


def test_the_pass_runs_from_epoch_two_compares_the_currents_and_reuses_the_epochs_rollouts(
        tmp_path, monkeypatch):
    loop, proposer = _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 8)], [(8, 6)], []])
    by_id = {c.id: c for c in loop.population}
    passes = _events(loop, "optimizer_memory")
    assert [p["epoch"] for p in passes] == [2, 3, 4] == [a["epoch"] for a in proposer.asked]
    levels = [(_level(by_id[p["prev"]]), _level(by_id[p["curr"]])) for p in passes]
    # 7 kept in epoch 2, 8 in epoch 3; 6 rejected in epoch 4: unchanged, 8 against itself.
    assert levels == [(5, 7), (7, 8), (8, 8)]
    assert [p["same"] for p in passes] == [False, False, True]
    assert all(p["pairs"] == len(TRAIN_IDS) and not p["missing"] for p in passes)
    assert all(p["evaluated"] == 0 for p in passes), "the epoch's own rollouts are reused"
    assert not _events(loop, "memory_rollout")
    # 5 passes t1-t3, 7 passes t1-t5: two improved, none regressed.
    assert passes[0]["counts"] == {"regressed": 0, "persistent_fail": 3, "improved": 2,
                                   "stable_success": 3}
    assert "rollout of epoch 3" in proposer.asked[2]["prompt"]
    assert "rollout of epoch 4" in proposer.asked[2]["prompt"]
    assert "two rollouts of this one pipeline" in proposer.asked[2]["prompt"]


def test_the_note_reaches_the_request_it_precedes_and_replaces_the_lessons(tmp_path, monkeypatch):
    loop, proposer = _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 8)], []])
    first, second, third = (c.render() for c in proposer.contexts)
    assert "## Optimizer memory" not in first, "no note in epoch 1"
    assert "NOTE-2" in _section(second, "## Optimizer memory")
    assert "written in epoch 2" in second
    assert "NOTE-3" in third and "NOTE-2" not in third
    assert '"lessons"' not in third and "Your lessons so far" not in third
    assert "NOTE-2" in proposer.asked[1]["prompt"], "the previous note, to reflect on"
    assert loop._memory["text"] == "NOTE-3"


def test_a_failed_pass_keeps_the_earlier_note_marked_and_a_later_one_clears_it(
        tmp_path, monkeypatch):
    loop, proposer = _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 8)], [(8, 9)], []],
                          proposer=Remembering([[(5, 7)], [(7, 8)], [(8, 9)], []], fail=(3,)))
    third = _section(proposer.contexts[2].render(), "## Optimizer memory")
    assert "NOT UPDATED in epoch 3" in third and "DOWN-3" in third and "NOTE-2" in third
    assert "this is the earlier note, unchanged" in third
    fourth = proposer.contexts[3].render()
    assert "NOTE-4" in fourth and "NOT UPDATED" not in fourth
    passes = _events(loop, "optimizer_memory")
    assert [p["updated"] for p in passes] == [True, False, True]
    assert passes[1]["memory"]["not_updated"]["why"] == "RuntimeError: DOWN-3"


def test_a_resume_replays_the_note_and_does_not_redo_a_pass(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 8)], [(8, 9)]],
         proposer=Remembering([[(5, 7)], [(7, 8)], [(8, 9)]], fail=(3,)))
    again = Remembering([[]])
    with _patch_mode(LevelProbe):
        resumed = _loop(tmp_path, epochs=4, val_gate=True, resume=True)
        resumed.optimizer = again
        resumed.run()
    assert [a["epoch"] for a in again.asked] == [4], "epochs 2 and 3 are not made again"
    assert "NOTE-2" in again.asked[0]["prompt"], "the replayed note, to reflect on"
    assert resumed._memory["text"] == "NOTE-4" and "not_updated" not in resumed._memory
    assert [p["epoch"] for p in resumed._memory_passes] == [2, 3, 4]


def test_without_the_gate_an_unchanged_current_is_rolled_out_again_and_counted(
        tmp_path, monkeypatch):
    # Ungated, the train leader is current: 7 scores below 5 (kept current), 3 above.
    loop, proposer = _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 3)], []], val_gate=False)
    passes = _events(loop, "optimizer_memory")
    by_id = {c.id: c for c in loop.population}
    assert [(_level(by_id[p["prev"]]), _level(by_id[p["curr"]])) for p in passes] == [
        (5, 5), (5, 3)]
    assert passes[0]["evaluated"] == 1 and passes[0]["rollouts"] == len(TRAIN_IDS)
    assert passes[1]["evaluated"] == 0, "both scored with the run's one seed"
    (rolled,) = _events(loop, "memory_rollout")
    assert rolled["for_epoch"] == 2 and rolled["key"] == "fresh2"
    assert len(rolled["records"]) == len(TRAIN_IDS)
    # The earlier rollout is the one it was scored with; the fresh one is this pass's.
    prompt = proposer.asked[0]["prompt"]
    assert ("Start of the epoch: its rollout when it was scored (the run's seed). Now: a "
            "rollout made for this epoch's memory pass (the run's seed).") in prompt
    assert loop.evaluations == sum(1 for c in loop.population if c.evaluated) + 1
    report = loop.report()
    assert report["optimizer_memory"]["evaluations"] == 1


def test_a_pass_is_made_once_per_epoch(tmp_path, monkeypatch):
    loop, proposer = _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 8)]])
    asked = len(proposer.asked)
    loop._memory_pass()  # a second proposal step in the same epoch (a strategy's)
    assert len(proposer.asked) == asked and len(loop._memory_passes) == asked


def test_a_parent_picker_hands_the_memory_call_to_its_proposer():
    from refactory_dsh.optim import compose, policies

    proposer = AutoresearchOptimizer(Client())
    paired = compose.PickedParent(policies.HillClimbSelector(), proposer)
    assert paired.write_memory(epoch=2, generation=1, build=_build)["note"]["text"] == "THE-NOTE"
    sampler = compose.PickedParent(policies.HillClimbSelector(), random_optimizer.RandomOptimizer())
    assert getattr(sampler, "write_memory", None) is None, "no model, no memory pass"


def test_off_there_is_no_pass_no_note_and_no_event(tmp_path, monkeypatch):
    loop, proposer = _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 8)], []], lessons=False)
    assert proposer.asked == [] and not _events(loop, "optimizer_memory")
    assert all("## Optimizer memory" not in c.render() for c in proposer.contexts)
    assert "optimizer_memory" not in loop.report()


def test_the_report_says_each_pass_and_shows_the_latest_note(tmp_path, monkeypatch):
    loop, _proposer = _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 8)], []])
    report = loop.report()
    rep = report["optimizer_memory"]
    assert [p["epoch"] for p in rep["passes"]] == [2, 3] and rep["updated"] == 2
    assert rep["latest"]["text"] == "NOTE-3"
    text = (loop.root / "report.md").read_text()
    assert "## Optimizer memory" in text and "> NOTE-3" in text


def test_the_autoresearch_proposer_writes_the_note_in_a_real_loop(tmp_path, monkeypatch):
    """The model proposer end to end: the memory call's prompt has both traces of the
    epoch's changed items, and the next proposal request carries its note."""
    from test_analysts import Client as Analysts, _with_traces

    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(_val))
    _passes_by_level(monkeypatch)
    _with_traces(monkeypatch)
    client = Analysts()
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, epochs=2, val_gate=True)
        loop.optimizer = AutoresearchOptimizer(client)
        loop.run()
    (prompt,) = client.of((memory.MEMORY_SYSTEM,))
    assert "# Writing the optimizer's memory -- epoch 2" in prompt
    assert "### Improvements" in prompt or "### Regressions" in prompt
    proposals = [p for s, p in client.calls if s not in (memory.MEMORY_SYSTEM,)]
    assert "MEMORY-NOTE" in _section(proposals[-1], "## Optimizer memory")
    trace = [json.loads(line) for line in (loop.root / "optimizer_trace.jsonl").read_text()
             .splitlines()]
    assert [t["kind"] for t in trace].count("memory") == 1


# ── retries for analysts and merges ─────────────────────────────────────────


def test_analyst_and_merge_calls_are_retried_with_skillopts_waits(monkeypatch):
    from test_analysts import ANALYST_SYSTEMS, MERGE_SYSTEMS, Client as Analysts, _ctx
    from test_analysts import _rec as arec

    waits: list[float] = []
    monkeypatch.setattr(random_optimizer, "_sleep", waits.append)
    failures = {"left": 2}

    class Flaky(Analysts):
        def complete(self, prompt: str, *, system: str = "", **kwargs: Any) -> str:
            if system in MERGE_SYSTEMS and failures["left"]:
                failures["left"] -= 1
                with self.lock:
                    self.calls.append((system, prompt))
                raise TimeoutError("slow")
            return super().complete(prompt, system=system, **kwargs)

    client = Flaky()
    optimizer = AutoresearchOptimizer(client)
    records = [arec(f"f{i}", False) for i in range(2)] + [arec("s0", True)]
    optimizer.propose(_ctx(records, group_size=1), 1, [])
    merges = optimizer.last_analysis["merges"]
    assert all(not m["error"] and not m["fallback"] for m in merges), "the third try answered"
    assert waits == [1, 2] and len(client.of(MERGE_SYSTEMS)) == len(merges) + 2
    assert optimizer.last_analysis["calls"] == len(client.of(ANALYST_SYSTEMS + MERGE_SYSTEMS))


# ── item metadata ───────────────────────────────────────────────────────────


def test_a_tasks_declared_fields_head_each_item_and_a_grader_field_never_does(monkeypatch):
    from refactory_dsh import task as task_module

    class Declaring(task_module.Task):
        name = "declaring"
        grader_fields = ("answer",)
        item_metadata = ("kind", "answer", "tags")

    task_module.register_task("declaring_task", Declaring)
    state = {"graph": {"start": "data", "nodes": {"data": {
        "id": "data", "kind": "data", "task_ref": "declaring_task", "subgraph_entry": "w",
        "subgraph_exit": "w"}, "w": {"id": "w", "kind": "op", "call": "loop_ops:noop"}},
        "edges": []}}
    read = task_module.item_metadata_reader(state)
    record = _rec("a", False, metadata={"kind": "algebra", "answer": "42", "tags": ["x", "y"],
                                        "question": "Q?"})
    record["item_meta"] = read(record)
    assert record["item_meta"] == {"kind": "algebra", "tags": ["x", "y"]}
    header = analysts.item_text(record).splitlines()[0]
    assert header.endswith("— kind: algebra; tags: x, y") and "42" not in header
    assert _trajectory(record).splitlines()[0].endswith("— kind: algebra; tags: x, y")
    assert "kind:" not in analysts.item_text({**record, "item_meta": {}}).splitlines()[0]


def test_livemath_shows_skillopts_task_type_exactly():
    """rollout.py:132: the first theorem type, else math_mcq."""
    from refactory_packages.livemath.tasks import LiveMathTask

    task = LiveMathTask()
    assert task.item_meta({"theorem_type": ["inequality", "existence"]}) == {
        "task_type": "inequality"}
    assert task.item_meta({"theorem_type": []}) == {"task_type": "math_mcq"}
    assert task.item_meta({}) == {"task_type": "math_mcq"}
    assert "task_type" not in LiveMathTask.grader_fields


def test_the_loop_hands_the_analysts_records_with_their_declared_fields(tmp_path, monkeypatch):
    from refactory_dsh.loop import Loop

    monkeypatch.setattr(Loop, "_item_meta_reader",
                        lambda self: lambda r: ({"kind": r["metadata"]["kind"]}
                                                if "kind" in (r.get("metadata") or {}) else {}))
    original = Loop._parent_train_records

    def with_metadata(self, board):  # type: ignore[no-untyped-def]
        return [{**r, "metadata": {"kind": f"K-{r['id']}"}} for r in original(self, board)]

    monkeypatch.setattr(Loop, "_parent_train_records", with_metadata)
    _loop_run, proposer = _run(tmp_path, monkeypatch, [[(5, 7)], []])
    records = proposer.contexts[-1].item_records
    assert records and all(r["item_meta"] == {"kind": f"K-{r['id']}"} for r in records)


# ── edits counted by logical change ─────────────────────────────────────────


def _graph() -> dict[str, Any]:
    from refactory_dsh import nn

    graph = nn.Graph()
    graph.add(nn.Node("draft", "agent", prompt_template="Draft.", writes=("draft.txt",)))
    graph.add(nn.Node("final", "agent", prompt_template="Use draft.txt.", reads=("draft.txt",),
                      writes=("answer.txt",)))
    graph.add(nn.Node("report", "agent", prompt_template="Report.", reads=("answer.txt",),
                      writes=("report.txt",)))
    graph.connect("draft", "final")
    graph.connect("final", "report")
    values = {"final__task": "Use draft.txt.\nAnswer.", "report__task": "Report.",
              "draft__task": "Draft."}
    knobs = {name: {"value": value, "spec": {"kind": "str", "default": value}}
             for name, value in values.items()}
    state = {"graph": graph.to_state(), "knobs": knobs}
    nodes = state["graph"]["nodes"]
    for name in ("draft", "final", "report"):
        nodes[name]["field_bindings"] = [["prompt_template", f"{name}__task"]]
        nodes[name]["knobs"] = [f"{name}__task"]
    return state


CHECK = {"id": "check", "kind": "agent", "prompt_template": "Check draft.txt.",
         "reads": ["draft.txt"], "writes": ["check.txt"], "knobs": ["check__task"],
         "field_bindings": [["prompt_template", "check__task"]]}


def _insert() -> list[dict[str, Any]]:
    return [{"op": "add_knob", "name": "check__task",
             "spec": {"kind": "str", "default": "Check draft.txt."}},
            {"op": "insert_after", "target": "draft", "node": CHECK},
            {"op": "set_knob", "name": "check__task", "value": "Check draft.txt twice."}]


def test_a_new_step_with_its_own_settings_and_its_readers_lines_naming_it_is_one_change():
    from refactory_dsh.optim.edit_count import count

    connect = {"op": "set_knob", "name": "final__task",
               "value": "Use draft.txt.\nAlso weigh check.txt.\nAnswer."}
    tally = count([*_insert(), connect], _graph())
    assert tally["logical"] == 1 and tally["raw"] == 4
    assert (tally["structural"], tally["knob"]) == (1, 3) and tally["chars"] > 0
    other = {"op": "set_knob", "name": "final__task", "value": "Use draft.txt.\nBe brief."}
    assert count([*_insert(), other], _graph())["logical"] == 2, "a line not naming it"
    mixed = {"op": "set_knob", "name": "final__task",
             "value": "Use draft.txt.\nAlso weigh check.txt.\nBe brief."}
    assert count([*_insert(), mixed], _graph())["logical"] == 2, "every changed line names it"
    far = {"op": "set_knob", "name": "report__task", "value": "Report.\nMention check.txt."}
    assert count([*_insert(), far], _graph())["logical"] == 2, "two hops: no sprawl"
    two = [*_insert(), {"op": "insert_after", "target": "final",
                        "node": {**CHECK, "id": "check2", "writes": ["check2.txt"],
                                 "knobs": [], "field_bindings": []}}]
    assert count(two, _graph())["logical"] == 2, "two structural edits are two"
    near = {"op": "add_edge", "source": "check", "target": "final"}
    removed = [*_insert(), {"op": "remove_edge", "source": "check", "target": "final"}, near]
    assert count(removed, _graph())["logical"] == 1, "edges between it and a neighbour"
    far = [*_insert(), {"op": "add_edge", "source": "check", "target": "report"}]
    assert count(far, _graph())["logical"] == 2, "an edge to a step two hops away"


def test_the_cap_counts_logical_changes_and_its_refusal_says_how():
    six = [*_insert(),
           {"op": "set_knob", "name": "final__task", "value": "Use draft.txt.\nSee check.txt.\nAnswer."},
           {"op": "set_knob", "name": "draft__task", "value": "Draft carefully."},
           {"op": "set_knob", "name": "report__task", "value": "Report briefly."}]
    assert proposal_refusal(Proposal(edits=tuple(six)), max_edits=4, parent_state=_graph()) == []
    knobs = [{"op": "set_knob", "name": name, "value": f"{name} {n}"}
             for n, name in enumerate(["draft__task", "final__task", "report__task",
                                       "draft__task", "final__task"])]
    refused = proposal_refusal(Proposal(edits=tuple(knobs)), max_edits=4, parent_state=_graph())
    assert refused[0].startswith("E-too-many-edits: the proposal makes 5 changes (5 edits)")
    assert "count as one change" in refused[0]
    raw = proposal_refusal(Proposal(edits=tuple(six)), max_edits=4)
    assert raw[0].startswith("E-too-many-edits: the proposal bundles 6 edits"), "no state: raw"
    single = proposal_refusal(Proposal(edits=tuple(_insert()[:2])), max_edits=1,
                              parent_state=_graph())
    assert single[0].startswith("E-too-many-edits"), "a limit of one is single edits"


def test_each_candidate_records_its_changes_and_the_report_shows_them(tmp_path, monkeypatch):
    loop, _proposer = _run(tmp_path, monkeypatch, [[(5, 7)], []])
    (record,) = [p for p in _events(loop, "proposal")]
    assert record["edit_count"] == {"logical": 1, "raw": 1, "structural": 0, "knob": 1,
                                    "chars": 2}
    text = (loop.root / "report.md").read_text()
    assert "| 1 (1 edits: 0 structural, 1 knob; 2 chars) |" in text


# ── #645's review follow-ups ────────────────────────────────────────────────


def test_without_a_data_node_the_final_step_only_the_scorer_reads_decides():
    from refactory_dsh.optim import feedback

    state = {"graph": {"nodes": {
        "notes": {"id": "notes", "kind": "agent", "writes": ["notes.txt"]},
        "side": {"id": "side", "kind": "agent", "writes": ["side.txt"]},
        "final": {"id": "final", "kind": "agent", "reads": ["notes.txt"],
                  "writes": ["answer.txt"]}},
        "edges": [{"source": "notes", "target": "side"}, {"source": "side", "target": "final"}]}}
    assert feedback.decision_step(state, "final") is True, "the graph's end"
    assert feedback.decision_step(state, "side") is False, "unread and not the end"


def test_the_answer_is_the_object_with_a_proposals_list():
    answer = {"diagnosis": "d", "proposals": [{"edit": {"op": "set_knob"}}]}
    example = {"proposal": {"edit": {"op": "bypass"}}}
    text = json.dumps(answer) + "\nFor instance: " + json.dumps(example)
    assert answer_json(text) == (answer, "")
    not_a_list = {"proposals": "none this round"}
    assert answer_json(json.dumps(not_a_list) + " " + json.dumps(answer)) == (answer, "")


def test_a_fenced_example_before_a_cut_off_answer_is_not_the_answer():
    example = '```json\n{"op": "set_knob", "name": "k", "value": 1}\n```\n'
    text = example + '{"diagnosis": "d", "proposals": [{"edit": {"op": "set_knob", "name"'
    found, problem = answer_json(text)
    assert found == {} and "never closed" in problem
    whole = example + '```json\n{"proposals": []}\n```'
    assert answer_json(whole)[0] == {"proposals": []}


def test_a_top_level_array_is_read_whole_or_refused():
    one, two = {"edit": {"op": "set_knob", "name": "a", "value": 1}}, {"op": "bypass", "target": "b"}
    assert answer_json("Here:\n" + json.dumps([one, two])) == ({"proposals": [one, two]}, "")
    assert answer_json("```json\n" + json.dumps([one, two]) + "\n```")[0] == {"proposals": [one, two]}
    found, problem = answer_json(json.dumps([one, "and a note"]))
    assert found == {} and "other than proposal objects" in problem
    found, problem = answer_json(json.dumps([one, two])[:-20])
    assert found == {} and "never closed" in problem
    optimizer = AutoresearchOptimizer(Client())
    optimizer._client = type("C", (), {"complete": lambda self, p, **k: json.dumps([one, two])})()
    parent = Candidate.from_state({"graph": {}, "knobs": {}})
    proposals = optimizer.propose(ProposalContext(parent=parent.id), 2, [parent])
    assert [p.steps for p in proposals] == [[one["edit"]], [two]], "every one, not the first"


def test_a_review_whose_instructions_dominate_shrinks_on_its_retry_or_is_not_resent():
    from refactory_dsh.optim import feedback

    entry = {"step": "s", "candidate": "c", "kind": "added agent step", "node_kind": "agent",
             "instructions": "I" * 300000,
             "samples": [{"item": "t1", "input": "x" * 50000, "output": "y" * 50000,
                          "next_step": "n", "next_output": "z" * 50000}]}
    first = feedback.review_prompt(entry, "m")
    assert len(first) + len(feedback.STEP_REVIEW_SYSTEM) <= feedback.STEP_REVIEW_ROOM
    half = feedback.review_prompt(entry, "m", room=len(first) // 2)
    assert len(half) + len(feedback.STEP_REVIEW_SYSTEM) <= len(first) // 2
    assert "characters omitted" in half and "I" * 1000 in half

    class Long:
        reviews: list[str] = []

        def complete(self, prompt: str, *, system: str = "", **_: Any) -> str:
            if system == feedback.STEP_REVIEW_SYSTEM:
                self.reviews.append(prompt)
                raise RuntimeError("prompt is too long")
            return json.dumps({"proposals": []})

    client = Long()
    floor = {**entry, "instructions": "short", "samples": [
        {**entry["samples"][0], "input": "x", "output": "y", "next_output": "z"}]}
    AutoresearchOptimizer(client).propose(ProposalContext(added_steps=[floor]), 1, [])
    assert len(client.reviews) == 1, "nothing to cut: the same request is not sent again"


# ── the independent review's fixes ──────────────────────────────────────────


def test_an_answer_with_a_proposals_list_wins_over_any_array():
    answer = {"diagnosis": "d", "proposals": [{"edit": {"op": "set_knob", "name": "a", "value": 1}}]}
    prose = ("I will send edits shaped like [{op, knob}] below.\n```json\n"
             + json.dumps(answer) + "\n```")
    assert answer_json(prose) == (answer, "")
    example = 'Example: [{"op": "add_node"}] -- now my answer: ' + json.dumps(answer)
    assert answer_json(example) == (answer, ""), "the example array is not the answer"
    edit = {"op": "set_knob", "name": "a", "value": 1}
    assert answer_json("Shaped like [{op, knob}]: " + json.dumps(edit)) == (edit, ""), (
        "brackets in prose are not an array")
    with pytest.raises(ValueError, match="no JSON object"):
        answer_json("Nothing to change; edits would look like [{op, knob}].")
    cut = 'Example: [{"op": "add_node"}]\n{"diagnosis": "d", "proposals": [{"edit": {'
    found, problem = answer_json(cut)
    assert found == {} and "never closed" in problem, "nor before a cut-off answer"


def test_a_knob_another_step_binds_is_not_the_new_steps_own():
    from refactory_dsh.optim.edit_count import count

    shared = {"id": "check", "kind": "agent", "prompt_template": "Check.",
              "reads": ["draft.txt"], "writes": ["check.txt"],
              "knobs": ["final__task", "report__task", "draft__task"]}
    edits = [{"op": "insert_after", "target": "draft", "node": shared},
             *({"op": "set_knob", "name": name, "value": f"REWRITTEN {name}"}
               for name in ("final__task", "report__task", "draft__task"))]
    tally = count(edits, _graph())
    assert (tally["logical"], tally["raw"]) == (4, 4), "three whole rewrites, each its own"


def test_the_changes_shown_start_from_the_common_ancestor():
    lineage = {"common": "root", "prev": [{"id": "a", "parent": "root", "edits": ["EA"]}],
               "curr": [{"id": "b", "parent": "root", "edits": ["EB"]}]}
    text = "\n".join(memory.changes_lines(lineage))
    assert "different branches from their common ancestor `root`" in text
    assert text.index("EA") < text.index("From it to the current candidate") < text.index("EB")
    down = "\n".join(memory.changes_lines({"common": "a", "prev": [], "curr": [
        {"id": "b", "parent": "a", "edits": ["EB"]}]}))
    assert "descends from the start-of-epoch one" in down and "EB" in down
    back = "\n".join(memory.changes_lines({"common": "b", "prev": [
        {"id": "a", "parent": "b", "edits": ["EA"]}], "curr": []}))
    assert "is an earlier version of the start-of-epoch one" in back and "EA" in back
    assert "different branches" not in back
    apart = "\n".join(memory.changes_lines({"common": None, "prev": [], "curr": []}))
    assert "share no ancestor" in apart


def test_a_sibling_current_is_compared_from_their_common_ancestor(tmp_path, monkeypatch):
    # Ungated, a lower level scores higher: 4 (from 5) leads, then 3, also from 5.
    loop, proposer = _run(tmp_path, monkeypatch, [[(5, 4)], [(5, 3)], []], val_gate=False)
    by_id = {c.id: c for c in loop.population}
    seed = next(c.id for c in loop.population if _level(c) == 5)
    passes = _events(loop, "optimizer_memory")
    assert [(_level(by_id[p["prev"]]), _level(by_id[p["curr"]])) for p in passes] == [
        (5, 4), (4, 3)]
    assert (f"different branches from their common ancestor `{seed}`"
            in proposer.asked[1]["prompt"])


class _Stopped(Client):
    def __init__(self) -> None:
        super().__init__(RuntimeError("503 overloaded"))


def test_a_retry_wait_ends_on_a_stop(monkeypatch):
    waits: list[float] = []
    monkeypatch.setattr(random_optimizer, "_sleep", waits.append)
    client = _Stopped()
    optimizer = AutoresearchOptimizer(client)
    optimizer.stop_check = lambda: True
    with pytest.raises(CancelledError):
        optimizer.write_memory(epoch=2, generation=1, build=_build)
    assert len(client.memory_calls()) == 1 and waits == [], "not attempted again"
    import threading

    event = threading.Event()
    event.set()
    scoped = AutoresearchOptimizer(_Stopped())
    scoped._client._cancellations = (event,)  # a DshModelClient's cancellation scope
    with pytest.raises(CancelledError):
        scoped.write_memory(epoch=2, generation=1, build=_build)
    patient = AutoresearchOptimizer(_Stopped())
    patient.stop_check = lambda: False
    patient.write_memory(epoch=2, generation=1, build=_build)
    assert sum(waits) == 1 + 2 + 4, "no stop: SkillOpt's waits, in slices"


def test_the_loop_hands_its_stop_request_to_the_optimizer(tmp_path, monkeypatch):
    from refactory_dsh.control import Control

    loop, proposer = _run(tmp_path, monkeypatch, [[(5, 7)], []])
    assert proposer.stop_check() is False
    monkeypatch.setattr(Control, "read_request", lambda self: {"action": "stop"})
    assert proposer.stop_check() is True
    monkeypatch.setattr(Control, "read_request", lambda self: {"action": "pause"})
    assert proposer.stop_check() is False


class _Cancelling(Remembering):
    def write_memory(self, *, epoch, generation, build):  # type: ignore[no-untyped-def]
        raise CancelledError("the run was stopped")


def test_a_cancelled_memory_call_records_no_pass_so_a_resume_makes_it(tmp_path, monkeypatch):
    with pytest.raises(CancelledError):
        _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 8)]],
             proposer=_Cancelling([[(5, 7)], [(7, 8)]]))
    root = next((tmp_path / "run").glob("**/journal.jsonl")).parent
    assert '"optimizer_memory"' not in (root / "journal.jsonl").read_text()
    again = Remembering([[]])
    with _patch_mode(LevelProbe):
        resumed = _loop(tmp_path, epochs=2, val_gate=True, resume=True)
        resumed.optimizer = again
        resumed.run()
    assert [a["epoch"] for a in again.asked] == [2] and resumed._memory["text"] == "NOTE-2"


def test_a_cancelled_model_client_ends_the_run_but_leaves_its_report(tmp_path, monkeypatch):
    with pytest.raises(CancelledError):
        _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 8)]],
             proposer=_Cancelling([[(5, 7)], [(7, 8)]]))
    root = next((tmp_path / "run").glob("**/journal.jsonl")).parent
    assert (root / "report.md").exists()
    bounds = [json.loads(line) for line in (root / "journal.jsonl").read_text().splitlines()
              if '"event": "bound"' in line]
    assert bounds and bounds[-1]["bound"].startswith("CancelledError: the run was stopped")


def test_every_compared_item_is_headed_by_its_question():
    prev = [_rec("q1", True, metadata={"question": "Q-ONE?"})]
    curr = [_rec("q1", True, metadata={"question": "Q-ONE?"})]
    entries, _ = memory.pairs(prev, curr, ["q1"])
    text, _ = _built(entries)
    assert "#### Task q1: Q-ONE?" in text[text.index("### Stable successes"):]


class _StopsInItsWait(Client):
    """Every memory and analyst call fails, so its retry waits; a person has asked to stop."""

    def complete(self, prompt: str, *, system: str = "", **kwargs: Any) -> str:
        from test_analysts import ANALYST_SYSTEMS

        if system in (memory.MEMORY_SYSTEM, *ANALYST_SYSTEMS):
            self.calls.append((system, prompt))
            raise RuntimeError("503 overloaded")
        return super().complete(prompt, system=system, **kwargs)


def _stopping_run(tmp_path: Path, monkeypatch: Any, client: Any, **overrides: Any) -> Any:
    from refactory_dsh.loop import Loop
    from test_analysts import _with_traces

    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(_val))
    _passes_by_level(monkeypatch)
    _with_traces(monkeypatch)
    monkeypatch.setattr(Loop, "_stop_requested", lambda self: True)
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, val_gate=True, **overrides)
        loop.optimizer = AutoresearchOptimizer(client)
        loop.run()  # a stop, not an exception
    monkeypatch.undo()
    return loop


def _resume(tmp_path: Path, monkeypatch: Any, client: Any, **overrides: Any) -> Any:
    from test_analysts import _with_traces

    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(_val))
    _passes_by_level(monkeypatch)
    _with_traces(monkeypatch)
    with _patch_mode(LevelProbe):
        resumed = _loop(tmp_path, val_gate=True, resume=True, **overrides)
        resumed.optimizer = AutoresearchOptimizer(client)
        resumed.run()
    return resumed


def test_a_stop_during_the_memory_pass_wait_ends_as_a_stop_and_the_pass_is_redone(
        tmp_path, monkeypatch):
    from refactory_dsh.optim.random_optimizer import CALL_ATTEMPTS

    client = _StopsInItsWait()
    loop = _stopping_run(tmp_path, monkeypatch, client, epochs=3)
    assert loop.bound == "a person asked the run to stop"
    assert (loop.root / "report.md").exists()
    assert len(client.memory_calls()) == 1 < CALL_ATTEMPTS, "not attempted again"
    events = [json.loads(line)["event"] for line in
              (loop.root / "journal.jsonl").read_text().splitlines()]
    assert "stopped_inside_step" in events and "optimizer_memory" not in events
    assert not loop.optimizer.proposal_failures
    again = Client()
    resumed = _resume(tmp_path, monkeypatch, again, epochs=3)
    assert [p["epoch"] for p in resumed._memory_passes][:1] == [2], "the pass is made"
    assert resumed._memory_passes[0]["updated"] is True


def test_a_stop_during_an_analyst_wait_is_a_stop_and_the_round_is_redone(
        tmp_path, monkeypatch):
    from test_analysts import Client as Analysts

    client = _StopsInItsWait()
    loop = _stopping_run(tmp_path, monkeypatch, client, epochs=2, analysts=True,
                         analyst_group_size=2)
    assert loop.bound == "a person asked the run to stop"
    assert not loop.optimizer.proposal_failures, "not an empty failed round"
    assert loop.optimizer.proposal_rounds == 0, "nor a round counted"
    assert (loop.root / "report.md").exists()
    assert "could not propose" not in (loop.root / "report.md").read_text()
    proposed = [p for s, p in client.calls if s not in (memory.MEMORY_SYSTEM,)
                and p.startswith("# Update request")]
    assert proposed == [], "the round never reached its proposal"
    again = Analysts()
    _resume(tmp_path, monkeypatch, again, epochs=2, analysts=True, analyst_group_size=2)
    assert "# Update request — generation 0, epoch 1" in again.proposal().splitlines()[0] or any(
        p.startswith("# Update request — generation 0, epoch 1") for s, p in again.calls), (
        "the stopped round is made again on resume")


@pytest.mark.parametrize("after", [' and in prose {"op": "set_knob"}', " {}"])
def test_a_list_followed_by_an_object_is_refused_not_read_as_its_first_entry(after):
    one, two = {"edit": {"op": "set_knob", "name": "a", "value": 1}}, {"op": "bypass", "target": "b"}
    found, problem = answer_json(json.dumps([one, two]) + after)
    assert found == {} and "follows its top-level list of proposals" in problem
    optimizer = AutoresearchOptimizer(Client())
    optimizer._client = type("C", (), {"complete": lambda self, p, **k: json.dumps([one, two])
                                       + after})()
    proposals = optimizer.propose(ProposalContext(), 2, [])
    assert len(proposals) == 2 and all(
        proposal_refusal(p)[0].startswith("E-proposal-shape: your whole answer") for p in proposals)


def test_of_a_draft_list_and_a_final_list_the_final_one_is_read_whole():
    draft = [{"edit": {"op": "set_knob", "name": "a", "value": 1}}]
    final = [{"edit": {"op": "set_knob", "name": "a", "value": 2}}, {"op": "bypass", "target": "b"}]
    text = "Draft:\n" + json.dumps(draft) + "\nOn reflection, final:\n" + json.dumps(final)
    assert answer_json(text) == ({"proposals": final}, "")



@pytest.mark.parametrize("key,value", [
    ("proposal", {"edit": {"op": "set_knob", "name": "a", "value": 1}}),
    ("proposals", {"edit": {"op": "set_knob", "name": "a", "value": 1}})])
def test_a_loosely_written_answer_after_a_prose_example_is_the_answer(key, value):
    real = {"diagnosis": "d", key: value}
    text = 'Edit shape: {"op": "bypass", "target": "x"}\n' + json.dumps(real)
    assert answer_json(text) == (real, "")
    optimizer = AutoresearchOptimizer(Client())
    optimizer._client = type("C", (), {"complete": lambda self, p, **k: text})()
    (proposal,) = optimizer.propose(ProposalContext(), 1, [])
    assert proposal.steps == [value["edit"]], "one proposal, the answer's, not the example"
