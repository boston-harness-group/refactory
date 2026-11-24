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


def _bare(text: str) -> str:
    """``text`` without its steps' ``Ran as:`` lines (said once a request)."""
    from refactory_dsh.optim import evidence

    return "\n".join(line for line in text.split("\n")
                     if not line.startswith(f"{evidence.RAN_AS}: "))

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
    assert "B-BEFORE" in text and "B-NOW" in regressed and "Grader: WHY-b" in regressed
    assert "A-BEFORE" in text and "A-NOW" in text
    stable = text[text.index("### Stable successes"):]
    assert "#### Task d: d" in stable and "D-NOW" not in text and "D-BEFORE" not in text
    assert "PIPE-START" in text and "PIPE-NOW" in text and "EDITS" in text
    assert "EARLIER-NOTE" in text and "keep its facts that still hold" in text
    assert '"memory": "<the record: facts only, no advice' in text
    first, _ = _built(entries)
    assert "(No previous optimizer memory -- this is the first update.)" in first


def test_an_unchanged_candidate_is_compared_with_its_own_earlier_rollout():
    entries, _ = memory.pairs([_rec("a", True)], [_rec("a", False)], ["a"])
    text, _ = _built(entries, same=True)
    assert "two rollouts of this one pipeline (R-PREV, and R-NOW)" in text
    assert "PIPE-START" not in text and "## What changed between them" not in text
    assert "Item a (First rollout) — passed" in text
    assert "Item a (Second rollout) — answered wrong" in text


def test_the_record_keeper_gives_no_advice_and_rules_on_no_kind_of_change():
    # Round 2 (2026-11-14): with no structural change tried, notes said "do not add new
    # steps -- the pipeline is sound" and proposers cited them; a 2026-11-23 SB note said
    # "invest in the skill knob rather than judge loops". The note is now facts only.
    flat = " ".join(memory.MEMORY_SYSTEM.split())
    assert ("Write nothing about what to do next: no recommendations, next steps, "
            "priorities, lessons, meta-lessons, principles or regression risks") in flat
    assert ("Make no claim about what can or cannot work, be built or be edited, and do not "
            "judge a kind of change in general.") in flat


def test_the_traces_share_the_room_each_cut_said_and_the_system_prompt_counted():
    prev = [_rec(f"i{n}", False, "x" * 60000) for n in range(4)]
    curr = [_rec(f"i{n}", n % 2 == 0, "y" * 60000) for n in range(4)]
    entries, _ = memory.pairs(prev, curr, [f"i{n}" for n in range(4)])
    text, clip = _built(entries, room=200000)
    assert len(text) + len(memory.MEMORY_SYSTEM) <= 200000
    assert clip["items"] == 8 and clip["chars"] > 200000
    assert "chars omitted" in text and "clipped head and tail" in text
    small, _ = memory.pairs([_rec("i0", False, "x" * 9000)], [_rec("i0", True, "y" * 9000)], ["i0"])
    whole, none = _built(small)
    assert none == {"items": 0, "chars": 0} and "chars omitted" not in whole
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


# ── a pass too long for one call: split, never cut ──────────────────────────


def _traced(item: str, passed: bool, chars: int, tag: str) -> dict[str, Any]:
    """A four-step record whose texts add up to about ``chars`` characters."""
    each = chars // 8
    return {"id": item, "passed": passed, "score": float(passed), "prediction": f"P-{item}",
            "fail_reason": "" if passed else f"WHY-{item}",
            "exchanges": [{"node": f"s{n}", "prompt": f"{tag}-{item}-IN{n} " + "I" * each,
                           "response": f"{tag}-{item}-OUT{n} " + "O" * each}
                          for n in range(4)]}


#: item -> (passed at the start of the epoch, passed now): two of each category.
SPLIT_ITEMS = {"i0": (False, True), "r0": (True, False), "p0": (False, False),
               "s0": (True, True), "r1": (True, False), "i1": (False, True),
               "p1": (False, False), "s1": (True, True)}


def _long_entries(chars: int = 80000) -> list[dict[str, Any]]:
    prev = [_traced(i, before, chars, "BEFORE") for i, (before, _now) in SPLIT_ITEMS.items()]
    curr = [_traced(i, now, chars, "NOW") for i, (_before, now) in SPLIT_ITEMS.items()]
    return memory.pairs(prev, curr, list(SPLIT_ITEMS))[0]


def _request(entries: list[dict[str, Any]], **overrides: Any) -> memory.MemoryRequest:
    return memory.MemoryRequest(**{
        "epoch": 3, "mode_name": "m",
        "prev": {"id": "c1", "pipeline": "PIPE-START", "rollout": "R-PREV"},
        "curr": {"id": "c2", "pipeline": "PIPE-NOW", "rollout": "R-NOW"},
        "changes": ["- `c2` (from `c1`): EDITS"], "previous_note": "EARLIER-NOTE",
        "entries": entries, "tried": ["## What was tried this epoch", "- TRIED-ENTRY"],
        "objective": "OBJECTIVE-TEXT", **overrides})


def _tasks(prompt: str) -> list[str]:
    import re

    return re.findall(r"^#### Task (\S+?)(?::| --|$)", prompt, re.M)


class Splitting(Client):
    """Answers each comparison call with one observation marked with its part and
    naming its items, and the synthesis with ``SPLIT-NOTE``."""

    def __init__(self, compare: Any = None, synthesis: Any = None) -> None:
        super().__init__()
        self.compare, self.synthesis = compare, synthesis

    def complete(self, prompt: str, *, system: str = "", **kwargs: Any) -> str:
        import re

        if system == memory.COMPARE_SYSTEM:
            self.calls.append((system, prompt))
            if self.compare is not None:
                return self.compare(prompt)
            part = re.search(r"part (\d+) of", prompt).group(1)
            return json.dumps({"observations": [{
                "category": "regressed", "observation": f"OBS-PART-{part}",
                "items": _tasks(prompt), "evidence": ["a quote"]}]})
        if system == memory.SYNTHESIS_SYSTEM:
            self.calls.append((system, prompt))
            if self.synthesis is not None:
                return self.synthesis(prompt)
            return json.dumps({"reasoning": "R", "memory": "SPLIT-NOTE"})
        return super().complete(prompt, system=system, **kwargs)

    def of(self, system: str) -> list[str]:
        return [p for s, p in self.calls if s == system]


def test_a_pass_too_long_for_one_call_is_split_into_comparisons_and_a_synthesis(tmp_path):
    """Six changed or failing items with two traces of 80000 characters each: past one
    call's room. Several comparison calls each read whole pairs, every pair in exactly
    one, regressions first; then one synthesis call reads every part's observations,
    the previous note, what was tried and the objective, and writes the note. Nothing
    is cut."""
    import re

    entries = _long_entries()
    request = _request(entries)
    client = Splitting()
    optimizer = AutoresearchOptimizer(client)
    optimizer.trace_path = tmp_path / "optimizer_trace.jsonl"
    made = optimizer.write_memory(epoch=3, generation=2, build=request)
    # The calls run concurrently: in part order.
    compares = sorted(client.of(memory.COMPARE_SYSTEM),
                      key=lambda p: int(re.search(r"part (\d+) of", p).group(1)))
    (synthesis,) = client.of(memory.SYNTHESIS_SYSTEM)
    assert len(compares) > 1 and client.calls[-1] == (memory.SYNTHESIS_SYSTEM, synthesis)
    assert client.memory_calls() == [], "no single call"
    for prompt in (*compares, synthesis):
        assert "chars omitted" not in prompt and "clipped head and tail" not in prompt
    for prompt in compares:
        assert len(memory.COMPARE_SYSTEM) + len(prompt) <= memory.MEMORY_ROOM
        assert "EDITS" in prompt and "OBJECTIVE-TEXT" in prompt
        # The pipelines, as the single call shows them, so a finding can say where.
        assert "## The pipeline at the start of this epoch (candidate `c1`)\nPIPE-START" in prompt
        assert "## The pipeline now (candidate `c2`)\nPIPE-NOW" in prompt
        for item in _tasks(prompt):
            entry = next(e for e in entries if e["id"] == item)
            if entry["category"] != "stable_success":
                # A step's `Ran as:` line is said once a request (`evidence.said_once`).
                bare = _bare(prompt)
                assert _bare(memory._trace(entry, "prev", False, request._settled)) in bare, \
                    "both traces whole"
                assert _bare(memory._trace(entry, "curr", False, request._settled)) in bare
            else:
                assert "BEFORE-" + item not in prompt, "stable successes without traces"
    read = [item for prompt in compares for item in _tasks(prompt)]
    assert sorted(read) == sorted(SPLIT_ITEMS), "every pair in exactly one call"
    assert read[:2] == ["r0", "r1"] and read[-2:] == ["s0", "s1"], "regressions first"
    for number in range(1, len(compares) + 1):
        assert f"OBS-PART-{number}" in synthesis
    assert "EARLIER-NOTE" in synthesis and "keep its facts that still hold" in synthesis
    assert "- TRIED-ENTRY" in synthesis and "OBJECTIVE-TEXT" in synthesis
    assert "PIPE-START" in synthesis and "PIPE-NOW" in synthesis
    assert '"memory": "<the record: facts only, no advice' in synthesis
    assert made["note"]["text"] == "SPLIT-NOTE" and made["error"] is None
    assert made["parts"] == len(compares) and made["clipped_items"] == 0
    assert made["calls"] == len(compares) + 1
    trace = [json.loads(line) for line in optimizer.trace_path.read_text().splitlines()]
    assert sorted(t["kind"] for t in trace) == sorted(
        ["memory_compare"] * len(compares) + ["memory"])
    # A resumed pass is answered from the cache: nothing is paid again.
    again = Splitting()
    resumed = AutoresearchOptimizer(again)
    resumed.trace_path = tmp_path / "optimizer_trace.jsonl"
    redo = resumed.write_memory(epoch=3, generation=2, build=_request(_long_entries()))
    assert again.calls == [] and redo["cached"] is True and redo["note"]["text"] == "SPLIT-NOTE"


def test_a_pass_that_fits_is_one_call_and_its_request_is_the_single_one():
    entries, _ = memory.pairs([_rec("a", False, "A-BEFORE"), _rec("b", True)],
                              [_rec("a", True, "A-NOW"), _rec("b", False)], ["a", "b"])
    request = _request(entries)
    assert request.parts(memory.MEMORY_ROOM) is None
    client = Splitting()
    made = AutoresearchOptimizer(client).write_memory(epoch=3, generation=2, build=request)
    (system, prompt), = client.calls
    assert system == memory.MEMORY_SYSTEM and made["note"]["text"] == "THE-NOTE"
    assert prompt == memory.build_prompt(
        epoch=3, mode_name="m",
        prev={"id": "c1", "pipeline": "PIPE-START", "rollout": "R-PREV"},
        curr={"id": "c2", "pipeline": "PIPE-NOW", "rollout": "R-NOW"},
        changes=["- `c2` (from `c1`): EDITS"], previous_note="EARLIER-NOTE", entries=entries,
        tried=["## What was tried this epoch", "- TRIED-ENTRY"], objective="OBJECTIVE-TEXT")[0]
    assert "parts" not in made


def test_a_half_done_split_pass_pays_only_for_what_it_did_not_finish(tmp_path):
    """The comparisons answered, the synthesis unreadable: the pass fails, keeping the
    earlier note. Made again (a resume), the comparisons come from the cache and only
    the synthesis is paid; a failed comparison makes no synthesis."""
    first = Splitting(synthesis=lambda prompt: "no json here")
    optimizer = AutoresearchOptimizer(first)
    optimizer.trace_path = tmp_path / "optimizer_trace.jsonl"
    made = optimizer.write_memory(epoch=3, generation=2, build=_request(_long_entries()))
    assert made["note"] is None and "no JSON object" in made["error"]
    parts = len(first.of(memory.COMPARE_SYSTEM))
    again = Splitting()
    resumed = AutoresearchOptimizer(again)
    resumed.trace_path = tmp_path / "optimizer_trace.jsonl"
    redo = resumed.write_memory(epoch=3, generation=2, build=_request(_long_entries()))
    assert again.of(memory.COMPARE_SYSTEM) == [] and len(again.of(memory.SYNTHESIS_SYSTEM)) == 1
    assert redo["note"]["text"] == "SPLIT-NOTE" and redo["calls"] == 1 and parts > 1

    def second_down(prompt: str) -> str:
        if "part 2 of" in prompt:
            return "no json here"
        return json.dumps({"observations": [{"observation": "OBS"}]})

    broken = Splitting(compare=second_down)
    failed = AutoresearchOptimizer(broken).write_memory(epoch=3, generation=2,
                                                        build=_request(_long_entries()))
    assert failed["note"] is None and failed["error"].startswith("comparison part 2 of ")
    assert broken.of(memory.SYNTHESIS_SYSTEM) == [], "no note from part of the evidence"


def test_an_unchanged_pipeline_is_shown_once_to_each_comparison_call():
    entries = _long_entries()
    client = Splitting()
    AutoresearchOptimizer(client).write_memory(epoch=3, generation=2, build=_request(
        entries, prev={"id": "c2", "pipeline": "PIPE-NOW", "rollout": "R-PREV"}))
    for prompt in client.of(memory.COMPARE_SYSTEM):
        assert "## The pipeline (candidate `c2`)" in prompt and "PIPE-NOW" in prompt
        assert "two rollouts of this one pipeline (R-PREV, and R-NOW)" in prompt


def test_long_observations_are_cut_to_the_parts_room_and_counted():
    """A comparison call that writes far more than asked: its observations are cut to
    ``PART_OBSERVATION_CHARS`` in the synthesis request, each cut marked and counted."""
    def verbose(prompt: str) -> str:
        return json.dumps({"observations": [
            {"observation": f"LONG-{n} " + "w" * 20000, "items": [], "evidence": ["q" * 50]}
            for n in range(3)]})

    client = Splitting(compare=verbose)
    made = AutoresearchOptimizer(client).write_memory(epoch=3, generation=2,
                                                      build=_request(_long_entries()))
    (synthesis,) = client.of(memory.SYNTHESIS_SYSTEM)
    parts = made["parts"]
    assert synthesis.count("… [cut: ") == 3 * parts and "q" * 50 in synthesis
    assert made["synthesis_cuts"]["observation_texts"] == 3 * parts
    assert made["synthesis_cuts"]["observation_chars"] > 3 * parts * 15000
    assert made["synthesis_cuts"]["pipeline_chars"] == 0
    kept, cut = memory.fit_observations([{"observation": "short", "evidence": ["e"]}])
    assert cut == {"texts": 0, "chars": 0} and kept[0]["observation"] == "short"
    block = synthesis.split("### Part 1 of")[1].split("### Part 2 of")[0]
    assert len(block) < memory.PART_OBSERVATION_CHARS + 1000


def test_pipelines_too_long_for_the_synthesis_are_clipped_and_counted():
    entries = _long_entries()
    big = {"prev": {"id": "c1", "pipeline": "S" * 300000, "rollout": "R-PREV"},
           "curr": {"id": "c2", "pipeline": "N" * 300000, "rollout": "R-NOW"}}
    request = _request(entries, **big)
    results = [{"part": 1, "items": ["r0"], "observations": [{"observation": "OBS"}]}]
    text, cut = request.synthesis_prompt(results, memory.MEMORY_ROOM)
    assert len(memory.SYNTHESIS_SYSTEM) + len(text) <= memory.MEMORY_ROOM
    assert "characters omitted" in text and "OBS" in text
    assert cut["pipeline_chars"] > 200000 and cut["observation_texts"] == 0
    whole, none = _request(entries).synthesis_prompt(results, memory.MEMORY_ROOM)
    assert not any(none.values()) and "chars omitted" not in whole


def test_a_stop_during_a_split_pass_starts_no_further_comparison_call(monkeypatch):
    from refactory_dsh.optim.base import RunStopped

    asked = {"stop": False}

    def first_then_stop(prompt: str) -> str:
        asked["stop"] = True  # a person asks the run to stop while this call runs
        return json.dumps({"observations": []})

    client = Splitting(compare=first_then_stop)
    optimizer = AutoresearchOptimizer(client)
    optimizer.stop_check = lambda: asked["stop"]
    monkeypatch.setattr(analysts, "CONCURRENCY", 1)  # one at a time: the rest stay queued
    with pytest.raises(RunStopped):
        optimizer.write_memory(epoch=3, generation=2, build=_request(_long_entries()))
    assert len(client.of(memory.COMPARE_SYSTEM)) == 1
    assert client.of(memory.SYNTHESIS_SYSTEM) == []


def test_a_stop_during_the_analyst_stage_starts_no_further_analyst_call(monkeypatch):
    from refactory_dsh.optim.base import RunStopped

    from test_analysts import ANALYST_SYSTEMS, Client as Analysts, _ctx, _rec as arec

    asked = {"stop": False}

    class Stopping(Analysts):
        def complete(self, prompt: str, *, system: str = "", **kwargs: Any) -> str:
            asked["stop"] = True
            return super().complete(prompt, system=system, **kwargs)

    client = Stopping()
    optimizer = AutoresearchOptimizer(client)
    optimizer.stop_check = lambda: asked["stop"]
    monkeypatch.setattr(analysts, "CONCURRENCY", 1)
    with pytest.raises(RunStopped):
        optimizer.propose(_ctx([arec(f"f{i}", False) for i in range(4)], group_size=1), 1, [])
    assert len(client.of(ANALYST_SYSTEMS)) == 1


def test_only_a_pair_longer_than_the_room_alone_is_cut_and_counted():
    entries = _long_entries(20000)
    huge = memory.pairs([_traced("h0", True, 300000, "BEFORE")],
                        [_traced("h0", False, 300000, "NOW")], ["h0"])[0]
    request = _request([*huge, *entries])
    client = Splitting()
    made = AutoresearchOptimizer(client).write_memory(epoch=3, generation=2, build=request)
    compares = client.of(memory.COMPARE_SYSTEM)
    (alone,) = [p for p in compares if "h0" in _tasks(p)]
    assert _tasks(alone) == ["h0"] and "chars omitted" in alone
    assert all("chars omitted" not in p for p in compares if p is not alone)
    assert made["clipped_items"] == 2 and made["clipped_chars"] > 100000


@pytest.mark.parametrize("fatal", ["budget", "cancel"])
def test_a_spent_budget_or_a_stop_in_a_comparison_call_stops_the_pass(fatal):
    from refactory_dsh.budgets import BudgetExhausted

    error = BudgetExhausted("spent") if fatal == "budget" else CancelledError()

    def raises(prompt: str) -> str:
        raise error

    with pytest.raises(type(error)):
        AutoresearchOptimizer(Splitting(compare=raises)).write_memory(
            epoch=3, generation=2, build=_request(_long_entries()))


def test_the_synthesis_keeps_the_memory_rules_and_the_comparison_prompt_is_domain_free():
    flat = " ".join(memory.SYNTHESIS_SYSTEM.split())
    assert memory.SYNTHESIS_SYSTEM != memory.MEMORY_SYSTEM
    assert "read in parts by comparison calls" in flat
    assert "Write nothing about what to do next" in flat
    assert "the observations) or the previous record" in flat
    assert "Write nothing about what to do next" in memory.COMPARE_SYSTEM
    assert ("Report what these traces show; leave verdicts on the changes tried to the call "
            "that weighs each of them against these observations, and judge no kind of "
            "change in general.") in " ".join(memory.COMPARE_SYSTEM.split())
    assert "Do not output instructions for the pipeline's own steps." in memory.COMPARE_SYSTEM
    assert memory.parse_observations({"findings": []}) is None
    kept = memory.parse_observations({"observations": [
        {"category": "nope", "observation": " X ", "items": ["a", None], "evidence": list("abcdef")},
        {"observation": ""}, "junk"]})
    assert kept == [{"category": "", "observation": "X", "items": ["a"],
                     "evidence": list("abcd")}]


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
    # 7 kept in epoch 2, 8 in epoch 3; 6 rejected in epoch 4: nothing replaced 8, so the
    # pass sets 8 against the rejected 6 on the items 6 changed.
    assert levels == [(5, 7), (7, 8), (8, 6)]
    assert [p["same"] for p in passes] == [False, False, False]
    assert "rejected" not in passes[0] and "rejected" not in passes[1]
    assert passes[2]["rejected"]["child"] == passes[2]["curr"]
    assert all(p["pairs"] == len(TRAIN_IDS) and not p["missing"] for p in passes[:2])
    # 8 passes t1-t6, 6 passes t1-t4: the two it broke, and only those.
    assert passes[2]["sample"] == ["t5", "t6"] and passes[2]["pairs"] == 2
    assert passes[2]["counts"] == {"regressed": 2, "persistent_fail": 0, "improved": 0,
                                   "stable_success": 0}
    assert all(p["evaluated"] == 0 for p in passes), "the epoch's own rollouts are reused"
    assert not _events(loop, "memory_rollout")
    # 5 passes t1-t3, 7 passes t1-t5: two improved, none regressed.
    assert passes[0]["counts"] == {"regressed": 0, "persistent_fail": 3, "improved": 2,
                                   "stable_success": 3}
    prompt = proposer.asked[2]["prompt"]
    assert "two rollouts of this one pipeline" not in prompt
    assert "## The rejected change's pipeline" in prompt
    assert "the rejected change against the incumbent, on the items it changed" in prompt
    assert "It fixed 0 and broke 2 of" in prompt
    assert "Rejected change `" in prompt and "Incumbent `" in prompt


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
    # Round 1 proposes nothing, so epoch 2's pass has no rejected change to compare.
    loop, proposer = _run(tmp_path, monkeypatch, [[], [(5, 3)], []], val_gate=False)
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


def test_a_split_pass_in_a_real_loop_journals_its_parts_and_its_note_is_read(
        tmp_path, monkeypatch):
    from test_analysts import _with_traces

    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(_val))
    _passes_by_level(monkeypatch)
    _with_traces(monkeypatch)
    # One pair per comparison call, as a model with a small context would need.
    monkeypatch.setattr(memory.MemoryRequest, "parts",
                        lambda self, room: [[e] for e in memory._ordered(self.entries)])
    client = Splitting()
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, epochs=2, val_gate=True)
        loop.optimizer = AutoresearchOptimizer(client)
        loop.run()
    (passed,) = _events(loop, "optimizer_memory")
    assert passed["parts"] == passed["pairs"] == len(client.of(memory.COMPARE_SYSTEM)) > 1
    assert passed["updated"] is True and loop._memory["text"] == "SPLIT-NOTE"
    assert "SPLIT-NOTE" in _section(client.calls[-1][1], "## Optimizer memory")
    trace = [json.loads(line) for line in (loop.root / "optimizer_trace.jsonl").read_text()
             .splitlines()]
    kinds = [t["kind"] for t in trace]
    assert kinds.count("memory") == 1 and kinds.count("memory_compare") == passed["parts"]
    assert f"{passed['pairs']} of 8 in {passed['parts']} parts" in (
        loop.root / "report.md").read_text()


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


@pytest.mark.parametrize("text", [
    # An example carrying ``proposal``, then the real answer cut off at the output limit.
    'For example {"proposal": {"edit": {"op": "bypass", "target": "x"}}}\n'
    '{"diagnosis": "d", "proposals": [{"edit": {"op": "set_knob", "name": "a", "value": 1}}',
    # An example, then a real answer with a trailing comma.
    'For example {"proposal": {"edit": {"op": "bypass", "target": "x"}}}\n'
    '{"diagnosis": "d", "proposals": [{"edit": {"op": "set_knob", "name": "a", "value": 1}},]}',
    # A broken answer, then a fragment carrying ``proposal``.
    '{"diagnosis": "d", "proposals": [{"edit": {"op": "set_knob", "name": "a", "value": 1}},]}\n'
    'for instance {"proposal": {"edit": {"op": "bypass", "target": "x"}}}',
])
def test_a_loosely_written_object_never_stands_in_for_a_broken_answer(text):
    found, problem = answer_json(text)
    assert problem and not found, "the broken answer is reported, not replaced by the example"


def test_memory_shows_each_item_s_input_once_and_names_the_rollout_it_refers_to():
    """Both rollouts of an item are the same task: its input, choices and expected answer
    appear once (in the first trace); a repeated instruction in the second says which
    rollout showed it."""
    task = "## Your task\n" + "Check every option against the hypotheses, one by one. " * 4

    def rec(item, passed, reply):
        return {"id": item, "passed": passed, "score": float(passed), "label": "B",
                "prediction": "B" if passed else "A", "fail_reason": "" if passed else "no",
                "metadata": {"question": f"QUESTION-{item} which holds?",
                             "choices": ["one", "two"], "answer": "B"},
                "exchanges": [{"node": "solve", "response": reply,
                               "prompt": f"QUESTION-{item} which holds?\nA. one\nB. two\n" + task}]}

    entries, _ = memory.pairs([rec("q1", True, "BEFORE")], [rec("q1", False, "NOW")], ["q1"])
    for same in (False, True):
        request = memory.MemoryRequest(epoch=3, mode_name="m",
                                       prev={"id": "c1" if not same else "c2", "pipeline": "P",
                                             "rollout": "R"},
                                       curr={"id": "c2", "pipeline": "P", "rollout": "R"},
                                       changes=[], previous_note="", entries=entries)
        for text in (request()[0], request.compare_prompt(entries, 1, 1)[0]):
            assert text.count("QUESTION-q1 which holds?") == 1, same
            assert text.count("Expected:") == 1, same
            later = "Second rollout" if same else "Now"
            first = "First rollout" if same else "Start of epoch"
            assert f"same as item q1 ({first}), step solve" in text.split(f"({later})", 1)[1]


# ── S1-s45: a rejected loop, nothing kept ───────────────────────────────────


def test_s1_s45_the_memory_reads_the_rejected_loop_its_judge_and_its_traces():
    """S1-s45 round 3: one loop (a judge after `solve`, at most one send-back) broke 8
    train items (and fixed one whose parent timed out), nothing was kept, and the note ruled on every loop. Now the
    request sets the unchanged incumbent against that rejected change on the items it
    moved, with both real traces; says what the loop was -- its judge's criteria, where
    it sits, what it sends back, its limit -- and asks for lessons about that change,
    not verdicts on kinds of change."""
    from refactory_dsh import registry
    from refactory_dsh.nn import edits as nn_edits
    from refactory_dsh.optim.base import _by_item, edit_change
    from refactory_dsh.optim.feedback import knob_values, placement_lines
    from test_edit_effect import S45_LOOP, _tried

    entry, fixture = _tried("s1_s45_loop.json")
    parent, child = _by_item(fixture["parent"]), _by_item(fixture["child"])
    moved = edit_change(fixture["parent"], fixture["child"])
    changed = [*moved["broken"], *moved["fixed"]]
    entries, missing = memory.pairs(parent, child, changed)
    assert not missing and memory.counts(entries)["regressed"] == len(moved["broken"]) == 8
    state = registry.resolve(Path("."), "livemath/solver")[1]().state_dict()
    loop_state = nn_edits.apply(S45_LOOP, state)
    entry = {**entry, "edits": memory.edits_text([S45_LOOP], knob_values(state)),
             "placed": placement_lines(state, loop_state)}
    tally = memory.change_tally_lines([entry], val_items=17, gated=True)
    tried = memory.tried_lines([entry], [], round_=3, current=fixture["parent_id"], gated=True,
                               tally=tally)
    at = "its rollout in the round the change was evaluated in (one train seed for both)"
    rejected = {"child": fixture["child_id"], "changed": len(changed), "shown": len(changed),
                "broken": len(moved["broken"]), "fixed": len(moved["fixed"]),
                "shared": moved["shared"], "others": []}
    build = memory.MemoryRequest(
        epoch=4, mode_name="livemath/solver",
        prev={"id": fixture["parent_id"], "pipeline": "PIPE-INCUMBENT", "rollout": at},
        curr={"id": fixture["child_id"], "pipeline": "PIPE-REJECTED", "rollout": at,
              "rejected": rejected},
        changes=["- the loop above"], previous_note="", entries=entries, tried=tried)
    assert build.parts(memory.MEMORY_ROOM) is None, "one call"
    text, clip = build(memory.MEMORY_ROOM)
    assert clip == {"items": 0, "chars": 0}
    # What the loop was: its judge's criteria, its place, what it sends back, its limit.
    assert "maximalist trap or fabricated authority" in text
    assert '"step": "solve"' in text and '"max_iterations": 1' in text
    assert "new agent `solve_check_judge` after `solve`" in text
    assert "new gate `solve_check` after `solve_check_judge`: reloop -> `solve`" in text
    assert "sends back at most 1 time" in text
    # The comparison: the incumbent against the rejected change, on the items it moved.
    assert "## The rejected change's pipeline" in text and "PIPE-REJECTED" in text
    assert f"Rejected change `{fixture['child_id']}`" in text
    # (Its one fixed item's parent hit a harness timeout, so it is not counted as fixed.)
    assert f"It fixed 0 and broke 8 of {moved['shared']} shared train items" in text
    for item in moved["broken"]:
        replies = [e["response"] for e in child[item]["exchanges"] if e.get("response")]
        assert replies and replies[-1].strip()[:200] in text, item
    judged = [e for item in changed for e in child[item]["exchanges"]
              if e.get("node") == "solve_check_judge"]
    assert judged and any(e["response"].strip()[:120] in text for e in judged)
    # Asked about this change, never verdicts on kinds of change.
    system = " ".join(memory.MEMORY_SYSTEM.split())
    assert "Which kinds of changes" not in system and "tend to help" not in system
    assert "Each change tried, this epoch and as the previous record holds earlier ones" in system
    assert "do not judge a kind of change in general" in system
    assert f"- {_tried('s1_s45_loop.json')[0]['change']} (loop/gate): 1 try" in text
    assert "- loop/gate:" not in text


def test_without_the_gate_a_change_not_leading_is_compared_but_never_called_rejected(
        tmp_path, monkeypatch):
    """Ungated, the train leader is current: 7 scores below 5, so nothing replaced 5. The
    pass sets 5 against 7 on the items 7 moved -- called the unkept change, since no gate
    rejected it -- and makes no fresh rollout."""
    loop, proposer = _run(tmp_path, monkeypatch, [[(5, 7)], []], val_gate=False)
    (first,) = _events(loop, "optimizer_memory")
    by_id = {c.id: c for c in loop.population}
    assert (_level(by_id[first["prev"]]), _level(by_id[first["curr"]])) == (5, 7)
    assert first["rejected"]["noun"] == memory.NOT_KEPT_NOUN and first["evaluated"] == 0
    assert not _events(loop, "memory_rollout")
    prompt = proposer.asked[0]["prompt"]
    assert "Rejected change" not in prompt and "rejected change" not in prompt
    assert f"Unkept change `{first['curr']}`" in prompt
    assert "## The unkept change's pipeline" in prompt
    assert "## What the unkept change changed" in prompt
    assert "without a validation gate the current is the train leader" in prompt
