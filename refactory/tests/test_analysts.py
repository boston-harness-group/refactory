"""The analyst stage (``LoopConfig.analysts``; `optim.analysts`): SkillOpt's per-round
analysis for the model proposer, over a graph.

- Failed and passed train items are split as SkillOpt splits them, shuffled with the
  round's seed, and cut into batches of ``analyst_group_size``; one call per batch.
- Each analyst reads the pipeline and every item's full trace: every step's input and
  output, the answer, the score and the grader's feedback; a finding names nodes,
  edges, settings or the structure.
- Findings merge level by level in batches of ``analyst_merge_batch_size``, failures and
  successes apart, then a final failure-first merge returns one ranked list (SkillOpt's
  shortcuts and fallback).
- The request shows the ranked findings in place of the readers' summaries.
- Off makes no call and leaves the request and the settings digest as they were.
- Calls are cached (a retry or a resume pays for none again), traced, journaled and
  summarised in report.md; no validation or test item reaches an analyst.

Each test fails when the behaviour it names is reverted.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh.optim import ProposalContext, analysts
from refactory_dsh.optim.base import Candidate
from refactory_dsh.optim.random_optimizer import AutoresearchOptimizer
from refactory_dsh.outerloop import ValidationGate

from test_gate_aware_context import _passes_by_level, _val
from test_skillopt import TEST_IDS, VAL_IDS, _patch_mode
from test_val_gate import LevelProbe, _loop

ANALYST_SYSTEMS = (analysts.FAILURE_SYSTEM, analysts.SUCCESS_SYSTEM)
MERGE_SYSTEMS = (analysts.MERGE_FAILURE_SYSTEM, analysts.MERGE_SUCCESS_SYSTEM,
                 analysts.MERGE_FINAL_SYSTEM)


def _rec(item: str, passed: bool, *, steps: dict[str, tuple[str, str]] | None = None,
         repeat: int = 1, error: str | None = None) -> dict[str, Any]:
    steps = steps or {"draft": (f"DRAFT-IN {item}", f"DRAFT-OUT {item}"),
                      "final": (f"FINAL-IN {item}", f"FINAL-OUT {item}")}
    return {"id": item, "passed": passed, "score": float(passed), "repeat": repeat,
            "prediction": f"PRED {item}", "label": f"GOLD {item}",
            "fail_reason": None if passed else f"GRADER {item}",
            "error_text": error, "metadata": {"question": f"QUESTION {item}"},
            "exchanges": [{"node": node, "prompt": given, "response": out}
                          for node, (given, out) in steps.items()]}


def _items(prompt: str) -> list[str]:
    return [line.split()[2] for line in prompt.splitlines() if line.startswith("### Item ")]


class Client:
    """A scripted optimizer model: each analyst reports one finding over its items;
    each merge keeps its inputs' findings in order (or answers ``merge``); the proposal
    proposes nothing. Thread-safe; every call is kept as (system, prompt)."""

    def __init__(self, *, merge: Any = None, analyst: Any = None, gate: Any = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.merge, self.analyst, self.gate = merge, analyst, gate
        self.lock = threading.Lock()

    def complete(self, prompt: str, *, system: str = "", **_: Any) -> str:
        with self.lock:
            self.calls.append((system, prompt))
        if system in ANALYST_SYSTEMS:
            if self.gate is not None:
                self.gate.wait(timeout=5)
            if self.analyst is not None:
                return self.analyst(system, prompt)
            ids = _items(prompt)
            kind = "failure" if system == analysts.FAILURE_SYSTEM else "success"
            return json.dumps({"batch_size": len(ids), "findings": [{
                "finding": f"{kind.upper()} over {'+'.join(ids)}",
                "where": {"nodes": ["draft"], "edges": ["draft -> final"], "knobs": [],
                          "structure": "no check step"},
                "change" if kind == "failure" else "preserve": f"ACT {'+'.join(ids)}",
                "items": ids + ["v1"], "impact": "high", "evidence": [f"DRAFT-OUT {ids[0]}"]}]})
        if system in MERGE_SYSTEMS:
            if self.merge is not None:
                return self.merge(system, prompt)
            lists = [json.loads(chunk.split("\n", 1)[1].split("\n\n## ")[0])
                     for chunk in prompt.split("## List ")[1:]] or [
                json.loads(prompt.split("## Failure findings\n")[1].split("\n\n## Success")[0]),
                json.loads(prompt.split("## Success findings\n")[1].split("\n\n## Answer")[0])]
            return json.dumps({"findings": [f for found in lists for f in found]})
        return json.dumps({"diagnosis": "d", "proposals": []})

    def of(self, systems: tuple[str, ...]) -> list[str]:
        return [prompt for system, prompt in self.calls if system in systems]

    def proposal(self) -> str:
        return [p for s, p in self.calls if s not in ANALYST_SYSTEMS + MERGE_SYSTEMS][-1]


def _ctx(records: list[dict[str, Any]], **settings: Any) -> ProposalContext:
    return ProposalContext(
        mode_name="m", epoch=3, item_records=records,
        analysts=analysts.settings(**settings) if settings is not None else {},
        nodes=[{"id": "draft", "kind": "agent"}, {"id": "final", "kind": "agent"}],
        edges=[{"source": "draft", "target": "final"}],
        node_states=[{"id": "draft", "kind": "agent", "prompt_template": "TEMPLATE-D",
                      "field_bindings": [["prompt_template", "draft__task"]]},
                     {"id": "final", "kind": "agent", "prompt_template": "TEMPLATE-F"}],
        knobs=[{"name": "draft__task", "kind": "str", "value": "BOUND-VALUE", "frozen": False},
               {"name": "style", "kind": "str", "value": "STYLE-VALUE", "frozen": False}])


# ── batching ────────────────────────────────────────────────────────────────


def test_items_split_by_passed_once_each_and_shuffle_with_the_rounds_seed():
    records = [_rec("t1", True), _rec("t1", False, repeat=2), _rec("t2", False),
               _rec("t3", False, error="boom"), _rec("t4", True), *(_rec(f"u{i}", False)
                                                                    for i in range(8))]
    failures, successes = analysts.split(records, 5)
    assert sorted(r["id"] for r in successes) == ["t1", "t4"], "a repeat is not a second item"
    assert sorted(r["id"] for r in failures) == sorted(["t2", "t3", *(f"u{i}" for i in range(8))])
    import random

    ids = sorted({r["id"] for r in records})
    expected = {"f": [i for i in ids if i not in ("t1", "t4")], "s": ["t1", "t4"]}
    random.Random(5).shuffle(expected["f"])
    random.Random(6).shuffle(expected["s"])
    assert [r["id"] for r in failures] == expected["f"], "failures shuffled with the seed"
    assert [r["id"] for r in successes] == expected["s"], "successes with the seed + 1"
    again, _ = analysts.split(list(reversed(records)), 5)
    assert [r["id"] for r in again] == [r["id"] for r in failures], "seeded, input order free"
    other, _ = analysts.split(records, 6)
    assert [r["id"] for r in other] != [r["id"] for r in failures], "another round, another order"


def test_batches_are_group_size_failures_first_and_capped():
    failures = [_rec(f"f{i}", False) for i in range(5)]
    successes = [_rec(f"s{i}", True) for i in range(3)]
    plan, left = analysts.batches(failures, successes, group_size=2)
    assert [(k, i, len(g)) for k, i, g in plan] == [
        ("failure", 1, 2), ("failure", 2, 2), ("failure", 3, 1), ("success", 1, 2),
        ("success", 2, 1)]
    assert left == {"failure": 0, "success": 0}
    capped, left = analysts.batches(failures, successes, group_size=2, max_batches=4)
    assert [(k, i) for k, i, _g in capped] == [("failure", 1), ("failure", 2), ("failure", 3),
                                               ("success", 1)]
    assert left == {"failure": 0, "success": 1}


def test_the_defaults_are_skillopts():
    from refactory_dsh.loop import LoopConfig

    config = LoopConfig()
    assert config.analysts is False
    assert (config.analyst_group_size, config.analyst_merge_batch_size,
            config.analyst_max_batches) == (8, 8, 0)
    assert analysts.settings() == {"group_size": 8, "merge_batch_size": 8, "max_batches": 0}
    assert analysts.FINDINGS_PER_ANALYST == 4, "SkillOpt's analyst edit budget L"


# ── what an analyst reads ──────────────────────────────────────────────────


def test_an_analyst_reads_every_steps_input_and_output_of_a_multi_node_graph():
    client = Client()
    records = [_rec("t1", False), _rec("t2", True)]
    AutoresearchOptimizer(client).propose(_ctx(records), 1, [])
    failure = client.of((analysts.FAILURE_SYSTEM,))[0]
    for text in ("QUESTION t1", "#### Step `draft` (1 of 2)", "Input it was given:\nDRAFT-IN t1",
                 "Output:\nDRAFT-OUT t1", "#### Step `final` (2 of 2)", "FINAL-IN t1",
                 "FINAL-OUT t1", "Expected: GOLD t1", "Predicted: PRED t1", "Grader: GRADER t1",
                 "### Item t1 — answered wrong (score 0.0)"):
        assert text in failure, text
    assert failure.index("DRAFT-OUT t1") < failure.index("FINAL-IN t1"), "in the order they ran"
    assert "t2" not in _items(failure)
    # The pipeline: nodes, edges, each node in full with its bound setting's value, and
    # the settings not bound to a node with theirs.
    assert "| draft | final | null |" in failure
    assert "BOUND-VALUE" in failure and "TEMPLATE-F" in failure and "STYLE-VALUE" in failure
    assert "`draft__task` (str): the value of `draft`.prompt_template" in failure
    assert "At most 4 findings" in failure
    success = client.of((analysts.SUCCESS_SYSTEM,))[0]
    assert _items(success) == ["t2"] and "FINAL-OUT t2" in success and "Grader:" not in success


def test_a_long_item_has_its_step_inputs_clipped_before_its_outputs():
    big = _rec("t1", False, steps={"a": ("I" * 50000, "O" * 3000), "b": ("J" * 50000, "P" * 3000)})
    text = analysts.fit_item(big, 20000)
    assert len(text) <= 20000 and "O" * 3000 in text and "P" * 3000 in text
    assert "characters omitted" in text and text.endswith("to fit the analyst's room.)")
    tight = analysts.fit_item(big, 6000)
    assert len(tight) <= 6000 and "O" * 3000 not in tight
    assert analysts.fit_item(_rec("t2", True), 20000) == analysts.item_text(_rec("t2", True))
    # Inputs and outputs that each fit on their own: the inputs are clipped, the
    # outputs stay whole.
    mid = _rec("t3", False, steps={"a": ("I" * 6000, "O" * 6000), "b": ("J" * 6000, "P" * 6000)})
    text = analysts.fit_item(mid, 20000)
    assert "O" * 6000 in text and "P" * 6000 in text and "I" * 6000 not in text


def test_a_finding_names_where_in_the_graph_and_only_its_batchs_items():
    payload = {"findings": [
        {"finding": "drops the units", "where": {"nodes": ["draft"], "edges": ["draft -> final"],
                                                 "knobs": ["style"], "structure": "no check"},
         "change": "add a check step", "items": ["t1", "v1"], "impact": "HIGH",
         "evidence": ["q"]},
        {"finding": ""}, "junk", *({"finding": f"more {i}"} for i in range(6))]}
    found = analysts.parse_findings(payload, "failure", ["t1", "t2"])
    assert len(found) == analysts.FINDINGS_PER_ANALYST, "at most L, empty entries dropped"
    first = found[0]
    assert first["where"] == {"nodes": ["draft"], "edges": ["draft -> final"],
                              "knobs": ["style"], "structure": "no check"}
    assert first["items"] == ["t1"] and first["not_in_batch"] == ["v1"] and first["support"] == 1
    assert first["impact"] == "high" and first["source"] == "failure"
    assert analysts.parse_findings({"patterns": []}, "failure", ["t1"]) is None


# ── merge and rank ─────────────────────────────────────────────────────────


def _merges(client: Client) -> list[tuple[str, str]]:
    names = {analysts.MERGE_FAILURE_SYSTEM: "failure", analysts.MERGE_SUCCESS_SYSTEM: "success",
             analysts.MERGE_FINAL_SYSTEM: "final"}
    return [(names[s], p.splitlines()[0]) for s, p in client.calls if s in names]


def test_findings_merge_level_by_level_failures_and_successes_apart_then_final():
    client = Client()
    records = [_rec(f"f{i}", False) for i in range(5)] + [_rec(f"s{i}", True) for i in range(2)]
    optimizer = AutoresearchOptimizer(client)
    optimizer.propose(_ctx(records, group_size=1, merge_batch_size=2), 1, [])
    merges = _merges(client)
    failure_heads = sorted(h for k, h in merges if k == "failure")
    # Five lists, two per call: level 1 merges 2+2 (the fifth goes up alone), level 2
    # merges 2 of the 3, level 3 the last 2.
    assert failure_heads == sorted([
        "# Merging the failure analysts' findings -- 2 lists, merge level 1"] * 2 + [
        "# Merging the failure analysts' findings -- 2 lists, merge level 2",
        "# Merging the failure analysts' findings -- 2 lists, merge level 3"])
    assert [h for k, h in merges if k == "success"] == [
        "# Merging the success analysts' findings -- 2 lists, merge level 1"]
    assert [k for k, _h in merges][-1] == "final", "the final merge comes last"
    analysis = optimizer.last_analysis
    assert analysis["calls"] == 7 + 6 and analysis["cached"] == 0
    assert [(m["step"], m["level"]) for m in analysis["merges"]] == [
        ("failure", 1), ("failure", 1), ("failure", 2), ("failure", 3), ("final", 0),
        ("success", 1)]
    found = analysis["findings"]
    assert [f["source"] for f in found] == ["failure"] * 5 + ["success"] * 2
    assert all(f["items"] and "v1" not in f["items"] for f in found), "only analysed items"


def test_the_final_merge_ranks_and_the_request_shows_the_ranked_list():
    ranked = {"findings": [
        {"source": "success", "finding": "SECOND-RANKED", "preserve": "KEEP-IT", "items": ["s0"]},
        {"source": "failure", "finding": "FIRST-RANKED", "change": "CHANGE-IT",
         "where": {"nodes": ["final"], "edges": ["draft -> final"]},
         "items": ["f0", "f1", "x1"], "batches": 2, "impact": "high", "evidence": ["Q1"]}]}

    def merge(system: str, prompt: str) -> str:
        if system == analysts.MERGE_FINAL_SYSTEM:
            return json.dumps({"findings": list(reversed(ranked["findings"]))})
        return json.dumps({"findings": [{"finding": "merged", "items": ["f0"]}]})

    client = Client(merge=merge)
    records = [_rec(f"f{i}", False) for i in range(3)] + [_rec("s0", True)]
    AutoresearchOptimizer(client).propose(_ctx(records, group_size=2), 1, [])
    request = client.proposal()
    section = request[request.index("## What the analysts found, ranked"):
                      request.index("## Item results")]
    assert ("3 failed item(s) in 2 batch(es) and 1 passed item(s) in 1 batch(es) of up to 2"
            in section)
    assert ('1. [failure; 2 item(s): f0, f1; impact high; from 2 findings] FIRST-RANKED -- '
            'where: step `final`; edge `draft -> final` -- change: CHANGE-IT -- evidence: "Q1"'
            in section)
    assert "2. [success; 1 item(s): s0] SECOND-RANKED -- where: unclear -- keep: KEEP-IT" in section
    assert "x1" not in section, "an id no analyst read is dropped"


def test_one_list_is_not_merged_and_one_side_needs_no_final_merge():
    client = Client()
    optimizer = AutoresearchOptimizer(client)
    optimizer.propose(_ctx([_rec(f"f{i}", False) for i in range(3)]), 1, [])
    assert _merges(client) == [], "one failure batch, no successes: no merge call"
    assert [f["finding"] for f in optimizer.last_analysis["findings"]] == [
        "FAILURE over " + "+".join(_items(client.of((analysts.FAILURE_SYSTEM,))[0]))]
    client = Client()
    AutoresearchOptimizer(client).propose(
        _ctx([_rec(f"f{i}", False) for i in range(3)], group_size=1), 1, [])
    assert [k for k, _h in _merges(client)] == ["failure"], "no final merge without successes"


def test_an_unreadable_merge_keeps_every_finding_failures_first():
    client = Client(merge=lambda system, prompt: "no json here")
    optimizer = AutoresearchOptimizer(client)
    records = [_rec(f"f{i}", False) for i in range(2)] + [_rec(f"s{i}", True) for i in range(2)]
    optimizer.propose(_ctx(records, group_size=1), 1, [])
    analysis = optimizer.last_analysis
    assert all(m["fallback"] and m["error"] for m in analysis["merges"])
    assert [f["source"] for f in analysis["findings"]] == ["failure", "failure", "success",
                                                          "success"]


def test_analysts_run_concurrently():
    barrier = threading.Barrier(analysts.CONCURRENCY)
    client = Client(gate=barrier)
    records = [_rec(f"f{i}", False) for i in range(analysts.CONCURRENCY)]
    optimizer = AutoresearchOptimizer(client)
    optimizer.propose(_ctx(records, group_size=1), 1, [])
    assert not barrier.broken and all(not b["error"] for b in optimizer.last_analysis["batches"])


def test_a_failed_analyst_is_recorded_shown_and_never_fatal(tmp_path):
    def analyst(system: str, prompt: str) -> str:
        if "f0" in _items(prompt):
            raise RuntimeError("503 overloaded")
        return json.dumps({"findings": [{"finding": "ok", "items": _items(prompt)}]})

    client = Client(analyst=analyst)
    optimizer = AutoresearchOptimizer(client)
    optimizer.trace_path = tmp_path / "optimizer_trace.jsonl"
    records = [_rec("f0", False), _rec("f1", False)]
    optimizer.propose(_ctx(records, group_size=1), 1, [])
    request = client.proposal()
    assert "could not be analysed: RuntimeError: 503 overloaded" in request
    trace = [json.loads(line) for line in optimizer.trace_path.read_text().splitlines()]
    failed = [t for t in trace if t["kind"] == "analyst_failure" and t["error"]]
    assert len(failed) == 1 and failed[0]["items"] == ["f0"]
    stored = tmp_path / "analyst_findings"
    assert len(list(stored.glob("[!r]*.json"))) == 1, "failures not cached"
    assert not list(stored.glob("round-*.json")), "nor the step that had one"


def test_a_spent_budget_during_an_analyst_stops_the_proposal():
    from refactory_dsh.budgets import BudgetExhausted

    def analyst(system: str, prompt: str) -> str:
        raise BudgetExhausted("tokens spent")

    client = Client(analyst=analyst)
    with pytest.raises(BudgetExhausted):
        AutoresearchOptimizer(client).propose(_ctx([_rec("f0", False)]), 1, [])
    assert client.of(ANALYST_SYSTEMS) and not [s for s, _p in client.calls
                                               if s not in ANALYST_SYSTEMS]


def test_calls_are_cached_for_a_retry_and_a_resume_and_traced(tmp_path):
    records = [_rec(f"f{i}", False) for i in range(2)] + [_rec("s0", True)]
    client = Client()
    optimizer = AutoresearchOptimizer(client)
    optimizer.trace_path = tmp_path / "optimizer_trace.jsonl"
    ctx = _ctx(records, group_size=1)
    optimizer.propose(ctx, 1, [])
    paid = len(client.of(ANALYST_SYSTEMS + MERGE_SYSTEMS))
    assert paid == 3 + 2
    trace = [json.loads(line) for line in optimizer.trace_path.read_text().splitlines()]
    kinds = sorted(t["kind"] for t in trace if t["kind"] != "proposal")
    assert kinds == ["analyst_failure", "analyst_failure", "analyst_merge_failure",
                     "analyst_merge_final", "analyst_success"]
    assert all(t["findings"] for t in trace if t["kind"].startswith("analyst_"))
    first = optimizer.last_analysis
    optimizer.propose(replace(ctx, ask_lessons=True, lessons={"text": "NEW LESSON"}), 1, [])
    assert len(client.of(ANALYST_SYSTEMS + MERGE_SYSTEMS)) == paid, "a retry pays for none"
    assert optimizer.last_analysis is first, "the step's analysis, its first lessons"
    # Each call is cached on its own too: a fresh optimizer without the step's file
    # answers every call from the per-call cache.
    for held in (tmp_path / "analyst_findings").glob("round-*.json"):
        held.rename(held.with_suffix(".bak"))
    percall = Client()
    alone = AutoresearchOptimizer(percall)
    alone.trace_path = tmp_path / "optimizer_trace.jsonl"
    alone.propose(ctx, 1, [])
    assert percall.of(ANALYST_SYSTEMS + MERGE_SYSTEMS) == []
    assert alone.last_analysis["calls"] == 0 and alone.last_analysis["cached"] == paid
    for held in (tmp_path / "analyst_findings").glob("round-*.bak"):
        held.rename(held.with_suffix(".json"))
    again = Client()
    resumed = AutoresearchOptimizer(again)  # a fresh process: only the run directory
    resumed.trace_path = tmp_path / "optimizer_trace.jsonl"
    # A resumed run whose lessons have changed since the step's first attempt reuses
    # the step's analysis from the run directory.
    resumed.propose(replace(ctx, ask_lessons=True, lessons={"text": "NEW LESSON"}), 1, [])
    assert again.of(ANALYST_SYSTEMS + MERGE_SYSTEMS) == []
    assert resumed.last_analysis["calls"] == 0 and resumed.last_analysis["cached"] == paid
    assert "## What the analysts found, ranked" in again.proposal()
    other = Client()
    other_model = AutoresearchOptimizer(other)
    other_model.trace_path = tmp_path / "optimizer_trace.jsonl"
    other_model._model_id = lambda: "another model"  # type: ignore[method-assign]
    other_model.propose(ctx, 1, [])
    assert len(other.of(ANALYST_SYSTEMS + MERGE_SYSTEMS)) == paid, "keyed by the model"


# ── the request ─────────────────────────────────────────────────────────────


def test_the_analysts_take_the_readers_place_when_items_do_not_fit_one_call():
    records = [_rec(f"f{i}", False) for i in range(4)] + [_rec("s0", True)]
    client = Client()
    ctx = replace(_ctx(records, group_size=2), evidence_budget=900)
    AutoresearchOptimizer(client).propose(ctx, 1, [])
    assert not any(p.startswith("# Reading trajectories") for _s, p in client.calls), "no reader"
    request = client.proposal()
    assert "The analysts' ranked findings above cover them" in request
    assert "### What the readers found" not in request
    assert request.index("## What the analysts found") < request.index("## Trajectories")
    # Off, the readers read them as before.
    off = Client()
    AutoresearchOptimizer(off).propose(replace(ctx, analysts={}), 1, [])
    assert any(p.startswith("# Reading trajectories") for _s, p in off.calls)
    assert "## What the analysts found" not in off.proposal()


def test_off_makes_no_call_and_leaves_the_request_as_it_was():
    records = [_rec(f"f{i}", False) for i in range(3)] + [_rec("s0", True)]
    off = Client()
    AutoresearchOptimizer(off).propose(replace(_ctx(records), analysts={}), 1, [])
    assert off.of(ANALYST_SYSTEMS + MERGE_SYSTEMS) == []
    plain = Client()
    base = replace(_ctx(records), analysts={})
    AutoresearchOptimizer(plain).propose(ProposalContext(**{
        f: getattr(base, f) for f in base.__dataclass_fields__ if f not in ("analysts", "analysis")}),
        1, [])
    assert off.calls == plain.calls, "the same request bytes as a context without the stage"
    assert "analyst" not in off.proposal().lower()


def test_a_parent_picker_runs_the_stage_and_hands_its_analysis_through():
    from refactory_dsh.optim import compose, policies

    client = Client()
    proposer = AutoresearchOptimizer(client)
    paired = compose.PickedParent(policies.HillClimbSelector(), proposer)
    parent = Candidate.from_state({"knobs": {}, "graph": {"nodes": {}}})
    paired.propose(replace(_ctx([_rec("f0", False)]), parent=parent.id), 1, [parent])
    assert client.of(ANALYST_SYSTEMS) and paired.last_analysis is proposer.last_analysis
    paired.last_analysis = None
    assert proposer.last_analysis is None


# ── settings ────────────────────────────────────────────────────────────────


def test_the_settings_are_checked_and_never_change_the_evidence_digest(tmp_path):
    from refactory_dsh import profiles

    settings = profiles.normalize({"analysts": 1, "analyst_group_size": 3})
    assert settings["analysts"] is True and settings["analyst_group_size"] == 3
    for bad in ({"analysts": "yes"}, {"analyst_group_size": 0},
                {"analyst_merge_batch_size": 1}, {"analyst_max_batches": -1}):
        with pytest.raises(profiles.ProfileError):
            profiles.normalize(bad)
    off = profiles.resolve(tmp_path, "circle_packing/adaptive", overrides={"optimizer": "random"})
    on = profiles.resolve(tmp_path, "circle_packing/adaptive", overrides={
        "optimizer": "random", "analysts": True, "analyst_group_size": 3,
        "analyst_merge_batch_size": 4, "analyst_max_batches": 2})
    assert on["settings"]["analysts"] is True and off["settings"]["analysts"] is False
    assert on["context_hash"] == off["context_hash"], "what is shown, not how it is scored"


def test_the_command_line_takes_the_settings():
    import argparse

    from refactory_dsh import profiles

    parser = argparse.ArgumentParser()
    profiles.add_arguments(parser)
    args = parser.parse_args(["--analysts", "1", "--analyst-group-size", "4",
                              "--analyst-merge-batch-size", "3", "--analyst-max-batches", "5"])
    assert profiles.cli_overrides(args) == {"analysts": 1, "analyst_group_size": 4,
                                            "analyst_merge_batch_size": 3,
                                            "analyst_max_batches": 5}


# ── in the loop ─────────────────────────────────────────────────────────────


def _with_traces(monkeypatch: Any) -> None:
    """Train records get a two-step trace; val records one marked VALTEXT."""
    from refactory_dsh.loop import Loop

    original = Loop._measure

    def measure(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        result, summary = original(self, candidate, workdir, part, **kwargs)
        for record in result.item_records:
            item = str(record["id"])
            text = "VALTEXT" if part != "train" else "TRAIN"
            record["exchanges"] = [{"node": "work", "prompt": f"{text}-IN {item}",
                                    "response": f"{text}-OUT {item}"},
                                   {"node": "finish", "prompt": f"{text}-IN2 {item}",
                                    "response": f"{text}-OUT2 {item}"}]
        return result, summary

    monkeypatch.setattr(Loop, "_measure", measure)


def _loop_run(tmp_path: Path, monkeypatch: Any, client: Client, **overrides: Any):
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(_val))
    _passes_by_level(monkeypatch)
    _with_traces(monkeypatch)
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, **{"epochs": 2, "val_gate": True, "analysts": True,
                                  "analyst_group_size": 2, **overrides})
        loop.optimizer = AutoresearchOptimizer(client)
        loop.run()
    return loop


def _journal(loop: Any, event: str) -> list[dict[str, Any]]:
    lines = (loop.root / "journal.jsonl").read_text().splitlines()
    return [e for e in map(json.loads, lines) if e.get("event") == event]


def test_the_loop_runs_the_stage_on_train_items_only(tmp_path, monkeypatch):
    """Val records with their own text are stored beside the train ones before every
    request is built; no analyst or merge prompt shows a val or test item, text or score."""
    from refactory_dsh.loop import Loop

    original = Loop._context

    def with_val_records(self):  # type: ignore[no-untyped-def]
        for candidate in self.population:
            self._part_results.setdefault(candidate.id, {})["val"] = [{"records": [
                {**_rec(item, True), "exchanges": [{"node": "work", "prompt": "VALTEXT",
                                                    "response": f"VALTEXT {item}"}]}
                for item in (*VAL_IDS, *TEST_IDS)]}]
        return original(self)

    monkeypatch.setattr(Loop, "_context", with_val_records)
    client = Client()
    loop = _loop_run(tmp_path, monkeypatch, client)
    prompts = client.of(ANALYST_SYSTEMS + MERGE_SYSTEMS)
    assert client.of(ANALYST_SYSTEMS), "the stage ran"
    read = {item for prompt in client.of(ANALYST_SYSTEMS) for item in _items(prompt)}
    assert read == {f"t{i}" for i in range(1, 9)}, "every train item, none other"
    stored = [r for parts in loop._part_results.values() for runs in parts.get("val", [])
              for r in runs.get("records") or ()]
    assert stored and any("VALTEXT" in json.dumps(r) for r in stored), "setup: val records exist"
    assert all("TRAIN-IN2 t" in p and "TRAIN-OUT2 t" in p for p in client.of(ANALYST_SYSTEMS))
    for prompt in prompts:
        assert "VALTEXT" not in prompt
        for item in (*VAL_IDS, *TEST_IDS):
            assert f"Item {item} " not in prompt and f'"{item}"' not in prompt, item
        for score in loop._gate().cache.values():
            assert f"{score:.6g}" not in prompt and str(score) not in prompt, score


def test_the_loop_journals_traces_and_reports_the_analysis(tmp_path, monkeypatch):
    client = Client()
    loop = _loop_run(tmp_path, monkeypatch, client)
    events = _journal(loop, "analyst_findings")
    assert events and {e["epoch"] for e in events} == {1, 2}
    first = events[0]
    assert first["failed_items"] + first["passed_items"] == 8
    # analyst_group_size 2: every batch holds at most 2 items, as few batches as that allows.
    assert all(len(b["items"]) <= 2 for e in events for b in e["batches"])
    assert len(first["batches"]) == (-(-first["failed_items"] // 2)
                                     + -(-first["passed_items"] // 2))
    assert loop._context().analysts == {"group_size": 2, "merge_batch_size": 8,
                                        "max_batches": 0}
    assert first["findings"][0]["finding"].startswith("FAILURE over")
    assert first["calls"] == len([b for b in first["batches"]]) + len(first["merges"])
    trace = [json.loads(line) for line in (loop.root / "optimizer_trace.jsonl").read_text()
             .splitlines()]
    assert sum(1 for t in trace if t["kind"].startswith("analyst_")) == sum(
        e["calls"] for e in events)
    report = json.loads((loop.root / "report.json").read_text())["analysts"]
    assert report["rounds"] == len(events) and report["calls"] == sum(e["calls"] for e in events)
    assert report["group_size"] == 2 and report["latest"]["findings"] == events[-1]["findings"]
    md = (loop.root / "report.md").read_text()
    section = md[md.index("## What the analysts found"):]
    assert f"ran in {len(events)} proposal round(s): {report['calls']} optimizer-model call(s)" \
        in section
    assert "1. [failure;" in section and "FAILURE over" in section


def test_a_resumed_run_keeps_its_analyses_and_adds_the_new_rounds(tmp_path, monkeypatch):
    client = Client()
    first = _loop_run(tmp_path, monkeypatch, client)
    before = [{k: v for k, v in e.items() if k != "event"}
              for e in _journal(first, "analyst_findings")]
    with _patch_mode(LevelProbe):
        again = Client()
        resumed = _loop(tmp_path, epochs=3, val_gate=True, analysts=True, analyst_group_size=2,
                        resume=True)
        resumed.optimizer = AutoresearchOptimizer(again)
        resumed.run()
    assert resumed._analyses[:len(before)] == before, "restored from the journal"
    assert [a["epoch"] for a in resumed._analyses[len(before):]] == [3], "and the new round"
    report = json.loads((resumed.root / "report.json").read_text())["analysts"]
    assert report["rounds"] == len(before) + 1
    assert report["calls"] == sum(a["calls"] for a in resumed._analyses)


def test_off_in_the_loop_hands_no_settings_journals_nothing_and_resumes_old_runs(
        tmp_path, monkeypatch):
    client = Client()
    loop = _loop_run(tmp_path, monkeypatch, client, analysts=False)
    assert loop._context().analysts == {}
    assert client.of(ANALYST_SYSTEMS + MERGE_SYSTEMS) == []
    assert _journal(loop, "analyst_findings") == []
    assert "analysts" not in json.loads((loop.root / "report.json").read_text())
    assert "## What the analysts found" not in (loop.root / "report.md").read_text()
    # A run whose config was written before the settings existed resumes with them off.
    from refactory_dsh.loop import LoopConfig

    written = json.loads((loop.root / "config.json").read_text())
    for key in ("analysts", "analyst_group_size", "analyst_merge_batch_size",
                "analyst_max_batches"):
        written.pop(key, None)
    assert LoopConfig.from_dict(written).analysts is False


def test_a_fresh_run_clears_the_cached_calls_and_a_resume_keeps_them(tmp_path, monkeypatch):
    from refactory_dsh.budgets import LEDGER

    client = Client()
    first = _loop_run(tmp_path, monkeypatch, client)
    stored = first.root / "analyst_findings"
    assert list(stored.glob("*.json"))
    (first.root / LEDGER).unlink()
    with _patch_mode(LevelProbe):
        fresh = _loop(tmp_path, epochs=1, val_gate=True)
        fresh.optimizer = AutoresearchOptimizer(Client())
        fresh.run()
    assert fresh.root == first.root and not stored.exists()


# ── every strategy that uses the model proposer ────────────────────────────


def _answering_analysts(monkeypatch: Any) -> list[tuple[str, str]]:
    """The search-combo model answers analyst and merge calls too; they are kept."""
    from test_search_combo import ComboModel

    seen: list[tuple[str, str]] = []
    lock = threading.Lock()
    original = ComboModel.complete

    def complete(self, prompt, *, system="", **kwargs):  # type: ignore[no-untyped-def]
        if system in ANALYST_SYSTEMS + MERGE_SYSTEMS:
            with lock:
                seen.append((system, prompt))
            return Client().complete(prompt, system=system)
        return original(self, prompt, system=system, **kwargs)

    monkeypatch.setattr(ComboModel, "complete", complete)
    return seen


@pytest.mark.parametrize("name,overrides", [
    *((name, {}) for name in __import__("refactory_dsh.strategies", fromlist=["names"]).names()),
    ("default", {"parent_picker": "evolutionary", "portfolio": "prompt+pipeline"}),
])
def test_every_model_proposer_strategy_runs_the_stage_on_real_traces(
        tmp_path, monkeypatch, name, overrides):
    """End to end on the search-combo toy (an op step, then an agent step that records
    its prompt and answer): every strategy asking the model proposer runs the analysts
    over the train items' real step traces, the request shows the ranked findings, and
    no val or test item reaches an analyst."""
    from test_search_combo import GATED_STRATEGIES, TEST, VAL, ComboModel, _parts, _run

    seen = _answering_analysts(monkeypatch)
    model = ComboModel()
    # The parts manifest for every strategy, gated or not, so val and test items exist
    # beside the train ones (without one every item is a train item).
    loop = _run(name, tmp_path, model, val_gate=name in GATED_STRATEGIES, epochs=3,
                parts_manifest=str(_parts(tmp_path)), analysts=True, analyst_group_size=2,
                **overrides)
    assert loop.optimizer.proposal_failures == []
    asked = [r for r in model.requests if "## Item results" in r.text]
    if name in ("best_of_n", "autoscientists"):
        # Single sampling rounds proposed before any train records exist: nothing to
        # analyse, so no analyst runs (nor would a reader).
        assert model.requests and not asked and seen == []
        return
    assert asked, f"{name}: no request carried train evidence"
    assert all("## What the analysts found, ranked" in r.text for r in asked), name
    analyst_prompts = [p for s, p in seen if s in ANALYST_SYSTEMS]
    assert analyst_prompts
    for prompt in analyst_prompts:
        assert "#### Step `solve` (1 of 1)\nInput it was given:\n" in prompt
    for _system, prompt in seen:
        for item in (*VAL, *TEST):
            assert f"Item {item} " not in prompt and f'"{item}"' not in prompt, item
    events = _journal(loop, "analyst_findings")
    assert events and sum(e["calls"] for e in events) == len(seen), "each call paid once"


# ── review fixes: one analysis per step, journaling, limits ────────────────


class Retrying(Client):
    """Every proposal answer is refused (an unknown knob) and writes new lessons, so the
    loop retries the step with a request whose lessons changed."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.answers = 0

    def complete(self, prompt: str, *, system: str = "", **kwargs: Any) -> str:
        if system in ANALYST_SYSTEMS + MERGE_SYSTEMS:
            return super().complete(prompt, system=system, **kwargs)
        with self.lock:
            self.calls.append((system, prompt))
            self.answers += 1
            number = self.answers
        return json.dumps({"diagnosis": "d", "lessons": f"LESSON-{number}", "proposals": [
            {"edit": {"op": "set_knob", "name": "no_such_knob", "value": 1}}]})


def test_retries_in_a_step_reuse_its_analysis_and_journal_it_once(tmp_path, monkeypatch):
    client = Retrying()
    loop = _loop_run(tmp_path, monkeypatch, client, epochs=2, attempts=3)
    proposals = [p for s, p in client.calls if s not in ANALYST_SYSTEMS + MERGE_SYSTEMS]
    events = _journal(loop, "analyst_findings")
    assert len(proposals) >= 2 * len(events), "setup: every step was retried"
    assert any("LESSON-1" in p for p in proposals[1:]), "setup: the lessons changed"
    assert len({e["key"] for e in events}) == len(events) == 2, "one analysis per step"
    paid = client.of(ANALYST_SYSTEMS + MERGE_SYSTEMS)
    assert len(paid) == sum(e["calls"] for e in events), "every call paid once"
    assert len(paid) == len(set(paid)), "no call repeated"
    per_step = [len(e["batches"]) + len(e["merges"]) for e in events]
    assert len(paid) == sum(per_step)


def test_a_failed_proposal_still_journals_the_analysis_it_paid_for(tmp_path, monkeypatch):
    class Down(Client):
        def complete(self, prompt: str, *, system: str = "", **kwargs: Any) -> str:
            if system in ANALYST_SYSTEMS + MERGE_SYSTEMS:
                return super().complete(prompt, system=system, **kwargs)
            raise RuntimeError("503 overloaded")

    client = Down()
    loop = _loop_run(tmp_path, monkeypatch, client, epochs=1)
    events = _journal(loop, "analyst_findings")
    assert events and events[0]["calls"] == len(client.of(ANALYST_SYSTEMS + MERGE_SYSTEMS)) > 0
    assert loop.optimizer.proposal_failures, "setup: the proposal call failed"


def test_a_spent_budget_mid_stage_journals_what_was_paid(tmp_path, monkeypatch):
    from refactory_dsh.budgets import BudgetExhausted

    def analyst(system: str, prompt: str) -> str:
        if system == analysts.SUCCESS_SYSTEM:
            raise BudgetExhausted("tokens spent")
        return json.dumps({"findings": [{"finding": "x", "items": _items(prompt)}]})

    client = Client(analyst=analyst)
    optimizer = AutoresearchOptimizer(client)
    records = [_rec("f0", False), _rec("s0", True)]
    with pytest.raises(BudgetExhausted):
        optimizer.propose(_ctx(records, group_size=1), 1, [])
    partial = optimizer.last_analysis
    assert partial["calls"] == 2 and partial["error"].startswith("BudgetExhausted")
    assert analysts.journal_entry(partial)["error"].startswith("BudgetExhausted")


def test_the_lessons_reach_the_analysts_and_the_merges():
    client = Client()
    records = [_rec(f"f{i}", False) for i in range(2)] + [_rec(f"s{i}", True) for i in range(2)]
    ctx = replace(_ctx(records, group_size=1), ask_lessons=True,
                  lessons={"text": "LESSON-TEXT: prefer checks"})
    AutoresearchOptimizer(client).propose(ctx, 1, [])
    for prompt in client.of(ANALYST_SYSTEMS + MERGE_SYSTEMS):
        assert "## The optimizer's lessons so far" in prompt and "LESSON-TEXT" in prompt
    unasked = Client()
    AutoresearchOptimizer(unasked).propose(replace(ctx, ask_lessons=False), 1, [])
    assert not any("LESSON-TEXT" in p for p in unasked.of(ANALYST_SYSTEMS + MERGE_SYSTEMS))


def test_a_level_merge_drops_ids_no_analyst_read():
    def merge(system: str, prompt: str) -> str:
        return json.dumps({"findings": [{"finding": "merged", "items": ["f0", "v9", "x1"]}]})

    client = Client(merge=merge)
    optimizer = AutoresearchOptimizer(client)
    optimizer.propose(_ctx([_rec(f"f{i}", False) for i in range(3)], group_size=1), 1, [])
    (found,) = optimizer.last_analysis["findings"]
    assert found["items"] == ["f0"] and found["not_in_batch"] == ["v9", "x1"]
    assert "v9" not in client.proposal() and "x1" not in client.proposal()


def test_the_items_the_analysts_cite_are_shown_in_full_first():
    """Twelve failed items that do not fit one request; the analysts cite only f9:
    it is the first item shown in full, before f0 (first in the usual order)."""
    def merge(system: str, prompt: str) -> str:
        return json.dumps({"findings": [{"finding": "CITED", "items": ["f9"]}]})

    client = Client(merge=merge)
    records = [_rec(f"f{i}", False, steps={"a": (f"IN {i}", "O" * 1000)}) for i in range(12)]
    ctx = replace(_ctx(records, group_size=4), evidence_budget=8000)
    AutoresearchOptimizer(client).propose(ctx, 1, [])
    request = client.proposal()
    shown = request[request.index("### Some items in full"):]
    assert "The items the analysts cited come first" in shown
    assert shown.count("### Item ") >= 2, "setup: room for more than the cited item"
    assert shown.split("### Item ")[1].startswith("f9 ")


def test_at_most_the_concurrency_cap_of_calls_run_at_once_merges_included():
    in_flight = {"now": 0, "most": 0}
    lock = threading.Lock()
    barrier = threading.Barrier(analysts.CONCURRENCY)
    # Level-1 merges of 12 + 12 lists in pairs: 12 merge calls are asked for at once; a
    # barrier one wider than the cap holds them until the cap would be passed.
    merges = threading.Barrier(analysts.CONCURRENCY + 1)

    class Counting(Client):
        def complete(self, prompt: str, *, system: str = "", **kwargs: Any) -> str:
            if system not in ANALYST_SYSTEMS + MERGE_SYSTEMS:
                return super().complete(prompt, system=system, **kwargs)
            with lock:
                in_flight["now"] += 1
                in_flight["most"] = max(in_flight["most"], in_flight["now"])
            try:
                try:
                    (barrier if system in ANALYST_SYSTEMS else merges).wait(timeout=0.5)
                except threading.BrokenBarrierError:
                    pass
                return super().complete(prompt, system=system, **kwargs)
            finally:
                with lock:
                    in_flight["now"] -= 1

    records = [_rec(f"f{i}", False) for i in range(12)] + [_rec(f"s{i}", True) for i in range(12)]
    optimizer = AutoresearchOptimizer(Counting())
    optimizer.propose(_ctx(records, group_size=1, merge_batch_size=2), 1, [])
    assert in_flight["most"] == analysts.CONCURRENCY == 8
    assert all(not b["error"] for b in optimizer.last_analysis["batches"])


@pytest.mark.parametrize("fatal", ["cancelled", "model_choice"])
def test_a_cancelled_run_or_a_missing_model_choice_stops_the_proposal(fatal):
    from concurrent.futures import CancelledError

    from refactory_dsh.model import ModelChoiceError

    def analyst(system: str, prompt: str) -> str:
        raise CancelledError() if fatal == "cancelled" else ModelChoiceError("no model chosen")

    with pytest.raises(CancelledError if fatal == "cancelled" else ModelChoiceError):
        AutoresearchOptimizer(Client(analyst=analyst)).propose(_ctx([_rec("f0", False)]), 1, [])


def test_a_proposer_that_asks_no_model_is_refused_and_a_custom_strategy_warned(
        tmp_path, monkeypatch):
    from refactory_dsh.loop import LoopError
    from refactory_dsh.outerloop import OuterLoop

    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, analysts=True)  # the random proposer
        with pytest.raises(LoopError, match="asks no model"):
            loop.run()
        quiet = _loop(tmp_path / "quiet")
        with __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("error")
            quiet.set_custom_strategy(OuterLoop())
        warned = _loop(tmp_path / "warned", analysts=True)
        with pytest.warns(RuntimeWarning, match="builds its own optimizer requests"):
            warned.set_custom_strategy(OuterLoop())


def _long(item: str, chars: int, passed: bool = False) -> dict[str, Any]:
    """An item of four steps whose texts add up to about ``chars`` characters."""
    each = chars // 8
    return _rec(item, passed, steps={f"s{n}": ("I" * each, "O" * each) for n in range(4)})


def test_a_small_batch_of_long_traces_is_not_clipped():
    """Three four-step items of about 150000 characters each: 450000 together fit the
    call's room, so none is clipped (an equal fixed share per item would have cut them)."""
    group = [_long(f"f{i}", 150000) for i in range(3)]
    prompt, clip = analysts.analyst_batch(_ctx(group), "failure", 1, 1, group, "PIPE")
    assert clip == {"items": 0, "chars": 0}
    assert "characters omitted" not in prompt
    assert all(analysts.item_text(record) in prompt for record in group)
    assert len(prompt) <= analysts.ANALYST_ROOM


def test_a_batch_over_the_room_has_its_longest_items_clipped_to_one_length():
    shorts = [_long(f"s{i}", 5000) for i in range(6)]
    longs = [_long("f6", 300000), _long("f7", 400000)]
    group = shorts + longs
    prompt, clip = analysts.analyst_batch(_ctx(group), "failure", 1, 1, group, "PIPE")
    assert len(prompt) <= analysts.ANALYST_ROOM
    blocks = {item: prompt.split(f"### Item {item} ")[1].split("\n\n### Item ")[0]
              .split("\n\n## Answer")[0] for item in ("s0", "f6", "f7")}
    assert "characters omitted" not in blocks["s0"], "the short items stay whole"
    assert all(analysts.item_text(r) in prompt for r in shorts)
    assert "characters omitted" in blocks["f6"] and "characters omitted" in blocks["f7"]
    # Both long items are cut to the same length (within one clipped text's rounding),
    # and together they use the room the short ones leave.
    assert abs(len(blocks["f6"]) - len(blocks["f7"])) <= 200
    assert len(prompt) > analysts.ANALYST_ROOM - 2000
    wholes = {r["id"]: len(analysts.item_text(r)) for r in longs}
    assert clip["items"] == 2
    left_out = sum(wholes.values()) - sum(len(prompt.split(f"### Item {i} ")[1]
                                              .split("\n\n### Item ")[0]
                                              .split("\n\n## Answer")[0]) + len(f"### Item {i} ")
                                          for i in ("f6", "f7"))
    assert abs(clip["chars"] - left_out) <= 4


def test_the_item_cap_fills_the_room_and_keeps_its_floor():
    assert analysts.item_cap([10, 20], 30) is None, "fits whole"
    assert analysts.item_cap([1000, 9000, 9000], 15000) == 7000
    assert analysts.item_cap([90000] * 8, 8000) == analysts.ITEM_FLOOR


def test_clipped_items_and_characters_are_journaled_and_reported():
    group = [_long("f0", 600000)] + [_rec(f"f{i}", False) for i in range(1, 8)]
    optimizer = AutoresearchOptimizer(Client())
    optimizer.propose(_ctx(group), 1, [])
    entry = analysts.journal_entry(optimizer.last_analysis)
    (batch,) = entry["batches"]
    assert entry["clipped_items"] == batch["clipped"] == 1
    assert entry["clipped_chars"] == batch["clipped_chars"] > 100000
    from refactory_dsh.loop import LoopConfig, _analysts_report, _analysts_section

    report = _analysts_report([{**entry, "epoch": 1, "generation": 1}],
                              LoopConfig(analysts=True))
    assert report["clipped_items"] == 1 and report["clipped_chars"] == entry["clipped_chars"]
    assert (f"1 item trace(s) were clipped to fit an analyst's room "
            f"({entry['clipped_chars']} characters left out)"
            in "\n".join(_analysts_section(report)))


# ── delta review: step keys and journaling after a resume ──────────────────


def test_each_parent_and_each_train_sample_gets_its_own_analysis():
    client = Client()
    optimizer = AutoresearchOptimizer(client)
    records = [_rec("f0", False), _rec("f1", False), _rec("s0", True)]
    base = replace(_ctx(records, group_size=1), parent="p1")
    optimizer.propose(base, 1, [])
    first = optimizer.last_analysis
    optimizer.propose(replace(base, parent="p2"), 1, [])
    other_parent = optimizer.last_analysis
    assert other_parent is not first and other_parent["key"] != first["key"]
    assert other_parent["candidate"] == "p2"
    resampled = [_rec("f2", False), _rec("s1", True)]
    optimizer.propose(replace(base, item_records=resampled), 1, [])
    other_sample = optimizer.last_analysis
    assert other_sample is not first and other_sample["key"] != first["key"]
    assert {i for b in other_sample["batches"] for i in b["items"]} == {"f2", "s1"}
    assert other_sample["calls"] > 0, "new items, new calls"


def _keep(analyses: list[dict[str, Any]], analysis: dict[str, Any]) -> list[tuple[str, dict]]:
    """``Loop._keep_analysis`` on a loop that has journaled ``analyses`` (as a resume
    replays them); what it journals."""
    from types import SimpleNamespace

    from refactory_dsh.loop import Loop

    journal: list[tuple[str, dict]] = []
    stub = SimpleNamespace(config=SimpleNamespace(analysts=True),
                           optimizer=SimpleNamespace(last_analysis=analysis),
                           _analyses=analyses, epoch=1, generation=1,
                           _journal=lambda event, **entry: journal.append((event, entry)))
    Loop._keep_analysis(stub)  # type: ignore[arg-type]
    return journal


def _made(**overrides: Any) -> dict[str, Any]:
    return {"key": "K", "candidate": "c", "merges": [], "calls": 1, "cached": 8,
            "batches": [{"kind": "failure", "index": 1, "items": ["f0"], "error": None}],
            "findings": [{"finding": "new", "items": ["f0"]}], **overrides}


@pytest.mark.parametrize("earlier", [
    {"batches": [{"kind": "failure", "index": 1, "items": ["f0"], "error": "RuntimeError: 503"}]},
    {"merges": [{"step": "final", "level": 0, "batch": 1, "error": "ValueError: no json"}]},
    {"error": "BudgetExhausted: tokens spent", "findings": []},
], ids=["failed batch", "failed merge", "stopped by the budget"])
def test_a_step_remade_after_an_incomplete_analysis_is_journaled_again(earlier):
    """After a resume the journal holds the step's incomplete first analysis; the step is
    remade (its failed call paid again) and that analysis, the one the proposer saw, is
    journaled too, so every paid call is counted."""
    first = {"epoch": 1, "generation": 1, "key": "K", "calls": 9, "cached": 0,
             "batches": [{"kind": "failure", "index": 1, "items": ["f0"], "error": None}],
             "merges": [], "findings": [{"finding": "old"}], **earlier}
    journal = _keep([first], _made())
    assert [event for event, _e in journal] == ["analyst_findings"]
    assert journal[0][1]["findings"] == [{"finding": "new", "items": ["f0"]}]
    assert journal[0][1]["calls"] == 1


def test_a_complete_analysis_or_the_same_incomplete_one_is_not_journaled_twice():
    complete = {"epoch": 1, "generation": 1, **analysts.journal_entry(_made())}
    assert _keep([complete], _made(calls=0, cached=9)) == [], "reloaded after a resume"
    failed = _made(batches=[{"kind": "failure", "index": 1, "items": ["f0"],
                             "error": "RuntimeError: 503"}])
    analyses: list[dict[str, Any]] = []
    assert len(_keep(analyses, failed)) == 1 and len(analyses) == 1
    assert _keep(analyses, failed) == [], "a retry reusing the same incomplete analysis"
