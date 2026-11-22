"""The memory pass writes a record, not advice (`optim.memory`), for the loop's own
model proposer; the SkillOpt strategy keeps its published meta skill.

- One task-general prompt and one fixed note format (`RECORD_FORMAT`, JSON).
- The prompt asks for nothing forward-looking, and an answer's advice is stripped.
- The facts (what each change was, kept / not kept / refused and why, the train items
  it fixed and broke, the items that keep failing) are the harness's, from the journal.
- The SkillOpt strategy's meta-skill prompt and note format are byte-identical to main.

Each test fails when the behaviour it names is reverted (mutation checks in the PR).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh.optim import evidence, memory
from refactory_dsh.optim.base import edit_change

from test_optimizer_memory import _events, _run
from test_val_gate import _level

# ── the prompt and the answer's form ──────────────────────────────────────────

#: What the optimizer-coach prompt asked for (#705), and SkillOpt's meta-skill asks: none
#: may be asked of the record's model, in any of its prompts or its answer format.
FORWARD_ASKS = (
    "optimizer-coach", "what to try next", "What to try next", "durable principles",
    "What level of detail works best", "failure-repair patterns should be prioritized",
    "regression risks future optimizer calls should guard against",
    "Address the FUTURE OPTIMIZER", "help future optimizer calls", "This memory should capture",
    "Focus on how to choose and write better changes", "Advise only on changes",
    "guidance for future", "HIGHEST PRIORITY", "Reflect on whether it improved or harmed",
)
#: Dataset and task names no record prompt may hold (the same prompt for every task).
TASK_WORDS = ("livemath", "spreadsheet", "searchqa", "alfworld", "skill", "math", "excel")


def _rec(item: str, passed: bool) -> dict[str, Any]:
    return {"id": item, "passed": passed, "score": float(passed), "prediction": f"P-{item}",
            "fail_reason": "" if passed else f"WHY-{item}",
            "exchanges": [{"node": "solve", "prompt": f"IN-{item}", "response": f"OUT-{item}"}]}


def test_the_record_prompts_ask_nothing_forward_looking_and_name_no_task():
    entries, _ = memory.pairs([_rec("a", False), _rec("b", True)],
                              [_rec("a", True), _rec("b", False)], ["a", "b"])
    request = memory.MemoryRequest(
        epoch=3, mode_name="m", prev={"id": "c1", "pipeline": "P1", "rollout": "R1"},
        curr={"id": "c2", "pipeline": "P2", "rollout": "R2"}, changes=["- `c2`"],
        entries=entries, tried=["## What was tried this epoch", "- `c2` ..."])
    results = [{"part": 1, "items": ["a"], "observations": [{"observation": "O", "items": ["a"]}]}]
    texts = {
        "system": memory.MEMORY_SYSTEM, "synthesis": memory.SYNTHESIS_SYSTEM,
        "compare": memory.COMPARE_SYSTEM, "answer": memory.answer_text(),
        "compare answer": memory.compare_answer_text(), "request": request()[0],
        "compare request": request.compare_prompt(entries, 1, 1)[0],
        "synthesis request": request.synthesis_prompt(results)[0],
        "reader lead": memory.RECORD_LEAD,
    }
    for name, text in texts.items():
        flat = " ".join(text.split())
        for ask in FORWARD_ASKS:
            assert ask not in flat, (name, ask)
        lowered = flat.lower()
        assert not [w for w in TASK_WORDS if w in lowered], name
    for system in (memory.MEMORY_SYSTEM, memory.SYNTHESIS_SYSTEM, memory.COMPARE_SYSTEM):
        flat = " ".join(system.split())
        # The prompt forbids what it used to ask for.
        assert ("Write nothing about what to do next: no recommendations, suggestions, "
                "principles, rules, lessons, priorities or risks to guard against") in flat
        assert "Make no claim about what can or cannot be built, edited, reached or unfrozen." in flat
    # The request carries no earlier note to rewrite.
    assert "Previous optimizer memory" not in texts["request"]


def _tried(cid: str, round_: int, parent: str, *, gate: str, fixed: list[str],
           broken: list[str], failing: list[str], shared: list[str]) -> dict[str, Any]:
    return {"id": cid, "round": round_, "parent": parent, "kind": "prompt edit",
            "structural": False, "touched": ["solve"], "edits": f"EDITS-{cid}",
            "evaluated": True, "exit_ok": True, "gate": gate, "fixed": len(fixed),
            "broken": len(broken), "shared": len(shared),
            "items": {"fixed": fixed, "broken": broken, "failing": failing, "shared": shared},
            "own": {"scored": shared, "failing": failing}}


FACTS = memory.record_facts(
    [_tried("c2", 1, "c1", gate="kept", fixed=["a"], broken=[], failing=["c"],
            shared=["a", "b", "c"]),
     _tried("c3", 2, "c2", gate="rejected", fixed=[], broken=["b"], failing=["b", "c"],
            shared=["a", "b", "c"])],
    [{"round": 2, "attempt": 1, "kind": "tools", "structural": True, "touched": ["solve"],
      "edits": "EDITS-REFUSED", "reasons": ["E-tool-policy-escalation: node 'solve' ..."]}],
    gated=True, current="c2")


def test_an_answer_is_read_structurally_and_the_facts_stay_the_harnesss():
    """No wording is judged (the prompt's rules are what keeps advice out; a pattern
    filter misjudged both ways): what is checked is structure. An observation naming no
    compared item is dropped, one pinned to a change this pass did not compare is kept
    unpinned, each said why; a reply that writes facts of its own changes none."""
    payload = {
        "changes": [{"id": "c3", "outcome": "kept", "fixed": ["b"]}],  # not the model's to say
        "observations": [
            {"change": "c3", "items": ["b"], "observation": "In b, the step solve read the "
             "second table and returned 7 where the grader expected 4.", "evidence": ["7"]},
            {"change": "c3", "items": ["b"], "observation": "The skill text said 'must cite'."},
            {"change": "c2", "items": ["c"], "observation": "In c, step solve stopped early."},
            {"change": "c2", "items": ["zz"], "observation": "Item zz was answered late."},
            {"change": "c9", "items": ["c"], "observation": "In c both pipelines stopped at step 2."},
        ]}
    note = memory.note_from(payload, epoch=3, facts=FACTS, items=["a", "b", "c"],
                            compared={"epoch": 3, "prev": "c2", "curr": "c3"}, pinnable=["c3"])
    kept = note["record"]["observations"]
    assert [(o["change"], o["text"][:12]) for o in kept] == [
        ("c3", "In b, the st"), ("c3", "The skill te"), (None, "In c, step s"),
        (None, "In c both pi")]
    assert note["stripped"] == [{"text": "Item zz was answered late.",
                                 "why": "it names no item of the comparison"}]
    assert [(u["change"], u["why"]) for u in note["unpinned"]] == [
        ("c2", "`c2` is not a change this pass compared"),
        ("c9", "`c9` is not a change this pass compared")]
    view = json.loads(note["text"])
    assert view["format"] == memory.RECORD_FORMAT
    c3 = next(c for c in view["changes"] if c["id"] == "c3")
    assert (c3["outcome"], c3["fixed"], c3["broken"]) == ("not kept", [], ["b"])
    assert [o["items"] for o in c3["observations"]] == [["b"], ["b"]]
    assert not hasattr(memory, "advice_in"), "no wording filter"
    assert memory.note_from({"memory": "a coach's note"}, facts=FACTS) is None, "no list, no note"
    assert json.loads(memory.note_from({"observations": []}, facts=FACTS)["text"])["changes"]


def test_keeps_failing_reads_each_changes_own_records():
    """A change whose parent was not scored with its seed (after a resume, say) has no
    fixed/broken ids, but its own records still count: an item it passed never keeps
    failing."""
    tried = [_tried("c2", 1, "c1", gate="kept", fixed=[], broken=[], failing=["t6"],
                    shared=["t5", "t6"]),
             _tried("c3", 2, "c2", gate="rejected", fixed=[], broken=[], failing=["t6"],
                    shared=["t5", "t6"]),
             {**_tried("c8", 3, "c2", gate="current", fixed=[], broken=[], failing=[],
                       shared=["t5", "t6"]), "items": None, "fixed": None, "broken": None}]
    facts = memory.record_facts(tried, [], gated=True, current="c8")
    assert facts["keeps_failing"] == [], "c8 passes t6"
    c8 = facts["changes"][-1]
    assert (c8["fixed"], c8["broken"], c8["outcome"]) == (None, None, "kept")
    assert memory.record_facts(tried[:2], [], gated=True, current="c2")["keeps_failing"] == [
        {"item": "t6", "failed_under": ["c2", "c3"]}]


def test_without_a_gate_a_change_is_kept_when_it_led_or_leads_through_a_descendant():
    tried = [_tried(c, r, p, gate=None, fixed=[], broken=[], failing=[], shared=["a"])
             for c, r, p in (("c2", 1, "c1"), ("c3", 2, "c2"), ("c4", 2, "c2"), ("c5", 3, "c1"))]
    facts = memory.record_facts(tried, [], gated=False, current="c3",
                                leaders=["c1", "c5", "c3"], lineage=["c3", "c2", "c1"])
    assert [(c["id"], c["outcome"], c["why"]) for c in facts["changes"]] == [
        ("c2", "kept", "no validation gate: an ancestor of the train leader now"),
        ("c3", "kept", "no validation gate: the train leader now"),
        ("c4", "not kept", "no validation gate: never the train leader"),
        ("c5", "kept", "no validation gate: the train leader at an earlier epoch")]


def test_the_record_is_one_fixed_json_form_and_its_oldest_entries_go_first():
    note = memory.note_from({"observations": []}, epoch=3, facts=FACTS, items=[])
    text = note["text"]
    assert list(json.loads(text)) == ["format", "epoch", "passes", "changes",
                                      "keeps_failing", "observations"]
    lines = text.splitlines()
    assert lines[0] == "{" and lines[-1] == "}"
    changes = [json.loads(line.strip().rstrip(",")) for line in lines if '"round"' in line]
    assert [(c["id"], c["outcome"]) for c in changes] == [
        ("c2", "kept"), ("c3", "not kept"), (None, "refused")]
    assert changes[2]["why"].startswith("E-tool-policy-escalation")
    assert json.loads(text)["keeps_failing"] == [{"item": "c", "failed_under": ["c2", "c3"]}]
    # Past its room the oldest change is left out first, and named.
    tight = memory.render_record(note["record"], chars=len(text) - 1)
    shown = json.loads(tight)
    assert shown["omitted"] == {"changes": ["c2"]} and len(tight) < len(text)
    assert [c["id"] for c in shown["changes"]] == ["c3", None]


def _journal_oracle(events: list[dict[str, Any]], at: dict[str, Any]) -> dict[str, Any]:
    """What the journal alone says of every change and refusal before the pass ``at``,
    read independently of the loop's code: each change's round (``proposal``) and parent,
    its fixed/broken ids against its parent on the records of the round it was evaluated
    in (``train_records``/``train_records_epoch``; none when the parent has no records of
    that round), kept or not (the gate's ``strategy_state``; without one, the leaders
    ``epoch_current`` named and the current's ancestors), the refusals (``rejection``),
    and the items wrong under every change that scored them, by each change's own
    records."""
    before = events[:events.index(at)]
    latest: dict[str, list[dict[str, Any]]] = {}
    by_epoch: dict[str, dict[int, list[dict[str, Any]]]] = {}
    last_epoch: dict[str, int] = {}
    rounds, parents, leaders, refusals, gate = {}, {}, set(), [], None
    for e in before:
        kind = e.get("event")
        if kind == "train_records":
            latest[e["id"]] = e["records"]
        elif kind == "train_records_epoch":
            by_epoch.setdefault(e["id"], {})[int(e["epoch"])] = latest.get(e["id"], [])
            last_epoch[e["id"]] = int(e["epoch"])
        elif kind == "candidate":
            parents[e["id"]] = e.get("parent")
        elif kind == "proposal":
            rounds[e["id"]] = int(e["epoch"])
        elif kind == "epoch_current":
            leaders.add(e["id"])
        elif kind == "rejection":
            refusals.append((int(e["epoch"]), e["reasons"][0]))
        elif kind == "strategy_state" and "val_gate" in (e.get("state") or {}):
            gate = e["state"]["val_gate"]
    current = at["prev"] if at.get("rejected") else at["curr"]
    lineage, node = set(), current
    while node:
        lineage.add(node)
        node = parents.get(node)
    changes, own = {}, {}
    for cid, round_ in rounds.items():
        if round_ >= at["epoch"] or cid not in latest:
            continue
        parent = parents[cid]
        if gate is None:
            pair, child = latest.get(parent), latest[cid]
        else:
            first = min(by_epoch.get(cid) or {0: None})
            child = (by_epoch[cid][first] if by_epoch.get(cid)
                     and last_epoch.get(cid) != first else latest[cid])
            pair = (by_epoch.get(parent) or {}).get(first)
        moved = edit_change(pair, child) if pair and child else None
        kept = (cid in gate["accepted"]) if gate is not None else (cid in leaders | lineage)
        changes[cid] = (round_, parent, None if not moved or not moved["shared"] else
                        (moved["fixed"], moved["broken"]), "kept" if kept else "not kept")
        own[cid] = {r["id"]: bool(r.get("passed")) for r in child}
    scored: dict[str, list[str]] = {}
    for cid in sorted(own, key=lambda c: (rounds[c], c)):
        for item, passed in own[cid].items():
            scored.setdefault(item, []).append(cid)
    failing = sorted([(item, under) for item, under in scored.items() if len(under) >= 2
                      and not any(own[c][item] for c in under)],
                     key=lambda f: (-len(f[1]), f[0]))
    return {"changes": changes, "refused": [r for r in refusals if r[0] < at["epoch"]],
            "keeps_failing": failing}


def _scenario(name: str, tmp_path: Path, monkeypatch: Any) -> Any:
    from test_optimizer_memory import Remembering
    from test_skillopt import _patch_mode
    from test_val_gate import LevelProbe, _loop

    if name == "gated":
        return _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 8)], [(8, 6)], []])[0]
    if name == "refusal":  # level 12 is past the knob's range: refused before it runs
        return _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 12)], [(7, 8)], []])[0]
    if name == "failed_pass":
        plan = [[(5, 7)], [(7, 8)], [(8, 6)], []]
        return _run(tmp_path, monkeypatch, plan, proposer=Remembering(plan, fail=(3,)))[0]
    if name == "ungated":
        # Without the gate a lower level scores higher: 3 and 4 from 5 (3 leads, 4 never
        # does), then 2 from 4 leads: 4, never the leader, is kept as 2's ancestor.
        return _run(tmp_path, monkeypatch, [[(5, 3), (5, 4)], [(4, 2)], []],
                    val_gate=False, batch_size=2)[0]
    assert name == "resume"
    _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 8)], []])
    with _patch_mode(LevelProbe):
        resumed = _loop(tmp_path, epochs=5, val_gate=True, resume=True)
        resumed.optimizer = Remembering([[(8, 6)], []])
        resumed.run()
    return resumed


@pytest.mark.parametrize("name", ["gated", "refusal", "failed_pass", "resume", "ungated"])
def test_the_records_facts_match_the_journal(name, tmp_path, monkeypatch):
    """Every memory pass's record says, for every change and refusal so far, exactly what
    the journal says (`_journal_oracle`, read without the loop's code): a gated run, one
    with a refused proposal, one whose memory call failed (its facts written all the
    same), a resumed one, and one without the gate (a change kept when it led, or leads
    through a descendant)."""
    loop = _scenario(name, tmp_path, monkeypatch)
    events = [json.loads(line) for line in (loop.root / "journal.jsonl").read_text().splitlines()]
    passes = [e for e in events if e["event"] == "optimizer_memory"]
    assert len(passes) >= 2
    for at in passes:
        oracle = _journal_oracle(events, at)
        record = at["memory"]["record"]
        changes = [c for c in record["changes"] if c["id"]]
        assert {c["id"]: (c["round"], c["parent"], None if c["fixed"] is None
                          else (c["fixed"], c["broken"]), c["outcome"])
                for c in changes} == oracle["changes"], (name, at["epoch"])
        assert [(c["round"], c["why"].split(" | ")[0]) for c in record["changes"]
                if c["outcome"] == "refused"] == [
            (r, memory._clip(why, memory.TRIED_REASON_CHARS)) for r, why in oracle["refused"]]
        assert [(f["item"], f["failed_under"]) for f in record["keeps_failing"]] == \
            oracle["keeps_failing"], (name, at["epoch"])
    levels = {c.id: _level(c) for c in loop.population}
    last = {levels[c["id"]]: c["outcome"] for c in passes[-1]["memory"]["record"]["changes"]
            if c["id"]}
    expected = {"gated": {7: "kept", 8: "kept", 6: "not kept"},
                "refusal": {7: "kept", 8: "kept"},
                "failed_pass": {7: "kept", 8: "kept", 6: "not kept"},
                "resume": {7: "kept", 8: "kept", 6: "not kept"},
                "ungated": {3: "kept", 4: "kept", 2: "kept"}}[name]
    assert last == expected
    if name == "refusal":
        assert any(c["outcome"] == "refused" for c in passes[-1]["memory"]["record"]["changes"])
    if name == "ungated":
        first = {levels[c["id"]]: c["outcome"] for c in passes[0]["memory"]["record"]["changes"]}
        assert first == {3: "kept", 4: "not kept"}, "4 not kept until 2 descends from it"
    if name == "failed_pass":
        assert [p["updated"] for p in passes] == [True, False, True]
        assert len(passes[1]["memory"]["record"]["changes"]) == 2, "facts written on failure"
    # The note the proposer reads is that record, rendered.
    shown = json.loads(loop._memory["text"])["changes"]
    assert [{k: v for k, v in c.items() if k != "observations"} for c in shown] == \
        passes[-1]["memory"]["record"]["changes"]


# ── the SkillOpt arm: its published meta skill, untouched ─────────────────────

#: sha256 of the SkillOpt strategy's meta-skill surfaces on main before the record
#: (e11cdc204b): its system prompt, its user message for fixed inputs (with and without
#: a previous meta skill), the context block every later call reads, and the step that
#: asks for it and reads ``meta_skill_content``.
SKILLOPT_MAIN = {
    "system": "9ddd93ac6d9360218b73513d222e0a987bbac8fe51d3abf32359e477723ffe08",
    "user": "f8bab8a068c69d33aaf2adf8d1bcda5fd919c83a2d8857c2b1e345db099a391b",
    "first user": "5de8294bdc8e3f4e554f152073477cda91e9d768cfcd66a703dfa586d845dd23",
    "context": "5fa2524bb12eab0e05943d5e348ad0ad505115d8d50bfa2af78d458685faa47d",
    "step": "3f33d5957a1de86924d7dad9493d49dd9fad81bd0d5a55dd19b5212ad510e277",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_the_skillopt_strategys_meta_skill_is_byte_identical_to_main():
    import inspect

    from refactory_packages.skillopt import env, prompts, proposer

    assert {
        "system": _sha(prompts.META_SKILL_SYSTEM),
        "user": _sha(prompts.format_meta_skill_user("PREV-SKILL", "CURR-SKILL", "PREV-META",
                                                    "COMPARISON")),
        "first user": _sha(prompts.format_meta_skill_user("PREV-SKILL", "CURR-SKILL", "",
                                                          "COMPARISON")),
        "context": _sha(prompts.format_meta_skill_context("META")),
        "step": _sha(inspect.getsource(proposer.SkillOptProposer.meta_skill_step)),
    } == SKILLOPT_MAIN
    assert env.GENERIC_PROMPTS["meta_skill"] is prompts.META_SKILL_SYSTEM
    # No module of the SkillOpt package reaches the loop's memory pass.
    package = Path(prompts.__file__).parent
    for path in package.glob("*.py"):
        source = path.read_text()
        assert "optim.memory" not in source and "optim import memory" not in source, path


def test_a_skillopt_run_makes_its_own_meta_skill_and_no_record_pass(tmp_path):
    """A real SkillOpt-strategy run (scripted client): its meta-skill call is SkillOpt's
    prompt and user message, its note reaches the next epoch's analysts in SkillOpt's
    context block, and the loop makes no memory pass of its own."""
    from refactory_packages.skillopt import prompts
    from test_skillopt import FakeClient, _run_loop

    client = FakeClient(responses={("meta_skill", 2, 0, 0): json.dumps(
        {"reasoning": "meta", "meta_skill_content": "USE CONCISE EDITS"})})
    strategy, _report = _run_loop(tmp_path, client=client, skillopt_options={"max_epochs": 3})
    systems = {c["system"] for c in client.calls}
    assert not systems & {memory.MEMORY_SYSTEM, memory.SYNTHESIS_SYSTEM, memory.COMPARE_SYSTEM}
    metas = [c for c in client.calls if (c.get("call_key") or ("",))[0] == "meta_skill"]
    assert metas and all(c["system"] == prompts.META_SKILL_SYSTEM for c in metas)
    assert all(c["prompt"].startswith("## Previous Epoch Last-Step Skill\n") for c in metas)
    later = [c["prompt"] for c in client.calls
             if (c.get("call_key") or ("",))[0].startswith("analyst_") and c["call_key"][1] == 3]
    assert later and all(prompts.format_meta_skill_context("USE CONCISE EDITS") in p
                         for p in later)
    from test_skillopt import _INFRA

    loop = _INFRA[id(strategy)]
    assert loop._memory_passes == []
    # Even with a proposer that could write one, a custom strategy makes no pass.
    loop.optimizer.write_memory = lambda **_: {"note": None}
    assert loop._memory_on() is False
    assert '"optimizer_memory"' not in (loop.root / "journal.jsonl").read_text()


# ── the replay hook: a recorded run's memory requests, re-rendered offline ───

#: A run's optimizer_trace.jsonl from main before the record (e11cdc204b), made by
#: `MemoryRequest` from the inputs below: one single call (epoch 3) and one split pass
#: (epoch 4: five comparison calls under a 9000-character room, then the synthesis).
OLD_RUN = Path(__file__).parent / "fixtures" / "memory_replay" / "run"
OLD_TRIED = [
    "## What was tried this epoch",
    "Every change the optimizer proposed in round 2 (header kept verbatim in the fixture).",
    "- `c2` (round 2, from `c1`; knob change: prompt edit): set_knob skill: 10 -> 20 chars "
    "-- train score 0.6 -- gate: kept",
    "- `c3` (round 2, from `c1`; structural change: insert): [insert after solve] -- train "
    "score 0.4 -- gate: rejected",
    "- refused before it ran (round 2, attempt 1; structural change: insert): [insert] -- "
    "E-tool-policy-escalation: node 'solve' ..."]


def _old_inputs(epoch: int, outcomes: dict[str, tuple[bool, bool]], chars: int) -> dict[str, Any]:
    def rec(item: str, passed: bool, text: str, tag: str) -> dict[str, Any]:
        return {"id": item, "passed": passed, "score": float(passed), "prediction": f"P-{item}",
                "fail_reason": "" if passed else f"WHY-{item}",
                "exchanges": [{"node": "solve", "prompt": f"{tag}-IN-{item}", "response": text}]}

    prev = [rec(i, b, "B" * chars if chars else f"BEFORE-{i}", "BEFORE")
            for i, (b, _n) in outcomes.items()]
    curr = [rec(i, n, "N" * chars if chars else f"NOW-{i}", "NOW")
            for i, (_b, n) in outcomes.items()]
    entries, _ = memory.pairs(prev, curr, list(outcomes))
    return dict(epoch=epoch, mode_name="m",
                prev={"id": "c1", "pipeline": "PIPE-START\nstep solve", "rollout": "R-PREV"},
                curr={"id": "c2", "pipeline": "PIPE-NOW\nstep solve", "rollout": "R-NOW"},
                changes=['- `c2` (from `c1`): [{"op": "set_knob"}]'], entries=entries,
                tried=OLD_TRIED, objective="OBJECTIVE-TEXT")


def _script() -> Any:
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "replay_memory.py"
    spec = importlib.util.spec_from_file_location("replay_memory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_replay_rebuilds_a_recorded_runs_requests_as_the_record_sends_them(tmp_path):
    """Every memory request an old run recorded, converted by the script, is byte for
    byte the request the current code makes from the same inputs: the single call, each
    comparison call and the synthesis (from the comparison replies the trace holds)."""
    from refactory_dsh.optim.random_optimizer import reply_object

    script = _script()
    manifest = script.replay(OLD_RUN, tmp_path / "out")
    calls = {(c["epoch"], c["name"]): c for c in manifest["calls"]}
    assert sorted(calls) == [(3, "note"), *((4, f"compare_part{n}") for n in range(1, 6)),
                             (4, "synthesis")]
    assert all("error" not in c and c["converted"] for c in calls.values())

    def written(key: tuple[int, str]) -> tuple[str, str]:
        return (Path(calls[key]["system"]).read_text(), Path(calls[key]["prompt"]).read_text())

    single = memory.MemoryRequest(**_old_inputs(3, {"a": (False, True), "b": (True, False),
                                                    "c": (False, False), "d": (True, True)}, 0))
    assert written((3, "note")) == (memory.MEMORY_SYSTEM, single()[0])
    split = memory.MemoryRequest(**_old_inputs(4, {
        "i0": (False, True), "r0": (True, False), "p0": (False, False), "s0": (True, True),
        "r1": (True, False), "i1": (False, True)}, 1500))
    by_id = {e["id"]: e for e in split.entries}
    trace = [json.loads(line) for line in (OLD_RUN / "optimizer_trace.jsonl").read_text()
             .splitlines()]
    compares = [t for t in trace if t["kind"] == "memory_compare"]
    for call in compares:
        part = [by_id[i] for i in call["items"]]
        assert written((4, f"compare_part{call['part']}")) == (
            memory.COMPARE_SYSTEM, split.compare_prompt(part, call["part"], call["parts"])[0])
    results = [{"part": c["part"], "items": c["items"],
                "observations": memory.parse_observations(reply_object(c["reply"]))}
               for c in sorted(compares, key=lambda c: c["part"])]
    assert written((4, "synthesis")) == (memory.SYNTHESIS_SYSTEM, split.synthesis_prompt(results)[0])
    # Replies of one's own to the comparison requests make the synthesis from them.
    replies = tmp_path / "replies" / "epoch_4"
    replies.mkdir(parents=True)
    for call in compares:
        (replies / f"part{call['part']}.json").write_text(json.dumps({"observations": [
            {"observation": f"NEW-{call['part']}", "items": call["items"]}]}))
    again = script.replay(OLD_RUN, tmp_path / "again", 4, tmp_path / "replies")
    synthesis = Path(next(c for c in again["calls"] if c["name"] == "synthesis")["prompt"]).read_text()
    assert synthesis == split.synthesis_prompt([
        {"part": c["part"], "items": c["items"], "observations": [
            {"category": "", "observation": f"NEW-{c['part']}", "items": c["items"],
             "evidence": []}]} for c in sorted(compares, key=lambda c: c["part"])])[0]
    assert "OBS-PART" not in synthesis


def test_the_replay_reads_a_reply_as_the_record_does(tmp_path, capsys):
    script = _script()
    reply = tmp_path / "reply.json"
    reply.write_text(json.dumps({"observations": [
        {"change": "c3", "items": ["b"], "observation": "In b the inserted step returned 7."},
        {"change": "c2", "items": ["a"], "observation": "Next time keep the edit smaller."}]}))
    assert script.main([str(OLD_RUN), "--out", str(tmp_path / "out"), "--epoch", "3",
                        "--reply", str(reply)]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert [(o["change"], o["text"]) for o in shown["kept"]] == [
        ("c3", "In b the inserted step returned 7."), ("c2", "Next time keep the edit smaller.")]
    assert shown["stripped"] == [] and shown["unpinned"] == []


class _Proposing:
    """A model client for a real loop: each proposal call sets the level to the next of
    ``levels``; each memory call answers, on its first compared item, one observation
    about the first change "What was tried this epoch" names, one piece of advice, and
    one about the first change it was ever shown (later, not one that pass compares)."""

    def __init__(self, levels: list[int]) -> None:
        self.levels, self.calls, self.first = list(levels), [], ""

    def complete(self, prompt: str, *, system: str = "", **_: Any) -> str:
        import re

        self.calls.append((system, prompt))
        if system == memory.MEMORY_SYSTEM:
            items = re.findall(r"^#### Task (\S+?)(?::| --|$)", prompt, re.M)
            changes = re.findall(r"^- `([^`]+)` \(round ", prompt, re.M)
            self.first = self.first or (changes[0] if changes else "")
            return json.dumps({"observations": [
                {"change": changes[0] if changes else "", "items": items[:1],
                 "observation": f"OBS for {items[0]}: step work returned TRAIN-OUT."},
                {"change": "", "items": items[:1], "observation": "Next round, try a judge."},
                {"change": self.first, "items": items[:1], "observation": "PINNED-FIRST"}]})
        if not self.levels:
            return json.dumps({"diagnosis": "d", "proposals": []})
        level = self.levels.pop(0)
        return json.dumps({"diagnosis": "d", "proposals": [
            {"edit": {"op": "set_knob", "name": "level", "value": level}}]})


def test_the_replay_rebuilds_the_record_from_the_journal_and_swaps_it_into_a_proposal(
        tmp_path, monkeypatch, capsys):
    """A real loop (the model proposer, a scripted model): 7 kept, 8 kept, 6 rejected.
    From its run directory alone, the record rebuilt from the journal and the memory
    replies is the record the loop stored, and without replies the facts are the same;
    epoch 4's proposal request is written as recorded, with the record, and without
    memory."""
    from refactory_dsh.optim.random_optimizer import AutoresearchOptimizer
    from refactory_dsh.outerloop import ValidationGate
    from test_analysts import _with_traces
    from test_gate_aware_context import _passes_by_level, _val
    from test_skillopt import _patch_mode
    from test_val_gate import LevelProbe, _loop

    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(_val))
    _passes_by_level(monkeypatch)
    _with_traces(monkeypatch)
    script = _script()
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, epochs=4, val_gate=True)
        loop.optimizer = AutoresearchOptimizer(_Proposing([7, 8, 6]))
        loop.run()
        run = loop.root
        trace = [json.loads(line) for line in (run / "optimizer_trace.jsonl").read_text()
                 .splitlines()]
        replies = {c["epoch"]: c["reply"] for c in trace if c["kind"] == "memory"}
        stored = [json.loads(line) for line in (run / "journal.jsonl").read_text().splitlines()]
        stored = [e for e in stored if e["event"] == "optimizer_memory"]
        assert [e["epoch"] for e in stored] == [2, 3, 4] and sorted(replies) == [2, 3, 4]
        # Round 1's change is compared in epoch 2 only (7 -> 8, then 8 against 6): later,
        # an observation pinned to it is kept unpinned, and the journal says so.
        first = loop.optimizer._client.first
        assert [[(u["change"], u["text"]) for u in e["unpinned"]] for e in stored] == [
            [], [(first, "PINNED-FIRST")], [(first, "PINNED-FIRST")]]
        assert all(e["stripped"] == [] for e in stored)
        for at in stored:
            rebuilt = script.record_from_journal(run, at["epoch"], replies, Path("."),
                                                 tmp_path / f"scratch{at['epoch']}")
            assert rebuilt == at["memory"]["record"], at["epoch"]
        last = stored[-1]["memory"]["record"]
        # Each pass's three observations, the advice one too: wording is not judged.
        assert len(last["changes"]) == 3 and len(last["observations"]) == 9
        assert [o["change"] for o in last["observations"] if o["text"] == "PINNED-FIRST"] == [
            first, None, None]
        assert {c["outcome"] for c in last["changes"]} == {"kept", "not kept"}
        bare = script.record_from_journal(run, 4, {}, Path("."), tmp_path / "bare")
        assert bare["changes"] == last["changes"] and bare["observations"] == []
        assert bare["keeps_failing"] == last["keeps_failing"]
        reply = tmp_path / "reply4.json"
        reply.write_text(replies[4])
        out = tmp_path / "out"
        assert script.main([str(run), "--out", str(out), "--epoch", "4", "--journal",
                            "--reply", f"2={_write(tmp_path, 'r2', replies[2])}",
                            "--reply", f"3={_write(tmp_path, 'r3', replies[3])}",
                            "--reply", str(reply), "--swap-proposal"]) == 0
    printed = capsys.readouterr().out
    folder = out / "epoch_4"
    assert json.loads((folder / "record.json").read_text()) == json.loads(
        memory.render_record(last))
    assert (folder / "record.json").read_text() == loop._memory["text"]
    # The proposal request three ways, the system prompt unchanged.
    (call,) = [c for c in trace if c["kind"] == "proposal" and c["epoch"] == 4][-1:]
    old, record, none = ((folder / f"proposal.{n}.prompt.txt").read_text()
                         for n in ("old", "record", "none"))
    assert old == call["prompt"]
    assert {(folder / f"proposal.{n}.system.txt").read_text()
            for n in ("old", "record", "none")} == {call["system"]}
    start, end = script.memory_section(old)
    assert old[start:].startswith("## Optimizer memory\n") and old[end:].startswith("## ")
    assert record == old[:start] + "\n".join(memory.request_lines(
        {**stored[-1]["memory"], "text": loop._memory["text"]})) + "\n\n" + old[end:]
    assert record == old, "the loop's own note is that record: the swap changes nothing"
    # With another record (the facts alone) only the section's note changes.
    bare_text = memory.render_record(bare)
    script.swap_proposal(run, 4, bare_text, tmp_path / "bare_out")
    swapped = (tmp_path / "bare_out" / "proposal.record.prompt.txt").read_text()
    s_start, s_end = script.memory_section(swapped)
    assert swapped[:s_start] == old[:start] and swapped[s_end:] == old[end:]
    assert "\n".join(evidence.exact_block(bare_text)) in swapped[s_start:s_end]
    assert "OBS for" not in swapped[s_start:s_end] and "OBS for" in old[start:end]
    assert "## Optimizer memory" not in none and none == old[:start] + old[end:]
    assert memory.RECORD_LEAD in record and memory.RECORD_LEAD not in none
    for name in ("old", "record", "none"):
        assert f"proposal.{name}: " in printed
    # A request whose memory section is not there exactly once is refused, loudly.
    import pytest

    for broken in (none, old.replace("## Optimizer memory\n", "## Optimizer memory\n\n"
                                     "## Optimizer memory\n", 1)):
        with pytest.raises(ValueError, match="sections, not one"):
            script.memory_section(broken)


def _write(folder: Path, name: str, text: str) -> Path:
    path = folder / f"{name}.json"
    path.write_text(text)
    return path


def test_the_memory_section_is_found_by_the_requests_own_headings():
    script = _script()
    note = "\n".join(evidence.exact_block("## Optimizer memory\n## Not a heading"))
    prompt = ("# Update request\n\n## History\nH\n\n## Optimizer memory\nLEAD\n"
              f"{note}\n\n## Answer\nA")
    start, end = script.memory_section(prompt)
    assert prompt[start:end] == f"## Optimizer memory\nLEAD\n{note}\n\n"
    import pytest

    with pytest.raises(ValueError, match="no section of the request follows"):
        script.memory_section(prompt[:prompt.index("## Answer")])


def test_without_the_gate_the_rebuilt_record_reads_the_train_leader_as_the_loop_did(
        tmp_path, monkeypatch):
    """Ungated, level 7 scores below the seed's 5: the pass compares the seed with that
    unkept change, and the change is not the train leader. The rebuilt record says so,
    as the loop's did."""
    from refactory_dsh.optim.random_optimizer import AutoresearchOptimizer
    from test_gate_aware_context import _passes_by_level
    from test_skillopt import _patch_mode
    from test_val_gate import LevelProbe, _loop

    _passes_by_level(monkeypatch)
    script = _script()
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, epochs=2, val_gate=False)
        loop.optimizer = AutoresearchOptimizer(_Proposing([7]))
        loop.run()
        (at,) = [json.loads(line) for line in (loop.root / "journal.jsonl").read_text()
                 .splitlines() if '"optimizer_memory"' in line]
        assert at["rejected"]["noun"] == memory.NOT_KEPT_NOUN
        rebuilt = script.record_from_journal(loop.root, 2, {}, Path("."), tmp_path / "s")
    (change,) = rebuilt["changes"]
    assert (change["outcome"], change["why"]) == (
        "not kept", "no validation gate: never the train leader")
    assert rebuilt["changes"] == at["memory"]["record"]["changes"]


def test_a_note_from_before_the_record_is_dropped_on_resume_and_said(tmp_path, monkeypatch):
    """A run resumed from a journal whose note is an older coach note: the next pass
    does not carry it (it is no record), and says so once, in the journal and the
    progress log, never silently."""
    from test_optimizer_memory import Remembering
    from test_skillopt import _patch_mode
    from test_val_gate import LevelProbe, _loop

    class Coach(Remembering):
        def write_memory(self, *, epoch, generation, build):  # type: ignore[no-untyped-def]
            return {"note": {"text": "OLD-COACH-NOTE: try a judge", "reasoning": "r"},
                    "error": None, "calls": 1}

    _run(tmp_path, monkeypatch, [[(5, 7)], []], proposer=Coach([[(5, 7)], []]))
    with _patch_mode(LevelProbe):
        resumed = _loop(tmp_path, epochs=3, val_gate=True, resume=True)
        resumed.optimizer = Remembering([[(7, 8)], []])
        resumed.run()
    events = [json.loads(line) for line in (resumed.root / "journal.jsonl").read_text()
              .splitlines()]
    (dropped,) = [e for e in events if e["event"] == "memory_note_dropped"]
    assert dropped["note_epoch"] == 2 and memory.RECORD_FORMAT in dropped["why"]
    assert "OLD-COACH-NOTE" not in resumed._memory["text"]
    assert resumed._memory["record"]["format"] == memory.RECORD_FORMAT
    progress = (resumed.root / "progress.jsonl").read_text()
    assert "written before the record's format" in progress
