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

from refactory_dsh.optim import memory
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
            "items": {"fixed": fixed, "broken": broken, "failing": failing, "shared": shared}}


FACTS = memory.record_facts(
    [_tried("c2", 1, "c1", gate="kept", fixed=["a"], broken=[], failing=["c"],
            shared=["a", "b", "c"]),
     _tried("c3", 2, "c2", gate="rejected", fixed=[], broken=["b"], failing=["b", "c"],
            shared=["a", "b", "c"])],
    [{"round": 2, "attempt": 1, "kind": "tools", "structural": True, "touched": ["solve"],
      "edits": "EDITS-REFUSED", "reasons": ["E-tool-policy-escalation: node 'solve' ..."]}],
    gated=True, current="c2")


def test_an_answer_with_advice_is_stripped_and_the_facts_stay_the_harnesss():
    """Stripped, not rejected: the clean observations are kept, the advice dropped and
    said why; a reply that writes facts of its own changes none of the harness's."""
    payload = {
        "changes": [{"id": "c3", "outcome": "kept", "fixed": ["b"]}],  # not the model's to say
        "observations": [
            {"change": "c3", "items": ["b"], "observation": "In b, the step solve read the "
             "second table and returned 7 where the grader expected 4.", "evidence": ["7"]},
            {"change": "c3", "items": ["b"], "observation": "Next round, try a judge after solve."},
            {"change": "", "items": ["c"], "observation":
             "The `skill` knob is the only unfrozen reachable text."},
            {"change": "c2", "items": ["c"], "observation": "Future proposals should keep it short."},
            {"change": "c2", "items": ["zz"], "observation": "Item zz was answered late."},
            {"change": "c9", "items": ["c"], "observation": "In c both pipelines stopped at step 2."},
        ]}
    note = memory.note_from(payload, epoch=3, facts=FACTS, items=["a", "b", "c"],
                            compared={"epoch": 3, "prev": "c2", "curr": "c3"})
    kept = note["record"]["observations"]
    assert [o["text"][:9] for o in kept] == ["In b, the", "In c both"]
    assert kept[0]["change"] == "c3" and kept[1]["change"] is None, "an unknown id is no change's"
    assert [s["why"] for s in note["stripped"]] == [
        "advice: 'Next round'", "advice: 'only unfrozen'", "advice: 'Future proposals'",
        "it names no item of the comparison"]
    view = json.loads(note["text"])
    assert view["format"] == memory.RECORD_FORMAT
    c3 = next(c for c in view["changes"] if c["id"] == "c3")
    assert (c3["outcome"], c3["fixed"], c3["broken"]) == ("not kept", [], ["b"])
    assert c3["observations"][0]["items"] == ["b"]
    assert memory.note_from({"memory": "a coach's note"}, facts=FACTS) is None, "no list, no note"
    assert json.loads(memory.note_from({"observations": []}, facts=FACTS)["text"])["changes"]


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


def test_the_records_facts_match_the_journal(tmp_path, monkeypatch):
    """Seed 5; 7 kept, 8 kept, 6 rejected (train item tN passes when N <= level - 2). The
    record each pass journals says, for every change, exactly what the journal's own
    train records and gate state say: the items it fixed and broke, and kept or not."""
    loop, _proposer = _run(tmp_path, monkeypatch, [[(5, 7)], [(7, 8)], [(8, 6)], []])
    journal = [json.loads(line) for line in (loop.root / "journal.jsonl").read_text().splitlines()]
    first_records: dict[str, list[dict[str, Any]]] = {}
    for event in journal:
        if event["event"] == "train_records":
            first_records.setdefault(event["id"], event["records"])
    parents = {c.id: c.parent for c in loop.population}
    levels = {c.id: _level(c) for c in loop.population}
    # Each pass with the gate's state as the journal had it when the pass was journaled.
    passes, gate = [], {}
    for event in journal:
        if event["event"] == "strategy_state" and "val_gate" in (event.get("state") or {}):
            gate = event["state"]["val_gate"]
        if event["event"] == "optimizer_memory":
            passes.append((event, dict(gate)))
    assert [p["epoch"] for p, _g in passes] == [2, 3, 4]
    for index, (done, gate) in enumerate(passes):
        changes = done["memory"]["record"]["changes"]
        assert len(changes) == index + 1, "every round so far, not only the last"
        for change in changes:
            moved = edit_change(first_records[parents[change["id"]]], first_records[change["id"]])
            assert (change["fixed"], change["broken"]) == (moved["fixed"], moved["broken"])
            assert change["of"] == moved["shared"] and change["parent"] == parents[change["id"]]
            assert change["id"] in gate["accepted"] + gate["rejected"]
            kept = change["id"] in gate["accepted"]
            assert change["outcome"] == ("kept" if kept else "not kept"), change
    passes = [p for p, _g in passes]
    final = {levels[c["id"]]: (c["fixed"], c["broken"], c["outcome"])
             for c in passes[-1]["memory"]["record"]["changes"]}
    assert final == {7: (["t4", "t5"], [], "kept"), 8: (["t6"], [], "kept"),
                     6: ([], ["t5", "t6"], "not kept")}
    # t7 and t8 failed under every change that scored them; t6 was fixed by 8.
    failing = passes[-1]["memory"]["record"]["keeps_failing"]
    assert [(f["item"], [levels[c] for c in f["failed_under"]]) for f in failing] == [
        ("t7", [7, 8, 6]), ("t8", [7, 8, 6])]
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
        ("c3", "In b the inserted step returned 7.")]
    assert shown["stripped"] == [{"text": "Next time keep the edit smaller.",
                                  "why": "advice: 'Next time'"}]
