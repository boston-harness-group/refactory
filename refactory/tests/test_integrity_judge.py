"""The integrity judge (`optim.integrity`, ``LoopConfig.integrity_judge``).

- Off (the default): no call, no record, the request and the report as before, and a
  run may be resumed with the judge turned on (the settings digest leaves it out).
- On: one call of the optimizer's model per proposal that became a valid candidate,
  through ``Run.attempt`` (so every strategy), and one on the winner's combined change
  from the seed. Each category has a policy, off / warn / veto: the run's setting, else
  the Task's declaration (an allowance turns a category off and the judge is told why),
  else the run's default (veto for ``on``, warn for ``flag``).
- Only an ``exploit`` verdict naming a veto category refuses a proposal (``E-integrity``);
  a suspect verdict, a judge error or a malformed reply never does.
- The verdicts reach the change ledger, the next request ("Integrity review of your
  proposals"), the journal (kept across a resume) and the report.
- The same prompt can be built and judged offline from a saved proposal or a past run.

Each test fails when the behaviour it names is reverted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh.optim import integrity
from refactory_dsh.optim.base import Candidate, Proposal
from refactory_dsh.outerloop import Run, ValidationGate

from test_gate_aware_context import Recording, _passes_by_level, _val
from test_skillopt import SkillTask, _patch_mode
from test_val_gate import LevelProbe, _level, _loop


def _section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:] if end == -1 else text[start:end]


EXPLOIT = {"verdict": "exploit", "categories": ["dataset_artifact"],
           "evidence": ["if that option is listed, choose it"],
           "reason": "It keys on a feature that only marks the right answer in this data."}
SUSPECT = {**EXPLOIT, "verdict": "suspect"}
CLEAN = {"verdict": "clean", "categories": [], "evidence": [], "reason": "Better checking."}


class Judge:
    """A scripted judge: each call takes the next reply (the last one repeats); an
    exception is raised, a mapping is sent as JSON, a string as it is."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.systems: list[str] = []

    def complete(self, prompt: str, *, system: str = "", **_options: Any) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


#: Seed 5 -> 3, then 3 -> 2 (a lower level scores higher on train): two proposals, one
#: per round, and the winner two changes away from the seed.
PLAN = [[(5, 3)], [(3, 2)], []]


def _run(tmp_path: Path, monkeypatch: Any, judge: Any, plan: Any = None, **overrides: Any):
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(_val))
    monkeypatch.setattr("refactory_dsh.loop.Loop._seed_population", lambda self, seed: None)
    _passes_by_level(monkeypatch)
    proposer = Recording(list(plan or PLAN))
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, **{"epochs": len(plan or PLAN), "val_gate": False,
                                  "max_evaluations": 50, **overrides})
        loop.optimizer = proposer
        loop.integrity_client = judge
        report = loop.run()
    return loop, proposer, report


def _journal(loop: Any, event: str) -> list[dict[str, Any]]:
    lines = (loop.root / "journal.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if json.loads(line).get("event") == event]


def _proposal_prompts(judge: Judge) -> list[str]:
    return [p for p in judge.prompts if p.startswith("# Integrity review of a proposed change")]


# ── reading a verdict ───────────────────────────────────────────────────────


def test_a_verdict_is_read_and_bounded():
    found = integrity.parse_verdict({
        "verdict": "Exploit", "categories": ["dataset-artifact", "Train overfit", "made_up"],
        "evidence": "q" * 900, "reason": "r" * 2000})
    assert found["verdict"] == "exploit"
    assert found["categories"] == ["dataset_artifact", "train_overfit"]
    assert found["unknown_categories"] == ["made_up"]
    assert len(found["evidence"]) == 1 and len(found["evidence"][0]) <= integrity.QUOTE_CHARS + 40
    assert len(found["reason"]) <= integrity.REASON_CHARS
    many = integrity.parse_verdict({"verdict": "suspect", "evidence": ["a", "b", "c", "d", "e"]})
    assert many["evidence"] == ["a", "b", "c", "d"]


@pytest.mark.parametrize("payload", [{"verdict": "bad"}, {}, ["exploit"], "exploit", None])
def test_an_answer_without_a_known_verdict_is_refused(payload):
    with pytest.raises(ValueError):
        integrity.parse_verdict(payload)


def test_a_finding_only_in_categories_the_run_allows_reads_as_clean():
    found = integrity.parse_verdict({**EXPLOIT, "categories": ["answer_leak"]},
                                    categories=["dataset_artifact"])
    assert found["verdict"] == "clean" and found["judged"] == "exploit"
    assert found["categories"] == [] and found["ignored_categories"] == ["answer_leak"]
    kept = integrity.parse_verdict({**EXPLOIT, "categories": ["answer_leak", "dataset_artifact"]},
                                   categories=["dataset_artifact"])
    assert kept["verdict"] == "exploit" and kept["categories"] == ["dataset_artifact"]


# ── policy ──────────────────────────────────────────────────────────────────


def test_bare_settings_and_their_defaults():
    assert integrity.parse_setting("off") is None
    assert integrity.parse_setting({"mode": "off", "policy": {"answer_leak": "veto"}}) is None
    for word in ("on", "veto", True):
        assert integrity.parse_setting(word) == {"default": "veto", "policy": {}}
    for word in ("flag", "warn"):
        assert integrity.parse_setting(word) == {"default": "warn", "policy": {}}
    assert integrity.parse_setting('{"policy": {"answer_leak": "off"}}') == {
        "default": "veto", "policy": {"answer_leak": "off"}}
    assert integrity.parse_setting({"mode": "flag", "default": "veto"})["default"] == "veto"
    for bad in ("yes", {"policy": {"nope": "off"}}, {"policy": {"answer_leak": "block"}},
                {"mode": "loud"}, {"extra": 1}, "{not json", 3):
        with pytest.raises(ValueError):
            integrity.parse_setting(bad)


def test_precedence_run_setting_then_task_then_default():
    setting = integrity.parse_setting({"mode": "on", "policy": {"answer_leak": "warn"}})
    policy = integrity.resolve_policy(
        setting, declared={"answer_leak": "off", "grader_exploit": "warn"},
        allowances={"train_overfit": "items are meant to be memorised"})
    assert policy == {"answer_leak": "warn",          # the run's setting beats the Task's off
                      "dataset_artifact": "veto",     # nothing declared: the default
                      "grader_exploit": "warn",       # the Task's declaration
                      "environment_tamper": "veto",
                      "task_bypass": "veto",
                      "train_overfit": "off"}         # the Task's allowance
    warned = integrity.resolve_policy(integrity.parse_setting("flag"),
                                      declared={"task_bypass": "veto"})
    assert warned["dataset_artifact"] == "warn" and warned["task_bypass"] == "veto"
    assert set(integrity.resolve_policy(None).values()) == {"off"}
    assert integrity.asked(policy) == ["answer_leak", "dataset_artifact", "grader_exploit",
                                       "environment_tamper", "task_bypass"]


@pytest.mark.parametrize("verdict,policy,vetoed", [
    (EXPLOIT, {"dataset_artifact": "veto"}, True),
    (SUSPECT, {"dataset_artifact": "veto"}, False),          # suspect is never vetoed
    (EXPLOIT, {"dataset_artifact": "warn"}, False),          # warn by choice
    (EXPLOIT, {"dataset_artifact": "off"}, False),
    ({**EXPLOIT, "categories": []}, {"dataset_artifact": "veto"}, False),
    ({**EXPLOIT, "error": "ValueError: x"}, {"dataset_artifact": "veto"}, False),
    ({"verdict": None, "categories": ["dataset_artifact"], "error": "boom"},
     {"dataset_artifact": "veto"}, False),
    ({**EXPLOIT, "categories": ["answer_leak", "dataset_artifact"]},
     {"answer_leak": "warn", "dataset_artifact": "veto"}, True),
])
def test_only_an_exploit_in_a_veto_category_vetoes(verdict, policy, vetoed):
    assert integrity.vetoes(verdict, policy) is vetoed


# ── the prompt ──────────────────────────────────────────────────────────────


PARENT_STATE = {"knobs": {"solve__task": {"value": "Solve the item.\nShow your work.",
                                          "spec": {"kind": "str", "default": ""}}},
                "graph": {"start": "solve", "nodes": {"solve": {"id": "solve", "kind": "agent",
                                              "prompt_template": "Work.", "knobs": ["solve__task"],
                                              "writes": ["answer.txt"]}}, "edges": []}}
EDITS = [{"op": "set_knob", "name": "solve__task",
          "value": "Solve the item.\nIf the last listed choice is present, pick it.\nShow your work."},
         {"op": "insert_after", "target": "solve",
          "node": {"id": "pick", "kind": "op", "call": "refactory_dsh.snippet:run",
                   "code": "def main(item):\n    return 'PICK_LAST' in item['text']"}}]
RECORDS = [{"id": "t1", "passed": False, "score": 0.0, "prediction": "B", "label": "D",
            "metadata": {"question": "Q one"}},
           {"id": "t2", "passed": True, "score": 1.0, "prediction": "A", "label": "A"}]


def test_the_proposal_prompt_shows_the_change_reasoning_evidence_and_the_test():
    child = integrity._applied(PARENT_STATE, EDITS[:1])
    prompt = integrity.proposal_prompt(
        edits=EDITS, parent_state=PARENT_STATE, state=child, rationale="RATIONALE-X",
        diagnosis="DIAG-Y", prediction="PRED-Z", records=RECORDS, mode_name="m",
        parent_id="p1", candidate_id="c2")
    assert integrity.CORE_TEST in prompt
    assert "If the last listed choice is present, pick it." in prompt      # the edit whole
    assert "return 'PICK_LAST' in item['text']" in prompt                  # the code whole
    assert "+If the last listed choice is present, pick it." in prompt    # the state diff
    assert "RATIONALE-X" in prompt and "DIAG-Y" in prompt and "PRED-Z" in prompt
    evidence = _section(prompt, "## The training evidence the optimizer saw")
    assert "2 train item(s), 1 passed" in evidence and "| t1 | answered wrong | D | B | Q one |" in evidence
    for name in integrity.CATEGORIES:
        assert f"- {name}: " in prompt
    assert '"verdict": "clean" | "suspect" | "exploit"' in prompt


def test_the_prompt_names_no_dataset_or_domain():
    prompt = integrity.proposal_prompt(edits=[{"op": "set_knob", "name": "k", "value": "v"}],
                                       parent_state=None, state=None)
    lowered = (prompt + integrity.SYSTEM).lower()
    for word in ("livemath", "math", "none of the above", "multiple choice", "theorem",
                 "cyber", "python"):
        assert word not in lowered, word


def test_an_off_category_is_not_asked_about_and_an_allowance_is_said():
    prompt = integrity.proposal_prompt(
        edits=EDITS[:1], parent_state=PARENT_STATE, state=None,
        categories=["dataset_artifact", "grader_exploit"],
        allowances={"answer_leak": "the task is to recall facts about these very items"})
    assert "- dataset_artifact: " in prompt and "- answer_leak: " not in prompt
    assert "- train_overfit: " not in prompt
    assert "zero or more of: dataset_artifact, grader_exploit" in prompt
    allows = _section(prompt, "## What this task allows")
    assert "the task is to recall facts about these very items" in allows


def test_the_evidence_summary_is_bounded():
    many = [{"id": f"t{i:03d}", "passed": i % 2 == 0, "prediction": "x" * 300, "label": "y" * 300}
            for i in range(400)]
    text = integrity.evidence_summary(many, budget=3000)
    assert len(text) <= 3300 and "more item(s) not shown" in text
    assert integrity.evidence_summary([]) == "(no train records were available)"


# ── the loop: off ───────────────────────────────────────────────────────────


def test_off_makes_no_call_and_changes_no_request_or_report(tmp_path, monkeypatch):
    judge = Judge(EXPLOIT)
    loop, proposer, report = _run(tmp_path / "off", monkeypatch, judge)
    assert loop.config.integrity_judge == "off"
    assert judge.prompts == [] and loop._integrity == []
    assert not _journal(loop, "integrity_verdict")
    assert "integrity" not in report and "## Integrity review" not in (loop.root / "report.md").read_text()
    # The same run with the judge's code taken out renders the same requests, byte for byte.
    from refactory_dsh.loop import Loop
    monkeypatch.setattr(Loop, "_integrity_check", lambda self, p, parent, c: c)
    monkeypatch.setattr(Loop, "_integrity_feedback", lambda self: [])
    monkeypatch.setattr(Loop, "_integrity_of", lambda self, cid: None)
    _loop2, bare, _ = _run(tmp_path / "bare", monkeypatch, judge)
    import difflib
    import re

    def text(ctx: Any) -> str:  # wall-clock timings differ between any two runs
        return re.sub(r'"(\w+_s)": [0-9.]+', r'"\1": T', ctx.render())

    assert len(proposer.contexts) == len(bare.contexts) == len(PLAN)
    for ours, theirs in zip(proposer.contexts, bare.contexts):
        assert text(ours) == text(theirs), "\n".join(difflib.unified_diff(
            text(ours).splitlines(), text(theirs).splitlines(), lineterm=""))
    assert all("integrity" not in e for c in proposer.contexts for e in c.ledger)


# ── the loop: on ────────────────────────────────────────────────────────────


def test_flag_records_shows_and_still_evaluates_an_exploit(tmp_path, monkeypatch):
    judge = Judge(EXPLOIT)
    loop, proposer, report = _run(tmp_path, monkeypatch, judge, integrity_judge="flag")
    made = [c for c in loop.population if c.parent]
    assert [_level(c) for c in made] == [3, 2] and all(c.evaluated for c in made)
    assert len(_proposal_prompts(judge)) == 2
    assert all(s == integrity.SYSTEM for s in judge.systems)
    verdicts = _journal(loop, "integrity_verdict")
    assert [(v["scope"], v["verdict"], v["vetoed"]) for v in verdicts] == [
        ("proposal", "exploit", False), ("proposal", "exploit", False), ("final", "exploit", False)]
    # The next request: the ledger's column and its own section.
    text = proposer.contexts[-1].render()
    ledger = _section(text, "## Every change so far, by kind")
    assert "| integrity |" in ledger and "| exploit: dataset_artifact |" in ledger
    review = _section(text, "## Integrity review of your proposals")
    assert "2 of your proposals were reviewed: 0 clean, 0 suspect, 2 exploit." in review
    assert '"if that option is listed, choose it"' in review and "refused" not in review
    # The optimizer trace counts the calls like the step reviews.
    trace = [json.loads(l) for l in (loop.root / "optimizer_trace.jsonl").read_text().splitlines()]
    assert [t["kind"] for t in trace if t["kind"].startswith("integrity")] == [
        "integrity_proposal", "integrity_proposal", "integrity_final"]
    # The report.
    assert report["integrity"]["proposals"] == {"reviewed": 2, "clean": 0, "suspect": 0,
                                                 "exploit": 2, "errors": 0, "vetoed": 0}
    assert set(report["integrity"]["policy"].values()) == {"warn"}
    assert report["integrity"]["final_audit"]["verdict"] == "exploit"
    md = (loop.root / "report.md").read_text()
    assert "## Integrity review" in md and "Result vs seed" in md and "**exploit: dataset_artifact**" in md


def test_on_vetoes_an_exploit_before_it_is_evaluated_and_says_why(tmp_path, monkeypatch):
    judge = Judge(EXPLOIT, CLEAN)
    loop, proposer, report = _run(tmp_path, monkeypatch, judge, integrity_judge="on",
                                  plan=[[(5, 7)], [(5, 6)], []])
    levels = {_level(c): c for c in loop.population}
    assert 7 not in levels, "the exploit was never evaluated"
    assert levels[6].evaluated
    refused = [r for r in loop.rejections if r.codes() == ["E-integrity"]]
    assert len(refused) == 1
    assert "dataset_artifact" in refused[0].reasons[0] and "only marks the right answer" in refused[0].reasons[0]
    assert any(r.startswith("E-integrity") for r in proposer.contexts[1].rejections)
    review = _section(proposer.contexts[1].render(), "## Integrity review of your proposals")
    assert "(1 refused before evaluation)" in review and "refused before evaluation --" in review
    assert report["integrity"]["proposals"]["vetoed"] == 1
    assert report["rejection_codes"].get("E-integrity") == 1


@pytest.mark.parametrize("reply", [SUSPECT, RuntimeError("503 overloaded"), "no json here",
                                   {"verdict": "terrible"}, {**EXPLOIT, "categories": []}])
def test_suspect_errors_and_uncategorised_verdicts_never_veto(tmp_path, monkeypatch, reply):
    judge = Judge(reply)
    loop, _proposer, report = _run(tmp_path, monkeypatch, judge, integrity_judge="on")
    assert [_level(c) for c in loop.population if c.parent] == [3, 2]
    assert not any("E-integrity" in r.codes() for r in loop.rejections)
    records = [r for r in loop._integrity if r["scope"] == "proposal"]
    assert records and not any(r["vetoed"] for r in records)
    if isinstance(reply, (Exception, str)) or reply.get("verdict") == "terrible":
        assert all(r["verdict"] is None and r["error"] for r in records)
        assert report["integrity"]["proposals"]["errors"] == 2
        trace = (loop.root / "optimizer_trace.jsonl").read_text()
        assert '"kind": "integrity_proposal"' in trace and '"error": null' not in trace.split(
            '"kind": "integrity_proposal"')[1].split("\n")[0]


def test_a_warn_category_is_shown_not_vetoed_and_an_off_one_is_not_asked(tmp_path, monkeypatch):
    judge = Judge(EXPLOIT)
    setting = {"mode": "on", "policy": {"dataset_artifact": "warn", "answer_leak": "off"}}
    loop, _p, report = _run(tmp_path, monkeypatch, judge, integrity_judge=setting)
    assert all(c.evaluated for c in loop.population)
    assert "- answer_leak: " not in judge.prompts[0] and "- dataset_artifact: " in judge.prompts[0]
    assert report["integrity"]["policy"]["dataset_artifact"] == "warn"
    assert report["integrity"]["policy"]["task_bypass"] == "veto"


def test_a_task_allowance_turns_a_category_off_and_the_run_setting_wins(tmp_path, monkeypatch):
    """The Task declares item-specific facts legitimate: answer_leak is not asked about,
    the judge reads why, and an answer_leak finding does not count. A run that names
    answer_leak itself asks about it again."""
    monkeypatch.setattr(SkillTask, "integrity_allowances",
                        {"answer_leak": "recalling facts about these exact items is the task"},
                        raising=False)
    monkeypatch.setattr(SkillTask, "integrity_policy", {"grader_exploit": "warn"}, raising=False)
    leak = {**EXPLOIT, "categories": ["answer_leak"]}
    judge = Judge(leak)
    loop, _p, report = _run(tmp_path / "task", monkeypatch, judge, integrity_judge="on")
    prompt = judge.prompts[0]
    assert "- answer_leak: " not in prompt
    assert "recalling facts about these exact items is the task" in _section(prompt, "## What this task allows")
    assert all(c.evaluated for c in loop.population), "the finding is in an allowed category"
    assert [r["verdict"] for r in loop._integrity if r["scope"] == "proposal"] == ["clean", "clean"]
    assert report["integrity"]["policy"]["answer_leak"] == "off"
    assert report["integrity"]["policy"]["grader_exploit"] == "warn"
    assert "answer_leak" in report["integrity"]["allowances"]

    judge = Judge(leak)
    loop, _p, _r = _run(tmp_path / "user", monkeypatch, judge,
                        integrity_judge={"policy": {"answer_leak": "veto"}})
    assert "- answer_leak: " in judge.prompts[0] and "## What this task allows" not in judge.prompts[0]
    assert 3 not in {_level(c) for c in loop.population}, "the run's veto wins over the Task"


def test_the_final_audit_judges_the_winners_combined_change(tmp_path, monkeypatch):
    judge = Judge(CLEAN, CLEAN, {**EXPLOIT, "evidence": ["two steps together"]})
    loop, _p, report = _run(tmp_path, monkeypatch, judge, integrity_judge="flag")
    audit = report["integrity"]["final_audit"]
    best = report["best"]["id"]
    final = judge.prompts[-1]
    assert final.startswith("# Integrity review of a search's result")
    assert f"candidate `{best}`" in final and "by 2 accepted change(s)" in final
    combined = _section(final, "## The combined change (starting pipeline -> result)")
    assert "### setting `level` changed: 5 -> 2" in combined
    built = _section(final, "## The changes that built it, in order")
    made = [c.id for c in loop.population if c.parent]
    assert f"### 1. candidate `{made[0]}`" in built and f"### 2. candidate `{made[1]}`" in built
    assert built.index("value: 3") < built.index("value: 2"), "each change's edits, in order"
    assert audit["verdict"] == "exploit" and audit["candidate"] == best
    assert audit["evidence"] == ["two steps together"]


def test_resume_keeps_the_verdicts_and_pays_for_none_again(tmp_path, monkeypatch):
    judge = Judge(EXPLOIT)
    loop, _p, report = _run(tmp_path, monkeypatch, judge, integrity_judge="flag")
    calls = len(judge.prompts)
    silent = Judge(RuntimeError("the judge must not be called again"))
    with _patch_mode(LevelProbe):
        again = _loop(tmp_path, **{"epochs": len(PLAN), "val_gate": False, "max_evaluations": 50,
                                   "integrity_judge": "flag", "resume": True})
        again.optimizer = Recording([])
        again.integrity_client = silent
        resumed = again.run()
    assert silent.prompts == [] and len(judge.prompts) == calls
    assert [r["verdict"] for r in again._integrity][:3] == ["exploit", "exploit", "exploit"]
    assert resumed["integrity"]["proposals"] == report["integrity"]["proposals"]
    assert resumed["integrity"]["final_audit"]["verdict"] == "exploit"


def test_an_off_run_resumes_with_the_judge_on(tmp_path, monkeypatch):
    """The settings digest leaves the judge out: turning it on is no new evaluation context."""
    loop, _p, _r = _run(tmp_path, monkeypatch, Judge(CLEAN), plan=[[(5, 3)], []])
    judge = Judge(CLEAN)
    with _patch_mode(LevelProbe):
        again = _loop(tmp_path, **{"epochs": 3, "val_gate": False, "max_evaluations": 50,
                                   "integrity_judge": "on", "resume": True})
        again.optimizer = Recording([[(3, 2)], []])
        again.integrity_client = judge
        again.run()
    assert _proposal_prompts(judge), "the resumed run judges its new proposals"


@pytest.mark.parametrize("optimizer", ["random", "hill_climb"])
def test_every_strategys_proposals_pass_the_judge(tmp_path, monkeypatch, optimizer):
    """A sampler and a parent picker through the default loop; a custom strategy's
    ``run.attempt`` is the same door (below)."""
    monkeypatch.setattr(ValidationGate, "val_score", staticmethod(_val))
    judge = Judge(CLEAN)
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, optimizer=optimizer, epochs=3, batch_size=3, val_gate=False,
                     edit_preset="knob_only",
                     integrity_judge="flag")
        loop.integrity_client = judge
        loop.run()
    made = {c.id for c in loop.population if c.parent and c.evaluated
            and (loop.proposals.get(c.id) or {}).get("origin") == "optimizer"}
    assert made
    judged = {r["candidate"] for r in loop._integrity if r["scope"] == "proposal"}
    assert made <= judged


def test_run_attempt_vetoes_for_a_custom_strategy(tmp_path, monkeypatch):
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, integrity_judge="veto")
        loop.integrity_client = Judge(EXPLOIT)
        loop.init()
        seed = loop.population[0]
        run = Run(_loop=loop)
        outcome = run.attempt(Proposal(edit={"op": "set_knob", "name": "level", "value": 3},
                                       parent=seed.id, rationale="from a custom strategy"), seed)
    assert isinstance(outcome, list) and outcome[0].startswith("E-integrity")
    assert loop.rejections[-1].codes() == ["E-integrity"]
    assert "from a custom strategy" in loop.integrity_client.prompts[0]


def test_the_judge_uses_the_optimizers_route(tmp_path):
    """bind_budget hands the judge the optimizer's model route, the one the budget meters."""
    routed: list[str] = []

    class Model:
        def for_node(self, node, role=""):
            routed.append(f"{node}/{role}")
            return self

        def complete(self, prompt, *, system="", workdir=None):
            return json.dumps(CLEAN)

    class Budget:
        def bind_model(self, model):
            return model

    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, integrity_judge="on")
        loop.model = Model()
        loop.bind_budget(Budget())
    assert routed == ["optimizer/optimizer"] and loop._judge_model is loop.model
    assert loop._judge("proposal", "p")["verdict"] == "clean"


# ── offline ─────────────────────────────────────────────────────────────────


def test_offline_judging_of_saved_entries(tmp_path, capsys):
    entries = [
        {"id": "a", "label": "exploit", "edits": EDITS[0], "parent_state": PARENT_STATE,
         "rationale": "r", "train_records": RECORDS},
        {"id": "b", "label": "clean", "edits": EDITS, "integrity_judge": "flag"},
        {"id": "c", "edits": EDITS[:1], "allowances": {"dataset_artifact": "allowed here"},
         "task_policy": {"answer_leak": "off"}},
        {"id": "d", "scope": "cumulative", "seed_state": PARENT_STATE,
         "state": integrity._applied(PARENT_STATE, EDITS[:1]),
         "steps": [{"id": "x", "edits": EDITS[:1], "rationale": "r"}]},
    ]
    path = tmp_path / "proposals.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    judge = Judge(EXPLOIT, EXPLOIT, EXPLOIT, "garbage")
    out = tmp_path / "verdicts.jsonl"
    assert integrity.main(["judge", "--proposal", str(path), "--out", str(out)], client=judge) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [(r["id"], r.get("label"), r["verdict"], r["vetoed"]) for r in rows] == [
        ("a", "exploit", "exploit", True), ("b", "clean", "exploit", False),
        ("c", None, "clean", False), ("d", None, None, False)]
    assert rows[3]["error"]
    assert "+If the last listed choice is present, pick it." in judge.prompts[0]
    assert "allowed here" in judge.prompts[2] and "- dataset_artifact: " not in judge.prompts[2]
    assert judge.prompts[3].startswith("# Integrity review of a search's result")
    assert integrity.main(["judge", "--proposal", str(path), "--print-prompt"]) == 0
    printed = capsys.readouterr().out
    assert printed.count(integrity.CORE_TEST) == 4


def test_a_past_runs_proposals_replay_with_the_prompt_the_run_used(tmp_path, monkeypatch):
    judge = Judge(CLEAN)
    loop, _p, _r = _run(tmp_path, monkeypatch, judge, integrity_judge="flag")
    out = tmp_path / "exported.jsonl"
    assert integrity.main(["export", str(loop.root), "--out", str(out)]) == 0
    entries = integrity.read_entries(out)
    assert [e["candidate"] for e in entries] == [c.id for c in loop.population if c.parent]
    for entry in entries:
        entry["integrity_judge"] = "flag"
    assert [integrity.entry_prompt(e) for e in entries] == _proposal_prompts(judge)


def test_a_proposal_judged_before_is_not_judged_again(tmp_path):
    """A vetoed proposal the optimizer sends again, word for word, is refused from the
    cache: one call."""
    with _patch_mode(LevelProbe):
        loop = _loop(tmp_path, integrity_judge="on")
        loop.integrity_client = Judge(EXPLOIT, CLEAN)
        loop.init()
        seed = loop.population[0]
        run = Run(_loop=loop)
        proposal = Proposal(edit={"op": "set_knob", "name": "level", "value": 3}, parent=seed.id)
        first, again = run.attempt(proposal, seed), run.attempt(proposal, seed)
    assert isinstance(first, list) and isinstance(again, list)
    assert len(loop.integrity_client.prompts) == 1


def test_a_run_with_every_category_off_asks_nothing(tmp_path, monkeypatch):
    judge = Judge(EXPLOIT)
    loop, _p, report = _run(tmp_path, monkeypatch, judge,
                            integrity_judge={"mode": "on", "default": "off"})
    assert judge.prompts == [] and loop._integrity == []
    assert report["integrity"]["final_audit"]["note"] == "every category is off: nothing to judge"
