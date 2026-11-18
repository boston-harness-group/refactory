"""An item a harness fault kept from an outcome is the harness's failure, never a wrong
answer; a session still writing at the backstop is the candidate's own.

A grader or sandbox that fails (`model.HarnessIncident`) leaves its item with no outcome
of the candidate's (`items.harness_incident`). The loop runs that item again, alone, on
the same candidate, part and seed, up to `INCIDENT_RERUNS` times
(`Loop._settle_incidents`); a rerun that answers is scored as any item. One still an
incident after them is left out of the score -- numerator and denominator -- and said in
the report, the progress line and the optimizer's request. A comparison of candidates
uses the items all were scored on (`Loop.shared_scores`). A session the backstop stopped
while its model still wrote (`model.HarnessBackstop`, ``backstop``) is scored 0, not
rerun, and shown to the optimizer as a failure.

The runs are the real loop and walker over a scripted ``dsh`` reached through the real
model client, whose backstop is 2 s here (14400 s in production); the grader fails when
an answer is ``GRADER_DOWN``. Each test fails when the behaviour it names is reverted.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import nn
from refactory_dsh.events import render
from refactory_dsh.items import harness_incident
from refactory_dsh.loop import INCIDENT_RERUNS, Loop
from refactory_dsh.model import DshModelClient
from refactory_dsh.optim.base import _item_evidence

from test_interrupted_evaluation import _loop as _interrupted_loop, quick_retries  # noqa: F401
from test_skillopt import TRAIN_IDS, VAL_IDS, _patch_mode
from test_val_gate import _journal_events, _level

#: A scripted ``dsh``: the first plan key whose ``&``-separated parts are all in the call's
#: folder or prompt decides: ``ACTIVE`` streams (touches the session's activity file) for
#: 30 s, past the 2 s backstop; anything else is the answer (``GRADER_DOWN``: the grader
#: fails, a harness incident).
FAKE_DSH = '''import json, os, pathlib, sys, time
root = pathlib.Path({root!r})
plan = json.loads((root / "plan.json").read_text())
seen = os.getcwd() + "\\n" + sys.stdin.read()
step = next(value for key, value in plan.items() if all(part in seen for part in key.split("&")))
if step == "ACTIVE":
    activity = pathlib.Path(os.environ["REFACTORY_WORKER_RECEIPT"]).with_suffix(".activity")
    end = time.time() + 30
    while time.time() < end:
        activity.touch()
        time.sleep(0.2)
print(step)
'''


class Dsh:
    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        script = root / "fake_dsh.py"
        script.write_text(FAKE_DSH.format(root=str(root)))
        self.cli = root / "dsh"
        self.cli.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{script}" "$@"\n')
        self.cli.chmod(0o755)

    def plan(self, plan: dict[str, str]) -> None:
        (self.root / "plan.json").write_text(json.dumps(plan))

    def client(self) -> DshModelClient:
        return DshModelClient(worker_backend="dsh", provider="p", model="m", dsh_cli=str(self.cli),
                              attempts=1, timeout_s=60, inactivity_s=5, backstop_s=2)


class Probe(nn.Mode):
    """Per item: one agent step at the candidate's ``level`` (no declared wall, so its
    session ends on inactivity or at the backstop), then an op that keeps its answer."""

    name = "incident_probe"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {"level": nn.Knob(default=5, kind="int", lo=0, hi=9, description="a level")}

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node("data", "data", subgraph_entry="solve", subgraph_exit="keep",
                          task_ref="inactivity_ops:SplitAnswerTask", writes=("current_item.json",),
                          produces=("item_scores",), parallelism=4))
        graph.add(nn.Node("solve", "agent", prompt_template="Answer at level {{level}}.",
                          writes=("answer.txt",), tool_policy="none", role="solver",
                          knobs=("level",)))
        graph.add(nn.Node("keep", "op", call="loop_ops:keep_answer", reads=("answer.txt",),
                          writes=("final.txt",)))
        graph.connect("solve", "keep")
        graph.add(nn.Node("finish", "op", call="loop_ops:write_best",
                          writes=("best.json",), produces=("best",)))
        graph.connect("data", "finish")
        return graph

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0


#: The loop's handling sits above the evaluator, on records both runtimes make alike
#: (the spine's incident record: `test_inactivity_only`); a full outer loop on the spine
#: needs a DSH profile, so these run the walker.
RUNTIMES = ["walker"]


def _run(tmp_path: Path, dsh: Dsh, plan: list[list[tuple[int, int]]], evaluator: str) -> Loop:
    with _patch_mode(Probe):
        loop = _interrupted_loop(tmp_path, dsh, plan, evaluator=evaluator)  # type: ignore[arg-type]
        loop.run()
    return loop


def _seed(loop: Loop):  # type: ignore[no-untyped-def]
    return next(c for c in loop.population if _level(c) == 5)


def _incidents(loop: Loop, part: str) -> list[dict[str, Any]]:
    return [e for e in _journal_events(loop, "harness_incident") if e["part"] == part]


@pytest.mark.parametrize("evaluator", RUNTIMES)
def test_an_incident_a_rerun_answers_is_scored_as_any_item(tmp_path, evaluator):
    """t2's grader fails in the seed's train run; run again alone, it answers right: it
    is scored like the rest (the train score is 1.0 over all 8), and the journal says it
    was recovered."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan({"/t2_&level 5.&incident_reruns": "right", "/t2_&level 5.": "GRADER_DOWN",
              "level": "right"})
    loop = _run(tmp_path, dsh, [[]], evaluator)
    seed = _seed(loop)
    assert seed.raw_score == pytest.approx(1.0)
    entry = _incidents(loop, "train")[0]
    assert entry["id"] == seed.id and entry["workdir"] == f"runs/{seed.id}"
    assert entry["items"] == ["t2"] and entry["recovered"] == ["t2"] and entry["not_scored"] == []
    assert entry["reran"] == 1
    records = loop._train_item_records[seed.id]
    assert len(records) == len(TRAIN_IDS) and not any(map(harness_incident, records))
    [t2] = [r for r in records if r["id"] == "t2"]
    assert t2["passed"] is True and t2["incident_reruns"] == 1
    # The one-item rerun is a rollout of its own (train adds 8, val 4, test 3).
    counts = [0] + [e["rollouts"] for e in _journal_events(loop, "rollouts")]
    assert 1 in {b - a for a, b in zip(counts, counts[1:])}


@pytest.mark.parametrize("evaluator", RUNTIMES)
def test_an_incident_every_rerun_repeats_is_left_out_of_the_score_and_said(tmp_path, evaluator):
    """t2's grader fails every time: after `INCIDENT_RERUNS` reruns it is left out of
    the score (t1 wrong, six right: 6/7, not 6/8), kept as a record with no score, and
    listed in report.json, report.md, the progress line and the optimizer's request."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan({"/t2_&level 5.": "GRADER_DOWN", "/t1_&level 5.": "wrong", "level": "right"})
    loop = _run(tmp_path, dsh, [[]], evaluator)
    seed = _seed(loop)
    assert seed.exit_ok and seed.raw_score == pytest.approx(6 / 7)
    entry = _incidents(loop, "train")[0]
    assert entry["id"] == seed.id and entry["workdir"] == f"runs/{seed.id}"
    assert entry["not_scored"] == ["t2"] and entry["reran"] == INCIDENT_RERUNS
    reruns = sorted(p.name for p in (loop.root / "runs" / seed.id / "incident_reruns").iterdir())
    assert reruns == [str(n) for n in range(1, INCIDENT_RERUNS + 1)]
    records = loop._train_item_records[seed.id]
    [t2] = [r for r in records if r["id"] == "t2"]
    assert t2["score"] is None and t2["not_scored"] == "harness_incident"
    assert [e["id"] for e in seed.facts["item_scores"]].count("t2") == 0

    report = json.loads((loop.root / "report.json").read_text())
    assert any(e["candidate"] == seed.id and e["not_scored"] == ["t2"]
               for e in report["harness_incidents"])
    assert "had a harness incident and are not scored" in (loop.root / "report.md").read_text()
    progress = [json.loads(line) for line in
                (loop.root / "progress.jsonl").read_text().splitlines()]
    shown = next(r for r in progress if r["step"] == "harness_incident"
                 and r.get("outcome") == "not_scored" and r["part"] == "train")
    assert "NOT SCORED: t2" in render(shown)

    lines = "\n".join(_item_evidence(records))
    assert lines.count("t2") == 1
    assert "1 item(s) (t2) had a harness incident and are not scored" in lines
    assert "7 train item(s)" in lines and "mean score 0.857" in lines


@pytest.mark.parametrize("evaluator", RUNTIMES)
def test_a_val_comparison_leaves_out_an_item_either_candidate_could_not_be_scored_on(
        tmp_path, evaluator):
    """The seed's val v2 is a harness incident every time (left out: 2 of 3); the child
    answers v2 right (3 of 4), but is no better on v1, v3, v4. Compared on those, the
    child ties and is rejected; on its raw val score it would have been accepted."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan({"/v2_&level 5.": "GRADER_DOWN", "/v1_&level": "wrong", "level": "right"})
    loop = _run(tmp_path, dsh, [[(5, 6)], []], evaluator)
    seed = _seed(loop)
    child = next(c for c in loop.population if _level(c) == 6)
    gate = loop._gate()
    assert gate.cache[seed.id] == pytest.approx(2 / 3)
    assert gate.cache[child.id] == pytest.approx(3 / 4)
    assert child.id not in loop._gate_accepted, "the child won only on the incident item"
    assert gate.current_id == seed.id
    scores, left_out = loop.shared_scores([child.id, seed.id])
    assert scores == {child.id: pytest.approx(2 / 3), seed.id: pytest.approx(2 / 3)}
    assert left_out == ["v2"]
    [entry] = [e for e in _incidents(loop, "val") if e["id"] == seed.id]
    assert entry["not_scored"] == ["v2"] and len(VAL_IDS) == 4


def test_edit_effect_reads_no_harness_incident_as_a_broken_item():
    """A child's item the harness failed is not one the edit broke: `edit_effect`
    leaves it out, as `edit_change` does, while a real broken item is still read."""
    from refactory_dsh.optim.edit_effect import edit_effect

    parent = [{"id": "a", "passed": True, "score": 1.0}, {"id": "b", "passed": True, "score": 1.0}]
    ran = [{"node": "check", "prompt": "", "response": "x"}]
    child = [{"id": "a", "passed": False, "score": None, "error_kind": "harness_incident",
              "error_text": "data: SandboxFailed: harness incident: the grader sandbox did not start",
              "not_scored": "harness_incident", "exchanges": ran},
             {"id": "b", "passed": False, "score": 0.0, "exchanges": ran}]
    effect = edit_effect(parent, child, new_steps={"check"})
    assert effect["broken"]["items"] == 1 and effect["broken"]["new"] == {"check": 1}


@pytest.mark.parametrize("evaluator", RUNTIMES)
def test_the_test_deltas_are_on_the_items_both_were_scored_on(tmp_path, evaluator):
    """The child wins on val (v1 right) and becomes the best. On test the seed's x2 is an
    incident every time (left out: 2 of 2 right), the child misses x1 (2 of 3). The best's
    test delta is taken on x1 and x3 (0.5 - 1.0), with the raw delta (2/3 - 1) beside it."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan({"/x2_&level 5.": "GRADER_DOWN", "/v1_&level 5.": "wrong", "/x1_&level 6.": "wrong",
              "level": "right"})
    loop = _run(tmp_path, dsh, [[(5, 6)], []], evaluator)
    child = next(c for c in loop.population if _level(c) == 6)
    assert loop._gate().best_id == child.id
    result = loop.final_evaluation()
    assert result["baseline_test"] == pytest.approx(1.0)
    assert result["best_test"] == pytest.approx(2 / 3)
    assert result["best_test_delta"] == pytest.approx(-0.5)
    assert result["best_test_delta_raw"] == pytest.approx(-1 / 3)
    assert result["best_test_delta_left_out"] == ["x2"]


# ── the comparisons, on the scores a loop keeps ────────────────────────────


class _Kept:
    """The parts of `Loop` its comparisons read: each candidate's val/test rollouts."""

    def __init__(self, rollouts: dict[str, dict[str, Any]], part: str = "val") -> None:
        import threading

        self._part_results = {cid: {part: [o]} for cid, o in rollouts.items()}
        self._state_lock = threading.RLock()
        self.warnings: list[str] = []

    def _journal(self, event: str, **fields: Any) -> None:
        pass

    def _progress(self, step: str, **fields: Any) -> None:
        self.warnings.append(fields.get("warning", ""))

    shared_scores = Loop.shared_scores
    part_incidents = Loop.part_incidents


def _rollout(scores: dict[str, float], *, left_out: tuple[str, ...] = (), repeat: int = 1,
             more: list[list[Any]] | None = None, score: float | None = None) -> dict[str, Any]:
    items = [[i, v, repeat] for i, v in scores.items()] + list(more or [])
    if score is None:
        by: dict[int, list[float]] = {}
        for entry in items:
            by.setdefault(entry[2], []).append(entry[1])
        score = sum(sum(v) / len(v) for v in by.values()) / len(by)
    return {"score": score, "item_scores": items,
            **({"harness_incidents": list(left_out)} if left_out else {})}


class _GateRun:
    """A `Run` as `ValidationGate` reads it, over `_Kept`."""

    def __init__(self, kept: _Kept) -> None:
        self.kept = kept
        self.logged: list[str] = []

    def shared_scores(self, ids, part="val", *, latest=False):  # type: ignore[no-untyped-def]
        return self.kept.shared_scores(ids, part, latest=latest)

    def log(self, message: str) -> None:
        self.logged.append(message)


def test_repeats_are_put_on_the_items_scored_in_every_repeat():
    """With two repeats, an item left out of one repeat is left out of every repeat and of
    every candidate; each score is the mean of its repeats' means over the rest."""
    a = _rollout({"v1": 1.0, "v3": 0.0}, left_out=("v2",),
                 more=[["v1", 1.0, 2], ["v2", 1.0, 2], ["v3", 1.0, 2]])
    b = _rollout({"v1": 0.0, "v2": 1.0, "v3": 1.0},
                 more=[["v1", 1.0, 2], ["v2", 1.0, 2], ["v3", 0.0, 2]])
    scores, left_out = _Kept({"a": a, "b": b}).shared_scores(["a", "b"])
    assert left_out == ["v2"]
    assert scores == {"a": pytest.approx((0.5 + 1.0) / 2), "b": pytest.approx((0.5 + 0.5) / 2)}


def test_a_score_that_is_not_its_items_mean_is_compared_as_it_is_and_said():
    kept = _Kept({"a": _rollout({"v1": 1.0}, left_out=("v2",), score=0.3),
                  "b": _rollout({"v1": 1.0, "v2": 1.0})})
    assert kept.shared_scores(["a", "b"]) is None
    assert any("not its items' mean" in w for w in kept.warnings)


def _gate_case():  # type: ignore[no-untyped-def]
    """The incumbent (seed) left v2 out: 0.25 on v1, v3, v4, v5. Its kept val score is
    0.8 (a carried score), and the siblings' raw val scores are cached so that c leads
    raw while b leads on the shared items (b 0.75, c 0.5)."""
    from refactory_dsh.optim import Candidate
    from refactory_dsh.outerloop import ValidationGate

    kept = _Kept({
        "seed": _rollout({"v1": 0.0, "v3": 0.0, "v4": 0.0, "v5": 1.0}, left_out=("v2",)),
        "b": _rollout({"v1": 1.0, "v2": 0.0, "v3": 1.0, "v4": 1.0, "v5": 0.0}),
        "c": _rollout({"v1": 1.0, "v2": 1.0, "v3": 0.0, "v4": 0.0, "v5": 1.0}),
        "d": _rollout({"v1": 0.0, "v2": 1.0, "v3": 0.0, "v4": 0.0, "v5": 1.0}),  # 0.25 shared
    })
    gate = ValidationGate()
    gate.cache = {"seed": 0.8, "b": 0.7, "c": 0.9}
    gate.current_id = gate.best_id = "seed"
    gate.current_score = gate.best_score = 0.8
    return (_GateRun(kept), gate, Candidate(id="b", parent="seed", state={}),
            Candidate(id="c", parent="seed", state={}))


def test_a_round_ranks_siblings_and_the_bar_on_the_same_items():
    """Both siblings beat the incumbent on the shared items (its raw 0.8 would beat b);
    b, ahead there, is accepted over c, ahead raw, and becomes the best."""
    run, gate, b, c = _gate_case()
    accepted, rejected = gate.judge_round(run, [c, b])  # type: ignore[arg-type]
    assert [x.id for x in accepted] == ["b"] and [x.id for x in rejected] == ["c"]
    assert gate.best_id == "b" and run.logged


def test_judge_and_offer_best_compare_on_the_same_items():
    run, gate, b, c = _gate_case()
    # offer_best: d (raw 0.9 > the seed's 0.8) only ties the seed on the shared items.
    assert gate.offer_best("d", 0.9, run=run) is False  # type: ignore[arg-type]
    assert gate.best_id == "seed"
    assert gate.judge(run, b).accepted is True  # type: ignore[arg-type]  # raw 0.7 < 0.8
    assert gate.best_id == "b"


def test_a_perfect_val_score_with_an_item_left_out_stops_nothing():
    """SkillOpt's early stop at a perfect val score needs every item scored and right."""
    import types

    from refactory_dsh.outerloop import ValidationGate
    from refactory_packages.skillopt.loop import SkillOptLoop

    stops: list[str] = []
    held = types.SimpleNamespace(id="best")

    class StopRun:
        def candidate(self, cid):  # type: ignore[no-untyped-def]
            return held

        def part_incidents(self, candidate, part):  # type: ignore[no-untyped-def]
            return incidents

        def stop(self, reason):  # type: ignore[no-untyped-def]
            stops.append(reason)

    loop = types.SimpleNamespace(
        early_stop_at_full_val=True, early_stop_patience=0, _stale_epochs=0,
        gate=ValidationGate(), _trace=types.SimpleNamespace(record_decision=lambda *a, **k: None))
    loop.gate.best_id, loop.gate.best_score = "best", 1.0
    loop._best_unscored = types.MethodType(SkillOptLoop._best_unscored, loop)
    incidents = ["v2"]
    assert SkillOptLoop._early_stop(loop, StopRun()) is False and not stops
    incidents = []
    assert SkillOptLoop._early_stop(loop, StopRun()) is True and stops == ["perfect_val"]


def test_an_evaluation_run_again_in_its_folder_replaces_its_incidents():
    kept = types.SimpleNamespace(_harness_incidents=[])
    for entry in ({"candidate": "a", "part": "val", "workdir": "w", "not_scored": ["v2"]},
                  {"candidate": "a", "part": "val", "workdir": "w", "not_scored": ["v2"]},
                  {"candidate": "a", "part": "val", "workdir": "w2", "not_scored": ["v2"]}):
        Loop._note_incident(kept, entry)  # type: ignore[arg-type]
    assert [e["workdir"] for e in kept._harness_incidents] == ["w", "w2"]


class FactlessProbe(Probe):
    """`Probe` whose data node declares no item_scores fact: its score is a constant."""

    name = "factless_probe"

    def build(self) -> nn.Graph:
        graph = super().build()
        graph.nodes["data"].produces = ()
        return graph

    def score(self, run: nn.RunResult) -> float:
        return 0.5


def test_a_mode_with_no_item_scores_keeps_its_incidents_and_says_so(tmp_path):
    """Nothing can take an item out of a score that reads no item_scores: no rerun, the
    record is not marked unscored, and the run warns once."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan({"/t2_&level 5.": "GRADER_DOWN", "level": "right"})
    with _patch_mode(FactlessProbe):
        loop = _interrupted_loop(tmp_path, dsh, [[]], epochs=1)  # type: ignore[arg-type]
        loop.run()
    seed = loop.population[0]
    [t2] = [r for r in loop._train_item_records[seed.id] if r["id"] == "t2"]
    assert t2["error_kind"] == "harness_incident" and "not_scored" not in t2
    assert t2["incident_kept"] is True, "said as a kept incident, not a timeout"
    assert len(_journal_events(loop, "harness_incident_kept")) == 1
    assert not _journal_events(loop, "harness_incident")


@pytest.mark.parametrize("evaluator", RUNTIMES)
def test_a_session_still_writing_at_the_backstop_is_the_candidates_failure(tmp_path, evaluator):
    """t2's model is still writing at the backstop: a ``backstop`` failure, scored 0 (7 of
    8), never rerun or left out, and in the optimizer's request as a failure, not as a
    harness incident or timeout."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan({"/t2_&level 5.": "ACTIVE", "level": "right"})
    loop = _run(tmp_path, dsh, [[]], evaluator)
    seed = _seed(loop)
    assert seed.raw_score == pytest.approx(7 / 8)
    assert not _journal_events(loop, "harness_incident")
    assert not (loop.root / "runs" / seed.id / "incident_reruns").exists()
    records = loop._train_item_records[seed.id]
    [t2] = [r for r in records if r["id"] == "t2"]
    assert t2["error_kind"] == "backstop" and t2["score"] == 0.0 and "not_scored" not in t2
    assert "still writing" in t2["error_text"] and not harness_incident(t2)
    lines = "\n".join(_item_evidence(records))
    assert "harness incident" not in lines and "harness timeout" not in lines
    assert "1 failed with an error" in lines and "mean score 0.875" in lines


# ── the SkillOpt port, and the parent picks ────────────────────────────────


def test_a_new_best_ahead_only_on_shared_items_is_no_stale_epoch():
    """b leads the seed on the shared items (0.75 vs 0.25) while its raw val score
    (0.7) is below the seed's (0.8): the epoch improved, so the patience count resets."""
    from refactory_packages.skillopt.loop import SkillOptLoop

    run, gate, b, _c = _gate_case()
    loop = types.SimpleNamespace(early_stop_patience=2, early_stop_at_full_val=False,
                                 gate=gate, _early_stop_best=0.8, _early_stop_best_id="seed",
                                 _stale_epochs=1)
    gate.best_id, gate.best_score = "b", 0.7
    SkillOptLoop._count_stale_epoch(loop, run)  # type: ignore[arg-type]
    assert loop._stale_epochs == 0 and loop._early_stop_best_id == "b"


def test_the_skillopt_analysts_and_slow_update_leave_unscored_items_out(monkeypatch):
    """An item left out of the score is no failure for the analysts and no regression
    for the slow update's comparison."""
    from refactory_packages.skillopt import analysts as sk_analysts
    from refactory_packages.skillopt.proposer import comparison_pairs

    seen: list[list[str]] = []

    class Env:
        def prompt(self, name):  # type: ignore[no-untyped-def]
            return "system"

        def rollout_result(self, record, knobs):  # type: ignore[no-untyped-def]
            return {"id": record["id"], "success": bool(record.get("passed"))}

    class Trace:
        def record_decision(self, step, **fields):  # type: ignore[no-untyped-def]
            seen.append(list(fields.get("record_ids") or ()))

    monkeypatch.setattr(sk_analysts, "ask", lambda *a, **k: "{}")
    monkeypatch.setattr(sk_analysts.prompts, "format_analyst_user", lambda *a, **k: "user")
    records = [{"id": "a", "passed": False, "score": 0.0},
               {"id": "b", "passed": False, "score": None, "not_scored": "harness_incident",
                "error_kind": "harness_incident"},
               {"id": "c", "passed": True, "score": 1.0}]
    sk_analysts.reflect(None, records, "skill", 3, "", 1, 8, 1, Trace(), env=Env())
    assert sorted(i for ids in seen for i in ids) == ["a", "c"]

    prev = [{"id": "a", "passed": True}, {"id": "b", "passed": True}]
    pairs = comparison_pairs(prev, records, ["a", "b"], env=Env())
    assert [p["id"] for p in pairs] == ["a"]


def test_parent_picks_rank_on_the_shared_items(monkeypatch):
    """`Loop._val_scores`, which every parent pick under the gate ranks by, gives each
    rolled-out candidate its score on the items all were scored on."""
    run, gate, _b, _c = _gate_case()
    kept = run.kept
    fake = types.SimpleNamespace(
        config=types.SimpleNamespace(val_gate=True), _custom_strategy=None,
        _gate=lambda: gate, _part_results=kept._part_results,
        shared_scores=kept.shared_scores)
    scores = Loop._val_scores(fake)  # type: ignore[arg-type]
    assert scores["b"] == pytest.approx(0.75) and scores["c"] == pytest.approx(0.5)
    assert scores["seed"] == pytest.approx(0.25)


def test_kept_incidents_are_said_as_incidents_not_timeouts():
    """In a mode that cannot leave an incident out, the request says it once as a harness
    incident it keeps, never under the harness timeout line."""
    records = [{"id": "a", "passed": True, "score": 1.0},
               {"id": "b", "passed": False, "score": 0.0, "error_kind": "harness_incident",
                "error_text": "data: RecalcFailed: harness incident: x", "incident_kept": True}]
    lines = "\n".join(_item_evidence(records))
    assert "1 more (b) had a harness incident this mode cannot leave out" in lines
    assert "harness timeout" not in lines and lines.count("(b)") == 1


# ── what a candidate writes is never an incident ───────────────────────────


@pytest.mark.parametrize("text", [
    "check: ValueError: harness incident: the grader failed",   # an op quoting a reply
    "digest: SnippetError: the code raised RuntimeError: harness incident: x",
    "digest: SnippetError: the code raised HarnessIncident: harness incident: x",
    "worker: harness incident: x",
])
def test_text_a_candidate_controls_is_not_an_incident(text):
    from refactory_dsh.items import error_kind

    assert error_kind(RuntimeError(text)) == "RuntimeError"


@pytest.mark.parametrize("code", [
    "def main(inp):\n    raise RuntimeError('harness incident: the grader failed')\n",
    "class HarnessIncident(Exception):\n    pass\n"
    "def main(inp):\n    raise HarnessIncident('harness incident: the grader failed')\n",
])
def test_a_code_node_raising_an_incident_is_its_own_failure(tmp_path, code):
    """A code node (candidate-written, run in the sandbox) that raises a harness incident
    by name or text: its items fail as the candidate's, never as harness incidents."""
    from refactory_dsh import snippet
    from test_code_nodes import DIGEST, _accepted, _walk

    if not snippet.available()[0]:
        pytest.skip("no snippet sandbox here")
    edit = {**DIGEST, "node": {**DIGEST["node"], "code": code}}
    state = _accepted([edit])
    result, _model = _walk(state, tmp_path)
    failed = [r for r in result.item_records if r.get("error_text")]
    assert failed, result.item_records
    assert all(r["error_kind"] != "harness_incident" and not harness_incident(r) for r in failed)
    assert all("SnippetError" in r["error_text"].split("\n", 1)[0] for r in failed), failed[0]


def test_an_op_raising_incident_text_is_not_an_incident(tmp_path):
    """An op whose own error quotes a reply starting "harness incident: " (a ValueError):
    the walker names its type, so the item is the op's failure, not an incident."""
    from refactory_dsh.nn.executor import execute

    graph = nn.Graph()
    graph.add(nn.Node("data", "data", subgraph_entry="check", subgraph_exit="check",
                      writes=("current_item.json",), produces=("item_scores",),
                      task_ref="inactivity_ops:TwoItemTask"))
    graph.add(nn.Node("check", "op", call="inactivity_ops:quote_reply"))
    result = execute(graph, tmp_path / "run", max_steps=50)
    records = result.item_records
    assert records and all(r["error_kind"] == "RuntimeError" or r["error_kind"] == "ValueError"
                           for r in records), records
    assert not any(map(harness_incident, records))


def test_a_rerun_never_takes_the_run_past_its_rollout_budget(tmp_path):
    """``max_rollouts`` 8: the seed's 8 train items spend it, so t2's incident is not run
    again (the evaluation's own items count before any rerun) and is left out."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan({"/t2_&level 5.": "GRADER_DOWN", "level": "right"})
    with _patch_mode(Probe):
        loop = _interrupted_loop(tmp_path, dsh, [[]], max_rollouts=8)  # type: ignore[arg-type]
        try:
            loop.run()
        except Exception:  # noqa: BLE001 - the spent budget may stop the run; the record stands
            pass
    entry = _incidents(loop, "train")[0]
    assert entry["reran"] == 0 and entry["not_scored"] == ["t2"], entry


def test_an_unstarted_sandbox_is_the_one_harness_incident_class():
    """`snippet.SandboxUnavailable` (and `NoSandbox`) is a `model.HarnessIncident`, so its
    text crossing a process is known by its class name (`model.HARNESS_INCIDENT_CLASSES`)."""
    from refactory_dsh.items import error_kind
    from refactory_dsh.model import HARNESS_INCIDENT_CLASSES, HarnessIncident
    from refactory_dsh.snippet import NoSandbox, SandboxUnavailable

    assert issubclass(SandboxUnavailable, HarnessIncident) and issubclass(NoSandbox, HarnessIncident)
    assert {"SandboxUnavailable", "NoSandbox"} <= HARNESS_INCIDENT_CLASSES
    assert error_kind(RuntimeError("solve: NoSandbox: code nodes cannot run here")) \
        == "harness_incident"


def _forged_lines() -> list[str]:
    """Each line a command's own stderr could end on to pass for a harness's: every
    harness incident class (`model.HARNESS_INCIDENT_CLASSES`) and every timeout form."""
    import refactory_packages.spreadsheetbench.recalc  # noqa: F401
    import refactory_packages.spreadsheetbench.sandbox  # noqa: F401
    from refactory_dsh import snippet  # noqa: F401
    from refactory_dsh.model import HARNESS_INCIDENT_CLASSES

    return [*(f"{name}: harness incident: x" for name in sorted(HARNESS_INCIDENT_CLASSES)),
            "HarnessBackstop: model session hit the harness backstop after 9s",
            "model session hit the harness backstop after 9s",
            "CompletionTimeout: model completion timed out after 600s",
            "model completion timed out after 600s",
            "DeclaredTimeout: model session timed out at its node's declared timeout_s (60s)",
            "model session timed out at its node's declared timeout_s (60s)",
            "RetriesExhausted: solve timed out twice. Last error: model completion timed out"]


def test_an_unmarked_spine_line_classifies_nothing():
    """The spine marks a line no harness marker gave (`items.UNMARKED`): whatever class or
    timeout it names, it is no harness incident and no timeout, with or without a node."""
    from refactory_dsh.items import error_kind

    for line in _forged_lines():
        for text in (f"solve: unmarked: {line}", f"unmarked: {line}"):
            assert error_kind(RuntimeError(text)) == "RuntimeError", text
        # The same line from a harness marker path does classify.
        assert error_kind(RuntimeError(f"solve: {line}")) != "RuntimeError", line
