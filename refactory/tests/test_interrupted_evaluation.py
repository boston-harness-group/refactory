"""An evaluation whose items died with the run's own processes is not a result.

On 2026-11-12 a host out-of-memory kill took out an outer-loop run's processes while a
candidate's train evaluation was in flight: every item failed, the candidate was
recorded as ``failed: data: all 35 items failed``, and the resumed run never evaluated
it again. Now an item that died with a process of the run -- the wire to the model kept
failing, the completion's process was killed, an op child never reported, the run's stop
cancelled it -- has an unknown outcome (`items.interrupted_item`). Such an evaluation is
retried inside the run after each wait of `INTERRUPTED_RETRY_WAITS_S`; still interrupted,
it is journaled and stops the run (exit `EXIT_INTERRUPTED`) with nothing recorded, and
``--resume`` evaluates the candidate again and gates it. The `INTERRUPTION_CAP`-th stop
of the same evaluation records it as failed instead. Each earlier attempt's per-item
receipts are moved to the run's archive, never deleted. A model's own failure (a wrong
answer, a timeout, an output limit, a runtime abort) is still the candidate's result.

Most runs here are the real loop and walker over a scripted ``dsh`` reached through the
real model client. Each test fails when the behaviour it names is reverted.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import CancelledError
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import loop as loop_module
from refactory_dsh import nn
from refactory_dsh import task as task_module
from refactory_dsh.events import render
from refactory_dsh.items import finished_pass_record, interrupted_item
from refactory_dsh.loop import (
    EXIT_INTERRUPTED, INTERRUPTED_RETRY_WAITS_S, INTERRUPTION_CAP, EvaluationInterrupted,
    Loop, LoopConfig,
)
from refactory_dsh.model import CompletionFailed, DshModelClient, is_retryable_failure
from refactory_dsh.optim import Candidate
from refactory_dsh.outerloop import BudgetStopped, Run

from test_skillopt import MODE_ID, TEST_IDS, TRAIN_IDS, VAL_IDS, _patch_mode, _write_parts
from test_val_gate import ScriptedLevels, _journal_events, _level

TASK_REF = "interrupted_evaluation_task"

#: The in-run retry waits the tests use, so a retry costs nothing.
WAITS = (0.01, 0.02)

#: A scripted ``dsh``: the plan maps keys to one behaviour -- ``KILL`` / ``ABORT`` (the
#: process dies of SIGKILL, as the out-of-memory killer kills it, or of SIGABRT, as a
#: runtime out of heap aborts), ``SLEEP`` (it outlives its wall), ``dsh: ...`` (a
#: failure with that stderr), or an answer. The first key whose ``&``-separated parts
#: are all in the prompt or the call's folder (``occasions/val_0``, an item's pass) decides.
FAKE_DSH = '''import json, os, pathlib, signal, sys, time
root = pathlib.Path({root!r})
plan = json.loads((root / "plan.json").read_text())
seen = os.getcwd() + "\\n" + sys.stdin.read()
step = next(value for key, value in plan.items() if all(part in seen for part in key.split("&")))
if step == "KILL":
    os.kill(os.getpid(), signal.SIGKILL)
if step == "ABORT":
    os.kill(os.getpid(), signal.SIGABRT)
if step == "SLEEP":
    time.sleep(30)
if step.startswith("dsh:"):
    print(step, file=sys.stderr)
    sys.exit(1)
print(step)
'''

#: The model client's wire failure, as `dsh` reports a proxy that is gone.
TRANSPORT = "dsh: TRANSPORT: connect ECONNREFUSED 127.0.0.1:8082"

#: An output-limit stop whose streamed reasoning happens to say "429" and "rate limit".
CAPPED_429 = ("dsh: reasoning:\nso 3*143 = 429, and a rate limit of 7 per hour gives\n"
              "dsh: max-tokens: the model reached its output token limit before it finished")


class AnswerTask(task_module.Task):
    """The train, val and test ids of the toy split; an item passes when its pass kept
    the answer ``right``."""

    name = TASK_REF

    def instances(self, *, part: str | None = None, seed: int | None = None) -> list[task_module.TaskInstance]:
        return [task_module.TaskInstance(id=item) for item in TRAIN_IDS + VAL_IDS + TEST_IDS]

    def verify(self, instance, workspace, *, part=None, seed=None):  # type: ignore[no-untyped-def]
        final = Path(workspace) / "final.txt"
        right = final.exists() and final.read_text().strip() == "right"
        return task_module.VerifyResult(passed=right, score=1.0 if right else 0.0)


task_module.register_task(TASK_REF, AnswerTask)


class AskingProbe(nn.Mode):
    """Per item: an agent step asked at the candidate's ``level``, then an op that keeps
    its answer (a missing answer leaves the item with none)."""

    name = "asking_probe"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {"level": nn.Knob(default=5, kind="int", lo=0, hi=9, description="a level")}

    def pass_nodes(self, graph: nn.Graph) -> str:
        """Add the pass after the solver; return its last node."""
        graph.add(nn.Node("keep", "op", call="loop_ops:keep_answer", reads=("answer.txt",),
                          writes=("final.txt",)))
        graph.connect("solve", "keep")
        return "keep"

    def after(self, graph: nn.Graph) -> None:
        graph.add(nn.Node("finish", "op", call="loop_ops:write_best",
                          writes=("best.json",), produces=("best",)))
        graph.connect("data", "finish")

    #: The pass's last node (`pass_nodes`).
    pass_exit = "keep"

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node("data", "data", subgraph_entry="solve", subgraph_exit=self.pass_exit,
                          task_ref=TASK_REF, writes=("current_item.json",),
                          produces=("item_scores",), parallelism=4))
        graph.add(nn.Node("solve", "agent", prompt_template="Answer at level {{level}}.",
                          writes=("answer.txt",), tool_policy="none", role="solver",
                          knobs=("level",), timeout_s=5))
        assert self.pass_nodes(graph) == self.pass_exit
        self.after(graph)
        return graph

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0


class DyingOpProbe(AskingProbe):
    """The pass ends with an op whose child is killed before it reports (`DIE_FLAG`)."""

    name = "dying_op_probe"
    pass_exit = "check"

    def pass_nodes(self, graph: nn.Graph) -> str:
        last = super().pass_nodes(graph)
        graph.add(nn.Node("check", "op", call="loop_ops:die_if_flagged", knobs=("level",)))
        graph.connect(last, "check")
        return "check"


class SummarizingProbe(AskingProbe):
    """After the data node, an agent outside any item summarises the run."""

    name = "summarizing_probe"

    def after(self, graph: nn.Graph) -> None:
        graph.add(nn.Node("summarize", "agent", prompt_template="Summarize at level {{level}}.",
                          writes=("summary.txt",), tool_policy="none", role="solver",
                          knobs=("level",), timeout_s=5))
        graph.connect("data", "summarize")


class Dsh:
    """The scripted ``dsh`` and the model client that runs it."""

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        script = root / "fake_dsh.py"
        script.write_text(FAKE_DSH.format(root=str(root)))
        self.cli = root / "dsh"
        self.cli.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{script}" "$@"\n')
        self.cli.chmod(0o755)

    def plan(self, *, only: dict[str, str] | None = None, **levels: str) -> None:
        """Each ``level<n>``'s behaviour; ``only`` maps ``"<part>&<part>"`` keys to a
        behaviour that comes first."""
        (self.root / "plan.json").write_text(json.dumps({**(only or {}), **{
            f"level {level.removeprefix('level')}.": step for level, step in levels.items()}}))

    def client(self) -> DshModelClient:
        return DshModelClient(worker_backend="dsh", provider="p", model="m", dsh_cli=str(self.cli),
                              attempts=1, timeout_s=60)


@pytest.fixture(autouse=True)
def quick_retries(monkeypatch):  # type: ignore[no-untyped-def]
    """The scripted plans start from the seed alone, and an in-run retry waits `WAITS`."""
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)
    monkeypatch.setattr(loop_module, "INTERRUPTED_RETRY_WAITS_S", WAITS)


def _loop(tmp_path: Path, dsh: Dsh, plan: list[list[tuple[int, int]]], **overrides: Any) -> Loop:
    config = dict(epochs=2, batch_size=1, optimizer="random", seed=7, runs_root=tmp_path / "run",
                  allowed_ops=("set_knob",), max_evaluations=60, holdout=0.0, val_gate=True,
                  parts_manifest=str(_write_parts(tmp_path, test=True)))
    config.update(overrides)
    loop = Loop(MODE_ID, LoopConfig(**config), project=Path("."), model=dsh.client())
    loop.optimizer = ScriptedLevels(plan)
    return loop


def _run(tmp_path: Path, dsh: Dsh, plan: list[list[tuple[int, int]]], mode: type = AskingProbe,
         **overrides: Any) -> Loop:
    with _patch_mode(mode):
        loop = _loop(tmp_path, dsh, plan, **overrides)
        loop.run()
    return loop


def _stopped(tmp_path: Path, dsh: Dsh, plan: list[list[tuple[int, int]]], mode: type = AskingProbe,
             match: str = "resume the run", **overrides: Any) -> Loop:
    with _patch_mode(mode):
        loop = _loop(tmp_path, dsh, plan, **overrides)
        with pytest.raises(EvaluationInterrupted, match=match):
            loop.run()
    return loop


def _child(loop: Loop) -> Candidate:
    return next(c for c in loop.population if _level(c) == 6)


def _done(folder: Path) -> list[Path]:
    return sorted(path.relative_to(folder) for path in folder.rglob(".done"))


def _outcomes(loop: Loop, candidate: Candidate | None = None) -> list[tuple[int, str]]:
    return [(r["attempt"], r["outcome"]) for r in _journal_events(loop, "evaluation_interrupted")
            if candidate is None or r["id"] == candidate.id]


ATTEMPTS = [(1, "retrying"), (2, "retrying"), (3, "stopped")]


# ── a train evaluation that dies ───────────────────────────────────────────


@pytest.mark.parametrize("death", ["transport", "killed"])
def test_an_interrupted_evaluation_is_retried_then_stops_and_the_resume_evaluates_it(tmp_path, death):
    """The child's train run dies with the run's processes in every attempt: its model
    calls fail on the wire (the proxy is gone), or their processes are killed (retried
    as crashes, then spent, so the items have no answer). It is tried again after each
    wait, then the run stops with nothing recorded; the resume evaluates the child from
    the start, scores it normally and gates it. Every attempt's receipts are archived."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan(level5="wrong", level6=TRANSPORT if death == "transport" else "KILL")
    loop = _stopped(tmp_path, dsh, [[(5, 6)], []])
    child = _child(loop)
    assert not child.evaluated and child.score is None and child.error is None
    assert child.id not in {r["id"] for r in _journal_events(loop, "score")}, \
        "an interrupted evaluation is never recorded as a score"
    assert child.id not in {r["id"] for r in _journal_events(loop, "train_records")}
    entries = _journal_events(loop, "evaluation_interrupted")
    assert _outcomes(loop) == ATTEMPTS
    assert [e.get("retry_in_s") for e in entries] == [*WAITS, None]
    assert all(e["causes"] == {death: len(TRAIN_IDS)} and e["part"] == "train" for e in entries)
    assert entries[-1]["stops"] == 1
    last = _done(loop.root / "runs" / child.id / "passes" / "data")
    assert len(last) == len(TRAIN_IDS), "setup: the last attempt left per-item receipts"

    dsh.plan(level5="wrong", level6="right")
    with _patch_mode(AskingProbe):
        resumed = _loop(tmp_path, dsh, [[]], resume=True)
        resumed.run()
    child = _child(resumed)
    assert child.evaluated and child.exit_ok and child.raw_score == pytest.approx(1.0)
    assert [r["raw_score"] for r in _journal_events(resumed, "score") if r["id"] == child.id] == [1.0]
    assert child.id in resumed._gate_accepted, "the re-evaluated child is gated on its val score"
    assert resumed._gate().cache[child.id] == pytest.approx(1.0)

    # Two in-run retries and the resume each archived the attempt before them.
    archived = _journal_events(resumed, "evaluation_archived")
    assert [a["id"] for a in archived] == [child.id] * 3
    assert all(a["workdir"] == f"runs/{child.id}" for a in archived)
    for entry in archived:
        kept = resumed.root / entry["archive"] / "passes" / "data"
        assert _done(kept) == last
        records = [finished_pass_record(kept / path.parent) for path in _done(kept)]
        assert all(interrupted_item(record) == death for record in records)


def test_an_in_run_retry_that_succeeds_needs_no_stop(tmp_path, monkeypatch):
    """The wire is down for the first attempt only: the retry after the wait scores the
    child, the run goes on, and nothing counts as a stop."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan(level5="wrong", level6=TRANSPORT)
    original = Loop._archive_unfinished

    def recover(self, candidate_id, workdir, **kwargs):  # type: ignore[no-untyped-def]
        moved = original(self, candidate_id, workdir, **kwargs)
        if moved is not None:  # an interrupted attempt was set aside: the proxy is back
            dsh.plan(level5="wrong", level6="right")
        return moved

    monkeypatch.setattr(Loop, "_archive_unfinished", recover)
    loop = _run(tmp_path, dsh, [[(5, 6)], []])
    child = _child(loop)
    assert child.evaluated and child.raw_score == pytest.approx(1.0)
    assert _outcomes(loop) == [(1, "retrying")]
    assert loop._interrupted_stops == {}
    assert child.id in loop._gate_accepted


def test_an_evaluation_interrupted_at_every_resume_is_recorded_failed_at_the_cap(tmp_path):
    """The same evaluation dies in every attempt of every resume. The first
    INTERRUPTION_CAP - 1 stops stop the run; the cap's records it as failed, naming why,
    and the run goes on. In-run retries are not stops; the count survives a resume."""
    assert (INTERRUPTION_CAP, loop_module.INTERRUPTED_RETRY_WAITS_S) == (3, WAITS)
    assert INTERRUPTED_RETRY_WAITS_S == (120.0, 300.0), "the shipped waits: 2 min, then 5 min"
    dsh = Dsh(tmp_path / "dsh")
    # Its train run dies every time; its val and test rollouts (a failed train run is
    # still rolled out on val) answer.
    dsh.plan(only={"occasions/&level 6.": "right"}, level5="wrong", level6=TRANSPORT)
    loop = _stopped(tmp_path, dsh, [[(5, 6)], []])
    loop = _stopped(tmp_path, dsh, [[]], resume=True)
    assert _outcomes(loop) == ATTEMPTS * 2
    with _patch_mode(AskingProbe):
        resumed = _loop(tmp_path, dsh, [[]], resume=True)
        resumed.run()
    child = _child(resumed)
    assert _outcomes(resumed) == ATTEMPTS * 2 + [*ATTEMPTS[:2], (3, "recorded_failed")]
    assert [r["stops"] for r in _journal_events(resumed, "evaluation_interrupted")
            if r["outcome"] != "retrying"] == [1, 2, 3]
    assert child.evaluated and not child.exit_ok and child.score is None
    assert "train evaluation interrupted 3 times; recorded as failed" in child.error
    [score] = [r for r in _journal_events(resumed, "score") if r["id"] == child.id]
    assert score["exit_ok"] is False and "recorded as failed" in score["error"]


def test_a_partly_interrupted_evaluation_is_no_result_either(tmp_path):
    """One item of eight dies on the wire, the rest answer: still no result, since that
    item's 0 would be scored as the candidate's."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan(only={"/t3_&level 6.": TRANSPORT}, level5="wrong", level6="right")
    loop = _stopped(tmp_path, dsh, [[(5, 6)], []])
    entry = _journal_events(loop, "evaluation_interrupted")[-1]
    assert (entry["interrupted"], entry["items"], entry["causes"]) == (1, 8, {"transport": 1})
    assert not _child(loop).evaluated


def test_a_cancelled_model_call_is_stopped_not_failed(tmp_path, monkeypatch):
    """The run's stop cancels the child's model calls (``model completion cancelled``):
    their items are interrupted, cause ``stopped``."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan(level5="wrong", level6="right")
    original = DshModelClient._run_completion

    def cancelled(self, cmd, run_dir, *args, task=None, **kwargs):  # type: ignore[no-untyped-def]
        if "level 6." in str(task):
            raise CancelledError("model completion cancelled; not retried")
        return original(self, cmd, run_dir, *args, task=task, **kwargs)

    monkeypatch.setattr(DshModelClient, "_run_completion", cancelled)
    loop = _stopped(tmp_path, dsh, [[(5, 6)], []])
    assert _journal_events(loop, "evaluation_interrupted")[-1]["causes"] == {"stopped": len(TRAIN_IDS)}


def test_an_op_child_killed_before_it_reports_is_an_unknown_outcome(tmp_path):
    """An op's child process is killed before it reports what it did: its items are
    interrupted, cause ``outcome unknown``; once it lives, the resume scores the child."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan(level5="wrong", level6="right")
    (tmp_path / "die.flag").write_text("6")
    loop = _stopped(tmp_path, dsh, [[(5, 6)], []], mode=DyingOpProbe)
    assert _journal_events(loop, "evaluation_interrupted")[-1]["causes"] == {
        "outcome unknown": len(TRAIN_IDS)}
    (tmp_path / "die.flag").unlink()
    with _patch_mode(DyingOpProbe):
        resumed = _loop(tmp_path, dsh, [[]], resume=True)
        resumed.run()
    assert _child(resumed).raw_score == pytest.approx(1.0)


def test_a_run_error_outside_any_item_is_an_interruption_too(tmp_path):
    """Every item answers; the agent after the data node fails on the wire. The run's own
    error says the process died, so the evaluation is interrupted (no item counted)."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan(only={"Summarize at level 6.": TRANSPORT}, level5="wrong", level6="right")
    loop = _stopped(tmp_path, dsh, [[(5, 6)], []], mode=SummarizingProbe)
    entry = _journal_events(loop, "evaluation_interrupted")[-1]
    assert (entry["interrupted"], entry["causes"]) == (0, {"transport": 0})
    assert "summarize" in entry["error"] and not _child(loop).evaluated


def test_a_run_stopped_mid_pass_keeps_its_receipts_and_evaluates_again(tmp_path, monkeypatch):
    """The run's own process is stopped (SIGTERM becomes SystemExit) once half of the
    child's model calls were made, so some items finished and the rest never did. The
    resume evaluates the child from the start, and the finished items' receipts from
    before the stop are in the archive, not deleted."""
    import threading

    dsh = Dsh(tmp_path / "dsh")
    dsh.plan(level5="wrong", level6="right")
    original = DshModelClient._run_completion
    lock, asked = threading.Lock(), []

    def stopping(self, cmd, run_dir, *args, task=None, **kwargs):  # type: ignore[no-untyped-def]
        if "level 6." in str(task):
            with lock:
                asked.append(run_dir)
                late = len(asked) > len(TRAIN_IDS) // 2
            if late:
                raise SystemExit(143)
        return original(self, cmd, run_dir, *args, task=task, **kwargs)

    monkeypatch.setattr(DshModelClient, "_run_completion", stopping)
    with _patch_mode(AskingProbe):
        loop = _loop(tmp_path, dsh, [[(5, 6)], []], max_concurrent_evaluations=1)
        with pytest.raises(SystemExit):
            loop.run()
    child = _child(loop)
    assert not child.evaluated
    receipts = _done(loop.root / "runs" / child.id / "passes" / "data")
    assert 0 < len(receipts) < len(TRAIN_IDS), "setup: some items finished before the stop"

    monkeypatch.setattr(DshModelClient, "_run_completion", original)
    with _patch_mode(AskingProbe):
        resumed = _loop(tmp_path, dsh, [[]], resume=True, max_concurrent_evaluations=1)
        resumed.run()
    child = _child(resumed)
    assert child.evaluated and child.raw_score == pytest.approx(1.0)
    assert child.id in resumed._gate_accepted
    [archived] = _journal_events(resumed, "evaluation_archived")
    kept = resumed.root / archived["archive"] / "passes" / "data"
    assert _done(kept) == receipts
    assert all(finished_pass_record(kept / path.parent)["score"] == 1.0 for path in _done(kept))


def test_the_resume_archive_leaves_the_candidates_occasions(tmp_path):
    """The archive takes the interrupted train attempt, not the candidate's occasions
    (its val and test rollouts, which have folders of their own)."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan(level5="wrong", level6=TRANSPORT)
    loop = _stopped(tmp_path, dsh, [[(5, 6)], []])
    occasion = loop.root / "runs" / _child(loop).id / "occasions" / "test_9"
    occasion.mkdir(parents=True)
    (occasion / "item_records.json").write_text("[]")
    dsh.plan(level5="wrong", level6="right")
    with _patch_mode(AskingProbe):
        _loop(tmp_path, dsh, [[]], resume=True).run()
    assert (occasion / "item_records.json").read_text() == "[]"


def test_an_evaluated_candidate_evaluated_again_archives_nothing(tmp_path):
    """Only an unfinished attempt is archived: a strategy that evaluates a candidate on
    train again does what it did before."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan(level5="right")
    with _patch_mode(AskingProbe):
        loop = _loop(tmp_path, dsh, [[]])
        loop.init()
        seed = loop.population[0]
        Run(_loop=loop).evaluate(seed)
        Run(_loop=loop).evaluate(seed)
    assert seed.raw_score == pytest.approx(1.0)
    assert _journal_events(loop, "evaluation_archived") == []


# ── val, the re-score and the final test ───────────────────────────────────


def test_an_interrupted_val_rollout_is_no_val_score(tmp_path):
    """The child's train run is scored; its val rollout dies on the wire in every
    attempt. No val score is recorded and the gate judges nothing from it; the resume
    rolls it out again in a fresh occasion (the interrupted one stays) and gates it."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan(only={"occasions/val_&level 6.": TRANSPORT}, level5="wrong", level6="right")
    loop = _stopped(tmp_path, dsh, [[(5, 6)], []], match="val evaluation was interrupted")
    child = _child(loop)
    assert child.evaluated and child.raw_score == pytest.approx(1.0), "its train run was a result"
    assert child.id not in {r["id"] for r in _journal_events(loop, "part_result")}
    assert child.id not in loop._gate_accepted + loop._gate_rejected
    interrupted = _journal_events(loop, "evaluation_interrupted")
    assert _outcomes(loop) == ATTEMPTS
    assert interrupted[-1]["part"] == "val" and interrupted[-1]["causes"] == {"transport": len(VAL_IDS)}

    dsh.plan(level5="wrong", level6="right")
    with _patch_mode(AskingProbe):
        resumed = _loop(tmp_path, dsh, [[]], resume=True)
        resumed.run()
    vals = [r["result"] for r in _journal_events(resumed, "part_result")
            if r["id"] == child.id and r["part"] == "val"]
    assert [(v["occasion"], v["score"]) for v in vals] == [(1, 1.0)]
    assert child.id in resumed._gate_accepted
    assert (resumed.root / interrupted[-1]["workdir"] / "passes").is_dir(), "the interrupted occasion stays"


def test_an_interrupted_rescore_is_no_evidence_and_is_kept(tmp_path):
    """Fresh evidence: the parent is re-scored beside its child, and the re-score dies.
    Its records are not replaced, nothing counts it as a re-score, and the run stops.
    When it is run again (a resume in that epoch tries it again) its interrupted passes
    are moved to the archive."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan(only={"train_epoch_&level 5.": TRANSPORT}, level5="wrong", level6="right")
    loop = _stopped(tmp_path, dsh, [[(5, 6)], []], match="train evaluation was interrupted")
    parent = loop.population[0]
    interrupted = _journal_events(loop, "evaluation_interrupted")[-1]
    assert (interrupted["id"], interrupted.get("rescore_epoch")) == (parent.id, 2)
    assert interrupted["key"] == f"{parent.id}:train@2"
    assert _journal_events(loop, "rescore") == _journal_events(loop, "rescore_failed") == []
    assert [r["id"] for r in _journal_events(loop, "train_records")] == [parent.id, _child(loop).id], \
        "the parent keeps its own records"
    receipts = _done(loop.root / interrupted["workdir"])
    assert len(receipts) == len(TRAIN_IDS)
    retried = len(_journal_events(loop, "evaluation_archived"))
    assert retried == len(WAITS), "each in-run retry archived the attempt before it"

    dsh.plan(level5="wrong", level6="right")
    with _patch_mode(AskingProbe):
        resumed = _loop(tmp_path, dsh, [[]], resume=True)
        resumed.init()
        assert resumed._interrupted_stops == {f"{parent.id}:train@2": 1}, "rebuilt from the journal"
        assert resumed._rescore(resumed.population[0], 2)
    [archived] = _journal_events(resumed, "evaluation_archived")[retried:]
    assert (archived["id"], archived["workdir"]) == (parent.id, interrupted["workdir"])
    assert _done(resumed.root / archived["archive"]) == receipts
    assert [r["epoch"] for r in _journal_events(resumed, "rescore")] == [2]


def test_an_interrupted_final_test_is_scored_after_the_resume(tmp_path):
    """The winner's final test rollout dies on the wire: no test score is recorded and
    the run stops; the resume finishes the final evaluation with a real test score."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan(only={"occasions/test_&level 6.": TRANSPORT}, level5="wrong", level6="right")
    loop = _stopped(tmp_path, dsh, [[(5, 6)], []], match="test evaluation was interrupted")
    child = _child(loop)
    assert child.id in loop._gate_accepted
    assert not [r for r in _journal_events(loop, "part_result")
                if r["part"] == "test" and r["id"] == child.id]
    assert _journal_events(loop, "evaluation_interrupted")[-1]["part"] == "test"

    dsh.plan(level5="wrong", level6="right")
    with _patch_mode(AskingProbe):
        resumed = _loop(tmp_path, dsh, [[]], resume=True)
        resumed.run()
    tests = [r["result"] for r in _journal_events(resumed, "part_result")
             if r["part"] == "test" and r["id"] == child.id]
    assert [t["score"] for t in tests] == [1.0]


def test_a_budget_stopped_rollout_is_a_budget_stop_not_an_interruption(tmp_path, monkeypatch):
    """A val rollout the budget cut, whose cut items read as cancelled, is the budget's
    stop: flagged ``budget_stopped`` and raised as one, with no interruption journaled."""
    from refactory_dsh.nn.mode import RunResult

    dsh = Dsh(tmp_path / "dsh")
    dsh.plan(level5="right")

    def cut(self, candidate, workdir, part, **kwargs):  # type: ignore[no-untyped-def]
        record = {"id": "v1", "passed": False, "score": 0.0,
                  "error_text": "solve: CancelledError: model completion cancelled; not retried"}
        return (RunResult(exit_ok=False, error="elapsed-time budget exhausted", item_records=[record]),
                {"budget_stopped": True, "count": 1, "complete": False})

    with _patch_mode(AskingProbe):
        loop = _loop(tmp_path, dsh, [[]])
        loop.init()
        monkeypatch.setattr(Loop, "_measure", cut)
        with pytest.raises(BudgetStopped):
            loop._evaluate_part(loop.population[0], "val", None, 7)
    assert _journal_events(loop, "evaluation_interrupted") == []
    [part] = _journal_events(loop, "part_result")
    assert part["result"]["budget_stopped"] is True


# ── a model's own failure ──────────────────────────────────────────────────


@pytest.mark.parametrize("answer", ["garbage", "SLEEP", "ABORT", CAPPED_429])
def test_a_models_own_failure_is_still_the_candidates_result(tmp_path, answer):
    """Every item answers wrongly, outlives its wall, has its runtime abort (SIGABRT, as
    a runtime out of heap does), or stops at its output limit with "429" in its
    reasoning: the child is scored 0 or recorded failed, as before, and the run goes on."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan(level5="right", level6=answer)
    loop = _run(tmp_path, dsh, [[(5, 6)], []])
    child = _child(loop)
    assert child.evaluated, "a model's failure is a result"
    assert _journal_events(loop, "evaluation_interrupted") == []
    [score] = [r for r in _journal_events(loop, "score") if r["id"] == child.id]
    if answer == "SLEEP":
        assert (score["score"], score["exit_ok"]) == (None, False)
        assert score["error"] == f"data: all {len(TRAIN_IDS)} items failed"
    else:
        assert (score["raw_score"], score["exit_ok"]) == (0.0, True)
    assert child.id in loop._gate_rejected


def test_an_output_limit_is_never_a_wire_failure_whatever_its_reasoning_says(monkeypatch):
    """The model's streamed reasoning mentions "429" and "rate limit"; the diagnostic is
    the output limit. Not retried, and not called a wire failure."""
    assert not is_retryable_failure(CAPPED_429)
    assert is_retryable_failure("dsh: reasoning:\nthinking\n" + TRANSPORT)
    assert is_retryable_failure(TRANSPORT)

    class Capped:
        returncode, stdout, stderr = 1, "", CAPPED_429

    calls = []
    monkeypatch.setattr(DshModelClient, "_run_completion",
                        lambda *a, **k: calls.append(1) or Capped())
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(CompletionFailed) as raised:
        DshModelClient(attempts=3).complete("hello")
    assert raised.value.kind == "output_limit" and "on the wire" not in str(raised.value)
    assert len(calls) == 1


def test_a_wire_failure_is_called_one_only_when_its_cause_is_the_wire(monkeypatch):
    """A diagnostic line names the wire but the session stopped for another reason (the
    last cause): it fails at once with that cause, never as a wire failure."""
    class Mixed:
        returncode, stdout = 1, ""
        stderr = TRANSPORT + "\ndsh: max-tokens: the model reached its output token limit"

    monkeypatch.setattr(DshModelClient, "_run_completion", lambda *a, **k: Mixed())
    monkeypatch.setattr("time.sleep", lambda s: None)
    with pytest.raises(CompletionFailed) as raised:
        DshModelClient(attempts=3).complete("hello")
    assert raised.value.kind == "output_limit" and "on the wire" not in str(raised.value)


# ── how it is said ─────────────────────────────────────────────────────────


def test_the_progress_line_says_what_the_run_does_next():
    base = {"step": "evaluation_interrupted", "epoch": 4, "candidate": "3819a22f7869",
            "part": "train", "attempt": 1, "interrupted": 35, "items": 35,
            "causes": {"transport": 35}}
    assert render({**base, "outcome": "retrying", "retry_in_s": 120.0}) == (
        "epoch 4 · candidate 3819a22f7869 · train evaluation interrupted, attempt 1 (35/35 "
        "items died with the run's processes: transport 35) · not scored; trying again in 120s")
    assert render({**base, "attempt": 3, "outcome": "stopped", "stops": 1}).endswith(
        "· not scored; the run stops: resume it to evaluate it again")
    assert render({**base, "attempt": 3, "outcome": "recorded_failed", "stops": 3}).endswith(
        "· interrupted 3 times; recorded as failed")


def test_a_stopped_run_exits_75_so_a_supervisor_can_resume_it(tmp_path, monkeypatch, capsys):
    """The command line exits EX_TEMPFAIL (75) on an interrupted evaluation, not the 2 of
    a refused configuration, and says the run is resumable."""
    from test_parts import PartsProbe
    from test_parts import _patch_mode as parts_mode
    from test_seed_cache import _main, _manifest

    monkeypatch.setenv("REFACTORY_HOME", str(tmp_path / "home"))
    _manifest(tmp_path)

    def interrupted(self, candidate, **kwargs):  # type: ignore[no-untyped-def]
        raise EvaluationInterrupted(f"{candidate.id}'s train evaluation was interrupted")

    monkeypatch.setattr(Loop, "_evaluate", interrupted)
    with parts_mode(PartsProbe):
        code, _ = _main(tmp_path, "A", "--optimizer", "random", "--epochs", "1")
    assert code == EXIT_INTERRUPTED == 75
    said = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert said["resumable"] is True and "interrupted" in said["error"]


# ── which item died with the run ───────────────────────────────────────────


@pytest.mark.parametrize(("text", "missing", "cause"), [
    ("solve: CompletionFailed: model completion failed on the wire 3 time(s); last cause: "
     "dsh: TRANSPORT: fetch failed", None, "transport"),
    ("solve: CompletionFailed: model completion failed on the wire 2 time(s) and the node's "
     "time ran out before it could ask again; last cause: econnreset", None, "transport"),
    # The spine's worker stderr, under the walker-shaped first line.
    ("solve: CompletionFailed: model completion failed (exit -9): dsh: reasoning:\n"
     "worker: model completion failed (exit -9): dsh: reasoning:", None, "killed"),
    ("solve: CompletionFailed: model completion failed (exit -15): ", None, "killed"),
    ("solve: CompletionFailed: model completion failed (exit 1): dsh: worker error: Product "
     "subagent failure (product: Claude Code; stage: query-run; category: unknown; "
     "signal: SIGKILL): killed", None, "killed"),
    ("work: OpOutcomeUnknown: op 'work' child exited before reporting its outcome", None,
     "outcome unknown"),
    ("solve: CancelledError: model completion cancelled; not retried", None, "stopped"),
    # A killed session is retried as a crash; spent, its item has no answer.
    (None, "solve crashed 3 times; its retry is spent. Last error: model completion failed "
     "(exit -9): ", "killed"),
    # The candidate's own failures.
    ("solve: CompletionFailed: model completion failed (exit -6): ", None, None),
    ("solve: CompletionFailed: model completion failed (exit -11): ", None, None),
    (None, "solve crashed 3 times; its retry is spent. Last error: model completion failed "
     "(exit -6): ", None),
    ("solve: CompletionTimeout: model completion timed out after 600s; requested runtime "
     "shutdown; the client does not replay a deadline", None, None),
    ("solve: CompletionFailed: model completion failed (exit 1): dsh: worker error: Product "
     "subagent failure (product: Claude Code; stage: query-run; category: invalid-result; "
     "exit code: 1): the answer was empty", None, None),
    ("solve: CompletionFailed: model completion failed (exit 1): dsh: max-tokens: the model "
     "stopped at its output limit", None, None),
    (None, "solve crashed 3 times; its retry is spent. Last error: model completion failed "
     "(exit 1): dsh: worker error: Product subagent failure (category: invalid-result)", None),
    (None, None, None),
])
def test_an_item_died_with_the_run_only_when_its_process_did(text, missing, cause):
    """Which failed items are interruptions: a wire that kept failing, a process killed by
    SIGKILL or SIGTERM, an op child that never reported, a cancellation. Never another
    signal, a timeout, a product's own failure or an output limit."""
    record: dict[str, Any] = {"id": "t1", "passed": False, "score": 0.0, "error_text": text}
    if missing is not None:
        record["fail_reason"] = "no answer: solve crashed in all 3 sessions"
        record["missing_steps"] = {"solve": {"cause": "crash", "sessions": 3, "error": missing}}
    assert interrupted_item(record) == cause


def test_an_item_the_task_graded_is_never_interrupted():
    """A graded item is its result even when a step it went on without was killed (a
    fork's other branch answered): only a record with no verdict reads its missing steps."""
    killed = {"solve_b": {"cause": "crash", "sessions": 3,
                          "error": "Last error: model completion failed (exit -9): "}}
    graded = {"id": "t1", "passed": True, "score": 1.0, "error_text": None, "fail_reason": None,
              "missing_steps": killed}
    assert interrupted_item(graded) is None
    assert interrupted_item({**graded, "passed": False, "score": 0.0,
                             "fail_reason": "the answer was 7"}) is None
    assert interrupted_item({**graded, "passed": False, "score": 0.0,
                             "fail_reason": "no answer: solve_b crashed in all 3 sessions"}) == "killed"
