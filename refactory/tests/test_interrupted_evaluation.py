"""An evaluation whose items died with the run's own processes is not a result.

On 2026-11-12 a host out-of-memory kill took out an outer-loop run's processes while a
candidate's train evaluation was in flight: every item failed, the candidate was
recorded as ``failed: data: all 35 items failed``, and the resumed run never evaluated
it again. Now an item that died with a process of the run -- the wire to the model kept
failing, the completion's process was killed, the run's stop cancelled it -- has an
unknown outcome (`items.interrupted_item`); an evaluation with one is journaled as
interrupted and stops the run, nothing of it is recorded, and ``--resume`` evaluates
the candidate again and gates it. The interrupted attempt's per-item receipts are
moved to the run's archive, not deleted. A model's own failure (a wrong answer, a
timeout) is still the candidate's result.

Each run here is the real loop and walker over a scripted ``dsh`` reached through the
real model client. Each test fails when the behaviour it names is reverted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import nn
from refactory_dsh import task as task_module
from refactory_dsh.items import finished_pass_record, interrupted_item
from refactory_dsh.loop import EvaluationInterrupted, Loop, LoopConfig
from refactory_dsh.model import DshModelClient
from refactory_dsh.optim import Candidate

from test_skillopt import MODE_ID, TEST_IDS, TRAIN_IDS, VAL_IDS, _patch_mode, _write_parts
from test_val_gate import ScriptedLevels, _journal_events, _level

TASK_REF = "interrupted_evaluation_task"

#: A scripted ``dsh``: the plan maps keys to one behaviour -- ``KILL`` (the process is
#: killed, as the out-of-memory killer kills it), ``SLEEP`` (it outlives its wall),
#: ``dsh: ...`` (a failure with that diagnostic), or an answer. The first key whose
#: ``&``-separated parts are all in the prompt or the call's folder (a pass of
#: ``occasions/val_0``, say) decides.
FAKE_DSH = '''import json, os, pathlib, signal, sys, time
root = pathlib.Path({root!r})
plan = json.loads((root / "plan.json").read_text())
seen = os.getcwd() + "\\n" + sys.stdin.read()
step = next(value for key, value in plan.items() if all(part in seen for part in key.split("&")))
if step == "KILL":
    os.kill(os.getpid(), signal.SIGKILL)
if step == "SLEEP":
    time.sleep(30)
if step.startswith("dsh:"):
    print(step, file=sys.stderr)
    sys.exit(1)
print(step)
'''

#: The model client's wire failure, as `dsh` reports a proxy that is gone.
TRANSPORT = "dsh: TRANSPORT: connect ECONNREFUSED 127.0.0.1:8082"


class AnswerTask(task_module.Task):
    """The train, val and test ids of the toy split; an item passes when its pass
    answered ``right``."""

    name = TASK_REF

    def instances(self, *, part: str | None = None, seed: int | None = None) -> list[task_module.TaskInstance]:
        return [task_module.TaskInstance(id=item) for item in TRAIN_IDS + VAL_IDS + TEST_IDS]

    def verify(self, instance, workspace, *, part=None, seed=None):  # type: ignore[no-untyped-def]
        answer = Path(workspace) / "answer.txt"
        right = answer.exists() and answer.read_text().strip() == "right"
        return task_module.VerifyResult(passed=right, score=1.0 if right else 0.0)


task_module.register_task(TASK_REF, AnswerTask)


class AskingProbe(nn.Mode):
    """One agent step per item, asked at the candidate's ``level``."""

    name = "asking_probe"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {"level": nn.Knob(default=5, kind="int", lo=0, hi=9, description="a level")}

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node("data", "data", subgraph_entry="solve", subgraph_exit="solve",
                          task_ref=TASK_REF, writes=("current_item.json",),
                          produces=("item_scores",), parallelism=4))
        graph.add(nn.Node("solve", "agent", prompt_template="Answer at level {{level}}.",
                          writes=("answer.txt",), tool_policy="none", role="solver",
                          knobs=("level",), timeout_s=5))
        graph.add(nn.Node("finish", "op", call="loop_ops:write_best",
                          writes=("best.json",), produces=("best",)))
        graph.connect("data", "finish")
        return graph

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0


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
        """Each ``level<n>``'s behaviour; ``only`` maps ``"<folder part>&level <n>."``
        keys to a behaviour that comes first."""
        (self.root / "plan.json").write_text(json.dumps({**(only or {}), **{
            f"level {level.removeprefix('level')}.": step for level, step in levels.items()}}))

    def client(self) -> DshModelClient:
        return DshModelClient(worker_backend="dsh", provider="p", model="m", dsh_cli=str(self.cli),
                              attempts=1, timeout_s=60)


@pytest.fixture
def one_seed(monkeypatch):  # type: ignore[no-untyped-def]
    """The scripted plans start from the seed alone."""
    monkeypatch.setattr(Loop, "_seed_population", lambda self, seed: None)


def _loop(tmp_path: Path, dsh: Dsh, plan: list[list[tuple[int, int]]], **overrides: Any) -> Loop:
    config = dict(epochs=2, batch_size=1, optimizer="random", seed=7, runs_root=tmp_path / "run",
                  allowed_ops=("set_knob",), max_evaluations=60, holdout=0.0, val_gate=True,
                  parts_manifest=str(_write_parts(tmp_path, test=True)))
    config.update(overrides)
    loop = Loop(MODE_ID, LoopConfig(**config), project=Path("."), model=dsh.client())
    loop.optimizer = ScriptedLevels(plan)
    return loop


def _child(loop: Loop) -> Candidate:
    return next(c for c in loop.population if _level(c) == 6)


@pytest.mark.parametrize("death", ["transport", "killed"])
def test_an_interrupted_evaluation_is_evaluated_again_on_resume(tmp_path, one_seed, death):
    """The child's train run dies with the run's processes: its model calls fail on the
    wire (the proxy is gone), or their processes are killed (retried as crashes, then
    spent). The run stops on it with nothing recorded; the resume evaluates the child
    from the start, scores it normally and gates it."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan(level5="wrong", level6=TRANSPORT if death == "transport" else "KILL")
    with _patch_mode(AskingProbe):
        loop = _loop(tmp_path, dsh, [[(5, 6)], []])
        with pytest.raises(EvaluationInterrupted, match="resume the run"):
            loop.run()
    child = _child(loop)
    assert not child.evaluated and child.score is None and child.error is None
    assert child.id not in {r["id"] for r in _journal_events(loop, "score")}, \
        "an interrupted evaluation is never recorded as a score"
    assert child.id not in {r["id"] for r in _journal_events(loop, "train_records")}
    [interrupted] = _journal_events(loop, "evaluation_interrupted")
    assert (interrupted["id"], interrupted["part"]) == (child.id, "train")
    assert interrupted["causes"] == {death: len(TRAIN_IDS)}
    passes = loop.root / "runs" / child.id / "passes" / "data"
    receipts = sorted(path.relative_to(passes) for path in passes.rglob("*") if path.is_file())
    assert receipts, "setup: the interrupted pass left per-item receipts"

    dsh.plan(level5="wrong", level6="right")
    with _patch_mode(AskingProbe):
        resumed = _loop(tmp_path, dsh, [[]], resume=True)
        resumed.run()
    child = _child(resumed)
    assert child.evaluated and child.exit_ok and child.raw_score == pytest.approx(1.0)
    assert [r["raw_score"] for r in _journal_events(resumed, "score") if r["id"] == child.id] == [1.0]
    assert child.id in resumed._gate_accepted, "the re-evaluated child is gated on its val score"
    assert resumed._gate().cache[child.id] == pytest.approx(1.0)

    # The interrupted attempt's receipts are in the run's archive, byte for byte.
    [archived] = _journal_events(resumed, "evaluation_archived")
    assert archived["id"] == child.id and archived["workdir"] == f"runs/{child.id}"
    kept = resumed.root / archived["archive"] / "passes" / "data"
    assert sorted(path.relative_to(kept) for path in kept.rglob("*") if path.is_file()) == receipts
    done = [finished_pass_record(path.parent) for path in kept.rglob(".done")]
    assert len(done) == len(TRAIN_IDS) and all(interrupted_item(r) == death for r in done)


def test_a_run_stopped_mid_pass_keeps_its_receipts_and_evaluates_again(tmp_path, one_seed,
                                                                        monkeypatch):
    """The run's own process is stopped (SIGTERM becomes SystemExit) once half of the
    child's model calls were made, so some items finished and the rest never did. The resume evaluates the child from the start, and the
    finished items' receipts from before the stop are in the archive, not deleted."""
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
    passes = loop.root / "runs" / child.id / "passes" / "data"
    receipts = sorted(path.relative_to(passes) for path in passes.rglob(".done"))
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
    assert sorted(path.relative_to(kept) for path in kept.rglob(".done")) == receipts
    assert all(finished_pass_record(path.parent)["score"] == 1.0 for path in kept.rglob(".done"))


def test_an_interrupted_val_rollout_is_no_val_score(tmp_path, one_seed):
    """The child's train run is scored; its val rollout dies on the wire. No val score
    is recorded and the gate judges nothing from it; the resume rolls it out again in a
    fresh occasion (the interrupted one stays) and gates the child on it."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan(only={"occasions/val_&level 6.": TRANSPORT}, level5="wrong", level6="right")
    with _patch_mode(AskingProbe):
        loop = _loop(tmp_path, dsh, [[(5, 6)], []])
        with pytest.raises(EvaluationInterrupted, match="val evaluation was interrupted"):
            loop.run()
    child = _child(loop)
    assert child.evaluated and child.raw_score == pytest.approx(1.0), "its train run was a result"
    assert child.id not in {r["id"] for r in _journal_events(loop, "part_result")}
    assert child.id not in loop._gate_accepted + loop._gate_rejected
    [interrupted] = _journal_events(loop, "evaluation_interrupted")
    assert (interrupted["id"], interrupted["part"], interrupted["causes"]) == (
        child.id, "val", {"transport": len(VAL_IDS)})

    dsh.plan(level5="wrong", level6="right")
    with _patch_mode(AskingProbe):
        resumed = _loop(tmp_path, dsh, [[]], resume=True)
        resumed.run()
    vals = [r["result"] for r in _journal_events(resumed, "part_result")
            if r["id"] == child.id and r["part"] == "val"]
    assert [(v["occasion"], v["score"]) for v in vals] == [(1, 1.0)]
    assert child.id in resumed._gate_accepted
    assert (resumed.root / interrupted["workdir"] / "passes").is_dir(), "the interrupted occasion stays"


def test_an_interrupted_rescore_is_no_evidence_and_is_kept(tmp_path, one_seed):
    """Fresh evidence: the parent is re-scored beside its child, and the re-score dies.
    Its records are not replaced, nothing counts it as a re-score, and the run stops.
    When it is run again (a resume in that epoch tries it again) its interrupted passes
    are moved to the archive."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan(only={"train_epoch_&level 5.": TRANSPORT}, level5="wrong", level6="right")
    with _patch_mode(AskingProbe):
        loop = _loop(tmp_path, dsh, [[(5, 6)], []])
        with pytest.raises(EvaluationInterrupted, match="train evaluation was interrupted"):
            loop.run()
    parent = loop.population[0]
    [interrupted] = _journal_events(loop, "evaluation_interrupted")
    assert (interrupted["id"], interrupted.get("rescore_epoch")) == (parent.id, 2)
    assert _journal_events(loop, "rescore") == _journal_events(loop, "rescore_failed") == []
    assert [r["id"] for r in _journal_events(loop, "train_records")] == [parent.id, _child(loop).id], \
        "the parent keeps its own records"
    left = loop.root / interrupted["workdir"]
    receipts = sorted(path.relative_to(left) for path in left.rglob(".done"))
    assert len(receipts) == len(TRAIN_IDS)

    dsh.plan(level5="wrong", level6="right")
    with _patch_mode(AskingProbe):
        resumed = _loop(tmp_path, dsh, [[]], resume=True)
        resumed.init()
        assert resumed._rescore(resumed.population[0], 2)
    [archived] = _journal_events(resumed, "evaluation_archived")
    assert (archived["id"], archived["workdir"]) == (parent.id, interrupted["workdir"])
    kept = resumed.root / archived["archive"]
    assert sorted(path.relative_to(kept) for path in kept.rglob(".done")) == receipts
    assert [r["epoch"] for r in _journal_events(resumed, "rescore")] == [2]


@pytest.mark.parametrize("answer", ["garbage", "SLEEP"])
def test_a_models_own_failure_is_still_the_candidates_result(tmp_path, one_seed, answer):
    """Every item answers wrongly, or every model call outlives its wall: the child is
    scored 0, or recorded as failed, as before, and the run goes on."""
    dsh = Dsh(tmp_path / "dsh")
    dsh.plan(level5="right", level6=answer)
    with _patch_mode(AskingProbe):
        loop = _loop(tmp_path, dsh, [[(5, 6)], []])
        loop.run()
    child = _child(loop)
    assert child.evaluated, "a model's failure is a result"
    assert _journal_events(loop, "evaluation_interrupted") == []
    [score] = [r for r in _journal_events(loop, "score") if r["id"] == child.id]
    if answer == "garbage":
        assert (score["raw_score"], score["exit_ok"]) == (0.0, True)
    else:
        assert (score["score"], score["exit_ok"]) == (None, False)
        assert score["error"] == f"data: all {len(TRAIN_IDS)} items failed"
    assert child.id in loop._gate_rejected


@pytest.mark.parametrize(("text", "missing", "cause"), [
    ("solve: CompletionFailed: model completion failed on the wire 3 time(s); last cause: "
     "dsh: TRANSPORT: fetch failed", None, "transport"),
    ("solve: CompletionFailed: model completion failed on the wire 2 time(s) and the node's "
     "time ran out before it could ask again; last cause: econnreset", None, "transport"),
    # The spine's worker stderr, under the walker-shaped first line.
    ("solve: CompletionFailed: model completion failed (exit -9): dsh: reasoning:\n"
     "worker: model completion failed (exit -9): dsh: reasoning:", None, "killed"),
    ("solve: CompletionFailed: model completion failed (exit 1): dsh: worker error: Product "
     "subagent failure (product: Claude Code; stage: query-run; category: unknown; "
     "signal: SIGKILL): killed", None, "killed"),
    ("work: OpOutcomeUnknown: op 'work' child exited before reporting its outcome", None,
     "outcome unknown"),
    ("solve: CancelledError: model completion cancelled; not retried", None, "stopped"),
    # A killed session is retried as a crash; spent, its item has no answer.
    (None, "solve's crash retry is spent. Last error: model completion failed (exit -9): ",
     "killed"),
    # The model's own failures.
    ("solve: CompletionTimeout: model completion timed out after 600s; requested runtime "
     "shutdown; the client does not replay a deadline", None, None),
    ("solve: CompletionFailed: model completion failed (exit 1): dsh: worker error: Product "
     "subagent failure (product: Claude Code; stage: query-run; category: invalid-result; "
     "exit code: 1): the answer was empty", None, None),
    ("solve: CompletionFailed: model completion failed (exit 1): dsh: max-tokens: the model "
     "stopped at its output limit", None, None),
    (None, "solve's crash retry is spent. Last error: model completion failed (exit 1): "
     "dsh: worker error: Product subagent failure (category: invalid-result)", None),
    (None, None, None),
])
def test_an_item_died_with_the_run_only_when_its_process_did(text, missing, cause):
    """Which failed items are interruptions: a wire that kept failing, a killed process,
    an op child that never reported, a cancellation. Never a timeout, a product's own
    failure or an output limit, and never an item that answered."""
    record: dict[str, Any] = {"id": "t1", "passed": False, "score": 0.0, "error_text": text}
    if missing is not None:
        record["missing_steps"] = {"solve": {"cause": "crash", "sessions": 3, "error": missing}}
    assert interrupted_item(record) == cause
