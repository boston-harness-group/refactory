"""Each evaluated item's full trace is written as files under the run's ``traces/``
folder (`refactory_dsh.traces`), on both runtimes, and nothing reads them.

The files are write-only for now: the journal, the optimizer's requests and the
report stay what they were, and no module of the package but the writer names the
folder. Each test fails when the writer is turned off or starts to be read.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import items as items_module
from refactory_dsh import nn, task as task_module, traces
from refactory_dsh.loop import JOURNAL, Loop, LoopConfig
from refactory_dsh.optim import Optimizer, Proposal

sys.path.insert(0, str(Path(__file__).parent))
from test_skillopt import _patch_mode  # noqa: E402

TASK_REF = "trace_files_probe"
#: Ids with the characters a folder name cannot hold as they are.
ITEM_IDS = ["set:1", "set:2", "a/b:3", ".."]
#: A reply longer than any preview, so a cut one would show.
LONG = "reasoning " * 900


class TraceTask(task_module.Task):
    name = TASK_REF

    def instances(self, *, part: str | None = None, seed: int | None = None):
        return [task_module.TaskInstance(id=item_id, metadata={"question": f"q of {item_id}"})
                for item_id in ITEM_IDS]

    def verify(self, instance, workspace, *, part=None, seed=None):  # type: ignore[no-untyped-def]
        passed = instance.id.endswith("1")
        return task_module.VerifyResult(passed=passed, score=1.0 if passed else 0.0)


task_module.register_task(TASK_REF, TraceTask)


class TraceProbe(nn.Mode):
    """A data pass of two model calls and an op, and one knob a sampler can move."""

    name = "trace_probe"

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {"level": nn.Knob(default=5, kind="int", lo=0, hi=9, description="a level")}

    def build(self) -> nn.Graph:
        graph = nn.Graph()
        graph.add(nn.Node("data", "data", subgraph_entry="think", subgraph_exit="work",
                          task_ref=TASK_REF, writes=("current_item.json",),
                          produces=("item_scores",)))
        graph.add(nn.Node("think", "agent", prompt_template="think it through",
                          writes=("think.txt",)))
        graph.add(nn.Node("answer", "agent", prompt_template="now answer", writes=("answer.txt",)))
        graph.add(nn.Node("work", "op", call="loop_ops:record_item", reads=("current_item.json",),
                          writes=("seen.json",), produces=("seen",), knobs=("level",)))
        graph.add(nn.Node("finish", "op", call="loop_ops:write_best",
                          writes=("best.json",), produces=("best",)))
        graph.connect("think", "answer")
        graph.connect("answer", "work")
        graph.connect("data", "finish")
        return graph

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0


class _Model:
    """Answers each model call with a long reply naming the prompt's first line."""

    thread_safe = True

    def complete(self, prompt: str, *, system: str = "", **_: Any) -> str:
        return f"{prompt.splitlines()[0] if prompt else ''} :: {LONG}"

    def for_node(self, node_id: str, role: str | None = None) -> "_Model":
        return self

    def with_wall(self, seconds: float) -> "_Model":
        return self

    def with_tool_policy(self, policy: Any) -> "_Model":
        return self

    def route(self) -> dict[str, Any]:
        return {"worker_backend": "dsh"}


class _Recorder(Optimizer):
    """Records every request it is shown and moves the knob."""

    name = "recorder"

    def __init__(self) -> None:
        super().__init__(k=1)
        self.seen: list[str] = []

    def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
        self.seen.append(ctx.render())
        return [Proposal(edit={"op": "set_knob", "name": "level", "value": len(self.seen)},
                         parent=parents[0].id)]


def _run(root: Path) -> tuple[Loop, _Recorder, dict[str, Any]]:
    recorder = _Recorder()
    with _patch_mode(TraceProbe):
        loop = Loop("trace_probe/probe", LoopConfig(
            epochs=2, batch_size=1, optimizer="random", seed=3, runs_root=root,
            allowed_ops=("set_knob",), holdout=0.0), project=Path("."), model=_Model())
        loop.optimizer = recorder
        report = loop.run()
    return loop, recorder, report


def _journal(root: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (root / JOURNAL).read_text().splitlines()]


def test_a_walker_run_writes_every_items_full_trace(tmp_path):
    root = tmp_path / "run"
    loop, _recorder, _report = _run(root)
    records = {event["id"]: event["records"] for event in _journal(root)
               if event["event"] == "train_records"}
    assert loop.population[0].id in records and len(records) >= 2
    for candidate, recorded in records.items():
        folder = root / traces.TRACES_DIR / candidate / f"train-seed{loop.config.seed}"
        assert sorted(path.name for path in folder.iterdir()) == sorted(
            traces.safe_name(r["id"]) for r in recorded)
        for record in recorded:
            item = folder / traces.safe_name(record["id"])
            meta = json.loads((item / "item.json").read_text())
            assert meta["id"] == record["id"] and "exchanges" not in meta and "steps" not in meta
            assert meta == {k: v for k, v in record.items() if k not in ("exchanges", "steps")}
            calls = sorted(item.glob("[0-9]*-*.json"))
            assert [p.name for p in calls] == ["01-think.json", "02-answer.json"]
            exchanges = [json.loads(p.read_text()) for p in calls]
            assert exchanges == record["exchanges"]
            assert [e["node"] for e in exchanges] == ["think", "answer"]
            assert "think it through" in exchanges[0]["prompt"]
            assert exchanges[1]["response"].endswith(LONG)  # whole, not a preview
            steps = json.loads((item / "steps.json").read_text())
            assert steps == record["steps"] and [s["node"] for s in steps] == ["work"]
    # The ids a folder cannot hold as they are get distinct, safe names.
    names = {traces.safe_name(i) for i in ITEM_IDS}
    assert len(names) == len(ITEM_IDS)
    assert all("/" not in n and ":" not in n and n not in (".", "..") for n in names)


def test_the_writer_changes_no_request_journal_or_report(tmp_path, monkeypatch):
    """The same run with the writer turned off: the optimizer's requests, the
    journal and the report are the same, byte for byte once the run root and the
    timings are set aside."""
    on_root, off_root = tmp_path / "on" / "run", tmp_path / "off" / "run"
    _loop, on, _report = _run(on_root)
    monkeypatch.setattr(traces, "write_evaluation", lambda *a, **k: None)
    _loop, off, _report = _run(off_root)
    assert (on_root / traces.TRACES_DIR).is_dir() and not (off_root / traces.TRACES_DIR).exists()

    def same(text: str, root: Path) -> str:
        text = re.sub(r"\b1[6-9]\d{8}(?:\.\d+)?\b", "<clock>", text.replace(str(root), "<root>"))
        return re.sub(r'"(\w+_s|started|finished|ts|time)": [^,}\]]+',
                      r'"\1": 0', text)

    assert on.seen and [same(s, on_root) for s in on.seen] == [same(s, off_root) for s in off.seen]
    assert same((on_root / JOURNAL).read_text(), on_root) == same(
        (off_root / JOURNAL).read_text(), off_root)
    assert same((on_root / "report.json").read_text(), on_root) == same(
        (off_root / "report.json").read_text(), off_root)
    # report.md's tables show timings as bare numbers: those with many decimals.
    on_md, off_md = (re.sub(r"\d+\.\d{3,}", "<n>", same((root / "report.md").read_text(), root))
                     for root in (on_root, off_root))
    assert on_md == off_md
    assert f"/{traces.TRACES_DIR}/" not in (on_root / JOURNAL).read_text()


def test_a_rewrite_replaces_the_files_and_drops_stale_calls(tmp_path):
    record = {"id": "x:1", "passed": True, "score": 1.0,
              "exchanges": [{"node": "a", "prompt": "p1", "response": "r1"},
                            {"node": "b", "prompt": "p2", "response": "r2"}],
              "steps": [{"node": "op", "kind": "op", "output": "o"}]}
    traces.write_evaluation(tmp_path, "c1", "val", "val_0", None, [record])
    item = tmp_path / traces.TRACES_DIR / "c1" / "val-0" / traces.safe_name("x:1")
    assert sorted(p.name for p in item.iterdir()) == ["01-a.json", "02-b.json", "item.json",
                                                      "steps.json"]
    again = {"id": "x:1", "passed": False, "score": 0.0,
             "exchanges": [{"node": "a", "prompt": "p1", "response": "r9"}]}
    traces.write_evaluation(tmp_path, "c1", "val", "val_0", None, [again])
    assert sorted(p.name for p in item.iterdir()) == ["01-a.json", "item.json"]
    assert json.loads((item / "01-a.json").read_text())["response"] == "r9"
    assert json.loads((item / "item.json").read_text())["passed"] is False


def test_a_failed_write_warns_once_and_does_not_raise(tmp_path, caplog):
    (tmp_path / traces.TRACES_DIR).write_text("a file where the folder goes")
    with caplog.at_level("WARNING"):
        traces.write_evaluation(tmp_path, "c1", "train", "", 1, [{"id": "a"}, {"id": "b"}])
    assert len(caplog.records) == 1 and "could not write c1's train traces" in caplog.text


def test_repeats_and_occasions_get_their_own_folders():
    assert traces.occasion_name("train", "train_epoch_3", 44) == "train-epoch_3-seed44"
    assert traces.occasion_name("val", "val_0", None) == "val-0"
    assert traces.occasion_name("holdout", "", 7) == "holdout-seed7"
    assert traces.occasion_name("val", "val_1", 2, repeat=2) == "val-1-seed2-repeat2"


def test_spine_records_reach_the_writer(tmp_path, monkeypatch):
    """The spine path: the records the spine writes to ``item_records.json`` (built by
    the same `items.make_item_record` its ``command_end`` calls) are read by
    ``DshEvaluator`` and written by the loop's one hook, whole."""
    from refactory_dsh import dsh_runtime
    from refactory_dsh.evaluators import DshEvaluator
    from refactory_dsh.nn.mode import RunResult

    verdicts = {i: task_module.VerifyResult(passed=i == "set:1", score=float(i == "set:1"))
                for i in ITEM_IDS}
    exchanges = [{"node": "think", "prompt": "think it through", "response": LONG},
                 items_module.op_visit(tmp_path, "work", {"seen": ["x"]}),
                 {"node": "answer", "prompt": "now answer", "response": "B " + LONG}]
    records = [items_module.make_item_record(i, verdicts[i], exchanges=exchanges,
                                             prediction="B", label="A") for i in ITEM_IDS]

    def fake_spine(mode, state, workdir, **_):  # type: ignore[no-untyped-def]
        workdir.mkdir(parents=True, exist_ok=True)
        # As the spine writes it: JSON.stringify(records, null, 2).
        (workdir / items_module.RECORDS_FILE).write_text(json.dumps(records, indent=2))
        (workdir / "facts.json").write_text(json.dumps({"item_scores": [
            {"id": r["id"], "score": r["score"]} for r in records]}))
        return RunResult(exit_ok=True, score=0.25, facts={"item_scores": [
            {"id": r["id"], "score": r["score"]} for r in records]}, workdir=workdir)

    monkeypatch.setattr(dsh_runtime, "evaluate_on_dsh", fake_spine)
    root = tmp_path / "run"
    with _patch_mode(TraceProbe):
        loop = Loop("trace_probe/probe", LoopConfig(
            epochs=1, batch_size=1, optimizer="random", seed=3, runs_root=root, holdout=0.0),
            project=Path("."))
        loop.evaluator = DshEvaluator()
        loop.init()
        seed = loop.population[0]
        loop._evaluate(seed, seed=3)  # noqa: SLF001 - the loop's own train evaluation
    folder = root / traces.TRACES_DIR / seed.id / "train-seed3"
    assert sorted(p.name for p in folder.iterdir()) == sorted(traces.safe_name(i) for i in ITEM_IDS)
    for record in records:
        item = folder / traces.safe_name(record["id"])
        assert json.loads((item / "item.json").read_text())["id"] == record["id"]
        assert [json.loads(p.read_text()) for p in sorted(item.glob("[0-9]*-*.json"))] == \
            record["exchanges"]
        assert json.loads((item / "02-answer.json").read_text())["response"] == "B " + LONG
        assert json.loads((item / "steps.json").read_text()) == record["steps"]


def test_nothing_but_the_writer_names_the_traces_folder():
    """Write-only: no module of the package but `refactory_dsh.traces` names the
    folder or its constant, and the loop only calls the writer."""
    package = Path(traces.__file__).resolve().parent
    pattern = re.compile(r"""TRACES_DIR|["']traces(?:/|["'])|/\s*["']traces["']""")
    found = {}
    roots = [package, package.parent / "refactory_packages"]
    for path in (p for r in roots if r.is_dir() for p in r.rglob("*.py")):
        hits = [n for n, line in enumerate(path.read_text().splitlines(), 1) if pattern.search(line)]
        if hits:
            found[str(path.relative_to(package.parent))] = hits
    assert set(found) == {"refactory_dsh/traces.py"}, found
    loop_source = (package / "loop.py").read_text()
    assert loop_source.count("traces.") == 1 and "traces.write_evaluation(" in loop_source
