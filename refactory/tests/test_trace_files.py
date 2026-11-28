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


def _without_timings(markdown: str) -> str:
    """report.md with its timings set aside, whatever their digits: a number with a
    seconds unit (``0.95s``, ``12s``) and every cell of a column headed in seconds
    (``wall-clock s``). A count of decimals is not a format: ``round`` prints 0.95."""
    markdown = re.sub(r"(?<![\w.])\d+(?:\.\d+)?(?:e-?\d+)?s\b", "<n>s", markdown)
    lines, seconds = [], set()
    for line in markdown.splitlines():
        cells = line.split("|")
        if not line.startswith("|"):
            seconds = set()
        elif not seconds:
            seconds = {i for i, cell in enumerate(cells) if cell.strip().endswith(" s")}
        elif not set(line) <= set("|-: "):
            line = "|".join("<n>" if i in seconds else cell for i, cell in enumerate(cells))
        lines.append(line)
    return "\n".join(lines)


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
        # A request's "Time per item" is a measured wall time too (said only from 1 s up).
        text = re.sub(r"## Time per item\n[^\n]*\n\n?", "", text)
        return re.sub(r'"(\w+_s|started|finished|ts|time)": [^,}\]]+',
                      r'"\1": 0', text)

    assert on.seen and [same(s, on_root) for s in on.seen] == [same(s, off_root) for s in off.seen]
    assert same((on_root / JOURNAL).read_text(), on_root) == same(
        (off_root / JOURNAL).read_text(), off_root)
    assert same((on_root / "report.json").read_text(), on_root) == same(
        (off_root / "report.json").read_text(), off_root)
    on_md, off_md = (_without_timings(same((root / "report.md").read_text(), root))
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
    assert len(caplog.records) == 1 and "could not write 2 of c1's train item traces" in caplog.text


def test_one_bad_item_leaves_the_others_written(tmp_path, caplog, monkeypatch):
    """A lone surrogate in a reply is written escaped; an item whose write fails part
    way is skipped without an item.json (written last), and the rest are written."""
    bad = {"id": "bad", "exchanges": [{"node": "n", "prompt": "p", "response": "r"}],
           "steps": [{"node": "s", "kind": "op", "output": "o"}]}
    odd = {"id": "odd", "exchanges": [{"node": "n", "prompt": "p", "response": "x \udc80 y"}]}
    fine = {"id": "fine", "exchanges": [{"node": "n", "prompt": "p", "response": "ok"}]}
    dump = traces._dump

    def failing(path, value):  # type: ignore[no-untyped-def]
        if path.parent.name == "bad" and path.name == "steps.json":
            raise OSError("disk full")
        dump(path, value)

    monkeypatch.setattr(traces, "_dump", failing)
    with caplog.at_level("WARNING"):
        traces.write_evaluation(tmp_path, "c1", "train", "", 1, [bad, odd, fine])
    folder = tmp_path / traces.TRACES_DIR / "c1" / "train-seed1"
    assert len(caplog.records) == 1 and "could not write 1 of c1's train item traces" in caplog.text
    assert not (folder / "bad" / "item.json").exists()
    assert json.loads((folder / "odd" / "01-n.json").read_text())["response"] == "x \udc80 y"
    assert (folder / "odd" / "item.json").exists() and (folder / "fine" / "item.json").exists()


def test_repeats_and_occasions_get_their_own_folders():
    assert traces.occasion_name("train", "train_epoch_3", 44) == "train-epoch_3-seed44"
    assert traces.occasion_name("val", "val_0", None) == "val-0"
    assert traces.occasion_name("holdout", "", 7) == "holdout-seed7"
    assert traces.occasion_name("val", "val_1", 2, repeat=2) == "val-1-seed2-repeat2"
    assert traces.occasion_name("train", "", 3, number=2) == "train-seed3-eval2"


def _probe_loop(root: Path, **config: Any) -> Loop:
    with _patch_mode(TraceProbe):
        loop = Loop("trace_probe/probe", LoopConfig(
            epochs=1, batch_size=1, optimizer="random", seed=3, runs_root=root, holdout=0.0,
            **config), project=Path("."), model=_Model())
        loop.init()
    return loop


def test_two_train_rollouts_at_one_seed_keep_both_traces(tmp_path, monkeypatch):
    """Each evaluation the journal records gets its own folder, counted again on
    resume; an attempt it does not record (here: the run dies before the score is
    journaled) rewrites its own folder rather than taking a new one."""
    root = tmp_path / "run"
    loop = _probe_loop(root)
    seed = loop.population[0]
    with _patch_mode(TraceProbe):
        loop._evaluate(seed, seed=3)  # noqa: SLF001
        loop._evaluate(seed, seed=3)  # noqa: SLF001 - SkillOpt's slow update does this
    base = root / traces.TRACES_DIR / seed.id
    assert sorted(p.name for p in base.iterdir()) == ["train-seed3", "train-seed3-eval2"]
    scores = [e for e in _journal(root) if e["event"] == "score" and e["id"] == seed.id]
    assert len(scores) == 2  # the journal keeps both, and so do the traces

    # An attempt that never reaches the journal: its folder is the next number's.
    journal = loop._journal

    def dies(event, **payload):  # type: ignore[no-untyped-def]
        if event == "train_records":
            raise RuntimeError("killed")
        journal(event, **payload)

    monkeypatch.setattr(loop, "_journal", dies)
    with _patch_mode(TraceProbe), pytest.raises(RuntimeError):
        loop._evaluate(seed, seed=3)  # noqa: SLF001
    monkeypatch.setattr(loop, "_journal", journal)
    assert (base / "train-seed3-eval3").is_dir()
    with _patch_mode(TraceProbe):
        loop._evaluate(seed, seed=3)  # noqa: SLF001 - the same evaluation, done again
    assert sorted(p.name for p in base.iterdir()) == [
        "train-seed3", "train-seed3-eval2", "train-seed3-eval3"]

    # A resume counts the recorded evaluations from the journal.
    resumed = _probe_loop(root, resume=True)
    with _patch_mode(TraceProbe):
        resumed._evaluate(resumed.population[0], seed=3)  # noqa: SLF001
    assert (base / "train-seed3-eval4").is_dir()
    # A fresh run in the same directory starts the traces over, as it does the journal.
    fresh = _probe_loop(root)
    assert not (root / traces.TRACES_DIR).exists()
    with _patch_mode(TraceProbe):
        fresh._evaluate(fresh.population[0], seed=3)  # noqa: SLF001
    assert sorted(p.name for p in base.iterdir()) == ["train-seed3"]


def test_a_gated_run_writes_a_folder_per_evaluation(tmp_path):
    """With val, test and the gate's fresh-evidence re-scores: train, re-score, val
    and test folders, each named as the docs say."""
    parts = tmp_path / "parts.json"
    parts.write_text(json.dumps({"train": ["set:1", "set:2"], "val": ["a/b:3"],
                                 "test": [".."]}))
    root = tmp_path / "run"
    recorder = _Recorder()
    with _patch_mode(TraceProbe):
        loop = Loop("trace_probe/probe", LoopConfig(
            epochs=3, batch_size=1, optimizer="random", seed=3, runs_root=root,
            allowed_ops=("set_knob",), holdout=0.0, val_gate=True, selection="single",
            parts_manifest=str(parts)), project=Path("."), model=_Model())
        loop.optimizer = recorder
        loop.run()
    # The gate's train rollouts use SkillOpt's per-epoch seed (1004, 2004, ...).
    kinds = {candidate.name: sorted(re.sub(r"\d+", "N", p.name) for p in candidate.iterdir())
             for candidate in (root / traces.TRACES_DIR).iterdir()}
    evaluated = [c.id for c in loop.population if c.evaluated]
    assert sorted(kinds) == sorted(evaluated)
    for candidate, folders in kinds.items():
        assert "train-seedN" in folders and "val-N-seedN" in folders, (candidate, folders)
        assert set(folders) <= {"train-seedN", "val-N-seedN", "train-epoch_N-seedN",
                                "test-N-seedN"}, folders
    every = [folder for folders in kinds.values() for folder in folders]
    assert "train-epoch_N-seedN" in every and "test-N-seedN" in every, kinds
    # Each folder holds only its part's items.
    for folder in (root / traces.TRACES_DIR).glob("*/*"):
        names = {p.name for p in folder.iterdir()}
        part = folder.name.split("-")[0]
        expected = {"train": {"set:1", "set:2"}, "val": {"a/b:3"}, "test": {".."}}[part]
        assert names <= {traces.safe_name(i) for i in expected}, (folder, names)


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


#: The words that name the trace files, as a string literal would hold them.
NAMES = re.compile(r"\btraces\b|\bitem\.json\b|\bsteps\.json\b")

#: String literals outside the writer that hold one of ``NAMES`` today, and are not the
#: trace files: prose ("the traces" the optimizer reads) or a Task's own ``item.json`` in
#: a pass. Counted per file, so a new literal anywhere fails the test.
OTHER_USES = {
    ("refactory_dsh/nn/executor/items.py", "item.json"): 2,
    ("refactory_dsh/optim/request/context.py", "traces"): 1,
    ("refactory_dsh/optim/request/objective.py", "traces"): 1,
    # Prompt prose, no file names: "both full traces", split-evidence #655; #705's memory
    # prompt ("from the comparison and the traces"), the rejected-change comparison ("their
    # traces are not shown"), MEMORY_WEIGHT ("the current traces hold") and the record's
    # rules ("where in the pipeline the traces show it starts", "what the traces show").
    ("refactory_dsh/optim/memory.py", "traces"): 12,
    # The analysts' summary: "with quotes from the traces".
    ("refactory_dsh/optim/analysts.py", "traces"): 1,
    ("refactory_dsh/tools/gen_tutorial.py", "traces"): 2,
    ("refactory_packages/java_build_pack/mode.py", "item.json"): 2,
    ("refactory_packages/java_build_pack/ops.py", "item.json"): 1,
    ("refactory_packages/java_build_pack/tasks.py", "item.json"): 1,
    ("refactory_packages/loop_mode/mode.py", "traces"): 1,
}


def _literal_names(source: str) -> list[str]:
    """``NAMES`` in every string literal of ``source`` (an f-string's text parts
    included), docstrings aside: a docstring reads no file."""
    import ast

    tree = ast.parse(source)
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                       ast.AsyncFunctionDef))
                  and node.body and isinstance(node.body[0], ast.Expr)
                  and isinstance(node.body[0].value, ast.Constant)}
    return [match.group() for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in docstrings for match in NAMES.finditer(node.value)]


def test_nothing_but_the_writer_names_the_trace_files():
    """Write-only: outside `refactory_dsh.traces`, no string literal of the Python
    package names the folder or its file names (beyond ``OTHER_USES``), no module but
    the loop package imports the writer and the loop only calls it, and the TypeScript sources
    do not name them at all."""
    from collections import Counter

    package = Path(traces.__file__).resolve().parent
    top = package.parent
    found: Counter[tuple[str, str]] = Counter()
    importers = []
    for path in (p for r in (package, top / "refactory_packages") if r.is_dir()
                 for p in r.rglob("*.py")):
        name = str(path.relative_to(top))
        if name == "refactory_dsh/traces.py":
            continue
        source = path.read_text()
        found.update((name, word) for word in _literal_names(source))
        if re.search(r"TRACES_DIR|\bimport traces\b|refactory_dsh\.traces\b", source):
            importers.append(name)
    assert dict(found) == OTHER_USES, dict(found)
    assert importers == ["refactory_dsh/loop/evaluate.py", "refactory_dsh/loop/resume.py"], \
        importers
    loop_source = "".join(p.read_text() for p in sorted((package / "loop").glob("*.py")))
    assert "TRACES_DIR" not in loop_source
    calls = re.findall(r"\btraces\.(\w+)", loop_source)
    assert sorted(calls) == ["clear", "write_evaluation"], calls

    repo = top.parent
    sources = [p for p in (repo / "packages").glob("**/src/**/*")
               if p.suffix in (".ts", ".mts", ".cts", ".tsx", ".mjs", ".js")
               and "node_modules" not in p.parts and "generated" not in p.name]
    assert sources, "no TypeScript sources found"
    named = {}
    for path in sources:
        code = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)
        if NAMES.search(code):
            named[str(path.relative_to(repo))] = sorted(set(NAMES.findall(code)))
    assert named == {}, named


def test_the_literal_scan_sees_paths_and_f_strings():
    """The scan of the test above catches the ways a reader would name the files."""
    assert _literal_names('p = root / "traces" / cid') == ["traces"]
    assert _literal_names('p = f"{root}/traces/{cid}/item.json"') == ["traces", "item.json"]
    assert _literal_names('p = Path("x", "steps.json")') == ["steps.json"]
    assert _literal_names('def f():\n    """Reads the traces."""\n') == []
