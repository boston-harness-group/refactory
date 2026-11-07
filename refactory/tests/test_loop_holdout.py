"""The held-out audit: the search trains on one part, and the winner is examined on the rest.

The failure this exists to stop is a search that rewards a harness which only works
on the items it was tuned against, because those were the only items it was ever
scored on. The split is deterministic, it is written into the run directory before a
candidate is evaluated, and the item loader of both runtimes reads it, so no
candidate can choose its own exam.

``test_loop.py`` covers the size cost, which is the other half of the loop's own
objective; these cover what the winner is measured on afterwards.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest

from refactory_dsh import nn
from refactory_dsh import task as task_module
from refactory_dsh.loop import Loop, LoopConfig

MODE = "holdout_probe/probe"
TASK_REF = "holdout_probe_counting"
NAMES = ("alpha", "beta", "gamma", "delta")


class Counting(task_module.Task):
    """Four instances, judged on whether the pass ran on a trained id.

    The Task is where the gap this audit is about comes from: a pass it sees during
    the search is rewarded, so a harness evaluated on the train part scores 1.0 and
    the same harness on the held-out part scores 0.0.
    """

    name = TASK_REF

    def instances(self) -> list[task_module.TaskInstance]:
        return [task_module.TaskInstance(id=name) for name in NAMES]

    def verify(self, instance, workspace):  # type: ignore[no-untyped-def]
        held = {
            row["id"]
            for row in task_module.partition(
                [{"id": name} for name in NAMES], task_module.DEFAULT_HOLDOUT
            )[task_module.HOLDOUT]
        }
        passed = instance.id not in held
        return task_module.VerifyResult(passed=passed, score=1.0 if passed else 0.0)


class Single(task_module.Task):
    """One instance, which cannot be split into a train and a holdout part."""

    name = "holdout_probe_single"

    def instances(self) -> list[task_module.TaskInstance]:
        return [task_module.TaskInstance(id="only")]

    def verify(self, instance, workspace):  # type: ignore[no-untyped-def]
        return task_module.VerifyResult(passed=True, score=1.0)


def _probe_graph(task_ref: str) -> nn.Graph:
    graph = nn.Graph()
    graph.add(nn.Node(
        "data", "data", subgraph_entry="work", subgraph_exit="work",
        task_ref=task_ref, writes=("current_item.json",), produces=("item_scores",),
    ))
    graph.add(nn.Node(
        "work", "op", call="loop_ops:record_item",
        reads=("current_item.json",), writes=("seen.json",), produces=("seen",),
    ))
    graph.add(nn.Node(
        "finish", "op", call="loop_ops:write_best",
        writes=("best.json",), produces=("best",),
    ))
    graph.connect("data", "finish")
    return graph


class Probe(nn.Mode):
    """A scored mode whose one data node iterates a Task's items."""

    name = "probe"

    def build(self) -> nn.Graph:
        return _probe_graph(TASK_REF)

    def score(self, run: nn.RunResult) -> float:
        scores = [entry["score"] for entry in run.facts.get("item_scores", [])]
        return sum(scores) / len(scores) if scores else 0.0


class SingleProbe(Probe):
    """The same mode bound to a Task with one instance."""

    def build(self) -> nn.Graph:
        return _probe_graph(Single.name)


def _patch_mode(mode_class: type[nn.Mode]):  # type: ignore[no-untyped-def]
    """Point the registry at a local mode class for the duration of a test."""
    import refactory_dsh.registry as registry

    class FakePackage:
        name = "fake"
        description = ""

        def mode(self, name: str) -> type[nn.Mode]:
            return mode_class

    @contextlib.contextmanager
    def patched():  # type: ignore[no-untyped-def]
        original = registry.resolve
        registry.resolve = lambda project, identifier: (FakePackage(), mode_class)  # type: ignore[assignment]
        try:
            yield
        finally:
            registry.resolve = original  # type: ignore[assignment]

    return patched()


task_module.register_task(TASK_REF, Counting)
task_module.register_task(Single.name, Single)


def _run(tmp_path: Path, *, mode_class: type[nn.Mode] = Probe, **overrides: object):
    base: dict[str, object] = dict(
        epochs=1, batch_size=1, optimizer="random", seed=1, runs_root=tmp_path / "run",
    )
    base.update(overrides)
    with _patch_mode(mode_class):
        loop = Loop(MODE, LoopConfig(**base), project=Path("."))  # type: ignore[arg-type]
        report = loop.run()
    return loop, report


def _held_id(holdout: float = task_module.DEFAULT_HOLDOUT) -> str:
    return task_module.partition(
        [{"id": name} for name in NAMES], holdout
    )[task_module.HOLDOUT][0]["id"]


# ── the search only ever sees the train part ────────────────────────────────


def test_the_search_is_scored_on_the_train_part_and_never_on_the_holdout(tmp_path):
    """Every candidate the search paid for was scored on the items it was tuned on."""
    _loop, report = _run(tmp_path)
    assert report["leaderboard"]
    assert all(row["raw_score"] == pytest.approx(1.0) for row in report["leaderboard"]), (
        "a candidate was scored on the held-out items during the search"
    )
    assert report["holdout"]["instances"] == {"train": 3, "holdout": 1}


def test_the_winner_is_audited_on_the_holdout_and_the_gap_is_recorded(tmp_path):
    """The winner is run once more, on the items no candidate saw, and the gap is kept."""
    _loop, report = _run(tmp_path)
    audit = report["holdout"]
    assert audit["train"] == pytest.approx(1.0)
    assert audit["holdout"] == pytest.approx(0.0)
    assert audit["gap"] == pytest.approx(1.0)
    # The winner is the seed (the probe has nothing to move): its gap is the parts'
    # difference, not overfitting. An optimized winner's is checked below.
    assert audit["overfit"] is False and "the winner is the seed" in audit["note"]
    assert audit["threshold"] == 0.15
    assert audit["instances"] == {"train": 3, "holdout": 1}


class BonusProbe(Probe):
    """The probe with a knob that adds a little to the score, so a moved candidate
    beats the seed on train and wins."""

    @classmethod
    def declare(cls) -> dict[str, nn.Knob]:
        return {"bonus": nn.Knob(default=0, kind="int", lo=0, hi=9, description="a bonus")}

    def build(self) -> nn.Graph:
        graph = _probe_graph(TASK_REF)
        graph.nodes["work"] = graph.nodes["work"].with_field("knobs", ("bonus",))
        return graph

    def score(self, run: nn.RunResult) -> float:
        return super().score(run) + int(self.values()["bonus"]) / 1000


def test_an_optimized_winner_that_only_works_on_its_train_items_is_overfit(tmp_path):
    """A moved candidate wins on train (1.0) and scores 0 on the held-out item: the
    gap is beyond the threshold, so the winner overfit."""
    loop, report = _run(tmp_path, mode_class=BonusProbe, batch_size=2)
    audit = report["holdout"]
    assert audit["winner"] != loop.population[0].id, "setup: a moved candidate won"
    assert audit["gap"] > audit["threshold"] and audit["overfit"] is True
    assert audit["standard_error"] is not None


def test_the_audit_is_written_beside_the_run_and_carried_by_the_report(tmp_path):
    """`holdout.json` is the durable record; the in-memory payload is the same one."""
    loop, report = _run(tmp_path)
    root = tmp_path / "run"
    written = json.loads((root / "holdout.json").read_text())
    assert written == report["holdout"]
    assert loop.holdout_audit == report["holdout"]

    # The audit ran the winner in a directory of its own, on the held-out item, and
    # the score the search recorded for that candidate is still the train one.
    best = report["best"]["id"]
    assert json.loads((root / "holdout" / best / "current_item.json").read_text())[
        "id"
    ] == _held_id()
    assert json.loads((root / "runs" / best / "candidate.json").read_text())[
        "raw_score"
    ] == pytest.approx(1.0)


def test_one_instance_cannot_be_split_and_the_run_records_that(tmp_path):
    """A Task with one item is a fact about the Task, not a reason to fail the run."""
    _loop, report = _run(tmp_path, mode_class=SingleProbe)
    audit = report["holdout"]
    assert audit["instances"] == {"train": 1, "holdout": 0}
    assert audit["holdout"] is None
    assert audit["gap"] is None
    # ``overfit`` is a tri-state: True when the gap is above the threshold, False
    # when it was measured and is not, and None when there was no holdout to
    # measure. A single-item Task is the third case, not a claim about the winner.
    assert audit["overfit"] is None
    assert audit["threshold"] == 0.15
    assert "cannot be split" in audit.get("note", "")


# ── the part is the run's setting, not the candidate's ─────────────────────


def test_the_part_is_written_beside_the_run_and_not_into_the_mode(tmp_path):
    """The file is what the runtime reads, and a candidate cannot reach it.

    The point of a file rather than a knob is that the search cannot choose its own
    exam: a `set_knob` moving the holdout fraction is not a legal edit, because the
    mode declares no such knob.
    """
    _loop, report = _run(tmp_path, batch_size=2)
    root = tmp_path / "run"
    best = report["best"]["id"]
    train = json.loads((root / "runs" / best / "subset.json").read_text())
    # The run's seed travels with the part, as SkillOpt's and testeval's do.
    assert train == {"part": "train", "holdout": 0.3, "seed": 1}
    held = json.loads((root / "holdout" / best / "subset.json").read_text())
    assert held["part"] == "holdout" and held["seed"] == 1

    state = Probe().state_dict()
    assert "holdout" not in state["knobs"]
    with pytest.raises(nn.edits.EditError, match="no knob"):
        nn.edits.configure(state, "holdout", 0.0)


def test_a_different_fraction_moves_the_split(tmp_path):
    """The run's own setting decides both parts, and the audit is run on them.

    The Task rewards three of its four ids, so the default fraction audits one item
    and a half-held-out run audits two, and the two audits are different numbers.
    """
    _loop, report = _run(tmp_path, holdout=0.5)
    assert report["holdout"]["instances"] == {"train": 2, "holdout": 2}
    assert report["holdout"]["holdout"] == pytest.approx(0.5), (
        "the audited score is the mean over the two items the fraction held out"
    )


# ── what a person is told ──────────────────────────────────────────────────


def test_optimize_prints_one_line_about_the_held_out_audit(tmp_path, capsys):
    """`refactory optimize` ends with a sentence about the winner, not just a number."""
    from refactory_dsh.loop import main as loop_main

    loop_main([
        "dataset_probe/dataset", "--epochs", "1", "--batch-size", "1",
        "--optimizer", "random", "--seed", "1",
        "--runs-root", str(tmp_path / "run"), "--project", ".",
    ])
    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if line.startswith("holdout:")]
    assert len(lines) == 1, captured.err
    assert "train" in lines[0] and "holdout" in lines[0] and "holds up" in lines[0]
    assert "the winner is the seed" in lines[0]
    payload = json.loads(captured.out[captured.out.index("{"):])
    assert payload["holdout"]["overfit"] is False
    assert payload["raw_best_score"] is not None


def test_best_surfaces_the_audit_of_the_winner(tmp_path):
    """A reader asking for the winner is told the score may not travel."""
    from refactory_dsh import console

    root = _overfit_run(tmp_path)
    payload = console._best_payload(root, keep_state=False)
    assert payload["holdout"]["overfit"] is True
    assert payload["overfit_finding"], "best reported an overfit winner with no finding"
    assert "held-out" in payload["overfit_finding"]


def test_adopting_an_overfit_winner_is_refused_unless_the_override_is_given(tmp_path):
    """`adopt` is the one place a winner reaches the catalogue, so it is the gate."""
    from refactory_dsh import console

    root = _overfit_run(tmp_path)
    project = tmp_path
    destination = project / ".factory" / "packages" / "taken"

    assert console.main([
        "adopt", "--run", str(root), "--as", "taken", "--project", str(project),
    ]) == 2
    assert not destination.exists(), "an overfit winner was adopted without the override"

    assert console.main([
        "adopt", "--run", str(root), "--as", "taken", "--project", str(project),
        "--allow-overfit",
    ]) == 0
    assert (destination / "mode.py").exists()
    recorded = json.loads((destination / ".adopted.json").read_text())
    assert recorded["candidate"] == json.loads(
        (root / "holdout.json").read_text()
    )["winner"]


def _overfit_run(tmp_path: Path) -> Path:
    """A real run of a real Task-bound mode whose optimized winner is overfit.

    A workspace package subclasses `dataset_probe/dataset` with a `bonus` knob that
    adds a little to the score, so the swept candidate beats the seed and wins. The
    search rewards alpha's pass (1.0); the held-out item scores 0.5, so the winner's
    train-vs-holdout gap is about 0.5 -- beyond the threshold, and not the seed's.
    """
    source = tmp_path / ".factory" / "packages" / "overfit_pack"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text(
        "from __future__ import annotations\n\n"
        "from refactory_dsh.nn import Package\n\n"
        "from .mode import Overfit\n\n"
        "PACKAGE = Package(\n"
        "    name='overfit_pack', description='', modes=(Overfit,),\n"
        "    ops=('refactory_packages.dataset_probe.ops',),\n"
        ")\n"
    )
    (source / "mode.py").write_text(
        "from refactory_dsh import nn\n"
        "from refactory_packages.dataset_probe.mode import Dataset\n\n\n"
        "class Overfit(Dataset):\n"
        "    name = 'overfit'\n\n"
        "    @classmethod\n"
        "    def declare(cls):\n"
        "        return {**super().declare(),\n"
        "                'bonus': nn.Knob(0, 'int', lo=0, hi=9, description='a bonus')}\n\n"
        "    def build(self):\n"
        "        graph = super().build()\n"
        "        work = graph.nodes['work']\n"
        "        graph.nodes['work'] = work.with_field('knobs', (*work.knobs, 'bonus'))\n"
        "        return graph\n\n"
        "    def score(self, run):\n"
        "        return super().score(run) + int(self.values()['bonus']) / 1000\n"
    )
    root = tmp_path / "run"
    Loop(
        "overfit_pack/overfit",
        LoopConfig(epochs=1, batch_size=2, optimizer="random", seed=1, runs_root=root),
        project=tmp_path,
    ).run()
    audit = json.loads((root / "holdout.json").read_text())
    assert audit["overfit"] is True, audit
    assert audit["winner"] != json.loads((root / "report.json").read_text())["lineage"][0]["id"], (
        "setup: the winner is not the seed")
    return root


def test_a_candidate_cannot_move_its_own_exam(tmp_path):
    """The class of work is the seed's, and a search may not rebind it.

    A data node's Task and source *are* the exam: the held-out split is derived from
    the items a candidate actually has, so a candidate that rebinds the node would be
    graded on items it chose. The seed's binding is the one that stands, and a person's
    route to other items is a run's `--source`.
    """
    import copy

    from refactory_dsh import loop as loop_module
    from refactory_dsh import registry
    from refactory_dsh.paths import project_root

    _, klass = registry.resolve(project_root(), "dataset_probe/dataset")
    state = klass().state_dict()
    # Direct state mutation of task_ref via graph manipulation
    moved = copy.deepcopy(state)
    from refactory_dsh.nn.graph import Graph

    graph = Graph.from_state(moved["graph"])
    graph.nodes["data"] = graph.nodes["data"].with_field(
        "task_ref", "refactory_packages.java_build_pack.tasks:JavaBuildTask"
    )
    moved["graph"] = graph.to_state()

    from refactory_dsh import optim as optim_module

    run = loop_module.Loop(
        "dataset_probe/dataset",
        loop_module.LoopConfig(runs_root=tmp_path),
        optimizer=optim_module.create("random"),
    )
    run.population = [loop_module.Candidate.from_state(state)]
    candidate = loop_module.Candidate.from_state(moved)
    reason = run._exam_moved(candidate)
    assert reason is not None and reason.startswith("E-exam"), reason

    same = loop_module.Candidate.from_state(state)
    assert run._exam_moved(same) is None


def test_an_edit_cannot_bake_the_items_into_the_graph(tmp_path):
    """add_node with inline_items triggers E-items.

    `inline_items` is read before the Task or the source, so a state that carries it is
    a harness for exactly those items. Every primitive returns through the same
    finisher, so this holds for the ops that add a node.
    """
    import pytest as pytest_module

    from refactory_dsh import registry
    from refactory_dsh.nn import edits as edit_module
    from refactory_dsh.nn.edits import EditError
    from refactory_dsh.paths import project_root

    _, klass = registry.resolve(project_root(), "dataset_probe/dataset")
    state = klass().state_dict()
    new_node = {
        "id": "baked", "kind": "data",
        "subgraph_entry": "work", "subgraph_exit": "work",
        "inline_items": [{"id": "alpha"}, {"id": "beta"}],
        "writes": ["baked_item.json"], "produces": ["baked_scores"],
    }
    edit = {"op": "add_node", "node": new_node, "position": {"after": "data"}}
    with pytest_module.raises(EditError) as caught:
        edit_module.apply(edit, state)
    assert caught.value.code == "E-items"
