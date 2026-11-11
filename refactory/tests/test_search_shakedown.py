"""Correctness fixes from the structural and population search shakedown on LiveMath.

Each test fails on the code before the fix:

1. the default loop evaluates with the run's seed, so LiveMath's choices are shuffled;
2. LiveMath's theorem and sketch switches are frozen (the theorem states the answer);
3. a data pass ends at its exit on both runtimes, and edits move the exit;
4. the holdout audit respects the parts manifest and never runs test items;
5. the optimizer is shown the parent candidate's graph, not the seed's;
6. spine model calls, spine records of an all-failed run, structural wins, timeouts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import nn
from refactory_dsh.loop import Loop, LoopConfig
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.optim import Candidate

from test_livemath_blocks import (
    FIXTURE_DIR,
    _FailOnceModel,
    _NodeScriptedModel,
    _facts,
    _grid,
    _patch_task,
    _scores,
    _spine,
    needs_spine,
)
from test_parts import MODE as PARTS_MODE, PartsProbe, _patch_mode


# ── helpers ───────────────────────────────────────────────────────────────


def _all_a_items(root: Path, count: int = 6) -> Path:
    """LiveMath items whose correct label is always A, as in LiveMath's raw data."""
    data = root / "data" / "202601"
    data.mkdir(parents=True)
    rows = [{
        "month": "202601", "no": no, "paper_link": f"http://arxiv.org/abs/2601.{no:05d}v1",
        "theorem": f"Theorem {no}.", "sketch": f"Sketch {no}.", "theorem_type": ["analysis"],
        "mcq": {"question": f"Question number {no}: which holds?",
                "choices": [{"label": label, "text": f"choice {label} of {no}"} for label in "ABCD"],
                "correct_choice": {"label": "A", "text": f"choice A of {no}"}},
    } for no in range(1, count + 1)]
    (data / "qa_202601_final.json").write_text(json.dumps(rows))
    return root / "data"


class _AnswerA:
    """A walker model that always answers A."""

    def complete(self, prompt: str, **_: Any) -> str:
        return "<answer>A</answer>"

    def for_node(self, node_id: str, role: str | None = None) -> "_AnswerA":
        return self

    def with_tool_policy(self, policy: str) -> "_AnswerA":
        return self

    def with_wall(self, seconds: Any) -> "_AnswerA":
        return self

    def route(self) -> dict[str, Any]:
        return {"worker_backend": "dsh"}


def _verify_node(node_id: str = "verify") -> dict[str, Any]:
    from refactory_packages.livemath.mode import SYSTEM_PROMPT

    return {"id": node_id, "kind": "agent", "prompt_template": "Verify.\n\n{{user_prompt}}",
            "system_prompt_template": SYSTEM_PROMPT, "writes": ["answer.txt"],
            "tool_policy": "none", "role": "solver"}


# ── 1. the default loop evaluates with the run's seed ────────────────────


def test_default_loop_shuffles_livemath_choices_with_the_run_seed(tmp_path, monkeypatch):
    """Every correct label in LiveMath's data is A; the default loop must evaluate with
    its seed, as SkillOpt and testeval do, so each item's choices get SkillOpt's
    (item, seed) shuffle and the correct labels are no longer all A."""
    from refactory_packages.livemath.shuffling import shuffle_item_choices
    from refactory_packages.livemath.tasks import load_items

    data = _all_a_items(tmp_path)
    monkeypatch.setenv("LIVEMATH_DATA_PATH", str(data))
    config = LoopConfig(epochs=1, batch_size=1, max_evaluations=1, optimizer="random",
                        seed=42, runs_root=tmp_path / "run", holdout=0.0)
    Loop("livemath/solver", config, project=Path("."), model=_AnswerA()).run()
    (trial,) = sorted((tmp_path / "run" / "runs").iterdir())
    assert json.loads((trial / "subset.json").read_text())["seed"] == 42
    labels = {r["id"]: r["label"] for r in json.loads((trial / "item_records.json").read_text())}
    expected = {item["id"]: shuffle_item_choices(item, 42)["correct_choice"]["label"]
                for item in load_items(str(data))}
    assert labels == expected
    assert set(labels.values()) != {"A"}


@needs_spine
def test_spine_items_follow_the_seed_the_loop_writes(tmp_path, monkeypatch):
    """The seed reaches the spine's item loader as it reaches the walker's: the same
    subset.json gives both runtimes SkillOpt's shuffled labels and the same scores."""
    from refactory_dsh.task import write_subset
    from refactory_packages.livemath.mode import Solver
    from refactory_packages.livemath.shuffling import shuffle_item_choices
    from refactory_packages.livemath.tasks import load_items

    monkeypatch.setenv("LIVEMATH_DATA_PATH", str(FIXTURE_DIR))  # what _spine hands the spine
    mode = Solver()
    walker_dir, spine_dir = tmp_path / "walker", tmp_path / "spine"
    for workdir in (walker_dir, spine_dir):
        write_subset(workdir, "train", 0.0, seed=42)
    walker = mode.run(walker_dir, model=_AnswerA())
    assert walker.exit_ok, walker.error
    rc = _spine(mode.graph(), spine_dir, tmp_path, agent_responses={"solve": "<answer>A</answer>"},
                mode_state=mode.state_dict())
    assert rc == 0
    spine_records = json.loads((spine_dir / "item_records.json").read_text())
    raw = {item["id"]: item for item in load_items(str(FIXTURE_DIR))}
    shuffled = {i: shuffle_item_choices(item, 42)["correct_choice"]["label"] for i, item in raw.items()}
    assert shuffled != {i: item["correct_choice"]["label"] for i, item in raw.items()}
    for records in (walker.item_records, spine_records):
        assert {r["id"]: r["label"] for r in records} == shuffled
    assert _scores(walker.facts["item_scores"]) == _scores(_facts(spine_dir)["item_scores"])


# ── 2. LiveMath's answer-bearing switches are frozen ─────────────────────


@pytest.mark.parametrize("block", ["baseline", "solve_then_verify",
                                   "two_attempts_then_judge", "solve_then_self_check"])
@pytest.mark.parametrize("knob", ["use_theorem", "use_sketch"])
def test_theorem_and_sketch_cannot_be_turned_on_by_a_search(block, knob):
    """The theorem text states the correct option, so no edit may turn it on."""
    from refactory_packages.livemath.mode import Solver

    state = Solver(block=block).state_dict()
    assert state["knobs"][knob]["spec"]["frozen"] is True
    assert state["knobs"][knob]["value"] is False
    with pytest.raises(nn_edits.EditError, match="frozen"):
        nn_edits.apply({"op": "set_knob", "name": knob, "value": True}, state)


# ── 3. a data pass ends at its exit, on both runtimes ────────────────────


def _livemath_state() -> dict[str, Any]:
    from refactory_packages.livemath.mode import Solver

    return Solver().state_dict()


def test_inserting_after_the_exit_moves_the_exit():
    # verify reads solve's answer before writing its own: one that did not would lose
    # solve's work (V19).
    verify = {**_verify_node(), "reads": ["answer.txt"]}
    state = nn_edits.apply({"op": "insert_after", "target": "solve", "node": verify},
                           _livemath_state())
    graph = nn.Graph.from_state(state["graph"])
    assert graph.nodes["data"].subgraph_exit == "verify"
    assert not nn.validate_state(state, resolve_calls=False)


def test_duplicating_the_exit_moves_the_exit():
    state = nn_edits.apply({"op": "duplicate", "target": "solve", "new_id": "solve_again"},
                           _livemath_state())
    graph = nn.Graph.from_state(state["graph"])
    assert graph.nodes["data"].subgraph_exit == "solve_again"
    assert [e.target for e in graph.out_edges("solve")] == ["solve_again"]


def test_inserting_before_the_entry_moves_the_entry():
    node = {**_verify_node("plan"), "writes": ["plan.txt"]}
    state = nn_edits.apply({"op": "insert_before", "target": "format_prompt", "node": node},
                           _livemath_state())
    graph = nn.Graph.from_state(state["graph"])
    assert graph.nodes["data"].subgraph_entry == "plan"
    assert "plan" in graph.data_body("data")


def test_a_terminal_node_is_added_after_the_run_not_after_a_pass_exit():
    """``{terminal: true}`` wires the run's terminals; the pass exit (solve) has no
    out-edge too, but a node after it would run inside the pass."""
    node = {"id": "report", "kind": "op", "call": "refactory_packages.livemath.ops:finish",
            "writes": ["report.json"]}
    state = nn_edits.apply({"op": "add_node", "node": node, "position": {"terminal": True}},
                           _livemath_state())
    graph = nn.Graph.from_state(state["graph"])
    assert [e.source for e in graph.in_edges("report")] == ["finish"]
    assert not graph.out_edges("solve")


def test_an_edge_out_of_a_pass_exit_is_refused():
    state = nn_edits.apply({"op": "add_node", "node": _verify_node(), "position": {"detached": True}},
                           _livemath_state())
    state = nn_edits.apply({"op": "add_edge", "source": "solve", "target": "verify"}, state)
    issues = nn.validate_state(state, resolve_calls=False)
    assert any(i.code == "V12" and "exit 'solve'" in i.message for i in issues), issues


def test_the_walker_ends_a_pass_at_its_exit(tmp_path, monkeypatch):
    """A node wired after the exit does not run on the walker, as on the spine."""
    from refactory_dsh.nn.executor import execute
    from refactory_packages.livemath.mode import Solver

    _patch_task(monkeypatch)
    graph = Solver().graph()
    graph.add(nn.Node.from_state(_verify_node()))
    graph.connect("solve", "verify")
    model = _NodeScriptedModel({"solve": "<answer>A</answer>", "verify": "<answer>B</answer>"})
    result = execute(graph, tmp_path / "run", knobs=Solver().values(), model=model)
    assert result.exit_ok, result.error
    assert {node for node, _ in model.log} == {"solve"}


@needs_spine
@pytest.mark.parametrize("edit", [
    {"op": "insert_after", "target": "solve", "node": _verify_node()},
    {"op": "duplicate", "target": "solve", "new_id": "verify"},
], ids=["insert_after", "duplicate"])
def test_a_structural_candidate_scores_the_same_on_walker_and_spine(edit, tmp_path, monkeypatch):
    """The shakedown's candidate scored 1.0 on the walker and 0.0 on the spine: the
    node after the exit ran only on the walker. Now both run it, as the pass's exit,
    and grade its answer."""
    from refactory_packages.livemath.mode import Solver

    _patch_task(monkeypatch)
    state = nn_edits.apply(edit, Solver().state_dict())
    mode = Solver(state=state)
    answers = {"solve": "<answer>A</answer>", "verify": "<answer>B</answer>"}
    walker = mode.run(tmp_path / "walker", model=_NodeScriptedModel(answers))
    assert walker.exit_ok, walker.error
    spine_dir = tmp_path / "spine"
    assert _spine(mode.graph(), spine_dir, tmp_path, agent_responses=answers,
                  mode_state=mode.state_dict()) == 0
    spine_records = json.loads((spine_dir / "item_records.json").read_text())
    label_b = {"202501:1", "202503:1", "202503:2"}
    walker_scores = _scores(walker.facts["item_scores"])
    assert walker_scores == _scores(_facts(spine_dir)["item_scores"])
    assert walker_scores == sorted((i, 1.0 if i in label_b else 0.0) for i, _ in walker_scores)
    assert mode.score(walker) == pytest.approx(0.6)
    for records in (walker.item_records, spine_records):
        assert sorted(len(r["exchanges"]) for r in records) == [2] * 5
        assert sorted((r["id"], r["prediction"]) for r in records) == sorted(
            (i, "B") for i, _ in walker_scores)


# ── 4. the holdout respects the parts manifest ───────────────────────────


def _parts_run(tmp_path: Path, parts: dict[str, list[str]],
               holdout: float = 0.5) -> tuple[Loop, dict[str, Any]]:
    manifest = tmp_path / "parts.json"
    manifest.write_text(json.dumps(parts))
    with _patch_mode(PartsProbe):
        loop = Loop(PARTS_MODE, LoopConfig(epochs=1, batch_size=1, optimizer="random", seed=1,
                                           runs_root=tmp_path / "run", holdout=holdout,
                                           parts_manifest=str(manifest),
                                           # the audit of an ungated run's val part
                                           val_gate=False), project=Path("."))
        report = loop.run()
    return loop, report


def _every_record_id(root: Path) -> set[str]:
    ids: set[str] = set()
    for path in root.rglob("item_records.json"):
        ids |= {str(r["id"]) for r in json.loads(path.read_text())}
    return ids


def test_the_holdout_audit_runs_the_manifest_val_part_and_never_test(tmp_path):
    loop, report = _parts_run(tmp_path, {"train": ["alpha", "beta", "gamma", "delta"],
                                         "val": ["epsilon"], "test": ["zeta"]})
    audit = report["holdout"]
    assert audit["status"] == "measured", audit
    assert audit["instances"] == {"train": 4, "holdout": 1}
    winner = audit["winner"]
    held = json.loads((tmp_path / "run" / "holdout" / winner / "item_records.json").read_text())
    assert [r["id"] for r in held] == ["epsilon"]
    assert "zeta" not in _every_record_id(tmp_path / "run")


def test_a_manifest_without_val_has_no_holdout_rather_than_a_test_one(tmp_path):
    _loop, report = _parts_run(tmp_path, {"train": ["alpha", "beta", "gamma", "delta"],
                                          "test": ["epsilon", "zeta"]})
    audit = report["holdout"]
    assert audit["status"] == "not_applicable", audit
    assert audit["instances"] == {"train": 4, "holdout": 0}
    assert not (tmp_path / "run" / "holdout").exists()
    assert not {"epsilon", "zeta"} & _every_record_id(tmp_path / "run")


def test_holdout_zero_turns_the_manifest_audit_off(tmp_path):
    """Arm B runs with --holdout 0 and a manifest: its winner is not re-run on val."""
    _loop, report = _parts_run(tmp_path, {"train": ["alpha", "beta", "gamma", "delta"],
                                          "val": ["epsilon"], "test": ["zeta"]}, holdout=0.0)
    audit = report["holdout"]
    assert audit["status"] == "not_applicable", audit
    assert audit["instances"] == {"train": 4, "holdout": 0}
    assert not (tmp_path / "run" / "holdout").exists()


# ── 5. the optimizer sees the parent's graph ─────────────────────────────


def test_the_proposal_context_describes_the_parent_not_the_seed(tmp_path):
    with _patch_mode(PartsProbe):
        loop = Loop(PARTS_MODE, LoopConfig(epochs=1, batch_size=1, optimizer="random", seed=1,
                                           runs_root=tmp_path / "run"), project=Path("."))
        loop.init()
        seed = loop.population[0]
        edit = {"op": "insert_after", "target": "finish",
                "node": {"id": "extra", "kind": "op", "call": "loop_ops:noop"}}
        child = Candidate.from_state(nn_edits.apply(edit, seed.state), parent=seed.id, edits=(edit,))
        child.evaluated, child.exit_ok, child.score = True, True, 1.0
        loop.population.append(child)
        loop._editing_parent = child.id
        context = loop._context()
    assert "extra" in {node["id"] for node in context.nodes}
    assert {"source": "finish", "target": "extra"} in [
        {"source": e["source"], "target": e["target"]} for e in context.edges]


def _probe_loop_with_child(tmp_path: Path) -> tuple[Loop, Candidate, Candidate]:
    loop = Loop(PARTS_MODE, LoopConfig(epochs=1, batch_size=1, optimizer="random", seed=1,
                                       runs_root=tmp_path / "run"), project=Path("."))
    loop.init()
    seed = loop.population[0]
    edit = {"op": "insert_after", "target": "finish",
            "node": {"id": "extra", "kind": "op", "call": "loop_ops:noop"}}
    child = Candidate.from_state(nn_edits.apply(edit, seed.state), parent=seed.id, edits=(edit,))
    child.evaluated, child.exit_ok, child.score = True, True, 1.0
    loop.population.append(child)
    return loop, seed, child


def test_the_proposal_context_describes_the_parent_a_strategy_names(tmp_path):
    with _patch_mode(PartsProbe):
        loop, seed, child = _probe_loop_with_child(tmp_path)
        loop._editing_parent = seed.id
        loop.strategy_decision = {"parent": child.id}
        context = loop._context()
    assert "extra" in {node["id"] for node in context.nodes}
    assert context.graph_note == ""


def test_an_unbuildable_parent_is_named_not_passed_off_as_the_seed(tmp_path):
    """When the parent cannot be built the seed's pipeline is shown, and both the
    prompt and the run's progress say so."""
    with _patch_mode(PartsProbe):
        loop, seed, _child = _probe_loop_with_child(tmp_path)
        broken = Candidate.from_state({**seed.state, "graph": {"nodes": "not a graph"}},
                                      parent=seed.id, edits=())
        broken.evaluated, broken.exit_ok, broken.score = True, True, 1.0
        loop.population.append(broken)
        loop._editing_parent = broken.id
        context = loop._context()
    assert "extra" not in {node["id"] for node in context.nodes}
    assert broken.id in context.graph_note and "could not be built" in context.graph_note
    assert f"Note: {context.graph_note}" in context.render()
    progress = [json.loads(line) for line in
                (tmp_path / "run" / "progress.jsonl").read_text().splitlines()]
    assert any(r.get("step") == "warning" and r.get("candidate") == broken.id for r in progress)


# ── 6. smaller fixes ─────────────────────────────────────────────────────


def _receipt(path: Path, node_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"node_id": node_id, "state": "completed"}))


def test_a_spine_candidate_reports_its_model_calls(tmp_path, monkeypatch):
    """One worker receipt per agent call, in the run and in each pass; the
    coordinator's receipt and other evaluations' receipts are not counted."""
    from refactory_dsh import dsh_runtime
    from refactory_packages.livemath.mode import Solver

    workdir = tmp_path / "cand"
    _receipt(workdir / ".refactory/workers/w-coord.json", "")
    _receipt(workdir / ".refactory/workers/w-top.json", "judge")
    for index in range(3):
        _receipt(workdir / f"passes/data/item_{index}/.refactory/workers/w{index}.json", "solve")
    _receipt(workdir / "trials/2/passes/data/item_0/.refactory/workers/w9.json", "solve")
    _receipt(workdir / "occasions/val_0/.refactory/workers/w8.json", "solve")
    (workdir / ".refactory" / "spine-status.json").write_text(json.dumps({"status": "done"}))

    def fake_run(state, workdir, **kwargs):  # noqa: ANN001
        return dsh_runtime.DshRun(facts={"item_scores": [{"id": "x", "score": 1.0}]},
                                  workdir=workdir, returncode=0)

    monkeypatch.setattr(dsh_runtime, "run_on_dsh", fake_run)
    monkeypatch.setattr("refactory_dsh.nn.executor._run_end_cleanup", lambda workdir: None)
    mode = Solver()
    result = dsh_runtime.evaluate_on_dsh(mode, mode.state_dict(), workdir)
    assert result.exit_ok, result.error
    assert result.model_calls == 4


@pytest.mark.parametrize("ending", ["dsh_failed", "graph_unfinished"])
def test_a_failed_spine_candidate_reports_its_model_calls(ending, tmp_path, monkeypatch):
    """A spine run that failed still spent its calls: both failure paths count them."""
    from refactory_dsh import dsh_runtime
    from refactory_packages.livemath.mode import Solver

    workdir = tmp_path / "cand"
    for index in range(2):
        _receipt(workdir / f"passes/data/item_{index}/.refactory/workers/w{index}.json", "solve")
    (workdir / ".refactory").mkdir(parents=True)
    (workdir / ".refactory" / "spine-status.json").write_text(json.dumps({
        "status": "ready", "last_failure": {"node": "data", "error": "all 2 items failed"}}))

    def fake_run(state, workdir, **kwargs):  # noqa: ANN001
        return dsh_runtime.DshRun(facts={"item_scores": []} if ending == "graph_unfinished" else {},
                                  workdir=workdir, returncode=0 if ending == "graph_unfinished" else 1,
                                  stderr="boom")

    monkeypatch.setattr(dsh_runtime, "run_on_dsh", fake_run)
    monkeypatch.setattr("refactory_dsh.nn.executor._run_end_cleanup", lambda workdir: None)
    mode = Solver()
    result = dsh_runtime.evaluate_on_dsh(mode, mode.state_dict(), workdir)
    assert not result.exit_ok
    assert result.model_calls == 2
    if ending == "graph_unfinished":
        assert result.error == "the graph did not finish: all 2 items failed"


def test_a_spine_candidate_whose_items_all_failed_keeps_its_records(tmp_path, monkeypatch):
    """The spine writes item_records.json only when a run is done; an all-failed data
    node leaves the run 'ready' with the records in its status. They are kept, as the
    walker keeps them."""
    from refactory_dsh import dsh_runtime
    from refactory_dsh.evaluators import DshEvaluator
    from refactory_dsh.nn.mode import RunResult
    from refactory_packages.livemath.mode import Solver

    records = [{"id": f"i{n}", "passed": False, "score": 0.0, "error_kind": "timeout",
                "error_text": "worker: model completion timed out after 3s"} for n in range(2)]
    workdir = tmp_path / "cand"
    (workdir / ".refactory").mkdir(parents=True)
    (workdir / ".refactory" / "spine-status.json").write_text(json.dumps({
        "status": "ready", "item_records": records,
        "last_failure": {"node": "data", "error": "AllItemsFailedError: data: all 2 items failed"}}))
    monkeypatch.setattr(dsh_runtime, "evaluate_on_dsh", lambda *a, **k: RunResult(
        exit_ok=False, error="the graph did not finish: ready", workdir=workdir))
    mode = Solver()
    result = DshEvaluator().evaluate(mode, mode.state_dict(), workdir)
    assert not result.exit_ok
    assert result.item_records == records
    assert json.loads((workdir / "item_records.json").read_text()) == records


def test_structural_wins_are_only_structural_candidates_that_beat_their_parent(tmp_path):
    with _patch_mode(PartsProbe):
        loop = Loop(PARTS_MODE, LoopConfig(epochs=1, batch_size=1, optimizer="random", seed=1,
                                           runs_root=tmp_path / "run"), project=Path("."))
        loop.init()
        seed = loop.population[0]
        seed.evaluated, seed.exit_ok, seed.score = True, True, 0.5

        def child(node_id: str, score: float) -> Candidate:
            edit = {"op": "insert_after", "target": "finish",
                    "node": {"id": node_id, "kind": "op", "call": "loop_ops:noop"}}
            made = Candidate.from_state(nn_edits.apply(edit, seed.state), parent=seed.id,
                                        edits=(edit,))
            made.evaluated, made.exit_ok, made.score = True, True, score
            return made

        better, worse, same = child("better", 0.75), child("worse", 0.25), child("same", 0.5)
        loop.population.extend([better, worse, same])
        wins = loop._structural_wins([better, worse, same, seed])
    assert [row["id"] for row in wins] == [better.id]


def test_recovered_selection_reports_structural_wins_against_parents(tmp_path):
    """A Keep recovered after the search rewrites the report's structural_wins with
    the same rule: the running example's structural candidates lose to their
    parents, so it has none."""
    from refactory_dsh.control import Control
    from refactory_dsh.finished_control import recover_selection

    root = tmp_path / "run"
    loop = Loop("circle_packing/adaptive", LoopConfig(epochs=2, batch_size=4, optimizer="random",
                                                      seed=5, runs_root=root), project=Path("."))
    report = loop.run()
    structural = [row for row in report["leaderboard"]
                  if any(e.get("op") in ("add_node", "duplicate") for e in row.get("edits") or ())]
    assert structural
    # A legacy request left queued when the search finished (``request`` now refuses
    # a keep on a finished search, so it is written as the old runs left it).
    Control(root).path.write_text(json.dumps(
        {"action": "keep", "target": report["leaderboard"][0]["id"], "at": 1.0}))
    assert recover_selection(root, project=Path(".")) is not None
    recovered = json.loads((root / "report.json").read_text())
    assert recovered["selection_recovery"]["status"] == "applied"
    assert recovered["structural_wins"] == []


def test_a_node_timeout_is_recorded_as_a_timeout_on_both_runtimes():
    """The walker's failed pass and the spine's worker stderr both name a completion
    timeout; each is a ``timeout``, not a RuntimeError, so the grid can count it."""
    from refactory_dsh.items import make_item_record

    walker = RuntimeError("solve: CompletionTimeout: model completion timed out after 3s; "
                          "requested runtime shutdown")
    spine = RuntimeError("worker: model completion timed out after 3s; requested runtime "
                         "shutdown; the client does not replay a deadline")
    for error in (walker, spine):
        assert make_item_record("x", None, error=error)["error_kind"] == "timeout"
    assert make_item_record("x", None, error=RuntimeError("boom"))["error_kind"] == "RuntimeError"


def test_the_grid_counts_and_prints_timeouts(tmp_path, monkeypatch, capsys):
    from refactory_dsh.model import CompletionTimeout

    class Factory:
        def __call__(self, **kwargs: Any) -> Any:
            return _FailOnceModel(CompletionTimeout(
                "model completion timed out after 3s; requested runtime shutdown"))

    rc, output, _ = _grid(tmp_path, monkeypatch, factory=Factory(), blocks=["baseline"])
    assert rc == 0
    (row,) = json.loads(output.read_text())["blocks"]
    assert row["timeouts"] == 1 and row["errors"] == 1
    lines = capsys.readouterr().out.splitlines()
    header = next(line for line in lines if line.startswith("Block"))
    assert "T/O" in header.split()
    baseline = next(line for line in lines if line.startswith("baseline "))
    assert baseline.split()[header.split().index("T/O")] == "1"
