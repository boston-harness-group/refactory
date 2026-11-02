"""The outer loop expressed as a refactory mode.

Issue #18: the loop's control flow as data. What these test is that the graph is
the loop — the same steps, in the same order, bounded by the same gate, reaching
the same score as the loop object does on the same seed — and that it is a mode
like any other, so the same outer loop can be pointed at it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import child_patches

from refactory_dsh import nn, optim, registry
from refactory_dsh.loop import Loop, LoopConfig, LoopError
from refactory_dsh.nn import validate_state

MODE = "loop_mode/loop"
TARGET = "circle_packing/adaptive"


def _knobs(root: Path, **overrides: object) -> dict:
    base = {
        "target": TARGET,
        "project": str(Path(".").resolve()),
        "epochs": 2,
        "batch_size": 3,
        "optimizer": "random",
        "elite": 3,
        "attempts": 3,
        "max_evaluations": 64,
        "seed": 5,
    }
    base.update(overrides)
    return base


def _mode(**overrides: object) -> nn.Mode:
    """The loop mode with its knobs set: a knob is state, so it is set as state."""
    mode_class = registry.resolve(Path("."), MODE)[1]
    state = mode_class().state_dict()
    for name, value in _knobs(Path("."), **overrides).items():
        state["knobs"][name]["value"] = value
    return mode_class(state=state)


def _run(root: Path, **overrides: object) -> tuple[nn.Mode, nn.RunResult]:
    mode = _mode(**overrides)
    return mode, mode.run(root)


def _direct(root: Path, **overrides: object) -> dict:
    config = LoopConfig(
        epochs=2, batch_size=3, optimizer="random", elite=3, attempts=3,
        max_evaluations=64, seed=5, runs_root=root,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return Loop(TARGET, config, project=Path(".")).run()


# ── the graph is the loop ──────────────────────────────────────────────────


def test_the_mode_is_registered_and_scored():
    package, mode_class = registry.resolve(Path("."), MODE)
    assert package.name == "loop_mode"
    assert mode_class.name == "loop"
    assert mode_class().is_scored()


def test_the_graph_is_the_eight_steps_in_order(tmp_path):
    mode = _mode()
    graph = mode.graph()
    assert graph.start == "init"
    chain = ["init", "forward", "loss", "backward", "optimizer", "step", "scheduler", "again", "report"]
    for earlier, later in zip(chain, chain[1:]):
        assert later in [edge.target for edge in graph.out_edges(earlier)], (earlier, later)
    verdicts = {edge.condition for edge in graph.out_edges("again")}
    assert verdicts == {"again", "done"}


def test_the_loop_back_edge_passes_through_the_gate(tmp_path):
    """V10: an unbounded cycle is not a loop, it is a hang."""
    mode = _mode()
    assert validate_state(mode.state_dict(), invariants=mode.invariants()) == []
    assert [edge.target for edge in mode.graph().out_edges("again", condition="again")] == ["forward"]


def test_the_target_is_frozen_because_moving_it_changes_what_is_measured():
    knobs = _package_knobs()
    assert knobs["target"].frozen is True
    assert knobs["seed"].frozen is True
    assert knobs["epochs"].frozen is False
    assert knobs["batch_size"].frozen is False


def _package_knobs() -> dict:
    return registry.resolve(Path("."), MODE)[1].declare()


# ── running it runs the loop ───────────────────────────────────────────────


def test_it_produces_the_same_score_as_the_loop_itself(tmp_path):
    """The claim of #18: the graph IS the loop, not a re-implementation of it."""
    mode, result = _run(tmp_path / "graph")
    direct = _direct(tmp_path / "direct")
    assert result.exit_ok, result.error
    assert direct["best_score"] == mode.score(result)
    assert direct["evaluations"] == result.facts["evaluations"]


def test_it_writes_the_run_the_loop_would_have_written(tmp_path):
    _package, result = _run(tmp_path / "graph")
    root = tmp_path / "graph" / "loop"
    for name in ("config.json", "journal.jsonl", "report.json", "report.md", "gradient.json"):
        assert (root / name).exists(), name
    report = json.loads((root / "report.json").read_text())
    assert report["mode"] == TARGET
    assert report["bound"] == "epochs"
    assert report["best_score"] > 0


#: Where `recording_scheduler` writes the counts each epoch was closed with.
SCHEDULED = "LOOP_AS_MODE_SCHEDULED"


def recording_scheduler(self, **counts):  # type: ignore[no-untyped-def]
    """`Loop.scheduler`, each call's counts recorded first: it runs in the op child."""
    with Path(os.environ[SCHEDULED]).open("a") as handle:
        handle.write(json.dumps(counts) + "\n")
    return child_patches.real(Loop, "scheduler")(self, **counts)


def test_the_graph_closes_each_epoch_with_that_epochs_counts(tmp_path, monkeypatch):
    """The scheduler step is handed the optimizer step's own counts, as the loop is."""
    monkeypatch.setenv(SCHEDULED, str(tmp_path / "scheduled.jsonl"))
    child_patches.for_test(monkeypatch).setattr(Loop, "scheduler", recording_scheduler)
    _package, result = _run(tmp_path / "graph")
    assert result.exit_ok, result.error
    seen = [json.loads(line) for line in (tmp_path / "scheduled.jsonl").read_text().splitlines()]
    assert len(seen) == 2
    for counts in seen:
        assert counts["proposed"] == counts["accepted"] + counts["refused"] > 0, counts


def test_the_run_reports_what_bounded_it(tmp_path):
    _package, result = _run(tmp_path / "graph")
    assert result.facts["bound"] == "epochs"


def test_the_run_honours_the_evaluation_ceiling(tmp_path):
    """The other bound, reached the way the loop reaches it."""
    _package, result = _run(tmp_path / "graph", epochs=8, batch_size=3, max_evaluations=6)
    assert result.exit_ok, result.error
    assert result.facts["evaluations"] == 6
    assert result.facts["bound"] == "evaluations"


def test_a_short_batch_still_reaches_the_same_best_score(tmp_path):
    """One epoch of the same seed finds what two did, because the seed population did."""
    _package, one = _run(tmp_path / "one", epochs=1, batch_size=3)
    _package, two = _run(tmp_path / "two", epochs=2, batch_size=3)
    assert one.facts["best"] == two.facts["best"]


def test_the_settings_reach_the_run(tmp_path):
    """The loop's own knobs are the run's settings, which is what makes it searchable."""
    _package, result = _run(tmp_path / "graph", batch_size=1, epochs=3)
    config = json.loads((tmp_path / "graph" / "loop" / "config.json").read_text())
    assert config["batch_size"] == 1
    assert config["epochs"] == 3
    assert result.facts["evaluations"] == 3


def test_a_target_that_cannot_be_resolved_is_reported(tmp_path):
    mode = _mode(target="nope/missing")
    result = mode.run(tmp_path / "bad")
    assert result.exit_ok is False
    assert "nope" in (result.error or "")


# ── it is a mode, so a loop can be pointed at it ────────────────────────────


def test_an_outer_loop_can_search_the_loop_itself(tmp_path):
    """The point of the exercise: the loop's own knobs become a search surface.

    This used to assert only that `batch_size`, `epochs` and `optimizer` are
    *visible* in the context, which is expressiveness -- that the settings are
    reachable, not that anything reached them. `architecture.md` §11 recorded the
    difference for as long as it stood: "expressible, not yet used". So this runs
    the search, which is the claim D14 actually makes.
    """
    loop = Loop(
        MODE,
        LoopConfig(
            epochs=1, batch_size=3, optimizer="random", seed=0,
            runs_root=tmp_path / "outer",
            max_evaluations=6,
        ),
        project=Path("."),
    )
    report = loop.run()

    assert report["evaluations"] >= 1, "the outer loop evaluated nothing"
    assert report["leaderboard"], "the search over the tower produced no leaderboard"
    assert report["best_score"] is not None
    assert report["best_score"] >= report["seed_score"], (
        "searching the loop's own settings made the inner result worse"
    )
    # The knobs the search may move are the loop's own, not the inner mode's.
    context = loop._context()  # noqa: SLF001 - the test drives the real builder
    names = {knob["name"] for knob in context.knobs}
    assert {"batch_size", "epochs", "optimizer"} <= names


def test_the_loops_own_knobs_are_a_real_lever(tmp_path):
    """A search needs a surface that moves, not merely one that exists.

    Deterministic on seed 0, so this is a fact about the tower rather than a lucky
    draw: the same inner loop with a wider batch reaches a better inner result.
    Without this, the outer run above could not have improved anything -- and a
    search over a surface that cannot move would still "pass" its own test.
    """
    _package, small = _run(tmp_path / "small", batch_size=4, max_evaluations=16)
    _package, large = _run(tmp_path / "large", batch_size=8, max_evaluations=16)
    assert large.facts["best"] > small.facts["best"], (
        "the loop's own batch_size does not move the inner result, so searching the "
        f"loop has nothing to find: 4 -> {small.facts['best']}, 8 -> {large.facts['best']}"
    )


def test_its_score_is_the_best_score_the_inner_run_reached(tmp_path):
    _package, result = _run(tmp_path / "graph")
    report = json.loads((tmp_path / "graph" / "loop" / "report.json").read_text())
    assert result.facts["best"] == report["best_score"]


def test_a_run_with_no_score_is_refused_rather_than_read_as_zero(tmp_path):
    """Zero and "never scored" are different answers, and a search must know which."""
    mode = _mode()
    empty = nn.RunResult(facts={}, exit_ok=False, error="the inner run never ran")
    with pytest.raises(ValueError, match="no best score"):
        mode.score(empty)


def test_the_declared_ports_are_the_files_the_run_writes(tmp_path):
    """A port says what a run produced; the validator is what keeps that true."""
    mode, result = _run(tmp_path / "run")
    assert result.exit_ok, result.error
    ports = {port.name: port.path for port in mode.ports()}
    assert set(ports) == {"report", "journal"}
    for path in ports.values():
        assert (tmp_path / "run" / path).exists(), path


def test_the_steps_are_ops_the_optimizer_may_not_edit_away(tmp_path):
    """The loop's shape is its semantics; a search edits its settings, not its steps."""
    mode = _mode()
    assert set(mode.graph().nodes) >= {"init", "forward", "scheduler", "again", "report"}
    assert mode.edit_policy() == "broad"
    assert "set_knob" in mode.preferred_edits()


def test_an_inserted_advisory_node_cannot_move_the_score(tmp_path):
    mode = _mode()
    insertable = mode.insertable()
    assert insertable
    assert all(node.score_effect == "none" for node in insertable), (
        "an advisory block must say so, or a search spends a batch on it"
    )


def test_the_run_resumes_from_its_own_journal(tmp_path):
    """Each node is a separate process, so the journal is what carries the state."""
    root = tmp_path / "graph"
    _package, result = _run(root)
    assert result.exit_ok, result.error
    journal = [json.loads(line) for line in (root / "loop" / "journal.jsonl").read_text().splitlines()]
    events = [record["event"] for record in journal]
    assert events.count("epoch_open") == 2
    assert events.count("epoch_end") == 2
    assert events.count("step") == 2
    assert "bound" in events


# ── the other runtime ──────────────────────────────────────────────────────
# The walker is the debug runtime; DSH is the runtime that ships. A graph that
# only the walker accepts is a graph that does not ship, so the loop's own graph
# is lowered and offered to the spine's validator here, and driven end to end by
# the integration test beside this file's unit tests.


def test_the_runtime_that_ships_accepts_the_lowered_loop(tmp_path):
    """The spine's own validator, not just refactory's."""
    import subprocess

    from refactory_dsh import lowering
    from refactory_dsh.paths import REPO_ROOT, node_bin

    state = registry.resolve(Path("."), MODE)[1]().state_dict()
    path, _workflow = lowering.write_mode_graph(state, tmp_path / "graphs", name="loop")
    spine = REPO_ROOT / "packages" / "workflow" / "workflow-spine" / "src" / "spine.ts"
    script = (
        "import { readFileSync } from 'node:fs';"
        f"import {{ validateGraph }} from {spine.as_uri()!r};"
        f"const g = JSON.parse(readFileSync({str(path)!r}, 'utf8'));"
        "const issues = validateGraph(g as never);"
        "console.log(JSON.stringify(issues));"
    )
    result = subprocess.run(
        [str(node_bin()), "--import", "tsx/esm", "-e", script],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180,
    )
    assert result.returncode == 0, result.stderr[-1000:]
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


def test_the_lowered_loop_keeps_the_loop_back_edge(tmp_path):
    """Lowering must not flatten the cycle: a loop without its edge is a pipeline."""
    from refactory_dsh import lowering

    state = registry.resolve(Path("."), MODE)[1]().state_dict()
    path, _workflow = lowering.write_mode_graph(state, tmp_path / "graphs", name="loop")
    written = json.loads(Path(path).read_text())
    edges = {(edge["source"], edge["target"], edge.get("condition") or "") for edge in written["edges"]}
    assert ("again", "forward", "again") in edges
    assert ("again", "report", "done") in edges
    assert written["start_node"] == "init"


@pytest.mark.integration
def test_it_runs_end_to_end_on_the_runtime_that_ships(tmp_path):
    """The same graph, driven by DSH's spine rather than by the walker.

    Nine nodes, the gate routing `done` to the report, and the same best score —
    on the only runtime a deployment has. Marked integration because it spawns a
    real session: `pytest --run-integration`.
    """
    import subprocess

    from refactory_dsh import lowering
    from refactory_dsh.paths import REPO_ROOT, dsh_command, venv_bin
    from refactory_dsh.run import FACTS

    state = registry.resolve(Path("."), MODE)[1]().state_dict()
    graphs = tmp_path / "graphs"
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    lowering.write_mode_graph(state, graphs, name="loop")
    lowering.write_declarations(state, work)
    knobs = {name: entry["value"] for name, entry in state["knobs"].items()}
    knobs["project"] = str(Path(".").resolve())
    knobs["epochs"] = 2
    knobs["batch_size"] = 2
    knobs["max_evaluations"] = 4

    result = subprocess.run(
        [
            *dsh_command(),
            "--profile", "graph", "run",
            "--mode", "loop",
            "--task", "Call wf_next repeatedly until it reports done. Do nothing else.",
            "--budget", "1", "--graphs", str(graphs), "--workdir", str(work),
            "--python-bin", str(venv_bin()), "--knobs", json.dumps(knobs),
        ],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=900,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    facts = json.loads((work / FACTS).read_text())
    assert facts["bound"] == "epochs"
    assert facts["best"] > 0
    # The number the runtime reported is the number the walker reports for the
    # same seed, which is what makes the two runtimes interchangeable.
    _package, walked = _run(tmp_path / "walk", epochs=2, batch_size=2, max_evaluations=4)
    assert facts["best"] == walked.facts["best"]
