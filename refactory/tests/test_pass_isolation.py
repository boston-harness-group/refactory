"""Pass isolation on both runtimes: the PR #561 review findings, each as a scenario.

A data node runs its subgraph once per item in its own pass directory. Each test
here runs one small graph on the walker (in process) and on the spine (lowered and
driven through the real TS drain, as test_runtime_equivalence's data tests do), and
checks both against the same expectation:

1. a file written upstream of the data node is readable inside a pass;
2. a resume keeps the scores of the items finished before the stop, and a plain
   rerun over leftover passes starts clean;
3. a fact edited in place inside a pass stays in that pass;
4. a pass cannot write through a read-through link to a run-level input;
5. an agent inside a pass sees the person's steering.

And the review's surviving mutants: a declared non-score fact merges back (M10),
sanitised ids carry a hash (M11) and replace ``/`` (M12), and an id too long for a
file name is bounded (finding 6).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from refactory_dsh import lowering, nn
from refactory_dsh.nn.executor import CURRENT_ITEM, execute

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
NODE = shutil.which("node")
DRAIN = REPO_ROOT / "packages/workflow/workflow-spine/tests/fixtures/data-drain.mts"
DRAIN_AGENT = REPO_ROOT / "packages/workflow/workflow-spine/tests/fixtures/data-drain-agent.mts"
PYTHON_BIN = str(REPO_ROOT / ".venv/bin")

needs_spine = pytest.mark.skipif(
    NODE is None
    or not (REPO_ROOT / "packages/workflow/workflow-spine/node_modules/@deepseek-ai/cordis").exists(),
    reason="the spine drain needs Node and the installed workspace (pnpm install)",
)

ITEMS = ({"id": "a"}, {"id": "b"}, {"id": "c"})

_SCORED_TASK = '''\
from refactory_dsh.task import Task, TaskInstance, VerifyResult

SCORES = {"a": 0.1, "b": 0.2, "c": 0.3}


class ScoredTask(Task):
    name = "pass_isolation_scored"

    def instances(self):
        return [TaskInstance(id=key, metadata={}) for key in SCORES]

    def setup(self, instance, workspace):
        pass

    def verify(self, instance, workspace):
        return VerifyResult(passed=True, score=SCORES[instance.id])
'''


@pytest.fixture
def scored_task(tmp_path):
    """A Task scoring a=0.1, b=0.2, c=0.3: the walker's name and the spine's reference."""
    from refactory_dsh.task import register_task

    project = tmp_path / "project"
    pkg = project / ".factory" / "packages" / "pass_isolation_pkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "task.py").write_text(_SCORED_TASK)
    sys.path.insert(0, str(project / ".factory" / "packages"))
    try:
        from pass_isolation_pkg.task import ScoredTask  # type: ignore[import-not-found]

        register_task("pass_isolation_scored", ScoredTask)
        yield project, "pass_isolation_scored", "refactory_packages.pass_isolation_pkg.task:ScoredTask"
    finally:
        sys.path.remove(str(project / ".factory" / "packages"))
        sys.modules.pop("pass_isolation_pkg.task", None)
        sys.modules.pop("pass_isolation_pkg", None)


def _graph(body: list[nn.Node], *, upstream: nn.Node | None = None, items=ITEMS,
           produces: tuple[str, ...] = ("item_scores",), task_ref: str = "") -> nn.Graph:
    """``[upstream] -> data -> report``; the data node runs *body* (a chain) per item."""
    graph = nn.Graph()
    if upstream is not None:
        graph.add(upstream)
    graph.add(nn.Node(
        "data", "data", subgraph_entry=body[0].id, subgraph_exit=body[-1].id,
        writes=(CURRENT_ITEM,), produces=produces, task_ref=task_ref,
        inline_items=() if task_ref else tuple(dict(item) for item in items),
    ))
    for node in body:
        graph.add(node)
    for first, second in zip(body, body[1:]):
        graph.connect(first.id, second.id)
    graph.add(nn.Node("report", "op", call="loop_ops:write_best", writes=("best.json",), produces=("best",)))
    if upstream is not None:
        graph.connect(upstream.id, "data")
    graph.connect("data", "report")
    return graph


def _spine(graph: nn.Graph, workdir: Path, tmp_path: Path, *, project: Path | None = None,
           env: dict[str, str] | None = None, agent_responses: dict[str, str] | None = None) -> int:
    """Lower *graph* into *workdir* and drive it through the real spine drain."""
    workdir.mkdir(parents=True, exist_ok=True)
    state = {"graph": graph.to_state()}
    if not (workdir / ".refactory" / "declarations.json").exists():
        lowering.write_declarations(state, workdir)
    graph_path, _ = lowering.write_mode_graph(state, tmp_path / f"graphs-{workdir.name}", name="pass_isolation")
    run_env = {**os.environ, "REFACTORY_WORKDIR": str(workdir),
               "PYTHONPATH": os.pathsep.join(p for p in (
                   str(project / ".factory" / "packages") if project else "", str(HERE)) if p),
               **(env or {})}
    if project is not None:
        run_env["REFACTORY_PROJECT"] = str(project)
    run_env["PATH"] = f"{PYTHON_BIN}:{run_env.get('PATH', '')}"
    argv = [str(NODE), "--import", "tsx/esm", str(DRAIN_AGENT if agent_responses is not None else DRAIN),
            str(graph_path), str(workdir), f"{PYTHON_BIN}/python", "10000"]
    if agent_responses is not None:
        responses = tmp_path / f"responses-{workdir.name}.json"
        responses.write_text(json.dumps(agent_responses))
        argv.append(str(responses))
    done = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180, env=run_env)
    if done.returncode != 0:
        sys.stderr.write(f"spine drain: rc={done.returncode}\nstdout: {done.stdout[-2000:]}\n"
                         f"stderr: {done.stderr[-3000:]}\n")
    return done.returncode


def _facts(workdir: Path) -> dict[str, Any]:
    return json.loads((workdir / "facts.json").read_text())


def _scores(records: list[dict[str, Any]]) -> list[tuple[str, float]]:
    return sorted((str(r["id"]), r.get("score")) for r in records)


# ── 1. upstream files are read through ──────────────────────────────────────


def _upstream_graph() -> nn.Graph:
    return _graph(
        [nn.Node("work", "op", call="loop_ops:read_prompt", reads=(CURRENT_ITEM, "prompt.txt"))],
        upstream=nn.Node("prep", "op", call="loop_ops:write_prompt", writes=("prompt.txt",)),
    )


def test_walker_pass_reads_a_file_written_upstream(tmp_path):
    """``prep`` writes prompt.txt before the data node; every pass reads it.

    Fails when the read-through set subtracts every node's writes (prep's too)."""
    result = execute(_upstream_graph(), tmp_path / "run")
    assert result.exit_ok, result.error
    assert (tmp_path / "run/passes/data/a_0/prompt.txt").read_text() == "from prep"


@needs_spine
def test_spine_pass_reads_a_file_written_upstream(tmp_path):
    """The same graph on the spine: the loader reads the pass's body from the declarations."""
    run = tmp_path / "spine"
    assert _spine(_upstream_graph(), run, tmp_path) == 0
    assert (run / "passes/data/c_2/prompt.txt").read_text() == "from prep"
    assert "subgraph_nodes" in json.loads((run / ".refactory/declarations.json").read_text())["data"]


def test_data_body_is_the_subgraph_only():
    """``Graph.data_body`` holds entry..exit, not the nodes around the data node."""
    graph = _graph(
        [nn.Node("w1", "op", call="loop_ops:read_prompt"), nn.Node("w2", "op", call="loop_ops:read_prompt")],
        upstream=nn.Node("prep", "op", call="loop_ops:write_prompt", writes=("prompt.txt",)),
    )
    assert graph.data_body("data") == {"w1", "w2"}


# ── 2. resume keeps finished scores; a rerun starts clean ───────────────────


def _resume_graph(task_ref: str) -> nn.Graph:
    return _graph(
        [nn.Node("work", "op", call="loop_ops:log_and_stop_on_b", reads=(CURRENT_ITEM,))],
        task_ref=task_ref,
    )


def _ran(log: Path) -> list[str]:
    return log.read_text().split()


def test_walker_resume_keeps_the_scores_of_finished_items(tmp_path, scored_task, monkeypatch):
    """Stop on b, resume at the data node: a runs once and its score is kept.

    Fails when the skipped items' scores are not taken from where the pass kept
    them (the result is ``[b, c]``)."""
    _, walker_ref, _ = scored_task
    log = tmp_path / "walker.log"
    monkeypatch.setenv("PASS_TEST_LOG", str(log))
    run = tmp_path / "run"
    first = execute(_resume_graph(walker_ref), run)
    assert first.halted, first.error
    resumed = execute(_resume_graph(walker_ref), run, start="data", facts=dict(first.facts))
    assert resumed.exit_ok, resumed.error
    assert _ran(log) == ["a", "b", "b", "c"]
    expected = [("a", 0.1), ("b", 0.2), ("c", 0.3)]
    assert _scores(resumed.facts["item_scores"]) == expected
    assert _scores(resumed.item_records) == expected


def test_walker_plain_rerun_over_leftover_passes_runs_every_item(tmp_path, scored_task, monkeypatch):
    """A fresh ``execute`` (no ``start``) after a stopped run runs a, b and c again.

    Fails when any leftover ``.done`` pass is skipped outside a resume."""
    _, walker_ref, _ = scored_task
    log = tmp_path / "walker.log"
    monkeypatch.setenv("PASS_TEST_LOG", str(log))
    run = tmp_path / "run"
    assert execute(_resume_graph(walker_ref), run).halted
    log.write_text("")
    again = execute(_resume_graph(walker_ref), run)
    assert again.exit_ok, again.error
    assert _ran(log) == ["a", "b", "c"]
    assert _scores(again.facts["item_scores"]) == [("a", 0.1), ("b", 0.2), ("c", 0.3)]


@needs_spine
def test_walker_and_spine_agree_on_scores_after_a_resume(tmp_path, scored_task, monkeypatch):
    """The same stop-and-resume on both runtimes ends with the same scores, a run once.

    The walker halts on b; the spine's drain is killed in b's command and goes on
    from its checkpoint (DRAIN_RESUME)."""
    project, walker_ref, spine_ref = scored_task
    walker_log = tmp_path / "walker.log"
    monkeypatch.setenv("PASS_TEST_LOG", str(walker_log))
    walker = tmp_path / "walker"
    first = execute(_resume_graph(walker_ref), walker)
    assert first.halted
    walker_result = execute(_resume_graph(walker_ref), walker, start="data", facts=dict(first.facts))
    assert walker_result.exit_ok, walker_result.error

    spine_log = tmp_path / "spine.log"
    spine = tmp_path / "spine"
    env = {"PASS_TEST_LOG": str(spine_log), "PASS_TEST_RUNTIME": "spine"}
    assert _spine(_resume_graph(spine_ref), spine, tmp_path, project=project, env=env) != 0
    assert _ran(spine_log) == ["a", "b"], "the spine run did not stop in item b"
    assert _spine(_resume_graph(spine_ref), spine, tmp_path, project=project,
                  env={**env, "DRAIN_RESUME": "1"}) == 0

    assert _ran(spine_log) == _ran(walker_log) == ["a", "b", "b", "c"]
    assert _scores(_facts(spine)["item_scores"]) == _scores(walker_result.facts["item_scores"]) == [
        ("a", 0.1), ("b", 0.2), ("c", 0.3)]


# ── 3. in-place fact edits stay in the pass ──────────────────────────────────


def _shared_graph() -> nn.Graph:
    return _graph(
        [nn.Node("work", "op", call="loop_ops:append_shared_in_place", reads=(CURRENT_ITEM,),
                 writes=("shared_seen.json",))],
        upstream=nn.Node("prep", "op", call="loop_ops:seed_shared", produces=("shared",)),
    )


def _seen_shared(run: Path) -> dict[str, Any]:
    return {name: json.loads((run / f"passes/data/{name}/shared_seen.json").read_text())
            for name in ("a_0", "b_1", "c_2")}


def test_in_place_fact_edits_stay_in_their_pass_on_both_runtimes(tmp_path):
    """Each item sees ``['seed']`` and the run ends with ``['seed']``: ``shared`` is undeclared.

    Fails on the walker with a shallow copy of the facts (the list is shared)."""
    walker = tmp_path / "walker"
    result = execute(_shared_graph(), walker)
    assert result.exit_ok, result.error
    expected = {"a_0": ["seed"], "b_1": ["seed"], "c_2": ["seed"]}
    assert _seen_shared(walker) == expected
    assert result.facts["shared"] == ["seed"]
    if NODE is None:
        pytest.skip("the spine half needs Node")
    spine = tmp_path / "spine"
    assert _spine(_shared_graph(), spine, tmp_path) == 0
    assert _seen_shared(spine) == expected
    assert _facts(spine)["shared"] == ["seed"]


# ── 4. no write through a read-through link ──────────────────────────────────


def _tamper_graph() -> nn.Graph:
    return _graph([nn.Node("work", "op", call="loop_ops:tamper_run_input",
                           reads=(CURRENT_ITEM, "run_input.txt"))])


def _check_untampered(run: Path) -> None:
    assert (run / "run_input.txt").read_text() == "run-level-data"
    for name, item in (("a_0", "a"), ("b_1", "b"), ("c_2", "c")):
        assert (run / f"passes/data/{name}/input_seen.txt").read_text() == "run-level-data", name
        own = run / f"passes/data/{name}/run_input.txt"
        assert not own.is_symlink() and own.read_text() == f"tampered by {item}"


def test_walker_pass_cannot_write_through_a_read_through_link(tmp_path):
    """Fails when the write follows the link (b reads "tampered by a")."""
    run = tmp_path / "walker"
    run.mkdir()
    (run / "run_input.txt").write_text("run-level-data")
    result = execute(_tamper_graph(), run)
    assert result.exit_ok, result.error
    _check_untampered(run)


@needs_spine
def test_spine_pass_cannot_write_through_a_read_through_link(tmp_path):
    run = tmp_path / "spine"
    run.mkdir()
    (run / "run_input.txt").write_text("run-level-data")
    assert _spine(_tamper_graph(), run, tmp_path) == 0
    _check_untampered(run)


def test_detach_read_through_copies_a_linked_directory_shallowly(tmp_path):
    """A write under a linked directory replaces the link by a directory of links."""
    from refactory_dsh.items import detach_read_through, prepare_pass_dir

    run = tmp_path / "run"
    (run / "docs").mkdir(parents=True)
    (run / "docs/keep.md").write_text("keep")
    pass_dir = prepare_pass_dir(run, "data", "a", 0, declared_reads=("docs",))
    assert (pass_dir / "docs").is_symlink()
    detach_read_through(pass_dir, "docs/new.md")
    assert not (pass_dir / "docs").is_symlink() and (pass_dir / "docs/keep.md").read_text() == "keep"
    (pass_dir / "docs/new.md").write_text("new")
    assert not (run / "docs/new.md").exists()


# ── 5. an agent in a pass sees the steering ─────────────────────────────────

NOTE = "Always answer in French, said the person."


def _agent_graph() -> nn.Graph:
    return _graph([nn.Node("ask", "agent", role="builder", prompt_template="Process the item.",
                           reads=(CURRENT_ITEM,), writes=())], items=({"id": "a"},))


def _seed_ledger(run: Path) -> None:
    from refactory_dsh.steering import Ledger

    run.mkdir(parents=True, exist_ok=True)
    Ledger.seed(run, "the brief")
    Ledger(run).note("correction", NOTE)


class _Recording:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt, *, system="", workdir=None, **kw):
        self.prompts.append(prompt)
        return "ok"

    def for_node(self, node, role=""):
        return self

    def with_wall(self, wall):
        return self

    def with_budget(self, budget):
        return self

    def context(self):
        return {}

    def route(self):
        return {"worker_backend": "dsh"}


def test_walker_agent_in_a_pass_sees_the_steering(tmp_path):
    """Fails when the pass does not read the run's intent ledger through."""
    from refactory_dsh.steering import STEERING_START

    run = tmp_path / "walker"
    _seed_ledger(run)
    model = _Recording()
    result = execute(_agent_graph(), run, model=model)
    assert result.exit_ok, result.error
    assert model.prompts, "the agent was never called"
    assert STEERING_START in model.prompts[0] and NOTE in model.prompts[0]


@needs_spine
def test_spine_agent_in_a_pass_sees_the_steering(tmp_path):
    """The spine's delivery bridge runs in the pass directory; its prompt carries the note."""
    from refactory_dsh.steering import STEERING_START

    run = tmp_path / "spine"
    _seed_ledger(run)
    assert _spine(_agent_graph(), run, tmp_path, agent_responses={"ask": "ok"}) == 0
    prompts = list((run / "passes/data/a_0/.refactory/agent-delivery").glob("*/prompt.txt"))
    assert prompts, "no delivery in the pass directory"
    text = prompts[0].read_text()
    assert STEERING_START in text and NOTE in text


def test_pass_reads_the_run_config_and_ledger_read_only(tmp_path):
    """run-config.json and the ledger are linked into a pass; a ctx write detaches them."""
    from refactory_dsh.items import prepare_pass_dir
    from refactory_dsh.nn.executor import RunContext

    run = tmp_path / "run"
    (run / ".refactory").mkdir(parents=True)
    (run / ".refactory/run-config.json").write_text('{"omit_steering": true}')
    (run / "intent-ledger.md").write_text("ledger")
    pass_dir = prepare_pass_dir(run, "data", "a", 0)
    assert json.loads((pass_dir / ".refactory/run-config.json").read_text()) == {"omit_steering": True}
    ctx = RunContext(pass_dir, enforce_writes=False)
    ctx.append_text("intent-ledger.md", "\nfrom the pass")
    assert (run / "intent-ledger.md").read_text() == "ledger"
    assert (pass_dir / "intent-ledger.md").read_text() == "ledger\nfrom the pass"


# ── review mutants M10, M11, M12 and finding 6 ──────────────────────────────


def _note_graph() -> nn.Graph:
    return _graph([nn.Node("work", "op", call="loop_ops:note_item", reads=(CURRENT_ITEM,), produces=("note",))],
                  produces=("item_scores", "note"))


def test_declared_non_score_fact_merges_back_on_both_runtimes(tmp_path):
    """M10: the data node declares ``note``; the run ends with the last item's value."""
    result = execute(_note_graph(), tmp_path / "walker")
    assert result.exit_ok, result.error
    assert result.facts["note"] == "from c"
    if NODE is None:
        pytest.skip("the spine half needs Node")
    spine = tmp_path / "spine"
    assert _spine(_note_graph(), spine, tmp_path) == 0
    assert _facts(spine)["note"] == "from c"


def test_sanitised_ids_carry_a_hash_and_replace_separators():
    """M11, M12: ``a/b`` and ``a_b`` must not share a directory."""
    from refactory_dsh.items import sanitize_item_id

    slashed = sanitize_item_id("a/b")
    assert "/" not in slashed and slashed.startswith("a_b_") and len(slashed) == len("a_b_") + 6
    assert sanitize_item_id("a_b") == "a_b"
    assert sanitize_item_id("x:y") != sanitize_item_id("x_y")


def test_a_very_long_item_id_gets_a_bounded_unique_directory(tmp_path):
    """Finding 6: a 300-byte id fits a file name and stays unique."""
    from refactory_dsh.items import sanitize_item_id

    first, second = "x" * 300, "x" * 299 + "y"
    assert len(sanitize_item_id(first).encode()) < 220
    assert sanitize_item_id(first) != sanitize_item_id(second)
    long_items = ({"id": first}, {"id": second})
    graph = _graph([nn.Node("work", "op", call="loop_ops:note_item", reads=(CURRENT_ITEM,), produces=("note",))],
                   items=long_items, produces=("item_scores", "note"))
    result = execute(graph, tmp_path / "run")
    assert result.exit_ok, result.error
    assert len(list((tmp_path / "run/passes/data").iterdir())) == 2
