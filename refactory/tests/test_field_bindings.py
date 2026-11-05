"""Field bindings on both runtimes (F14): each bindable field, run on the walker and
through the real spine drain, reaches the run as its bound knob's value.

``system_prompt_template`` has its own two-runtime test in test_nn.py
(``test_two_runtime_system_prompt_knob``); this file covers ``prompt_template``,
``max_iterations`` (gate and agent) and ``parallelism``, and V14's knob-kind check,
which makes a mismatched binding fail at validation on both runtimes alike.
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

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DRAIN_AGENT = REPO_ROOT / "packages/workflow/workflow-spine/tests/fixtures/data-drain-agent.mts"
PYTHON_BIN = REPO_ROOT / ".venv/bin"
NODE = shutil.which("node")

needs_spine = pytest.mark.skipif(
    not NODE or not (REPO_ROOT / "packages/workflow/workflow-spine/node_modules/@deepseek-ai/cordis").exists(),
    reason="the spine drain needs Node and the installed workspace (pnpm install)",
)


def _knob(value: Any, kind: str) -> dict[str, Any]:
    return {"value": value, "spec": nn.Knob(value, kind).spec()}


def _state(graph: nn.Graph, **knobs: tuple[Any, str]) -> dict[str, Any]:
    return {"version": nn.STATE_VERSION, "graph": graph.to_state(),
            "knobs": {name: _knob(value, kind) for name, (value, kind) in knobs.items()}}


def _spine(state: dict[str, Any], run: Path, tmp_path: Path, *,
           responses: dict[str, str] | None = None, env: dict[str, str] | None = None) -> int:
    """Lower ``state`` into ``run`` and drive it through the real spine drain."""
    run.mkdir(parents=True, exist_ok=True)
    lowering.write_declarations(state, run)
    graph_path, _ = lowering.write_mode_graph(state, tmp_path / f"graphs-{run.name}", name="bindings")
    responses_file = tmp_path / f"responses-{run.name}.json"
    responses_file.write_text(json.dumps(responses or {}))
    run_env = {**os.environ, "REFACTORY_WORKDIR": str(run), "PYTHONPATH": str(HERE),
               **lowering.knob_environment(state), **(env or {})}
    run_env["PATH"] = f"{PYTHON_BIN}:{run_env.get('PATH', '')}"
    done = subprocess.run(
        [str(NODE), "--import", "tsx/esm", str(DRAIN_AGENT), str(graph_path), str(run),
         f"{PYTHON_BIN}/python", "30000", str(responses_file)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180, env=run_env)
    if done.returncode != 0:
        sys.stderr.write(f"spine drain rc={done.returncode}\nstdout: {done.stdout[-2000:]}\n"
                         f"stderr: {done.stderr[-3000:]}\n")
    return done.returncode


class _Model:
    """Records each prompt the walker sends."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt, *, system="", workdir=None, **kw):
        self.prompts.append(prompt)
        return "reply"

    def for_node(self, n, role=""):
        return self

    def with_wall(self, t):
        return self

    def with_budget(self, b):
        return self

    def with_tool_policy(self, p):
        return self

    def context(self):
        return {}

    def route(self):
        return {"worker_backend": "dsh"}


# ── prompt_template ───────────────────────────────────────────────────────

@needs_spine
def test_a_bound_prompt_template_reaches_the_model_on_both_runtimes(tmp_path):
    """The bound knob's text, not the node's own template, is the prompt each runtime
    sends, with its {{slot}} rendered from the run's knobs. Both runtimes."""
    graph = nn.Graph()
    graph.add(nn.Node("a", "agent", prompt_template="Default prompt about {{topic}}",
                      writes=("out.txt",), knobs=("pt", "topic"),
                      field_bindings=(("prompt_template", "pt"),)))
    state = _state(graph, pt=("Bound prompt about {{topic}}", "str"), topic=("physics", "str"))

    model = _Model()
    walker = execute(graph, tmp_path / "walker", model=model,
                     knobs={"pt": "Bound prompt about {{topic}}", "topic": "physics"})
    assert walker.exit_ok, walker.error
    assert len(model.prompts) == 1
    assert "Bound prompt about physics" in model.prompts[0]
    assert "Default prompt" not in model.prompts[0]

    record = tmp_path / "spine-record.jsonl"
    assert _spine(state, tmp_path / "spine", tmp_path, responses={"a": "reply"},
                  env={"DRAIN_RECORD": str(record)}) == 0
    sent = [json.loads(line) for line in record.read_text().splitlines()]
    assert [entry["node"] for entry in sent] == ["a"]
    assert "Bound prompt about physics" in sent[0]["prompt"]
    assert "Default prompt" not in sent[0]["prompt"]


# ── max_iterations ────────────────────────────────────────────────────────

def _gate_loop() -> nn.Graph:
    """body -> gate (always retry, max_iterations 8 bound to ``cap``) -> body."""
    graph = nn.Graph()
    graph.add(nn.Node("body", "op", call="loop_ops:count_visit",
                      writes=("visits.json",), produces=("visits",)))
    graph.add(nn.Node("gate", "gate", evaluator="loop_ops:always_retry", max_iterations=8,
                      knobs=("cap",), field_bindings=(("max_iterations", "cap"),)))
    graph.add(nn.Node("done", "op", call="loop_ops:write_best",
                      writes=("best.json",), produces=("best",)))
    graph.connect("body", "gate")
    graph.connect("gate", "body", condition="retry")
    graph.connect("gate", "done", condition="proceed")
    return graph


@needs_spine
def test_a_bound_gate_max_iterations_caps_the_loop_on_both_runtimes(tmp_path, monkeypatch):
    """A gate's reloop cap is its bound knob's value (2), not its own (8): the body
    runs once plus two reloops on each runtime. Both runtimes."""
    monkeypatch.syspath_prepend(str(HERE))
    graph = _gate_loop()
    state = _state(graph, cap=(2, "int"))
    assert not [i for i in nn.validate_state(state, resolve_calls=False) if i.code == "V14"]

    walker = execute(graph, tmp_path / "walker", knobs={"cap": 2}, max_steps=100)
    assert walker.halted, walker.error
    walker_visits = json.loads((tmp_path / "walker" / "visits.json").read_text())

    assert _spine(state, tmp_path / "spine", tmp_path) == 0
    spine_visits = json.loads((tmp_path / "spine" / "visits.json").read_text())

    assert walker_visits == spine_visits == 3
    assert lowering.to_workflow(state).nodes["gate"].max_iterations == 2


def test_a_bound_agent_max_iterations_lowers_to_the_knob_value():
    """An agent node's bound max_iterations reaches the spine's IR as the knob value.
    TS runtime (via lowering); the walker reads it through the same Node.bound."""
    graph = nn.Graph()
    graph.add(nn.Node("a", "agent", prompt_template="t", writes=("out.txt",), max_iterations=8,
                      knobs=("cap",), field_bindings=(("max_iterations", "cap"),)))
    state = _state(graph, cap=(3, "int"))
    assert lowering.to_workflow(state).nodes["a"].max_iterations == 3
    assert graph.nodes["a"].bound("max_iterations", {"cap": 3}) == 3
    assert graph.nodes["a"].bound("max_iterations", {}) == 8


# ── parallelism ───────────────────────────────────────────────────────────

def _data_graph() -> nn.Graph:
    """data (parallelism bound to ``par``) -> [barrier_item] -> report."""
    graph = nn.Graph()
    graph.add(nn.Node("data", "data", subgraph_entry="body", subgraph_exit="body",
                      writes=(CURRENT_ITEM,), produces=("item_scores",),
                      inline_items=tuple({"id": str(i)} for i in range(3)),
                      knobs=("par",), field_bindings=(("parallelism", "par"),)))
    graph.add(nn.Node("body", "op", call="loop_ops:barrier_item",
                      reads=(CURRENT_ITEM,), produces=("counted",)))
    graph.add(nn.Node("report", "op", call="loop_ops:write_best",
                      writes=("best.json",), produces=("best",)))
    graph.connect("data", "report")
    return graph


@needs_spine
def test_a_bound_parallelism_runs_items_at_once_on_both_runtimes(tmp_path, monkeypatch):
    """A data node's parallelism bound to a knob (3) runs three items at once on each
    runtime: each item waits until all three are in flight, which only concurrency
    allows. With the knob at 1 the same probe times out. Both runtimes."""
    monkeypatch.syspath_prepend(str(HERE))
    graph = _data_graph()
    assert graph.nodes["data"].effective_parallelism({"par": 3}) == 3
    state = _state(graph, par=(3, "int"))
    assert lowering.to_workflow(state).nodes["data"].parallelism == 3

    def barrier(name: str) -> dict[str, str]:
        path = tmp_path / name
        path.mkdir()
        return {"BARRIER_DIR": str(path), "BARRIER_K": "3", "BARRIER_TIMEOUT_S": "20"}

    def through(name: str) -> list[str]:
        return sorted(p.name for p in (tmp_path / name).glob("through_*"))

    for key, value in barrier("walker-barrier").items():
        monkeypatch.setenv(key, value)
    walker = execute(graph, tmp_path / "walker", knobs={"par": 3}, max_steps=200)
    assert walker.exit_ok, walker.error

    assert _spine(state, tmp_path / "spine", tmp_path, env=barrier("spine-barrier")) == 0

    everyone = ["through_0", "through_1", "through_2"]
    assert through("walker-barrier") == through("spine-barrier") == everyone

    # The probe is real: one item at a time, the first item never sees three in flight.
    for key, value in {**barrier("serial-barrier"), "BARRIER_TIMEOUT_S": "1"}.items():
        monkeypatch.setenv(key, value)
    execute(graph, tmp_path / "serial", knobs={"par": 1}, max_steps=200)
    assert "through_0" not in through("serial-barrier")


def test_a_bound_parallelism_is_bounded_like_parallelism_knob():
    """A bound parallelism is held to 1..MAX_PARALLELISM and must be an integer,
    the same checks parallelism_knob gets. Both runtimes (one helper)."""
    node = _data_graph().nodes["data"]
    with pytest.raises(nn.GraphError, match="between 1 and"):
        node.effective_parallelism({"par": 0})
    with pytest.raises(nn.GraphError, match="integer"):
        node.effective_parallelism({"par": "3"})
    with pytest.raises(nn.GraphError, match="no value"):
        node.effective_parallelism({})


# ── V14: the knob's kind must match the field's ──────────────────────────

@pytest.mark.parametrize("field_name,kind", [
    ("max_iterations", "str"), ("parallelism", "str"),
    ("prompt_template", "int"), ("system_prompt_template", "int"),
])
def test_v14_refuses_a_knob_of_the_wrong_kind(field_name, kind):
    """str knob -> max_iterations/parallelism, int knob -> a template: V14 refuses it
    at validation, which both runtimes run before they start."""
    graph = nn.Graph()
    node_kind = "data" if field_name == "parallelism" else "agent"
    if node_kind == "data":
        graph = _data_graph()
        graph.nodes["data"] = graph.nodes["data"].with_field(
            "field_bindings", ((field_name, "par"),))
        state = _state(graph, par=("3" if kind == "str" else 3, kind))
    else:
        graph.add(nn.Node("a", "agent", prompt_template="t", knobs=("k",),
                          field_bindings=((field_name, "k"),)))
        state = _state(graph, k=("x" if kind == "str" else 3, kind))
    v14 = [i for i in nn.validate_state(state, resolve_calls=False) if i.code == "V14"]
    assert [i.message for i in v14] and "knob, but" in v14[0].message, v14


def test_v14_accepts_knobs_of_the_right_kind():
    graph = _data_graph()
    assert not [i for i in nn.validate_state(_state(graph, par=(3, "int")), resolve_calls=False)
                if i.code == "V14"]
    state = _state(_gate_loop(), cap=(2, "int"))
    assert not [i for i in nn.validate_state(state, resolve_calls=False) if i.code == "V14"]


def test_an_unvalidated_mismatch_fails_alike_on_both_runtimes(tmp_path):
    """A str knob bound to max_iterations that skipped validation is a GraphError from
    the same helper on both runtimes, not a lowering ValueError and a silent walker
    fallback."""
    graph = _gate_loop()
    state = _state(graph, cap=("two", "str"))
    with pytest.raises(nn.GraphError, match="must be an integer"):
        lowering.to_workflow(state)
    walker = execute(graph, tmp_path / "walker", knobs={"cap": "two"}, max_steps=100)
    assert not walker.exit_ok and "must be an integer" in (walker.error or ""), walker.error


def test_v14_refuses_an_empty_knob_name():
    graph = nn.Graph()
    graph.add(nn.Node("a", "agent", prompt_template="t", field_bindings=(("prompt_template", ""),)))
    v14 = [i for i in nn.validate_state(_state(graph), resolve_calls=False) if i.code == "V14"]
    assert any("empty name" in i.message for i in v14), v14


def test_v14_refuses_a_field_bound_twice():
    graph = nn.Graph()
    graph.add(nn.Node("a", "agent", prompt_template="t", knobs=("k1", "k2"),
                      field_bindings=(("prompt_template", "k1"), ("prompt_template", "k2"))))
    state = _state(graph, k1=("x", "str"), k2=("y", "str"))
    v14 = [i for i in nn.validate_state(state, resolve_calls=False) if i.code == "V14"]
    assert any("bound twice" in i.message for i in v14), v14


def test_v14_refuses_parallelism_bound_two_ways():
    graph = nn.Graph()
    graph.add(nn.Node("data", "data", subgraph_entry="body", subgraph_exit="body",
                      writes=(CURRENT_ITEM,), produces=("item_scores",),
                      inline_items=({"id": "0"},), parallelism_knob="par",
                      knobs=("par",), field_bindings=(("parallelism", "par"),)))
    graph.add(nn.Node("body", "op", call="loop_ops:noop", reads=(CURRENT_ITEM,)))
    v14 = [i for i in nn.validate_state(_state(graph, par=(1, "int")), resolve_calls=False)
           if i.code == "V14"]
    assert any("both parallelism_knob" in i.message for i in v14), v14


def test_a_binding_to_a_knob_the_node_does_not_consume_is_refused():
    """The bound knob must be one of the node's own knobs (graph.py), so a binding
    cannot read a knob the node does not declare it consumes."""
    with pytest.raises(nn.GraphError, match="must be one of the node's knobs"):
        nn.Node("a", "agent", prompt_template="t", knobs=("other",),
                field_bindings=(("prompt_template", "k"),))


# ── the checks that read a prompt read the bound one ─────────────────────

def _sibling_fact_graph() -> nn.Graph:
    """Branch ``a`` produces ``finding``; branch ``b``'s own prompt does not read it,
    but the knob bound to its prompt_template does."""
    graph = nn.Graph()
    graph.add(nn.Node("start", "op", call="loop_ops:noop"))
    graph.add(nn.Node("fork", "fork", targets=("a", "b")))
    graph.add(nn.Node("a", "op", call="loop_ops:noop", produces=("finding",)))
    graph.add(nn.Node("b", "agent", prompt_template="Review the code", writes=("b.md",),
                      knobs=("pt",), field_bindings=(("prompt_template", "pt"),)))
    graph.add(nn.Node("join", "join", sources=("a", "b")))
    graph.connect("start", "fork")
    graph.connect("fork", "join")
    graph.connect("a", "join")
    graph.connect("b", "join")
    return graph


def test_a_bound_branch_prompt_reading_a_siblings_fact_is_refused_on_both_runtimes(tmp_path):
    """The fork's sibling-fact check reads the prompt a branch will send (its bound
    template), on the walker (fork.py) and in lowering for the spine."""
    from refactory_dsh.lowering import LoweringError

    graph = _sibling_fact_graph()
    knobs = {"pt": "Review {{finding}}"}
    with pytest.raises(LoweringError, match="depends on a sibling's facts"):
        lowering.to_workflow(_state(graph, pt=(knobs["pt"], "str")))
    walker = execute(graph, tmp_path / "walker", model=_Model(), knobs=knobs, max_steps=50)
    assert not walker.exit_ok and "depends on a sibling's facts" in (walker.error or ""), walker.error

    # The node's own template alone is fine on both.
    lowering.to_workflow(_state(graph, pt=("Review the code", "str")))
    clean = execute(graph, tmp_path / "clean", model=_Model(), knobs={"pt": "Review the code"},
                    max_steps=50)
    assert clean.exit_ok, clean.error


def test_the_walker_warns_about_the_bound_prompts_unresolved_keys(tmp_path):
    """The walker's unresolved-key warning scans the prompt it sends (the bound one),
    as the spine scans the lowered prompt; a key only the node's own template has is
    not reported. Python runtime."""
    graph = nn.Graph()
    graph.add(nn.Node("a", "agent", prompt_template="Own {{own_only}}", writes=("out.txt",),
                      knobs=("pt",), field_bindings=(("prompt_template", "pt"),)))
    run = tmp_path / "run"
    result = execute(graph, run, model=_Model(), knobs={"pt": "Bound {{bound_only}}"})
    assert result.exit_ok, result.error
    events = [json.loads(line) for line in (run / ".refactory/events.jsonl").read_text().splitlines()
              if line.strip()]
    summaries = [e.get("summary", "") for e in events if e.get("kind") == "run_warning"]
    assert any("bound_only" in line for line in summaries), summaries
    assert not any("own_only" in line for line in summaries), summaries
