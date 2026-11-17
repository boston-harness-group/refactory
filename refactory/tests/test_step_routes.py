"""A step's own model, effort and output cap (`Node.route`), chosen among the run's routes.

- **Both runtimes** make the step's model call on its route: the walker's client and the
  spine's dispatched worker each start their DSH process on it (a fake dsh records what
  each one was handed), and the step beside it stays on the run's route.
- **Bounded**: a core operator sets ``route`` on a new agent step; the attempt refuses any
  value no route of the run uses, or the one the step runs on already, naming the allowed
  values (`workers.StepRoutes`), and refuses every route when the run has none to choose.
- **Said once**: the request has one ``route`` line, only when the run has a choice; the
  view shows a step's route only when it has one of its own, in the legend's one form.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

import structure_fuzz as sf
from refactory_dsh import cli, registry, workers
from refactory_dsh.model import DshModelClient
from refactory_dsh.nn import Graph, Node, core_ops, pipeline_view as pv
from refactory_dsh.nn.graph import GraphError
from refactory_dsh.optim import Candidate
from refactory_dsh.optim import base
from refactory_dsh.tools.request_sizes import repeated_sentences
from refactory_dsh.workers import StepRoutes
from test_model_choice import _PACKAGE_INIT, _workers, fake_dsh  # noqa: F401 - a fixture
from tests_support import store_model_choice

MODE = "choicepack/ask"
ROUTE = {"model": "small-model", "effort": "low", "max_completion_tokens": "900"}

_ROUTED_MODE = f'''\
"""Draft a line, then polish it on a route of its own."""
from refactory_dsh import nn


class Ask(nn.Mode):
    name = "ask"
    description = "draft a line, then polish it"

    def build(self):
        graph = nn.Graph()
        graph.add(nn.Node("draft", "agent", prompt_template="Write a line of verse.",
                          writes=("draft.md",), produces=("draft",)))
        graph.add(nn.Node("polish", "agent", prompt_template="Polish the verse in draft.md.",
                          reads=("draft.md",), writes=("final.md",), produces=("final",),
                          route={ROUTE!r}))
        graph.connect("draft", "polish")
        return graph

    def score(self, run):
        return float(bool(run.facts.get("final")))
'''


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """`fake_dsh`'s project: the two-step mode, its second step on a route of its own."""
    root = tmp_path / "project"
    package = root / ".factory" / "packages" / "choicepack"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(_PACKAGE_INIT)
    (package / "mode.py").write_text(_ROUTED_MODE)
    monkeypatch.setenv("REFACTORY_PROJECT", str(root))
    return root


def _asked(run: Path) -> dict[str, tuple[str, str, str, str]]:
    """Each step's model call: the model, effort and cap its client asked for, and the
    model its DSH process was started on."""
    return {r["node_id"]: (r["requested_route"]["model"], r["requested_route"]["effort"],
                           r["requested_route"].get("max_completion_tokens", ""),
                           r["actual_route"]["model"]) for r in _workers(run)}


def _store_choice() -> None:
    import os

    store_model_choice(Path(os.environ["DSH_HOME"]), "chosen-prov", "chosen-model")


def test_the_spine_s_worker_makes_the_step_s_call_on_its_route(tmp_path, fake_dsh, capsys):
    _store_choice()
    run = tmp_path / "run"
    assert cli.main(["ceo", MODE, "--workdir", str(run), "--headless"]) == 0, \
        capsys.readouterr().err
    assert _asked(run) == {"draft": ("chosen-model", "", "", "chosen-model"),
                           "polish": ("small-model", "low", "900", "small-model")}


def test_the_walker_makes_the_step_s_call_on_its_route(tmp_path, fake_dsh, project):
    _store_choice()
    _package, mode = registry.resolve(project, MODE)
    run = tmp_path / "walked"
    result = mode().run(run, model=DshModelClient(project=project))
    assert result.exit_ok, result.error
    assert _asked(run) == {"draft": ("chosen-model", "", "", "chosen-model"),
                           "polish": ("small-model", "low", "900", "small-model")}


def test_a_route_is_refused_where_no_worker_would_take_it():
    """The spine hands an undispatched agent to the driving agent, which would drop the
    route: the lowering refuses it there, as it refuses a workspace."""
    from refactory_dsh.ir import AgentNode, AgentRole, Workflow

    flow = Workflow(name="w", start_node="a", nodes={"a": AgentNode(
        id="a", role=AgentRole("ceo"), writes={"a.txt"}, route={"model": "m"})}, edges=[])
    with pytest.raises(ValueError, match="only a dispatched worker can take"):
        workers.lower_workflow(flow, Path("."), model_context=None)


def test_a_step_route_is_an_agent_s_and_of_known_fields():
    def agent(**route):
        return Node("a", "agent", prompt_template="p", writes=("a.txt",), **route)

    assert agent(route={"max_completion_tokens": 900}).route == {"max_completion_tokens": "900"}
    assert agent(route={"model": ""}).route is None
    for bad in ({"temperature": "1"}, {"max_completion_tokens": "lots"}, {"model": 3}, "m"):
        with pytest.raises(GraphError):
            agent(route=bad)
    with pytest.raises(GraphError, match="not an agent"):
        Node("o", "op", call="m:f", route={"model": "m"})
    graph = Graph()
    graph.add(agent(route={"effort": "low"}))
    assert Graph.from_state(graph.to_state()).nodes["a"].route == {"effort": "low"}
    assert "route" not in agent().to_state()


# ── the run's routes bound the choice ────────────────────────────────────


def _routes(**maps) -> StepRoutes:
    return StepRoutes(sf.ROUTE_BASE, **maps)


def test_a_run_with_one_route_offers_nothing_and_its_maps_offer_their_values():
    assert _routes().offered() == {}
    routes = _routes(node_routes=sf.ROUTE_MAPS)
    assert routes.offered() == {"model": ["m-small"], "effort": ["low"],
                                "max_completion_tokens": ["900"]}
    # A role on another backend offers its values to that role's steps only.
    other = _routes(role_routes={"coder": {"worker_backend": "claude_code", "model": "cc"}})
    assert other.offered() == {}
    assert "allowed: cc" in other.refusal("x", "coder", {"model": "m-big"})


def _attempt(edit, routes=StepRoutes(sf.ROUTE_BASE, node_routes=sf.ROUTE_MAPS)):
    applier = sf.Applier(sf.seeds()["fuzz_draft_review"])
    applier.optimizer.step_routes = routes
    parent = Candidate.from_state(applier.seed_state)
    return applier, applier.attempt(parent, edit, {parent.id})


@pytest.mark.parametrize("op", [
    {"op": "insert", "after": "draft", "kind": "agent"},
    {"op": "split", "step": "review", "into": [{"kind": "agent"}]},
], ids=["insert", "split into"])
def test_a_new_step_runs_on_a_route_the_run_offers(op):
    applier, out = _attempt({**op, **({"route": {"model": "m-small", "effort": "low"}}
                                      if op["op"] == "insert" else
                                      {"into": [{"kind": "agent", "route": {"model": "m-small"}}]})})
    assert isinstance(out, Candidate), out
    routed = {n.id: n.route for n in Graph.from_state(out.state["graph"]).nodes.values() if n.route}
    assert list(routed.values()) in ([{"model": "m-small", "effort": "low"}], [{"model": "m-small"}])
    assert sf.check_state(applier, out.state, walker=True) == []
    # A copy of that step (an insert with every default) runs on its route too.
    step = next(iter(routed))
    copied = applier.attempt(out, {"op": "insert", "after": step}, {out.id})
    assert isinstance(copied, Candidate), copied
    assert sum(n.route == routed[step] for n in
               Graph.from_state(copied.state["graph"]).nodes.values()) == 2


@pytest.mark.parametrize("route, said", [
    ({"model": "m-other"}, "route model 'm-other', which no route of this run uses; allowed: "
                           "m-big, m-small"),
    ({"model": "m-big"}, "route model 'm-big', which it runs on already; leave it out"),
    ({"effort": "high"}, "allowed: low"),
])
def test_a_route_outside_the_run_s_is_refused_naming_the_allowed_values(route, said):
    _applier, out = _attempt({"op": "insert", "after": "draft", "kind": "agent", "route": route})
    assert isinstance(out, list) and out[0].startswith("E-route: ") and said in out[0], out


def test_a_run_with_no_routes_refuses_every_step_route():
    _applier, out = _attempt({"op": "insert", "after": "draft", "route": {"model": "m-small"}},
                             routes=None)
    assert isinstance(out, list) and re.fullmatch(
        r"E-route: step '\w+' sets a route of its own, and this run has none to choose from",
        out[0]), out


@pytest.mark.parametrize("edit, said", [
    ({"op": "insert", "after": "draft", "kind": "code", "route": {"model": "m-small"}},
     "only an agent step takes a 'route'"),
    ({"op": "insert", "after": "draft", "route": {"temperature": "1"}}, "unknown: ['temperature']"),
])
def test_a_malformed_route_is_refused_before_anything_is_built(edit, said):
    _applier, out = _attempt(edit)
    assert isinstance(out, list) and said in out[0], out


# ── said once: the request and the view ───────────────────────────────────


def test_the_view_shows_a_step_s_own_route_in_the_legend_s_form():
    _applier, out = _attempt({"op": "insert", "after": "draft", "kind": "agent",
                              "route": {"model": "m-small", "max_completion_tokens": "900"}})
    lines = [line for line in pv.render(out.state).splitlines() if " on " in line]
    assert len(lines) == 1, lines
    assert re.match(r"^\w+ agent on model=m-small max_completion_tokens=900  sees ", lines[0])
    # The legend's one form is the reader's own line, with placeholders.
    assert pv.FORMS["step route"] in base.PIPELINE_VIEW_LEAD
    assert re.fullmatch(r"step agent on model=\S+ effort=\S+ max_completion_tokens=\S+",
                        pv.FORMS["step route"])


def _request(tmp_path, offered):
    from refactory_dsh import strategies
    from refactory_dsh.loop import LoopConfig

    loop = strategies.create("default", "livemath/solver", LoopConfig(
        epochs=1, batch_size=1, optimizer="random", seed=1, runs_root=tmp_path / "run",
        edit_preset="broad"), project=Path("."))
    seed = Candidate.from_state(loop.mode.state_dict())
    seed.evaluated, seed.score = True, 0.5
    loop.population.append(seed)
    ctx = replace(loop._context(), step_routes=offered)
    from refactory_dsh.optim import analysts

    return ctx.render(), analysts._changes_text(ctx)


def test_the_request_says_route_once_and_only_when_the_run_offers_one(tmp_path):
    offered = _routes(node_routes=sf.ROUTE_MAPS).offered()
    for text in _request(tmp_path, offered):
        assert len(re.findall(r"(?m)^- `route` on a new agent step", text)) == 1, text
        assert "model m-small; effort low; max_completion_tokens 900" in text
        assert re.search(r"`insert` \([^)]*\broute\b", text)
        assert repeated_sentences(text) == []
    for text in _request(tmp_path, {}):
        assert "on a new agent step" not in text
        assert not re.search(r"`(insert|branch|merge)` \([^)]*\broute\b", text)


def test_the_route_line_s_example_applies():
    offered = _routes(node_routes=sf.ROUTE_MAPS).offered()
    example = json.loads(core_ops.route_field_line(offered).rsplit("E.g. ", 1)[1])
    _applier, out = _attempt({"op": "insert", "after": "draft", **example})
    assert isinstance(out, Candidate), out


def test_the_node_schema_leaves_route_to_its_one_line():
    from refactory_dsh.nn.node_schema import render_schema

    text = "\n".join(render_schema())
    assert "| route |" not in text and ", route" not in text
