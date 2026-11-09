"""Evaluation integrity is path-aware, and the optimizer is told the rules.

The integrity issue: the metric guard claimed the scored facts are produced by the
pipeline, but it accepted a producer sitting off every path the run can take. And
the optimizer learned the validator's rules only by breaking them.
"""

from __future__ import annotations

import ast
import pathlib
from pathlib import Path

import pytest

from refactory_dsh import nn
from refactory_dsh.loop import Loop, LoopConfig
from refactory_dsh.optim import Candidate, ProposalContext
from refactory_dsh.optim.base import summarise_reasons

OPS = "loop_ops"


def _off_path_graph() -> nn.Graph:
    """A graph with a terminal, and a producer that can never reach it.

    ``start → a → gate1``; ``gate1 --go--> done`` (the terminal). ``gate1 --side-->``
    a region that loops on itself through gate2 and never reaches a terminal. The
    scored fact is produced only there, so the run can never produce it.
    """
    graph = nn.Graph()
    graph.add(nn.Node("a", "op", call=f"{OPS}:write_best", writes=("a.json",),
                      produces=("a",)))
    graph.add(nn.Node("gate1", "gate", evaluator=f"{OPS}:always_retry", max_iterations=4))
    graph.add(nn.Node("done", "op", call=f"{OPS}:write_best", produces=("done",)))
    graph.add(nn.Node("side", "op", call=f"{OPS}:write_best", produces=("best",)))
    graph.add(nn.Node("gate2", "gate", evaluator=f"{OPS}:always_retry", max_iterations=4))
    graph.connect("a", "gate1")
    graph.connect("gate1", "done", condition="go")
    graph.connect("gate1", "side", condition="side")
    graph.connect("side", "gate2")
    graph.connect("gate2", "side", condition="a")
    graph.connect("gate2", "side", condition="b")
    return graph


def test_on_path_nodes_excludes_a_region_that_cannot_reach_a_terminal():
    graph = _off_path_graph()
    on_path = nn.on_path_nodes(graph)
    assert "a" in on_path and "gate1" in on_path and "done" in on_path
    assert "side" not in on_path and "gate2" not in on_path
    assert "best" not in nn.produced_on_path(graph)


def test_a_producer_off_every_path_does_not_satisfy_the_metric_invariant():
    graph = _off_path_graph()
    issues = nn.validate(
        graph, {}, declared_knobs={}, invariants=(nn.fact_produced("best"),)
    )
    named = [i for i in issues if i.code == "I:fact_produced"]
    assert named, [str(i) for i in issues]
    assert "on a path" in named[0].message
    assert "V3" not in {i.code for i in issues}, (
        "the graph does have a terminal, so this must fail on the path, not on V3"
    )


def test_a_producer_on_a_path_satisfies_it():
    graph = _off_path_graph()
    graph.nodes["done"] = nn.Node("done", "op", call=f"{OPS}:write_best",
                                  produces=("best",))
    issues = nn.validate(
        graph, {}, declared_knobs={}, invariants=(nn.fact_produced("best"),)
    )
    assert not [i for i in issues if i.code == "I:fact_produced"]


def test_a_read_produced_only_off_path_is_refused():
    graph = nn.Graph()
    graph.add(nn.Node("a", "op", call=f"{OPS}:write_number", writes=("x.json",),
                      produces=("value",)))
    graph.add(nn.Node("gate1", "gate", evaluator=f"{OPS}:always_retry", max_iterations=4))
    graph.add(nn.Node("done", "op", call=f"{OPS}:write_best", produces=("done",)))
    graph.add(nn.Node("side", "op", call=f"{OPS}:write_best", writes=("y.json",),
                      produces=("y",)))
    graph.add(nn.Node("gate2", "gate", evaluator=f"{OPS}:always_retry", max_iterations=4))
    graph.add(nn.Node("reader", "op", call=f"{OPS}:double", reads=("y.json",),
                      writes=("z.json",), produces=("z",)))
    graph.connect("a", "reader")
    graph.connect("reader", "gate1")
    graph.connect("gate1", "done", condition="go")
    graph.connect("gate1", "side", condition="side")
    graph.connect("side", "gate2")
    graph.connect("gate2", "side", condition="a")
    graph.connect("gate2", "side", condition="b")
    codes = {
        issue.code
        for issue in nn.validate(graph, {}, declared_knobs=[])
    }
    assert "V6" in codes, "a read satisfied only off every path must be refused"


def test_the_seed_of_the_running_example_is_still_valid():
    """Path-awareness must not refuse a mode that genuinely works."""
    from refactory_dsh import registry

    _package, mode_class = registry.resolve(Path("."), "circle_packing/adaptive")
    mode = mode_class()
    assert nn.validate_state(
        mode.state_dict(), invariants=mode.invariants(), resolve_calls=True
    ) == []


# ── the rules reach the optimizer, and cannot drift from the code ──────────
def test_every_rule_the_validator_emits_is_documented():
    """The table and the code are the same set, in both directions."""
    import sys

    source = pathlib.Path(sys.modules["refactory_dsh.nn.validate"].__file__).read_text()
    emitted = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Issue":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value.startswith("V"):
                        emitted.add(arg.value)
    documented = {code for code in nn.RULES if code.startswith("V")}
    assert emitted == documented, (
        f"undocumented: {sorted(emitted - documented)}; "
        f"stale: {sorted(documented - emitted)}"
    )


def test_the_rules_text_lists_every_rule():
    text = nn.rules_text()
    for code, description in nn.RULES.items():
        assert f"`{code}`" in text
        assert description.split(",")[0][:24] in text


def test_the_optimizer_prompt_states_the_rules():
    described = nn.Mode.__subclasses__()
    assert described is not None
    ctx = ProposalContext(knobs=[], nodes=[], edges=[], catalogue=nn.edits.catalogue())
    prompt = ctx.render()
    assert "## The rules a proposal must satisfy" in prompt
    for code in ("V4", "V5", "V6", "V7", "V10"):
        assert f"`{code}`" in prompt, f"{code} is not stated to the optimizer"


# ── the refusal summary keeps the reason that explains intent ──────────────
def test_summarise_reasons_groups_and_prioritises():
    reasons = [
        f"V6 [node{i}]: reads 'x.json', which no node on a path produces"
        for i in range(12)
    ] + ["I:fact_produced: the score reads ['best'], which no node on a path produces"]
    summary = summarise_reasons(reasons, limit=4)
    assert summary[0].startswith("I:fact_produced"), (
        "the mode-declared rule states intent and must come first: " + str(summary)
    )
    assert sum(1 for r in summary if r.startswith("V6")) <= 2
    assert any("more V6 refusal" in r for r in summary), (
        "the count must be stated rather than implied: " + str(summary)
    )


def test_summarise_reasons_keeps_the_first_examples():
    summary = summarise_reasons(["V6 [a]: first", "V6 [b]: second"], limit=8)
    assert summary == ["V6 [a]: first", "V6 [b]: second"]


def test_the_prompt_refusals_include_the_invariant(tmp_path):
    """End to end: what the optimizer is shown carries the intent rule."""
    loop = Loop(
        "circle_packing/adaptive",
        LoopConfig(epochs=1, batch_size=2, optimizer="random", seed=1,
                   # A private directory, not `/tmp` itself: looking for prior runs
                   # scans the runs root, and a shared `/tmp` can hold entries this
                   # process may not stat (a systemd-private directory, for one).
                   runs_root=tmp_path / "never-written"),
        project=Path("."),
    )
    loop.init()
    parent = loop.leaderboard(1) or [Candidate.from_state(loop.mode.state_dict())]
    from refactory_dsh.optim import Optimizer, Proposal

    class Removing(Optimizer):
        name = "removing"

        def propose(self, ctx, n, parents):  # type: ignore[no-untyped-def]
            return [Proposal(edit={"op": "remove_node", "id": "exit"},
                             parent=parents[0].id)] * n

    candidates, rejections = Removing(k=1).propose_valid(
        loop._context(), 1, parents=parent, invariants=loop.invariants  # noqa: SLF001
    )
    assert not candidates
    reasons = [r for rejection in rejections for r in rejection.reasons]
    assert any(r.startswith("I:") for r in reasons), reasons


# ── V20: a {{name}} in a knob is filled only where a runtime fills it ─────────


@nn.fills_facts
def filled_op(ctx):  # type: ignore[no-untyped-def]
    """An op that renders the knobs it reads (resolved as ``test_integrity:filled_op``)."""
    return {}


def _knob_reader(node: nn.Node) -> nn.Graph:
    """``a`` produces the fact ``a``; ``node`` reads the knob ``note`` after it."""
    graph = nn.Graph()
    graph.add(nn.Node("a", "op", call=f"{OPS}:write_best", writes=("a.json",), produces=("a",)))
    graph.add(node)
    graph.connect("a", node.id)
    return graph


def _v20(graph: nn.Graph, note: str, **kwargs) -> list[tuple[str, str, str]]:  # type: ignore[no-untyped-def]
    issues = nn.validate(graph, {"note": note}, declared_knobs={"note": nn.Knob(note, "str")},
                         **kwargs)
    return [(i.code, i.node, i.knob) for i in issues if i.code == "V20"]


@pytest.mark.parametrize("call, refused", [(f"{OPS}:write_best", True),
                                           ("test_integrity:filled_op", False)])
def test_a_slot_in_a_knob_an_op_pastes_as_written_is_refused(call, refused):
    """An op reads its knob as it is unless its callable fills facts: a `{{fact}}` there
    would reach the step as braces while the state looked valid, so it is refused."""
    graph = _knob_reader(nn.Node("b", "op", call=call, produces=("done",), knobs=("note",)))
    assert _v20(graph, "Use {{a}}.") == ([("V20", "b", "note")] if refused else [])
    assert _v20(graph, "Use a.") == []
    # Without resolving calls an op's callable is unknown, so it is not judged.
    assert _v20(graph, "Use {{a}}.", resolve_calls=False) == []


def test_a_slot_in_an_agents_prompt_knob_is_filled_and_an_unbound_one_is_not():
    bound = nn.Node("ask", "agent", prompt_template="(the knob)", writes=("ask.txt",), tool_policy="none",
                    field_bindings=(("prompt_template", "note"),), knobs=("note",))
    assert _v20(_knob_reader(bound), "Use {{a}}.") == []
    unbound = nn.Node("ask", "agent", prompt_template="Go.", writes=("ask.txt",),
                      tool_policy="none", knobs=("note",))
    assert _v20(_knob_reader(unbound), "Use {{a}}.") == [("V20", "ask", "note")]


def test_every_shipped_mode_seed_passes_v20():
    """No shipped mode's seed holds a slot in a knob nothing fills."""
    import importlib
    import inspect
    import pkgutil

    import refactory_dsh
    import refactory_packages

    modes = set()
    for package in (refactory_packages, refactory_dsh):
        for found in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            try:
                module = importlib.import_module(found.name)
            except Exception:  # noqa: BLE001 - a module that does not import declares no mode here
                continue
            modes |= {cls for _, cls in inspect.getmembers(module, inspect.isclass)
                      if issubclass(cls, nn.Mode) and cls is not nn.Mode
                      and cls.__module__ == module.__name__}
    assert len(modes) > 10
    for cls in sorted(modes, key=lambda c: f"{c.__module__}.{c.__name__}"):
        try:
            state = cls().state_dict()
        except Exception:  # noqa: BLE001 - a mode that needs arguments is not a seed here
            continue
        issues = nn.validate_state(state, resolve_calls=True)
        assert not [str(i) for i in issues if i.code == "V20"], cls
