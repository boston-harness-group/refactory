"""The analysts are told what an edit can build in this run (`analysts.buildable_lines`),
from the run's graph and settings, and every line says what a real attempt does: a tool
change on a step whose tool rule is fixed is refused, a new sealed step is accepted where
the seal runs (and refused where it does not), a protected step stays.

Each test fails when the behaviour it names is reverted (mutation checks in the PR).
"""

from __future__ import annotations

from typing import Any

import structure_fuzz as sf
from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.optim import ProposalContext, analysts, base

HEADING = f"{analysts.POSSIBLE_HEADING}\n"


def _ctx(applier: Any, *, sealed: bool, seed: bool = True) -> ProposalContext:
    state = applier.seed_state
    knobs = [{"name": name, "frozen": bool((entry.get("spec") or {}).get("frozen"))}
             for name, entry in (state.get("knobs") or {}).items()]
    return ProposalContext(
        mode_name="m", epoch=2, node_states=[dict(n) for n in state["graph"]["nodes"].values()],
        edges=list(state["graph"]["edges"]), catalogue=nn_edits.catalogue("broad"),
        protected=list(applier.protected), graders=list(applier.graders), knobs=knobs,
        new_node_tools=sealed, fixed_tool_rules=analysts.fixed_tool_rules(state) if seed else None,
        analysts=analysts.settings())


def _section(ctx: ProposalContext, kind: str = "failure") -> str:
    head, _tail = analysts._analyst_frame(ctx, kind, 1, 1, 0, "PIPE")
    assert head.count(HEADING) == 1, kind
    start = head.index(HEADING)
    # Inside "The changes the optimizer can make", beside the operations it qualifies.
    operations = head.index("## The changes the optimizer can make\n")
    assert operations < start and "## What the score measures" not in head[operations:start]
    assert "## The failed items" not in head[:start] and "## The passed items" not in head[:start]
    return head[start:head.index("\n## ", start + len(HEADING))]


def _reasons(outcome: Any) -> str:
    assert isinstance(outcome, list), f"accepted: {outcome}"
    return " | ".join(outcome)


def test_a_fixed_tool_rule_and_a_sealed_new_step_are_said_as_an_attempt_decides():
    applier = sf.Applier(sf.seeds()["fuzz_draft_review"])
    applier.optimizer.sealed_tools = (True, "")
    parent = sf.Candidate.from_state(applier.seed_state)
    section = _section(_ctx(applier, sealed=True))
    # Settings first, said as plainly as the structure.
    assert section.split("\n")[1] == "- Settings: `set_knob` changes the value of any setting."
    assert ("- Fixed tool rules: `draft` (no tools, tool_policy `none`); `finalize` (no tools, "
            "tool_policy `none`); `review` (no tools, tool_policy `none`). No edit gives "
            "these steps tools or changes their rule: an attempt that does is refused "
            "(E-tool-policy-escalation).") in section
    assert ("- New steps' tools: a new agent step (`fallback`, `insert`, `map`) may list "
            "tools from `bash`; it runs sealed, in a fresh workspace holding only the files "
            "it reads, with no other item's files, no grader files and no network.") in section
    # The attempt agrees: `review` given `bash` is refused for the reason said ...
    review = applier.seed_state["graph"]["nodes"]["review"]
    regrant = {"edits": [{"op": "remove_node", "id": "review"},
                         {"op": "add_node", "node": {**review, "tools": ["bash"]},
                          "position": {"after": "draft"}},
                         {"op": "add_edge", "source": "review", "target": "finalize"}]}
    assert "E-tool-policy-escalation: node 'review'" in _reasons(
        applier.attempt(parent, regrant, set()))
    # ... and a new step with `bash` from a listed operation is a candidate.
    sealed = {"op": "insert", "after": "draft", "kind": "agent", "text": "Check it.",
              "tools": ["bash"]}
    assert isinstance(applier.attempt(parent, sealed, set()), sf.Candidate)
    # The success analysts read the same facts.
    assert _section(_ctx(applier, sealed=True), "success") == section


def test_where_no_sealed_step_can_run_a_new_step_has_no_tools_as_an_attempt_decides():
    applier = sf.Applier(sf.seeds()["fuzz_draft_review"])
    applier.optimizer.sealed_tools = (False, "no seal here")
    section = _section(_ctx(applier, sealed=False))
    assert ("- New steps' tools: in this run a new agent step (`fallback`, `insert`, `map`) "
            "has no tools; one that lists any is refused (E-new-node-tools).") in section
    assert "may list tools" not in section
    sealed = {"op": "insert", "after": "draft", "kind": "agent", "text": "Check it.",
              "tools": ["bash"]}
    assert "E-new-node-tools" in _reasons(applier.attempt(
        sf.Candidate.from_state(applier.seed_state), sealed, set()))


def test_protected_steps_and_a_step_with_tools_are_said_from_the_graph():
    data = sf.Applier(sf.seeds()["fuzz_data_pass"])
    section = _section(_ctx(data, sealed=True))
    assert ("- Protected steps: `items`, `check`. An edit may not remove them, change their "
            "declarations or out-edges, move a knob they read, or let a run finish without "
            "them (E-protected).") in section
    assert "E-protected" in _reasons(data.attempt(
        sf.Candidate.from_state(data.seed_state), {"op": "remove", "step": "check"}, set()))
    # A step whose tools are its backend's is no fixed rule; nor is a step the seed lacks.
    tools = sf.Applier(sf.seeds()["fuzz_tool_step"])
    said = _section(_ctx(tools, sealed=True))
    assert "`reason` (no tools, tool_policy `none`)" in said and "`gather`" not in said
    ctx = _ctx(tools, sealed=True)
    ctx.fixed_tool_rules = {}
    assert "Fixed tool rules" not in _section(ctx), "a step the seed lacks is not fixed"
    # Without a recorded seed an attempt holds the edited graph to its own rules.
    assert _section(_ctx(tools, sealed=True, seed=False)) == said
    # Knob-only runs offer no new step, and say nothing of its tools.
    knobs = _ctx(tools, sealed=True)
    knobs.catalogue = nn_edits.catalogue("knob_only")
    assert "New steps' tools" not in _section(knobs)


def test_the_loop_hands_the_analysts_its_seeds_fixed_tool_rules(tmp_path, monkeypatch):
    from test_optimizer_memory import _run

    loop, proposer = _run(tmp_path, monkeypatch, [[(5, 7)], []])
    assert loop._original_seed_state is not None
    for ctx in proposer.contexts:
        assert ctx.fixed_tool_rules == analysts.fixed_tool_rules(loop._original_seed_state)
        assert ctx.fixed_tool_rules is not None


def test_a_text_only_run_is_told_only_what_its_edits_can_move():
    """With `set_knob` alone no line about tools, new steps or protection: no edit of the
    run can touch them. One line says which settings move (all but the frozen ones and
    those a protected or grading step reads) and that the structure stays."""
    data = sf.Applier(sf.seeds()["fuzz_data_pass"])
    ctx = _ctx(data, sealed=True)
    # The loop's catalogue under allowed_ops ["set_knob"] (`Loop._context`).
    ctx.catalogue = [c for c in ctx.catalogue if c["op"] == "set_knob"]
    held = sorted(base._grader_side_knobs(data.seed_state, [*data.protected, *data.graders]))
    section = _section(ctx)
    lines = section.split("\n")[1:]
    assert [line.split(":")[0] for line in lines if line] == ["- Settings", "- Structure"]
    assert lines[1] == ("- Structure: no operation of this run changes the steps, their "
                        "edges or their tools.")
    for word in ("tool", "Protected", "new agent step", "sealed"):
        assert word not in lines[0] and word not in section.replace(lines[1], ""), word
    assert held == ["data_parallelism"], "the data pass reads it"
    assert all(f"`{k}`" in lines[0] for k in held) and "(E-frozen)" in lines[0]
    # An attempt agrees: a held knob is refused, as the line says.
    assert "E-frozen" in _reasons(data.attempt(
        sf.Candidate.from_state(data.seed_state),
        {"op": "set_knob", "name": held[0], "value": 2}, set()))


def test_a_setting_a_protected_step_reads_is_said_held_as_an_attempt_decides():
    """A knob no flag freezes but a protected step reads stays as it is: the settings line
    names it, and an attempt that moves it is refused (E-frozen)."""
    applier = sf.Applier(sf.seeds()["fuzz_draft_review"])
    applier.protected = ("review",)
    ctx = _ctx(applier, sealed=True)
    ctx.catalogue = [c for c in ctx.catalogue if c["op"] == "set_knob"]
    line = _section(ctx).split("\n")[1]
    assert line == ("- Settings: `set_knob` changes the value of any setting except these, "
                    "which stay as they are: `review_task`, `review__system_prompt` (E-frozen).")
    assert "E-frozen" in _reasons(applier.attempt(
        sf.Candidate.from_state(applier.seed_state),
        {"op": "set_knob", "name": "review_task", "value": "Review it."}, set()))
