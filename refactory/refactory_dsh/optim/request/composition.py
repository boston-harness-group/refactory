"""Pipeline view, graph facts, composition section, operations."""


from __future__ import annotations

import json
from typing import Any, Callable, Collection, Iterable, Mapping, Sequence

from refactory_dsh.nn import edits as nn_edits
from refactory_dsh.nn.core_ops import STEP_KINDS


def operation_lines(catalogue: Sequence[Mapping[str, Any]],
                    node_states: Sequence[dict[str, Any]] = (),
                    edges: Sequence[dict[str, Any]] = (), *, protected: Sequence[str] = (),
                    knob_values: Mapping[str, Any] | None = None,
                    step_routes: Mapping[str, Sequence[str]] | None = None,
                    tools: bool = True, code_nodes: bool = False) -> list[str]:
    """"The operations you may use", each said once: the written-out operations in a
    table (fields and description), then the core operators, one line and one example each
    (`operator_lines`, `nn.core_ops.OPERATORS`). Without ``tools`` (the run offers no tool
    step) no operator offers 'tools'."""
    core = set(CORE_OPERATORS)
    written = [c for c in catalogue if c.get("op") not in core]
    lines = [_table(["op", "kind", "macro", "fields", "description"],
                    [[c.get("op"), c.get("kind"), c.get("macro"),
                      _edit_fields(c.get("fields") or ()), c.get("description")]
                     for c in written])] if written else []
    return lines + operator_lines(catalogue, node_states, edges, protected=protected,
                                  knob_values=knob_values, step_routes=step_routes,
                                  tools=tools, code_nodes=code_nodes)


#: The catalogue kinds that change the pipeline's structure (``nn.edits.catalogue``).
STRUCTURE_KINDS = ("node", "edge", "topology")


#: Catalogue fields an example leaves out: optional for every op that has them.
_OPTIONAL_EDIT_FIELDS = ("condition", "new_id", "patch", "config")


#: Placeholders for the catalogue fields that name an existing step.
_STEP_FIELDS = ("target", "id", "source", "first", "second")


def structural_ops(catalogue: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The permitted operations that change the pipeline's structure."""
    return [op for op in catalogue if op.get("kind") in STRUCTURE_KINDS]


def example_node() -> dict[str, Any]:
    """A new agent node for an example, from the generated node schema: its required
    fields with placeholders, its own system prompt, and the one write a node without
    tools declares (its reply; rule V15)."""
    from refactory_dsh.nn.node_schema import node_schema

    fill = {"id": "<new step id>", "kind": "agent", "prompt_template": "<prompt text>"}
    node = {name: fill.get(name, f"<{name}>") for name in node_schema()["agent"]["required"]}
    node["system_prompt_template"] = "<system prompt text>"
    node["writes"] = ["<new step id>.txt"]
    return node


def _node_op(catalogue: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The structural op an example uses: one that puts a new node after an existing
    step (``target`` and ``node``), else any that takes a node, else any structural op."""
    ops = structural_ops(catalogue)
    for wanted in (lambda f: set(f) == {"target", "node"}, lambda f: "node" in f, lambda f: True):
        for op in ops:
            if wanted(op.get("fields") or []):
                return op
    return None


def structural_example(catalogue: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """An example edit for a structural op this run permits: the one-line ``insert`` when
    the run permits it, else one built from another op's catalogue fields with
    placeholders; ``None`` when none is permitted."""
    if any(op.get("op") == "insert" for op in catalogue):
        return {"op": "insert", "after": "<existing step id>", "kind": "agent",
                "text": "<what the new step does>"}
    op = _node_op(catalogue)
    if op is None:
        return None
    edit: dict[str, Any] = {"op": op["op"]}
    for name in op.get("fields") or []:
        if name in _OPTIONAL_EDIT_FIELDS:
            continue
        if name == "node":
            edit[name] = example_node()
        elif name == "position":
            edit[name] = {"after": "<existing step id>"}
        elif name in _STEP_FIELDS:
            edit[name] = "<existing step id>"
        else:
            edit[name] = f"<{name}>"
    return edit


def _item_count(records: Sequence[dict[str, Any]]) -> int:
    return len({str(record.get("id")) for record in records or ()})


def _blocks_section(insertable: Sequence[dict[str, Any]],
                    catalogue: Sequence[dict[str, Any]]) -> list[str]:
    """Where a new node comes from: the mode's blocks, verbatim, when it declares any;
    otherwise the node schema. Nothing when the run permits no structural op."""
    if not structural_ops(catalogue):
        return []
    if not insertable:
        return ["## New nodes",
                "This mode declares no ready-made blocks. A new node is written from the node "
                "schema in the reference (or copied from a node's JSON under \"Each node in "
                "full\", with a new id).", ""]
    return ["## Blocks you may insert",
            "Use these node objects verbatim in an edit that takes a node, changing nothing "
            "but their `id`, or write a node from the node schema in the reference:",
            "```json", json.dumps(list(insertable), indent=2, default=str), "```", ""]


def _composition_anchor(node_states: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
                        protected: Sequence[str]) -> tuple[dict[str, Any], str | None] | None:
    """Where the composition examples attach: an agent of this graph with one plain
    out-edge to a step that is not a gate, and that out-edge's target; else an agent with
    no verdict out-edge (one that ends its pass or the run), and ``None`` for its next
    step, so a graph whose only agent step ends its pass still gets examples (each built
    in a form an attempt takes there, `_completed`). ``None`` when the graph has neither."""
    by_id = {str(state.get("id")): state for state in node_states}
    ends: list[dict[str, Any]] = []
    for node_id in sorted(by_id):
        state = by_id[node_id]
        if state.get("kind") != "agent" or node_id in protected or not state.get("writes"):
            continue
        out = [edge for edge in edges if edge.get("source") == node_id]
        if any(edge.get("condition") is not None for edge in out):
            continue
        if len(out) != 1:
            ends += [state] if not out else []
            continue
        target = by_id.get(str(out[0].get("target"))) or {}
        if target.get("kind") in ("agent", "op"):
            return state, str(out[0].get("target"))
    return (ends[0], None) if ends else None


def _fact_filled_knobs(node_states: Sequence[dict[str, Any]],
                       text_knobs: Iterable[str] | None = None) -> list[str]:
    """The knobs an op that fills facts reads (`nn.fills_facts`): beside an agent's
    templates, the only places a `{{fact}}` is filled in. Only text knobs, given
    ``text_knobs`` (a flag the same op reads holds no slot)."""
    from refactory_dsh.nn.graph import Node
    from refactory_dsh.nn.validate import filled_knobs

    names: set[str] = set()
    for state in node_states:
        try:
            names.update(filled_knobs(Node.from_state(dict(state))))
        except Exception:  # a node that does not read back is the validator's to report
            continue
    if text_knobs is not None:
        names &= set(text_knobs)
    return sorted(names)


def _or_list(names: Sequence[str], last: str = "or") -> str:
    """``names`` in backticks, the last joined by ``last``."""
    quoted = [f"`{name}`" for name in names]
    return f" {last} ".join([", ".join(quoted[:-1]), quoted[-1]]) if len(quoted) > 1 \
        else "".join(quoted)


#: The lead of "Structural operators". It makes no claim about wiring: what a new step
#: sees and where its output goes are each operator's own generated line
#: (`nn.core_ops.OPERATORS`, `core_ops.REACHES`), the single source.
STRUCTURAL_LEAD = (
    "The structural operators, each one line: what kind of step "
    f"({_or_list(STEP_KINDS)}), where (a step id), and "
    "its text (a prompt, or code). The system does the wiring, as each operator's line "
    "below says. Ids, files and facts are generated; each text goes into a knob of its own "
    "(`<id>__task`, `<id>__code`) for a later `set_knob`; the candidate says what was "
    "wired. Every text has a default (the neighbour's, or a neutral line), and each "
    "operator counts as one change. The written-out operations in the table above stay "
    "for what these do not express.")


def composition_section(catalogue: Sequence[dict[str, Any]], node_states: Sequence[dict[str, Any]],
                        edges: Sequence[dict[str, Any]], *, code_nodes: bool = False,
                        text_knobs: Iterable[str] | None = None,
                        structure_guidance: bool = False, tools: bool = False,
                        protected: Sequence[str] = (),
                        knob_values: Mapping[str, Any] | None = None,
                        frozen: Collection[str] = ()) -> list[str]:
    """"How nodes compose": the language's primitives (each node kind, where a `{{fact}}`
    is filled in, tools, gates and loops, running out of output, crashes, forks), facts
    only, then one example of each composition shape with this graph's ids
    (`composition_examples`); each operator's own line and example are under the
    operations (`operator_lines`). With ``structure_guidance``, what makes a new step pay
    off (`feedback.STRUCTURE_GUIDANCE`). Nothing when the run permits no structural op."""
    from refactory_dsh import snippet

    if not structural_ops(catalogue):
        return []
    ops = {op.get("op") for op in catalogue}
    reader = ("a code op that reads the file and `produces` the fact" if code_nodes else
              "an op that reads the file and `produces` the fact")
    filled = _fact_filled_knobs(node_states, text_knobs)
    lines = [
        "### How nodes compose",
        "The pipeline is made of four kinds of step, and any of them may be put anywhere the "
        "validator's rules (listed below) allow:",
        "- **agent**: one model call. Its prompt fills each `{{name}}` from a fact (or knob) "
        "produced upstream; its reply is written to its first `writes` file and, when it "
        "declares `produces`, also becomes that fact. A new agent without `tools` "
        "sees only its prompt and the files it names in `sees` (they become its declared reads, pasted in after the prompt): "
        "hand it anything else as `{{facts}}`. "
        f"A file's text becomes a fact through {reader}, or when the agent that writes the "
        "file declares `produces`.",
        "- **where `{{fact}}` is filled in**: only in an agent's `prompt_template` (its "
        "`<id>__task` knob) and its system prompt (its `<id>__system_prompt` knob)"
        + (f", and in {', '.join(f'`{name}`' for name in filled)}, which an op that fills "
           "facts reads" if filled else "")
        + ". Any other knob is read as written, so a `{{fact}}` there would never arrive "
        "and is refused (V20).",
        tools_line(ops) if tools else NO_TOOLS_LINE,
    ]
    if code_nodes:
        sealed_imports, has_packages = _sealed_code_imports(node_states)
        lines += [
            f"- **code op**: `\"call\": \"{snippet.RUN}\"` with `code`, Python that defines "
            "`main(inp)`. `inp` is `{\"files\": {each declared read whose content is text: "
            "its text}, \"facts\": {each name in its `inputs`: its value}}`; binary reads "
            "(e.g. .xlsx) are silently absent from `inp[\"files\"]`, and without "
            "`\"sealed\": true` a code step cannot read them from disk (the sandbox has no "
            "file access). `main` returns "
            "`{\"facts\": {...}, \"files\": {...}}`, each one declared in its `produces` / "
            "`writes`. An insert takes `produces` (default: one fact named after the step's "
            "id) and `writes` (default: none; a takes_over adds the taken-over files). "
            "`main` must return exactly the declared names. "
            "It costs no model call. Limits: imports only these modules: "
            f"{snippet.ALLOWED_IMPORTS_TEXT}; at most {snippet.MAX_CODE_BYTES} "
            f"bytes, {snippet.WALL_S} s, no files, network, open/exec/eval or dunders (rule "
            "V18). The code goes into the node's own `<id>__code` knob, which a later "
            "`set_knob` patches or rewrites.",
        ]
        if sealed_imports:
            sealed_io = (
                " The files it names in `sees` become its declared reads and are present "
                "as files in the working directory, so it can `open()` them directly "
                "(including binary files such as .xlsx that `inp[\"files\"]` does not carry). "
                "It may also write its declared `writes` "
                "to the working directory; those files are copied back to the run directory "
                "after a successful run, so downstream steps read them.")
            if has_packages:
                importable = "; ".join(sealed_imports)
                lines.append(
                    "  With `\"sealed\": true` on the insert (or loop/route), the code step "
                    "runs in the Task's tool environment instead: it may import anything "
                    f"installed in that environment, including {importable}; there is no "
                    f"import allowlist (the seal is the boundary). Up to "
                    f"{snippet.SEALED_WALL_S} s and {snippet.SEALED_MEMORY_MB} MB. It must "
                    "still define `main(inp)` and parse."
                    f"{sealed_io} "
                    "Without `\"sealed\": true` a code step imports only the standard "
                    "library.")
            else:
                lines.append(
                    "  With `\"sealed\": true` on the insert (or loop/route), the code step "
                    "runs in the Task's tool environment instead: it may import the packages "
                    f"that environment provides ({'; '.join(sealed_imports)}), with no import "
                    f"allowlist (the seal is the boundary), up to {snippet.SEALED_WALL_S} s "
                    f"and {snippet.SEALED_MEMORY_MB} MB. It must still define `main(inp)` "
                    "and parse."
                    f"{sealed_io} "
                    "Without `\"sealed\": true` a code step imports only the standard "
                    "library.")
    lines += [
        "- **gate**: routes by a verdict, to the out-edge whose `condition` is that verdict. "
        "Give each out-edge a single-word condition (`proceed`, `reloop`, `revise` ...). "
        f"Any declared upstream output can be the verdict: with `\"evaluator\": "
        f"\"{snippet.VERDICT}\"` the gate reads the verdict from its first `required_reads` "
        "file (or, with none, its first `inputs` fact) -- whatever an agent or a code op "
        "upstream wrote there: its first word, or else its last line when that is the word "
        "alone or the word then `:`, `—`, `–`, `-` or `.` and more (leading markdown or "
        "quotes are skipped; a list item or a code fence is not read) -- and the rest of "
        "the text is the reason. The word must be one of the conditions (any case) or "
        "`halt`, which ends the whole run. Anything else fails that item's run, and so does "
        "a first word and a last line that name different verdicts, so tell the writer to "
        "start with the word or put it alone on the last line.",
    ]
    if code_nodes:
        lines.append(
            f"  With `\"evaluator\": \"{snippet.GATE}\"` and `code`, `main(inp)` returns "
            "`{\"verdict\": <condition or halt>, \"reason\": <text>}`.")
    lines += [
        "- **loop**: a gate's out-edge back to an earlier step. The `loop` operation makes "
        "one around a step (step -> gate; gate -> step on `reloop`, gate -> the step's old "
        "successor on `proceed`); a reloop hands the gate's reason to the step it goes back "
        "to as feedback, and the gate's `max_iterations` bounds how often it may. When a gate "
        "would reloop once more than that, its `on_exhausted` decides: `continue` (the "
        "default inside the per-item pass) goes on along `proceed` with the last attempt, "
        "which is graded as it stands; `fail_item` fails that item only; `halt` ends the "
        "whole run (no later item runs). A gate with no forward `proceed` edge defaults to "
        "`fail_item`. The last step of the pass can be looped too: the "
        "loop then adds a step after the gate to end the pass.",
        "- **running out of output**: a model call that uses its whole output allowance "
        "without answering is the model failing that question. It is not retried unless the "
        "step opts in (retry.on): the step "
        "has no answer, and the item goes on without it (a gate after it sees no answer and "
        "may reloop; a fork's join sees that branch without one; a code op's `inp['files']` "
        "holds only the files that exist; an item left with nothing to grade scores 0).",
        "- **running out of turns**: an agent that uses its whole turn cap without finishing "
        "is the model failing that question, like running out of output. It is not retried "
        "unless the step opts in (retry.on): "
        "the step has no answer, and the item goes on without it, as above.",
        "- **crashes**: when the worker or product fails (not the model's own output or turn "
        "cap), an agent step in the per-item pass that is safe to rerun (no tools, its own "
        "report files only, or a workspace) is run again in a fresh session, up to 2 more "
        "times; if every session crashes it has no answer, as above. An item that times out "
        f"is a {HARNESS_TIMEOUT}.",
    ]
    if {"parallel", "parallelize"} & ops:
        lines.append(
            "- **fork**: steps run at once between a fork and a join; branches write distinct "
            "files and facts. `parallelize` and `branch` wire what the branches make into the "
            "step after the join; the written-out `parallel` operation leaves that to you.")
    if structure_guidance:
        from ..feedback import STRUCTURE_GUIDANCE
        lines.append(STRUCTURE_GUIDANCE)
    examples, refused = composition_examples(catalogue, node_states, edges,
                                             protected=protected, code_nodes=code_nodes,
                                             tools=tools, knob_values=knob_values,
                                             frozen=frozen)
    if examples:
        lines += ["", COMPOSITION_LEAD]
        for shape, bundle in examples:
            lines.append(f"- {shape}: "
                         + json.dumps({"edits": bundle} if len(bundle) > 1 else bundle[0]))
    if refused:
        lines.append(REFUSED_LEAD + "; ".join(f"{shape}: {why}" for shape, why in refused)
                     + ".")
    return lines + [""]


#: What heads the composition examples: vocabulary, one of each shape, none preferred.
COMPOSITION_LEAD = ("One example of each composition shape, with this graph's ids (how to "
                    "write it; none is preferred):")


#: What heads the shapes this graph refuses, each with its refusal.
REFUSED_LEAD = "Shapes not shown here, with the refusal of the forms tried: "


#: The composition shapes, in the order shown (`composition_examples`).
COMPOSITION_SHAPES = ("a new agent step", "a parallel fork and join",
                      "a verdict loop (a judge that can send a step back)",
                      "a route (a judge choosing the next step)", "a code op",
                      "an agent step with tools")


def composition_examples(catalogue: Sequence[dict[str, Any]],
                         node_states: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
                         *, protected: Sequence[str] = (), code_nodes: bool = False,
                         tools: bool = False, knob_values: Mapping[str, Any] | None = None,
                         frozen: Collection[str] = ()
                         ) -> tuple[list[tuple[str, list[dict[str, Any]]]], list[tuple[str, str]]]:
    """One short example of each composition shape (`COMPOSITION_SHAPES`) this run offers,
    with this graph's ids and placeholder texts: how each is written, never when to use it.
    A shape is shown only when its operator is offered (a code op with ``code_nodes``, a
    tool step with ``tools`` and an operator that makes one, `tool_step_ops`) and its bundle
    applies to this graph as an attempt takes it and as it runs (`_completed`, then
    `_example_refusal`: no ``frozen`` knob moved, every step it adds read, no step's outputs
    taken over; ``tests/test_dsl_coverage.py`` holds each one to `Optimizer.attempt`). Returns
    the shapes shown and, for each offered shape no form of which this graph takes, the
    refusal of every form tried. A take-over form is shown only when the new step can
    write every file it takes over (`_can_write`), its text saying it must write them.
    Nothing when the graph has no step to anchor them."""
    ops = {op.get("op") for op in catalogue}
    anchor = _composition_anchor(node_states, edges, protected)
    state = _example_state(node_states, edges, knob_values)
    if anchor is None or state is None:
        return [], []
    step = str(anchor[0]["id"])
    code_id = f"{step}_code"
    code = ("def main(inp):\n    result = len(inp['facts'])  # <what it computes>\n"
            f"    return {{'facts': {{{code_id!r}: str(result)}}, 'files': {{}}}}\n")
    agent = {"kind": "agent", "text": "<what the new step does>"}
    # The step just before the anchor, when one plain edge leads to it: branches there feed it.
    into = [e for e in edges if e.get("target") == step]
    feeding = [str(into[0]["source"])] if len(into) == 1 and into[0].get("condition") is None \
        else []
    branches = [{"kind": "agent", "text": "<what one branch does>"},
                {"kind": "agent", "text": "<what another branch does>"}]
    # Each shape: its operator and its forms, the first this graph takes is shown.
    shapes: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("insert", COMPOSITION_SHAPES[0], [
            {"op": "insert", "before": step, **agent}, {"op": "insert", "after": step, **agent}]),
        ("parallelize", COMPOSITION_SHAPES[1], [
            {"op": "parallelize", "after": step, "new": branches},
            *({"op": "parallelize", "after": before, "new": branches} for before in feeding),
            {"op": "parallelize", "steps": [step], "copies": 1}]),
        ("loop", COMPOSITION_SHAPES[2], [
            {"op": "loop", "step": step, "judge": "agent", "text": "<what to check>"}]),
        ("route", COMPOSITION_SHAPES[3], [
            {"op": "route", "after": step, "judge": "agent", "text": "<how to choose>"},
            *({"op": "route", "after": before, "judge": "agent", "text": "<how to choose>",
               "to": {"proceed": step, "revise": {**agent, "to": step}}} for before in feeding)]),
    ]
    if code_nodes:
        sealed_imports, _ = _sealed_code_imports(node_states)
        code_extra: dict[str, Any] = {"sealed": True} if sealed_imports else {}
        shapes.append(("insert", COMPOSITION_SHAPES[4], [
            {"op": "insert", "before": step, "kind": "code", "id": code_id, "text": code,
             "produces": [code_id], **code_extra},
            {"op": "insert", "after": step, "kind": "code", "id": code_id, "text": code,
             "produces": [code_id], **code_extra}]))
    if tools and "insert" in tool_step_ops(ops):
        tool = {"kind": "agent", "tools": ["bash"], "text": "<what to find out with tools>"}
        shapes.append(("insert", COMPOSITION_SHAPES[5], [
            {"op": "insert", "before": step, **tool}, {"op": "insert", "after": step, **tool}]))
    out: list[tuple[str, list[dict[str, Any]]]] = []
    refused: list[tuple[str, str]] = []
    for op, shape, forms in shapes:
        if op not in ops:
            continue
        reasons: list[str] = []
        for form in forms:
            done = _completed(state, [form])
            if done is not None:
                done = _with_outputs_said(done, state)
            why = _example_refusal(state, done if done is not None else [form], frozen)
            if done is not None and why is None:
                out.append((shape, done))
                break
            why = why or "refused by the operator's precondition"
            if why not in reasons:
                reasons.append(why)
        else:
            refused.append((shape, " / ".join(reasons)))
    return out, refused


def tool_step_ops(ops: Iterable[str]) -> list[str]:
    """The operations among ``ops`` that can make a new agent step with `tools`: the
    coverage map's (`nn.coverage.made_by`, which ``tests/test_dsl_coverage.py`` holds to each
    one making a sealed tool step), core operators first in their order, then the
    written-out ones by name."""
    from refactory_dsh.nn import coverage

    makers = set(coverage.made_by("node field", "tools")) & set(ops)
    return [*(op for op in CORE_OPERATORS if op in makers),
            *sorted(makers - set(CORE_OPERATORS))]


def tools_line(ops: Iterable[str]) -> str:
    """What the request says of tools when a new step may have them: an existing step's own
    tool rules (its ``tool_policy``, its ``tools``) are not a limit on a step an edit adds;
    the operations that can add one are generated (`tool_step_ops`), so the list is always
    the exact set this run offers. `NO_TOOLS_LINE` when none of ``ops`` can."""
    makers = tool_step_ops(ops)
    if not makers:
        return NO_TOOLS_LINE
    return ("- **tools**: every existing step keeps its own tool rules (a step that has no "
            "tools or refuses them stays so). A new agent step may still have `tools` "
            f"(made by {_or_list(makers)}: in the edit's own 'tools', or in a step or node "
            "it carries): it runs sealed, in a fresh workspace holding "
            "only the files it reads -- in the per-item part, unless its 'sees' "
            "names its inputs, the item's input files (the pipeline view lists "
            "them) beside what it sees -- with no network and no grader files.")


#: The same when this run offers no tool step (no seal here, or ``new_step_tools`` off).
NO_TOOLS_LINE = ("- **tools**: every existing step keeps its own tool rules, and in this run "
                 "a new step has no tools.")


#: The core operators a request explains, in its order (`nn.core_ops.CORE_OPS`).
CORE_OPERATORS = nn_edits.CORE_OPS


def _example_ids(node_states: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
                 protected: Sequence[str]) -> dict[str, str]:
    """Ids of this graph for the operators' examples: a step (an agent with one plain
    out-edge to a step) and its next step, and a fork, where the graph has them."""
    ids: dict[str, str] = {}
    anchor = _composition_anchor(node_states, edges, protected)
    if anchor is not None and anchor[1] is not None:
        ids["step"], ids["next step"] = str(anchor[0]["id"]), anchor[1]
    fork = next((str(n.get("id")) for n in sorted(node_states, key=lambda n: str(n.get("id")))
                 if n.get("kind") == "fork"), None)
    if fork:
        ids["fork"] = fork
    return ids


def _example_state(node_states: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
                   knob_values: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """The request's graph as a state the operators' preconditions can read (its start is
    the step nothing leads to), or ``None`` when it does not read back."""
    from refactory_dsh.nn.graph import Graph

    entered = {str(e.get("target")) for e in edges} | {
        str(t) for n in node_states for t in (n.get("targets") or ())} | {
        str(n.get("subgraph_entry")) for n in node_states if n.get("subgraph_entry")}
    start = next((str(n["id"]) for n in sorted(node_states, key=lambda n: str(n.get("id")))
                  if str(n["id"]) not in entered), "")
    try:
        graph = Graph.from_state({"start": start,
                                  "nodes": {str(n["id"]): dict(n) for n in node_states},
                                  "edges": list(edges)})
    except Exception as exc:  # noqa: BLE001 - logged; the examples then use placeholders
        import logging

        logging.getLogger(__name__).warning("the request's graph does not read back, so the "
                                            "operator examples use placeholders: %s", exc)
        return None
    return {"graph": graph.to_state(),
            "knobs": {k: {"value": v, "spec": {"kind": "str"}}
                      for k, v in (knob_values or {}).items()}}


def _takes_over_parts(takes: Any) -> tuple[str, frozenset[str] | None]:
    """Best-effort parse: on refusal (malformed value) returns ``("", None)``."""
    from refactory_dsh.nn.core_ops import _parse_takes_over

    try:
        return _parse_takes_over(takes)
    except Exception:  # noqa: BLE001 — enrichment, never a refusal
        return "", None


def _can_write(spec: Mapping[str, Any], file: str, files: Sequence[str]) -> bool:
    """Whether a new step of ``spec`` can produce ``file`` among the ``files`` it takes over:
    an agent with tools writes any file, one without only its reply (so one file); the code
    example returns facts, no files, so it takes over none."""
    if spec.get("kind", "agent") == "agent":
        return bool(spec.get("tools")) or len(files) == 1
    return False


def _with_outputs_said(bundle: list[dict[str, Any]], state: Mapping[str, Any]
                       ) -> list[dict[str, Any]]:
    """``bundle`` with each take-over's placeholder text saying which files the new step
    must write: the outputs of the step it takes over, which its readers go on reading."""
    nodes = (state.get("graph") or {}).get("nodes") or {}
    out = []
    for edit in bundle:
        taken = edit.get("takes_over")
        if taken:
            target_id, selected = _takes_over_parts(taken)
            all_files = list((nodes.get(target_id) or {}).get("writes") or ())
            files = ([f for f in all_files if f in selected] if selected is not None
                     else all_files)
        else:
            target_id = None
            files = []
        if files and isinstance(edit.get("text"), str) and edit.get("kind", "agent") == "agent":
            edit = {**edit, "text": f"{edit['text']} It writes {_or_list(files, 'and')}, the outputs "
                                    f"of `{target_id}` it takes over."}
        out.append(edit)
    return out


def _example_refusal(state: Mapping[str, Any], bundle: Sequence[dict[str, Any]],
                     frozen: Collection[str] = ()) -> str | None:
    """Why ``bundle`` is no example for ``state``, or ``None``: an edit refused (a knob of
    ``frozen`` moved included), a step it adds left with no reader
    (`core_ops.unread_by_bundle`), or a step's outputs taken over (the new step would have to
    write what that step writes, which a placeholder text does not say)."""
    from refactory_dsh.nn import core_ops

    nodes = (state.get("graph") or {}).get("nodes") or {}
    for edit in bundle:
        if not edit.get("takes_over"):
            continue
        target_id, selected = _takes_over_parts(edit["takes_over"])
        all_files = list((nodes.get(target_id) or {}).get("writes") or ())
        files = ([f for f in all_files if f in selected] if selected is not None else all_files)
        unable = [f for f in files if not _can_write(edit, f, files)]
        if unable:
            return (f"its new step reaches a reader only by taking over the outputs of "
                    f"`{target_id}` ({_or_list(files, 'and')}), which this new "
                    f"{edit.get('kind', 'agent')} step does not write")
    # Each knob as a spec `set_knob` reads back: its value its default, its kind its
    # value's, frozen as declared.
    kinds = {bool: "bool", int: "int", float: "float", str: "str"}
    knobs = {name: {"value": entry.get("value"),
                    "spec": {"default": entry.get("value"),
                             "kind": kinds.get(type(entry.get("value")), "str"),
                             "frozen": name in frozen}}
             for name, entry in (state.get("knobs") or {}).items()}
    state = {**state, "knobs": knobs}
    states, current = [], dict(state)
    try:
        for edit in bundle:
            current = nn_edits.apply(edit, current)
            states.append(current)
    except Exception as exc:  # noqa: BLE001 - a refused example is said, not shown
        return str(exc)
    return core_ops.unread_by_bundle(state, states, bundle)


def _removable_step(node_states: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
                    protected: Sequence[str], state: Mapping[str, Any] | None) -> str | None:
    """A step ``remove`` can take out of this graph as an example: an agent (else an op) with
    a next step, not one a gate's verdict leads to, not after a protected or data node, and
    making nothing a protected node reads; ``None`` when there is none."""
    if state is None:
        return None
    kinds = {str(n.get("id")): n.get("kind") for n in node_states}
    guarded = {name for other in node_states if str(other.get("id")) in protected
               for key in ("reads", "required_reads", "inputs") for name in other.get(key) or ()}
    sources = {str(e.get("source")) for e in edges}
    ranked = sorted(node_states, key=lambda n: (n.get("kind") != "agent", str(n.get("id"))))
    for node in ranked:
        node_id = str(node.get("id"))
        into = [e for e in edges if e.get("target") == node_id]
        made = {*(node.get("writes") or ()), *(node.get("produces") or ())}
        if node.get("kind") not in ("agent", "op") or node_id in protected or made & guarded \
                or node_id not in sources or any(
                    e.get("condition") or e.get("source") in protected
                    or kinds.get(str(e.get("source"))) == "data" for e in into):
            continue
        if nn_edits.precheck({"op": "remove", "step": node_id}, dict(state)) is None:
            return node_id
    return None


def _fitting_ids(name: str, ids: Mapping[str, str], node_states: Sequence[dict[str, Any]],
                 edges: Sequence[dict[str, Any]], protected: Sequence[str],
                 state: Mapping[str, Any] | None) -> dict[str, str]:
    """The example ids for operator ``name``: ``ids`` when its example passes its
    precondition on this graph, else the first other step (agents first, each with its one
    plain next step) for which it does, else placeholders for the step ids."""
    from refactory_dsh.nn import core_ops

    placeholders = {**ids, "step": "<step>", "next step": "<next step>",
                    **({"fork": "<fork>"} if "fork" in ids else {})}
    if state is None:
        return placeholders
    out_of = {}
    for edge in edges:
        out_of.setdefault(str(edge.get("source")), []).append(edge)
    options = [dict(ids)] if "step" in ids else []
    # A data pass's first and last steps are named by its data node, which an edit there
    # would change: not an example to copy.
    ends = {str(n.get(key)) for n in node_states if n.get("kind") == "data"
            for key in ("subgraph_entry", "subgraph_exit")}
    for node in sorted(node_states, key=lambda n: (n.get("kind") != "agent", str(n.get("id")))):
        node_id = str(node.get("id"))
        out = out_of.get(node_id) or []
        if node.get("kind") in ("agent", "op") and node_id not in protected and len(out) == 1 \
                and not out[0].get("condition") and node_id not in ends:
            options.append({**ids, "step": node_id, "next step": str(out[0].get("target"))})
    for option in options:
        example = core_ops.example(name, option)
        named = [example.get(k) for k in ("step", "steps", "after", "before", "fork")]
        if "<" in json.dumps(named) or "<" in json.dumps(list(example.get("to") or {}.values())):
            continue
        if nn_edits.precheck(example, dict(state)) is None \
                and _completed(state, [example]) == [example]:
            return option
    return placeholders


#: How a core operator's refusal names the edit that replaces the step (`core_ops`).
_REPLACE_HINT = "replace it with "


def checked_hint(reason: str, check: Callable[[dict[str, Any]], Any]) -> str:
    """``reason`` with the replacing edit it names checked by ``check`` (an attempt's
    verdict: a candidate, or its refusals): kept when it would be accepted, otherwise
    turned into "replacing it is refused too (<why>)" -- for a rewire giving the step its
    own text (`core_ops.OWN_TEXT_HINT`), "giving it its own text is refused too (<why>)".
    The operators' own precheck knows only the graph; an attempt also refuses on protected
    steps, frozen and guarded knobs, wiring and validation, so the hint is held to that."""
    from refactory_dsh.nn.core_ops import CODE_HINT_NOTE, OWN_TEXT_HINT

    found = [(at, phrase) for phrase in (_REPLACE_HINT, OWN_TEXT_HINT)
             if (at := reason.find(phrase)) >= 0]
    if not found:
        return reason
    start, phrase = min(found)
    try:
        edit, end = json.JSONDecoder().raw_decode(reason, start + len(phrase))
    except ValueError:
        return reason
    verdict = check(edit)
    if not isinstance(verdict, list):
        code = any(isinstance(step, dict) and step.get("kind") == "code"
                   for step in edit.get("into") or ())
        # Accepted without running the new code on recorded inputs (`attempt` skips that
        # for a hint: a process per request), which an attempt of the edit itself does.
        return (f"{reason[:end]} (not dry-run){reason[end:]}" if code else reason)
    why = "; ".join(str(r).removeprefix("E-precondition: ") for r in verdict)
    rest = reason[end:].removeprefix(CODE_HINT_NOTE)
    refused = ("giving it its own text is refused too" if phrase == OWN_TEXT_HINT
               else "replacing it is refused too")
    return f"{reason[:start]}{refused} ({why}){rest}"


def pipeline_view_lines(node_states: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
                        knobs: Sequence[Mapping[str, Any]], protected: Sequence[str],
                        graders: Sequence[str] = ()) -> list[str]:
    """The pipeline in one line per step (`nn.pipeline_view`), in the operators' own words;
    nothing when the graph does not read back."""
    from refactory_dsh.nn.pipeline_view import render

    state = _fit_state(node_states, edges, knobs)
    text = (render(state, protected, graders) if state is not None
            else "pipeline view unavailable: the graph does not read back")
    return [PIPELINE_VIEW_LEAD, "```text", text, "```"]


#: The reference at the end of the request, and the line that points to it.
REFERENCE_HEADING = "## Reference: the operations, the nodes and the rules"


REFERENCE_POINTER = ("The operations you may use, how nodes compose, every node in full (its "
                     "declarations, what it has before it runs), the node schemas and the "
                     "validator's rules are in the reference at the end.")


#: What the pipeline view's words mean.
def _view_lead() -> str:
    from refactory_dsh.nn.pipeline_view import EXHAUSTED, FORMS

    return (
        "The pipeline, one line per step, generated from the graph in the operators' own "
        "words: `sees` lists a step's "
        "inputs as `\"sees\"` names them, `name@maker` (`@run`: no step makes it); a plain "
        "op that only assembles a model step's input is folded in as `name (via op: its "
        "inputs)`; an agent with tools sees the run dir, its own workspace, or a sealed one "
        "(its inputs only); `told` is "
        "each knob whose text reaches the step, its opening words (`*` frozen; `?` read by "
        "an op with several outputs, for all of them), or a code step's code; `->` what it "
        "writes; a per-item block's `item files` are files the task lays into each item's "
        "pass, under the same name for every item, and the steps that read them: a new step "
        "reads one by naming it (or the data step) in 'sees'; "
        + "; ".join(f"`{form}`" for form in FORMS.values())
        + "; a loop that does not go on with its last attempt past its cap says so: `"
        + EXHAUSTED.format(policy="fail_item").lstrip(", ") + "` (only that item fails) or `"
        + EXHAUSTED.format(policy="halt").lstrip(", ") + "` (the whole run stops)"
        + "; `[optional]`; `[protected]`.")


PIPELINE_VIEW_LEAD = _view_lead()


def _fit_state(node_states: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
               knobs: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The request's graph as a state, its knobs with their kind and frozen mark (the
    operators' refusals read them)."""
    state = _example_state(node_states, edges, {k.get("name"): k.get("value") for k in knobs})
    if state is None:
        return None
    for knob in knobs:
        entry = state["knobs"].get(knob.get("name"))
        if entry is not None:
            from refactory_dsh.nn.knob import _SPEC_FIELDS, KINDS

            spec = {k: knob[k] for k in _SPEC_FIELDS if knob.get(k) is not None}
            spec.setdefault("default", knob.get("value"))
            if spec.get("kind") not in KINDS or (spec.get("kind") == "enum"
                                                 and not spec.get("enum")):
                spec["kind"] = "str"
            entry["spec"] = spec
    return state


def _fork_inputs(fork_id: str, state: dict[str, Any]) -> str:
    """What a new branch of ``fork_id`` can name in ``sees``: each fact or file there before
    the fork as the pipeline view writes it (``name@maker``), one built by a plain op that
    reads a text knob marked with it (it may carry that knob's instructions), and what
    ``"task"`` gives there. Read from the graph; "unavailable: <why>" (and a warning) when
    it does not read back."""
    from refactory_dsh import snippet
    from refactory_dsh.nn import core_ops
    from refactory_dsh.nn.graph import Graph
    from refactory_dsh.nn.operators import carried_knobs, made_by
    from refactory_dsh.nn.pipeline_view import RUN_MAKER

    try:
        graph = Graph.from_state(state["graph"])
        there = core_ops._readable(graph, state, set(core_ops._available(graph).get(fork_id, set())))
        task = core_ops._task_inputs(graph, state) & there
    except Exception as exc:  # noqa: BLE001 - said in the row and logged, never raised
        import logging

        logging.getLogger(__name__).warning("no inputs for fork %s: %s", fork_id, exc)
        return f"unavailable: {type(exc).__name__}: {exc}"
    makers = made_by(graph)
    values = {k: (e or {}).get("value") for k, e in (state.get("knobs") or {}).items()}

    def named(kind: str, name: str) -> str:
        made = makers.get((kind, name)) or []
        texts = [k for m in made if graph.nodes[m].kind == "op"
                 and not snippet.is_code_node(graph.nodes[m])
                 for k in carried_knobs(graph.nodes[m]).get((kind, name), ())
                 if isinstance(values.get(k), str)]
        return (f"{name}@{','.join(made) or RUN_MAKER}"
                + (f" (+knob {', '.join(dict.fromkeys(texts))})" if texts else ""))

    order = sorted(there, key=lambda key: (key[0] != "fact", key[1]))
    return (f"a new branch here can see: {', '.join(named(*key) for key in order) or 'nothing'}"
            f"; \"task\" gives {', '.join(n for _k, n in sorted(task)) or 'nothing'}")


def graph_facts(node_states: Sequence[dict[str, Any]], edges: Sequence[dict[str, Any]],
                knobs: Sequence[Mapping[str, Any]], catalogue: Sequence[Mapping[str, Any]],
                check: Callable[[dict[str, Any]], Any],
                failures: list[dict[str, str]] | None = None) -> list[str]:
    """Facts about this graph an edit is written against, worked out once by the loop
    and carried as data (``ProposalContext.graph_facts``): what ``"sees": ["task"]``
    gives a new step, what a new branch of each fork can see (`_fork_inputs`), and for
    each step the refusal ``check`` (an attempt's verdict on the edited candidate) gives
    to removing it, replacing it by one step that sees the task, and merging it with the
    step after it. Accepted edits are not listed. A fact that cannot be worked out is
    said so in its row (``not worked out (<error>)``), never left out, and appended to
    ``failures`` (``{"fact", "step", "error"}``) for the run's journal. Nothing when the
    run permits no core operator."""
    from refactory_dsh.nn.core_ops import TASK, _readable, _task_inputs
    from refactory_dsh.nn.graph import Graph
    from refactory_dsh.nn.operators import TASK_MEANING

    failed = failures if failures is not None else []

    def not_worked_out(fact: str, step: str, exc: Exception) -> str:
        error = f"{type(exc).__name__}: {exc}"
        failed.append({"fact": fact, "step": step, "error": error})
        return f"{fact} not worked out ({error})"

    ops = {op.get("op") for op in catalogue} & set(CORE_OPERATORS)
    if not ops:
        return []
    try:
        state = _fit_state(node_states, edges, knobs)
        if state is None:
            raise ValueError("the graph does not read back")
        graph = Graph.from_state(state["graph"])
    except Exception as exc:  # noqa: BLE001 - said in the section and journaled
        return [f"- The graph's facts are {not_worked_out('the graph', '', exc)}."]
    out: list[str] = []
    try:
        task = sorted(_readable(graph, state, _task_inputs(graph, state)),
                      key=lambda key: (key[0] != "fact", key[1]))
    except Exception as exc:  # noqa: BLE001 - said in the section and journaled
        task = []
        out.append(f"- What `\"sees\": [\"{TASK}\"]` gives is "
                   f"{not_worked_out('the task inputs', '', exc)}.")
    if task:
        out.append(f"- `\"sees\": [\"{TASK}\"]` gives a new step "
                   f"{', '.join(name for _k, name in task)}, where it is there before the "
                   f"step: {TASK_MEANING}.")
    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        if node.kind == "fork" and "branch" in ops:
            out.append(f"- fork `{node_id}`: {_fork_inputs(node_id, state)}.")
            continue
        if node.kind not in ("agent", "op"):
            continue
        plain_next = [e.target for e in graph.out_edges(node_id) if e.condition is None
                      and graph.nodes[e.target].kind in ("agent", "op")]
        tries = [("remove", {"op": "remove", "step": node_id}),
                 ("replace by one step", {"op": "split", "step": node_id,
                                          "into": [{"kind": "agent", "sees": ["task"]}]}),
                 *([("merge with the next step", {"op": "merge",
                                                   "steps": [node_id, plain_next[0]]})]
                   if len(plain_next) == 1 else [])]
        refused = []
        for label, edit in tries:
            if edit["op"] not in ops:
                continue
            try:
                verdict = check(edit)
            except Exception as exc:  # noqa: BLE001 - said in the row and journaled
                refused.append(not_worked_out(label, node_id, exc))
                continue
            if isinstance(verdict, list):
                reason = "; ".join(str(r).removeprefix("E-precondition: ") for r in verdict)
                refused.append(f"{label} is refused ({reason})")
        if refused:
            out.append(f"- `{node_id}`: " + "; ".join(refused) + ".")
    return out


def operator_lines(catalogue: Sequence[dict[str, Any]], node_states: Sequence[dict[str, Any]],
                   edges: Sequence[dict[str, Any]], *, protected: Sequence[str] = (),
                   knob_values: Mapping[str, Any] | None = None,
                   step_routes: Mapping[str, Sequence[str]] | None = None,
                   tools: bool = True, code_nodes: bool = False) -> list[str]:
    """The structural operators: what the system wires for every one, then each core operator
    the run permits in one line -- its fields and its catalogue description (its
    `nn.core_ops.OPERATORS` line, and a written-out form where it has one) -- with one
    example, its ids taken from this graph where it has them. When the run has routes to
    choose from (``step_routes``), one line says the ``route`` field and its values; without,
    no operator lists it. Nothing when the run permits none."""
    from refactory_dsh.nn import core_ops

    ops = {op.get("op") for op in catalogue}
    by_op = {op.get("op"): op for op in catalogue}
    chosen = [name for name in CORE_OPERATORS if name in ops]
    if not chosen:
        return []
    ids = _example_ids(node_states, edges, protected)
    removable = _removable_step(node_states, edges, protected,
                                _example_state(node_states, edges, knob_values))
    lines = ["", STRUCTURAL_LEAD]
    state = _example_state(node_states, edges, knob_values)
    for name in chosen:
        spec = core_ops.OPERATORS[name]
        # Ids this graph's own steps fill where the operator applies to them; a step this
        # graph can lose for `remove`; placeholders otherwise, never a step it refuses.
        own = {**ids, "step": removable or "<step>"} if name == "remove" else _fitting_ids(
            name, ids, node_states, edges, protected, state)
        entry = by_op.get(name) or {}
        description = str(entry.get("description") or f"{spec['line']}.")
        fields = [f for f in entry.get("fields") or spec["fields"]
                  if step_routes or f != core_ops.ROUTE_FIELD]
        if not tools:
            # A run that offers no tool step offers no 'tools' (`NO_TOOLS_LINE`).
            description, fields = core_ops.without_tools(description, fields)
        lines.append(f"- `{name}` ({_core_fields(fields)}): "
                     f"{description} E.g. " + json.dumps(core_ops.example(name, own)))
    if step_routes:
        lines.append(core_ops.route_field_line(step_routes))
    if tools or code_nodes:
        lines += tool_environment_lines(node_states, tools=tools, code_nodes=code_nodes)
    return lines


def tool_environment_lines(node_states: Sequence[Mapping[str, Any]], *,
                           tools: bool = True, code_nodes: bool = False) -> list[str]:
    """What a sealed step's commands run with in each per-item part whose Task declares
    it (`task.Task.tool_environment`), in the Task's own words: a fact, said only where the
    run offers a tool step or a sealed code step. Nothing for a Task that declares none
    (or cannot be read)."""
    from refactory_dsh.task import TaskError, task_tool_environment

    lines: list[str] = []
    for node in node_states:
        if node.get("kind") != "data" or not node.get("task_ref"):
            continue
        try:
            environment = task_tool_environment(str(node["task_ref"]))
        except (TaskError, ImportError):
            continue
        if environment is not None:
            kinds = []
            if tools:
                kinds.append("a step with `tools`")
            if code_nodes:
                kinds.append("a sealed code step (`\"sealed\": true`)")
            line = (f"- in the per-item part `{node.get('id')}`, "
                    f"{' or '.join(kinds)} runs its commands with {environment.summary}")
            if line not in lines:
                lines.append(line)
    return lines


def _sealed_code_imports(node_states: Sequence[Mapping[str, Any]]
                         ) -> tuple[list[str], bool]:
    """What a sealed code step may import, derived from the Task's tool environment.

    Returns ``(import_lines, has_packages)``: when any environment declares
    ``packages``, each line names the packages by name and version
    (``"openpyxl 3.1.5, pandas 2.3.3"``); otherwise each line is the
    environment's ``summary`` (the proposer knows what is there, but not
    exactly which modules). ``has_packages`` is True when every environment
    declared its packages.
    """
    from refactory_dsh.task import TaskError, task_tool_environment

    seen: list[str] = []
    all_have_packages = True
    for node in node_states:
        if node.get("kind") != "data" or not node.get("task_ref"):
            continue
        try:
            environment = task_tool_environment(str(node["task_ref"]))
        except (TaskError, ImportError):
            continue
        if environment is None:
            continue
        if environment.packages:
            line = ", ".join(f"{name} {version}" for name, version in environment.packages)
        else:
            line = environment.summary
            all_have_packages = False
        if line not in seen:
            seen.append(line)
    return seen, bool(seen) and all_have_packages


#: The core operators whose new step takes 'to' (where its output goes).
_TAKES_TO = ("insert", "branch", "map")


#: The text of the step a completion adds to take over the graded step's outputs.
_TAKE_OVER_TEXT = "<write the final result from the work above>"


def _reader_edit(after: Mapping[str, Any], node_id: str, edit: Mapping[str, Any],
                 before: Mapping[str, Any], first: str | None = None
                 ) -> tuple[str, dict[str, Any]] | None:
    """How a new step ``node_id`` that nothing reads in ``after`` reaches a reader, as one
    edit, the way its refusal's hint says (`core_ops._reader_hint`): ``("append", edit)``
    rewiring the nearest later agent or code step to see it as well as what it sees now,
    or adding a step that takes over the graded step's outputs after it (after its join
    when it is a fork's branch); ``("replace", edit)`` for an ``insert`` given
    ``takes_over`` itself; with ``first``, only a rewire of that step. ``None`` when no
    step could read it."""
    from refactory_dsh.nn import core_ops

    graph = nn_edits._graph(dict(after))
    later = [r for r in core_ops._readers_from(graph, node_id, after)
             if r != node_id and graph.nodes[r].kind == "agent"]
    if first is not None:
        # Only that step (the one the new step was put before), when it is an agent.
        later = [r for r in later if r == first]
        if not later:
            return None
    if later:
        reader = graph.nodes[later[0]]
        sees = sorted({name for _k, name in core_ops._readable(
            graph, after, core_ops._consumed(reader, after))})
        return "append", {"op": "rewire", "step": reader.id, "sees": [*sees, node_id]}
    graded = core_ops._graded_step(graph, node_id, edit, before)
    if graded is None or edit.get("op") == "route":
        return None
    if edit.get("op") == "insert" and "takes_over" not in edit:
        return "replace", {**edit, "takes_over": graded}
    out = [e.target for e in graph.out_edges(node_id)]
    at = out[0] if len(out) == 1 and graph.nodes[out[0]].kind == "join" else node_id
    return "append", {"op": "insert", "after": at, "kind": "agent", "text": _TAKE_OVER_TEXT,
                      "takes_over": graded}


# --- cross-module imports (resolved at definition time, used at call time) ---
from .edits import _core_fields, _edit_fields  # noqa: E402
from .items import HARNESS_TIMEOUT, _completed  # noqa: E402
from .objective import _table  # noqa: E402
