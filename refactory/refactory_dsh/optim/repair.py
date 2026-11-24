"""Repairing a refused proposal's mechanics, never its meaning.

The edit operators stay naive (`nn.edits`): they never fix what they are given. Repair
happens here, on the attempt path every strategy's proposals take (`Optimizer.attempt`),
in two forms, and every fix is said: in the candidate (``Candidate.repairs``, its edits
are the repaired ones, so they replay to its state), in the optimizer trace (kind
``repair``) and in the run's epoch line.

**Deterministic fixes**, with no model call, only where what was meant is unambiguous:

- an edit's ``target`` (or ``first``, ``second``, a core operator's ``step``, ``after``,
  ``before`` or ``fork``, a ``remove_node``'s ``id``, an edge's ends) that names a file
  instead of a step is read as the step that writes it, when exactly one step does
  (`fix_edit`);
- a new node whose id is taken gets a fresh id, when no other edit of the proposal names
  that id (`fix_edit`);
- a ``set_knob`` in a bundle that sets a knob to the value it already holds is dropped
  (`noop_knob`);
- a ``{{name}}`` slot that names a file rather than a fact -- no step makes the fact
  ``name``, and exactly one file there is called ``name`` -- gets the file's whole text
  from a ``file_text`` op inserted before the step (`wire_file_slots`). Only steps the
  proposal added or changed are judged, and not when the op would rewire a protected
  step.

Anything else is refused with the exact reason, as before.

**The slim repair call.** A proposal refused for its mechanics (`repairable`: an edit's
shape, a precondition, a validator rule, lowering, a dry run, a smoke run that failed with
an error, ``E-smoke``; not the ideas behind it,
``E-integrity``, ``E-noop``, ``E-duplicate``, nor a run's policy, the grader-side rule
V17 or the change cap) is sent back to the
model with the reasons and only what fixing it needs (`repair_request`): the proposal as
written, the pipeline's tables, the knobs it touches, the operations and the node
schemas, a fraction of a full request. The answer is the fixed proposal only; its
rationale and prediction stay the original's. At most `REPAIR_CALLS_PER_PROPOSAL` calls
per refused proposal and `REPAIR_CALLS_PER_EPOCH` per epoch; when they do not produce a
candidate the loop asks again with a full request, as it did before.

**The wiring review.** A proposal that applies but whose edit adds wiring warnings
(`nn.validate.wiring_warnings`: a step whose output nothing reads, a reloop whose review
reaches no prompt) is shown them in one slim call of the same shape (`wiring_request`),
before its candidate is judged, smoked or evaluated (``Loop._wiring_check``). The answer
changes the wiring or keeps the change as it is, with a reason: a warning is information,
never a refusal.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Slim repair calls for one refused proposal.
REPAIR_CALLS_PER_PROPOSAL = 2
#: Slim repair calls in one epoch, over every refused proposal.
REPAIR_CALLS_PER_EPOCH = 3
#: Refusals about the idea, not its mechanics: never sent to a slim repair call.
IDEA_CODES = ("E-integrity", "E-noop", "E-duplicate")
#: Validator rules that guard the grader side, as ``E-secret`` and ``E-protected`` do: a
#: proposal refused by one is not sent to a slim repair call either.
GUARD_RULES = ("V17",)
#: Refusals about how a proposal is written, which a slim repair call may fix, with every
#: validator rule (``V…``) but `GUARD_RULES`. ``E-too-many-edits`` is not one: fitting
#: the cap means choosing which changes to drop, which is the proposer's call.
MECHANICS_CODES = ("E-precondition", "E-shape", "E-node-text", "E-binding",
                   "E-lowering", "E-node", "E-edge", "E-knob", "E-patch",
                   "E-op", "E-gate-kind", "E-code-run", "E-wiring", "E-smoke")
#: The most characters the slim request spends on the nodes in full and the schemas.
REPAIR_NODE_CHARS = 14000
#: The most characters of one touched knob's current value the slim request shows.
REPAIR_KNOB_CHARS = 6000


def _code(reason: str) -> str:
    return str(reason).split(" ", 1)[0].rstrip(":")


def repairable(reasons: Sequence[str]) -> bool:
    """Whether a refusal is about mechanics only: every reason's code is one of
    `MECHANICS_CODES` or a validator rule other than `GUARD_RULES`; anything else -- the
    idea (`IDEA_CODES`), a run's policy, the change cap -- is not."""
    codes = [_code(reason) for reason in reasons if reason]
    if not codes:
        return False
    return all(code in MECHANICS_CODES
               or (re.fullmatch(r"V\d+", code) and code not in GUARD_RULES) for code in codes)


# ── deterministic fixes ───────────────────────────────────────────────────

#: Fields of an edit that name an existing step.
_STEP_FIELDS = {"target": None, "first": None, "second": None, "step": None, "after": None,
                "before": None, "fork": None}
_NODE_FIELDS = ("node", "gate")
_NODE_LIST_FIELDS = ("into", "branches")


def _writers(graph: Any) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for node in graph.nodes.values():
        for name in node.writes:
            out.setdefault(name, []).append(node.id)
    return out


def _named_fields(edit: Mapping[str, Any]) -> list[str]:
    op = edit.get("op")
    if op == "remove_node":
        return ["id"]
    if op in ("add_edge", "remove_edge"):
        return ["source", "target"]
    return [key for key in _STEP_FIELDS if key in edit]


def _mentions(value: Any, name: str) -> bool:
    return bool(re.search(r"(?<![\w.-])" + re.escape(name) + r"(?![\w-])",
                          json.dumps(value, default=str)))


def fix_edit(edit: dict[str, Any], state: Mapping[str, Any], steps: Sequence[Any], index: int
             ) -> tuple[dict[str, Any], list[str], str]:
    """``edit`` with the fixes whose meaning is unambiguous, applied against ``state`` (the
    state the edits before it left), what was fixed, and -- when a new node's id is taken
    and what was meant is not clear -- why it is refused instead (empty otherwise).

    A taken id is renamed only when nothing else names it: not the edit's own ``target``,
    ``first`` or ``second`` (a node re-declared over the step the edit is about may mean
    "replace this step", which the language does not do by adding one), and no other edit
    of the proposal."""
    from refactory_dsh.nn.edits import _fresh_id
    from refactory_dsh.nn.graph import Graph

    try:
        graph = Graph.from_state(dict(state.get("graph") or {}))
    except Exception:  # noqa: BLE001 - an unreadable state is refused where it is applied
        return edit, [], ""
    where = f"edit {index + 1} ({edit.get('op')})" if len(steps) > 1 else str(edit.get("op"))
    fixes: list[str] = []
    out = dict(edit)
    writers = _writers(graph)
    for key in _named_fields(out):
        value = out.get(key)
        if isinstance(value, str) and value not in graph.nodes and len(writers.get(value, ())) == 1:
            step = writers[value][0]
            out[key] = step
            fixes.append(f"{where}: {key} {value!r} names the file step {step!r} writes, so it "
                         f"was read as step {step!r}")
    others = [step for position, step in enumerate(steps) if position != index]
    own = {out.get(key) for key in _STEP_FIELDS if isinstance(out.get(key), str)}
    taken = set(graph.nodes)
    refusals: list[str] = []

    def renamed(node: Any) -> Any:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            return node
        old = node["id"]
        if old not in taken:
            taken.add(old)
            return node
        if old in own:
            refusals.append(
                f"the new node's id {old!r} is the step this edit is about, so it is not clear "
                "whether a second step or a replacement is meant; a step is not replaced by "
                "adding one with its id: change it through its knobs, or give the new step "
                "an id of its own")
            return node
        if _mentions(others, old):
            refusals.append(
                f"the new node's id {old!r} is already a step's, and another edit of the "
                "proposal names it, so which of the two it means is not clear; give the new "
                "step an id of its own")
            return node
        new = _fresh_id(graph, old)
        while new in taken:
            new = _fresh_id(graph, f"{new}_")
        taken.add(new)
        fixes.append(f"{where}: node id {old!r} is already a step's, so the new node was "
                     f"named {new!r}")
        return {**node, "id": new}

    for key in _NODE_FIELDS:
        if key in out:
            out[key] = renamed(out[key])
    for key in _NODE_LIST_FIELDS:
        if isinstance(out.get(key), list):
            out[key] = [renamed(node) for node in out[key]]
    return out, fixes, refusals[0] if refusals else ""


def noop_knob(edit: Any, state: Mapping[str, Any]) -> bool:
    """Whether ``edit`` is a ``set_knob`` that gives a knob the value it already holds."""
    from refactory_dsh.nn.knob import Knob

    if not isinstance(edit, dict) or edit.get("op") != "set_knob" or "value" not in edit \
            or "patch" in edit:
        return False
    entry = (state.get("knobs") or {}).get(edit.get("name"))
    if not isinstance(entry, dict):
        return False
    try:
        return Knob.from_spec(entry["spec"]).coerce(edit["value"]) == entry.get("value")
    except Exception:  # noqa: BLE001 - a bad value is refused where it is applied
        return False


def _touched(parent_state: Mapping[str, Any], state: Mapping[str, Any], graph: Any) -> list[str]:
    """Nodes the proposal added or changed: their declaration differs from the parent's,
    or a knob they read or bind holds another value."""
    before = (parent_state.get("graph") or {}).get("nodes") or {}
    old_knobs, new_knobs = parent_state.get("knobs") or {}, state.get("knobs") or {}
    out = []
    for node_id, node in sorted(graph.nodes.items()):
        names = [*node.knobs, *(knob for _f, knob in node.field_bindings)]
        if (before.get(node_id) != node.to_state()
                or any((old_knobs.get(n) or {}).get("value") != (new_knobs.get(n) or {}).get("value")
                       for n in names)):
            out.append(node_id)
    return out


def wire_file_slots(parent_state: Mapping[str, Any], state: dict[str, Any]
                    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """``state`` with a ``file_text`` op before each added or changed step whose ``{{name}}``
    slot names a file rather than a fact, handing it the file's whole text; the edits that
    did it and what was fixed.

    Fixed only when nothing makes the fact ``name`` anywhere in the graph, and exactly one
    file there before the step (on every path) is called ``name`` (its name, or its name
    without the extension); and the step is not a fork's branch or a join's source (a
    step put in front of it would not be one). Anything else is left for the validator to
    refuse (V15)."""
    from refactory_dsh.nn import edits as nn_edits
    from refactory_dsh.nn.graph import Graph
    from refactory_dsh.nn.operators import file_text_node
    from refactory_dsh.nn.validate import available_inputs, knob_slot_facts, template_facts

    try:
        graph = Graph.from_state(dict(state.get("graph") or {}))
    except Exception:  # noqa: BLE001
        return state, [], []
    values = {name: (entry or {}).get("value") for name, entry in (state.get("knobs") or {}).items()}
    produced = {name for node in graph.nodes.values() for name in node.produces}
    named = {member for node in graph.nodes.values()
             for member in (*node.targets, *node.optional_targets, *node.sources)}
    have = available_inputs(graph)
    added: list[dict[str, Any]] = []
    fixes: list[str] = []
    for node_id in _touched(parent_state, state, graph):
        node = graph.nodes[node_id]
        if node_id in named or node_id not in graph.reachable():
            continue
        if node.kind == "agent":
            slots = template_facts(node, values)
        elif node.kind in ("op", "gate"):
            slots = knob_slot_facts(node, values)
        else:
            continue
        facts = {n for kind, n in have.get(node_id, ()) if kind == "fact"}
        files = sorted(n for kind, n in have.get(node_id, ()) if kind == "file")
        for slot in sorted(slots - facts - produced - set(values)):
            matches = [name for name in files if name == slot or Path(name).stem == slot]
            if len(matches) != 1:
                continue
            current = Graph.from_state(state["graph"])
            reader = nn_edits._fresh_id(current, f"{node_id}_{slot}_text")
            edit = {"op": "insert_before", "target": node_id,
                    "node": file_text_node(reader, matches[0], slot)}
            try:
                state = nn_edits.apply(edit, state)
            except nn_edits.EditError:
                continue
            added.append(edit)
            fixes.append(f"{node_id}: {{{{{slot}}}}} names the file {matches[0]!r}, not a fact, "
                         f"so a file_text op {reader!r} before it now sets {{{{{slot}}}}} to "
                         "that file's whole text")
    return state, added, fixes


# ── the slim repair call ──────────────────────────────────────────────────


def _touched_knobs(steps: Sequence[Any], knobs: Mapping[str, Any]) -> list[str]:
    names = []
    for step in steps:
        if isinstance(step, dict) and step.get("op") in ("set_knob", "remove_knob"):
            name = step.get("name")
            if isinstance(name, str) and name in knobs and name not in names:
                names.append(name)
    return names


def repair_request(proposal: Any, reasons: Sequence[str], parent_state: Mapping[str, Any], *,
                   catalogue: Sequence[Mapping[str, Any]], code_nodes: bool = False,
                   person_gates: bool = False, protected: Sequence[str] = (),
                   max_edits: int = 1) -> str:
    """The slim repair request: the refused proposal as written, why it was refused, and
    what fixing its mechanics needs -- the pipeline it applies to (nodes, edges, what each
    node has), the current text of each knob it edits, the rules, the operations and the
    node schemas. No trajectories, no history."""
    return _slim_request(
        proposal, parent_state,
        title="# Repair a refused proposal",
        intro=("The proposal below was refused before anything ran, for how it is written, "
               "not for what it tries. Fix only what the reasons name. Keep every change it "
               "makes and what each one is for: the same steps, prompts, code and settings, "
               "changed only as far as the reasons require. Do not add, drop or redesign "
               "changes. Keep every text it wrote and fix only the refused field: move its "
               "text to a field the operation takes; if none can hold it, keep the edit "
               "without it (the proposer is told what was left out). If it cannot be fixed "
               "without changing what it means, answer with an empty `edits` list and say "
               "why."),
        heading="## The refused proposal", reasons_heading="## Why it was refused",
        reasons=reasons,
        answer=('{"edits": [<the fixed edits, in order>], "why": "<what you changed to fix '
                'it>"}'),
        catalogue=catalogue, code_nodes=code_nodes, person_gates=person_gates,
        protected=protected, max_edits=max_edits)


def wiring_request(proposal: Any, warnings: Sequence[str], parent_state: Mapping[str, Any], *,
                   catalogue: Sequence[Mapping[str, Any]], code_nodes: bool = False,
                   person_gates: bool = False, protected: Sequence[str] = (),
                   max_edits: int = 1) -> str:
    """The slim wiring review (``Loop._wiring_check``): a proposal that applies and is
    valid, the wiring warnings its edit adds (`nn.validate.wiring_warnings`), and what
    changing its wiring needs -- the same tables as `repair_request`. The proposer fixes
    the wiring, or keeps the change as it is and says why: a warning is not a refusal."""
    return _slim_request(
        proposal, parent_state,
        title="# Check the wiring of your proposal",
        intro=("The proposal below applies and is valid, and it will run as it is unless you "
               "change it. Its edit adds the wiring warnings listed under it: the graph runs, "
               "but likely not as you meant (a step whose output nothing reads still runs on "
               "every item, and its work is lost). If a warning names something you did not "
               "mean, change only the wiring it names -- wire the output into the step that "
               "should read it, or remove the step that is no longer needed -- and keep every "
               "other change and text of the proposal as it is. If the wiring is what you "
               "meant (a step that opens a file it does not declare, say), answer with an "
               "empty `edits` list and say why."),
        heading="## Your proposal", reasons_heading="## The wiring warnings its edit adds",
        reasons=warnings,
        answer=('{"edits": [<the proposal\'s edits with the wiring changed, in order, or none '
                'to keep it as it is>], "why": "<what you changed, or why the wiring is '
                'meant>"}'),
        catalogue=catalogue, code_nodes=code_nodes, person_gates=person_gates,
        protected=protected, max_edits=max_edits)


def _slim_request(proposal: Any, parent_state: Mapping[str, Any], *, title: str, intro: str,
                  heading: str, reasons_heading: str, reasons: Sequence[str], answer: str,
                  catalogue: Sequence[Mapping[str, Any]], code_nodes: bool,
                  person_gates: bool, protected: Sequence[str], max_edits: int) -> str:
    """A slim request about one proposal (`repair_request`, `wiring_request`)."""
    from refactory_dsh.nn.graph import Graph

    from .base import (_available_section, _clip_middle, _edit_fields, _node_specs, _rules_text,
                       _table, available_before)

    graph_state = dict(parent_state.get("graph") or {})
    knobs = dict(parent_state.get("knobs") or {})
    values = {name: (entry or {}).get("value") for name, entry in knobs.items()}
    try:
        graph = Graph.from_state(graph_state)
        nodes = [[n.id, n.kind, n.call or n.evaluator, list(n.knobs), n.notes]
                 for n in sorted(graph.nodes.values(), key=lambda n: n.id)]
        edges = [[e.source, e.target, e.condition] for e in graph.edges]
    except Exception:  # noqa: BLE001 - shown as far as it reads
        nodes, edges = [], []
    steps = list(getattr(proposal, "steps", []) or [])
    touched = _touched_knobs(steps, knobs)
    lines = [
        title,
        "",
        intro,
        "",
        heading,
        f"Rationale: {getattr(proposal, 'rationale', '') or '(none)'}",
        f"Prediction: {getattr(proposal, 'prediction', '') or '(none)'}",
        "```json",
        json.dumps({"edits": steps}, indent=1, default=str),
        "```",
        "",
        reasons_heading,
        *[f"- {reason}" for reason in reasons],
        "",
        "## The pipeline it applies to",
        *([f"Protected nodes: {', '.join(f'`{n}`' for n in protected)}."] if protected else []),
        "Nodes:",
        _table(["id", "kind", "call", "knobs", "notes"], nodes),
        "Edges:",
        _table(["source", "target", "condition"], edges),
        *_available_section(available_before(graph_state, values), code_nodes=code_nodes),
        "",
    ]
    if touched:
        lines += ["## The knobs it edits, as they stand now",
                  "A patch's `target` must be text the knob holds, exactly."]
        for name in touched:
            lines += [f"### `{name}`", "```", _clip_middle(str(values.get(name)), REPAIR_KNOB_CHARS),
                      "```"]
        lines.append("")
    lines += [
        *_node_specs(list(graph_state.get("nodes", {}).values()), section_chars=REPAIR_NODE_CHARS,
                     knob_values=values, person_gates=person_gates),
        "",
        "## The rules",
        _rules_text(),
        "",
        "## The operations",
        _table(["op", "kind", "macro", "fields", "description"],
               [[c["op"], c["kind"], c["macro"], _edit_fields(c["fields"]), c["description"]]
                for c in catalogue]),
        "",
        "## Answer",
        "One JSON object and nothing else: " + answer
        + (f". At most {max_edits} changes." if max_edits > 1 else "."),
    ]
    return "\n".join(lines)


#: The shortest text of a proposal whose loss in a repair is said (`lost_content`): an id
#: or a label renamed is a fix, not lost content.
LOST_TEXT_CHARS = 20


def _texts(steps: Sequence[Any]) -> list[tuple[str, str]]:
    """Each free text of ``steps`` (prompts, code, knob values, patch contents; not the
    wiring, `optim.base.EDIT_WIRING_FIELDS`) at least `LOST_TEXT_CHARS` long, with where
    it is: ``edit N (op) 'field'``."""
    from .base import _map_text

    out: list[tuple[str, str]] = []
    for index, step in enumerate(steps, 1):
        if not isinstance(step, Mapping):
            continue
        where = f"edit {index} ({step.get('op')})" if len(steps) > 1 else str(step.get("op"))
        for key, value in step.items():
            found: list[str] = []
            _map_text({key: value}, lambda text: found.append(text) or text)
            out += [(f"{where} '{key}'", text) for text in found
                    if len(text.strip()) >= LOST_TEXT_CHARS]
    return out


def lost_content(original: Sequence[Any], fixed: Sequence[Any]) -> str:
    """What a repair (``fixed``) dropped or changed of the proposal as written
    (``original``): each field whose text does not appear, whitespace aside, anywhere in
    the fix, with its length; empty when every text landed."""
    landed = " ".join(" ".join(text.split()) for _where, text in _texts(fixed))
    lost: dict[str, int] = {}
    for where, text in _texts(original):
        if " ".join(text.split()) not in landed:
            lost[where] = lost.get(where, 0) + len(text)
    return ", ".join(f"{where} ({chars} chars)" for where, chars in lost.items())


def repaired_proposal(payload: Mapping[str, Any], original: Any) -> tuple[Any | None, str]:
    """The fixed proposal a slim repair answer carries, with the original's rationale,
    prediction, confidence and parent; or ``None`` and why there is none."""
    from .base import Proposal
    from .random_optimizer import _proposal_edits

    steps, why = _proposal_edits(dict(payload))
    if not steps:
        said = str(payload.get("why") or "").strip()
        return None, (f"no fixed edits ({said})" if said else f"no fixed edits: {why}")
    return Proposal(edits=tuple(steps), rationale=getattr(original, "rationale", ""),
                    prediction=getattr(original, "prediction", ""),
                    confidence=getattr(original, "confidence", 0.0),
                    parent=getattr(original, "parent", None)), ""
