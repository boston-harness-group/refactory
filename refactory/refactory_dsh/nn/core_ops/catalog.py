from __future__ import annotations

import json
import re
from typing import Any, Collection, Mapping, Sequence

from refactory_dsh.nn.graph import Graph, Node
from refactory_dsh.nn.operators import TASK_MEANING
from refactory_dsh.seal import SEALED_TOOLS

from ._common import CORE_OPS, LOOP_ITERATIONS, MAP_MAX, TASK, TOOLS_HINTS, _DEFAULT_WIRED, _Refused, _edits
from .state import _available, _consumed, _data_ends, _downstream, _is_code, _knob_values, _outputs, _task_knob, _uses_tools
from .visibility import _readable
from .build import _Steps, _simulate
from .placement import _filled_text_knobs, dead_text_knobs
from .checks import _upstream
from .insert import _expand_insert, _pre_insert
from .remove import _expand_remove, _pre_remove
from .parallelize import _expand_parallelize, _fork_join_ids, _pre_parallelize
from .serialize import _expand_serialize, _pre_serialize
from .branch import _expand_branch, _pre_branch
from .rewire import _expand_rewire, _pre_rewire
from .loop import _expand_loop_line, _pass_of, _pre_loop_line, is_one_line_loop
from .route import _expand_route, _pre_route
from .map_op import _expand_map, _map_part, _pre_map
from .fallback import _expand_fallback, _pre_fallback
from .split_op import _expand_split_line, _pre_split_line, is_one_line_split
from .merge import _expand_merge, _pre_merge

import sys as _sys


class _Proxy:
    """Late-bound lookup on the core_ops package for monkeypatch compat."""
    __slots__ = ()
    def __getattr__(self, name):
        return getattr(_sys.modules['refactory_dsh.nn.core_ops'], name)


_co = _Proxy()


def _guarded(check):
    """A precondition that says its refusal (`_Refused`) as the reason."""
    def precondition(edit: dict, state: dict) -> str | None:
        try:
            return check(edit, state)
        except _Refused as exc:
            return str(exc)
    return precondition




def _guarded_expand(expand):
    def run(edit: dict, state: dict) -> list[dict]:
        token = _DEFAULT_WIRED.set(set())
        try:
            steps = expand(edit, state)
            wired = _DEFAULT_WIRED.get()
        except _Refused as exc:
            raise _edits().EditError(str(exc), code="E-precondition") from exc
        finally:
            _DEFAULT_WIRED.reset(token)
        steps = steps + _settle_optional(state, steps)
        return steps + _with_item_files(state, steps, wired)
    return run






def item_files_before(graph: Graph, state: Mapping[str, Any], node_id: str) -> list[str]:
    """The item's input files (`task.Task.item_files`) of the per-item pass ``node_id``
    runs in -- its innermost data node's Task's -- that are there before it on every path;
    none at the top level, or where the Task declares none or cannot be read (another
    rule's to say). Never a grader-side file: an item file is checked never to be one."""
    from refactory_dsh.task import TaskError, task_item_files

    data_id = _pass_of(graph, node_id)
    if data_id is None:
        return []
    try:
        files = task_item_files(graph.nodes[data_id].task_ref or "")
    except (TaskError, ImportError):
        return []
    there = _readable(graph, state, set(_available(graph).get(node_id, set())))
    return sorted(path for path in files.values() if ("file", path) in there)




def _with_item_files(start: Mapping[str, Any], steps: Sequence[dict],
                     wired: Collection[str]) -> list[dict]:
    """Steps that give each new sealed tool step (`workspace.is_sealed`) the operator wired
    by default (``wired``: the proposer gave no 'sees' or 'reads' for it) the item's input
    files of the pass it runs in (`item_files_before`) in its declared ``reads``, beside
    what it reads already: a default, never a limit. A step whose inputs the proposer named
    keeps exactly those (the candidate's notes say when it does not see the item's input
    files, `item_input_notes`)."""
    from refactory_dsh.workspace import is_sealed

    if not wired:
        return []
    end = _simulate(start, steps)
    new = _edits()._graph(end)
    out: list[dict] = []
    for node_id in sorted(set(wired) & set(new.nodes)):
        node = new.nodes[node_id]
        if not is_sealed(node):
            continue
        missing = [path for path in item_files_before(new, end, node_id)
                   if path not in node.reads]
        if not missing:
            continue
        more = _co._respec(end, node_id, {**node.to_state(), "reads": [*node.reads, *missing]})
        out += more
        end = _simulate(end, more)
        new = _edits()._graph(end)
    return out




def _settle_optional(start: Mapping[str, Any], steps: Sequence[dict]) -> list[dict]:
    """Steps that keep each ``optional`` binding (`Graph.optional_targets_of`) on a branch's
    first step after an edit reshaped a fork: put back on a first step the edit rebuilt
    under its id (rewire's collect step), moved to the one new first step of that same fork
    (an insert before it), else dropped with its knob (the fork is gone, or the branch went
    with ``remove``), so V14 never refuses what the edit left and a setting is not lost."""
    before = _edits()._graph(start)
    build = _Steps(start)
    build.add(list(steps))
    stranded = []
    rebuilt = []
    for fork in (n for n in before.nodes.values() if n.kind == "fork"):
        for head in fork.targets:
            knob = before.nodes[head].bound_knob("optional") if head in before.nodes else ""
            if not knob:
                continue
            now = build.graph.nodes.get(fork.id)
            # The value before the edit: no operator sets it, and a rebuilt step's knob may
            # have been declared again at its default.
            value = (start.get("knobs") or {})[knob]["value"]
            if now is not None and now.kind == "fork" and head in now.targets:
                if not build.graph.nodes[head].bound_knob("optional"):
                    rebuilt.append((head, knob, value))
                continue
            stranded.append((fork, head, knob, value))
    for head, knob, value in rebuilt:
        if knob not in (build.state.get("knobs") or {}):
            build.add([{"op": "add_knob", "name": knob,
                        "spec": dict((start.get("knobs") or {})[knob]["spec"])}])
        again = build.graph.nodes[head].to_state()
        again["knobs"] = [*(again.get("knobs") or []), knob]
        again["field_bindings"] = [*(again.get("field_bindings") or []), ["optional", knob]]
        build.add(_co._respec(build.state, head, again))
        build.add([{"op": "set_knob", "name": knob, "value": value}])
    for fork, head, knob, value in stranded:
        graph = build.graph
        now = graph.nodes.get(fork.id)
        fresh = [t for t in (now.targets if now is not None and now.kind == "fork" else ())
                 if t not in fork.targets and not graph.nodes[t].bound_knob("optional")]
        spec = dict((start.get("knobs") or {})[knob]["spec"])
        if len(fresh) == 1:
            # The knob is named after the step that binds it (E-binding), so it moves too.
            new_knob = f"{fresh[0]}__optional"
            moved = graph.nodes[fresh[0]].to_state()
            moved["knobs"] = [*(moved.get("knobs") or []), new_knob]
            moved["field_bindings"] = [*(moved.get("field_bindings") or []),
                                       ["optional", new_knob]]
            build.add([{"op": "add_knob", "name": new_knob, "spec": spec}])
            build.add(_co._respec(build.state, fresh[0], moved))
            build.add([{"op": "set_knob", "name": new_knob, "value": value}])
        if head in build.graph.nodes:
            if knob in (build.state.get("knobs") or {}) and value != spec.get("default"):
                # Back to its default while still read, so the respec has nothing to restore.
                build.add([{"op": "set_knob", "name": knob, "value": spec.get("default")}])
            kept = build.graph.nodes[head].to_state()
            kept["knobs"] = [k for k in kept.get("knobs") or [] if k != knob]
            kept["field_bindings"] = [pair for pair in kept.get("field_bindings") or []
                                      if pair[0] != "optional"]
            build.add(_co._respec(build.state, head, kept))
        if knob in (build.state.get("knobs") or {}) and knob not in build.graph.knob_consumers():
            build.add([{"op": "remove_knob", "name": knob}])
    return build.steps[len(steps):]




#: Where a new step's output goes (`_place`): one phrase for every operator that says it.
REACHES = ("its output goes into the nearest prompt {where} that can take it -- an agent's own "
           "prompt knob, or the one text knob an op fills facts into -- or where 'to' says (a "
           "prompt knob, or a step: an agent whose prompt is no knob gets a section for it); "
           "with neither, refused unless a step reads it")



#: The field a new agent step names its own route in (`Node.route`).
ROUTE_FIELD = "route"




def route_field_line(offered: Mapping[str, Sequence[str]]) -> str:
    """The request's one line for ``route``: where it goes and the run's values for it
    (`workers.StepRoutes.offered`)."""
    values = "; ".join(f"{key} {', '.join(vals)}" for key, vals in offered.items())
    # Every operator that takes the field, so the line cannot leave one out.
    ops = ", ".join(f"`{name}`" for name, entry in OPERATORS.items()
                    if ROUTE_FIELD in entry["fields"])
    return (f"- `{ROUTE_FIELD}` on a new agent step ({ops}, or a step in "
            "'into', 'new' or 'to'): its own model, effort or output cap, any of them, each "
            f"one of this run's -- {values}; left out, it runs on its neighbour's. E.g. "
            + json.dumps({ROUTE_FIELD: {key: vals[0] for key, vals in offered.items()}}))




#: What the insert line says of 'tools', and the map line: dropped from the request, with
#: the 'tools' field, in a run that offers no tool step (`without_tools`).
INSERT_TOOLS = ("; 'tools' (an agent): e.g. " + json.dumps(list(SEALED_TOOLS))
                + "; a new step may have them though the steps beside it have none or refuse "
                "theirs (an existing step keeps its own tool rules); it runs sealed: only the "
                "files it sees (in the per-item part, without 'sees', the item's input files "
                "too), no network, no grader files")


MAP_TOOLS = "; 'tools' as insert's: each run sealed"


#: What the branch and merge lines say of 'tools'.
STEP_TOOLS = "; 'tools' as insert's (sealed)"


#: What the merge line says of 'tools'.
MERGE_TOOLS = "; 'tools' as insert's (sealed), else a sealed step's own come along"


#: What the route line says of 'tools', not said twice in one request.
ROUTE_JUDGE_TOOLS = "; 'tools' as loop's judge"


#: What the loop line says of 'tools': the agent judge's, as insert's.
JUDGE_TOOLS = ("; with 'tools' (an agent judge, as insert's) it runs sealed instead, with "
               "the files it sees in its workspace, and its reply starts with the verdict")



#: What the insert line says of 'sealed' on a code step, and other operators that accept it.
SEALED_CODE = ("; 'sealed' true on a code step: its code runs in the Task's tool "
               "environment (its Python interpreter, its extra commands) sealed -- the files "
               "it sees read-only, no network, no grader files -- so it may import any "
               "package the Task provides (openpyxl, pandas, numpy) and read item files; "
               "the import allowlist does not apply; the Task must declare a tool environment")


#: The same for a code judge in loop or route.
SEALED_JUDGE_CODE = ("; with 'sealed' true on a code judge, it runs in the Task's tool "
                     "environment sealed, as insert's 'sealed' on a code step")


#: What the branch, merge and fallback lines say of 'sealed'.
STEP_SEALED = "; 'sealed' as insert's on a code step"


#: What the fallback line says of 'tools': a multi-file fallback may be an agent with tools.
FALLBACK_TOOLS = " or an agent with tools"




def refusal_without_tools(reason: str) -> str:
    """A refusal as a run that offers no tool step says it: without `TOOLS_HINTS`."""
    for hint in TOOLS_HINTS:
        reason = reason.replace(hint, "")
    return reason




def without_tools(description: str, fields: Sequence[str]) -> tuple[str, list[str]]:
    """An operator's request line and fields with no offer of tools: for a run where a new
    step has none (no seal, or ``new_step_tools`` off)."""
    for clause in (INSERT_TOOLS, MAP_TOOLS, JUDGE_TOOLS, ROUTE_JUDGE_TOOLS, MERGE_TOOLS,
                   STEP_TOOLS, SEALED_CODE, SEALED_JUDGE_CODE, STEP_SEALED, FALLBACK_TOOLS):
        description = description.replace(clause, "")
    return description, [f for f in fields if f not in ("tools", "sealed")]




#: Each operator: its catalogue fields, its one line for the request, and one example
#: with placeholders (``<step>``, ``<next step>``, ``<fork>``, ``<gate>``).
OPERATORS: dict[str, dict[str, Any]] = {
    "insert": {
        "reads_as": ('step',),
        "fields": ("after", "before", "kind", "text", "system", "sees", "to", "id", "call",
                   "reads", "tools", "takes_over", "route", "sealed", "produces", "writes"),
        "line": "a new step after (or before) a step; 'before' puts it on every "
                "edge into that step, so 'sees' must name only what every path carries "
                "there: 'kind' agent (text = its prompt; "
                "'system' its system prompt), code (text = Python) or op (with 'call' and "
                "'reads'); it sees what that step sees plus its outputs; "
                + REACHES.format(where="after it")
                + INSERT_TOOLS
                + SEALED_CODE
                + "; 'takes_over' an earlier step: the new step writes that step's outputs, "
                "so what read them reads the new step's (a choice among attempts reaching "
                "what is scored), and that step's own go under new names; "
                "'takes_over' as [step, output, ...] takes only those outputs "
                "(the new step writes them and its own; the rest pass through unchanged, "
                "so downstream steps keep reading them from the original step)",
        "example": {"op": "insert", "after": "<step>", "kind": "agent",
                    "text": "<what the new step does>"},
    },
    "remove": {
        "reads_as": ('step',),
        "fields": ("step", "keep"),
        "line": "take a step out and reconnect around it (a fork branch too); on a gate, "
                "'keep' names the verdict whose path stays",
        "example": {"op": "remove", "step": "<step>"},
    },
    "parallelize": {
        "reads_as": ('parallel',),
        "fields": ("steps", "copies", "after", "new", "sees"),
        "line": "run consecutive 'steps' at once, or one step beside 'copies' of itself, or "
                "'new' steps after 'after', between a fork and a join",
        "example": {"op": "parallelize", "steps": ["<step>"], "copies": 2},
    },
    "serialize": {
        "reads_as": ('parallel',),
        "fields": ("fork",),
        "line": "run a fork's branches one after another, in their order",
        "example": {"op": "serialize", "fork": "<fork>"},
    },
    "branch": {
        "reads_as": ('parallel',),
        "fields": ("fork", "kind", "text", "system", "sees", "optional", "id", "to", "route",
                   "tools", "sealed", "produces", "writes"),
        "line": "a new step as another branch of a fork ('optional': true lets it fail "
                "without failing the item; 'sees' names its inputs: \"task\", step ids, or "
                "single fact or file names there before the fork; 'system' as insert's); "
                + REACHES.format(where="after the join") + STEP_TOOLS + STEP_SEALED,
        "example": {"op": "branch", "fork": "<fork>", "kind": "agent",
                    "text": "<what this branch does>"},
    },
    "rewire": {
        "reads_as": ('sees', 'told'),
        "fields": ("step", "sees", "text"),
        "line": f"change what a step sees: \"task\" ({TASK_MEANING}), step ids (their "
                "outputs), or single fact or file names there before it (that input alone); "
                "'text' gives an agent step new instructions too, in place: the same id, "
                "outputs and place in a fork or join (so a branch can work from the task "
                "alone); with 'text' and no 'sees' it keeps what it sees",
        "example": {"op": "rewire", "step": "<step>", "sees": ["task"]},
    },
    "loop": {
        "reads_as": ('loop',),
        "fields": ("step", "back_to", "judge", "text", "sees", "tools", "max_iterations", "id",
                   "target", "gate", "sealed"),
        "line": "a judge after a step that can send it back with its reason -- or, with "
                "'back_to', send the work back to an earlier agent step, so every step from "
                "there to the judge runs again (a fork in between whole); 'back_to' is one "
                "every run passes before the step, in the same per-item pass and fork "
                "branch: 'judge' agent "
                "(text = what to check; its answer starts with proceed or reloop, then why), "
                "code (text = Python returning {'verdict': 'proceed' or 'reloop', 'reason': "
                "why}) or a callable. The judge is handed what the step read and wrote, "
                "unless 'sees' says otherwise, as text (a binary or very long file only "
                "named)" + JUDGE_TOOLS + SEALED_JUDGE_CODE + ". The reason reaches the "
                "step sent "
                "back, with that "
                "step's own last output; it is sent "
                "back at most 'max_iterations' times, so it runs at most that many times "
                f"plus one (default {LOOP_ITERATIONS[0]}: up to {LOOP_ITERATIONS[0] + 1} runs; "
                f"{LOOP_ITERATIONS[1]} to {LOOP_ITERATIONS[2]}; afterwards the knob "
                "<gate>__max_iterations, the gate being <step>_check unless 'id' names it), "
                "then the run goes on",
        "example": {"op": "loop", "step": "<step>", "judge": "agent",
                    "text": "<what to check>", "max_iterations": LOOP_ITERATIONS[0]},
    },
    "route": {
        "reads_as": ('route',),
        "fields": ("after", "judge", "text", "to", "sees", "tools", "id", "sealed"),
        "line": "a judge after a step chooses one labelled next step ('to': label -> step id "
                "or a new step, whose 'takes_over' rewrites an earlier step's outputs on that "
                "way); the branches rejoin; with no 'to', proceed or redo the step in place"
                + ROUTE_JUDGE_TOOLS,
        "example": {"op": "route", "after": "<step>", "judge": "agent",
                    "text": "<how to choose>",
                    "to": {"proceed": "<next step>",
                           "revise": {"kind": "agent", "text": "<what it does>"}}},
    },
    "split": {
        "reads_as": ('step',),
        "fields": ("step", "into", "target"),
        "line": "replace a step by two new steps run in turn ('into' two), or by one ('into' "
                "one: replace); the last takes over its outputs",
        "example": {"op": "split", "step": "<step>",
                    "into": [{"kind": "agent", "text": "<first part>"},
                             {"kind": "agent", "text": "<second part>"}]},
    },
    "merge": {
        "reads_as": ('step',),
        "fields": ("steps", "kind", "text", "system", "id", "route", "tools", "sealed"),
        "line": "two consecutive steps as one, with the first's inputs and the second's "
                "outputs; 'system' as insert's" + MERGE_TOOLS + STEP_SEALED,
        "example": {"op": "merge", "steps": ["<step>", "<next step>"],
                    "text": "<what the merged step does>"},
    },
    "fallback": {
        "reads_as": ('fallback',),
        "fields": ("step", "kind", "text", "system", "sees", "id", "call", "reads", "tools",
                   "route", "sealed"),
        "line": "a step run instead when 'step' (one run per item) fails -- it raises, "
                "crashes past its retries, hits its output limit, its turn cap, its prompt "
                "limit, or is still writing at the session time limit (backstop); never for "
                "a timeout (the harness's, a declared wall or the step's own) or a harness "
                "fault: it writes that step's outputs, so what reads them still "
                "gets an answer; 'kind' as in insert; it sees what 'step' sees; when the step "
                "writes more than one file, the fallback must be a code step"
                + FALLBACK_TOOLS + " (a plain agent writes only its reply)"
                + STEP_SEALED,
        "example": {"op": "fallback", "step": "<step>", "kind": "agent",
                    "text": "<a simpler way to the same output>"},
    },
    "map": {
        "reads_as": ('map',),
        "fields": ("after", "over", "text", "system", "sees", "max", "combine", "id", "to",
                   "route", "tools"),
        "line": "after a step, a new agent step run once per element of 'over' (a list there: "
                "a fact, a file or a step's output), one model call each, all at once: a JSON "
                "array gives its items, a bulleted, numbered or lettered list its items, other "
                f"text its non-empty lines; at most 'max' run (default {MAP_MAX[0]}, up to "
                f"{MAP_MAX[2]}; past it the first 'max', and the combine is told so); 'text' "
                "says what to do with one element; 'combine' (a step as in insert) sees every "
                "result and writes one output, placed as insert's is; 'system' as insert's, "
                "for each run" + MAP_TOOLS + "; remove the elements step or the combine and the whole map "
                "goes",
        "example": {"op": "map", "after": "<step>", "over": "<step>",
                    "text": "<what to do with one element>",
                    "combine": {"kind": "agent", "text": "<how to combine the results>"}},
    },
}




def _resolved_texts(edit: dict, state: dict) -> dict:
    """``edit`` with each step text that is exactly a knob pointer (`nn.edits.KNOB_POINTER`,
    copied from the request's node reference) replaced by that knob's current text: the
    edit's own 'text', and the 'text' of each new step under 'into', 'new', 'to' and
    'combine'. A
    pointer to a knob holding no text is refused (`edits.EditError`)."""
    edits = _edits()

    def step(raw: Any, where: str) -> Any:
        if isinstance(raw, dict) and "text" in raw:
            return {**raw, "text": edits._resolve_pointer(raw["text"], state, where)}
        return raw

    out = step(edit, f"{edit.get('op')}'s text")
    for key in ("into", "new"):
        if isinstance(out.get(key), list):
            out = {**out, key: [step(raw, f"{edit.get('op')}'s {key}") for raw in out[key]]}
    if isinstance(out.get("to"), dict):
        out = {**out, "to": {label: step(raw, f"{edit.get('op')}'s route {label!r}")
                             for label, raw in out["to"].items()}}
    if isinstance(out.get("combine"), dict):
        out = {**out, "combine": step(out["combine"], f"{edit.get('op')}'s combine")}
    return out




def _outputs_read(state: Mapping[str, Any], node_id: str) -> dict[str, set[tuple[str, str]]]:
    """Which of ``node_id``'s outputs each other step takes in (`validate.consumed`: a
    ``{{slot}}`` in a prompt or a bound or fills-facts knob, an input, a read)."""
    from refactory_dsh.nn.validate import consumed

    graph = _edits()._graph(state)
    node = graph.nodes.get(node_id)
    if node is None:
        return {}
    outputs = {("fact", n) for n in node.produces} | {("file", n) for n in node.writes}
    values = {name: entry.get("value") for name, entry in (state.get("knobs") or {}).items()}
    out: dict[str, set[tuple[str, str]]] = {}
    for other in graph.nodes.values():
        if other.id != node_id and (taken := consumed(other, values) & outputs):
            out[other.id] = taken
    return out




def unread_new_steps(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """The steps ``after`` adds to ``before`` whose outputs no step takes in. A step at the
    end of the run or of a pass writing names of its own is read by nothing, since a Task
    reads only the names it knows. A step writing names a step of ``before`` wrote is not
    judged here: a fallback writes them instead of that step, and a take-over is refused
    where it is built when nothing after it reads them or the Task scores none of them
    (`_unscored_take_over`)."""
    old_graph = _edits()._graph(before)
    made = set().union(set(), *(_outputs(node) for node in old_graph.nodes.values()))
    nodes = (after.get("graph") or {}).get("nodes") or {}
    graph = _edits()._graph(after)
    return [node_id for node_id in sorted(set(nodes) - set(old_graph.nodes))
            if nodes[node_id].get("kind") in ("agent", "op")
            and (nodes[node_id].get("produces") or nodes[node_id].get("writes"))
            and not _outputs(graph.nodes[node_id]) & made
            and not _outputs_read(after, node_id)]




def unread_by_bundle(parent: Mapping[str, Any], states: Sequence[Mapping[str, Any]],
                     edits: Sequence[Mapping[str, Any]]) -> str | None:
    """Why a bundle's edits leave a step it adds with no reader: ``states[i]`` is the state
    after ``edits[i]``, and only the last state counts, so a reader a later edit of the
    bundle adds is one (`unread_new_steps`). When an edit of the bundle removed the last reader,
    the refusal names that edit. Otherwise only a core operator's step is refused, naming
    where its output could go: a node object an edit adds whole reads and writes what it
    says, and the candidate's As built says when nothing reads it."""
    if not states:
        return None
    for node_id in unread_new_steps(parent, states[-1]):
        made = next(i for i, state in enumerate(states)
                    if node_id in ((state.get("graph") or {}).get("nodes") or {}))
        edit = edits[made] if made < len(edits) else {}
        last = max((i for i, state in enumerate(states[:-1]) if _outputs_read(state, node_id)),
                   default=None)
        if last is None and edit.get("op") not in CORE_OPS:
            continue  # a node object's own reads and writes are the proposer's, as written
        if last is None:
            return (f"nothing reads the output of new step {node_id!r}, so it changes no "
                    f"result ({_reader_hint(states[-1], node_id, edit, parent)})")
        removed = last + 1
        edit = edits[removed]
        lost = sorted(set().union(*_outputs_read(states[last], node_id).values()))
        what = ", ".join("{{" + name + "}}" if kind == "fact" else name for kind, name in lost)
        readers = ", ".join(sorted(_outputs_read(states[last], node_id)))
        named = " ".join(str(edit.get(k)) for k in ("op", "name", "step", "id") if edit.get(k))
        how = " value" if edit.get("op") == "set_knob" and "value" in edit else ""
        return (f"edit {removed + 1} ({named}{how}) removes {what}, the only read of the "
                f"output of new step {node_id!r} (by {readers}); patch the text so it keeps "
                f"{what}, or keep the slot")
    return None




#: How many knobs a reader hint names.
HINT_KNOBS = 4




def _open_knob(state: Mapping[str, Any], name: str) -> bool:
    return not (((state.get("knobs") or {}).get(name) or {}).get("spec") or {}).get("frozen")




def reader_knobs(state: Mapping[str, Any], node_id: str, *, steps: bool = False) -> list[str]:
    """The prompt knobs after step ``node_id`` that 'to' can name for its output, nearest
    first: an agent's own prompt knob, or a text knob an op fills facts into; none frozen.
    With ``steps``, an agent whose prompt is no knob is named too, by its id ('to' naming
    it re-slots its prompt, `_reslotted`)."""
    graph = _edits()._graph(state)
    names: list[str] = []
    if not graph.nodes[node_id].produces:
        return names  # 'to' places a fact; a step making none has nothing to place
    for reader in _co._readers_from(graph, node_id, state):
        if reader == node_id:
            continue
        node = graph.nodes[reader]
        if node.kind == "agent":
            knob = _task_knob(node)
            found = [knob] if knob and len(graph.knob_consumers().get(knob, ())) == 1 else []
            if steps and knob is None and reader not in names:
                names.append(reader)
                continue
        else:
            found = _filled_text_knobs(node, state, graph)
        names += [k for k in found if k not in names and _open_knob(state, k)]
    return names




def _graded_step(graph: Graph, node_id: str, edit: Mapping[str, Any],
                 before: Mapping[str, Any]) -> str | None:
    """The step of ``before`` whose outputs a new step at ``node_id`` could take over to
    reach the result: the step a parallelize copies, else the nearest agent step before it
    (its edit's anchor first)."""
    old = set(((before.get("graph") or {}).get("nodes") or {}))
    if edit.get("op") == "parallelize" and edit.get("steps"):
        return str(list(edit["steps"])[-1])
    seen, queue = {node_id}, [node_id]
    while queue:
        current = queue.pop(0)
        node = graph.nodes.get(current)
        if node is not None and current in old and node.kind == "agent":
            return current
        for e in graph.in_edges(current):
            if e.source not in seen:
                seen.add(e.source)
                queue.append(e.source)
    return None




def _reader_hint(state: Mapping[str, Any], node_id: str, edit: Mapping[str, Any],
                 before: Mapping[str, Any]) -> str:
    """Where step ``node_id``'s output could go, for a refusal, in one clause: the prompt
    knobs after it (named in 'to' when ``edit``'s operator takes one); when the nearest
    agent after it takes no knob of its own, that agent in 'to' (its prompt is re-slotted)
    and the op before it that builds that agent's prompt, to insert before; and when no later step can read it at all, a step that takes
    over the one whose output is graded (``takes_over``; a route way's new step rewrites it
    in place on its way)."""
    op = edit.get("op")
    takes_to = op in ("insert", "branch", "map", "route") or (
        op == "parallelize" and bool(edit.get("new")))
    graph = _edits()._graph(state)
    readers = [r for r in _co._readers_from(graph, node_id, state) if r != node_id]
    names = ", ".join(reader_knobs(state, node_id)[:HINT_KNOBS])
    fact = (graph.nodes[node_id].produces or ("",))[0]
    parts = []
    if names:
        parts.append(f"name a later prompt in 'to': {names}" if takes_to else
                     f"put {{{{{fact}}}}} in a later prompt: {names}")
    first = next((graph.nodes[r] for r in readers if graph.nodes[r].kind == "agent"), None)
    if takes_to and first is not None and _task_knob(first) is None:
        # Its prompt is no knob: 'to' naming it re-slots it (`_reslotted`).
        parts.append(f"name {first.id} in 'to' (its prompt gets a section for it)")
        after = _downstream(graph, node_id)
        taken = _consumed(first, state)
        for builder in graph.nodes.values():
            if (builder.id in after or builder.kind != "op" or not (_outputs(builder) & taken)
                    or "entry" in _data_ends(graph, builder.id)):
                continue
            filled = [k for k in _filled_text_knobs(builder, state, graph)
                      if _open_knob(state, k)]
            if filled:
                parts.append(f"{first.id} takes no knob: insert before {builder.id} with "
                             f"'to': {filled[0]}")
                break
    if parts:
        return " or ".join(parts)
    later = _downstream(graph, node_id) - {node_id}
    if any(graph.nodes[n].kind == "agent" or _is_code(graph.nodes[n])
           for n in later if n in graph.nodes):
        return "rewire a later step to see it, in the same edits"
    graded = _graded_step(graph, node_id, edit, before)
    if graded is None:
        return "no step can read it"
    if op == "route":
        return f"give the way's new step 'takes_over': {graded}"
    if op == "insert" and "takes_over" not in edit:
        return f"give it 'takes_over': {graded}"
    return f"insert one seeing it with 'takes_over': {graded}"




def _needed(node: Node, values: Mapping[str, Any]) -> set[tuple[str, str]]:
    """What a step cannot run without. An agent's prompt is read here: its reads and the
    ``{{slots}}`` it needs filled, a slot inside ``{% if name %}`` being optional, as the
    validator holds it (V15, `validate.template_facts`). An op or a gate is code that
    cannot be read, so it needs everything it takes in (its reads and inputs, which it
    may need though it declares them optional); only the slots of the knobs it fills are
    read, guards and all."""
    from refactory_dsh.nn.validate import consumed, knob_slot_facts, template_facts

    if node.kind == "agent":
        return ({("file", name) for name in node.reads}
                | {("fact", name) for name in template_facts(node, values)})
    filled = knob_slot_facts(node, values, guarded=True)
    return ((consumed(node, values) - {("fact", name) for name in filled})
            | {("fact", name) for name in knob_slot_facts(node, values)})




def lost_inputs(before: Mapping[str, Any], after: Mapping[str, Any]) -> str | None:
    """Why an operator's result would fail items at run time: a step it kept now runs, on
    some path, without an input it had on every path before -- one still made elsewhere in
    the graph (a route that skips the steps making it, say) -- or ``None``. The validator
    holds a step's required inputs to every path (V15); this holds every input the step
    takes in, since an op may need what it declares optional (LiveMath's ``decide_prompt``
    needs a note from at least one of the steps a skip passes by). An input no step makes
    any more is the operator's own to settle (`remove` reconnects or refuses)."""
    from refactory_dsh.nn.operators import made_by
    from refactory_dsh.nn.validate import available_inputs

    edits = _edits()
    try:
        old, new = edits._graph(dict(before)), edits._graph(dict(after))
    except Exception:  # noqa: BLE001 - an unreadable graph is the validator's to report
        return None
    had, has = available_inputs(old), available_inputs(new)
    makers = made_by(new)
    values = _knob_values(after)
    for node_id in sorted(set(old.nodes) & set(new.nodes)):
        try:
            takes = _needed(new.nodes[node_id], values)
        except Exception as exc:  # noqa: BLE001 - said, never skipped
            return (f"{node_id!r}'s inputs cannot be read ({type(exc).__name__}: {exc}), so "
                    "whether it keeps them after this edit is unknown")
        # A {{name}} a knob fills is no input (`validate.consumed` drops the knobs).
        lost = sorted((had.get(node_id, set()) & takes) - has.get(node_id, set()))
        lost = [key for key in lost if makers.get(key)]
        if lost:
            names = ", ".join(f"{name} (made by {', '.join(makers[(kind, name)])})"
                              for kind, name in lost)
            return (f"{node_id!r} would run without {names} on a path this edit adds: no step "
                    "on that path makes it before it, so the step would fail or answer without "
                    "it on every item that takes the path. Route to a step that needs none of "
                    "what the skipped steps make, or give the path its own step making it")
    return None




#: What a new step written inside an operator ('new', 'into', a route's 'to') may say;
#: anything else it would ignore, so it is refused (`_unknown_fields`).
STEP_FIELDS = ("kind", "text", "system", "sees", "id", "to", "call", "reads", "tools", "route",
               "sealed", "produces", "writes")




def _tools_refusal(tools: Any) -> str | None:
    """Why ``tools`` is not a list of the tools a new step may use, or ``None``: DSH's
    (`SEALED_TOOLS`) or Claude Code's (`seal.CLAUDE_CODE_TOOLS`), never a mix, as a step
    ``add_node`` writes out may have (`workspace.is_sealed`). Whether this run can run the
    step on that backend is the attempt's to judge (``E-new-node-tools``)."""
    from refactory_dsh.seal import CLAUDE_CODE_TOOLS

    if (not isinstance(tools, list) or not tools or len(set(tools)) != len(tools)
            or not (set(tools) <= set(SEALED_TOOLS) or set(tools) <= set(CLAUDE_CODE_TOOLS))):
        return (f"'tools' lists the tools a new step uses, from {', '.join(SEALED_TOOLS)} "
                f"(or Claude Code's {', '.join(CLAUDE_CODE_TOOLS)}); {tools!r} is not such a "
                "list")
    return None




def _unknown_fields(edit: Mapping[str, Any]) -> str | None:
    """Why ``edit`` names a field its operator would ignore -- a role, a timeout, a tool
    list -- or ``None``. An ignored field would let the proposer believe it set what it
    did not, so it is refused, naming the fields the operator takes."""
    name = str(edit.get("op"))
    fields = OPERATORS[name]["fields"] if name in OPERATORS else ()
    unknown = sorted(k for k in edit if k != "op" and k not in fields and not k.startswith("_"))
    if unknown:
        return (f"'{name}' does not take {', '.join(repr(k) for k in unknown)}; it would be "
                f"ignored. Its fields: {', '.join(f for f in fields if f not in ('target', 'gate'))}")
    specs: list[Any] = []
    for key in ("new", "into"):
        # The written-out split's 'into' is two whole node objects, not step specs.
        if key == "into" and name == "split" and not is_one_line_split(edit):
            continue
        if isinstance(edit.get(key), list):
            specs += edit[key]
    if isinstance(edit.get("to"), Mapping):
        specs += [v for v in edit["to"].values() if isinstance(v, Mapping)]
    if isinstance(edit.get("combine"), Mapping):
        specs.append(edit["combine"])
    for spec in [edit, *specs]:
        if not isinstance(spec, Mapping):
            continue
        if spec is not edit:
            allowed = (*STEP_FIELDS, "takes_over") if name == "route" else STEP_FIELDS
            extra = sorted(k for k in spec if k not in allowed and not k.startswith("_"))
            if extra:
                return (f"a new step does not take {', '.join(repr(k) for k in extra)}; it "
                        f"would be ignored. A step's fields: {', '.join(STEP_FIELDS)}")
        if spec.get("reads") is not None and spec.get("kind") != "op":
            # An agent or code step is given files by 'sees'; its own 'reads' are made from
            # what it sees (`_step_nodes`), so a list here would be ignored.
            return ("'reads' lists an op step's files (with 'call'); an agent or code step "
                    "gets a file by naming it in 'sees', so 'reads' would be ignored")
        if "tools" in spec and spec["tools"] is not False:
            refused = _tools_refusal(spec["tools"])
            if refused:
                return refused
    return None




def _pointed(pre, expand):
    """A precondition and an expansion that read the edit with its pointer texts resolved
    (`_resolved_texts`): a pointer is never stored as a step's prompt. A field the
    operator would ignore is refused first (`_unknown_fields`)."""
    def precondition(edit: dict, state: dict) -> str | None:
        unknown = _unknown_fields(edit) or _map_part(edit, state)
        if unknown:
            return unknown
        try:
            edit = _resolved_texts(edit, state)
        except _edits().EditError as exc:
            return str(exc)
        return pre(edit, state)

    def run(edit: dict, state: dict) -> list[dict]:
        return expand(_resolved_texts(edit, state), state)
    return precondition, run




def register(macro, macros: Mapping[str, Any]) -> None:
    """Register the operators in the edit catalogue (``nn.edits``)."""
    simple = {
        "insert": (_pre_insert, _expand_insert),
        "remove": (_pre_remove, _expand_remove),
        "parallelize": (_pre_parallelize, _expand_parallelize),
        "serialize": (_pre_serialize, _expand_serialize),
        "branch": (_pre_branch, _expand_branch),
        "rewire": (_pre_rewire, _expand_rewire),
        "route": (_pre_route, _expand_route),
        "merge": (_pre_merge, _expand_merge),
        "map": (_pre_map, _expand_map),
        "fallback": (_pre_fallback, _expand_fallback),
    }
    for name, (pre, expand) in simple.items():
        spec = OPERATORS[name]
        macro(name, spec["line"] + ".", spec["fields"],
              *_pointed(_guarded(pre), _guarded_expand(expand)), reads_as=spec["reads_as"])
    old_loop = macros["loop"]
    old_split = macros["split"]

    def pre_loop(edit: dict, state: dict) -> str | None:
        if not is_one_line_loop(edit):
            if "back_to" in edit:
                # Written out, the gate's reloop goes to 'target'; a 'back_to' there would
                # be ignored, so it is refused rather than dropped.
                return ("'back_to' goes with the one-line loop ('step' and 'judge'); "
                        "written out, the gate sends back to 'target'")
            if "max_iterations" in edit:
                # The written-out form's cap is its gate's own field; a top-level one
                # would be ignored there, so it is refused rather than dropped.
                return ("'max_iterations' goes with the one-line loop ('step' and 'judge'); "
                        "written out, set the 'gate' node's own max_iterations")
            return old_loop.precondition(edit, state)
        return _guarded(_pre_loop_line)(edit, state)

    def expand_loop(edit: dict, state: dict) -> list[dict]:
        if not is_one_line_loop(edit):
            steps = old_loop.expand(edit, state)
            return steps + _settle_optional(state, steps)
        return _guarded_expand(_expand_loop_line)(edit, state)

    def pre_split(edit: dict, state: dict) -> str | None:
        if not is_one_line_split(edit):
            return old_split.precondition(edit, state)
        return _guarded(_pre_split_line)(edit, state)

    def expand_split(edit: dict, state: dict) -> list[dict]:
        if not is_one_line_split(edit):
            steps = old_split.expand(edit, state)
            return steps + _settle_optional(state, steps)
        return _guarded_expand(_expand_split_line)(edit, state)

    # Each core operator says only its one-line form, as the lead promises; a proposal
    # written out the older way ('target' and a 'gate' or 'into' node objects) is still
    # taken, but not described.
    macro("loop", OPERATORS["loop"]["line"] + ".", OPERATORS["loop"]["fields"],
          *_pointed(pre_loop, expand_loop), reads_as=OPERATORS["loop"]["reads_as"])
    macro("split", OPERATORS["split"]["line"] + ".", OPERATORS["split"]["fields"],
          *_pointed(pre_split, expand_split), reads_as=OPERATORS["split"]["reads_as"])






def wiring_notes(edit: Mapping[str, Any], before: Mapping[str, Any],
                 after: Mapping[str, Any], final: Mapping[str, Any] | None = None) -> list[str]:
    """What a core operator did beyond the steps it names: the steps it added and removed,
    the prompts of existing steps it put a new step's output into, and the item's input
    files a tool step it added or changed works on (`_with_item_files`), so the candidate
    says where an output went. ``after`` is the state the edit left; ``final`` (default
    ``after``) the candidate's, after every edit of its proposal: what is true of the
    candidate -- a new step's output that nothing reads, a knob that reaches no step --
    is judged there, so a later edit that reads the output is not contradicted."""
    if not isinstance(edit, Mapping) or edit.get("op") not in CORE_OPS:
        return []
    if edit.get("op") == "loop" and not is_one_line_loop(edit):
        return []
    if edit.get("op") == "split" and not is_one_line_split(edit):
        return []
    old = (before.get("graph") or {}).get("nodes") or {}
    new = (after.get("graph") or {}).get("nodes") or {}
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    old_knobs, new_knobs = before.get("knobs") or {}, after.get("knobs") or {}
    placed = []
    for name in sorted(set(old_knobs) & set(new_knobs)):
        was, now = (old_knobs[name] or {}).get("value"), (new_knobs[name] or {}).get("value")
        if was != now and isinstance(now, str):
            slots = sorted(set(re.findall(r"\{\{\s*([A-Za-z_]\w*)\s*\}\}", now))
                           - set(re.findall(r"\{\{\s*([A-Za-z_]\w*)\s*\}\}", str(was or ""))))
            if slots:
                placed.append(f"{', '.join('{{' + s + '}}' for s in slots)} into knob {name!r}")
    final = after if final is None else final
    unread: list[str] = []
    try:
        graph = _edits()._graph(final)
        taken_in = set().union(*(_consumed(n, final) for n in graph.nodes.values()))
        for node_id in added:
            node = graph.nodes.get(node_id)
            if node is None:
                continue  # a later edit of the proposal took it out again
            if node.kind in ("agent", "op") and node.produces and not any(
                    ("fact", f) in taken_in for f in node.produces) and not any(
                    ("file", f) in taken_in for f in node.writes):
                unread.append(node_id)
    except Exception:  # noqa: BLE001 - a note, never a refusal
        unread = []
    taken, toolless = _taken_away(before, after, added, removed)
    # A removed step's own settings (`<id>_...`, `<id>__...`) always go with it; the
    # others are news.
    dropped = sorted(k for k in set(old_knobs) - set(new_knobs)
                     if not any(k.startswith(f"{node_id}_") for node_id in removed))
    parts = []
    if added:
        parts.append("added " + ", ".join(added))
    if removed:
        parts.append("removed " + ", ".join(removed))
    if placed:
        parts.append("put " + "; ".join(placed))
    if unread:
        parts.append(f"no step reads what {', '.join(unread)} make(s) yet (name a reader with "
                     "'to', or insert one)")
    if taken:
        parts.append("took " + "; ".join(f"{', '.join(names)} away from {node_id}"
                                         for node_id, names in taken))
    if toolless:
        parts.append(f"the tools of {', '.join(toolless)} went with it: no new step uses tools")
    if dropped:
        parts.append(f"dropped knob(s) {', '.join(dropped)}, which no step reads now")
    try:
        # Said by the edit that made it so, and only when it still is in the candidate.
        was_dead = dead_text_knobs(before)
        still = dead_text_knobs(final) if final is not after else None
        parts += [f"knob {name!r} now reaches no step ({op} fills it into "
                  f"{', '.join(outputs)}, which no step reads)"
                  for name, (op, outputs) in sorted(dead_text_knobs(after).items())
                  if name not in was_dead and (still is None or name in still)]
    except Exception:  # noqa: BLE001 - a note, never a refusal
        pass
    return [f"{edit.get('op')}: " + "; ".join(parts)] if parts else []




def item_input_notes(parent: Mapping[str, Any], final: Mapping[str, Any]) -> list[str]:
    """What the candidate says of each sealed tool step (`workspace.is_sealed`) it adds or
    changes inside a per-item pass whose Task lays item files: whether its sealed workspace
    holds the item's input files (given by default when the proposer named none of its
    inputs, `_with_item_files`), or which of them it does not see. Facts, never refusals."""
    from refactory_dsh.workspace import is_sealed

    try:
        old, new = _edits()._graph(parent), _edits()._graph(final)
    except Exception:  # noqa: BLE001 - a note, never a refusal
        return []
    notes = []
    for node_id in sorted(new.nodes):
        node = new.nodes[node_id]
        if not is_sealed(node) or (node_id in old.nodes
                                   and old.nodes[node_id].to_state() == node.to_state()):
            continue
        files = item_files_before(new, final, node_id)
        has = [path for path in files if path in node.reads]
        missing = [path for path in files if path not in node.reads]
        if has:
            notes.append(f"{node_id} (tools, in the per-item part) has the item's input "
                         f"file(s) {', '.join(has)} in its sealed workspace")
        if missing:
            notes.append(f"{node_id} (tools, in the per-item part) does not see the item's "
                         f"input file(s) {', '.join(missing)}")
    return notes




#: Where a written-out edit carries whole node objects (`nn.edits`): the field, and whether
#: it holds a list of them.
_NODE_FIELDS = (("node", False), ("into", True), ("branches", True))




def default_item_inputs(edit: Mapping[str, Any], before: Mapping[str, Any],
                        after: Mapping[str, Any]) -> dict[str, Any] | None:
    """``edit`` with the item's input files as the ``reads`` of each node object it adds
    whole (``add_node``, the written-out macros) that runs its tools sealed inside a
    per-item pass and names no ``reads`` of its own -- the default a core operator's new
    tool step gets (`_with_item_files`) -- judged on ``after``, the state ``edit`` left;
    ``None`` when it adds none such. The edit as returned is the one recorded and replayed.
    A node object that lists ``reads`` (even none) keeps exactly those."""
    from refactory_dsh.workspace import is_sealed

    if not isinstance(edit, Mapping) or (edit.get("op") in CORE_OPS and not (
            edit.get("op") == "split" and not is_one_line_split(edit))):
        return None
    try:
        new = _edits()._graph(after)
    except Exception:  # noqa: BLE001 - the validator's to say
        return None
    out = json.loads(json.dumps(edit))
    changed = False
    for key, many in _NODE_FIELDS:
        value = out.get(key)
        for node_state in (value if many and isinstance(value, list) else [value]):
            if not isinstance(node_state, dict) or "reads" in node_state:
                continue
            node = new.nodes.get(str(node_state.get("id")))
            if node is None or not is_sealed(node):
                continue
            files = item_files_before(new, after, node.id)
            if files:
                node_state["reads"] = files
                changed = True
    return out if changed else None




def _taken_away(before: Mapping[str, Any], after: Mapping[str, Any], added: Sequence[str],
                removed: Sequence[str]) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """What an edit took from the steps it kept: for each, the files and facts it read or
    put in its prompt before and no longer gets (itself, or through a step the edit added
    in front of it, such as a collect step); and the removed steps that used tools when no
    added step does."""
    try:
        old, new = _edits()._graph(before), _edits()._graph(after)
    except Exception:  # noqa: BLE001 - a note, never a refusal
        return [], []
    lost_by: dict[str, set[str]] = {}
    for node_id in sorted(set(old.nodes) & set(new.nodes)):
        now = set(_consumed(new.nodes[node_id], after))
        for other in added:
            made = _outputs(new.nodes[other])
            if made & now:
                now |= set(_consumed(new.nodes[other], after))
        lost = {name for _kind, name in set(_consumed(old.nodes[node_id], before)) - now}
        # A step's collect step (`<id>_inputs`) is part of it: its losses are the step's.
        owner = node_id.removesuffix("_inputs")
        owner = owner if node_id.endswith("_inputs") and owner in new.nodes else node_id
        if lost:
            lost_by.setdefault(owner, set()).update(lost)
    taken = [(node_id, sorted(names)) for node_id, names in sorted(lost_by.items())]
    toolless = [] if any(_uses_tools(new.nodes[n]) for n in added) else sorted(
        n for n in removed if _uses_tools(old.nodes[n]))
    return taken, toolless




def parallel_join_id(graph: Graph, step: str) -> str:
    """The id ``parallelize`` gives the join after ``step`` (or after the steps starting at
    it), so a composition can name it in its next edit."""
    return _fork_join_ids(graph, step)[1]




def select_call() -> str:
    """The callable of the ``select`` op (`operators.select`)."""
    from refactory_dsh.nn.operators import SELECT_CALL

    return SELECT_CALL




def example(name: str, ids: Mapping[str, str] | None = None) -> dict[str, Any]:
    """An operator's example with its placeholders filled from ``ids`` (``step``,
    ``next step``, ``fork``, ``gate``, ``label``, ``other label``) where given."""
    text = json.dumps(OPERATORS[name]["example"])
    for key, value in (ids or {}).items():
        text = text.replace(f"<{key}>", value)
    return json.loads(text)




def _without_text(value: Any) -> Any:
    """``value`` with every ``text`` field taken out, so each step takes its operator's
    default text (a neutral line, or the neighbour's)."""
    if isinstance(value, dict):
        return {key: _without_text(item) for key, item in value.items() if key != "text"}
    if isinstance(value, list):
        return [_without_text(item) for item in value]
    return value




def sample(name: str, state: Mapping[str, Any], rng: Any) -> dict[str, Any]:
    """One instance of operator ``name`` on ``state`` with no free text, for a strategy
    with no model proposer (`edits.random_edit`): the operator's example, its step,
    next step and fork drawn from the graph (a ``rewire`` sometimes keeping one step its
    step reads from beside ``"task"``; a ``loop`` half the time sending back to an agent
    before its step), and its texts left to the defaults. Whether it applies there is the
    caller's to check (`edits.precheck`)."""
    graph = _edits()._graph(state)
    ids = sorted(graph.nodes)
    step = rng.choice(ids) if ids else "<step>"
    after = [e.target for e in graph.out_edges(step) if e.condition is None] if ids else []
    forks = sorted(n.id for n in graph.nodes.values() if n.kind == "fork")
    filled = {"step": step, "next step": rng.choice(after) if after else step,
              "fork": rng.choice(forks) if forks else step}
    edit = _without_text(example(name, filled))
    if name == "rewire" and step in graph.nodes:
        # "task" alone takes away a prompt built from another step's output; half the
        # time keep one step the step reads from beside it.
        from refactory_dsh.nn.operators import made_by

        makers = made_by(graph)
        reads_from = sorted({m for key in _consumed(graph.nodes[step], state)
                             for m in makers.get(key, ()) if m != step})
        if reads_from and rng.random() < 0.5:
            edit["sees"] = [TASK, rng.choice(reads_from)]
    if name == "loop" and step in graph.nodes:
        earlier = sorted(n for n in _upstream(graph, step) if graph.nodes[n].kind == "agent")
        if earlier and rng.random() < 0.5:
            edit["back_to"] = rng.choice(earlier)
    return edit
