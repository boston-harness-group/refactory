from __future__ import annotations

import re
from typing import Any, Mapping

from refactory_dsh.nn.graph import Graph, Node

from ._common import MAP_ELEMENT_PROMPT, MAP_ELEMENT_TOOLS_PROMPT, MAP_MAX, NEUTRAL_COMBINE, NEUTRAL_ELEMENT, _Refused, _default_wired, _edits
from .state import _available, _consumed, _fresh, _outputs, _sole_out
from .visibility import _heading, _readable, _seen, _strip_unseen, _with_sections
from .build import _Steps, _add_chain, _add_detached, _drop_orphans, _respec, _simulate, _spec, _step_nodes
from .placement import _new_facts, _place_all
from .checks import _losses, _owned, _plain_step, _strip_steps
from .insert import _neighbour_view


def _map_list(graph: Graph, state: Mapping[str, Any], anchor: Node, over: Any
              ) -> tuple[str, str]:
    """``(kind, name)`` of the list a map runs over: a fact or file there after ``anchor``,
    or a step's output (its first fact, else its first file); refused otherwise."""
    there = _readable(graph, state, _available(graph).get(anchor.id, set()) | _outputs(anchor))
    if isinstance(over, str) and over in graph.nodes and over not in {n for _k, n in there}:
        node = graph.nodes[over]
        made = [("fact", n) for n in node.produces] + [("file", n) for n in node.writes]
        over = next((name for key, name in ((k, k[1]) for k in made) if key in there), over)
    for kind in ("fact", "file"):
        if (kind, over) in there:
            return kind, str(over)
    names = sorted({name for _k, name in there})
    raise _Refused(f"'over' names a list there after {anchor.id!r}: a fact, a file or a step "
                   f"whose output it is, not {over!r}; there: {', '.join(names) or 'none'}")


def _map_max(edit: dict) -> int:
    default, lo, hi = MAP_MAX
    cap = edit.get("max", default)
    if isinstance(cap, bool) or not isinstance(cap, int) or not lo <= cap <= hi:
        raise _Refused(f"'max' is a whole number from {lo} to {hi} (the most elements, one "
                       "run each)")
    return cap


def _combine_spec(edit: dict) -> dict:
    return _spec(edit.get("combine"), default_text=NEUTRAL_COMBINE)


def _pre_map(edit: dict, state: dict) -> str | None:
    graph = _edits()._graph(state)
    anchor = _plain_step(graph, edit.get("after"), "map")
    _sole_out(graph, anchor.id)
    _map_list(graph, state, anchor, edit.get("over"))
    _map_max(edit)
    if edit.get("text") is not None and not isinstance(edit.get("text"), str):
        raise _Refused("'text' is a string: what to do with one element")
    combine = _combine_spec(edit)
    if combine.get("to") is not None and edit.get("to") is not None \
            and combine["to"] != edit["to"]:
        raise _Refused("the map and its combine each name a 'to'; name it once: the combine "
                       "writes the map's one output")
    # Refused when it is no route, or a 'system' that is no text.
    _spec({"kind": "agent", "route": edit.get("route"), "system": edit.get("system")})
    base = edit.get("id")
    if base is not None and not isinstance(base, str):
        raise _Refused("'id' is a string")
    for name in (base, combine.get("id")):
        if name is not None and name in graph.nodes:
            raise _Refused(f"node {name!r} already exists")
    return None


def _expand_map(edit: dict, state: dict) -> list[dict]:
    """After ``after``: an ``elements`` step (`operators.MAP_CALL`) splitting the list into
    one prompt per element (its text knob ``<id>__task``, filled with what the new step
    sees, then the element), whose fact says how many ran; a fork with one optional branch
    per element up to 'max', each an agent reading its element (a branch with no element is
    refused before any model call and recorded as not needed); a join; then the combine
    step, which sees how many ran, every branch's result (one not written is said so) and
    what the element step sees, its output placed as insert's is."""
    from refactory_dsh.nn.knob import Knob
    from refactory_dsh.nn.operators import MAP_CALL, collect_node

    build = _Steps(state)
    graph = build.graph
    anchor = graph.nodes[edit["after"]]
    kind, over = _map_list(graph, state, anchor, edit.get("over"))
    cap = _map_max(edit)
    base = edit.get("id") or _fresh(graph, f"{anchor.id}_map")
    fork, join = _fresh(graph, f"{base}_fork", (base,)), _fresh(graph, f"{base}_join", (base,))
    taken = [base, fork, join]
    there = _available(graph).get(anchor.id, set()) | _outputs(anchor)
    seen = [key for key in _seen(graph, state, there, _neighbour_view(graph, state, anchor),
                                 edit.get("sees")) if key != (kind, over)]
    facts = [name for k, name in seen if k == "fact"]
    files = [name for k, name in seen if k == "file"]
    nodes: list[dict[str, Any]] = []
    if files:
        gather = _fresh(graph, f"{base}_inputs", taken)
        taken.append(gather)
        nodes.append(collect_node(gather, files, gather,
                                  notes=f"Hands {base}'s elements the files they see."))
        facts.append(gather)
    text = _with_sections(_strip_unseen(edit.get("text") or NEUTRAL_ELEMENT,
                                        set(facts) | set(state.get("knobs") or {})), facts)
    task = f"{base}__task"
    copies = [_fresh(graph, f"{base}_{index}", taken) for index in range(1, cap + 1)]
    taken += copies
    nodes.append({"id": base, "kind": "op", "call": MAP_CALL, "knobs": [task],
                  "reads": [over] if kind == "file" else [],
                  "inputs": [*([over] if kind == "fact" else []), *facts],
                  "writes": [f"{c}_element.txt" for c in copies],
                  # How many elements ran, for the combine step: "ran K of N (max M)".
                  "produces": [base],
                  # Every element's prompt carries the one text; the count none.
                  "knob_outputs": {**{f"{c}_element.txt": [task] for c in copies}, base: []},
                  "notes": f"Splits {over!r} into one prompt per element for {base}'s runs."})
    route = _spec({"kind": "agent", "route": edit.get("route")}).get("route") or (
        anchor.route if anchor.kind == "agent" else None)
    tools = edit.get("tools") or None
    # Every run's system prompt: the edit's 'system', as insert's, else the default.
    run_system = edit.get("system") or "You are one step of a pipeline."
    branch = []
    systems = []
    for copy in copies:
        # Each run has a system prompt of its own, as every new agent does; its prompt is
        # its element's file, which a step without tools is shown (`agent_delivery.begin`),
        # so a branch with no element does not run.
        system = f"{copy}__system_prompt"
        systems.append({"op": "add_knob", "name": system, "spec": Knob(
            run_system, "str",
            description=f"system prompt for {copy}").spec()})
        element = f"{copy}_element.txt"
        node: dict[str, Any] = {
            "id": copy, "kind": "agent", "prompt_template": MAP_ELEMENT_PROMPT,
            "system_prompt_template": run_system,
            "knobs": [system], "field_bindings": [["system_prompt_template", system]],
            "reads": [element], "writes": [f"{copy}.txt"], "tool_policy": "none"}
        if tools:
            # Sealed tools, as insert's (`workspace.SEALED_TOOLS`): a fresh workspace
            # holding only its element's file, which it reads with them.
            node.update({"tool_policy": None, "tools": list(tools),
                         "workspace": {"kind": "scratch"},
                         "prompt_template": MAP_ELEMENT_TOOLS_PROMPT.format(file=element)})
            if edit.get("sees") is None:
                _default_wired(copy)
        if anchor.kind == "agent" and anchor.role:
            node["role"] = anchor.role
        if anchor.kind == "agent" and anchor.timeout_s:
            # Its timeout too, as any new agent takes its neighbour's (`_step_nodes`).
            node["timeout_s"] = anchor.timeout_s
        if route:
            node["route"] = dict(route)
        branch.append(node)
    combine = _combine_spec(edit)
    combine_id = combine.get("id") or _fresh(graph, f"{base}_combine", taken)
    taken.append(combine_id)
    results = [f"{c}.txt" for c in copies]
    if combine["kind"] == "agent" and combine.get("tools"):
        # A sealed combine declares every file it reads, and a short list leaves some runs'
        # results unwritten (V6): they reach it as one fact from a collect step in front
        # (each file optional, as a combine without tools is handed them), and its sealed
        # workspace holds the other files it sees.
        gather = _fresh(graph, f"{combine_id}_inputs", taken)
        taken.append(gather)
        combined = [collect_node(gather, results, gather,
                                 notes=f"Hands {combine_id} the map's results."),
                    *_step_nodes(graph, state, combine, combine_id,
                                 [("fact", base), ("fact", gather), *seen],
                                 settings_from=anchor if anchor.kind == "agent" else None)]
    else:
        combined = _step_nodes(graph, state, combine, combine_id,
                               [("fact", base), *seen, *(("file", f) for f in results)],
                               settings_from=anchor if anchor.kind == "agent" else None)
    build.add(systems)
    build.add([{"op": "add_knob", "name": task,
                "spec": Knob(text, "str", description=f"what {base} does with one element").spec()},
               {"op": "add_node", "node": {"id": join, "kind": "join", "sources": copies},
                "position": {"after": anchor.id}},
               {"op": "add_node", "node": {"id": fork, "kind": "fork", "targets": copies,
                                           "optional_targets": copies},
                "position": {"after": anchor.id}}])
    build.add(_add_chain(nodes, {"after": anchor.id}))
    for node in branch:
        build.add(_add_detached([node], exact=copies))
    build.add([{"op": "add_edge", "source": copy, "target": join, "condition": None}
               for copy in copies])
    build.add(_add_chain(combined, {"after": join}))
    _place_all(build, _new_facts(combined), after=combine_id,
               to=edit.get("to") if edit.get("to") is not None else combine.get("to"))
    return build.steps


#: The fields a core operator names an existing node in.
NAMING_FIELDS = ("step", "after", "before", "fork", "back_to", "takes_over", "steps", "over",
                 "target")


def _maps(graph: Graph, state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Each map, by its elements step (`operators.MAP_CALL`), its parts found by what they
    are, never by where they stand: the fork it leads to (`operators.map_of`), the runs,
    their join; the combine, the one step that takes in the elements step's count fact and
    every run's result; and
    the collect steps that exist only for the elements step and for the combine
    (`_owned`). ``inner`` are the parts no edit names alone, ``combine`` the combine,
    ``nodes`` every node the map added, ``first`` the node its way in leads to."""
    from refactory_dsh.nn.operators import map_of

    out: dict[str, dict[str, Any]] = {}
    for fork_id in sorted(graph.nodes):
        op = map_of(graph, fork_id)
        join = graph.fork_join(fork_id) if op is not None else None
        if op is None or join is None:
            continue
        count = {("fact", name) for name in graph.nodes[op].produces}
        results = {("file", w) for run in graph.nodes[fork_id].targets
                   for w in graph.nodes[run].writes}
        # The combine takes in the count and every run's result, itself or through the
        # collect step in front of it; a step copied from it has no such collect step.
        found = []
        for taker in sorted(n.id for n in graph.nodes.values() if n.id != op
                            and n.kind in ("agent", "op") and _consumed(n, state) & count):
            helper = f"{taker}_inputs"
            helpers = [helper] if helper in graph.nodes and [
                e.target for e in graph.out_edges(helper)] == [taker] else []
            seen = set(_consumed(graph.nodes[taker], state)).union(
                *(_consumed(graph.nodes[h], state) for h in helpers))
            if results <= seen:
                found.append((taker, helpers))
        combine, gather = found[0] if len(found) == 1 else (None, [])
        before = _owned(graph, op, state)
        inner = [*before, fork_id, *graph.nodes[fork_id].targets, join, *gather]
        out[op] = {"inner": inner, "combine": combine, "first": (before or [op])[0],
                   "nodes": [*before, op, *inner[len(before):], *([combine] if combine else [])]}
    return out


#: What a core operator may do with a map's combine step: everything after it. Anything
#: else (a step before it, a split, a merge, a copy beside it, new inputs) would come
#: between its collect step and it.
COMBINE_OPS = {"remove": "step", "loop": "step", "route": "after", "insert": "after",
               "map": "after", "fallback": "step"}


def _map_part(edit: Mapping[str, Any], state: Mapping[str, Any]) -> str | None:
    """Why ``edit`` names a part of a map (its elements step, fork, runs, join or the
    collect steps around them) -- or None. Its runs are one step run per element; an edit
    to one alone (a step before it, a serialized fork, a branch beside them) would leave
    the others different or run one with no element, so a map is edited through its text
    knob and its combine step, and ``remove`` of its elements step or its combine takes the
    whole map out (`_remove_map`)."""
    try:
        graph = _edits()._graph(dict(state))
    except Exception:  # noqa: BLE001 - an unreadable state is the operator's to refuse
        return None
    maps = _maps(graph, state)
    parts = {part: op for op, found in maps.items() for part in (op, *found["inner"])}
    named = []
    for key in NAMING_FIELDS:
        value = edit.get(key)
        named += [v for v in (value if isinstance(value, list) else [value])
                  if isinstance(v, str)]
    if edit.get("op") == "remove" and edit.get("step") in maps:
        return None
    hit = next((name for name in named if name in parts), None)
    combines = {found["combine"]: op for op, found in maps.items() if found["combine"]}
    allowed = COMBINE_OPS.get(str(edit.get("op")))
    for key in NAMING_FIELDS:
        value = edit.get(key)
        for name in (value if isinstance(value, list) else [value]):
            if hit is None and name in combines and key != allowed:
                return (f"{name!r} is the combine of the map {combines[name]!r}; a step goes "
                        "after it (insert after, loop, route), or remove it to take the map "
                        "out; change what it does with its own knobs")
    if hit is None:
        return None
    return (f"{hit!r} is part of the map {parts[hit]!r}, whose runs each take one element; "
            f"change what they do with its knob {parts[hit]}__task, work on the step "
            "before the map or its combine step, or remove the map by its elements step")


def _map_removed(graph: Graph, state: Mapping[str, Any], step: str) -> dict[str, Any] | None:
    """The map ``remove`` of ``step`` takes out whole: ``step`` is its elements step or its
    combine; None otherwise."""
    for op, found in _maps(graph, state).items():
        if step in (op, found["combine"]):
            return {"op": op, **found}
    return None


def _unplaced(state: Mapping[str, Any], found: dict[str, Any]) -> list[dict]:
    """Steps that take the combine's output out of the knobs `_place` put it in (maybe a
    knob an op fills, as 'to' named it) -- those that still hold the heading it gave it --
    a frozen one never: wherever they now stand, its ``{% if %}`` guard, that heading and
    the slot go. A slot elsewhere is a step reading it, which the removal refuses."""
    from refactory_dsh.nn.validate import template_names

    graph = _edits()._graph(dict(state))
    facts = set(graph.nodes[found["combine"]].produces) if found["combine"] else set()
    steps = []
    for name, entry in sorted((state.get("knobs") or {}).items()):
        value = (entry or {}).get("value")
        if not isinstance(value, str) or ((entry or {}).get("spec") or {}).get("frozen") \
                or not template_names(value) & facts:
            continue
        placed = [fact for fact in sorted(facts) if re.search(
            r"(?m)^[ \t]*#{1,6}[ \t]*" + re.escape(_heading(fact)) + r"[ \t]*$", value)]
        if not placed:
            continue  # a slot a step was given by other means: its reader is said, below
        new = value
        for fact in placed:
            slot = r"\{\{\s*" + re.escape(fact) + r"\s*\}\}"
            new = re.sub(r"\{%-?\s*if\s+" + re.escape(fact) + r"\s*-?%\}(.*?)\{%-?\s*endif\s*-?%\}",
                         lambda m: m.group(1), new, flags=re.DOTALL)
            new = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]*" + re.escape(_heading(fact)) + r"[ \t]*\n?",
                         "", new)
            new = re.sub(r"(?m)^[ \t]*" + slot + r"[ \t]*\n?", "", new)
            new = re.sub(r"[ \t]*" + slot, "", new)
        new = re.sub(r"\n{3,}", "\n\n", new).strip()
        if new != value:
            steps.append({"op": "set_knob", "name": name, "value": new})
    return steps


def _pre_remove_map(graph: Graph, state: Mapping[str, Any], found: dict[str, Any]) -> None:
    combine = found["combine"]
    if combine is None or len(graph.out_edges(combine)) > 1:
        raise _Refused(f"map {found['op']!r} has no one combine step to remove it by")
    gone = set(found["nodes"])
    outside = sorted({e.source for n in gone for e in graph.in_edges(n)} - gone)
    if len(outside) != 1:
        raise _Refused(f"map {found['op']!r} is entered from {outside} (a loop sends work back "
                       "into it?); remove that loop's gate first (remove it with 'keep')")
    try:
        unplaced = _simulate(state, _unplaced(state, found))
    except _edits().EditError:
        unplaced = dict(state)  # a slot that cannot come out is a step left without it
    hard, _strip, _drop = _losses(_edits()._graph(unplaced), unplaced, gone)
    if hard:
        raise _Refused(f"removing map {found['op']!r} leaves steps without an input only it "
                       f"makes: {'; '.join(hard)}; remove or rewire those first")


def _remove_map(state: Mapping[str, Any], found: dict[str, Any]) -> list[dict]:
    """Primitive steps that take a map out whole and lead the step before it to the step
    after its combine, as before the map: its combine's output is taken out of the prompts
    it was put in, and its knobs go with it."""
    build = _Steps(state)
    graph = build.graph
    gone = set(found["nodes"])
    way_in = next(e for n in gone for e in graph.in_edges(n) if e.source not in gone)
    anchor = way_in.source
    after = _sole_out(graph, found["combine"])
    build.add(_unplaced(state, found))
    _hard, strip, drop = _losses(build.graph, build.state, gone)
    build.add(_strip_steps(build.state, strip))
    for reader, names in sorted(drop.items()):
        changed = build.graph.nodes[reader].to_state()
        changed["reads"] = [r for r in changed.get("reads") or () if r not in names]
        build.add(_respec(build.state, reader, changed))
    for data in graph.nodes.values():
        if data.kind == "data" and data.subgraph_exit == found["combine"]:
            build.add(_respec(build.state, data.id, {**data.to_state(), "subgraph_exit": anchor}))
    build.add([{"op": "remove_node", "id": node_id} for node_id in found["nodes"]])
    if after is not None:
        # The way in keeps its verdict (a gate before the map leads on as it led in).
        build.add([{"op": "add_edge", "source": anchor, "target": after,
                    "condition": way_in.condition}])
    _drop_orphans(build)
    return build.steps
