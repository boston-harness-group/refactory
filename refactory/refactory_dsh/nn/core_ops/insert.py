from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from refactory_dsh.nn.graph import Graph, Node

from ._common import NEUTRAL_STEP, TOOLS_HINTS, _Refused, _edits
from .state import _available, _consumed, _downstream, _fresh, _outputs, _require_step
from .build import _Steps, _add_chain, _chain, _copy_spec, _output_renames, _spec, _step_nodes
from .placement import _new_facts, _place_all
from .loop import _forward_flow, _reach
from .fallback import _at_rejoin, _fallback_parts

import sys as _sys


class _Proxy:
    """Late-bound lookup on the core_ops package for monkeypatch compat."""
    __slots__ = ()
    def __getattr__(self, name):
        return getattr(_sys.modules['refactory_dsh.nn.core_ops'], name)


_co = _Proxy()


def _where(edit: dict, graph: Graph) -> tuple[str, Node]:
    """The side and the step an insert names. Before a fallback gate is after the step it
    answers for (`_fallback_parts`), so the step keeps its fallback."""
    from refactory_dsh.nn.graph import FALLBACK_GATE_CALL

    given = [key for key in ("after", "before") if edit.get(key) is not None]
    if len(given) != 1:
        raise _Refused("insert takes exactly one of 'after' or 'before', naming a step")
    side = given[0]
    anchor = _require_step(graph, edit[side])
    if side == "before" and anchor.kind == "gate" and anchor.evaluator == FALLBACK_GATE_CALL:
        before = graph.in_edges(anchor.id)
        if len(before) == 1 and graph.falls_back(before[0].source):
            return "after", graph.nodes[before[0].source]
    return side, anchor




def _neighbour_view(graph: Graph, state: Mapping[str, Any], node: Node) -> set[tuple[str, str]]:
    """What a step put after ``node`` sees by default: what ``node`` takes in and makes; for
    a join, what its branches take in and make."""
    if node.kind == "join":
        from refactory_dsh.nn.validate import fork_branches

        out: set[tuple[str, str]] = set()
        for fork in graph.nodes.values():
            if fork.kind == "fork" and graph.fork_join(fork.id) == node.id:
                for branch in fork_branches(graph, fork, node.id):
                    for member in branch:
                        out |= _consumed(graph.nodes[member], state) | _outputs(graph.nodes[member])
        return out
    return _consumed(node, state) | _outputs(node)




def _insert_default(anchor: Node, state: Mapping[str, Any]) -> dict:
    return _copy_spec(anchor, state) if anchor.kind == "agent" else \
        {"kind": "agent", "text": NEUTRAL_STEP}




def _insert_spec(edit: dict, anchor: Node, state: Mapping[str, Any]) -> dict:
    default = _insert_default(anchor, state)
    given = {k: v for k, v in edit.items() if k not in ("op", "after", "before")}
    copied = default if given.get("text") is None and given.get("kind") is None else {}
    return _spec({**copied, **given}, default_kind=default["kind"],
                 default_text=default["text"] if given.get("kind") in (None, default["kind"])
                 else NEUTRAL_STEP)




def _pre_insert(edit: dict, state: dict) -> str | None:
    graph = _edits()._graph(state)
    side, anchor = _where(edit, graph)
    if side == "after":
        if anchor.kind == "fork":
            raise _Refused(f"{anchor.id!r} is a fork; add a step beside its branches with "
                           "'branch', or insert after one of its branches")
        if anchor.kind == "gate":
            raise _Refused(f"{anchor.id!r} is a gate whose out-edges carry verdicts; insert "
                           "before the step a verdict leads to")
        _co._sole_out(graph, anchor.id)
    elif anchor.kind in ("join", "data"):
        raise _Refused(f"{anchor.id!r} is a {anchor.kind}; insert after it, or before the "
                       "step it leads to")
    spec = _insert_spec(edit, anchor, state)
    if spec.get("id") is not None and spec["id"] in graph.nodes:
        raise _Refused(f"node {spec['id']!r} already exists")
    if edit.get("takes_over") is not None:
        target_id, only = _parse_takes_over(edit["takes_over"])
        target = _co._take_over_plan(graph, state, target_id, side, anchor, only=only)[0]
        if only is not None:
            if spec["kind"] == "agent" and not spec.get("tools"):
                raise _Refused("a selective 'takes_over' ([step, output, ...]) adds files the new "
                               "step must write besides its own reply; an agent step without tools "
                               f"writes only its reply: use a code step{TOOLS_HINTS[0]}")
        elif spec["kind"] == "agent" and not spec.get("tools") and len(target.writes) > 1:
            raise _Refused(f"{target.id!r} writes {len(target.writes)} files, and a new agent "
                           "step without tools writes only its reply; take it over with a "
                           f"code step{TOOLS_HINTS[0]}")
    return None




def _successors(graph: Graph, node_id: str, back: set[tuple[str, str]]) -> list[str]:
    """The steps that run next after ``node_id``: its edges (no reloop back), a fork's
    branches and a data pass's first step."""
    node = graph.nodes.get(node_id)
    if node is None:
        return []
    out = [e.target for e in graph.out_edges(node_id) if (node_id, e.target) not in back]
    if node.kind == "fork":
        out += list(node.targets)
    if node.kind == "data" and node.subgraph_entry:
        out.append(node.subgraph_entry)
    return out




def _runs_first_on_every_path(graph: Graph, target_id: str, before: Sequence[str]) -> bool:
    """Whether ``target_id`` has run whenever a step placed after each of ``before`` runs:
    no step of ``before`` can be reached once ``target_id`` is taken out. A fork runs every
    branch, so a join is reached only when all its branches are; a route runs one, so a
    step is reached when any step before it is. A sibling fork branch, a branch a route
    skips and a step after the place have not run first; a branch of a fork whose join
    the place follows has."""
    from refactory_dsh.nn.loops import loop_edges

    if not before:
        return False
    back = {(source, target) for source, target, _c in loop_edges(graph)}
    following = {n: _successors(graph, n, back) for n in graph.nodes}
    entered = {t for targets in following.values() for t in targets}
    reached = {n for n in graph.nodes if n not in entered and n != target_id}
    changed = True
    while changed:
        changed = False
        for node_id in list(reached):
            for nxt in following[node_id]:
                if nxt == target_id or nxt in reached:
                    continue
                node = graph.nodes.get(nxt)
                if node is not None and node.kind == "join" and not all(
                        source in reached for source in node.sources):
                    continue
                reached.add(nxt)
                changed = True
    return all(step == target_id or step not in reached for step in before)




def _code_text(node: Node, state: Mapping[str, Any]) -> str | None:
    """A code step's code (its own, or the knob bound to it); None when the step's
    callable is not code this module can read."""
    from refactory_dsh.snippet import is_code_node

    if not is_code_node(node):
        return None
    for field_name, knob in node.field_bindings:
        if field_name == "code":
            return str(((state.get("knobs") or {}).get(knob) or {}).get("value") or "")
    return str(node.code or "")




def _marked(node: Node, mark: str) -> bool:
    """Whether an op or gate's callable carries ``mark`` (False when it does not resolve)."""
    from refactory_dsh.nn.resolve import resolve

    call = node.call if node.kind == "op" else node.evaluator if node.kind == "gate" else ""
    if not call:
        return False
    try:
        return bool(getattr(resolve(call), mark, False))
    except Exception:  # noqa: BLE001 - an unresolvable call is V8's to report
        return False




def _by_declaration(node: Node) -> bool:
    """Whether an op or gate's callable reads, writes and sets only what its node declares
    (marked `nn.by_declaration`), so its inputs and outputs can be renamed."""
    return _marked(node, "by_declaration")




def _roles(node: Node) -> tuple[str, ...] | None:
    """The files ``node``'s callable was built to read, one per role, when it finds them by
    role (`nn.by_declaration` with ``files``, `nn.declared_files`); else None."""
    from refactory_dsh.nn.resolve import resolve

    call = node.call if node.kind == "op" else node.evaluator if node.kind == "gate" else ""
    if not call:
        return None
    try:
        roles = getattr(resolve(call), "declared_files", None)
    except Exception:  # noqa: BLE001 - an unresolvable call is V8's to report
        return None
    return tuple(roles) if roles is not None else None




def _read_renames(node: Node, files: Mapping[str, str]) -> dict[str, str] | None:
    """For a step that finds its files by role (`_roles`) or shows their names (collect),
    its ``renames`` ``reads`` once the files in ``files`` (old -> new) are renamed: each
    built name mapped to the name it is read under, so the step still knows which file is
    which whatever else changes in its reads (`nn.declared_files`), and shows each under
    the name it was built with. None for any other step, or when nothing changes."""
    roles = _roles(node)
    before = dict((node.renames or {}).get("reads") or {})
    if roles is None and _marked(node, "shows_names"):
        # It shows each file under its name (collect): the names it was built with.
        built_as = {read: built for built, read in before.items()}
        roles = tuple(built_as.get(read, read) for read in node.reads)
    if roles is None:
        return None
    after = {built: files.get(before.get(built, built), before.get(built, built))
             for built in roles}
    after = {built: name for built, name in after.items() if name != built}
    return after if after != before else None




def _names_itself(node: Node, code: str | None, name: str) -> bool:
    """Whether ``node`` may name ``name`` itself (its code says it, or its callable is not
    marked `nn.by_declaration`): renaming its declaration does not reach it."""
    if code is not None:
        return re.search(rf"\b{re.escape(name)}\b", code) is not None
    return not _by_declaration(node)




def _parse_takes_over(takes: Any) -> tuple[str, frozenset[str] | None]:
    """The step and the specific outputs a ``takes_over`` names, or ``None`` for all.

    A string takes all outputs; a list ``[step, output, ...]`` takes only those outputs."""
    if isinstance(takes, str):
        return takes, None
    if isinstance(takes, list) and len(takes) >= 2 and all(isinstance(x, str) for x in takes):
        return takes[0], frozenset(takes[1:])
    raise _Refused("'takes_over' names a step (all its outputs), or [step, output, ...] "
                   "for specific outputs")




def _take_over_plan(graph: Graph, state: Mapping[str, Any], target_id: Any, side: str,
                    anchor: Node, *, only: frozenset[str] | None = None
                    ) -> tuple[Node, dict[tuple[str, str], str], set[str], dict[str, str],
                               dict[str, dict[str, str | None]] | None]:
    """What ``insert``'s 'takes_over' does: the step it takes over, the new name of each of
    that step's outputs (``(kind, name) -> new name``: ``<step>.txt`` and ``<step>`` when
    free), the steps that read on under the new names -- every step that does not run
    after the new one -- and the new text of each knob they read that names a renamed
    fact, and for a code step taken over the ``renames`` its run records its outputs
    under the new names by (`_output_renames`; else None). Refused when the step does
    not run before the new one on every path to it, has no outputs, has readers none of
    which runs after the new one while the Task scores none of them (so the new outputs
    would reach nothing, `_unscored_take_over`), sets them from a callable that names
    them itself, or a reader's reading cannot be renamed: a frozen knob, a knob a step
    after the new one reads too (it must keep the old name), a fact named in a step's
    code or feedback keys, or a file an op or gate opens by name. No renaming reaches
    those.

    ``only``: when not None, only those outputs are taken over and the rest pass through
    unchanged (the target keeps them under their original names)."""
    if not isinstance(target_id, str) or target_id not in graph.nodes:
        raise _Refused(f"'takes_over' names an existing step, not {target_id!r}")
    target = graph.nodes[target_id]
    if target.kind not in ("agent", "op"):
        raise _Refused(f"{target_id!r} is a {target.kind}; only agent or op steps can be taken over")
    if not (target.writes or target.produces):
        raise _Refused(f"{target_id!r} has no outputs to take over")
    if only is not None:
        available = {*target.writes, *target.produces}
        missing = only - available
        if missing:
            raise _Refused(f"'takes_over' names outputs of {target_id!r}: "
                           f"{', '.join(sorted(missing))} not among its outputs "
                           f"({', '.join(sorted(available))})")
    if target.kind == "op" and _code_text(target, state) is None and not _by_declaration(target):
        raise _Refused(f"{target_id!r} runs {target.call!r}, which sets its outputs under names "
                       "of its own; 'takes_over' cannot rename them: take over a model or code "
                       "step")
    following = [e.target for e in graph.out_edges(anchor.id)] if side == "after" else [anchor.id]
    # Where the new step is: every way through ``cut`` passes it. After a step with a
    # fallback it goes where the two ways rejoin (`_at_rejoin`), before that step.
    cut = anchor.id
    if side == "after" and graph.falls_back(anchor.id):
        cut = _fallback_parts(graph, anchor.id)[1]
        following = [cut]
    before = [anchor.id] if side == "after" else [e.source for e in graph.in_edges(anchor.id)]
    if side == "before":
        before += [n.id for n in graph.nodes.values()
                   if (n.kind == "fork" and anchor.id in n.targets)
                   or (n.kind == "data" and n.subgraph_entry == anchor.id)]
    if not _runs_first_on_every_path(graph, target_id, before):
        hint = _co._not_on_every_path_hint(graph, target_id)
        raise _Refused(f"{target_id!r} does not run before the new step on every path to it "
                       f"({hint}); "
                       "'takes_over' names a step every run passes before the new one (not a "
                       "step after it, on a sibling branch or on a branch a route can skip)")
    after: set[str] = set()
    for start in following:
        after |= _downstream(graph, start)
    made = {("file", n) for node in graph.nodes.values() for n in node.writes}
    made |= {("fact", n) for node in graph.nodes.values() for n in node.produces}
    names: dict[tuple[str, str], str] = {}
    for kind, outputs, own in (("file", target.writes, f"{target_id}.txt"),
                               ("fact", target.produces, target_id)):
        for index, name in enumerate(outputs):
            if only is not None and name not in only:
                continue
            new = own if index == 0 and (kind, own) not in made else f"{target_id}_own_{name}"
            names[(kind, name)] = new
            made.add((kind, new))
    group = _writer_group(graph, state, target, set(names), anchor.id, cut, after)
    # The steps that read the group's outputs: those after a writer of the group and
    # before the new one. A step that reads the names only from a writer outside the
    # group (one whose way reaches a later reader without the new step) keeps them.
    flow = _forward_flow(graph)
    readers = {n for member in group for n in _reach(flow, member)
               if n not in after and n != target_id}
    for node_id in sorted(group - {target_id}):
        node = graph.nodes[node_id]
        if node.kind == "op" and _code_text(node, state) is None and not _by_declaration(node):
            raise _Refused(f"{node_id!r} writes {target_id!r}'s outputs too, from {node.call!r}, "
                           "which sets them under names of its own; 'takes_over' cannot rename "
                           "them")
    facts = {old for (kind, old) in names if kind == "fact"}
    files = {old for (kind, old) in names if kind == "file"}
    knobs = state.get("knobs") or {}
    frozen = {name for name, entry in knobs.items() if (entry.get("spec") or {}).get("frozen")}
    later_knobs = {k for n in after if n in graph.nodes
                   for k in (*graph.nodes[n].knobs, *(k for _f, k in graph.nodes[n].field_bindings))}
    texts: dict[str, str] = {}
    for node_id in sorted(readers):
        node = graph.nodes[node_id]
        for knob in dict.fromkeys((*node.knobs, *(k for _f, k in node.field_bindings))):
            text = str((knobs.get(knob) or {}).get("value") or "")
            renamed = _renamed_slots(text, names)
            if renamed == text:
                continue
            if knob in frozen:
                raise _Refused(f"{node_id!r}'s frozen {knob!r} reads {target_id!r}'s output by "
                               "name; it cannot be changed to read it under its new name")
            if knob in later_knobs:
                raise _Refused(f"{node_id!r} and a step after the new one both read {knob!r}, "
                               f"which names {target_id!r}'s output: renamed, the later step "
                               "would read the old output; give one of them its own text first")
            texts[knob] = renamed
        named = sorted(facts & set(node.feedback_keys))
        code = _code_text(node, state)
        # A fact an op or gate reads, it reads by name unless its code says otherwise.
        named += [fact for fact in sorted(facts & set(node.inputs))
                  if code is None or re.search(rf"\b{re.escape(fact)}\b", code)]
        if node.kind == "op" or (node.kind == "gate" and node.gate_kind == "fn"):
            # An op or fn gate opens its files in its code: a renamed read reaches it only
            # when it opens what it declares (LiveMath's notes_prompts opens
            # compare_notes.txt by name). A model step is handed its files as declared.
            # One whose names matter (`nn.names_matter`: select matches a label against
            # them) would act differently on a renamed read.
            named += [name for name in sorted(files & {*node.reads, *node.required_reads})
                      if _names_itself(node, code, name) or _marked(node, "names_matter")]
        if named:
            raise _Refused(f"{node_id!r} reads {target_id!r}'s output {', '.join(map(repr, named))} "
                           "in its code or feedback keys, which 'takes_over' cannot rename; take "
                           "over a step no such step reads, or change that step first")
    unscored = _co._unscored_take_over(graph, state, target, after, only=only)
    if unscored:
        raise _Refused(f"nothing after the new step reads {target_id!r}'s outputs ({unscored}), "
                       "so taking it over changes nothing; take over a step whose output a "
                       "later step reads or the Task scores, or insert before those readers")
    # The step's outputs are renamed: its run records them under the new names.
    # Code steps return values under the names the code was written for;
    # agent steps write files under the names their prompt says.  Both need a
    # renames entry so collection maps the produced name to the declared one.
    renames = _output_renames(target, [names.get(("fact", n), n) for n in target.produces],
                              [names.get(("file", n), n) for n in target.writes])
    return target, names, readers, texts, renames, group




def _writer_group(graph: Graph, state: Mapping[str, Any], target: Node,
                  taken: set[tuple[str, str]], anchor: str, cut: str, after: set[str]
                  ) -> set[str]:
    """The steps a take-over of ``target``'s outputs renames together: ``target`` and every
    other step before the new one writing those names -- a fallback, a route way's in-place
    writer -- whose every way to a step after the new one that reads them passes the new
    step (every way through ``cut`` does; the step it follows, ``anchor``, leads to it), so
    the new step reads whichever ran. A writer with a way around the new step to such a
    reader keeps its names: that reader reads it there."""
    flow = _forward_flow(graph)
    later = {n for n in after if n in graph.nodes and _consumed(graph.nodes[n], state) & taken}
    target_reads = _consumed(target, state) & taken
    group = {target.id}
    for node_id, node in graph.nodes.items():
        if node_id in after or node_id == target.id or not _outputs(node) & taken:
            continue
        if target_reads and (_outputs(node) & target_reads) and target.id in _reach(flow, node_id):
            continue
        around = set() if node_id == anchor else _reach(flow, node_id, without=cut)
        if not around & later:
            group.add(node_id)
    return group




def _unscored_take_over(graph: Graph, state: Mapping[str, Any], target: Node,
                        after: set[str], *, only: frozenset[str] | None = None) -> str:
    """Why outputs of ``target`` that a new step writes again reach nothing, or ``""``: a
    step in ``after`` (those after the new step) reads one, the Task scores one
    (`task.graded_files`), or the Task declares no scored files (it may score any name).

    ``only``: when not None, only those outputs are checked."""
    from refactory_dsh.task import graded_files

    all_taken = _outputs(target)
    taken = ({(k, n) for k, n in all_taken if n in only} if only is not None else all_taken)
    read_by = {n for n, node in graph.nodes.items()
               if n != target.id and _consumed(node, state) & taken}
    if read_by & after:
        return ""
    graded = graded_files(state)
    if graded is None:
        return ""  # a Task that declares no scored files may score any name: taken
    writes = ([w for w in target.writes if w in only] if only is not None else target.writes)
    if set(writes) & set(graded):
        return ""
    return (f"its readers {', '.join(sorted(read_by))} run before it" if read_by else
            f"the Task scores only {', '.join(graded)}")




def _renamed_slots(text: str, names: Mapping[tuple[str, str], str]) -> str:
    """``text`` with each renamed fact's ``{{slot}}`` and ``{% if %}`` test renamed."""
    for (kind, old), new in names.items():
        if kind != "fact":
            continue
        text = re.sub(r"\{\{\s*" + re.escape(old) + r"\s*\}\}", "{{" + new + "}}", text)
        text = re.sub(r"(\{%\s*(?:el)?if\s+)" + re.escape(old) + r"(\s*%\})",
                      lambda m: m.group(1) + new + m.group(2), text)
    return text




def _renamed_refs(node_state: dict[str, Any], names: Mapping[tuple[str, str], str]
                  ) -> dict[str, Any]:
    """A node state reading each renamed output under its new name."""
    out = dict(node_state)
    files = {old: new for (kind, old), new in names.items() if kind == "file"}
    facts = {old: new for (kind, old), new in names.items() if kind == "fact"}
    for key, table in (("reads", files), ("required_reads", files), ("inputs", facts)):
        if out.get(key):
            out[key] = [table.get(name, name) for name in out[key]]
    for key in ("prompt_template", "system_prompt_template"):
        if out.get(key):
            out[key] = _renamed_slots(str(out[key]), names)
    return out




def _select_reads(graph: Graph, anchor: Node) -> list[str]:
    """A select step's reads by default: the label ``anchor`` wrote, then the reply of each
    branch of the nearest fork before it."""
    label = list(anchor.writes[:1])
    current, seen = anchor.id, set()
    while current not in seen:
        seen.add(current)
        node = graph.nodes[current]
        if node.kind == "join":
            for fork in graph.nodes.values():
                if fork.kind == "fork" and graph.fork_join(fork.id) == node.id:
                    ends = [e.source for e in graph.in_edges(node.id)]
                    files = [graph.nodes[end].writes[0] for end in ends
                             if graph.nodes[end].writes]
                    return label + files
        preds = graph.in_edges(current)
        if len(preds) != 1:
            break
        current = preds[0].source
    return []




def _expand_insert(edit: dict, state: dict) -> list[dict]:
    from refactory_dsh.nn.operators import SELECT_CALL

    build = _Steps(state)
    graph = build.graph
    side, anchor = _where(edit, graph)
    spec = _insert_spec(edit, anchor, state)
    avail = _available(graph)
    if side == "after":
        there = avail.get(anchor.id, set()) | _outputs(anchor)
        view = _neighbour_view(graph, state, anchor)
    else:
        there = set(avail.get(anchor.id, set()))
        view = _consumed(anchor, state)
    node_id = spec.get("id") or _fresh(graph, f"{anchor.id}_{'next' if side == 'after' else 'prep'}")
    if spec["kind"] == "op" and spec["call"] == SELECT_CALL and not spec.get("reads"):
        spec["reads"] = _select_reads(graph, anchor) if side == "after" else []
    takes = edit.get("takes_over")
    if takes is not None:
        target_id_parsed, only = _parse_takes_over(takes)
        plan = _co._take_over_plan(graph, state, target_id_parsed, side, anchor, only=only)
    else:
        only = None
        plan = None
    if plan is not None and only is None:
        step_kw = {"writes": list(plan[0].writes), "produces": list(plan[0].produces)}
    else:
        step_kw: dict[str, Any] = {}
        if edit.get("produces") is not None:
            step_kw["produces"] = list(edit["produces"])
        if edit.get("writes") is not None:
            step_kw["writes"] = list(edit["writes"])
    nodes = _step_nodes(graph, state, spec, node_id,
                        _co._seen(graph, state, there, view, spec.get("sees")),
                        settings_from=anchor if anchor.kind == "agent" else None,
                        **step_kw)
    if plan is not None and only is not None:
        taken_writes = [old for (kind, old) in plan[1] if kind == "file"]
        taken_produces = [old for (kind, old) in plan[1] if kind == "fact"]
        step = nodes[-1]
        step["writes"] = list(step.get("writes", [])) + taken_writes
        step["produces"] = list(step.get("produces", [])) + taken_produces
    if plan is not None:
        target, names, readers, texts, renames, group = plan
        # The step taken over writes under new names (selective: only the taken ones;
        # the rest keep their original names).
        build.add(_co._respec(build.state, target.id, {
            **target.to_state(),
            "writes": [names.get(("file", n), n) for n in target.writes],
            "produces": [names.get(("fact", n), n) for n in target.produces],
            **({"renames": renames} if renames else {})}))
        files = {old: new for (kind, old), new in names.items() if kind == "file"}
        for reader in sorted(readers):
            node = build.graph.nodes[reader]
            before = node.to_state()
            after = _renamed_refs(before, names)
            roles = _read_renames(node, files)
            if roles is not None:
                # It finds its files by role: it is told the new name of each it reads.
                after["renames"] = {**dict(node.renames or {}), "reads": roles} if roles \
                    else {k: v for k, v in dict(node.renames or {}).items() if k != "reads"}
            if reader in group:
                # A co-writer of the taken names (a fallback, a route way's step) writes the
                # new names as the taken step does; both code and agent steps record them so
                # (`renames`), so collection maps the produced name to the declared one.
                facts = [names.get(("fact", n), n) for n in node.produces]
                files_new = [names.get(("file", n), n) for n in node.writes]
                after["writes"], after["produces"] = files_new, facts
                outputs = _output_renames(node, facts, files_new) or {}
                kept = {k: v for k, v in dict(after.get("renames") or {}).items()
                        if k == "reads"}
                after["renames"] = {**kept, **outputs} if kept or outputs else None
                if after["renames"] is None:
                    after.pop("renames")
            if after != before:
                build.add(_co._respec(build.state, reader, after))
        for knob, text in sorted(texts.items()):
            build.add([{"op": "set_knob", "name": knob, "value": text}])
        nodes = [_renamed_refs(n, names) for n in nodes]
        # A sealed code step opens its reads by name in its code.  When a
        # takeover renames a file (output.xlsx -> implement_own_output.xlsx),
        # the code still opens the original name: record the mapping so
        # sealed_execute can stage the file under both names.
        file_renames = {old: new for (kind, old), new in names.items()
                        if kind == "file"}
        if file_renames:
            for n in nodes:
                if not n.get("sealed_script"):
                    continue
                read_map = {old: new for old, new in file_renames.items()
                            if new in (n.get("reads") or [])}
                if read_map:
                    n["renames"] = {**(n.get("renames") or {}), "reads": read_map}
    head, _end = _chain(nodes)
    if side == "before":
        build.add(_co._renamed(build.state, anchor.id, [head], joins=False))
    if side == "after" and build.graph.falls_back(anchor.id):
        # After a step with a fallback is where its two ways rejoin, so the step keeps
        # its fallback, and the new step runs after whichever answered.
        build.add(_at_rejoin(build.graph, anchor.id, nodes))
    else:
        build.add(_add_chain(nodes, {side: anchor.id}))
    if plan is None:
        _place_all(build, _new_facts(nodes), after=node_id, to=spec.get("to"))
    return build.steps
