from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from refactory_dsh.nn.graph import Graph

from ._common import FILES_HEADING, TASK, _Refused
from .state import _available, _grader_side, _outputs, _task_inputs


def _readable(graph: Graph, state: Mapping[str, Any], names: set[tuple[str, str]]
              ) -> set[tuple[str, str]]:
    """``names`` less what grading made (rule V17) and less knobs."""
    knobs = set(state.get("knobs") or {})
    return {(kind, name) for kind, name in names - _grader_side(graph, state)
            if not (kind == "fact" and name in knobs)}


def _no_task(graph: Graph, state: Mapping[str, Any], there: set[tuple[str, str]]) -> str:
    """Why ``"task"`` gives nothing before a step, naming each op that does not declare
    which outputs carry its knobs and whose declaring none of them would give the step
    something (so all its outputs count as carrying its knobs only for want of it)."""
    from dataclasses import replace

    undeclared = []
    for node in graph.nodes.values():
        if node.kind != "op" or node.knob_outputs is not None or not node.knobs:
            continue
        trial = graph.copy()
        trial.nodes[node.id] = replace(node, knob_outputs={
            out: () for out in [*node.writes, *node.produces]})
        if _task_inputs(trial, state) & there:
            undeclared.append(node.id)
    why = ("\"task\" gives nothing before this step: no fact or file there is free of a "
           "step's output and of a knob's text, so the step would see nothing")
    if undeclared:
        ops = ", ".join(sorted(undeclared))
        return (f"{why}. Op {ops} reads a knob and does not declare which of its outputs "
                "carry it, so every output counts as carrying it; the mode can declare "
                f"knob_outputs ({{output: [knobs]}}) on {ops}. Until then, name the facts or "
                "steps the new step sees")
    return f"{why}; name the facts or steps it sees"


#: How many of the names a new step could see a refusal lists.
SEES_NAMES_SHOWN = 24


#: How many valid "after" anchors a placement hint lists.
_PLACEMENT_SHOWN = 6


def _producers_of(graph: Graph, name: str) -> list[str]:
    """Step ids that write or produce ``name`` (file or fact)."""
    return sorted(node_id for node_id, node in graph.nodes.items()
                  if name in node.writes or name in node.produces)


def _placement_anchors(graph: Graph, name: str, kind: str | None = None, *,
                       keys: set[tuple[str, str]] | None = None) -> list[str]:
    """Step ids where ``"after": id`` would put the new step where ``name`` is there on
    every path: ``name`` is in ``avail[id] | _outputs(id)`` and ``id`` is on a
    start-to-terminal path.  When ``kind`` is given, only ``("file", name)`` or
    ``("fact", name)`` is checked; otherwise both.  ``keys`` overrides both."""
    avail = _available(graph)
    if keys is None:
        keys = {(kind, name)} if kind else {("file", name), ("fact", name)}
    reachable = graph.reachable()
    return sorted(node_id for node_id in graph.nodes
                  if node_id in reachable
                  and keys & (avail.get(node_id, set()) | _outputs(graph.nodes[node_id])))


def _missing_edge_hint(graph: Graph, name: str,
                       kind: str | None = None, *,
                       keys: set[tuple[str, str]] | None = None) -> str:
    """Name one in-edge whose predecessor does not carry ``name``, to show which path
    is missing.  Returns a human-readable fragment like ``"has_case --skip--> finalize"``
    or an empty string when no such edge is found (should not happen when ``name`` is
    genuinely missing from ``there``)."""
    if keys is None:
        keys = {(kind, name)} if kind else {("file", name), ("fact", name)}
    avail = _available(graph)
    for node_id in graph.nodes:
        node_avail = avail.get(node_id, set())
        if keys & node_avail:
            continue
        for edge in graph.in_edges(node_id):
            source_left = avail.get(edge.source, set()) | _outputs(graph.nodes[edge.source])
            if not (keys & source_left):
                label = edge.condition or ""
                arrow = f" --{label}-->" if label else " -->"
                return f"{edge.source}{arrow} {node_id}"
    for node in graph.nodes.values():
        if node.kind == "fork":
            for target in node.targets:
                target_avail = avail.get(target, set())
                if not (keys & target_avail):
                    return f"{node.id} --> {target} (branch)"
    return ""


def _not_on_every_path_hint(graph: Graph, name: str,
                            kind: str | None = None, *,
                            keys: set[tuple[str, str]] | None = None) -> str:
    """A compact hint for a ``sees`` name that exists in the graph but is not in ``there``
    on every path: who produces it, one missing edge, and valid placements.

    ``keys`` overrides the ``(kind, name)`` pairs used for the availability check (for a
    step id whose outputs are a known set, not necessarily named after the step).
    """
    producers = _producers_of(graph, name)
    parts: list[str] = []
    if producers:
        shown = producers[:_PLACEMENT_SHOWN]
        more = f" (+{len(producers) - _PLACEMENT_SHOWN} more)" \
            if len(producers) > _PLACEMENT_SHOWN else ""
        parts.append(f"produced by {', '.join(shown)}{more}")
    edge = _missing_edge_hint(graph, name, kind, keys=keys)
    if edge:
        parts.append(f"missing on the {edge} path")
    anchors = _placement_anchors(graph, name, kind, keys=keys)
    if anchors:
        shown = anchors[:_PLACEMENT_SHOWN]
        more = f" (+{len(anchors) - _PLACEMENT_SHOWN} more)" \
            if len(anchors) > _PLACEMENT_SHOWN else ""
        parts.append(f"\"after\" one of [{', '.join(shown)}{more}] would reach it")
    elif producers:
        parts.append("no placement puts it on every path")
    return "; ".join(parts)


def _seen(graph: Graph, state: Mapping[str, Any], there: set[tuple[str, str]],
          default: set[tuple[str, str]], sees: Any) -> list[tuple[str, str]]:
    """What a new (or rewired) step will see: ``default``, or what ``sees`` names, kept
    to what is there before it on every path, facts before files, each sorted."""
    if sees is None:
        # A sealed step's session log (`workspace.is_session_log`) goes with its dispatch
        # and nothing reads it, so a default never hands it on; 'sees' may still name it.
        from refactory_dsh.workspace import is_session_log

        chosen = {key for key in set(default) & there
                  if not (key[0] == "file" and is_session_log(key[1]))}
    else:
        if isinstance(sees, str):
            sees = [sees]
        if not isinstance(sees, list) or not all(isinstance(s, str) for s in sees):
            raise _Refused("'sees' is a list of \"task\", step ids, and fact or file names "
                           "there before the new step")
        chosen = set()
        for name in sees:
            if name == TASK:
                task = _task_inputs(graph, state) & there
                if not task:
                    raise _Refused(_no_task(graph, state, there))
                chosen |= task
            elif name in graph.nodes:
                made = _outputs(graph.nodes[name]) & there
                if not made:
                    all_outputs = _outputs(graph.nodes[name])
                    hint = _not_on_every_path_hint(graph, name,
                                                   keys=all_outputs)
                    raise _Refused(
                        f"step {name!r}'s outputs are not there on every path to "
                        f"the new step, so it cannot see them ({hint}); 'sees' "
                        "requires every path to carry the name, even when the "
                        "step tolerates its absence")
                chosen |= made
            elif ("fact", name) in there or ("file", name) in there:
                chosen |= {key for key in (("fact", name), ("file", name)) if key in there}
            else:
                producers = _producers_of(graph, name)
                if producers:
                    hint = _not_on_every_path_hint(graph, name)
                    raise _Refused(
                        f"'sees' names {name!r}, which exists in the graph "
                        f"({hint}) but is not there before the new step on "
                        "every path; 'sees' requires every path to carry the "
                        "name, even when the step tolerates its absence")
                names = sorted({n for _k, n in _readable(graph, state, there)})
                shown = ", ".join(names[:SEES_NAMES_SHOWN]) + (
                    f" (+{len(names) - SEES_NAMES_SHOWN} more)"
                    if len(names) > SEES_NAMES_SHOWN else "")
                raise _Refused(f"'sees' names {name!r}, which is not \"task\", a step, "
                               "or a fact or file in the graph; no step produces it; "
                               f"there: {shown or 'nothing'}")
    chosen = _readable(graph, state, chosen)
    # An agent's reply is its first file and, when it declares one, its fact: the same
    # text twice. The fact is enough.
    for node in graph.nodes.values():
        if node.kind == "agent" and node.writes and node.produces \
                and any(("fact", f) in chosen for f in node.produces):
            chosen.discard(("file", node.writes[0]))
    return sorted(chosen, key=lambda key: (key[0] != "fact", key[1]))


def _strip_unseen(text: str, keep: set[str]) -> str:
    """``text`` with each ``{{name}}`` slot not in ``keep`` taken out: a line that is only
    the slot goes (with a heading line right above it), an inline one is said to be not
    given."""
    from refactory_dsh.nn.validate import template_names

    unseen = {name for name in template_names(text) if name not in keep}
    if not unseen:
        return text
    for name in sorted(unseen):
        text = re.sub(r"\{%-?\s*if\s+" + re.escape(name) + r"\s*-?%\}.*?\{%-?\s*endif\s*-?%\}",
                      "", text, flags=re.DOTALL)
    unseen = {name for name in template_names(text) if name not in keep}
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        match = re.fullmatch(r"\{\{\s*([A-Za-z_]\w*)\s*\}\}", stripped)
        if match and match.group(1) in unseen:
            if out and re.match(r"^\s*(#{1,6}\s|\S.*:\s*$)", out[-1]):
                out.pop()
            continue
        out.append(line)
    result = "\n".join(out)
    for name in sorted(unseen):
        result = re.sub(r"\{\{\s*" + re.escape(name) + r"\s*\}\}", "(not given to this step)",
                        result)
    return result


def _heading(name: str) -> str:
    if name.endswith("_inputs"):
        return f"{FILES_HEADING} ({name})"
    return name.replace("_", " ").strip().capitalize() or name


def _with_sections(text: str, facts: Sequence[str]) -> str:
    """``text`` with a ``## heading`` and ``{{fact}}`` for each fact it does not fill."""
    from refactory_dsh.nn.validate import template_names

    have = template_names(text)
    sections = [f"## {_heading(fact)}\n{{{{{fact}}}}}" for fact in facts if fact not in have]
    return "\n\n".join([text.rstrip(), *sections]).strip() if sections else text
