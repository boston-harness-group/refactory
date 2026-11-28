from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from refactory_dsh.nn.graph import Graph, Node

from ._common import TASK, _Refused, _edits
from .state import _available, _consumed, _kind_of, _knob_values, _task_knob
from .visibility import _readable, _seen, _strip_unseen, _with_sections
from .build import _Steps, _renamed, _respec
from .checks import _owned, _plain_step
from .remove import _bypass, _replace_hint
from .parallelize import _tools_beside


def _rewire_seen(graph: Graph, state: Mapping[str, Any], edit: Mapping[str, Any], node: Node
                 ) -> tuple[str, str | None, set[tuple[str, str]], list[tuple[str, str]]]:
    """A rewire's collect-step name, the collect step it made before (if any), what the
    step sees now and what it will see: what 'sees' names (default ``"task"``), or --
    when only 'text' is given -- what it sees now, so new instructions alone keep its
    inputs."""
    gather = f"{node.id}_inputs"
    helper = gather if gather in _owned(graph, node.id, state) else None
    there = set(_available(graph).get(helper or node.id, set()))
    now = _consumed(node, state)
    if helper is not None:
        now = (now - {("fact", gather)}) | _consumed(graph.nodes[helper], state)
    if edit.get("text") is not None and "sees" not in edit:
        seen = _seen(graph, state, there, _readable(graph, state, now), None)
    else:
        seen = _seen(graph, state, there, set(), edit.get("sees", [TASK]))
    return gather, helper, now, seen


def _own_text_slots(graph: Graph, state: Mapping[str, Any], node: Node, text: str,
                    seen: Sequence[tuple[str, str]], gather: str) -> None:
    """Refused when the proposer's 'text' names a slot the step would not be given: a fact
    or file it will not see (say which step to add to 'sees'), or a knob it does not read.
    An inherited prompt's unseen slots are taken out instead (`_strip_unseen`)."""
    from refactory_dsh.nn.operators import made_by
    from refactory_dsh.nn.validate import template_names

    facts = {name for kind, name in seen if kind == "fact"}
    if any(kind == "file" for kind, _name in seen):
        facts.add(gather)
    knobs = set(state.get("knobs") or {})
    makers = made_by(graph)
    system = {knob for field_name, knob in node.field_bindings
              if field_name == "system_prompt_template"}
    seen_files = {name for kind, name in seen if kind == "file"}
    for name in sorted(template_names(text)):
        if name in system:
            raise _Refused(f"'text' names {{{{{name}}}}}, {node.id!r}'s own system prompt, "
                           "which it is already given; set that knob to change it")
        if name in knobs:
            if name not in node.knobs:
                raise _Refused(f"'text' names knob {{{{{name}}}}}, which {node.id!r} does not "
                               "read; write what it should say into 'text' itself")
        elif name not in facts:
            files = {made_name for kind, made_name in makers
                     if kind == "file" and (made_name == name
                                            or made_name.rsplit(".", 1)[0] == name)}
            if files & seen_files:
                # A file it sees reaches an agent through its collect step, as one fact.
                raise _Refused(f"'text' names {{{{{name}}}}}: the files {node.id!r} sees reach "
                               f"it together as {{{{{gather}}}}}; use that slot")
            made = sorted({m for (kind, made_name), steps in makers.items()
                           if made_name == name or (kind == "file"
                                                    and made_name.rsplit(".", 1)[0] == name)
                           for m in steps})
            add = (f"add {' or '.join(made)} to 'sees'" if made
                   else "name only what it sees")
            raise _Refused(f"'text' names {{{{{name}}}}}, which {node.id!r} would not see; "
                           f"{add}")


def _pre_rewire(edit: dict, state: dict) -> str | None:
    graph = _edits()._graph(state)
    node = _plain_step(graph, edit.get("step"), "rewire")
    kind = _kind_of(node)
    if kind == "op":
        raise _Refused(f"{node.id!r} is a plain op: what it reads is what its callable reads. "
                       "Rewire an agent or a code step")
    if _tools_beside(node):
        raise _Refused(f"{node.id!r} uses tools in the run directory, where it can read "
                       "anything; rewiring its declared inputs would not change what it sees")
    own = edit.get("text")
    if own is not None:
        if kind != "agent":
            raise _Refused(f"'text' is an agent's new instructions; {node.id!r} is a code "
                           "step, so rewire what it sees alone, or replace it (split into one "
                           "code step of your own text)")
        if not isinstance(own, str) or not own.strip():
            raise _Refused("'text' is the step's new instructions: a non-empty string")
    gather, _helper, now, seen = _rewire_seen(graph, state, edit, node)
    if own is None and set(seen) == _readable(graph, state, now):
        raise _Refused(f"{node.id!r} already sees exactly that "
                       f"({', '.join(name for _k, name in seen) or 'nothing'}), so rewiring "
                       "would change nothing")
    if kind == "agent":
        knob = _task_knob(node)
        if knob is not None and len(graph.knob_consumers().get(knob, ())) > 1:
            raise _Refused(f"{node.id!r}'s prompt knob {knob!r} is shared with other steps")
        if own is not None:
            _own_text_slots(graph, state, node, own, seen, gather)
            return None
        text = str(_knob_values(state).get(knob) if knob else node.prompt_template)
        keep = {name for k, name in seen if k == "fact"} | set(state.get("knobs") or {}) | {gather}
        if text.strip() and not _strip_unseen(text, keep).strip():
            raise _Refused(f"{node.id!r}'s whole prompt is a fact it would no longer see, so "
                           "rewiring would leave it no instructions; name that fact's step in "
                           "'sees', or "
                           f"{_own_text_hint(node, state, edit.get('sees', [TASK]))}")
    return None


#: How a rewire refusal names the rewire that gives the step its own text (checked like
#: a replacing hint, `optim.base.checked_hint`).
OWN_TEXT_HINT = "give it its own text with "


def _own_text_hint(node: Node, state: dict, sees: Any) -> str:
    """How to give ``node`` its own instructions with what it sees: this rewire with
    'text', which keeps its id and so its place in a fork or join, when it applies to the
    graph; else the replacing split (`_replace_hint`)."""
    edit = {"op": "rewire", "step": node.id, "sees": sees, "text": "<what it does>"}
    try:
        if _edits().precheck(edit, state) is None:
            _edits().apply(edit, state)
            return f"{OWN_TEXT_HINT}{json.dumps(edit)}"
    except _edits().EditError:
        pass
    return f"give it your own text: {_replace_hint(node, state, sees)}"


def _expand_rewire(edit: dict, state: dict) -> list[dict]:
    from refactory_dsh.nn.operators import collect_node

    build = _Steps(state)
    graph = build.graph
    node = graph.nodes[edit["step"]]
    gather, helper, _now, seen = _rewire_seen(graph, state, edit, node)
    facts = [name for kind, name in seen if kind == "fact"]
    files = [name for kind, name in seen if kind == "file"]
    new_state = node.to_state()
    if _kind_of(node) == "code":
        new_state.update({"reads": files, "inputs": facts})
        new_state.pop("required_reads", None)
        build.add(_respec(build.state, node.id, new_state))
        return build.steps
    knob = _task_knob(node)
    before = text = str(_knob_values(state).get(knob) if knob else node.prompt_template)
    if edit.get("text") is not None:
        # Its own instructions, in place: the same id, outputs and place in any fork or
        # join, so every reader and role name stays. A prompt that was not a knob becomes
        # its own ``<id>__task`` when the node is added again (`edits._own_bindings`).
        text = str(edit["text"])
    keep = set(facts) | set(state.get("knobs") or {})
    if helper is not None:
        build.add(_bypass(build.state, helper))
    from refactory_dsh.workspace import is_sealed

    # A sealed step reads its files in its own workspace, as a new one does (`_step_nodes`):
    # they are its reads, exactly what 'sees' says, never a collect step's text.
    sealed = is_sealed(node)
    if files and not sealed:
        keep.add(gather)
        facts.append(gather)
    new_text = _with_sections(_strip_unseen(text, keep), facts)
    for key in ("reads", "required_reads", "inputs"):
        new_state.pop(key, None)
    if sealed and files:
        new_state["reads"] = files
    # Its prompt as a knob carries the new slots (the field too, so the knob's slot check
    # holds); a prompt that is not a knob is re-slotted in place: data wiring, no new text.
    new_state["prompt_template"] = new_text
    # Its own prompt knob (`<id>__...`) goes and comes back with it (`_respec`), at the new
    # text, so the restored value already holds the new slots.
    own_knob = knob is not None and knob.startswith(f"{node.id}__")
    build.add(_respec(build.state, node.id, new_state,
                      values={knob: new_text} if own_knob else None))
    if files and not sealed:
        build.add(_renamed(build.state, node.id, [gather], joins=False))
        build.add([{"op": "add_node", "position": {"before": node.id},
                    "node": collect_node(gather, files, gather,
                                         notes=f"Hands {node.id} the files it sees.")}])
    if knob is not None and not own_knob and new_text != before:
        build.add([{"op": "set_knob", "name": knob, "value": new_text}])
    return build.steps
