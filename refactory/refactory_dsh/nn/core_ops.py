"""The core structural operators: one line each, the data wired by the system.

Part of the edit layer (`nn.edits`): each operator here is a macro registered in its
catalogue, checked by a precondition and expanded into the seven primitives, so it
reaches every outer loop through the one attempt path (`Optimizer.attempt`) and is held
to every rule a hand-written edit is. They say nothing about what a step is for:

- ``insert``: a new step after or before an existing one;
- ``remove``: a step out of the pipeline, a fork branch included; on a gate, ``keep``
  names the verdict whose path stays (so it also un-routes);
- ``parallelize``: existing consecutive steps, copies of a step, or new steps, run at
  once between a fork and a join;
- ``serialize``: a fork's branches one after another, in their declared order;
- ``branch``: a new step as another (optionally optional) branch of an existing fork;
- ``rewire``: what an existing step sees (the task's own input, other steps' outputs);
- ``loop`` (one-line form): a judge after a step that can send it back with a reason;
- ``route``: a judge after a step choosing among labelled next steps that rejoin;
- ``split`` (one-line form): a step replaced by two new ones, or by one (replace);
- ``merge``: two consecutive steps as one;
- ``fallback``: a step that answers instead when a per-item step genuinely fails (never
  for time or a harness fault), writing that step's outputs.

**What the proposer writes** is the kind of step, where, and its text (a prompt, a
callable or code). Every text has a default (the neighbour's, or a neutral line), so a
strategy with no model can apply any operator.

**What the system wires** (`_seen`, `_step_nodes`, `_place`):

- a new step's inputs: by default what its neighbour sees plus the neighbour's outputs
  (``sees`` names others: ``"task"`` for the task's own input, `operators.task_inputs`,
  or a step id for that step's outputs). Facts reach a model step as ``{{slots}}``
  in its own prompt knob; files through a ``collect`` step in front of it
  (`operators.collect`), so a missing file is said, not a crash. A code step gets the
  same files and facts in ``inp``;
- its output: one file named after its id and one fact of that name, placed as `REACHES`
  says (``to`` names another place), guarded when
  the step may not run;
- names: fresh ids, files and facts, the same for the same edit on the same parent;
- the fork's targets and the join's sources, edges, the gate's verdict edges, the
  loop's iteration limit (a knob of its own);
- every text it creates goes into a knob of its own (``<id>__task``, ``<id>__code``),
  for later ``set_knob``.

Nothing here repairs a proposal: an operator either expands into primitives or is
refused with the reason, and the expanded state is validated like any other. Steps
downstream of a change keep the file and fact names they read, unless the operator is
about changing that (``rewire``, ``remove``).
"""

from __future__ import annotations

import json
import re
from typing import Any, Collection, Iterable, Mapping, Sequence

from refactory_dsh.seal import SEALED_TOOLS

from .graph import Graph, Node
from .operators import TASK_MEANING

#: The core operators, in the order the request lists them.
CORE_OPS = ("insert", "remove", "parallelize", "serialize", "branch", "rewire", "loop",
            "route", "split", "merge", "fallback")
#: The kinds of step an operator adds: a model call, a plain op (a callable), or code.
STEP_KINDS = ("agent", "op", "code")
#: Judges of ``loop`` and ``route`` besides a ``module:function`` callable.
JUDGE_KINDS = ("agent", "code")
#: The most steps one ``parallelize`` or ``route`` adds or runs at once.
MAX_BRANCHES = 8
#: The most labels a ``route`` chooses among.
MAX_ROUTES = 6

#: Neutral texts, so an operator applies with none given.
NEUTRAL_STEP = "Work on the task with what you are given, and write your result."
NEUTRAL_CHECK = "Check the work above against the task: is it complete and correct?"
NEUTRAL_ROUTE = "Choose which way the work should go next."
NEUTRAL_REVIEW = "Review the work above against the task and write the final result."
#: What a judge agent is told about its answer's first word.
VERDICT_RULE = ("Start your answer with one word, {labels}, alone or followed by why; "
                "nothing before it.")
def system_appended(text: str) -> tuple[str, str]:
    """``text`` split into the step's own words and what the system appended to them (the
    verdict rule a judge's prompt ends with, `VERDICT_RULE`); ``(text, "")`` when nothing
    was. Our own constant is matched, never the model's words."""
    head, tail = VERDICT_RULE.split("{labels}")
    found = re.search(re.escape(head) + r"(?P<labels>.+?)" + re.escape(tail) + r"\s*$",
                      text or "")
    if not found:
        return text, ""
    return text[:found.start()].rstrip(), text[found.start():].strip()


#: The loop's default iteration limit and its knob's bounds.
LOOP_ITERATIONS = (2, 1, 5)
#: What ``sees`` calls the task's own input (`operators.task_inputs`).
TASK = "task"
#: The heading a ``collect`` step's fact gets in a prompt.
FILES_HEADING = "Files from earlier steps"


class _Refused(Exception):
    """A precondition that does not hold, with its reason."""


# ── reading a state ───────────────────────────────────────────────────────


def _edits():
    from . import edits

    return edits


def _knob_values(state: Mapping[str, Any]) -> dict[str, Any]:
    return {name: (entry or {}).get("value") for name, entry in (state.get("knobs") or {}).items()}


def _outputs(node: Node) -> set[tuple[str, str]]:
    return {("file", n) for n in node.writes} | {("fact", n) for n in node.produces}


def _consumed(node: Node, state: Mapping[str, Any]) -> set[tuple[str, str]]:
    from .validate import consumed

    try:
        return consumed(node, _knob_values(state))
    except Exception:  # noqa: BLE001 - a node that cannot be read is the validator's to say
        return set()


def _available(graph: Graph) -> dict[str, set[tuple[str, str]]]:
    from .validate import available_inputs

    return available_inputs(graph)


def _grader_side(graph: Graph, state: Mapping[str, Any]) -> set[tuple[str, str]]:
    from .validate import grader_derived

    try:
        return set(grader_derived(graph, _knob_values(state)))
    except Exception:  # noqa: BLE001 - an unloadable task is another rule's to report
        return set()


def _task_inputs(graph: Graph, state: Mapping[str, Any]) -> set[tuple[str, str]]:
    """What ``"sees": ["task"]`` gives (`operators.task_inputs`). A node that cannot be
    read refuses the edit with the real error, never as a task that gives nothing."""
    from .operators import task_inputs

    try:
        return task_inputs(graph, _knob_values(state))
    except Exception as exc:  # noqa: BLE001 - said in the refusal
        raise _Refused(f"what \"task\" gives cannot be worked out: {type(exc).__name__}: "
                       f"{exc}") from exc


def _taken_names(graph: Graph) -> set[str]:
    names = set(graph.nodes)
    for node in graph.nodes.values():
        names |= set(node.writes) | set(node.produces)
        names |= {name.rsplit(".", 1)[0] for name in node.writes}
    return names


def _fresh(graph: Graph, base: str, also: Iterable[str] = ()) -> str:
    """An id no node has and no file or fact is named after (nor any in ``also``)."""
    taken = _taken_names(graph) | set(also)
    base = re.sub(r"\W", "_", base).strip("_") or "step"
    if base not in taken and f"{base}.txt" not in taken:
        return base
    for n in range(2, 1000):
        candidate = f"{base}{n}"
        if candidate not in taken:
            return candidate
    raise _edits().EditError(f"cannot find a free id based on {base!r}")


def _task_knob(node: Node) -> str | None:
    return next((knob for field_name, knob in node.field_bindings
                 if field_name == "prompt_template"), None)


def _text_of(node: Node, state: Mapping[str, Any]) -> str:
    """What a step does, as text: an agent's task prompt, a code step's code."""
    values = _knob_values(state)
    field_name = "code" if node.code or any(f == "code" for f, _ in node.field_bindings) \
        else "prompt_template"
    try:
        return str(node.bound(field_name, values) or "")
    except Exception:  # noqa: BLE001
        return str(getattr(node, field_name, "") or "")


def _is_code(node: Node) -> bool:
    from refactory_dsh.snippet import is_code_node

    return is_code_node(node)


def _kind_of(node: Node) -> str:
    """A step's kind as an operator names it."""
    if node.kind == "agent":
        return "agent"
    if node.kind == "op":
        return "code" if _is_code(node) else "op"
    return node.kind


def _uses_tools(node: Node) -> bool:
    return node.kind == "agent" and (node.tool_policy != "none" or bool(node.tools))


def _forks_naming(graph: Graph, node_id: str) -> list[Node]:
    return [n for n in graph.nodes.values()
            if node_id in (*n.targets, *n.optional_targets, *n.sources)]


def _data_ends(graph: Graph, node_id: str) -> list[str]:
    """``entry``/``exit`` for each data pass that starts or ends at ``node_id``."""
    out = []
    for node in graph.nodes.values():
        if node.kind == "data":
            if node.subgraph_entry == node_id:
                out.append("entry")
            if node.subgraph_exit == node_id:
                out.append("exit")
    return out


def _sole_out(graph: Graph, node_id: str) -> str | None:
    """The target of a node's one plain out-edge; refused for any other shape."""
    out = graph.out_edges(node_id)
    if not out:
        return None
    if len(out) > 1 or out[0].condition is not None:
        raise _Refused(f"{node_id!r} has {len(out)} out-edge(s) carrying verdicts, so the "
                       "operator cannot tell which one to follow")
    return out[0].target


def _require_step(graph: Graph, node_id: Any, what: str = "step") -> Node:
    if not isinstance(node_id, str) or node_id not in graph.nodes:
        raise _Refused(f"no {what} {node_id!r}; existing steps: {', '.join(sorted(graph.nodes))}")
    return graph.nodes[node_id]


def _join_of(graph: Graph, fork: Node) -> Node:
    join = graph.fork_join(fork.id)
    if join is None or join not in graph.nodes or graph.nodes[join].kind != "join":
        raise _Refused(f"fork {fork.id!r} has no join after it")
    return graph.nodes[join]


# ── what a new step sees ──────────────────────────────────────────────────


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


def _seen(graph: Graph, state: Mapping[str, Any], there: set[tuple[str, str]],
          default: set[tuple[str, str]], sees: Any) -> list[tuple[str, str]]:
    """What a new (or rewired) step will see: ``default``, or what ``sees`` names, kept
    to what is there before it on every path, facts before files, each sorted."""
    if sees is None:
        chosen = set(default) & there
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
                    raise _Refused(f"step {name!r}'s outputs are not there on every path to "
                                   "the new step, so it cannot see them")
                chosen |= made
            elif ("fact", name) in there or ("file", name) in there:
                chosen |= {key for key in (("fact", name), ("file", name)) if key in there}
            else:
                names = sorted({n for _k, n in _readable(graph, state, there)})
                shown = ", ".join(names[:SEES_NAMES_SHOWN]) + (
                    f" (+{len(names) - SEES_NAMES_SHOWN} more)"
                    if len(names) > SEES_NAMES_SHOWN else "")
                raise _Refused(f"'sees' names {name!r}, which is not \"task\", a step, or a "
                               "fact or file there before the new step; there: "
                               f"{shown or 'nothing'}")
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
    from .validate import template_names

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
    from .validate import template_names

    have = template_names(text)
    sections = [f"## {_heading(fact)}\n{{{{{fact}}}}}" for fact in facts if fact not in have]
    return "\n\n".join([text.rstrip(), *sections]).strip() if sections else text


# ── building a new step ───────────────────────────────────────────────────


def _spec(raw: Any, *, default_kind: str = "agent", default_text: str = NEUTRAL_STEP) -> dict:
    """A step spec ``{"kind", "text", ...}`` with its defaults; refused when malformed."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise _Refused("a new step is an object such as {\"kind\": \"agent\", \"text\": \"...\"}")
    kind = raw.get("kind") or default_kind
    if kind not in STEP_KINDS:
        raise _Refused(f"a new step's kind is one of {list(STEP_KINDS)}, not {kind!r}")
    if kind == "op" and not isinstance(raw.get("call"), str):
        raise _Refused("an op step names its callable: {\"kind\": \"op\", \"call\": \"module:function\"}")
    text = raw.get("text")
    if text is not None and not isinstance(text, str):
        raise _Refused("a new step's 'text' is a string (its prompt, or its code)")
    for key in ("id", "to"):
        if raw.get(key) is not None and not isinstance(raw.get(key), str):
            raise _Refused(f"'{key}' is a string")
    route = None
    if raw.get("route") is not None:
        if kind != "agent":
            raise _Refused("only an agent step takes a 'route'")
        from .graph import route_declaration

        try:
            route = route_declaration(raw["route"])
        except ValueError as exc:
            raise _Refused(str(exc)) from exc
    return {**raw, "kind": kind, "route": route, "text": text if text is not None else (
        default_text if kind == "agent" else "")}


def _default_code(facts: Sequence[str], files: Sequence[str] = ()) -> str:
    """Code that joins what the step was given into each fact and file it declares."""
    return ("def main(inp):\n"
            "    parts = [str(v) for v in inp['facts'].values()]\n"
            "    parts += [str(v) for v in inp['files'].values()]\n"
            "    text = '\\n\\n'.join(parts)\n"
            f"    return {{'facts': {{name: text for name in {list(facts)!r}}},\n"
            f"            'files': {{name: text for name in {list(files)!r}}}}}\n")


def _copy_spec(node: Node, state: Mapping[str, Any]) -> dict:
    """A step spec copying ``node``'s kind and text (an agent's tools come along)."""
    kind = _kind_of(node)
    if kind not in ("agent", "code"):
        return {"kind": "agent", "text": NEUTRAL_STEP}
    spec = {"kind": kind, "text": _text_of(node, state), "_like": node.id}
    if kind == "code":
        spec["_copied"] = spec["text"]
    return spec


def _renamed_code(code: str, source: Node, node_id: str, facts: Sequence[str],
                  files: Sequence[str]) -> str:
    """``code`` copied from ``source`` for a step with outputs of its own: it runs as it
    did, and what it returns under ``source``'s fact and file names is renamed to the
    copy's, in declaration order (one the copy does not declare is dropped). The code
    names its outputs itself, so a bare copy returns the original's and is refused at run
    time. The names are the copy's own, so a copy of a copy wraps a wrapper soundly. Code
    whose ``main`` calls itself calls the wrapper and is renamed twice, which changes
    nothing: the copy's names are fresh, never one of ``source``'s."""
    if list(source.produces) == list(facts) and list(source.writes) == list(files):
        return code
    tag = re.sub(r"\W", "_", node_id)
    fact_names = {old: (facts[i] if i < len(facts) else None)
                  for i, old in enumerate(source.produces)}
    file_names = {old: (files[i] if i < len(files) else None)
                  for i, old in enumerate(source.writes)}
    return (f"{code.rstrip()}\n\n\n"
            f"_main_before_{tag} = main\n"
            f"_names_for_{tag} = {{'facts': {fact_names!r}, 'files': {file_names!r}}}\n\n\n"
            "def main(inp):\n"
            f"    out = _main_before_{tag}(inp)\n"
            "    if not isinstance(out, dict):\n"
            "        return out\n"
            "    renamed = dict(out)\n"
            f"    for key, names in _names_for_{tag}.items():\n"
            "        given = out.get(key)\n"
            "        if isinstance(given, dict):\n"
            "            renamed[key] = {names.get(k, k): v for k, v in given.items()\n"
            "                            if names.get(k, k) is not None}\n"
            "    return renamed\n")


def _step_nodes(graph: Graph, state: Mapping[str, Any], spec: dict, node_id: str,
                seen: Sequence[tuple[str, str]], *, writes: Sequence[str] | None = None,
                produces: Sequence[str] | None = None, settings_from: Node | None = None,
                verdict: Sequence[str] = (), knobs: Sequence[str] = ()) -> list[dict[str, Any]]:
    """The node states a new step takes, in run order: a ``collect`` step for the files it
    sees (agents only) and the step itself.

    ``writes``/``produces`` are its outputs when it takes over another step's; by default
    one file ``<id>.txt`` and one fact ``<id>``. ``settings_from`` is the agent whose role,
    timeout and route a new agent takes, unless the spec names a route of its own; a copy (``spec["_like"]``) of an agent that uses tools
    keeps them, in a fresh workspace of its own. ``verdict`` makes it a judge: its answer
    starts with one of the labels, and it writes its reply only, for the gate to read."""
    from refactory_dsh import snippet

    from .operators import COLLECT_CALL, collect_node

    facts = [name for kind, name in seen if kind == "fact"]
    files = [name for kind, name in seen if kind == "file"]
    kind = spec["kind"]
    like = graph.nodes.get(spec.get("_like") or "") if spec.get("_like") else None
    source = like or settings_from
    out: list[dict[str, Any]] = []
    if kind == "agent":
        text = _strip_unseen(spec["text"] or NEUTRAL_STEP,
                             set(facts) | set(state.get("knobs") or {}))
        tools = spec.get("tools")
        if files and not tools:
            gather = _fresh(graph, f"{node_id}_inputs", (node_id,))
            out.append(collect_node(gather, files, gather,
                                    notes=f"Hands {node_id} the files it sees."))
            facts = [*facts, gather]
        text = _with_sections(text, facts)
        if verdict:
            text = f"{text}\n\n{VERDICT_RULE.format(labels=' or '.join(verdict))}"
        node: dict[str, Any] = {
            "id": node_id, "kind": "agent", "prompt_template": text,
            "system_prompt_template": spec.get("system") or "You are one step of a pipeline.",
            "writes": [f"{node_id}.txt"] if verdict else list(writes or [f"{node_id}.txt"]),
            "tool_policy": "none",
        }
        if not verdict:
            node["produces"] = list(produces) if produces is not None else [node_id]
        if source is not None and source.kind == "agent":
            if source.role:
                node["role"] = source.role
            if source.timeout_s:
                node["timeout_s"] = source.timeout_s
        route = spec.get("route") or (source.route if source is not None
                                      and source.kind == "agent" else None)
        if route:
            node["route"] = dict(route)
        if like is not None and _uses_tools(like):
            # A copy of a step that uses tools keeps them only in a fresh workspace of its
            # own, so it never works where the step it copies works (`coverage.DECISIONS`).
            node["tool_policy"] = like.tool_policy
            if like.tools:
                node["tools"] = list(like.tools)
            node["workspace"] = {"kind": "scratch"}
        if tools:
            # Its tools run sealed, in a fresh workspace holding only the files it reads
            # (`workspace.SEALED_TOOLS`): no other item, grader file or network.
            node["tool_policy"] = None
            node["tools"] = list(tools)
            node["workspace"] = {"kind": "scratch"}
            if files:
                node["reads"] = files
        carried = _carried_knobs("agent", text, node["system_prompt_template"], knobs)
        if carried:
            node["knobs"] = carried
        out.append(node)
        return out
    if kind == "code":
        made = list(produces) if produces is not None else [node_id]
        code = spec["text"] or _default_code(made, list(writes or ()))
        if like is not None and spec.get("_copied") is not None and code == spec["_copied"]:
            code = _renamed_code(code, like, node_id, made, list(writes or ()))
        node = {"id": node_id, "kind": "op", "call": snippet.RUN, "code": code,
                "reads": files, "inputs": facts, "produces": made}
        if writes:
            node["writes"] = list(writes)
        out.append(node)
        return out
    call = spec["call"]
    if call == COLLECT_CALL:
        out.append(collect_node(node_id, list(spec.get("reads") or files),
                                list(produces)[0] if produces else node_id))
        return out
    node = {"id": node_id, "kind": "op", "call": call,
            "reads": list(spec.get("reads") or files),
            "inputs": list(spec.get("inputs") or facts)}
    for key, given in (("writes", writes or spec.get("writes") or [f"{node_id}.txt"]),
                       ("produces", produces if produces is not None
                        else spec.get("produces") or [node_id]),
                       ("required_reads", spec.get("required_reads"))):
        if given:
            node[key] = list(given)
    out.append(node)
    return out


def _shared_knobs(*nodes: Node) -> list[str]:
    """The knobs these steps read that are not their own (``<id>__...``): a mode's knobs,
    which the step taking their place reads on, so none is left read by nothing (V7)."""
    out: list[str] = []
    for node in nodes:
        bound = {knob for _f, knob in node.field_bindings}
        out += [k for k in node.knobs if not k.startswith(f"{node.id}__") and k not in bound
                and k not in out]
    return out


def _carried_knobs(kind: str, text: str | None, system: str | None,
                   shared: Sequence[str]) -> list[str]:
    """The shared knobs (:func:`_shared_knobs`) a step taking others' place reads on: an
    agent those its prompts name. A code step's code never sees the knobs (`snippet`), and
    a new op step is given none. The rest go with the steps they belonged to, and the
    candidate's notes say so (:func:`wiring_notes`)."""
    if kind != "agent":
        return []
    from .validate import template_names

    named = template_names(text or NEUTRAL_STEP) | template_names(system or "")
    return [k for k in shared if k in named]


def _orphaned_knobs(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    """Knobs a step of ``before`` read that no step of ``after`` reads: the settings of the
    steps an operator removed or replaced, which go with them."""
    old, new = _edits()._graph(before), _edits()._graph(after)
    read_before = old.knob_consumers()
    read_after = new.knob_consumers()
    declared = set(after.get("knobs") or {})
    return sorted(k for k in read_before if k not in read_after and k in declared)


def _drop_orphans(build: "_Steps") -> None:
    build.add([{"op": "remove_knob", "name": k} for k in _orphaned_knobs(build.start, build.state)])


def _frozen_settings(graph: Graph, state: Mapping[str, Any], gone: set[str]) -> list[str]:
    """Frozen knobs that only steps in ``gone`` read, which removing them would drop."""
    readers = graph.knob_consumers()
    knobs = state.get("knobs") or {}
    return sorted(k for k, ids in readers.items() if set(ids) <= gone
                  and ((knobs.get(k) or {}).get("spec") or {}).get("frozen"))


def _chain(nodes: Sequence[dict[str, Any]]) -> tuple[str, str]:
    """The first and last id of a new step's nodes (its collect step, then itself)."""
    return str(nodes[0]["id"]), str(nodes[-1]["id"])


def _add_chain(nodes: Sequence[dict[str, Any]], position: dict[str, Any]) -> list[dict]:
    """Primitive steps adding a step's nodes at ``position`` in run order: the step at the
    position, then its collect step in front of it."""
    *before, step = nodes
    steps = [{"op": "add_node", "node": step, "position": position}]
    for node in reversed(before):
        steps.append({"op": "add_node", "node": node, "position": {"before": step["id"]}})
    return steps


def _add_detached(nodes: Sequence[dict[str, Any]], exact: Collection[str] = ()) -> list[dict]:
    steps = [{"op": "add_node", "node": node, "position": {"detached": True},
              **({"_exact": True} if node["id"] in exact else {})} for node in nodes]
    for first, second in zip(nodes, nodes[1:]):
        steps.append({"op": "add_edge", "source": first["id"], "target": second["id"],
                      "condition": None})
    return steps


def _respec(state: Mapping[str, Any], node_id: str, new_state: dict[str, Any]) -> list[dict]:
    """Primitive steps that give an existing node ``new_state`` under the same id, judged
    on ``state`` (the state they apply to): removed and added again, with its own knobs
    (those ``remove_node`` takes with it) declared again at their values, and every edge it
    has in ``state``. Used for a fork's targets, a join's sources and what a rewired step
    reads: topology, never a setting."""
    graph = _edits()._graph(state)
    node = graph.nodes[node_id]
    knobs = state.get("knobs") or {}
    readers = graph.knob_consumers()
    steps: list[dict] = [{"op": "remove_node", "id": node_id}]
    restore: list[dict] = []
    for _field, knob in node.field_bindings:
        if knob.startswith(f"{node_id}__") and set(readers.get(knob, ())) <= {node_id} \
                and knob in knobs:
            entry = knobs[knob]
            steps.append({"op": "add_knob", "name": knob, "spec": dict(entry.get("spec") or {})})
            if entry.get("value") != (entry.get("spec") or {}).get("default"):
                restore.append({"op": "set_knob", "name": knob, "value": entry.get("value")})
    position = {"start": True} if graph.start == node_id else {"detached": True}
    steps.append({"op": "add_node", "node": new_state, "position": position})
    for edge in graph.edges:
        if node_id in (edge.source, edge.target):
            steps.append({"op": "add_edge", "source": edge.source, "target": edge.target,
                          "condition": edge.condition})
    return steps + restore


def _renamed(state: Mapping[str, Any], old: str, new: Sequence[str], *, joins: bool = True,
             forks: bool = True) -> list[dict]:
    """Respec steps, judged on ``state``, for every fork naming ``old`` as a branch and every
    join naming it as a source, so they name ``new`` instead (none: the name is dropped).
    A branch keeps its optional mark under its new name."""
    graph = _edits()._graph(state)
    steps: list[dict] = []
    current = dict(state)
    for node_id in sorted(n.id for n in _forks_naming(graph, old)):
        node = _edits()._graph(current).nodes[node_id]
        changed = node.to_state()
        if node.kind == "fork" and forks and old in node.targets:
            changed["targets"] = [t for name in node.targets
                                  for t in ([*new] if name == old else [name])]
            optional = [t for name in node.optional_targets
                        for t in ([*new] if name == old else [name])]
            if optional:
                changed["optional_targets"] = optional
            else:
                changed.pop("optional_targets", None)
        elif node.kind == "join" and joins and old in node.sources:
            changed["sources"] = [s for name in node.sources
                                  for s in ([*new] if name == old else [name])]
        else:
            continue
        more = _respec(current, node_id, changed)
        steps += more
        current = _simulate(current, more)
    return steps


def _simulate(state: Mapping[str, Any], steps: Sequence[dict]) -> dict[str, Any]:
    return _edits().apply_many(steps, state)


class _Steps:
    """Primitive steps built in order, each judged on the state the ones before it leave."""

    def __init__(self, state: Mapping[str, Any]) -> None:
        self.start = dict(state)
        self.state = dict(state)
        self.steps: list[dict] = []

    def add(self, steps: Sequence[dict]) -> None:
        if steps:
            self.state = _simulate(self.state, steps)
            self.steps += list(steps)

    @property
    def graph(self) -> Graph:
        return _edits()._graph(self.state)


# ── where a new step's output goes ────────────────────────────────────────


def _filled_text_knobs(node: Node, state: Mapping[str, Any]) -> list[str]:
    """The text knobs an op that fills facts fills (`validate.filled_knobs`)."""
    from .validate import filled_knobs

    try:
        names = filled_knobs(node)
    except Exception:  # noqa: BLE001 - an unresolvable callable fills nothing known
        return []
    knobs = state.get("knobs") or {}
    return [name for name in names
            if ((knobs.get(name) or {}).get("spec") or {}).get("kind") == "str"]


def _fills(node: Node, state: Mapping[str, Any]) -> bool:
    return node.kind in ("op", "gate") and bool(_filled_text_knobs(node, state))


def _readers_from(graph: Graph, node_id: str, state: Mapping[str, Any]) -> list[str]:
    """The steps from ``node_id`` on whose prompts are filled with facts -- agents, and ops
    that fill facts into the knobs they read -- itself first when it is one, then the rest
    downstream, nearest first (through fork branches and verdicts, not back along a
    reloop)."""
    from .loops import loop_edges

    retries = {(source, target) for source, target, _c in loop_edges(graph)}
    order: list[str] = []
    seen, queue = {node_id}, [node_id]
    first = graph.nodes[node_id]
    if first.kind == "agent" or _fills(first, state):
        order.append(node_id)
    while queue:
        current = queue.pop(0)
        node = graph.nodes.get(current)
        if node is None:
            continue
        following = [e.target for e in graph.out_edges(current)
                     if (current, e.target) not in retries]
        if node.kind == "fork":
            following = [*node.targets, *following]
        for target in following:
            if target in seen or target not in graph.nodes:
                continue
            seen.add(target)
            if graph.nodes[target].kind == "agent" or _fills(graph.nodes[target], state):
                order.append(target)
            queue.append(target)
    return order


def _place(state: Mapping[str, Any], fact: str, *, readers: Sequence[str],
           to: str | None = None, guarded: bool = False) -> list[dict]:
    """The ``set_knob`` that puts ``{{fact}}`` where the next model step's prompt is filled:
    ``to`` (a step, an agent's prompt knob or a knob an op fills facts into), else the first
    of ``readers`` that is an agent whose prompt is a knob of its own, or an op filling
    facts into exactly one text knob (an op filling several is a choice for ``to``);
    guarded by ``{% if %}`` when the fact may not be there on every path. None when no such
    place is found and ``to`` was not given."""
    edits = _edits()
    graph = edits._graph(state)
    knob_to: str | None = None
    if to is not None:
        if to in graph.nodes:
            readers = [to]
        else:
            owners = [n.id for n in graph.nodes.values()
                      if _task_knob(n) == to or to in _filled_text_knobs(n, state)]
            if not owners:
                raise edits.EditError(f"'to' {to!r} is neither a step nor a prompt knob a step "
                                      "fills", code="E-wiring")
            readers, knob_to = owners[:1], to
    avail = _available(graph)
    for reader in readers:
        node = graph.nodes.get(reader)
        knob: str | None = None
        if node is not None and node.kind == "agent":
            knob = _task_knob(node)
            if knob is not None and len(graph.knob_consumers().get(knob, ())) > 1:
                knob = None
        elif node is not None:
            filled = _filled_text_knobs(node, state)
            knob = knob_to if knob_to in filled else (filled[0] if len(filled) == 1 else None)
            if knob is None and to is None and filled:
                return []  # it fills several prompts: which one is the proposer's to say
        if knob is None:
            if to is not None:
                raise edits.EditError(f"step {reader!r} has no prompt knob of its own to put "
                                      f"{{{{{fact}}}}} in; name the knob with 'to'",
                                      code="E-wiring")
            continue
        current = str(((state.get("knobs") or {}).get(knob) or {}).get("value") or "")
        if re.search(r"\{\{\s*" + re.escape(fact) + r"\s*\}\}", current):
            return []
        section = f"## {_heading(fact)}\n{{{{{fact}}}}}"
        if guarded or ("fact", fact) not in avail.get(reader, set()):
            section = f"{{% if {fact} %}}{section}{{% endif %}}"
        return [{"op": "set_knob", "name": knob, "value": f"{current.rstrip()}\n\n{section}".lstrip()}]
    return []


def _place_all(build: _Steps, facts: Sequence[str], *, after: str, to: str | None = None,
               guarded: bool = False) -> None:
    """`_place` each fact, readers from the first model step after ``after`` on."""
    for fact in facts:
        graph = build.graph
        readers = _readers_from(graph, after, build.state)
        if readers and readers[0] == after:
            readers = readers[1:]
        build.add(_place(build.state, fact, readers=readers, to=to, guarded=guarded))


def _new_facts(nodes: Sequence[dict[str, Any]]) -> list[str]:
    """The facts a new step makes for later steps (not its collect step's)."""
    return list(nodes[-1].get("produces") or ()) if nodes else []


# ── shared checks ─────────────────────────────────────────────────────────


def _plain_step(graph: Graph, node_id: Any, what: str) -> Node:
    node = _require_step(graph, node_id)
    if node.kind not in ("agent", "op"):
        raise _Refused(f"{node.id!r} is a {node.kind}; {what} works on a step (agent, op or code)")
    return node


def _slots_in_field(node: Node) -> set[str]:
    from .validate import template_names

    return template_names(str(node.prompt_template or ""))


def _losses(graph: Graph, state: Mapping[str, Any], gone: set[str]
            ) -> tuple[list[str], dict[str, set[str]], dict[str, set[str]]]:
    """What removing ``gone`` takes from the steps left: the steps that need something only
    ``gone`` made (each said); the prompt slots that can simply be taken out of a step's own
    prompt knob (a slot an operator put there, not one its declared prompt has); and the
    optional reads of ops and gates that name a file nothing makes any more."""
    made: set[tuple[str, str]] = set()
    for node_id in gone:
        made |= _outputs(graph.nodes[node_id])
    others: set[tuple[str, str]] = set()
    for node in graph.nodes.values():
        if node.id not in gone:
            others |= _outputs(node)
    lost = made - others
    hard: list[str] = []
    strip: dict[str, set[str]] = {}
    drop: dict[str, set[str]] = {}
    for node in sorted(graph.nodes.values(), key=lambda n: n.id):
        if node.id in gone or node.kind == "data":
            continue
        needs = _consumed(node, state)
        if node.kind in ("op", "gate"):
            # An op's optional read of a file nothing makes any more is dropped from its
            # reads (V6 refuses a read no path makes); its required reads cannot be.
            optional = {("file", name) for name in node.reads if name not in node.required_reads}
            if optional & lost:
                drop[node.id] = {name for _k, name in optional & lost}
            needs = needs - optional
        hit = needs & lost
        knob = _task_knob(node)
        own = (knob is not None and len(graph.knob_consumers().get(knob, ())) == 1)
        loose = {name for kind, name in hit if kind == "fact" and own
                 and name not in _slots_in_field(node) and name not in node.inputs
                 and name not in node.feedback_keys}
        if loose:
            strip[node.id] = loose
        rest = sorted(name for kind, name in hit if not (kind == "fact" and name in loose))
        if rest:
            hard.append(f"{node.id!r} needs {', '.join(rest)}")
    return hard, strip, drop


def _strip_steps(state: Mapping[str, Any], strip: Mapping[str, set[str]]) -> list[dict]:
    graph = _edits()._graph(state)
    steps = []
    for node_id, names in sorted(strip.items()):
        knob = _task_knob(graph.nodes[node_id])
        value = str(((state.get("knobs") or {}).get(knob) or {}).get("value") or "")
        from .validate import template_names

        keep = template_names(value) - set(names)
        new = _strip_unseen(value, keep | set(state.get("knobs") or {})).rstrip()
        if new != value:
            steps.append({"op": "set_knob", "name": knob, "value": new})
    return steps


def _owned(graph: Graph, step: str, state: Mapping[str, Any]) -> list[str]:
    """Helper steps that exist only for ``step``: the collect step an operator put in front
    of it, and the judge a ``loop`` or ``route`` put in front of its gate, when only
    ``step`` reads what they make."""
    out = []
    for helper in (f"{step}_inputs", f"{step}_judge"):
        node = graph.nodes.get(helper)
        if node is None or [e.target for e in graph.out_edges(helper)] != [step]:
            continue
        readers = {n.id for n in graph.nodes.values()
                   if _consumed(n, state) & _outputs(node)} - {helper}
        if readers <= {step}:
            out.append(helper)
    return out


def _upstream(graph: Graph, node_id: str) -> set[str]:
    """Every node a walk can pass before ``node_id``: along edges, from a fork to its
    branch heads, from a join's sources to the join and from a data node to its pass's
    entry -- the control flow `loops.loop_edges` reads cycles from -- so an edge back to
    one of them closes a cycle (a fork branch's head included)."""
    def before(key: str) -> list[str]:
        node = graph.nodes.get(key)
        out = [e.source for e in graph.in_edges(key)]
        out += [n.id for n in graph.nodes.values()
                if (n.kind == "fork" and key in n.targets)
                or (n.kind == "data" and n.subgraph_entry == key)]
        if node is not None and node.kind == "join":
            out += list(node.sources)
        return out

    seen: set[str] = set()
    stack = before(node_id)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(before(current))
    return seen


# ── insert ────────────────────────────────────────────────────────────────


def _where(edit: dict, graph: Graph) -> tuple[str, Node]:
    """The side and the step an insert names. Before a fallback gate is after the step it
    answers for (`_fallback_parts`), so the step keeps its fallback."""
    from .graph import FALLBACK_GATE_CALL

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
        from .validate import fork_branches

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
        _sole_out(graph, anchor.id)
    elif anchor.kind in ("join", "data"):
        raise _Refused(f"{anchor.id!r} is a {anchor.kind}; insert after it, or before the "
                       "step it leads to")
    spec = _insert_spec(edit, anchor, state)
    if spec.get("id") is not None and spec["id"] in graph.nodes:
        raise _Refused(f"node {spec['id']!r} already exists")
    if edit.get("takes_over") is not None:
        target = _take_over_plan(graph, state, edit["takes_over"], side, anchor)[0]
        if spec["kind"] == "agent" and not spec.get("tools") and len(target.writes) > 1:
            raise _Refused(f"{target.id!r} writes {len(target.writes)} files, and a new agent "
                           "step without tools writes only its reply; take it over with a "
                           "code step, or with 'tools'")
    return None


def _downstream(graph: Graph, start: str) -> set[str]:
    """``start`` and every step after it (fork branches and a data pass's steps included,
    no reloop back)."""
    from .loops import loop_edges

    back = {(source, target) for source, target, _c in loop_edges(graph)}
    out, queue = {start}, [start]
    while queue:
        node = graph.nodes.get(queue.pop())
        if node is None:
            continue
        following = [e.target for e in graph.out_edges(node.id) if (node.id, e.target) not in back]
        for target in [*(node.targets if node.kind == "fork" else ()),
                       *((node.subgraph_entry,) if node.kind == "data" and node.subgraph_entry
                         else ()), *following]:
            if target not in out:
                out.add(target)
                queue.append(target)
    return out


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
    from .loops import loop_edges

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


def _take_over_plan(graph: Graph, state: Mapping[str, Any], target_id: Any, side: str,
                    anchor: Node
                    ) -> tuple[Node, dict[tuple[str, str], str], set[str], dict[str, str]]:
    """What ``insert``'s 'takes_over' does: the step it takes over, the new name of each of
    that step's outputs (``(kind, name) -> new name``: ``<step>.txt`` and ``<step>`` when
    free), the steps that read on under the new names -- every step that does not run
    after the new one -- and the new text of each knob they read that names a renamed
    fact. Refused when the step does not run before the new one on every path to it, has
    no outputs, or a reader's reading cannot be renamed: a frozen knob, a knob a step
    after the new one reads too (it must keep the old name), or the fact named in a
    step's code or feedback keys, which no renaming reaches."""
    if not isinstance(target_id, str) or target_id not in graph.nodes:
        raise _Refused(f"'takes_over' names an existing step, not {target_id!r}")
    target = graph.nodes[target_id]
    if target.kind not in ("agent", "op") or not (target.writes or target.produces):
        raise _Refused(f"{target_id!r} is a {target.kind} with no outputs to take over")
    following = [e.target for e in graph.out_edges(anchor.id)] if side == "after" else [anchor.id]
    before = [anchor.id] if side == "after" else [e.source for e in graph.in_edges(anchor.id)]
    if side == "before":
        before += [n.id for n in graph.nodes.values()
                   if (n.kind == "fork" and anchor.id in n.targets)
                   or (n.kind == "data" and n.subgraph_entry == anchor.id)]
    if not _runs_first_on_every_path(graph, target_id, before):
        raise _Refused(f"{target_id!r} does not run before the new step on every path to it; "
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
            new = own if index == 0 and (kind, own) not in made else f"{target_id}_own_{name}"
            names[(kind, name)] = new
            made.add((kind, new))
    readers = {n for n in graph.nodes if n not in after and n != target_id}
    facts = {old for (kind, old) in names if kind == "fact"}
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
        for fact in sorted(facts & set(node.inputs)):
            if code is None or re.search(rf"\b{re.escape(fact)}\b", code):
                named.append(fact)
        if named:
            raise _Refused(f"{node_id!r} reads {target_id!r}'s output {', '.join(map(repr, named))} "
                           "in its code or feedback keys, which 'takes_over' cannot rename; take "
                           "over a step no such step reads, or change that step first")
    return target, names, readers, texts


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
    from .operators import SELECT_CALL

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
    plan = _take_over_plan(graph, state, takes, side, anchor) if takes is not None else None
    nodes = _step_nodes(graph, state, spec, node_id,
                        _seen(graph, state, there, view, spec.get("sees")),
                        settings_from=anchor if anchor.kind == "agent" else None,
                        **({"writes": list(plan[0].writes), "produces": list(plan[0].produces)}
                           if plan is not None else {}))
    if plan is not None:
        target, names, readers, texts = plan
        # The step taken over writes under new names, and every step before the new one
        # (the new one too) reads them there; the steps after it read the new step's.
        build.add(_respec(build.state, target.id, {
            **target.to_state(),
            "writes": [names[("file", n)] for n in target.writes],
            "produces": [names[("fact", n)] for n in target.produces]}))
        for reader in sorted(readers):
            node = build.graph.nodes[reader]
            before = node.to_state()
            after = _renamed_refs(before, names)
            if after != before:
                build.add(_respec(build.state, reader, after))
        for knob, text in sorted(texts.items()):
            build.add([{"op": "set_knob", "name": knob, "value": text}])
        nodes = [_renamed_refs(n, names) for n in nodes]
    head, _end = _chain(nodes)
    if side == "before":
        build.add(_renamed(build.state, anchor.id, [head], joins=False))
    if side == "after" and build.graph.falls_back(anchor.id):
        # After a step with a fallback is where its two ways rejoin, so the step keeps
        # its fallback, and the new step runs after whichever answered.
        build.add(_at_rejoin(build.graph, anchor.id, nodes))
    else:
        build.add(_add_chain(nodes, {side: anchor.id}))
    if plan is None:
        _place_all(build, _new_facts(nodes), after=node_id, to=spec.get("to"))
    return build.steps


# ── remove ────────────────────────────────────────────────────────────────


#: What a replacing hint adds when the step taking the place is code.
CODE_HINT_NOTE = (" (its default code joins what it is given; give 'text', your def main(inp), "
                  "for your own)")


def _replace_hint(node: Node, state: dict, sees: Any = None) -> str:
    """How to replace ``node`` by one step of the proposer's own text (an agent for an
    agent, code for anything else: a new agent writes only its reply): the exact edit when
    it applies to the graph, else why replacing it is refused too. An attempt holds the
    named edit to its own checks as well (`optim.base.checked_hint`)."""
    step: dict[str, Any] = ({"kind": "agent", "text": "<what it does>"} if node.kind == "agent"
                            else {"kind": "code"})
    if sees is not None:
        step["sees"] = sees
    edit = {"op": "split", "step": node.id, "into": [step]}
    refused = _edits().precheck(edit, state)
    if refused is None:
        try:
            _edits().apply(edit, state)
        except _edits().EditError as exc:
            refused = str(exc)
    if refused is not None:
        return ("replacing it is refused too ("
                f"{refused.removeprefix('E-precondition: ')})")
    return f"replace it with {json.dumps(edit)}" + ("" if node.kind == "agent" else CODE_HINT_NOTE)


def _pre_remove(edit: dict, state: dict) -> str | None:
    from .graph import FALLBACK

    graph = _edits()._graph(state)
    node = _require_step(graph, edit.get("step"))
    owner = _fallback_owner(graph, node.id)
    if owner is not None:
        # A fallback's gate or way goes as a whole (`_unfallback`); the step it answers for
        # goes as it would with no fallback.
        if node.kind == "gate" and edit.get("keep") == FALLBACK:
            raise _Refused(f"{node.id!r} is the fallback gate of {owner!r}; remove it (or "
                           "the fallback step) to take the fallback out, or remove "
                           f"{owner!r} itself")
        if node.id != owner:
            return None
        return _pre_remove(edit, _simulate(state, _unfallback(state, owner)))
    if node.kind in ("fork", "join"):
        raise _Refused(f"{node.id!r} is a {node.kind}; 'serialize' the fork to run its branches "
                       "one after another, or remove a branch's step")
    if node.kind == "data":
        raise _Refused(f"{node.id!r} is a data node: the data pass is the grader's side")
    if node.id == graph.start:
        raise _Refused(f"{node.id!r} is the start step; insert a step before it, or "
                       f"{_replace_hint(node, state)}")
    ends = _data_ends(graph, node.id)
    if ends:
        raise _Refused(f"{node.id!r} is the {' and '.join(ends)} of a data pass, so it stays; "
                       f"{_replace_hint(node, state)}")
    if node.kind == "gate":
        labels = sorted({str(e.condition) for e in graph.out_edges(node.id) if e.condition})
        keep = edit.get("keep")
        if keep is None and len(graph.out_edges(node.id)) > 1:
            raise _Refused(f"{node.id!r} is a gate; say which verdict's path to keep with "
                           f"'keep' (one of {labels})")
        if keep is not None and keep not in labels:
            raise _Refused(f"gate {node.id!r} has no verdict {keep!r}; its verdicts: {labels}")
        target = next((e.target for e in graph.out_edges(node.id)
                       if keep is None or e.condition == keep), None)
        if target is not None and (target == node.id or target in _upstream(graph, node.id)):
            raise _Refused(f"verdict {keep!r} of {node.id!r} leads back to {target!r}; keep the "
                           "verdict that goes on")
        return None
    if len(graph.out_edges(node.id)) > 1:
        raise _Refused(f"{node.id!r} has several out-edges")
    overlapping = {e.source for e in graph.in_edges(node.id)} & {
        e.target for e in graph.out_edges(node.id)}
    if overlapping:
        raise _Refused(f"{node.id!r} sits inside a loop with {sorted(overlapping)}; remove the "
                       "loop's gate first (remove it with 'keep')")
    if any(e.source == node.id and graph.nodes[e.target].kind == "gate"
           and any(back.target == node.id for back in graph.out_edges(e.target))
           for e in graph.out_edges(node.id)) or any(
            graph.nodes[e.source].kind == "gate" and e.condition
            and e.source in {o.target for o in graph.out_edges(node.id)}
            for e in graph.in_edges(node.id)):
        raise _Refused(f"a gate sends work back to {node.id!r}; remove that gate first (remove "
                       "it with 'keep')")
    gone = {node.id, *_owned(graph, node.id, state)}
    frozen = _frozen_settings(graph, state, gone)
    if frozen:
        raise _Refused(f"{node.id!r} alone reads the frozen knob(s) {frozen}, which keep their "
                       "value and declaration; remove another step")
    hard, _strip, _drop = _losses(graph, state, gone)
    if hard:
        raise _Refused(f"removing {node.id!r} leaves steps without an input only it makes: "
                       f"{'; '.join(hard)}; remove or rewire those first")
    return None


def _bypass(state: Mapping[str, Any], node_id: str) -> list[dict]:
    """Primitive steps that take one plain step out, judged on ``state``: forks and joins
    that name it name its neighbours on the branch instead, then each predecessor leads to
    its successor (a branch end's predecessor to the join), then the step goes."""
    build = _Steps(state)
    graph = build.graph
    preds = graph.in_edges(node_id)
    succ = graph.out_edges(node_id)
    ends_join = [j.id for j in graph.nodes.values() if j.kind == "join" and node_id in j.sources]
    heads_in = [f.id for f in graph.nodes.values() if f.kind == "fork" and node_id in f.targets]
    for fork_id in heads_in:
        join = graph.fork_join(fork_id)
        heads = [e.target for e in succ if e.target != join]
        build.add(_renamed(build.state, node_id, heads, joins=False))
    if ends_join:
        build.add(_renamed(build.state, node_id, [e.source for e in preds], forks=False))
    edges = []
    for incoming in preds:
        for outgoing in succ:
            if incoming.source != outgoing.target:
                edges.append({"op": "add_edge", "source": incoming.source,
                              "target": outgoing.target, "condition": incoming.condition})
    have = {e.key() for e in build.graph.edges}
    build.add([e for e in edges if (e["source"], e["target"], e["condition"] or "") not in have
               and (e["source"], e["target"], e["condition"]) not in have])
    build.add([{"op": "remove_node", "id": node_id}])
    return build.steps


def _fold_forks(state: Mapping[str, Any]) -> list[dict]:
    """Serialize every fork left with one branch or none."""
    graph = _edits()._graph(state)
    for fork in sorted(graph.nodes.values(), key=lambda n: n.id):
        if fork.kind == "fork" and len(fork.targets) <= 1:
            steps = _serialize_steps(state, fork.id)
            return steps + _fold_forks(_simulate(state, steps))
    return []


def _expand_remove(edit: dict, state: dict) -> list[dict]:
    build = _Steps(state)
    owner = _fallback_owner(build.graph, edit["step"])
    if owner is not None:
        # Removing a fallback's gate or step takes the whole fallback out, so the graph is
        # what it was before it; removing the step it answers for takes it out first.
        build.add(_unfallback(state, owner))
        if edit["step"] != owner:
            _drop_orphans(build)
            return build.steps
        state = build.state
    graph = build.graph
    node = graph.nodes[edit["step"]]
    if node.kind == "gate":
        keep = edit.get("keep")
        kept = [e for e in graph.out_edges(node.id) if keep is None or e.condition == keep][:1]
        before = graph.reachable()
        build.add([{"op": "add_edge", "source": incoming.source, "target": outgoing.target,
                    "condition": incoming.condition}
                   for incoming in graph.in_edges(node.id) for outgoing in kept
                   if incoming.source != outgoing.target])
        build.add([{"op": "remove_node", "id": node.id}])
        after = build.graph
        build.add([{"op": "remove_node", "id": orphan}
                   for orphan in sorted((before & set(after.nodes)) - after.reachable())])
        judge = f"{node.id}_judge"
        after = build.graph
        if judge in after.nodes and not any(
                _consumed(n, build.state) & _outputs(after.nodes[judge])
                for n in after.nodes.values() if n.id != judge):
            for helper in [*_owned(after, judge, build.state), judge]:
                build.add(_bypass(build.state, helper))
    else:
        gone = {node.id, *_owned(graph, node.id, state)}
        _hard, strip, drop = _losses(graph, state, gone)
        build.add(_strip_steps(state, strip))
        for reader, names in sorted(drop.items()):
            changed = build.graph.nodes[reader].to_state()
            changed["reads"] = [r for r in changed.get("reads") or () if r not in names]
            build.add(_respec(build.state, reader, changed))
        for helper in [*_owned(graph, node.id, state), node.id]:
            build.add(_bypass(build.state, helper))
    build.add(_fold_forks(build.state))
    _drop_orphans(build)
    return build.steps


# ── parallelize ───────────────────────────────────────────────────────────


def _chain_of(graph: Graph, ids: Sequence[str]) -> None:
    for first, second in zip(ids, ids[1:]):
        out = graph.out_edges(first)
        if len(out) != 1 or out[0].target != second or out[0].condition is not None:
            raise _Refused(f"{first!r} does not lead straight to {second!r}; give consecutive "
                           "steps, in run order")
        if len(graph.in_edges(second)) != 1:
            raise _Refused(f"{second!r} is reached from more than one step")


def _tools_beside(node: Node) -> bool:
    return _uses_tools(node) and node.workspace is None and not node.outputs_only


def _pre_parallelize(edit: dict, state: dict) -> str | None:
    graph = _edits()._graph(state)
    if edit.get("steps") is not None and edit.get("new") is not None:
        raise _Refused("parallelize takes 'steps' (existing steps, or one with 'copies') or "
                       "'after' with 'new' (new steps), not both")
    if edit.get("new") is not None:
        anchor = _require_step(graph, edit.get("after"))
        if anchor.kind in ("gate", "fork"):
            raise _Refused(f"{anchor.id!r} is a {anchor.kind}; put new steps after a step")
        _sole_out(graph, anchor.id)
        new = edit["new"]
        if not isinstance(new, list) or not 2 <= len(new) <= MAX_BRANCHES:
            raise _Refused(f"'new' is a list of 2 to {MAX_BRANCHES} steps")
        for raw in new:
            _spec(raw)
        return None
    ids = edit.get("steps")
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids):
        raise _Refused("'steps' is a list of step ids")
    nodes = [_plain_step(graph, i, "parallelize") for i in ids]
    for node in nodes:
        if _forks_naming(graph, node.id) and node.id not in (ids[0], ids[-1]):
            raise _Refused(f"{node.id!r} is named by a fork or join")
    copies = edit.get("copies")
    if len(ids) == 1:
        if not isinstance(copies, int) or isinstance(copies, bool) \
                or not 1 <= copies < MAX_BRANCHES:
            raise _Refused(f"one step runs beside 1 to {MAX_BRANCHES - 1} copies of itself: "
                           "give 'copies'")
        node = nodes[0]
        if node.kind != "agent":
            raise _Refused(f"{node.id!r} is not a model step; a copy of an op or of code would "
                           "compute the same thing again. Copy an agent")
        if edit.get("sees") is not None:
            raise _Refused("a copy is sent exactly what the step is sent, so it sees what the "
                           "step sees; for a new attempt with other inputs, give 'new' steps "
                           "('after' and 'new', each with its 'sees')")
        if _tools_beside(node):
            raise _Refused(f"{node.id!r} uses tools in the run directory, so a copy beside it "
                           "could see its half-done work; copy a step without tools, or one "
                           "that works in a workspace of its own")
    else:
        if copies is not None:
            raise _Refused("'copies' is for one step; list several steps without it")
        if len(ids) > MAX_BRANCHES or len(set(ids)) != len(ids):
            raise _Refused(f"'steps' names 2 to {MAX_BRANCHES} distinct steps")
        _chain_of(graph, ids)
        for node in nodes:
            if _tools_beside(node):
                raise _Refused(f"{node.id!r} uses tools in the run directory, so it cannot run "
                               "beside another step; give it a workspace of its own first")
        made: dict[tuple[str, str], str] = {}
        for node in nodes:
            for key in _outputs(node):
                if key in made:
                    raise _Refused(f"{made[key]!r} and {node.id!r} both make {key[1]!r}")
                made[key] = node.id
        for node in nodes:
            needs = sorted(name for key, name in _consumed(node, state)
                           if made.get((key, name)) not in (None, node.id))
            if needs:
                raise _Refused(f"{node.id!r} reads {needs}, which a step it would run beside "
                               "makes; they cannot run at once")
    if nodes[0].id == graph.start:
        raise _Refused(f"{nodes[0].id!r} is the start step; insert a step before it first")
    _sole_out(graph, ids[-1])
    return None


def _fork_join_ids(graph: Graph, base: str) -> tuple[str, str]:
    fork = _fresh(graph, f"{base}_fork")
    return fork, _fresh(graph, f"{base}_join", (fork,))


def _clone_agent(node: Node, state: Mapping[str, Any], copy_id: str
                 ) -> tuple[list[dict], dict[str, Any]]:
    """A copy of agent ``node`` as ``copy_id``: the same prompts, system prompt, role,
    limits and tools, so it is sent byte for byte what ``node`` is sent, with outputs of its
    own (``<copy_id>.txt``, ``<copy_id>``). A text bound to one of ``node``'s knobs is bound
    to a knob of the copy's own (`<copy_id>__...`) holding the same value, frozen when
    ``node``'s is -- also when ``node`` shares it with other steps, so tuning the copy never
    tunes them. A text ``node`` holds in its declaration stays so in the copy: the copy has
    no knob ``node`` lacks (its add_node is ``_exact``). Returns the primitive steps
    declaring those knobs, and the copy's node state."""
    knobs = state.get("knobs") or {}
    copy = node.to_state()
    copy["id"] = copy_id
    if node.writes:
        copy["writes"] = [f"{copy_id}.txt", *(f"{copy_id}_{name}" for name in node.writes[1:])]
    if node.produces:
        copy["produces"] = [copy_id, *(f"{copy_id}_{name}" for name in node.produces[1:])]
    steps: list[dict] = []
    renamed: dict[str, str] = {}
    suffixes = {"prompt_template": "task", "system_prompt_template": "system_prompt",
                "code": "code"}
    for field_name, knob in node.field_bindings:
        own = (knob[len(node.id) + 2:] if knob.startswith(f"{node.id}__")
               else suffixes.get(field_name, field_name))
        new_knob = f"{copy_id}__{own}"
        entry = knobs.get(knob) or {}
        # Declared at the original's value, so a frozen knob needs no set_knob.
        spec = {**dict(entry.get("spec") or {}), "default": entry.get("value")}
        steps.append({"op": "add_knob", "name": new_knob, "spec": spec})
        renamed[knob] = new_knob
    copy["knobs"] = [renamed.get(k, k) for k in node.knobs]
    copy["field_bindings"] = [[f, renamed.get(k, k)] for f, k in node.field_bindings]
    # Its tools as the original's, said outright: a new agent without a tool_policy is
    # given none (`edits._add_node`).
    copy["tool_policy"] = node.tool_policy
    if node.workspace is not None or _uses_tools(node):
        # A copy never works where the step it copies works (`coverage.DECISIONS`).
        copy["workspace"] = {"kind": "scratch"}
    copy.pop("notes", None)
    return steps, copy


def _expand_parallelize(edit: dict, state: dict) -> list[dict]:
    build = _Steps(state)
    graph = build.graph
    avail = _available(graph)
    if edit.get("new") is not None:
        anchor = graph.nodes[edit["after"]]
        fork, join = _fork_join_ids(graph, anchor.id)
        there = avail.get(anchor.id, set()) | _outputs(anchor)
        view = _neighbour_view(graph, state, anchor)
        branches: list[list[dict]] = []
        taken: list[str] = [fork, join]
        for index, raw in enumerate(edit["new"]):
            spec = _spec(raw)
            node_id = spec.get("id") or _fresh(graph, f"{anchor.id}_branch{index + 1}", taken)
            taken.append(node_id)
            nodes = _step_nodes(graph, state, spec, node_id,
                                _seen(graph, state, there, view, spec.get("sees")),
                                settings_from=anchor if anchor.kind == "agent" else None)
            taken += [str(n["id"]) for n in nodes]
            branches.append(nodes)
        build.add([
            {"op": "add_node", "node": {"id": join, "kind": "join",
                                        "sources": [_chain(n)[1] for n in branches]},
             "position": {"after": anchor.id}},
            {"op": "add_node", "node": {"id": fork, "kind": "fork",
                                        "targets": [_chain(n)[0] for n in branches]},
             "position": {"after": anchor.id}},
        ])
        for nodes in branches:
            build.add(_add_detached(nodes))
            build.add([{"op": "add_edge", "source": nodes[-1]["id"], "target": join,
                        "condition": None}])
        _place_all(build, [f for nodes in branches for f in _new_facts(nodes)], after=join)
        return build.steps
    ids = list(edit["steps"])
    first, last = graph.nodes[ids[0]], graph.nodes[ids[-1]]
    fork, join = _fork_join_ids(graph, first.id)
    copies: list[list[dict]] = []
    exact: set[str] = set()
    if len(ids) == 1:
        taken = [fork, join]
        there = set(avail.get(first.id, set()))
        for index in range(int(edit["copies"])):
            copy_id = _fresh(graph, f"{first.id}_copy{index + 1}", taken)
            taken.append(copy_id)
            knob_steps, node = _clone_agent(first, state, copy_id)
            build.add(knob_steps)
            copies.append([node])
            exact.add(copy_id)
    heads = [ids[0], *(_chain(n)[0] for n in copies)] if copies else list(ids)
    ends = [ids[-1], *(_chain(n)[1] for n in copies)] if copies else list(ids)
    # The new fork stands where the first step stood in an enclosing fork, and the new
    # join where the last step ended an enclosing branch.
    build.add(_renamed(build.state, first.id, [fork], joins=False))
    build.add(_renamed(build.state, last.id, [join], forks=False))
    build.add([
        {"op": "add_node", "node": {"id": join, "kind": "join", "sources": ends},
         "position": {"after": last.id}},
        {"op": "add_node", "node": {"id": fork, "kind": "fork", "targets": heads},
         "position": {"before": first.id}},
        {"op": "remove_edge", "source": fork, "target": first.id, "condition": None},
        *({"op": "remove_edge", "source": a, "target": b, "condition": None}
          for a, b in zip(ids, ids[1:])),
        *({"op": "add_edge", "source": a, "target": join, "condition": None} for a in ids[:-1]),
        {"op": "add_edge", "source": fork, "target": join, "condition": None},
    ])
    for nodes in copies:
        build.add(_add_detached(nodes, exact=exact))
        build.add([{"op": "add_edge", "source": nodes[-1]["id"], "target": join,
                    "condition": None}])
    _place_all(build, [f for nodes in copies for f in _new_facts(nodes)], after=join)
    return build.steps


# ── serialize ─────────────────────────────────────────────────────────────


def _pre_serialize(edit: dict, state: dict) -> str | None:
    graph = _edits()._graph(state)
    fork = _require_step(graph, edit.get("fork"), "fork")
    if fork.kind != "fork":
        raise _Refused(f"{fork.id!r} is a {fork.kind}, not a fork")
    _serialize_plan(graph, fork)
    return None


def _serialize_plan(graph: Graph, fork: Node) -> tuple[Node, list[list[tuple[str, Any]]]]:
    from .validate import fork_branches

    join = _join_of(graph, fork)
    if fork.id == graph.start:
        raise _Refused(f"fork {fork.id!r} is the start step; insert a step before it first")
    if _data_ends(graph, fork.id) or _data_ends(graph, join.id):
        raise _Refused(f"fork {fork.id!r} or its join starts or ends a data pass")
    ends = [[(e.source, e.condition) for e in graph.in_edges(join.id) if e.source in branch]
            for branch in fork_branches(graph, fork, join.id)]
    return join, ends


def _serialize_steps(state: Mapping[str, Any], fork_id: str) -> list[dict]:
    build = _Steps(state)
    graph = build.graph
    fork = graph.nodes[fork_id]
    join, ends = _serialize_plan(graph, fork)
    heads = list(fork.targets)
    onward = [(e.target, e.condition) for e in graph.out_edges(join.id)]
    last_ends = [source for source, _c in (ends[-1] if ends else [])]
    build.add(_renamed(build.state, fork.id, heads[:1], joins=False))
    build.add(_renamed(build.state, join.id, last_ends, forks=False))
    edges: list[dict] = []
    for edge in graph.in_edges(fork.id):
        targets = [(heads[0], None)] if heads else onward
        edges += [{"op": "add_edge", "source": edge.source, "target": target,
                   "condition": edge.condition} for target, _c in targets]
    for index, branch_ends in enumerate(ends):
        for source, condition in branch_ends:
            if index + 1 < len(heads):
                edges.append({"op": "add_edge", "source": source, "target": heads[index + 1],
                              "condition": condition})
            else:
                edges += [{"op": "add_edge", "source": source, "target": target,
                           "condition": condition if condition is not None else cond}
                          for target, cond in onward]
    build.add(edges)
    build.add([{"op": "remove_node", "id": fork.id}, {"op": "remove_node", "id": join.id}])
    return build.steps


def _expand_serialize(edit: dict, state: dict) -> list[dict]:
    return _serialize_steps(state, edit["fork"])


# ── branch ────────────────────────────────────────────────────────────────


def _pre_branch(edit: dict, state: dict) -> str | None:
    graph = _edits()._graph(state)
    fork = _require_step(graph, edit.get("fork"), "fork")
    if fork.kind != "fork":
        raise _Refused(f"{fork.id!r} is a {fork.kind}, not a fork; 'parallelize' runs steps "
                       "beside a step that is not one")
    _join_of(graph, fork)
    if len(fork.targets) >= MAX_BRANCHES:
        raise _Refused(f"fork {fork.id!r} already has {len(fork.targets)} branches")
    if edit.get("optional") is not None and not isinstance(edit.get("optional"), bool):
        raise _Refused("'optional' is true or false")
    spec = _branch_spec(edit, graph, fork, state)
    if spec.get("id") is not None and spec["id"] in graph.nodes:
        raise _Refused(f"node {spec['id']!r} already exists")
    return None


def _branch_spec(edit: dict, graph: Graph, fork: Node, state: Mapping[str, Any]) -> dict:
    first = next((graph.nodes[t] for t in fork.targets if t in graph.nodes
                  and graph.nodes[t].kind == "agent"), None)
    default = _copy_spec(first, state) if first is not None else {"kind": "agent",
                                                                   "text": NEUTRAL_STEP}
    given = {k: v for k, v in edit.items() if k not in ("op", "fork", "optional")}
    copied = default if given.get("text") is None and given.get("kind") is None else {}
    return _spec({**copied, **given}, default_kind=default["kind"],
                 default_text=default["text"] if given.get("kind") in (None, default["kind"])
                 else NEUTRAL_STEP)


def _expand_branch(edit: dict, state: dict) -> list[dict]:
    build = _Steps(state)
    graph = build.graph
    fork = graph.nodes[edit["fork"]]
    join = _join_of(graph, fork)
    preds = [e.source for e in graph.in_edges(fork.id)]
    there = set(_available(graph).get(fork.id, set()))
    view: set[tuple[str, str]] = set()
    for pred in preds:
        view |= _consumed(graph.nodes[pred], state) | _outputs(graph.nodes[pred])
    if not preds:
        view = _task_inputs(graph, state)
    spec = _branch_spec(edit, graph, fork, state)
    node_id = spec.get("id") or _fresh(graph, f"{fork.id}_branch")
    nodes = _step_nodes(graph, state, spec, node_id,
                        _seen(graph, state, there, view, spec.get("sees")))
    head, end = _chain(nodes)
    # Given at all, 'optional' is a setting the search can move: the branch's first step
    # binds it to a knob of its own (`Graph.optional_targets_of`), and what reads the
    # branch's output is guarded, since the knob may let the branch fail later.
    settable = edit.get("optional") is not None
    fork_state = fork.to_state()
    fork_state["targets"] = [*fork.targets, head]
    knob_steps: list[dict] = []
    set_steps: list[dict] = []
    if settable:
        knob = f"{head}__optional"
        from .knob import Knob

        knob_steps = [{"op": "add_knob", "name": knob, "spec": Knob(
            False, "bool", description=f"whether branch {head} may fail without failing the "
                                       "item").spec()}]
        set_steps = [{"op": "set_knob", "name": knob, "value": bool(edit["optional"])}]
        nodes[0] = {**nodes[0], "knobs": [*(nodes[0].get("knobs") or []), knob],
                    "field_bindings": [*(nodes[0].get("field_bindings") or []), ["optional", knob]]}
    join_state = join.to_state()
    join_state["sources"] = [*join.sources, end]
    build.add(knob_steps)
    build.add(_respec(build.state, fork.id, fork_state))
    build.add(_respec(build.state, join.id, join_state))
    build.add(_add_detached(nodes))
    build.add([{"op": "add_edge", "source": end, "target": join.id, "condition": None}])
    build.add(set_steps)
    _place_all(build, _new_facts(nodes), after=join.id, to=spec.get("to"), guarded=settable)
    return build.steps


# ── rewire ────────────────────────────────────────────────────────────────


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
    gather = f"{node.id}_inputs"
    helper = gather if gather in _owned(graph, node.id, state) else None
    there = set(_available(graph).get(helper or node.id, set()))
    seen = _seen(graph, state, there, set(), edit.get("sees", [TASK]))
    now = _consumed(node, state)
    if helper is not None:
        now = (now - {("fact", gather)}) | _consumed(graph.nodes[helper], state)
    if set(seen) == _readable(graph, state, now):
        raise _Refused(f"{node.id!r} already sees exactly that "
                       f"({', '.join(name for _k, name in seen) or 'nothing'}), so rewiring "
                       "would change nothing")
    if kind == "agent":
        knob = _task_knob(node)
        if knob is not None and len(graph.knob_consumers().get(knob, ())) > 1:
            raise _Refused(f"{node.id!r}'s prompt knob {knob!r} is shared with other steps")
        text = str(_knob_values(state).get(knob) if knob else node.prompt_template)
        keep = {name for k, name in seen if k == "fact"} | set(state.get("knobs") or {}) | {gather}
        if text.strip() and not _strip_unseen(text, keep).strip():
            raise _Refused(f"{node.id!r}'s whole prompt is a fact it would no longer see, so "
                           "rewiring would leave it no instructions; name that fact's step in "
                           "'sees', or give it your own text: "
                           f"{_replace_hint(node, state, edit.get('sees', [TASK]))}")
    return None


def _expand_rewire(edit: dict, state: dict) -> list[dict]:
    from .operators import collect_node

    build = _Steps(state)
    graph = build.graph
    node = graph.nodes[edit["step"]]
    gather = f"{node.id}_inputs"
    helper = gather if gather in _owned(graph, node.id, state) else None
    there = set(_available(graph).get(helper or node.id, set()))
    seen = _seen(graph, state, there, set(), edit.get("sees", [TASK]))
    facts = [name for kind, name in seen if kind == "fact"]
    files = [name for kind, name in seen if kind == "file"]
    new_state = node.to_state()
    if _kind_of(node) == "code":
        new_state.update({"reads": files, "inputs": facts})
        new_state.pop("required_reads", None)
        build.add(_respec(build.state, node.id, new_state))
        return build.steps
    knob = _task_knob(node)
    text = str(_knob_values(state).get(knob) if knob else node.prompt_template)
    keep = set(facts) | set(state.get("knobs") or {})
    if helper is not None:
        build.add(_bypass(build.state, helper))
    if files:
        keep.add(gather)
        facts.append(gather)
    new_text = _with_sections(_strip_unseen(text, keep), facts)
    for key in ("reads", "required_reads", "inputs"):
        new_state.pop(key, None)
    # Its prompt as a knob carries the new slots (the field too, so the knob's slot check
    # holds); a prompt that is not a knob is re-slotted in place: data wiring, no new text.
    new_state["prompt_template"] = new_text
    build.add(_respec(build.state, node.id, new_state))
    if files:
        build.add(_renamed(build.state, node.id, [gather], joins=False))
        build.add([{"op": "add_node", "position": {"before": node.id},
                    "node": collect_node(gather, files, gather,
                                         notes=f"Hands {node.id} the files it sees.")}])
    if knob is not None and new_text != text:
        build.add([{"op": "set_knob", "name": knob, "value": new_text}])
    return build.steps


# ── loop ──────────────────────────────────────────────────────────────────


def _judge(edit: dict) -> str:
    judge = edit.get("judge", "agent")
    if not isinstance(judge, str) or not (judge in JUDGE_KINDS or ":" in judge):
        raise _Refused(f"'judge' is one of {list(JUDGE_KINDS)} or a 'module:function' callable")
    return judge


def is_one_line_loop(edit: Mapping[str, Any]) -> bool:
    """Whether a ``loop`` edit is the one-line form (a step and a judge), not the written-out
    one (a ``target`` and a ``gate`` node object)."""
    return "gate" not in edit and "target" not in edit


def _pre_loop_line(edit: dict, state: dict) -> str | None:
    graph = _edits()._graph(state)
    node = _plain_step(graph, edit.get("step"), "loop")
    _sole_out(graph, node.id)
    _judge(edit)
    if edit.get("text") is not None and not isinstance(edit.get("text"), str):
        raise _Refused("'text' is a string")
    if edit.get("id") is not None and edit["id"] in graph.nodes:
        raise _Refused(f"node {edit['id']!r} already exists")
    _loop_cap(edit)
    if _back_to(graph, edit, node).id != node.id:
        _loop_cycle_valid(edit, state)
    return None


def _pass_of(graph: Graph, node_id: str) -> str | None:
    """The innermost data pass whose body holds ``node_id`` (``None``: the top level)."""
    bodies = [(len(body), data_id) for data_id, node in graph.nodes.items()
              if node.kind == "data" and node_id in (body := graph.data_body(data_id))]
    return min(bodies)[1] if bodies else None


def _forward_flow(graph: Graph) -> dict[str, list[str]]:
    """Each node's successors as a run walks them, retries left out: plain and verdict
    edges, a fork to its branch heads (not its own edge to its join, which a run reaches
    only through them), a branch's end to its join. A data node leads on past its pass
    (whose body is a scope of its own)."""
    from .loops import loop_edges

    retries = loop_edges(graph)
    out: dict[str, list[str]] = {key: [] for key in graph.nodes}
    for edge in graph.edges:
        source = graph.nodes.get(edge.source)
        if source is not None and source.kind == "fork" and source.targets:
            continue
        if edge.key() not in retries and edge.source in out and edge.target in out:
            out[edge.source].append(edge.target)
    for key, node in graph.nodes.items():
        if node.kind == "fork":
            out[key] += [t for t in node.targets if t in out]
        elif node.kind == "join":
            for source in node.sources:
                if source in out:
                    out[source].append(key)
    return out


def _reach(flow: Mapping[str, Sequence[str]], start: str, without: str | None = None
           ) -> set[str]:
    seen, todo = {start}, [start]
    while todo:
        for nxt in flow.get(todo.pop(), ()):
            if nxt not in seen and nxt != without:
                seen.add(nxt)
                todo.append(nxt)
    return seen


def _back_to(graph: Graph, edit: dict, node: Node) -> Node:
    """The step a loop sends back to: ``node`` itself, or the edit's ``back_to`` -- an
    agent (only an agent's prompt takes the judge's reason, on both runtimes) that every
    run passes before ``node``, in the same per-item pass and fork branch as ``node``, so
    a send-back re-runs exactly the steps from it to ``node`` (a fork in between whole)."""
    back = edit.get("back_to")
    if back is None or back == node.id:
        return node
    target = _plain_step(graph, back, "a loop's 'back_to'")
    where, here = _pass_of(graph, target.id), _pass_of(graph, node.id)
    if where != here:
        inside = f"inside the per-item pass {here!r}" if here else "outside any per-item pass"
        raise _Refused(f"'back_to' {target.id!r} is outside the pass {node.id!r} is in "
                       f"({node.id!r} is {inside}); a loop sends back only within its pass")
    flow = _forward_flow(graph)
    if target.id in _reach(flow, node.id):
        raise _Refused(f"'back_to' {target.id!r} comes after {node.id!r}; it names an earlier "
                       f"step, where the work goes back to when the judge after {node.id!r} "
                       "says reloop")
    start = _scope_start(graph, here)
    if target.id != start and node.id in _reach(flow, start, without=target.id):
        on_every = sorted(n for n in _reach(flow, start) - {node.id}
                          if graph.nodes[n].kind == "agent" and n != target.id
                          and (n == start or node.id not in _reach(flow, start, without=n)))
        raise _Refused(f"'back_to' {target.id!r} is not on every path to {node.id!r} (a branch "
                       "or a route's way passes it by); it names a step every run passes "
                       f"before {node.id!r}: " + (", ".join(on_every) or "none here"))
    from .validate import fork_branches

    for fork in (n for n in graph.nodes.values() if n.kind == "fork"):
        for branch in fork_branches(graph, fork, graph.fork_join(fork.id)):
            if node.id in branch and target.id not in branch:
                raise _Refused(f"{node.id!r} is inside a branch of fork {fork.id!r} and "
                               f"'back_to' {target.id!r} is not; a loop in a branch sends "
                               "back only within it (judge a step after the fork's join to "
                               "re-run the whole fork)")
            if target.id in branch and node.id not in branch:
                raise _Refused(f"'back_to' {target.id!r} is inside a branch of fork "
                               f"{fork.id!r} and {node.id!r} is not; a send-back from after "
                               "the join re-runs the whole fork, so it names a step before "
                               f"{fork.id!r}")
    if target.kind != "agent":
        nearest = _agents_after(graph, flow, target.id, node.id)
        raise _Refused(f"'back_to' {target.id!r} is {'code' if _is_code(target) else 'an op'}, "
                       "which has no prompt for the judge's reason; send back to the model "
                       "step that should act on it: "
                       + (", ".join(nearest) or f"none between it and {node.id!r}"))
    return target


def _scope_start(graph: Graph, data_id: str | None) -> str:
    """Where a run of the scope begins: the pass's entry, or the run's start."""
    return graph.nodes[data_id].subgraph_entry if data_id else graph.start


def _agents_after(graph: Graph, flow: Mapping[str, Sequence[str]], start: str, end: str
                  ) -> list[str]:
    """The first agents after ``start`` on the way to ``end`` (``end`` included)."""
    leads = {n for n in _reach(flow, start) if end in _reach(flow, n)}
    out, seen, todo = [], {start}, list(flow.get(start, ()))
    while todo:
        current = todo.pop(0)
        if current in seen or current not in leads:
            continue
        seen.add(current)
        if graph.nodes[current].kind == "agent":
            out.append(current)
        else:
            todo += list(flow.get(current, ()))
    return sorted(out)


def _loop_cycle_valid(edit: dict, state: dict) -> None:
    """Refused when the validator refuses the loop's cycle (V10, V12, V16) and did not
    refuse the graph before it."""
    from .validate import validate

    rules = ("V10", "V12", "V16")

    def issues(of: Mapping[str, Any]) -> list:
        return [issue for issue in validate(_edits()._graph(of), _knob_values(of),
                                            declared_knobs={}, resolve_calls=False)
                if issue.code in rules]

    before = {str(issue) for issue in issues(state)}
    new = [issue for issue in issues(_simulate(state, _expand_loop_line(edit, state)))
           if str(issue) not in before]
    if new:
        raise _Refused(f"the validator refuses the cycle back to {edit['back_to']!r}: {new[0]}")


def _loop_cap(edit: dict) -> int:
    """The loop's ``max_iterations`` (how often the judge may send the step back), the
    default when the edit leaves it out; refused outside the knob's bounds."""
    default, lo, hi = LOOP_ITERATIONS
    cap = edit.get("max_iterations", default)
    if isinstance(cap, bool) or not isinstance(cap, int) or not lo <= cap <= hi:
        raise _Refused(f"'max_iterations' is a whole number from {lo} to {hi} (how often the "
                       "judge may send the step back)")
    return cap


def _iterations_knob(gate_id: str, cap: int = LOOP_ITERATIONS[0]) -> tuple[str, dict]:
    from .knob import Knob

    _default, lo, hi = LOOP_ITERATIONS
    return f"{gate_id}__max_iterations", Knob(
        cap, "int", lo=lo, hi=hi,
        description=f"how often {gate_id} may send the work back").spec()


def _gate_state(gate_id: str, judge: str, text: str, seen: Sequence[tuple[str, str]], *,
                label: str, reads: Sequence[str] = (), iterations: bool = False,
                cap: int = LOOP_ITERATIONS[0]) -> dict[str, Any]:
    from refactory_dsh import snippet

    gate: dict[str, Any] = {"id": gate_id, "kind": "gate"}
    if judge == "agent":
        gate.update({"evaluator": snippet.VERDICT, "reads": list(reads),
                     "required_reads": list(reads)})
    elif judge == "code":
        gate.update({"evaluator": snippet.GATE, "code": text or _default_gate_code(label),
                     "inputs": [n for k, n in seen if k == "fact"],
                     "reads": [n for k, n in seen if k == "file"]})
    else:
        gate.update({"evaluator": judge, "reads": [n for k, n in seen if k == "file"]})
    if iterations:
        name, _spec_ = _iterations_knob(gate_id, cap)
        gate.update({"max_iterations": cap, "on_exhausted": "continue",
                     "knobs": [name], "field_bindings": [["max_iterations", name]]})
    return gate


def _default_gate_code(label: str) -> str:
    return ("def main(inp):\n"
            f"    return {{'verdict': {label!r}, 'reason': 'no check written yet'}}\n")


def _expand_loop_line(edit: dict, state: dict) -> list[dict]:
    edits = _edits()
    graph = edits._graph(state)
    node = graph.nodes[edit["step"]]
    back = _back_to(graph, edit, node).id
    judge = _judge(edit)
    there = _available(graph).get(node.id, set()) | _outputs(node)
    seen = _seen(graph, state, there, _consumed(node, state) | _outputs(node), edit.get("sees"))
    gate_id = edit.get("id") or _fresh(graph, f"{node.id}_check")
    cap = _loop_cap(edit)
    knob, knob_spec = _iterations_knob(gate_id, cap)
    steps: list[dict] = [{"op": "add_knob", "name": knob, "spec": knob_spec}]
    if judge == "agent":
        judge_id = _fresh(graph, f"{gate_id}_judge", (gate_id,))
        nodes = _step_nodes(graph, state, _spec({"kind": "agent",
                                                 "text": edit.get("text") or NEUTRAL_CHECK}),
                            judge_id, seen, verdict=("proceed", "reloop"),
                            settings_from=node if node.kind == "agent" else None)
        gate = _gate_state(gate_id, judge, "", seen, label="proceed",
                           reads=[f"{judge_id}.txt"], iterations=True, cap=cap)
        steps += edits._loop_steps(graph, node.id, gate, back)
        steps += _add_chain(nodes, {"before": gate_id})
    else:
        gate = _gate_state(gate_id, judge, edit.get("text") or "", seen, label="proceed",
                           iterations=True, cap=cap)
        steps += edits._loop_steps(graph, node.id, gate, back)
    return steps


# ── route ─────────────────────────────────────────────────────────────────


_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


def _routes(edit: dict, graph: Graph, anchor: Node) -> dict[str, Any]:
    to = edit.get("to")
    if to is None:
        succ = _sole_out(graph, anchor.id)
        if succ is None:
            raise _Refused(f"{anchor.id!r} has no next step to route to")
        return {"proceed": succ, "revise": {"kind": "agent", "text": NEUTRAL_REVIEW}}
    if not isinstance(to, dict) or not 2 <= len(to) <= MAX_ROUTES:
        raise _Refused(f"'to' maps 2 to {MAX_ROUTES} one-word labels to a step id or a new step")
    return to


def _pre_route(edit: dict, state: dict) -> str | None:
    graph = _edits()._graph(state)
    anchor = _plain_step(graph, edit.get("after"), "route")
    if _sole_out(graph, anchor.id) is None:
        raise _Refused(f"{anchor.id!r} is the last step of its pass or run; route after a step "
                       "that has a next step")
    _judge(edit)
    upstream = _upstream(graph, anchor.id) | {anchor.id}
    seen_labels: set[str] = set()
    for label, target in _routes(edit, graph, anchor).items():
        if not isinstance(label, str) or not _LABEL.fullmatch(label) or label.lower() == "halt":
            raise _Refused(f"route label {label!r} is not one word (and 'halt' ends the run)")
        if label.lower() in seen_labels:
            raise _Refused(f"route label {label!r} is given twice")
        seen_labels.add(label.lower())
        if isinstance(target, str):
            if target not in graph.nodes:
                raise _Refused(f"route {label!r} names no step {target!r}")
            if target in upstream:
                raise _Refused(f"route {label!r} leads back to {target!r}; use 'loop' to send "
                               "work back")
        else:
            _spec(target)
    return None


def _expand_route(edit: dict, state: dict) -> list[dict]:
    build = _Steps(state)
    graph = build.graph
    anchor = graph.nodes[edit["after"]]
    succ = _sole_out(graph, anchor.id)
    assert succ is not None
    judge = _judge(edit)
    routes = _routes(edit, graph, anchor)
    avail = _available(graph)
    there = avail.get(anchor.id, set()) | _outputs(anchor)
    view = _neighbour_view(graph, state, anchor)
    seen = _seen(graph, state, there, view, edit.get("sees"))
    gate_id = edit.get("id") or _fresh(graph, f"{anchor.id}_route")
    taken = [gate_id]
    if judge == "agent":
        judge_id = _fresh(graph, f"{gate_id}_judge", taken)
        nodes = _step_nodes(graph, state, _spec({"kind": "agent",
                                                 "text": edit.get("text") or NEUTRAL_ROUTE}),
                            judge_id, seen, verdict=list(routes))
        taken += [str(n["id"]) for n in nodes]
        build.add(_add_chain(nodes, {"after": anchor.id}))
        gate = _gate_state(gate_id, judge, "", seen, label=next(iter(routes)),
                           reads=[f"{judge_id}.txt"])
        build.add([{"op": "add_node", "node": gate, "position": {"after": judge_id}}])
    else:
        gate = _gate_state(gate_id, judge, edit.get("text") or "", seen,
                           label=next(iter(routes)))
        build.add([{"op": "add_node", "node": gate, "position": {"after": anchor.id}}])
    build.add([{"op": "remove_edge", "source": gate_id, "target": succ, "condition": None}])
    new_facts: list[str] = []
    for label, target in routes.items():
        if isinstance(target, str):
            build.add([{"op": "add_edge", "source": gate_id, "target": target,
                        "condition": label}])
            continue
        spec = _spec(target)
        node_id = spec.get("id") or _fresh(graph, f"{anchor.id}_{label}", taken)
        taken.append(node_id)
        nodes = _step_nodes(graph, state, spec, node_id,
                            _seen(graph, state, there, view, spec.get("sees")),
                            settings_from=anchor if anchor.kind == "agent" else None)
        taken += [str(n["id"]) for n in nodes]
        build.add(_add_detached(nodes))
        build.add([{"op": "add_edge", "source": gate_id, "target": nodes[0]["id"],
                    "condition": label},
                   {"op": "add_edge", "source": nodes[-1]["id"], "target": succ,
                    "condition": None}])
        new_facts += _new_facts(nodes)
    for fact in new_facts:
        build.add(_place(build.state, fact, readers=_readers_from(build.graph, succ, build.state),
                         guarded=True))
    return build.steps


# ── fallback ──────────────────────────────────────────────────────────────


def _fallback_spec(edit: dict, step: Node) -> dict:
    spec = _spec({k: v for k, v in edit.items() if k not in ("op", "step")})
    if spec["kind"] == "agent" and not spec.get("tools") and len(step.writes) > 1:
        raise _Refused(f"{step.id!r} writes {len(step.writes)} files, and a new agent step "
                       "without tools writes only its reply; give it a code step, or "
                       "'tools'")
    return spec


def _pre_fallback(edit: dict, state: dict) -> str | None:
    graph = _edits()._graph(state)
    node = _plain_step(graph, edit.get("step"), "fallback")
    if _pass_of(graph, node.id) is None:
        raise _Refused(f"{node.id!r} runs outside any per-item pass; a fallback answers for "
                       "a step that runs once per item")
    if graph.falls_back(node.id):
        raise _Refused(f"{node.id!r} has a fallback already, after "
                       f"{graph.fallback_gate(node.id)!r}")
    if not (node.writes or node.produces):
        raise _Refused(f"{node.id!r} has no outputs for a fallback to write")
    onward = _sole_out(graph, node.id)
    if onward is not None and graph.nodes[onward].kind == "join":
        raise _Refused(f"{node.id!r} ends a fork branch; insert a step after it first, or "
                       "give the fallback to a step inside the branch")
    spec = _fallback_spec(edit, node)
    _verdict_read(graph, node, spec)
    if spec.get("id") is not None and spec["id"] in graph.nodes:
        raise _Refused(f"node {spec['id']!r} already exists")
    return None


def _expand_fallback(edit: dict, state: dict) -> list[dict]:
    """A fallback gate after the step (`graph.FALLBACK_GATE_CALL`): ``proceed`` to what
    came next, ``fallback`` to the new step, which sees what the step sees, writes its
    outputs and leads to the same next step. A step that ends its pass first gets a no-op
    step after it (`LOOP_END_CALL`), as a loop's gate does, so the pass ends at a step."""
    from .graph import FALLBACK, FALLBACK_GATE_CALL, PROCEED
    from .loops import LOOP_END_CALL

    build = _Steps(state)
    graph = build.graph
    node = graph.nodes[edit["step"]]
    spec = _fallback_spec(edit, node)
    gate_id = _fresh(graph, f"{node.id}_failed")
    node_id = spec.get("id") or _fresh(graph, f"{node.id}_fallback", (gate_id,))
    there = _available(graph).get(node.id, set())
    seen = _seen(graph, state, there, _consumed(node, state), spec.get("sees"))
    nodes = _step_nodes(graph, state, spec, node_id, seen, writes=list(node.writes),
                        produces=list(node.produces),
                        settings_from=node if node.kind == "agent" else None)
    onward = _sole_out(graph, node.id)
    if onward is None:
        onward = _fresh(graph, f"{node.id}_done", (gate_id, *(str(n["id"]) for n in nodes)))
        build.add([{"op": "add_node", "position": {"after": node.id}, "node": {
            "id": onward, "kind": "op", "call": LOOP_END_CALL, "score_effect": "none",
            "notes": f"Ends what {node.id!r} ended, after its fallback gate {gate_id!r}."}}])
    build.add([{"op": "add_node", "position": {"detached": True}, "node": {
                   "id": gate_id, "kind": "gate", "evaluator": FALLBACK_GATE_CALL,
                   "notes": f"Sends the item to {node_id!r} when {node.id!r} failed."}},
               {"op": "remove_edge", "source": node.id, "target": onward, "condition": None},
               {"op": "add_edge", "source": node.id, "target": gate_id, "condition": None},
               {"op": "add_edge", "source": gate_id, "target": onward, "condition": PROCEED}])
    build.add(_add_detached(nodes))
    build.add([{"op": "add_edge", "source": gate_id, "target": nodes[0]["id"],
                "condition": FALLBACK},
               {"op": "add_edge", "source": nodes[-1]["id"], "target": onward,
                "condition": None}])
    return build.steps


def _fallback_parts(graph: Graph, step: str) -> tuple[str, str, set[str], list]:
    """A step with a fallback's gate, the step both ways rejoin at (the gate's ``proceed``
    target), the fallback's way (the steps from the gate's ``fallback`` edge up to the
    rejoin) and the edges that way rejoins by."""
    from .graph import FALLBACK, PROCEED

    gate = graph.fallback_gate(step)
    assert gate is not None, step
    out = {e.condition: e.target for e in graph.out_edges(gate)}
    onward = out[PROCEED]
    way: set[str] = set()
    pending = [out[FALLBACK]]
    rejoins = []
    while pending:
        current = pending.pop()
        if current == onward or current in way:
            continue
        way.add(current)
        for edge in graph.out_edges(current):
            if edge.target == onward:
                rejoins.append(edge)
            else:
                pending.append(edge.target)
    return gate, onward, way, rejoins


def _at_rejoin(graph: Graph, step: str, nodes: Sequence[dict[str, Any]]) -> list[dict]:
    """Primitive steps adding a new step's nodes where ``step``'s two ways (it answered, or
    its fallback did) rejoin: both lead to the new step, which leads where they led."""
    from .graph import PROCEED

    gate, onward, _way, rejoins = _fallback_parts(graph, step)
    head, end = _chain(nodes)
    steps = _add_detached(nodes)
    for source, condition in [(gate, PROCEED), *((e.source, e.condition) for e in rejoins)]:
        steps += [{"op": "remove_edge", "source": source, "target": onward,
                   "condition": condition},
                  {"op": "add_edge", "source": source, "target": head, "condition": condition}]
    steps.append({"op": "add_edge", "source": end, "target": onward, "condition": None})
    return steps


def _fallback_owner(graph: Graph, node_id: str) -> str | None:
    """The step with a fallback that ``node_id`` is, or whose fallback gate or fallback's
    way it is on; else None."""
    for step in sorted(graph.nodes):
        if graph.falls_back(step):
            gate, _onward, way, _rejoins = _fallback_parts(graph, step)
            if node_id in {step, gate, *way}:
                return step
    return None


def _unfallback(state: Mapping[str, Any], step: str) -> list[dict]:
    """Primitive steps taking ``step``'s fallback out, judged on ``state``: its gate and
    its way go, and ``step`` leads where they rejoined -- or, when the fallback added a
    no-op step to end the pass (`_expand_fallback`), ends the pass again."""
    from .loops import LOOP_END_CALL

    build = _Steps(state)
    graph = build.graph
    gate, onward, way, rejoins = _fallback_parts(graph, step)
    build.add([{"op": "remove_node", "id": node} for node in sorted(way)])
    build.add([{"op": "remove_node", "id": gate}])
    after = graph.nodes[onward]
    exits = [d.id for d in graph.nodes.values()
             if d.kind == "data" and d.subgraph_exit == onward]
    own_end = (after.kind == "op" and after.call == LOOP_END_CALL and exits
               and not graph.out_edges(onward)
               and {e.source for e in graph.in_edges(onward)} <= {gate, *(e.source for e in rejoins)})
    if own_end:
        build.add([{"op": "remove_node", "id": onward}])
        for data_id in exits:
            data = build.graph.nodes[data_id].to_state()
            build.add(_respec(build.state, data_id, {**data, "subgraph_exit": step}))
    else:
        build.add([{"op": "add_edge", "source": step, "target": onward, "condition": None}])
    return build.steps


# ── split (one-line form) and replace ─────────────────────────────────────


def is_one_line_split(edit: Mapping[str, Any]) -> bool:
    """Whether a ``split`` edit is the one-line form (step specs, or one step), not the
    written-out one (a ``target`` and two node objects with ids)."""
    if "target" not in edit:
        return True
    into = edit.get("into")
    return isinstance(into, list) and any(
        isinstance(n, dict) and ("text" in n or "id" not in n) for n in into)


def _split_specs(node: Node, edit: dict, state: Mapping[str, Any]) -> list[dict]:
    into = edit.get("into")
    if into is None:
        return [_copy_spec(node, state), {"kind": "agent", "text": NEUTRAL_REVIEW}]
    if not isinstance(into, list) or not 1 <= len(into) <= 2:
        raise _Refused("'into' is one step (replace) or two steps (split)")
    return [_spec(raw) for raw in into]


def _verdict_read(graph: Graph, node: Node, spec: Mapping[str, Any]) -> None:
    """Refused when a new code step with no text of its own would write what a verdict gate
    reads from ``node`` (a judge's reply): its default code joins its inputs, which never
    starts with a verdict, so the gate would fail on every run."""
    from refactory_dsh import snippet

    if spec.get("kind") != "code" or spec.get("text"):
        return
    for gate in graph.nodes.values():
        if gate.kind == "gate" and gate.evaluator == snippet.VERDICT \
                and set(gate.reads) & set(node.writes):
            raise _Refused(f"{gate.id!r} reads its verdict from what {node.id!r} writes; a code "
                           "step taking that over needs its own 'text' (code whose output "
                           "starts with the verdict), or make it an agent")


def _pre_split_line(edit: dict, state: dict) -> str | None:
    graph = _edits()._graph(state)
    node = _plain_step(graph, edit.get("step", edit.get("target")), "split")
    _sole_out(graph, node.id)
    specs = _split_specs(node, edit, state)
    if _forks_naming(graph, node.id):
        raise _Refused(f"{node.id!r} is a fork's branch or a join's source; insert a step before "
                       "or after it instead")
    last = specs[-1]
    if last["kind"] == "agent" and len(node.writes) != 1:
        raise _Refused(f"{node.id!r} writes {list(node.writes) or 'no file'}; a new agent writes "
                       "one file (its reply), so the step taking its place should be code")
    ends = _data_ends(graph, node.id)
    if len(specs) == 1 and "exit" in ends and ("entry" in ends or node.id == graph.start):
        raise _Refused(f"{node.id!r} both starts and ends its pass; split it into two steps")
    for spec in specs:
        if spec.get("id") is not None and spec["id"] in graph.nodes:
            raise _Refused(f"node {spec['id']!r} already exists")
    last = specs[-1]
    _verdict_read(graph, node, last)
    carried = _carried_knobs(last["kind"], last.get("text"), last.get("system"), _shared_knobs(node))
    frozen = [k for k in _frozen_settings(graph, state, {node.id}) if k not in carried]
    if frozen:
        raise _Refused(f"{node.id!r} alone reads the frozen knob(s) {frozen}, which keep their "
                       "value and declaration; split another step, or insert one beside it")
    return None


def _expand_split_line(edit: dict, state: dict) -> list[dict]:
    build = _Steps(state)
    graph = build.graph
    node = graph.nodes[edit.get("step", edit.get("target"))]
    specs = _split_specs(node, edit, state)
    there = set(_available(graph).get(node.id, set()))
    view = _consumed(node, state)
    taken: list[str] = []
    built: list[list[dict]] = []
    for index, spec in enumerate(specs):
        last = index == len(specs) - 1
        base = f"{node.id}_new" if len(specs) == 1 else f"{node.id}_{'ab'[index]}"
        node_id = spec.get("id") or _fresh(graph, base, taken)
        taken.append(node_id)
        if index == 1:
            made = built[0][-1]
            there = there | {("fact", n) for n in made.get("produces") or ()} | {
                ("file", n) for n in made.get("writes") or ()}
            view = view | {("fact", n) for n in made.get("produces") or ()}
        nodes = _step_nodes(graph, state, spec, node_id,
                            _seen(graph, state, there, view, spec.get("sees")),
                            writes=list(node.writes) if last else None,
                            produces=list(node.produces) if last else None,
                            settings_from=node if node.kind == "agent" else None,
                            knobs=_shared_knobs(node) if last else ())
        taken += [str(n["id"]) for n in nodes]
        built.append(nodes)
    if len(built) == 1:
        nodes = built[0]
        if "exit" in _data_ends(graph, node.id):
            build.add(_add_chain(nodes, {"after": node.id}))
            build.add([{"op": "add_edge", "source": e.source, "target": nodes[0]["id"],
                        "condition": e.condition} for e in graph.in_edges(node.id)])
        else:
            build.add(_add_chain(nodes, {"before": node.id}))
            build.add([{"op": "add_edge", "source": nodes[-1]["id"], "target": e.target,
                        "condition": e.condition} for e in graph.out_edges(node.id)])
        build.add([{"op": "remove_node", "id": node.id}])
        _drop_orphans(build)
        return build.steps
    first, second = built
    build.add(_add_chain(first, {"before": node.id}))
    build.add(_add_chain(second, {"after": node.id}))
    build.add([{"op": "remove_node", "id": node.id},
               {"op": "add_edge", "source": first[-1]["id"], "target": second[0]["id"],
                "condition": None}])
    _drop_orphans(build)
    return build.steps


# ── merge ─────────────────────────────────────────────────────────────────


def _merge_kind(edit: dict, a: Node, b: Node) -> str:
    return edit.get("kind") or ("agent" if "agent" in (a.kind, b.kind) else "code")


def _merge_text(edit: dict, a: Node, b: Node, kind: str, state: Mapping[str, Any]) -> str:
    """The merged step's text: the one given, else for an agent the two agents' prompts,
    with what ``a`` made named in words (the merged step now makes it itself, so it is no
    longer an input), after a line saying what a non-agent ``a`` did."""
    if edit.get("text"):
        return str(edit["text"])
    if kind != "agent":
        return ""
    made = sorted({name for _k, name in _outputs(a)})
    parts = [_text_of(n, state) for n in (a, b) if n.kind == "agent"]
    if a.kind != "agent":
        parts.insert(0, f"First do what step {a.id!r} did, which made {', '.join(made) or 'nothing named'}.")
    text = "\n\n".join(part for part in parts if part)
    for name in made:
        text = re.sub(r"\{\{\s*" + re.escape(name) + r"\s*\}\}", name, text)
    return text


def _pre_merge(edit: dict, state: dict) -> str | None:
    graph = _edits()._graph(state)
    ids = edit.get("steps")
    if not isinstance(ids, list) or len(ids) != 2 or not all(isinstance(i, str) for i in ids):
        raise _Refused("'steps' names two consecutive steps")
    a, b = (_plain_step(graph, i, "merge") for i in ids)
    _chain_of(graph, [a.id, b.id])
    _sole_out(graph, b.id)
    for node in (a, b):
        if _forks_naming(graph, node.id):
            raise _Refused(f"{node.id!r} is a fork's branch or a join's source")
        if _uses_tools(node):
            raise _Refused(f"{node.id!r} uses tools; merge steps that do not")
    readers = sorted(n.id for n in graph.nodes.values()
                     if n.id not in (a.id, b.id) and _consumed(n, state) & (_outputs(a) - _outputs(b)))
    if readers:
        raise _Refused(f"{', '.join(readers)} read(s) what {a.id!r} makes, which a merged step "
                       "would no longer make")
    kind = _merge_kind(edit, a, b)
    if kind == "agent" and len(b.writes) != 1:
        raise _Refused(f"{b.id!r} writes {list(b.writes) or 'no file'}; a merged agent writes "
                       "one file")
    if (graph.start == a.id or "entry" in _data_ends(graph, a.id)) \
            and "exit" in _data_ends(graph, b.id):
        raise _Refused("the two steps start and end the same pass or run")
    _spec({**edit, "kind": kind})
    carried = _carried_knobs(kind, _merge_text(edit, a, b, kind, state), edit.get("system"),
                             _shared_knobs(a, b))
    frozen = [k for k in _frozen_settings(graph, state, {a.id, b.id}) if k not in carried]
    if frozen:
        raise _Refused(f"{a.id!r} and {b.id!r} alone read the frozen knob(s) {frozen}")
    if edit.get("id") is not None and edit["id"] in graph.nodes:
        raise _Refused(f"node {edit['id']!r} already exists")
    return None


def _expand_merge(edit: dict, state: dict) -> list[dict]:
    build = _Steps(state)
    graph = build.graph
    a, b = (graph.nodes[i] for i in edit["steps"])
    kind = _merge_kind(edit, a, b)
    a_made = {name for _k, name in _outputs(a)}
    text = _merge_text(edit, a, b, kind, state)
    there = set(_available(graph).get(a.id, set()))
    view = _consumed(a, state) | {key for key in _consumed(b, state) if key[1] not in a_made}
    spec = _spec({**{k: v for k, v in edit.items() if k not in ("op", "steps")},
                  "kind": kind, "text": text})
    node_id = spec.get("id") or _fresh(graph, f"{a.id}_{b.id}")
    nodes = _step_nodes(graph, state, spec, node_id,
                        _seen(graph, state, there, view, spec.get("sees")),
                        writes=list(b.writes), produces=list(b.produces),
                        settings_from=a if a.kind == "agent" else (b if b.kind == "agent" else None),
                        knobs=_shared_knobs(a, b))
    if "exit" in _data_ends(graph, b.id):
        build.add(_add_chain(nodes, {"after": b.id}))
        build.add([{"op": "add_edge", "source": e.source, "target": nodes[0]["id"],
                    "condition": e.condition} for e in graph.in_edges(a.id)])
    else:
        build.add(_add_chain(nodes, {"before": a.id}))
        build.add([{"op": "add_edge", "source": nodes[-1]["id"], "target": e.target,
                    "condition": e.condition} for e in graph.out_edges(b.id)])
    build.add([{"op": "remove_node", "id": a.id}, {"op": "remove_node", "id": b.id}])
    _drop_orphans(build)
    return build.steps


# ── registration ──────────────────────────────────────────────────────────


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
        try:
            steps = expand(edit, state)
        except _Refused as exc:
            raise _edits().EditError(str(exc), code="E-precondition") from exc
        return steps + _settle_optional(state, steps)
    return run


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
        build.add(_respec(build.state, head, again))
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
            build.add(_respec(build.state, fresh[0], moved))
            build.add([{"op": "set_knob", "name": new_knob, "value": value}])
        if head in build.graph.nodes:
            if knob in (build.state.get("knobs") or {}) and value != spec.get("default"):
                # Back to its default while still read, so the respec has nothing to restore.
                build.add([{"op": "set_knob", "name": knob, "value": spec.get("default")}])
            kept = build.graph.nodes[head].to_state()
            kept["knobs"] = [k for k in kept.get("knobs") or [] if k != knob]
            kept["field_bindings"] = [pair for pair in kept.get("field_bindings") or []
                                      if pair[0] != "optional"]
            build.add(_respec(build.state, head, kept))
        if knob in (build.state.get("knobs") or {}) and knob not in build.graph.knob_consumers():
            build.add([{"op": "remove_knob", "name": knob}])
    return build.steps[len(steps):]


#: Where a new step's output goes (`_place`): one phrase for every operator that says it.
REACHES = ("its output goes into the nearest prompt {where} that can take it -- an agent's own "
           "prompt knob, or the one text knob an op fills facts into -- or where 'to' says; "
           "with neither, nothing reads it until you add a reader")

#: The field a new agent step names its own route in (`Node.route`).
ROUTE_FIELD = "route"


def route_field_line(offered: Mapping[str, Sequence[str]]) -> str:
    """The request's one line for ``route``: where it goes and the run's values for it
    (`workers.StepRoutes.offered`)."""
    values = "; ".join(f"{key} {', '.join(vals)}" for key, vals in offered.items())
    return (f"- `{ROUTE_FIELD}` on a new agent step (`insert`, `branch`, `merge`, or a step in "
            "'into', 'new' or 'to'): its own model, effort or output cap, any of them, each "
            f"one of this run's -- {values}; left out, it runs on its neighbour's. E.g. "
            + json.dumps({ROUTE_FIELD: {key: vals[0] for key, vals in offered.items()}}))


#: Each operator: its catalogue fields, its one line for the request, and one example
#: with placeholders (``<step>``, ``<next step>``, ``<fork>``, ``<gate>``).
OPERATORS: dict[str, dict[str, Any]] = {
    "insert": {
        "reads_as": ('step',),
        "fields": ("after", "before", "kind", "text", "system", "sees", "to", "id", "call",
                   "reads", "tools", "takes_over", "route"),
        "line": "a new step after (or before) a step: 'kind' agent (text = its prompt; "
                "'system' its system prompt), code (text = Python) or op (with 'call'); it "
                "sees what that step sees plus its outputs; " + REACHES.format(where="after it")
                + "; 'tools' (an agent): e.g. " + json.dumps(list(SEALED_TOOLS))
                + "; runs sealed: only the files it sees, no network"
                + "; 'takes_over' an earlier step: the new step writes that step's outputs, "
                "so what read them reads the new step's (a choice among attempts reaching "
                "what is scored), and that step's own go under new names",
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
        "fields": ("fork", "kind", "text", "sees", "optional", "id", "to", "route"),
        "line": "a new step as another branch of a fork ('optional': true lets it fail "
                "without failing the item; 'sees' names its inputs: \"task\", step ids, or "
                "single fact or file names there before the fork); "
                + REACHES.format(where="after the join"),
        "example": {"op": "branch", "fork": "<fork>", "kind": "agent",
                    "text": "<what this branch does>"},
    },
    "rewire": {
        "reads_as": ('sees', 'told'),
        "fields": ("step", "sees"),
        "line": f"change what a step sees: \"task\" ({TASK_MEANING}), step ids (their "
                "outputs), or single fact or file names there before it (that input alone)",
        "example": {"op": "rewire", "step": "<step>", "sees": ["task"]},
    },
    "loop": {
        "reads_as": ('loop',),
        "fields": ("step", "back_to", "judge", "text", "sees", "max_iterations", "id", "target",
                   "gate"),
        "line": "a judge after a step that can send it back with its reason -- or, with "
                "'back_to', send the work back to an earlier agent step, so every step from "
                "there to the judge runs again (a fork in between whole); 'back_to' is one "
                "every run passes before the step, in the same per-item pass and fork "
                "branch: 'judge' agent "
                "(text = what to check; its answer starts with proceed or reloop, then why), "
                "code (text = Python returning {'verdict': 'proceed' or 'reloop', 'reason': "
                "why}) or a callable. The judge sees what the step read and wrote, unless "
                "'sees' says otherwise. The reason reaches the step sent back; it is sent "
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
        "fields": ("after", "judge", "text", "to", "sees", "id"),
        "line": "a judge after a step chooses one labelled next step ('to': label -> step id "
                "or a new step); the branches rejoin",
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
        "fields": ("steps", "kind", "text", "id", "route"),
        "line": "two consecutive steps as one, with the first's inputs and the second's "
                "outputs",
        "example": {"op": "merge", "steps": ["<step>", "<next step>"],
                    "text": "<what the merged step does>"},
    },
    "fallback": {
        "reads_as": ('fallback',),
        "fields": ("step", "kind", "text", "system", "sees", "id", "call", "reads", "tools",
                   "route"),
        "line": "a step run instead when 'step' (one run per item) fails -- it raises, "
                "crashes past its retries or hits its output or prompt limit; never for time or a "
                "harness fault: it writes that step's outputs, so what reads them still "
                "gets an answer; 'kind' as in insert; it sees what 'step' sees",
        "example": {"op": "fallback", "step": "<step>", "kind": "agent",
                    "text": "<a simpler way to the same output>"},
    },
}


def _resolved_texts(edit: dict, state: dict) -> dict:
    """``edit`` with each step text that is exactly a knob pointer (`nn.edits.KNOB_POINTER`,
    copied from the request's node reference) replaced by that knob's current text: the
    edit's own 'text', and the 'text' of each new step under 'into', 'new' and 'to'. A
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
    return out


def _outputs_read(state: Mapping[str, Any], node_id: str) -> dict[str, set[tuple[str, str]]]:
    """Which of ``node_id``'s outputs each other step takes in (`validate.consumed`: a
    ``{{slot}}`` in a prompt or a bound or fills-facts knob, an input, a read)."""
    from .validate import consumed

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
    """The steps ``after`` adds to ``before`` whose outputs no step takes in. A step with
    no step after it (the end of a data pass or of the run) is not one: what it leaves is
    the Task's or the run's result."""
    old = set(((before.get("graph") or {}).get("nodes") or {}))
    nodes = (after.get("graph") or {}).get("nodes") or {}
    graph = _edits()._graph(after)
    return [node_id for node_id in sorted(set(nodes) - old)
            if nodes[node_id].get("kind") in ("agent", "op")
            and (nodes[node_id].get("produces") or nodes[node_id].get("writes"))
            and graph.out_edges(node_id)
            and not _outputs_read(after, node_id)]


def unread_by_bundle(parent: Mapping[str, Any], states: Sequence[Mapping[str, Any]],
                     edits: Sequence[Mapping[str, Any]]) -> str | None:
    """Why a bundle's edits leave a step it adds with no reader, when an edit of the same
    bundle removed the last one: ``states[i]`` is the state after ``edits[i]``. A step
    added without a reader is not refused here (a Task may read its file); one whose
    reader a later edit removed is, naming that edit and what it took out."""
    if not states:
        return None
    for node_id in unread_new_steps(parent, states[-1]):
        last = max((i for i, state in enumerate(states[:-1]) if _outputs_read(state, node_id)),
                   default=None)
        if last is None:
            continue
        removed = last + 1
        lost = sorted(set().union(*_outputs_read(states[last], node_id).values()))
        what = ", ".join("{{" + name + "}}" if kind == "fact" else name for kind, name in lost)
        readers = ", ".join(sorted(_outputs_read(states[last], node_id)))
        edit = edits[removed]
        named = " ".join(str(edit.get(k)) for k in ("op", "name", "step", "id") if edit.get(k))
        how = " value" if edit.get("op") == "set_knob" and "value" in edit else ""
        return (f"edit {removed + 1} ({named}{how}) removes {what}, the only read of the "
                f"output of new step {node_id!r} (by {readers}); patch the text so it keeps "
                f"{what}, or keep the slot")
    return None


def _needed(node: Node, values: Mapping[str, Any]) -> set[tuple[str, str]]:
    """What a step cannot run without. An agent's prompt is read here: its reads and the
    ``{{slots}}`` it needs filled, a slot inside ``{% if name %}`` being optional, as the
    validator holds it (V15, `validate.template_facts`). An op or a gate is code that
    cannot be read, so it needs everything it takes in (its reads and inputs, which it
    may need though it declares them optional); only the slots of the knobs it fills are
    read, guards and all."""
    from .validate import consumed, knob_slot_facts, template_facts

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
    from .operators import made_by
    from .validate import available_inputs

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
STEP_FIELDS = ("kind", "text", "system", "sees", "id", "to", "call", "reads", "tools", "route")


def _tools_refusal(tools: Any) -> str | None:
    """Why ``tools`` is not a list of the tools a new step may use, or ``None``."""
    if (not isinstance(tools, list) or not tools
            or any(t not in SEALED_TOOLS for t in tools) or len(set(tools)) != len(tools)):
        return (f"'tools' lists the tools a new step uses, from {', '.join(SEALED_TOOLS)}; "
                f"{tools!r} is not such a list")
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
    for spec in [edit, *specs]:
        if not isinstance(spec, Mapping):
            continue
        if spec is not edit:
            extra = sorted(k for k in spec if k not in STEP_FIELDS and not k.startswith("_"))
            if extra:
                return (f"a new step does not take {', '.join(repr(k) for k in extra)}; it "
                        f"would be ignored. A step's fields: {', '.join(STEP_FIELDS)}")
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
        unknown = _unknown_fields(edit)
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


# ── what an operator did, for the candidate ───────────────────────────────


def wiring_notes(edit: Mapping[str, Any], before: Mapping[str, Any],
                 after: Mapping[str, Any]) -> list[str]:
    """What a core operator did beyond the steps it names: the steps it added and removed,
    and the prompts of existing steps it put a new step's output into, so the candidate
    says where an output went."""
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
    unread: list[str] = []
    try:
        graph = _edits()._graph(after)
        taken_in = set().union(*(_consumed(n, after) for n in graph.nodes.values()))
        for node_id in added:
            node = graph.nodes[node_id]
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
    return [f"{edit.get('op')}: " + "; ".join(parts)] if parts else []


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
    from .operators import SELECT_CALL

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
        from .operators import made_by

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
