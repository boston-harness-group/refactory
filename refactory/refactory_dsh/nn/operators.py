"""Purpose-free wiring blocks: two plain ops, and the helpers that wire a step's data.

Nothing here knows what a step is for. The two ops are ordinary node callables, run the
same way on both runtimes (the walker in its forkserver child, the spine through
``python -m refactory_dsh.run``), so a graph that uses them needs nothing new from either:

- ``collect`` (`COLLECT_CALL`): reads the files its node declares in ``reads`` (each one
  optional) and sets its one declared fact to their text, one header per file. A file
  that is not there is said to be missing rather than failing the step, and a long or
  binary file is shown as its path and size. This is how files reach a model step as one
  ``{{fact}}``.
- ``file_text`` (`FILE_TEXT_CALL`): sets its one declared fact to the whole text of its one
  declared read, which it requires (both runtimes refuse to run it while the file is
  missing). No header and no length cap: this is what a ``{{name}}`` naming a file means,
  and what the attempt path wires in for one (`optim.repair`).
- ``select`` (`SELECT_CALL`): reads a label from its first declared read (the first word
  or the last line, as a verdict gate reads one) and copies the declared read whose name
  that label names to its first write (and its fact, when it declares one).

The helpers describe and wire data without a model:

- `made_by`: which steps make each fact and file.
- `model_free_inputs`: the facts and files no model step's output reaches -- the task's
  own inputs, and what plain ops make from them alone -- so a step can be made to see
  the task without seeing another step's work.
- `collect_node` / `file_text_node`: the node state of a ``collect`` step for a set of
  files, or of a ``file_text`` step for one.
- `place_output`: puts a step's fact where the next model step downstream fills facts
  (its prompt knob), and says where.

All of it is static and works from what nodes declare: an op that opens a file it does
not declare is not seen.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .graph import Graph, Node

#: The ``collect`` op: files to one fact.
COLLECT_CALL = "refactory_dsh.nn.operators:collect"
#: The ``file_text`` op: one file's whole text as one fact.
FILE_TEXT_CALL = "refactory_dsh.nn.operators:file_text"
#: The ``select`` op: copy the file a label names.
SELECT_CALL = "refactory_dsh.nn.operators:select"
#: Past this many characters ``collect`` shows a file as its path and size, not its text.
COLLECT_TEXT_CHARS = 40000


# ── the ops ───────────────────────────────────────────────────────────────


def _declared_fact(ctx: Any, op: str) -> str | None:
    facts = tuple(getattr(ctx, "_declared_facts", ()) or ())
    if len(facts) > 1:
        from .executor import ExecutionError

        raise ExecutionError(f"{op} step {ctx.node_id!r} declares {len(facts)} facts; it sets "
                             "one, so declare at most one in `produces`")
    return facts[0] if facts else None


def _section(ctx: Any, name: str) -> str:
    """One file as ``collect`` shows it: a header, then its text, or why not its text."""
    if not ctx.exists(name):
        return f"### {name}\n(not written: the step that writes it did not run or wrote nothing)"
    path: Path = ctx.path(name)
    if path.is_dir():
        return f"### {name}\n(a directory, not shown)"
    size = path.stat().st_size
    try:
        text = ctx.read_text(name)
    except UnicodeDecodeError:
        return f"### {name}\n(a binary file at {name}, {size} bytes; not shown)"
    if len(text) > COLLECT_TEXT_CHARS:
        return (f"### {name}\n(a file at {name}, {size} bytes, {len(text)} characters: "
                f"longer than {COLLECT_TEXT_CHARS}, so not shown)")
    return f"### {name}\n{text.rstrip()}"


def collect(ctx: Any) -> dict[str, Any]:
    """`COLLECT_CALL`: each declared read, in order, under its own header, as the node's
    one declared fact. Nothing is required: a missing file is said to be missing."""
    fact = _declared_fact(ctx, "collect")
    if fact is None:
        from .executor import ExecutionError

        raise ExecutionError(f"collect step {ctx.node_id!r} declares no fact; it sets one, "
                             "so declare it in `produces`")
    return {fact: "\n\n".join(_section(ctx, name) for name in ctx.declared_reads)}


def file_text(ctx: Any) -> dict[str, Any]:
    """`FILE_TEXT_CALL`: the node's one declared read, whole, as its one declared fact.
    Refused for any other shape, or a file that is not text."""
    from .executor import ExecutionError

    fact = _declared_fact(ctx, "file_text")
    reads = list(ctx.declared_reads)
    if fact is None or len(reads) != 1:
        raise ExecutionError(f"file_text step {ctx.node_id!r} reads one file and sets one fact: "
                             "declare exactly one read and one fact")
    try:
        return {fact: ctx.read_text(reads[0])}
    except UnicodeDecodeError as exc:
        raise ExecutionError(f"file_text step {ctx.node_id!r}: {reads[0]!r} is not text") from exc


def _stem(name: str) -> str:
    return Path(name).stem


def select(ctx: Any) -> dict[str, Any]:
    """`SELECT_CALL`: the label in the first declared read names one of the other declared
    reads (by its file name or its name without the extension, in any case); that file's
    text is written to the node's first write and set as its fact, when it declares one.
    A missing label file, a label naming no file, or a named file that is missing fails
    the step, naming what it could choose from."""
    from refactory_dsh.snippet import parse_verdict

    from .executor import ExecutionError

    reads = list(ctx.declared_reads)
    writes = list(getattr(ctx, "_declared", ()) or ())
    fact = _declared_fact(ctx, "select")
    if len(reads) < 2 or not writes:
        raise ExecutionError(f"select step {ctx.node_id!r} reads a label file and the files "
                             "to choose from, and writes the chosen one: declare at least two "
                             "reads and one write")
    label_file, alternatives = reads[0], reads[1:]
    labels = {}
    for name in alternatives:
        labels.setdefault(_stem(name), name)
        labels.setdefault(name, name)
    if not ctx.exists(label_file):
        raise ExecutionError(f"select step {ctx.node_id!r}: its label file {label_file!r} was "
                             "not written")
    found = parse_verdict(ctx.read_text(label_file), sorted(labels))
    if found is None:
        raise ExecutionError(f"select step {ctx.node_id!r}: {label_file!r} names none of "
                             f"{sorted(labels)} as its first word or alone on its last line")
    chosen = labels[found[0]]
    if not ctx.exists(chosen):
        raise ExecutionError(f"select step {ctx.node_id!r}: the chosen file {chosen!r} was not "
                             "written")
    text = ctx.read_text(chosen)
    ctx.write_text(writes[0], text)
    return {fact: text} if fact else {}


# ── what a graph's data is made of ────────────────────────────────────────


def made_by(graph: Graph) -> dict[tuple[str, str], list[str]]:
    """Each ``("fact", name)`` and ``("file", name)`` a node declares making, with the ids
    of the nodes that make it (sorted)."""
    out: dict[tuple[str, str], set[str]] = {}
    for node in graph.nodes.values():
        for name in node.writes:
            out.setdefault(("file", name), set()).add(node.id)
        for name in node.produces:
            out.setdefault(("fact", name), set()).add(node.id)
    return {key: sorted(ids) for key, ids in out.items()}


def model_free_inputs(graph: Graph, knob_values: Mapping[str, Any] | None = None
                      ) -> set[tuple[str, str]]:
    """The facts and files no model step's output reaches, as nodes declare them
    (``knob_values``: the knobs' values by name, for the slots in bound prompts).

    A name carries model output when an agent makes it, or when a node that takes in
    (`validate.consumed`) a name carrying model output makes it, to a fixpoint. Every
    other name a node makes is model-free: what a data node gives each item, and what
    plain ops build from that and from knobs alone. Static and conservative: a name
    made by two steps, one of them downstream of a model, carries model output.
    """
    from .validate import consumed

    values = dict(knob_values or {})
    makers = made_by(graph)
    carrying: set[tuple[str, str]] = set()
    takes = {node_id: consumed(node, values)
             for node_id, node in graph.nodes.items()}
    changed = True
    while changed:
        changed = False
        for node_id, node in graph.nodes.items():
            if node.kind != "agent" and not (takes[node_id] & carrying):
                continue
            made = {("file", n) for n in node.writes} | {("fact", n) for n in node.produces}
            if not made <= carrying:
                carrying |= made
                changed = True
    return set(makers) - carrying


# ── helpers an edit uses to wire data ─────────────────────────────────────


def fact_name(name: str) -> str:
    """A file name as a fact name: its stem, with anything not a word character as ``_``
    (``draft notes.md`` -> ``draft_notes``)."""
    out = re.sub(r"\W", "_", _stem(name)).strip("_") or "file"
    return out if not out[0].isdigit() else f"f_{out}"


def collect_node(node_id: str, files: Sequence[str], fact: str, *, notes: str = "") -> dict[str, Any]:
    """The node state of a ``collect`` step that hands ``files`` to later steps as the
    fact ``fact`` (each file optional)."""
    return {"id": node_id, "kind": "op", "call": COLLECT_CALL, "reads": list(files),
            "produces": [fact], "score_effect": "none",
            "notes": notes or f"Hands {', '.join(files)} to later steps as {{{{{fact}}}}}."}


def file_text_node(node_id: str, file: str, fact: str, *, notes: str = "") -> dict[str, Any]:
    """The node state of a ``file_text`` step that hands ``file``'s whole text to later
    steps as the fact ``fact`` (the file required)."""
    return {"id": node_id, "kind": "op", "call": FILE_TEXT_CALL, "reads": [file],
            "required_reads": [file], "produces": [fact], "score_effect": "none",
            "notes": notes or f"Hands the whole text of {file} to later steps as {{{{{fact}}}}}."}


def _template_knob(node: Node) -> str | None:
    """The knob an agent's task prompt is bound to, or ``None`` when it is not a knob."""
    for field_name, knob in node.field_bindings:
        if field_name == "prompt_template":
            return knob
    return None


def _next_readers(graph: Graph, start: str) -> list[str]:
    """Model steps downstream of ``start``, nearest first (breadth first, through fork
    branches and gate verdicts, not back along a reloop)."""
    from .loops import loop_edges

    retries = {(source, target) for source, target, _c in loop_edges(graph)}
    seen, order, queue = {start}, [], [start]
    while queue:
        current = queue.pop(0)
        node = graph.nodes.get(current)
        if node is None:
            continue
        following = [e.target for e in graph.out_edges(current) if (current, e.target) not in retries]
        if node.kind == "fork":
            following = [*node.targets, *following]
        for target in following:
            if target in seen or target not in graph.nodes:
                continue
            seen.add(target)
            if graph.nodes[target].kind == "agent":
                order.append(target)
            queue.append(target)
    return order


def place_output(state: Mapping[str, Any], fact: str, *, after: str, to: str | None = None,
                 heading: str = "") -> tuple[dict[str, Any], str]:
    """``state`` with ``{{fact}}`` added to the prompt knob of the model step that will
    read it, and where it went (for the candidate's summary).

    The reader is ``to`` (a step id, or the prompt knob of an agent), else the nearest
    agent downstream of ``after`` whose task prompt is a knob. The fact goes at the end
    of that knob under a heading (``heading``, or the fact's name). Refused
    (`edits.EditError`, ``E-wiring``) when there is no such reader, or the reader already
    names the fact. Goes through ``set_knob``, so every check a prompt edit gets holds.
    """
    from .edits import EditError, _graph, apply

    graph = _graph(state)
    if after not in graph.nodes:
        raise EditError(f"no step {after!r} to place {fact!r} after", code="E-wiring")
    knob: str | None = None
    reader: str | None = None
    if to is not None:
        if to in graph.nodes:
            reader, knob = to, _template_knob(graph.nodes[to])
            if graph.nodes[to].kind != "agent" or knob is None:
                raise EditError(f"step {to!r} has no prompt knob to place {{{{{fact}}}}} in "
                                "(only an agent whose task prompt is a knob fills facts there)",
                                code="E-wiring")
        else:
            owners = [n.id for n in graph.nodes.values()
                      if n.kind == "agent" and _template_knob(n) == to]
            if not owners:
                raise EditError(f"{to!r} is neither a step nor an agent's prompt knob",
                                code="E-wiring")
            reader, knob = owners[0], to
    else:
        for candidate in _next_readers(graph, after):
            knob = _template_knob(graph.nodes[candidate])
            if knob is not None:
                reader = candidate
                break
    if reader is None or knob is None:
        raise EditError(f"no model step after {after!r} has a prompt knob to place "
                        f"{{{{{fact}}}}} in; name one with 'to'", code="E-wiring")
    current = str(((state.get("knobs") or {}).get(knob) or {}).get("value") or "")
    if re.search(r"\{\{\s*" + re.escape(fact) + r"\s*\}\}", current):
        raise EditError(f"{reader!r}'s prompt knob {knob!r} already fills {{{{{fact}}}}}",
                        code="E-wiring")
    title = heading or fact.replace("_", " ").strip().capitalize()
    value = f"{current.rstrip()}\n\n## {title}\n{{{{{fact}}}}}".lstrip()
    out = apply({"op": "set_knob", "name": knob, "value": value}, state)
    return out, f"{{{{{fact}}}}} placed in {reader!r}'s prompt knob {knob!r}"
