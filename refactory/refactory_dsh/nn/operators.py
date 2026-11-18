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
- ``elements`` (`MAP_CALL`): splits a list (its first declared read, else its first input
  fact; `list_elements`) into one prompt per element -- its one text knob, filled from
  the run's facts, then the element -- written to its i-th write; a ``map``'s branch
  reads its own (`map_of`).
- ``step_failed`` (`graph.FALLBACK_GATE_CALL`): a fallback gate's verdict, ``fallback``
  when the step before it is missing because it failed (``items.missing_steps``) and not
  because it ran out of time (`node_retry.TIMEOUT_KINDS`), else ``proceed``.

The helpers describe and wire data without a model:

- `made_by`: which steps make each fact and file.
- `carried_knobs` / `task_inputs`: which knobs' text each op output carries (declared in
  ``knob_outputs``, else all of the op's knobs), and the task's own input, once, as the
  model steps receive it -- what ``"sees": ["task"]`` gives, so a step can be made to see
  the task without another step's work or instructions.
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
#: The ``elements`` op: a list to one prompt per element, for a ``map``'s branches.
MAP_CALL = "refactory_dsh.nn.operators:elements"
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


#: A line that starts a list item: a bullet, or a number or letter then ``.`` or ``)``.
LIST_ITEM = re.compile(r"^(?P<indent>\s*)(?:[-*+\u2022]|\(?(?:\d+|[A-Za-z])[.)])\s+\S")
#: A code fence (``` or ~~~, with a language or none) and what it holds.
FENCE = re.compile(r"^(```|~~~)[\w-]*[ \t]*\n(.*?)\n?\1[ \t]*$", re.DOTALL | re.MULTILINE)


def _json_array(text: str) -> list[Any] | None:
    """The first JSON array in ``text`` that parses and is not empty, or None."""
    import json

    decoder = json.JSONDecoder()
    for found in re.finditer(r"\[", text):
        try:
            value, _end = decoder.raw_decode(text, found.end() - 1)
        except ValueError:
            continue
        if isinstance(value, list) and value:
            return value
    return None


def _list_items(lines: Sequence[str]) -> list[str] | None:
    """The items of a markdown, numbered or lettered list in ``lines`` (two or more at the
    outermost indent), or None: lines before the first are an introduction; an indented
    line, a nested item among them, stays with its item; a blank line followed by text that
    is neither an item nor indented ends the list (what follows is an outro)."""
    marks = [(i, len(m.group("indent"))) for i, line in enumerate(lines)
             if (m := LIST_ITEM.match(line))]
    if not marks:
        return None
    top = min(indent for _i, indent in marks)
    if sum(indent == top for _i, indent in marks) < 2:
        return None
    items: list[str] = []
    blank = False
    for line in lines:
        mark = LIST_ITEM.match(line)
        if mark and len(mark.group("indent")) == top:
            items.append(line.strip())
        elif not items:
            continue
        elif not line.strip():
            blank = True
            continue
        elif blank and len(line) - len(line.lstrip()) <= top:
            break
        else:
            # A nested item keeps its indent under its parent; a wrapped line does not.
            items[-1] += "\n" + (line[top:].rstrip() if mark else line.strip())
        blank = False
    return items


def list_elements(raw: Any) -> list[str]:
    """The elements of a list, as text. When the text holds one code fence (``` or ~~~),
    what it holds is the list. Text that is JSON: an array gives its items (an item that is
    not text as compact JSON), anything else is one element. Else two or more list items
    (``-``, ``*``, ``1.``, ``2)``, ``A)``) at the outermost indent give one element per item
    (`_list_items`); else the first JSON array in the text that parses gives its items;
    else each non-empty line is one element."""
    import json

    if isinstance(raw, list):
        items: list[Any] = raw
    else:
        text = str(raw if raw is not None else "").strip()
        fences = FENCE.findall(text)
        if len(fences) == 1:
            text = fences[0][1].strip()
        lines = [line.rstrip() for line in text.splitlines()]
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None
        listed = None if parsed is not None else _list_items(lines)
        if parsed is None and listed is None:
            parsed = _json_array(text)
        if isinstance(parsed, list):
            items = parsed
        elif parsed is not None:
            items = [text]
        elif listed is not None:
            items = listed
        else:
            items = [line.strip() for line in lines if line.strip()]
    return [item if isinstance(item, str) else json.dumps(item, separators=(",", ":"))
            for item in items]


#: Where a map's run with no element this time is marked not needed (`<run>.txt`, why),
#: which `items.missing_steps` reads to leave it out: the list had fewer elements than the
#: map has runs, so it had nothing to do. Apart from the missing records, which each visit
#: of a step rewrites (`node_retry.record_spent`).
NOT_NEEDED_DIR = ".refactory/not-needed"


def elements(ctx: Any) -> dict[str, Any]:
    """`MAP_CALL`: the list (its first declared read, else its first declared input fact:
    its declaration's ``inputs``, `lowering.snippet_spec`) split into elements
    (`list_elements`); for the i-th of the first as many as it has writes, its text knob
    (``task_knob``) filled from the run's facts, then the element, is written to its i-th
    declared write. First every element file, and every output of the map's runs
    (``runs``), is taken away, so a pass run again (a loop) never shows a run an element or
    a result from before. A run with no element this time is marked not needed
    (`NOT_NEEDED_DIR`), never a step that failed. Its fact says how many ran: ``ran K of N
    elements (max M)``, which the combine step sees."""
    from refactory_dsh.node_retry import MISSING_DIR

    from .executor import ExecutionError, render

    found = getattr(ctx, "snippet", None) or {}
    reads = list(ctx.declared_reads)
    inputs = list(found.get("inputs") or ())
    source = reads[0] if reads else (inputs[0] if inputs else "")
    if reads and not ctx.exists(source):
        raise ExecutionError(f"map step {ctx.node_id!r}: its list {source!r} was not written")
    if not reads and source not in ctx.facts:
        raise ExecutionError(f"map step {ctx.node_id!r}: its list {source!r} was not set")
    items = list_elements(ctx.read_text(source) if reads else ctx.facts[source])
    writes = list(getattr(ctx, "_declared", ()) or ())
    runs = list(found.get("runs") or ())
    workdir = Path(ctx.workdir)
    for run in runs:
        for name in (*run.get("writes", ()), f"reviews/{run['id']}-missing.txt",
                     f"{MISSING_DIR}/{run['id']}.json", f"{NOT_NEEDED_DIR}/{run['id']}.txt"):
            (workdir / name).unlink(missing_ok=True)
    for name in writes:
        (workdir / name).unlink(missing_ok=True)
    used = items[:len(writes)]
    instruction = render(str(ctx.knobs.get(found.get("task_knob") or "") or ""), ctx).rstrip()
    for index, item in enumerate(used):
        ctx.write_text(writes[index],
                       f"{instruction}\n\n## Element {index + 1} of {len(used)}\n{item}\n")
    for run in runs[len(used):]:
        marker = workdir / NOT_NEEDED_DIR / f"{run['id']}.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(f"not needed: the list had {len(items)} element(s)\n")
    facts = list(getattr(ctx, "_declared_facts", ()) or ())
    said = f"ran {len(used)} of {len(items)} elements (max {len(writes)})"
    return {facts[0]: said} if facts else {}


#: It fills each ``{{name}}`` in the knob it reads, as an agent's prompt is filled
#: (`executor.fills_facts`): set here, since the executor imports this module.
elements.fills_facts = True  # type: ignore[attr-defined]

#: These work from their node's declaration alone (`executor.by_declaration`): set here,
#: since the executor imports this module.
for _op in (collect, file_text, select):
    _op.by_declaration = True  # type: ignore[attr-defined]


def map_of(graph: Graph, fork_id: str) -> str | None:
    """The ``elements`` step (`MAP_CALL`) whose list fork ``fork_id`` maps over -- its one
    way in -- or None: the fork a ``map`` builds, whose branches each read one element."""
    fork = graph.nodes.get(fork_id)
    if fork is None or fork.kind != "fork":
        return None
    before = graph.in_edges(fork_id)
    if len(before) != 1:
        return None
    op = graph.nodes.get(before[0].source)
    return op.id if op is not None and op.kind == "op" and op.call == MAP_CALL else None


def step_failed(ctx: Any) -> Any:
    """`graph.FALLBACK_GATE_CALL`: ``fallback`` when the step before this gate (its
    declaration's ``step``, `lowering.snippet_spec`) was recorded missing in this run
    directory -- the item's pass -- for a failure that is not a timeout (its cause, from
    ``items.missing_steps``, outside `node_retry.TIMEOUT_KINDS`), with that failure as the
    reason; ``proceed`` otherwise: the step answered, or ran out of time and the item goes
    on without its answer as it would with no fallback."""
    from refactory_dsh.items import missing_steps
    from refactory_dsh.node_retry import TIMEOUT_KINDS

    from .executor import ExecutionError, Verdict
    from .graph import FALLBACK, PROCEED

    step = str((getattr(ctx, "snippet", None) or {}).get("step") or "")
    if not step:
        raise ExecutionError(f"fallback gate {ctx.node_id!r}: the runtime named no step for "
                             "it to read")
    info = missing_steps(Path(ctx.workdir)).get(step)
    if info is None:
        return Verdict(PROCEED, reason=f"{step} answered")
    cause = str(info.get("cause") or "failed")
    if cause in TIMEOUT_KINDS:
        return Verdict(PROCEED, reason=f"{step} ran out of time ({cause}); no fallback")
    return Verdict(FALLBACK, reason=f"{step} failed ({cause}): {info.get('error') or ''}"[:2000])


step_failed.by_declaration = True  # type: ignore[attr-defined]


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


#: What ``"sees": ["task"]`` gives, in the words every request uses (the operator line,
#: the pipeline view, the step-carries table): `task_inputs` computes it.
TASK_MEANING = ("the task's own input, once, as the model steps receive it: never a "
                "step's output or a knob's text (an op's output carries the knobs the op "
                "declares for it in `knob_outputs`, else every knob the op reads)")

#: The core ops that only hand on what they read (a file as a fact, a chosen file): what
#: they make is that input again, so it is never the task's input a second time.
PASS_THROUGH_CALLS = frozenset((COLLECT_CALL, FILE_TEXT_CALL, SELECT_CALL))


def carried_knobs(node: Node) -> dict[tuple[str, str], tuple[str, ...]]:
    """Each output of an op or gate, ``("file"|"fact", name)``, with the knobs whose value
    may reach its text: as the op declares (``knob_outputs``), else every knob it reads,
    for every output. Empty for any other kind (an agent's outputs are model output, and
    a data node's knobs set how it runs, not what it gives)."""
    if node.kind not in ("op", "gate"):
        return {}
    outputs = [("file", n) for n in node.writes] + [("fact", n) for n in node.produces]
    if node.knob_outputs is not None:
        return {(kind, name): tuple(node.knob_outputs.get(name, ())) for kind, name in outputs}
    return {key: tuple(node.knobs) for key in outputs}


def task_inputs(graph: Graph, knob_values: Mapping[str, Any] | None = None
                ) -> set[tuple[str, str]]:
    """The task's own input, once, as the model steps receive it: what ``"sees":
    ["task"]`` gives a new step (``knob_values``: the knobs' values by name).

    A name carries a step's work or instructions when an agent makes it, when an op makes
    it with a text knob's value (`carried_knobs`; a knob with no value counts), or when a
    node that takes in such a name makes it, to a fixpoint. Of the rest, the task's input
    is what is there before some model step (`validate.available_inputs`) and is made by
    an op of the mode -- not a data node's per-item record and not a pass-through op's
    copy (`PASS_THROUGH_CALLS`) -- and, when a model step takes one of those in directly,
    only those: the same question built twice (a message and its framed prompt) is given
    once, in the form the steps get it.
    """
    from .validate import available_inputs, consumed

    values = dict(knob_values or {})

    def text_knob(name: str) -> bool:
        return values.get(name) is None or isinstance(values.get(name), str)

    takes = {node_id: consumed(node, values) for node_id, node in graph.nodes.items()}
    carrying: set[tuple[str, str]] = set()
    for node in graph.nodes.values():
        if node.kind == "agent":
            carrying |= {("file", n) for n in node.writes} | {("fact", n) for n in node.produces}
        carrying |= {key for key, knobs in carried_knobs(node).items()
                     if any(text_knob(k) for k in knobs)}
    changed = True
    while changed:
        changed = False
        for node_id, node in graph.nodes.items():
            if not (takes[node_id] & carrying):
                continue
            made = {("file", n) for n in node.writes} | {("fact", n) for n in node.produces}
            if not made <= carrying:
                carrying |= made
                changed = True
    agents = [node_id for node_id, node in graph.nodes.items() if node.kind == "agent"]
    have = available_inputs(graph)
    before = set().union(*(have.get(a, set()) for a in agents)) if agents else set()
    own = set()
    for key, makers in made_by(graph).items():
        if key in carrying or key not in before:
            continue
        if all(graph.nodes[m].kind == "op" and graph.nodes[m].call not in PASS_THROUGH_CALLS
               for m in makers):
            own.add(key)
    received = {key for key in own if any(key in takes[a] for a in agents)}
    return received or own


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
