"""Edit description, change history, reasoning display."""


from __future__ import annotations

import json
import re
from typing import Any, Collection, Iterable, Mapping, Sequence

from refactory_dsh.nn import edits as nn_edits

from .context import (
    NODE_SPEC_SECTION_CHARS,
    NODE_SPEC_TEXT_CHARS,
    NODE_SPEC_TEXT_FLOOR,
    _fence,
)
from ..types import NODE_OBJECT_FIELDS


def _history_cell(record: Mapping[str, Any], below: bool) -> str:
    """An item's history for the "Every item" table, without the heading's label; "(below)"
    when the item is shown in full below; empty when it has none (a pass that never
    failed)."""
    said = str(record.get("item_history") or "")
    if not said:
        return ""
    return "(below)" if below else said.removeprefix("history: ")


#: The fields of a core operator's older written-out form (`core_ops`: 'target' and a
#: 'gate' or 'into' node objects). An edit may still carry them, but its line does not
#: offer them: each core operator is described by its one-line form only.
WRITTEN_OUT_FIELDS = ("target", "gate")


def _core_fields(fields: Sequence[str]) -> str:
    """A core operator's fields as its line offers them: its one-line form, never a node
    object (``into`` is one or two short steps, as its example shows)."""
    return ", ".join("into: one or two steps, each like {kind, text, sees}" if name == "into"
                     else name for name in fields if name not in WRITTEN_OUT_FIELDS)


def _edit_fields(fields: Sequence[str]) -> str:
    """An operation's fields, saying which take a whole node object."""
    described = []
    for name in fields:
        if name == "into":
            described.append("into: a list of two full node objects (see the node schemas below)")
        elif name == "branches":
            described.append("branches: a list of 2-8 full node objects (see the node schemas below)")
        elif name in NODE_OBJECT_FIELDS:
            described.append(f"{name}: a full node object (see the node schemas below)")
        else:
            described.append(name)
    return ", ".join(described)


def _node_spec(state: dict[str, Any], limit: int,
               knob_values: Mapping[str, Any] | None = None, *, refer: bool = False) -> str:
    """One node in full, as a node object to copy; long texts clipped, and said.

    A field bound to a knob shows the knob's current value (the stored field is not
    what runs), and says to edit the knob; with ``refer``, a pointer to the knob
    (`nn.edits.KNOB_POINTER`), whose text the request shows once elsewhere. The bindings
    themselves, and the bound knobs in ``knobs``, are left out: a copy must not bind the
    original's knob.
    """
    knob_values = knob_values or {}
    bound = {str(pair[0]): str(pair[1]) for pair in state.get("field_bindings") or ()
             if isinstance(pair, (list, tuple)) and len(pair) == 2}
    shown: dict[str, Any] = {}
    notes: list[str] = []
    for key, value in state.items():
        if key == "field_bindings":
            continue
        if key == "knobs":
            value = [name for name in value or () if name not in bound.values()]
        if key in bound and refer:
            # Its text is shown once, under "Each step's settings"; a node copied with
            # the pointer gets that text (`nn.edits.KNOB_POINTER`).
            value = nn_edits.KNOB_POINTER.format(knob=bound[key])
        elif key in bound:
            value = knob_values.get(bound[key], value)
            notes.append(f"`{key}` is bound to the knob `{bound[key]}` and shows its current "
                         "value: edit the knob, not this field.")
        if isinstance(value, str) and len(value) > limit:
            shown[key] = _clip_middle(value, limit)
            notes.append(f"`{key}` is clipped here ({len(value)} characters in full); do not "
                         "copy it.")
        else:
            shown[key] = value
    lines = [f"#### `{state.get('id')}` ({state.get('kind')})", "```json",
             json.dumps(shown, default=str), "```"]
    return "\n".join(lines + notes)


def available_before(graph_state: Mapping[str, Any],
                     knob_values: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """For each node a run can reach, the facts and files there before it starts on every
    path that reaches it (`nn.validate.available_inputs`, the sets rule V15 checks
    against), with who makes each one (``made_by``: name -> step ids, `nn.operators.made_by`)
    and which are the task's own input (``task_inputs``, `nn.operators.task_inputs`;
    ``knob_values`` fills the slots of bound prompts); empty when the graph does not parse."""
    from refactory_dsh.nn.graph import Graph
    from refactory_dsh.nn.operators import made_by, task_inputs
    from refactory_dsh.nn.validate import available_inputs

    try:
        graph = Graph.from_state(dict(graph_state))
    except Exception:  # a broken graph is the validator's to report
        return {}
    have = available_inputs(graph)
    makers = made_by(graph)
    try:
        free = task_inputs(graph, knob_values)
    except Exception:  # a node that cannot be read is the validator's to report
        free = set()
    out: dict[str, dict[str, Any]] = {}
    for node_id in sorted(graph.reachable()):
        entries = have.get(node_id, ())
        out[node_id] = {
            "facts": sorted(n for k, n in entries if k == "fact"),
            "files": sorted(n for k, n in entries if k == "file"),
            "made_by": {n: makers.get((k, n), []) for k, n in sorted(entries)},
            "task_inputs": sorted(n for k, n in entries if (k, n) in free),
        }
    return out


#: The most characters the "what each node has" table takes: past it, each cell lists
#: fewer names (and says how many more), then rows are left out, last first, and named.
AVAILABLE_SECTION_CHARS = 8000


def _available_section(available: Mapping[str, Mapping[str, Any]], *,
                       code_nodes: bool = False,
                       limit: int = AVAILABLE_SECTION_CHARS) -> list[str]:
    """What each node can use: the facts a `{{name}}` may name there, and the files already
    written, each as the pipeline view writes it (``name@maker``, ``@run`` when no step
    makes it), nodes with the same row listed together, and how a file's text reaches a
    prompt. Fits ``limit``. Nothing when not known."""
    from refactory_dsh.nn.operators import (
        COLLECT_CALL,
        COLLECT_TEXT_CHARS,
        FILE_TEXT_CALL,
    )
    from refactory_dsh.nn.pipeline_view import RUN_MAKER

    if not available:
        return []
    how = ("a code op (`\"call\": \"refactory_dsh.snippet:run\"`) that reads the file and "
           "`produces` the fact" if code_nodes else "an op that reads the file and `produces` "
           "the fact")
    lead = (
        "### What each node has before it runs\n"
        "On every path that reaches it, each written as "
        "the pipeline view writes it (`name@maker`; `@run`: no step makes it); nodes with "
        "the same row share it. A step's input is not its output: name the fact or file the "
        "step makes. A prompt's "
        "`{{name}}` may name only a fact listed for its node (or a knob): a file's name is "
        "not a fact, and `{{notes}}` does not read `notes.txt`. An agent without tools sees "
        "its prompt and the files in its `sees` (its declared reads, pasted in after the prompt). To hand "
        "files to a prompt as one `{{name}}`, insert a collect op before the node "
        f"(`\"call\": \"{COLLECT_CALL}\"`, the files in `reads`, the fact in `produces`: "
        "each file under its own header, a missing one said to be missing; one file over "
        f"{COLLECT_TEXT_CHARS} characters is shown as its path and size), or a file_text op "
        f"for one file's whole text (`\"call\": \"{FILE_TEXT_CALL}\"`), or "
        f"{how}, or have the agent that writes the file declare `produces`, which makes its "
        "reply that fact too.")

    def named(entry: Mapping[str, Any], name: str) -> str:
        makers = (entry.get("made_by") or {}).get(name) or ()
        return f"{name}@{','.join(makers) or RUN_MAKER}"

    def cell(entry: Mapping[str, Any], key: str, keep: int | None) -> str:
        names = [named(entry, name) for name in entry.get(key) or ()]
        if not names:
            return "-"
        if keep is None or len(names) <= keep:
            return ", ".join(names)
        return ", ".join(names[:keep]) + f" (+{len(names) - keep} more)"

    def table(rows: Sequence[tuple[str, Mapping[str, Any]]], keep: int | None) -> str:
        # Nodes with the same row share it; a row that holds all of another row's names
        # is said as that row plus the rest, so no list of names is written twice.
        groups: dict[tuple[frozenset, frozenset], list[str]] = {}
        entries: dict[tuple[frozenset, frozenset], Mapping[str, Any]] = {}
        for node_id, entry in rows:
            key = (frozenset(entry.get("facts") or ()), frozenset(entry.get("files") or ()))
            groups.setdefault(key, []).append(node_id)
            entries.setdefault(key, entry)
        out = []
        for key, nodes in groups.items():
            within = [other for other in groups if other != key
                      and other[0] <= key[0] and other[1] <= key[1] and (other[0] or other[1])]
            base = max(within, key=lambda o: (len(o[0]) + len(o[1]), groups[o]), default=None)
            entry = entries[key]
            if base is None:
                facts, files = cell(entry, "facts", keep), cell(entry, "files", keep)
            else:
                said = f"as at `{groups[base][0]}`"
                more = {k: [n for n in entry.get(k) or () if n not in base[i]]
                        for i, k in enumerate(("facts", "files"))}
                facts = said + (", plus " + cell({**entry, "facts": more["facts"]}, "facts",
                                                 keep) if more["facts"] else "")
                files = said + (", plus " + cell({**entry, "files": more["files"]}, "files",
                                                 keep) if more["files"] else "")
            out.append([", ".join(nodes), facts, files])
        return _table(["node(s)", "facts available", "files written before it"], out)

    rows = sorted(available.items())
    room = max(0, limit - len(lead))
    for keep in (None, 12, 6, 3, 1, 0):
        text = table(rows, keep)
        if len(text) <= room:
            return [lead, text]
    while rows:
        left = len(available) - len(rows)
        note = f"({left} node(s) not listed to stay within {limit} characters)" if left else ""
        text = table(rows, 0)
        if len(text) + len(note) + 1 <= room:
            return [lead, text, *([note] if note else [])]
        rows = rows[:-1]
    return [lead, f"({len(available)} node(s) not listed to stay within {limit} characters)"]


def node_object_ops(catalogue: Sequence[Mapping[str, Any]]) -> list[str]:
    """The operations of ``catalogue`` that take a whole node object as their `node` field,
    in its order: generated, so the request names exactly the ones this run offers."""
    return [str(c.get("op")) for c in catalogue if "node" in (c.get("fields") or ())]


def _node_specs(states: Sequence[dict[str, Any]], *, text_chars: int = NODE_SPEC_TEXT_CHARS,
                section_chars: int = NODE_SPEC_SECTION_CHARS,
                knob_values: Mapping[str, Any] | None = None,
                person_gates: bool = True, refer: bool = False,
                catalogue: Sequence[Mapping[str, Any]] | None = None) -> list[str]:
    """Each node of the edited graph in full, then the generated node schemas.

    The operations said to take a copied node are those of ``catalogue`` (else of every
    edit) with a `node` field (`node_object_ops`).

    A structure edit (``add_node``, ``insert_after``, ``split``, ...) needs a whole
    node object; with these a model copies a node, changes its id and the fields it
    means to change, and uses it verbatim. Fits ``section_chars`` exactly: the text
    limit shrinks (to ``NODE_SPEC_TEXT_FLOOR``) before nodes are left out, last first,
    with a note naming them.
    """
    from refactory_dsh.nn.node_schema import render_schema

    if not states:
        return []
    schema = render_schema(person_gates=person_gates)
    takers = node_object_ops(nn_edits.catalogue("broad") if catalogue is None else catalogue)
    ordered = sorted(states, key=lambda s: str(s.get("id")))

    def section(chosen: Sequence[dict[str, Any]], limit: int) -> list[str]:
        lines = ["### Each node in full",
                 "Copy a node object, give it a new id, change what you mean to change "
                 "(prompt, reads, writes, produces, knobs)"
                 + (f", and use it as the `node` field of {_or_list(takers)}" if takers else "")
                 + ". Knob bindings are left out of these copies: a new agent node gets its own "
                 "`<id>__system_prompt` knob holding its `system_prompt_template` and its own "
                 "`<id>__task` knob holding its `prompt_template`, and a "
                 "new node may not bind another node's knob. A field bound to a knob shows "
                 "where its text is (`[the value of knob ...]`); a copy keeping that pointer, "
                 "or an operation's 'text' that is that pointer, gets the knob's current "
                 "text. A clipped text (marked "
                 "`[... N characters omitted ...]`) must not be copied: write your own, or "
                 "edit the knob; an edit carrying the marker is refused."]
        lines += [_node_spec(state, limit, knob_values, refer=refer) for state in chosen]
        left = [str(s.get("id")) for s in ordered[len(chosen):]]
        if left:
            lines.append(f"({len(left)} node(s) not shown in full to stay within "
                         f"{section_chars} characters: {', '.join(left)})")
        return lines + [""] + schema

    def size(lines: list[str]) -> int:
        return len("\n".join(lines))

    chosen = list(ordered)
    floor = min(NODE_SPEC_TEXT_FLOOR, text_chars)
    while True:
        if size(section(chosen, floor)) <= section_chars:
            low, high = floor, max(floor, text_chars)
            while low < high:
                middle = (low + high + 1) // 2
                if size(section(chosen, middle)) <= section_chars:
                    low = middle
                else:
                    high = middle - 1
            return section(chosen, low)
        if not chosen:
            return schema if size(schema) <= section_chars else []
        chosen = chosen[:-1]


def _gate_column(rows: Sequence[dict[str, Any]]) -> list[str]:
    """``["gate"]`` when the rows carry the validation gate's decisions, else nothing."""
    return ["gate"] if any("gate" in row for row in rows) else []


def _current_line(history: Sequence[dict[str, Any]], selection: str = "") -> str:
    """What "What this run has already tried" says of the validation gate's choice: the
    train score of the candidate marked ``current`` (a train score, labelled so; the
    val score that made it current is never shown). Under ``checkpoint`` the current
    candidate is the last one applied, and no gate is named."""
    whose = ("the current candidate" if selection == "checkpoint"
             else "the validation gate's current candidate")
    current = next((c for c in history if c.get("gate") == "current"), None)
    if current is None:
        return ("none judged by the validation gate yet"
                if all(c.get("gate") in (None, "not gated") for c in history)
                else f"{whose} is older than this list")
    score = current.get("score")
    return (f"train score of {whose} "
            + (fmt_score(score) if isinstance(score, (int, float)) else "n/a (its train run failed)"))


def _gate_note(*tables: Sequence[dict[str, Any]], val_items: int = 0,
               selection: str = "", population: bool = False) -> list[str]:
    """What the gate column means, said once, when there is one.

    A rejection says only that the candidate was not shown better on validation; with
    ``val_items`` (the validation items the gate's score is the mean of) the note says
    what one item is worth, so a one-item loss is read as the gate's resolution. Under
    ``selection`` ``pool`` it says what the pool is instead of SkillOpt's single line;
    under ``checkpoint``, what applied means instead of a rejection."""
    if not any(_gate_column(rows) for rows in tables):
        return []
    if selection == "checkpoint":
        # Nothing is rejected on validation: what the column's three words mean, once.
        return ["The gate column says what this run did with each change (the rule is under "
                "\"What the score measures\"): `current` is the last one applied, which you "
                "edit; `kept` was applied and later replaced; `not applied` was not. Scores "
                "here are train scores; held-out scores are never shown."]
    resolution = (f"The gate's score is the mean over {val_items} validation items, so one "
                  f"item is 1/{val_items} = {1 / val_items:.3g} of it, and a rejection can come "
                  "down to a single item. " if val_items > 0 else "")
    if selection == "pool":
        # The same labels as ``single`` (the pool's own membership follows the val
        # ranking, so it is never shown): what changes is that a parent may be any
        # candidate that scored well there (with a population picker, any judged
        # candidate), a rejected one included.
        return ["This run has a validation gate (its rule is under \"What the score "
                "measures\") with a pool of parents; the best so far is marked `current`. "
                + ("A parent may be any candidate judged so far, a `rejected` one included, "
                   "as the parent picker chooses. " if population else
                   "A parent may be any of the candidates that scored well there, a "
                   "`rejected` one included. ")
                + "Rows are not in order of the held-out scores. Scores here are train "
                "scores; held-out scores are never shown. You edit the pipeline shown below. A `rejected` "
                "candidate was not shown better than the current one on validation; that is "
                "all a rejection says. " + resolution + "A change that was rejected must not "
                "be proposed again unchanged. " + REJECTED_BUILD_TEXT]
    return ["This run has a validation gate (its rule is under \"What the score "
            "measures\"). Scores here are train scores; "
            "held-out scores are never shown. You edit the candidate marked `current`. A "
            "`rejected` candidate was not shown better on validation; that is all a "
            "rejection says. " + resolution + "A rejected candidate is never kept, and its "
            "change must not be proposed again unchanged. " + REJECTED_BUILD_TEXT]


#: The most score changes in each direction listed for graded items.
EDIT_CHANGE_GRADED = 3


#: Entries in "Your earlier reasoning", and how much of each text it keeps.
REASONING_ENTRIES = 6


REASONING_TEXT_CHARS = 600


#: The full edits of the latest rejected candidate, whatever its net count, shown under
#: "Your earlier reasoning" with what went wrong in it so it can be built on: every text
#: whole (a new step's instructions, a judge's criteria), the whole within this many
#: characters, head and tail with the cut said past it.
REJECTED_EDITS_CHARS = 20000


#: What the gate note says of building on a rejected change: by its mechanism, not its
#: net count.
REJECTED_BUILD_TEXT = (
    "The latest rejected change's full edits and the items it moved are under \"Your "
    "earlier reasoning\".")


def _by_item(records: Sequence[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for record in records or ():
        item = str(record.get("id", ""))
        held = out.get(item)
        if held is None or int(record.get("repeat") or 1) < int(held.get("repeat") or 1):
            out[item] = record
    return out


def _score_of(record: dict[str, Any]) -> float | None:
    value = record.get("score")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


#: How much of the first error of each kind a summary of a candidate's item errors keeps.
ITEM_ERROR_CHARS = 300


#: One ``<name>: `` step of an item error's ``<node>: <Kind>: ...`` chain.
_ERROR_STEP = re.compile(r"([A-Za-z_][\w.-]*): ")


#: A name in that chain that is an exception class, not a node id.
_ERROR_KIND = re.compile(r"[A-Z][A-Za-z0-9_]*(?:Error|Exception|Timeout|Expired|Exit|Interrupt"
                         r"|Stop|Failure|Iteration|Warning)")


#: What the node runner (``refactory_dsh.run.FAILURE_MARKER``) prints before its own
#: account of a failure, after the same text on stderr.
_FAILURE_MARKER = "--- refactory failure ---"


def _error_origin(text: str) -> tuple[str, str, str]:
    """``(node, kind, message)`` of an item error, read from its first line, where both
    runtimes say what failed: ``<node>: <Kind>: <message>`` (the walker's
    ``nn.executor``; the spine's ``nodeErrorText`` puts that line before the node's
    stderr tail). A fork nests its branch's error (``fork1: RuntimeError: inner:
    SnippetError: y``), so the innermost node and kind are the origin. A message that
    starts with a kind has no node (``ValueError: Expected: 3`` is a ValueError); one
    that starts with a name and no kind is that node's (``decide: worker failed``). A
    raw stderr tail with the node runner's failure marker and no such line (an older
    spine record) is read from the runner's own account after the marker. Empty node or
    kind when the text names none.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    node, kind, message = _error_line(lines[0] if lines else "")
    if not kind and _FAILURE_MARKER in lines:
        account = lines[len(lines) - 1 - lines[::-1].index(_FAILURE_MARKER) + 1:]
        if account:
            node, kind, message = _error_line(account[0])
    return node, kind, message


def _error_line(text: str) -> tuple[str, str, str]:
    """``(node, kind, message)`` of one ``<node>: <Kind>: <message>`` line."""
    node = kind = ""
    pos = 0
    while step := _ERROR_STEP.match(text, pos):
        name = step.group(1)
        if _ERROR_KIND.fullmatch(name):
            kind, pos = name, step.end()
            continue
        then = _ERROR_STEP.match(text, step.end())
        if then and _ERROR_KIND.fullmatch(then.group(1)):
            node, kind, pos = name, then.group(1), then.end()
        elif pos == 0:
            node, pos = name, step.end()
        else:
            break
    return node, kind, text[pos:]


def item_errors(records: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """A candidate's item errors -- items whose run failed, not wrong answers -- counted
    by kind and the node that raised them, most frequent first: ``{"kind", "node",
    "count", "item", "first"}``, with the first such item and its message (clipped to
    ``ITEM_ERROR_CHARS``). Kind and node come from the error's ``<node>: <Error>:
    <message>`` text when it has that shape (so a node's ``SnippetError`` wrapped in a
    ``RuntimeError`` counts as a ``SnippetError`` at that node; see :func:`_error_origin`),
    else the record's ``error_kind``."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records or ():
        text = str(record.get("error_text") or "").strip()
        if not text:
            continue
        node, kind, message = _error_origin(text)
        kind = kind or str(record.get("error_kind") or "error")
        group = groups.setdefault((kind, node), {
            "kind": kind, "node": node, "count": 0, "item": str(record.get("id", "")),
            "first": _clip(message, ITEM_ERROR_CHARS)})
        group["count"] += 1
    return sorted(groups.values(), key=lambda group: -group["count"])


def _item_error_lines(errors: Sequence[dict[str, Any]], items: int) -> list[str]:
    """A candidate's item errors (``item_errors``) as a request shows them; none when
    it had none."""
    if not errors:
        return []
    total = sum(int(error["count"]) for error in errors)
    lines = [f"Item errors: {total} of {items} train item(s) failed with an error -- the "
             "item's run failed, so it scored 0; not a wrong answer:"]
    for error in errors:
        where = f" at node {error['node']}" if error["node"] else ""
        lines.append(f"- {error['count']} x {error['kind']}{where}; first (item "
                     f"{error['item']}): {error['first']}")
    return lines


def edit_change(parent_records: Sequence[dict[str, Any]] | None,
                child_records: Sequence[dict[str, Any]] | None) -> dict[str, Any]:
    """How a child did against its parent, item by item, on the train items both
    were scored on: fixed (parent wrong, child right), broken (the reverse) and the
    largest score changes either way among the rest. Uses only each record's
    ``passed`` and ``score``."""
    parent, child = _by_item(parent_records), _by_item(child_records)
    both = sorted(set(parent) & set(child))
    # An item the harness timed out on either side moved for no reason of the edit's.
    timed_out = [i for i in both if harness_timeout(parent[i]) or harness_timeout(child[i])]
    shared = [i for i in both if i not in timed_out]
    fixed = [i for i in shared if not parent[i].get("passed") and child[i].get("passed")]
    broken = [i for i in shared if parent[i].get("passed") and not child[i].get("passed")]
    moved = []
    for item in shared:
        if item in fixed or item in broken:
            continue
        before, after = _score_of(parent[item]), _score_of(child[item])
        if before is not None and after is not None and after != before:
            moved.append((after - before, item))
    up = [item for delta, item in sorted(moved, reverse=True) if delta > 0][:EDIT_CHANGE_GRADED]
    down = [item for delta, item in sorted(moved) if delta < 0][:EDIT_CHANGE_GRADED]
    return {"shared": len(shared), "fixed": fixed, "broken": broken, "up": up, "down": down,
            "deltas": {item: round(delta, 6) for delta, item in moved}, "timed_out": timed_out}


#: How much of each long field of an edit (a value, a patch, a node) a listing of a
#: candidate's edits keeps; the op and what it names are always shown whole.
EDIT_FIELD_CHARS = 400


#: The fields that say what an edit touches, shown whole before any value.
EDIT_NAMING_FIELDS = ("op", "name", "target", "id", "source", "first", "second", "new_id",
                      "position", "condition")


#: The fields of an edit, and of a node it carries, that say how it is wired: shown
#: whole wherever past edits are listed, however long. Only the other text (prompts,
#: code, knob values, descriptions) is shortened, so a reader never mistakes a step's
#: cut-off ``reads`` for a step that reads nothing. They are wiring only at an edit's
#: top level and inside the node objects it carries (``EDIT_NODE_FIELDS``): under a
#: ``value``, ``patch``, ``config``, ``spec`` or ``code`` every string is free text (a
#: patch entry's ``target`` is the text it replaces).
EDIT_WIRING_FIELDS = frozenset((
    *EDIT_NAMING_FIELDS, "ids", "kind", "reads", "writes", "produces", "inputs",
    "required_reads", "optional_targets", "call", "evaluator", "gate_kind",
    "max_iterations", "on_exhausted", "targets", "sources", "edges", "start", "knobs",
    "parallelism", "parallelism_knob", "feedback_keys", "task_ref", "tools", "blocking",
    "subgraph_entry", "subgraph_exit", "path", "role", "source_path"))


#: The fields of an edit or node that hold node objects (one, or a list).
EDIT_NODE_FIELDS = frozenset((*NODE_OBJECT_FIELDS, "nodes"))


#: A text the budget would cut to fewer characters than this is shown as its length
#: marker alone ("… (N chars)").
EDIT_TEXT_MIN = 20


def _map_text(value: Any, fn: Any, structured: bool = True) -> Any:
    """``value`` with ``fn`` applied to each free-text string. In an edit or a node
    (``structured``) the wiring fields stay as they are and node fields are walked as
    nodes; everything else, at any depth, is free text."""
    if isinstance(value, (list, tuple)):
        return [_map_text(item, fn, structured) for item in value]
    if isinstance(value, str):
        return fn(value)
    if not isinstance(value, dict):
        return value
    if not structured:
        return {key: _map_text(item, fn, False) for key, item in value.items()}
    return {key: item if key in EDIT_WIRING_FIELDS
            else _map_text(item, fn, key in EDIT_NODE_FIELDS)
            for key, item in value.items()}


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit < EDIT_TEXT_MIN:
        return f"… ({len(text)} chars)"
    return f"{text[:limit]}… ({len(text) - limit} more chars)"


def _text_limit(lengths: Sequence[int], budget: int) -> int:
    """The longest any one text may stay so all of them together fit ``budget``: short
    texts stay whole and the long ones share what is left equally, however little
    that is (``_cut`` shows a share under ``EDIT_TEXT_MIN`` as a marker alone)."""
    remaining = max(0, budget)
    ordered = sorted(lengths)
    for index, length in enumerate(ordered):
        share = remaining // (len(ordered) - index)
        if length > share:
            return share
        remaining -= length
    return max(ordered, default=0)


def fit_edit_text(value: Any, budget: int, cap: int | None = None) -> Any:
    """An edit (or a list of edits) with its free text shortened to fit ``budget``
    characters in all (and no one text past ``cap``), and its structure and wiring
    kept whole however long they are."""
    lengths: list[int] = []
    _map_text(value, lambda text: lengths.append(len(text)) or text)
    limit = _text_limit(lengths, budget)
    limit = limit if cap is None else min(cap, limit)
    return _map_text(value, lambda text: _cut(text, limit))


#: A text of this many characters that a request shows whole in one place is said there
#: only: anywhere else it is named by its head and where it is whole.
SAID_ONCE_CHARS = 100


#: How much of such a text a pointer to it shows, so the reader can tell which it is.
POINTER_HEAD_CHARS = 60


#: Where a knob's current text is whole in a proposal request: its home, since a patch
#: quotes it exactly from there.
SETTINGS_HOME = "\"Each step's settings\""


def _folded(text: Any) -> str:
    return " ".join(str(text).split())


def knob_homes(knobs: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Each text knob ``SAID_ONCE_CHARS``+ long, by name, its value with whitespace folded:
    what "Each step's settings" shows whole (`refer_to_knobs`)."""
    return {str(k.get("name")): _folded(k["value"]) for k in knobs or ()
            if isinstance(k, Mapping) and isinstance(k.get("value"), str)
            and len(k["value"]) >= SAID_ONCE_CHARS}


def text_pointer(text: str, where: str) -> str:
    """A long text named by its head and its length, and where it is whole."""
    folded = _folded(text)
    return f"\"{folded[:POINTER_HEAD_CHARS]}…\" ({len(text)} chars, whole {where})"


def _knob_home(text: str, homes: Mapping[str, str]) -> str | None:
    """The knob whose current text holds ``text`` (``SAID_ONCE_CHARS``+ long), or ``None``."""
    folded = _folded(text)
    if len(folded) < SAID_ONCE_CHARS:
        return None
    return next((name for name, value in homes.items() if folded in value), None)


def refer_to_knobs(value: Any, homes: Mapping[str, str], away: str | None = None) -> Any:
    """``value`` (an edit, a list of edits) with each free text a current knob holds
    (``homes``, `knob_homes`) named by its head and that knob instead of repeated: the
    settings show it whole. With ``away`` (a section showing these edits whole), every
    other long text is named by its head and that section."""
    if not homes and away is None:
        return value

    def point(text: str) -> str:
        name = _knob_home(text, homes)
        if name is not None:
            return text_pointer(text, f"in `{name}` under {SETTINGS_HOME}")
        if away is not None and len(_folded(text)) >= SAID_ONCE_CHARS:
            return text_pointer(text, f"under \"{away}\"")
        return text

    return _map_text(value, point)


def describe_edits(edits: Any, field_chars: int = EDIT_FIELD_CHARS,
                   text_budget: int | None = None,
                   homes: Mapping[str, str] | None = None) -> str:
    """A candidate's edits as a reader needs them: one line per edit, its op and what it
    names first, whole, then each other field with its wiring whole (every node's id,
    kind, reads, writes, ...; ``EDIT_WIRING_FIELDS``) and only its free text shortened,
    so no edit of a compound proposal, and no step of a split, is cut away by a long
    prompt before it. The free text of all the edits shares ``text_budget`` characters
    (by default ``field_chars`` per field). A text a current knob holds (``homes``,
    `knob_homes`) is named by its head and that knob (`refer_to_knobs`)."""
    steps = list(edits) if isinstance(edits, (list, tuple)) else [edits]
    steps = refer_to_knobs(steps, homes or {})
    rests = [{key: value for key, value in edit.items() if key not in EDIT_NAMING_FIELDS}
             if isinstance(edit, dict) else edit for edit in steps]
    if text_budget is None:
        text_budget = field_chars * sum(max(1, len(rest)) if isinstance(rest, dict) else 1
                                        for rest in rests)
    shown = fit_edit_text(rests, text_budget)
    lines = []
    for index, (edit, rest) in enumerate(zip(steps, shown), 1):
        if not isinstance(edit, dict):
            lines.append(f"{index}. {_clip(json.dumps(edit, default=str), field_chars)}")
            continue
        head = " ".join(f"{key}={json.dumps(edit[key], default=str)}"
                        for key in EDIT_NAMING_FIELDS if key in edit)
        tail = "; ".join(f"{key}: {json.dumps(value, default=str, ensure_ascii=False)}"
                         for key, value in rest.items())
        lines.append(f"{index}. {head}" + (f" -- {tail}" if tail else ""))
    return "\n".join(lines) if lines else "(none)"


def whole_edits(edits: Any, before: Mapping[str, Any] | None = None,
                homes: Mapping[str, str] | None = None) -> str:
    """`whole_edit_parts`, one edit after another."""
    return "\n".join(whole_edit_parts(edits, before, homes)) or "(none)"


def whole_edit_parts(edits: Any, before: Mapping[str, Any] | None = None,
                     homes: Mapping[str, str] | None = None) -> list[str]:
    """A candidate's own edits as their home section shows them, one text per edit (so a
    bound can cut each and drop none): every text whole, never clipped, with its wiring.
    A text ``set_knob`` whose earlier value is known (``before``: the parent's values,
    then each earlier edit's) is shown as its change -- the lines it added and removed,
    whole -- not as the new value's unchanged start; a text a current knob holds
    (``homes``) is named by that knob, whose text "Each step's settings" shows whole."""
    import difflib

    homes = homes or {}
    values: dict[str, Any] = dict(before or {})
    parts: list[str] = []
    steps = list(edits) if isinstance(edits, (list, tuple)) else [edits]
    for index, edit in enumerate(steps, 1):
        lines: list[str] = []
        parts.append("")
        if not isinstance(edit, dict):
            parts[-1] = f"{index}. {json.dumps(edit, default=str, ensure_ascii=False)}"
            continue
        head = " ".join(f"{key}={json.dumps(edit[key], default=str)}"
                        for key in EDIT_NAMING_FIELDS if key in edit)
        name, new = edit.get("name"), edit.get("value", _UNKNOWN)
        old = values.get(name, _UNKNOWN) if isinstance(name, str) else _UNKNOWN
        if edit.get("op") == "set_knob" and isinstance(new, str) and isinstance(old, str):
            diff = [line[0] + refer_to_knobs(line[1:], homes) if line[:1] in ("+", "-")
                    else line for line in list(difflib.unified_diff(
                        old.splitlines(), new.splitlines(), lineterm="", n=0))[2:]]
            fence = _fence("\n".join(diff))
            lines += [f"{index}. {head} -- {len(old)} -> {len(new)} chars; the lines it "
                      "removed (-) and added (+):", fence + "diff", *(diff or ["(no change)"]),
                      fence]
        else:
            rest = refer_to_knobs({key: value for key, value in edit.items()
                                   if key not in EDIT_NAMING_FIELDS}, homes)
            if isinstance(rest.get("patch"), list):
                # A patch's 'target' quotes the text it replaced, not text it wrote.
                rest["patch"] = [{**op, "target": _cut(op["target"], HISTORY_NODE_PROMPT_CHARS)}
                                 if isinstance(op, dict) and isinstance(op.get("target"), str)
                                 else op for op in rest["patch"]]
            tail = "; ".join(f"{key}: {json.dumps(value, default=str, ensure_ascii=False)}"
                             for key, value in rest.items())
            lines.append(f"{index}. {head}" + (f" -- {tail}" if tail else ""))
        parts[-1] = "\n".join(lines)
        if edit.get("op") == "set_knob" and isinstance(name, str):
            values[name] = new
    return parts


#: How much of one edit "What this run has already tried" shows: a ``set_knob``'s
#: diff against the value it replaced, a patch's ops, a structural edit's free text
#: (its wiring is always whole).
HISTORY_EDIT_CHARS = 600


#: How much of any one text (a node's prompt) a structural edit in that table shows.
HISTORY_NODE_PROMPT_CHARS = 200


_UNKNOWN = object()


def _bounded(text: str, limit: int = HISTORY_EDIT_CHARS) -> str:
    return text if len(text) <= limit else f"{text[:limit]}... ({len(text) - limit} more chars)"


#: Joins a diff's lines in a table cell, where a newline would become a space.
DIFF_LINE_SEPARATOR = " ⏎ "


def _text_diff(old: str, new: str) -> str:
    """The changed lines of a text, unified-diff style without the two file header
    lines, joined with ``DIFF_LINE_SEPARATOR`` so line boundaries survive a table."""
    import difflib

    lines = list(difflib.unified_diff(old.splitlines(), new.splitlines(),
                                      lineterm="", n=0))[2:]
    return DIFF_LINE_SEPARATOR.join(lines) or "(no change)"


def _attempt_lines(history: Sequence[dict[str, Any]],
                   seed_values: Mapping[str, Any] | None, below: Sequence[str] = (),
                   elsewhere: Mapping[str, str] | None = None,
                   homes: Mapping[str, str] | None = None) -> list[str]:
    """Every attempt so far on one line, oldest first: what it added to its parent, its
    first As-built line, its train score beside its parent's, the gate's decision. None is
    dropped (so a failed attempt is not repeated); none repeats its ancestors' edits. One
    in ``below`` ("What the last edit changed" shows what it built) points there. One in
    ``elsewhere`` (id -> the section showing its edits whole) keeps its wiring whole and
    names each long text by its head and that section, and a text a current knob holds
    (``homes``) is named by that knob, so no text is said twice."""
    elsewhere = elsewhere or {}
    scores = {str(c.get("id")): c.get("score") for c in history}

    def score(value: Any) -> str:
        return fmt_score(value)

    lines = []
    for c in history:
        own = c.get("own")
        built = list(c.get("built") or ())
        parent = c.get("parent")
        lines.append(
            f"- `{c.get('id')}` from `{parent or '(start)'}`"
            + (f" (epoch {c['epoch']})" if c.get("epoch") is not None else "") + ": "
            + compact_edits(c.get("edits"), seed_values,
                            own=own if isinstance(own, int) else None, homes=homes,
                            away=elsewhere.get(str(c.get("id"))))
            + ((f" | built: under \"{elsewhere.get(str(c.get('id')), 'What the last edit changed')}\""
                if str(c.get("id")) in below or str(c.get("id")) in elsewhere else
                f" | built {_clip(built[0], AS_BUILT_LINE_CHARS)}"
                + (f" (+{len(built) - 1} more)" if len(built) > 1 else "")) if built else "")
            + f" | train {score(c.get('score'))}"
            + (f" (parent {score(scores.get(str(parent)))})" if parent in scores else "")
            + ("" if c.get("exit_ok", True) else " | its run failed")
            + (f" | gate {c['gate']}" if c.get("gate") else ""))
    return lines


#: How much of an attempt's first As-built line its history line shows.
AS_BUILT_LINE_CHARS = 240


def compact_edits(edits: Any, before: Mapping[str, Any] | None = None,
                  own: int | None = None, homes: Mapping[str, str] | None = None,
                  away: str | None = None) -> str:
    """A candidate's edits (its whole lineage, oldest first) for the history table,
    without whole values: each edit's op and what it names; a text ``set_knob`` as
    old -> new length and a bounded diff against the value it replaced (the previous
    ``set_knob`` of that knob in the lineage, else ``before``, the seed's values); a
    patch's ops; a structural edit's fields with every node's wiring whole and its free
    text (prompts, code) sharing ``HISTORY_EDIT_CHARS`` characters (``fit_edit_text``).
    A diff, value or patch is at most ``HISTORY_EDIT_CHARS`` characters plus a "(N more
    chars)" marker. The "Each step's settings" table already shows the current
    values in full. ``own``: only the last ``own`` edits are shown. A text (or a diff
    line) a current knob holds (``homes``) is named by that knob, and with ``away`` (the
    section showing these edits whole) every other long text by that section
    (`refer_to_knobs`); the wiring stays whole."""
    homes = homes or {}
    values: dict[str, Any] = dict(before or {})
    parts: list[str] = []
    listed = list(edits) if isinstance(edits, (list, tuple)) else [edits]
    # ``own``: only the last ``own`` edits are shown (the candidate's own, its ancestors'
    # are on their own lines); the earlier ones still set the values a diff compares with.
    start = len(listed) - own if own is not None else 0
    for index, edit in enumerate(listed):
        if index < start:
            if isinstance(edit, dict) and edit.get("op") == "set_knob":
                values[edit.get("name")] = edit.get("value", _UNKNOWN)
            continue
        if not isinstance(edit, dict):
            parts.append(_bounded(json.dumps(edit, default=str)))
            continue
        op, name = edit.get("op"), edit.get("name")
        if op == "set_knob" and "value" in edit:
            new, old = edit["value"], values.get(name, _UNKNOWN)
            values[name] = new
            if isinstance(new, str) and isinstance(old, str):
                diff = _text_diff(old, new).split(DIFF_LINE_SEPARATOR)
                diff = [line[0] + refer_to_knobs(line[1:], homes, away)
                        if line[:1] in ("+", "-") else line
                        for line in diff]
                parts.append(f"set_knob {name}: {len(old)} -> {len(new)} chars; diff: "
                             + _bounded(DIFF_LINE_SEPARATOR.join(diff)))
            elif isinstance(new, str):
                # Not pointed ``away``: that section may show this value as its change.
                parts.append(f"set_knob {name} = ({len(new)} chars) "
                             + _bounded(json.dumps(refer_to_knobs(new, homes))))
            else:
                parts.append(f"set_knob {name} = " + _bounded(json.dumps(new, default=str)))
        elif op == "set_knob" and "patch" in edit:
            values[name] = _UNKNOWN  # the patched value is not tracked here
            parts.append(f"set_knob {name} patch: "
                         + _bounded(json.dumps(refer_to_knobs(edit["patch"], homes, away),
                                               default=str)))
        else:
            parts.append(json.dumps(fit_edit_text(refer_to_knobs(edit, homes, away), HISTORY_EDIT_CHARS,
                                                  cap=HISTORY_NODE_PROMPT_CHARS),
                                    default=str, ensure_ascii=False))
    return "; ".join(parts) if parts else "(none)"


#: How many named items and steps a prediction check lists.
PREDICTION_NAMED = 8


def _named(text: str, names: Iterable[str]) -> list[str]:
    """The ``names`` a text mentions as whole words, in the order given."""
    return [name for name in names if name and re.search(
        r"(?<![\w.-])" + re.escape(str(name)) + r"(?![\w-])", text)]


def _step_replies(record: Mapping[str, Any]) -> dict[str, str]:
    return {str(e.get("node")): str(e.get("response") or "") for e in record.get("exchanges") or ()
            if isinstance(e, Mapping) and e.get("node")}


def prediction_check(prediction: str, parent_records: Sequence[dict[str, Any]] | None,
                     child_records: Sequence[dict[str, Any]] | None,
                     steps: Sequence[str] = ()) -> list[str]:
    """What happened to the items and steps a prediction names, measured on the train items
    parent and child share: each named item's outcome (fixed, broken, score up or down,
    unchanged), and for each named step how many items its reply changed on (or ran on, for
    a step only one of them has). Mechanical: whether the prediction held is the reader's
    call. Empty when the prediction names neither."""
    text = str(prediction or "")
    parent, child = _by_item(parent_records), _by_item(child_records)
    shared = sorted(set(parent) & set(child))
    if not text.strip() or not shared:
        return []
    lines: list[str] = []
    items = _named(text, shared)[:PREDICTION_NAMED]
    if items:
        said = []
        for item in items:
            before, after = parent[item], child[item]
            if not before.get("passed") and after.get("passed"):
                said.append(f"{item} fixed")
            elif before.get("passed") and not after.get("passed"):
                said.append(f"{item} broken")
            else:
                delta = (_score_of(after) or 0.0) - (_score_of(before) or 0.0)
                said.append(f"{item} {'score up' if delta > 0 else 'score down' if delta < 0 else 'unchanged'}")
        lines.append("Named items: " + ", ".join(said) + ".")
    named_steps = _named(text, steps)[:PREDICTION_NAMED]
    if named_steps:
        said = []
        for step in named_steps:
            ran_before = sum(1 for i in shared if step in _step_replies(parent[i]))
            ran_after = sum(1 for i in shared if step in _step_replies(child[i]))
            if not ran_before and not ran_after:
                said.append(f"{step}: no model reply recorded on either")
            elif not ran_before:
                said.append(f"{step}: new, replied on {ran_after} of {len(shared)} item(s)")
            elif not ran_after:
                said.append(f"{step}: gone (replied on {ran_before} before)")
            else:
                ran = [i for i in shared if step in _step_replies(parent[i])
                       or step in _step_replies(child[i])]
                changed = sum(1 for i in ran if _step_replies(parent[i]).get(step)
                              != _step_replies(child[i]).get(step))
                said.append(f"{step}: reply changed on {changed} of the {len(ran)} item(s) "
                            "it ran on")
        lines.append("Named steps: " + "; ".join(said) + ".")
    return lines


#: The most lines of one candidate's "As built" a request shows; the rest are counted.
AS_BUILT_LINES = 12


def _as_built_lines(entry: Mapping[str, Any]) -> list[str]:
    """The proposal's own flow change beside what its edit built: the candidate's
    pipeline-view lines that are new (``+``) or changed (``~``) against its parent's, and
    the steps gone (`nn.pipeline_view.changed_lines`), so a description that says
    otherwise is seen against what ran, in the view's own words."""
    built = list(entry.get("wiring") or ())
    if len(built) > AS_BUILT_LINES:
        built = [*built[:AS_BUILT_LINES],
                 f"(and {len(built) - AS_BUILT_LINES} more line(s))"]
    said = ([f"Flow change, as proposed: {_clip(entry['flow_change'], REASONING_TEXT_CHARS)}"]
            if entry.get("flow_change") else [])
    if not built:
        return said
    # Short, because it heads every candidate's lines: a 100+ character label said once
    # per candidate is a repeat (`request_sizes.repeated_sentences`).
    return [*said, "As built (the candidate's new + and changed ~ pipeline lines: what ran):",
            "```text", *built, "```",
            *(["Check the description above against As built."] if said else [])]


#: The kinds of changed item, in the order the section picks them in turn.
EDIT_KINDS = (("broken", "broken"), ("fixed", "fixed"), ("down", "score dropped"),
              ("up", "score rose"))


#: An answer as a changed item's one line says it.
MOVED_ANSWER_CHARS = 160


#: The most characters one reply takes in "What the last edit changed" (head and tail).
EDIT_REPLY_CHARS = 4000


def _runs(record: Mapping[str, Any]) -> list[tuple[tuple[str, int], Mapping[str, Any]]]:
    """Each model call of an item keyed by its step and that step's run (a step a loop
    sent back runs more than once), in the order it ran."""
    out, counts = [], {}
    for exchange in record.get("exchanges") or ():
        if isinstance(exchange, Mapping):
            node = str(exchange.get("node") or "agent")
            counts[node] = counts.get(node, 0) + 1
            out.append(((node, counts[node]), exchange))
    return out


def step_moves(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, list]:
    """How an item's steps moved from parent to child, each step's run its own (a loop's
    second run of a step is compared with the parent's second run): ``same`` (the same
    reply), ``changed`` (a different reply), ``new`` (the child ran it, the parent did
    not) and ``gone`` (the reverse), each a list of ``(node, run)`` in the order it ran."""
    before = {key: str(e.get("response") or "").strip() for key, e in _runs(parent)}
    after = _runs(child)
    keys = {key for key, _e in after}
    moves: dict[str, list] = {"same": [], "changed": [], "new": [], "gone": []}
    for key, exchange in after:
        reply = str(exchange.get("response") or "").strip()
        moves["new" if key not in before else "same" if before[key] == reply
              else "changed"].append(key)
    moves["gone"] = [key for key, _e in _runs(parent) if key not in keys]
    return moves


def _verdicts(record: Mapping[str, Any], gate: str) -> str:
    """The verdicts a gate (or loop check) gave on an item, in order, a run of the same
    verdict counted (``reloop x2, proceed``; case aside, as both runtimes route). A
    reloop its cap refused (`items.refused_at_cap`) is said on its own, ``reloop
    (refused: cap of 1 reached)``, and the gate's reruns -- the reloops of that verdict
    it took -- are said once after them; ``not run`` when it gave none."""
    visits = [v for v in record.get("steps") or ()
              if isinstance(v, Mapping) and str(v.get("node")) == gate and v.get("verdict")]
    out: list[list[Any]] = []
    for v in visits:
        verdict = str(v.get("verdict")).strip()
        refused = v.get("cap") if v.get("exhausted") else None
        if out and out[-1][0].lower() == verdict.lower() and out[-1][2] is None \
                and refused is None:
            out[-1][1] += 1
        else:
            out.append([verdict, 1, refused])
    said = [(verdict if count == 1 else f"{verdict} x{count}")
            + (f" (refused: cap of {cap} reached)" if cap is not None else "")
            for verdict, count, cap in out]
    refused_labels = {str(v.get("verdict")).strip().lower() for v in visits if v.get("exhausted")}
    if refused_labels:
        reruns = sum(1 for v in visits if not v.get("exhausted")
                     and str(v.get("verdict")).strip().lower() in refused_labels)
        said.append(f"{reruns} rerun{'' if reruns == 1 else 's'} taken")
    return ", ".join(said) or "not run"


def step_facts(parent: Mapping[str, Any], child: Mapping[str, Any]) -> str:
    """How an item's steps moved, from the records alone and in few words: each step whose
    reply changed, is new or is gone, with its runs on parent and child (``solve 1 -> 2
    run(s), reply changed``), and each gate's verdicts on both (``check: parent proceed;
    child reloop, proceed``). Empty when nothing moved."""
    moves = step_moves(parent, child)
    runs = {"parent": {}, "child": {}}
    for side, record in (("parent", parent), ("child", child)):
        for (node, _run), _e in _runs(record):
            runs[side][node] = runs[side].get(node, 0) + 1
    status: dict[str, str] = {}
    for name, label in (("changed", "reply changed"), ("new", "new"), ("gone", "gone")):
        for node, _run in moves[name]:
            said = label
            if name == "gone" and runs["child"].get(node):
                said = "fewer runs"  # it still runs, less often; "gone" is not at all
            elif name == "new" and runs["parent"].get(node):
                said = "more runs"
            status.setdefault(node, said)
    parts = [f"`{node}` {runs['parent'].get(node, 0)} -> {runs['child'].get(node, 0)} "
             f"run(s), {label}" for node, label in status.items()]
    return "; ".join([*parts, *_gate_facts(parent, child)])


def _gate_facts(parent: Mapping[str, Any], child: Mapping[str, Any]) -> list[str]:
    """Each gate whose verdicts on the item differ between parent and child, both said."""
    gates = list(dict.fromkeys(
        str(v.get("node")) for record in (parent, child) for v in record.get("steps") or ()
        if isinstance(v, Mapping) and v.get("verdict")))
    return [f"`{gate}`: parent {_verdicts(parent, gate)}; child {_verdicts(child, gate)}"
            for gate in gates if _verdicts(parent, gate) != _verdicts(child, gate)]


def _moved_line(parent: Mapping[str, Any], child: Mapping[str, Any], kind: str,
                child_id: Any, why: str = "") -> str:
    """A changed item not shown whole, in one line: both scores, each side's verdict and
    answer, how its steps moved (`step_facts`) and why it is not shown whole (``why``)."""

    def answer(value: Any) -> str:
        text = " ".join(str(value).split())
        if len(text) <= MOVED_ANSWER_CHARS:
            return text
        return f"{text[:MOVED_ANSWER_CHARS]}… (clipped: {len(text):,} chars)"

    def side(record: Mapping[str, Any]) -> str:
        return _outcome(record) + (f", gave {answer(record.get('prediction'))}"
                                   if record.get("prediction") is not None else "")

    steps = step_facts(parent, child)
    return (f"- Item {child.get('id')} -- {kind} (child {child_id}), score "
            f"{fmt_score(parent.get('score'))} -> {fmt_score(child.get('score'))}: parent "
            f"{side(parent)}; child {side(child)}"
            + (f"; {steps}" if steps else "; no reply changed")
            + (f" ({why})" if why else "") + ".")


def _moved_replies(parent: Mapping[str, Any], child: Mapping[str, Any], kind: str,
                   child_id: Any, shown: dict[str, str]) -> list[str]:
    """How the edit moved one item's replies: each step whose reply changed, is new or is
    gone (`step_moves`), the parent's reply and the child's, each at most
    ``EDIT_REPLY_CHARS`` (head and tail, the cut marked), fenced with its headings escaped
    (`evidence.fenced`). A reply shown before -- under
    "Trajectories" or earlier in this section (``shown``: reply -> where) -- is named by
    where it is; each reply shown here is added to ``shown``."""
    from ..evidence import fenced

    moves = step_moves(parent, child)
    before, after = dict(_runs(parent)), dict(_runs(child))
    where = f"item {child.get('id')} ({kind}, child {child_id})"

    def reply(side: str, record: Mapping[str, Any]) -> list[str]:
        text = str(record.get("response") or "").strip()
        if not text:
            return [f"{side}: (no reply)"]
        if text in shown:
            return [f"{side}: as {shown[text]}."]
        shown[text] = f"under \"What the last edit changed\", {where}"
        return [f"{side}:", *fenced(_clip_middle(text, EDIT_REPLY_CHARS))]

    lines = [f"#### Replies that moved on {where}"]
    for label, keys in (("changed", moves["changed"]), ("new", moves["new"]),
                        ("gone", moves["gone"])):
        for node, run in keys:
            lines.append(f"Step `{node}` (run {run}, {label}):")
            if label != "new":
                lines += reply("Parent's reply", before[(node, run)])
            if label != "gone":
                lines += reply("Child's reply", after[(node, run)])
    return lines if len(lines) > 1 else []


def _edit_change_lines(pairs: Sequence[dict[str, Any]],
                       homes: Mapping[str, str] | None = None,
                       shown: dict[str, str] | None = None) -> list[str]:
    """'What the last edit changed': each recent child of the edited candidate against
    its parent, on train items only: its edits whole (`whole_edits`: a ``set_knob`` as the
    lines it changed against the pair's ``before`` values), what they built, the counts of
    items fixed and broken, its prediction checked, every changed item (fixed, broken,
    the largest score rises and drops) in one line (`_moved_line`), and, for its first
    broken and its first fixed item, the replies that moved (`_moved_replies`). This is
    the home of each child's edits and prediction; other sections point here. A text a
    current knob holds (``homes``) is named by that knob."""
    if not pairs:
        return []
    shown = {} if shown is None else shown
    labels = dict(EDIT_KINDS)
    lines = ["## What the last edit changed",
             "The latest children of the candidate you edit (every child of its last round, "
             "siblings included), or that candidate itself when it has none yet, each "
             "compared with its parent item by item on the train items both were scored on "
             "(the parent's records from the child's epoch, so the same seed, when it was "
             "scored then; a pair scored with different seeds is marked, and some of its "
             "changes may be run-to-run noise). Each changed item is one line: both scores, "
             "each side's answer and the steps whose reply changed, is new or is gone. For "
             f"the first broken and the first fixed item of each child, the replies that "
             f"moved follow, parent's then child's, each at most {EDIT_REPLY_CHARS:,} "
             "characters (head and tail)."]
    for pair in pairs:
        head = (f"### Child {pair.get('child')} of {pair.get('parent') or '(none)'}"
                + (f" -- gate: {pair['gate']}" if pair.get("gate") else ""))
        lines += ["", head, "Edits:", whole_edits(pair.get("edit"), pair.get("before"), homes)]
        lines += [f"Wired by the system: {note}" for note in pair.get("notes") or ()]
        lines += _as_built_lines(pair)
        if pair.get("same_seed") is False:
            lines.append("Parent and child were scored in different epochs, with different "
                         "seeds: some changes below may be run-to-run noise.")
        child_records = list(pair.get("child_records") or ())
        lines += _item_error_lines(pipeline_item_errors(child_records), len(child_records))
        if not pair.get("parent"):
            lines.append("This candidate has no parent (it is the seed), so there is nothing "
                         "to compare.")
            continue
        change = edit_change(pair.get("parent_records"), pair.get("child_records"))
        if not change["shared"]:
            lines.append("The two share no scored train item, so there is nothing to compare.")
            continue
        lines.append(f"On {change['shared']} shared train item(s): {len(change['fixed'])} fixed "
                     f"(parent wrong, child right), {len(change['broken'])} broken (parent right, "
                     f"child wrong); largest score rises {change['up'] or 'none'}, largest drops "
                     f"{change['down'] or 'none'}."
                     + (f" Left out: {len(change['timed_out'])} item(s) the harness timed out "
                        "on one side." if change.get("timed_out") else ""))
        if pair.get("effect"):
            from ..edit_effect import effect_lines

            lines += effect_lines(pair["effect"])
        if pair.get("prediction"):
            checked = prediction_check(pair["prediction"], pair.get("parent_records"),
                                       pair.get("child_records"), pair.get("steps") or ())
            lines += [f"Its prediction: {_clip(pair['prediction'], REASONING_TEXT_CHARS)}",
                      *([f"Prediction check -- {line}" for line in checked] or [
                          "Prediction check: it names no shared item or step, so only the "
                          "counts above can say whether it held."])]
        parent, child = (_by_item(pair.get("parent_records")),
                         _by_item(pair.get("child_records")))
        changed = [(key, item) for key, _label in EDIT_KINDS for item in change[key]]
        if changed:
            lines += ["", *(_moved_line(parent[item], child[item], labels[key],
                                        pair.get("child")) for key, item in changed)]
        for key in ("broken", "fixed"):
            if change[key]:
                item = change[key][0]
                lines += ["", *_moved_replies(parent[item], child[item], labels[key],
                                              pair.get("child"), shown)]
    return lines


def reasoning_shown(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The reasoning entries 'Your earlier reasoning' shows in full: the latest round's.
    Older ones are in "What this run has already tried", one line each, and in the memory
    note, not repeated there."""
    latest = max((e.get("epoch") for e in entries if isinstance(e.get("epoch"), int)),
                 default=None)
    recent = [e for e in entries if latest is None or e.get("epoch") == latest]
    return list(recent)[-REASONING_ENTRIES:]


#: The gate words of a change that did not become the current candidate (``rejected``, or
#: under ``checkpoint`` ``not applied``), as "Your earlier reasoning" names one.
REFUSED_GATE_WORDS = {"rejected": "rejected", "not applied": "not-applied"}


def _promising(entries: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The latest rejected (or, under ``checkpoint``, not-applied) candidate with train
    counts against its parent, whatever its net count, whose edits a request shows whole
    with what went wrong in it, so it can be built on; ``None`` when there is none."""
    rejected = [entry for entry in entries if entry.get("gate") in REFUSED_GATE_WORDS
                and isinstance(entry.get("fixed"), int) and isinstance(entry.get("broken"), int)]
    return rejected[-1] if rejected else None


def _reasoning_lines(entries: Sequence[dict[str, Any]], above: Sequence[str] = (),
                     predicted: Sequence[str] = (),
                     homes: Mapping[str, str] | None = None,
                     reviewed: Collection[str] = ()) -> list[str]:
    """'Your earlier reasoning': what the optimizer said about recent candidates,
    beside what happened to them, with each one's item errors (``item_errors``). A
    candidate in ``above`` ("What the last edit changed" shows its edits, its flow change
    and what they built, and for one in ``predicted`` its prediction) points there instead
    of repeating them; a text a current knob holds (``homes``) is named by that knob."""
    if not entries:
        return []
    lines = ["## Your earlier reasoning and what happened",
             "Recent candidates on the line you edit, and recently refused ones, with what you "
             "predicted when you proposed them and what came of it."]
    for entry in reasoning_shown(entries):
        outcome = ", ".join(part for part in (
            f"gate {entry['gate']}" if entry.get("gate") else "",
            f"train score {fmt_score(entry.get('score'))}",
            (f"{entry['fixed']} fixed / {entry['broken']} broken vs its parent"
             + (" (scored with different seeds)" if entry.get("same_seed") is False else "")
             if entry.get("fixed") is not None else "")) if part)
        lines += ["", f"### {entry.get('id')} (proposed in epoch {entry.get('epoch', '?')}) -- {outcome}",
                  *_item_error_lines(entry.get("errors") or (), int(entry.get("items") or 0)),
                  *[f"Wiring warning this edit added: {w}" for w in entry.get("warnings") or ()],
                  *([("Its edits, flow change, prediction and what they built"
                      if str(entry.get("id")) in predicted else
                      "Its edits, flow change and what they built")
                     + ": under \"What the last edit changed\" above."]
                    if str(entry.get("id")) in above else [
                      "Edits:", whole_edits(entry.get("edit"), entry.get("before"), homes),
                      *[f"Wired by the system: {note}" for note in entry.get("notes") or ()]]),
                  f"Rationale: {_clip(entry.get('rationale'), REASONING_TEXT_CHARS) or '(none)'}",
                  *([f"Addresses: {_clip(entry['addresses'], REASONING_TEXT_CHARS)}"]
                    if entry.get("addresses") else []),
                  *([] if str(entry.get("id")) in above else _as_built_lines(entry)),
                  *([] if str(entry.get("id")) in predicted else [
                      f"Prediction: {_clip(entry.get('prediction'), REASONING_TEXT_CHARS) or '(none)'}"]),
                  f"Diagnosis that round: {_clip(entry.get('diagnosis'), REASONING_TEXT_CHARS) or '(none)'}",
                  *([f"Considered that round: {_clip(entry['considered'], REASONING_TEXT_CHARS)}"]
                    if entry.get("considered") else [])]
    entry = _promising(entries)
    if entry is not None:
        from ..edit_effect import effect_lines

        head = (f"### Full edits of {entry.get('id')}, the latest "
                f"{REFUSED_GATE_WORDS[str(entry.get('gate'))]} candidate "
                f"({entry['fixed']} fixed / {entry['broken']} broken)")
        # How it moved the items it changed (`optim.edit_effect`): the mechanism to fix. A
        # change that added no step has no such account; its counts are what there is.
        went = effect_lines(entry.get("effect") or {})
        wrong = ([f"How it moved the items it changed: {went[0]}",
                  *went[1:]] if went else
                 [f"What it did: it broke {entry['broken']} train item(s) and fixed "
                  f"{entry['fixed']} against its parent"
                  + ("; the replies that moved on its first broken and first fixed item are "
                     "under \"What the last edit changed\"." if str(entry.get("id")) in above
                     else ".")])
        # Pointed to only when that section reviews a step of this very candidate.
        review = ("A step it added is reviewed under \"Did the added steps contribute?\"."
                  if str(entry.get("id")) in set(reviewed) else "")
        shown = [str(e.get("id")) for e in reasoning_shown(entries)]
        if str(entry.get("id")) in above or str(entry.get("id")) in shown:
            # Its edits are whole in its own block already (`whole_edits`).
            where = ("under \"What the last edit changed\" above"
                     if str(entry.get("id")) in above else "in its block above")
            lines += ["", head, f"Its edits are whole {where}. It was rejected; proposing it "
                      "again unchanged is refused.", *wrong, *([review] if review else [])]
        else:
            parts = whole_edit_parts(entry.get("edit"), entry.get("before"), homes)
            # Past the bound, the longest edits are cut to one length (each its middle,
            # said in it); no edit is dropped.
            from ..analysts import item_cap

            cap = item_cap([len(p) for p in parts], REJECTED_EDITS_CHARS, floor=200)
            # A cut edit is one line (its fence, if any, folded): head and tail around the
            # cut, so no part of it can open a fence or read as a heading.
            shown = [p if cap is None or len(p) <= cap else _one_line_cut(p, cap)
                     for p in parts]
            cut = sum(1 for p, q in zip(parts, shown) if p != q)
            lines += ["", head,
                      "Every edit whole, a new step's instructions and a judge's criteria "
                      "included" + (f" (past {REJECTED_EDITS_CHARS} characters, {cut} of "
                                    f"its {len(parts)} edits have their middle cut, each "
                                    "cut said in it; none is left out)" if cut else "")
                      + ". It was rejected; proposing it again unchanged is refused.",
                      "\n".join(shown) if shown else "(none)",
                      *wrong, *([review] if review else [])]
    return lines


def _one_line_cut(text: str, limit: int) -> str:
    """``text`` on one line (newlines as ``DIFF_LINE_SEPARATOR``) within ``limit``: its
    head and tail around ``[... N characters omitted ...]``."""
    flat = text.replace("\n", DIFF_LINE_SEPARATOR)
    if len(flat) <= limit:
        return flat
    mark = f" [... {len(flat) - limit} characters omitted ...] "
    keep = max(0, limit - len(mark))
    return flat[:(keep + 1) // 2] + mark + (flat[len(flat) - keep // 2:] if keep // 2 else "")


# --- cross-module imports (resolved at definition time, used at call time) ---
from .composition import _or_list  # noqa: E402
from .items import (  # noqa: E402
    _clip,
    _clip_middle,
    _outcome,
    harness_timeout,
    pipeline_item_errors,
)
from .objective import _table, fmt_score  # noqa: E402
