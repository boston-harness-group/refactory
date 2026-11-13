"""How many changes a proposal makes: its edits counted by logical change.

Toward the per-proposal cap (``MAX_EDITS_PER_PROPOSAL``), a structural edit and the
edits that only connect it count as ONE change:

- the new (or removed) step's own prompt and settings: an ``add_knob``, ``set_knob`` or
  ``remove_knob`` of a knob that step binds or reads;
- its DIRECT neighbours' prompt lines that refer to it: a ``set_knob`` of a knob a
  direct neighbour binds or reads (a step that reads what it writes, writes what it
  reads, or sits next to it on an edge) whose changed lines each name the step or a
  file or fact it writes;
- edges to it: an ``add_edge``/``remove_edge`` from or to the step whose other end is
  the step or a direct neighbour.

Any other edit counts as one more change. Only direct neighbours (one hop on the data
flow or an edge) qualify, so one change cannot sprawl over the pipeline. Two structural
edits are two changes.
"""

from __future__ import annotations

import difflib
import json
import re
from typing import Any, Mapping, Sequence

KNOB_OPS = frozenset({"set_knob", "add_knob", "remove_knob"})
EDGE_OPS = frozenset({"add_edge", "remove_edge"})
_FACT = re.compile(r"\{\{\s*([A-Za-z_][\w.-]*)\s*\}\}")
_PROMPT_FIELDS = ("prompt_template", "system_prompt_template")


def _nodes(state: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return ((state or {}).get("graph") or {}).get("nodes") or {}


def _edges(state: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    return [e for e in ((state or {}).get("graph") or {}).get("edges") or ()
            if isinstance(e, Mapping)]


def _knob_value(state: Mapping[str, Any] | None, name: str) -> Any:
    entry = ((state or {}).get("knobs") or {}).get(name)
    return entry.get("value") if isinstance(entry, Mapping) else entry


def _bound(node: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [(str(pair[0]), str(pair[1])) for pair in node.get("field_bindings") or ()
            if isinstance(pair, (list, tuple)) and len(pair) == 2]


def _outputs(node: Mapping[str, Any]) -> set[str]:
    return {str(v) for key in ("writes", "produces") for v in node.get(key) or ()}


def _inputs(node: Mapping[str, Any], state: Mapping[str, Any] | None) -> set[str]:
    names = {str(v) for key in ("reads", "required_reads", "inputs") for v in node.get(key) or ()}
    texts = [str(node.get(key) or "") for key in _PROMPT_FIELDS]
    texts += [str(_knob_value(state, knob) or "") for field, knob in _bound(node)
              if field in _PROMPT_FIELDS]
    for text in texts:
        names |= set(_FACT.findall(text))
    return names


def owners(state: Mapping[str, Any] | None, knob: str) -> set[str]:
    """The steps that bind or read ``knob``."""
    return {str(node_id) for node_id, node in _nodes(state).items()
            if isinstance(node, Mapping)
            and (knob in (node.get("knobs") or ()) or any(k == knob for _f, k in _bound(node)))}


def neighbours(state: Mapping[str, Any] | None, ids: set[str]) -> set[str]:
    """The steps one hop from ``ids``: reading what they write, writing what they read,
    or on an edge with one of them."""
    nodes = _nodes(state)
    near: set[str] = set()
    for step in ids:
        node = nodes.get(step)
        if not isinstance(node, Mapping):
            continue
        out, into = _outputs(node), _inputs(node, state)
        for other_id, other in nodes.items():
            if other_id in ids or not isinstance(other, Mapping):
                continue
            if _inputs(other, state) & out or _outputs(other) & into:
                near.add(str(other_id))
    for edge in _edges(state):
        source, target = str(edge.get("source")), str(edge.get("target"))
        if source in ids and target not in ids:
            near.add(target)
        if target in ids and source not in ids:
            near.add(source)
    return near


def _text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)


def changed_lines(old: Any, new: Any) -> list[str]:
    """The lines one value has and the other lacks (a line-level diff)."""
    return [line[2:] for line in difflib.ndiff(_text(old).splitlines(), _text(new).splitlines())
            if line[:2] in ("+ ", "- ")]


def chars_changed(before: Mapping[str, Any] | None, after: Mapping[str, Any] | None) -> int:
    """The characters of knob text that changed between two states: the lines added and
    removed, over every knob either has."""
    names = set(((before or {}).get("knobs") or {})) | set(((after or {}).get("knobs") or {}))
    return sum(len(line) for name in sorted(names)
               for line in changed_lines(_knob_value(before, name), _knob_value(after, name)))


def _references(state: Mapping[str, Any] | None, ids: set[str]) -> set[str]:
    """What a line that refers to ``ids`` names: their ids and the files and facts they
    write."""
    nodes = _nodes(state)
    refs = set(ids)
    for step in ids:
        node = nodes.get(step)
        if isinstance(node, Mapping):
            refs |= _outputs(node)
    return {ref for ref in refs if ref}


def _connects(edit: Mapping[str, Any], group: Mapping[str, Any], before: Mapping[str, Any],
              after: Mapping[str, Any], final: Mapping[str, Any]) -> bool:
    op = edit.get("op")
    subject, near = group["subject"], group["near"]
    if op in KNOB_OPS:
        name = str(edit.get("name") or "")
        held = owners(final, name) | owners(before, name) | owners(after, name)
        if held & subject:
            return True
        if op != "set_knob" or not held & near:
            return False
        lines = changed_lines(_knob_value(before, name), _knob_value(after, name))
        refs = group["refs"]
        return bool(lines) and all(any(ref in line for ref in refs) for line in lines)
    if op in EDGE_OPS:
        ends = {str(edit.get("source")), str(edit.get("target"))}
        return bool(ends & subject) and ends <= subject | near
    return False


def count(steps: Sequence[Any], parent_state: Mapping[str, Any] | None) -> dict[str, Any]:
    """``{"logical", "raw", "structural", "knob", "chars", "groups"}`` for a proposal's
    edits applied to ``parent_state``: the changes it makes by the rule above, its raw
    edit count, how many edits are structural and how many knob edits, the knob text
    characters it changes, and each change's edit indices (from 0). Edits past one that
    cannot be applied count one change each."""
    from refactory_dsh.nn import edits as nn_edits

    steps = list(steps)
    states: list[Mapping[str, Any]] = [dict(parent_state or {})]
    for edit in steps:
        try:
            states.append(nn_edits.apply(edit, states[-1]))
        except Exception:  # noqa: BLE001 - the refusal is another check's to say
            break
    applied = len(states) - 1
    final = states[-1]
    ops = [step.get("op") if isinstance(step, Mapping) else None for step in steps]
    groups: list[dict[str, Any]] = []
    for index in range(applied):
        if ops[index] in KNOB_OPS or ops[index] in EDGE_OPS:
            continue
        before, after = states[index], states[index + 1]
        added = set(_nodes(after)) - set(_nodes(before))
        removed = set(_nodes(before)) - set(_nodes(after))
        subject = added | removed
        if not subject:
            edit = steps[index]
            subject = {str(edit.get(key)) for key in ("target", "first", "second", "id")
                       if isinstance(edit.get(key), str)}
        # One hop in the graph the structural edit itself left, so an edge another edit
        # of the proposal adds cannot make a far step a neighbour.
        near = (neighbours(after, added & set(_nodes(after)))
                | neighbours(before, (removed | (subject - added)) & set(_nodes(before)))) - subject
        refs = _references(after, added) | _references(before, subject - added)
        groups.append({"anchor": index, "subject": subject, "near": near, "refs": refs,
                       "members": [index]})
    alone: list[int] = []
    for index in range(len(steps)):
        if any(index == g["anchor"] for g in groups):
            continue
        if index < applied and isinstance(steps[index], Mapping):
            group = next((g for g in groups if _connects(
                steps[index], g, states[index], states[index + 1], final)), None)
            if group is not None:
                group["members"].append(index)
                continue
        alone.append(index)
    return {"logical": len(groups) + len(alone), "raw": len(steps),
            "structural": sum(1 for op in ops if op not in KNOB_OPS),
            "knob": sum(1 for op in ops if op in KNOB_OPS),
            "chars": chars_changed(states[0], final),
            "groups": [sorted(g["members"]) for g in groups] + [[i] for i in alone]}


def summary(tally: Mapping[str, Any]) -> dict[str, int]:
    """The counts a report keeps (no edit indices)."""
    return {key: int(tally.get(key) or 0)
            for key in ("logical", "raw", "structural", "knob", "chars")}
