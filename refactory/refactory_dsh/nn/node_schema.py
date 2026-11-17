"""The node schema a model optimizer reads, generated from ``Node`` and its validator.

An optimizer that adds a step has to write a whole node object. This describes, per
node kind (and per gate kind), which fields the validator requires, which it refuses
and which are optional, with each field's type, default and one-line meaning, plus a
minimal valid example. Every part is derived from the code -- the dataclass fields,
their ``#:`` comments, and what ``Node.__post_init__`` accepts -- so it cannot drift
from what an ``add_node``/``insert_after`` edit is checked against.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import re
from functools import lru_cache
from typing import Any

from refactory_dsh.nn.graph import GATE_KINDS, NODE_KINDS, GraphError, Node

#: Valid probe values for fields whose values the validator constrains, with the
#: companion fields each needs; every other field is probed with a value of its type.
_CONSTRAINED: dict[str, dict[str, Any]] = {
    "gate_kind": {"gate_kind": "user"},
    "on_exhausted": {"on_exhausted": "continue"},
    "score_effect": {"score_effect": "none"},
    "resource_role": {"resource_role": "work"},
    "tool_policy": {"tool_policy": "none"},
    "timeout_s": {"timeout_s": 60},
    "outputs_only": {"outputs_only": True, "writes": ("out.txt",)},
    "retry": {"retry": {"on": ["timeout"], "times": 1, "timeout_factor": 2}, "timeout_s": 60,
              "outputs_only": True, "writes": ("out.txt",)},
    "workspace": {"workspace": "scratch"},
    "parallelism": {"parallelism": 2},
    "parallelism_knob": {"parallelism_knob": "k", "knobs": ("k",)},
    "optional_targets": {"optional_targets": ("b",), "targets": ("b",)},
    "field_bindings": {"field_bindings": (("prompt_template", "k"),), "knobs": ("k",)},
    "inline_items": {"inline_items": ({"id": "1"},)},
    "required_reads": {"required_reads": ("in.txt",), "reads": ("in.txt",)},
    "knob_outputs": {"knob_outputs": {"out.txt": ("k",)}, "writes": ("out.txt",),
                     "knobs": ("k",)},
}


def _fields() -> list[dataclasses.Field]:
    return list(dataclasses.fields(Node))


def _sample(field: dataclasses.Field) -> dict[str, Any]:
    if field.name in _CONSTRAINED:
        return dict(_CONSTRAINED[field.name])
    default = field.default if field.default is not dataclasses.MISSING else None
    if isinstance(default, bool):
        return {field.name: not default}
    if isinstance(default, int):
        return {field.name: default + 2}
    if isinstance(default, tuple):
        return {field.name: (f"<{field.name}>",)}
    return {field.name: f"<{field.name}>"}


@lru_cache(maxsize=1)
def field_meanings() -> dict[str, str]:
    """Each field's ``#:`` comment in ``Node``'s source, first sentence, one line."""
    meanings: dict[str, str] = {}
    pending: list[str] = []
    for line in inspect.getsource(Node).splitlines():
        stripped = line.strip()
        if stripped.startswith("#:"):
            pending.append(stripped[2:].strip())
            continue
        match = re.match(r"^\s+(\w+):\s", line)
        if match and pending:
            text = " ".join(pending)
            first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
            meanings[match.group(1)] = first
        pending = []
    return meanings


#: The fields that apply to every node kind: the only ones whose ``#:`` comment names no
#: kind. A field missing from here whose comment lost its kind prefix would otherwise
#: widen to every kind unnoticed, so `field_kinds` refuses either mismatch.
EVERY_KIND_FIELDS = frozenset({
    "id", "kind", "knobs", "reads", "writes", "produces", "notes", "blocking",
    "score_effect", "resource_role", "field_bindings",
})


@lru_cache(maxsize=1)
def field_kinds() -> dict[str, tuple[str, ...]]:
    """The node kinds each field applies to, as its ``#:`` comment opens (an ``agent``
    field's comment starts with the kind in backticks and a colon; ``op`` or ``gate``
    likewise); a field whose comment names none applies to every kind. The
    schema's kinds column is these, of the kinds the validator accepts the field on: the
    validator accepts a field a kind ignores, and the column must not say it applies."""
    kinds: dict[str, tuple[str, ...]] = {}
    pending: list[str] = []
    for line in inspect.getsource(Node).splitlines():
        stripped = line.strip()
        if stripped.startswith("#:"):
            pending.append(stripped[2:].strip())
            continue
        match = re.match(r"^\s+(\w+):\s", line)
        if match and pending:
            lead = re.match(r"^((?:`\w+`(?:,? or |, )?)+)[^:]*:", " ".join(pending))
            named = tuple(k for k in re.findall(r"`(\w+)`", lead.group(1)) if k in NODE_KINDS) \
                if lead else ()
            name = match.group(1)
            if bool(named) == (name in EVERY_KIND_FIELDS):
                raise ValueError(
                    f"Node.{name}: its #: comment names {list(named) or 'no kind'}, but "
                    f"EVERY_KIND_FIELDS {'lists' if name in EVERY_KIND_FIELDS else 'omits'} "
                    "it; open the comment with the kinds it applies to, or list it there")
            kinds[name] = named or tuple(NODE_KINDS)
        pending = []
    missing = [f.name for f in _fields() if f.name not in kinds]
    if missing:
        raise ValueError(f"Node fields without a #: comment saying their kinds: {missing}")
    return kinds


def _example(entry: dict[str, Any], kind: str) -> dict[str, Any]:
    """A minimal example with only the fields of its kind (`field_kinds`): ``to_state``
    lists every field, and an agent's fields on an op would say the op takes them."""
    applies = field_kinds()
    return {name: value for name, value in entry["example"].items()
            if name in entry.get("set", ()) or kind in applies.get(name, NODE_KINDS)}


def _minimal(kind: str, gate_kind: str = "fn") -> dict[str, Any]:
    """The fewest fields a ``kind`` node needs: start from id and kind and add each
    field the validator's refusal names, until it accepts the node."""
    fields = {f.name: f for f in _fields()}
    values: dict[str, Any] = {"id": f"new_{kind}", "kind": kind}
    if kind == "gate":
        values["gate_kind"] = gate_kind
        values["id"] = f"new_{gate_kind}_gate"
    for _ in range(len(fields)):
        try:
            Node(**values)
            return values
        except GraphError as exc:
            named = [name for name in fields if name not in values
                     and re.search(rf"\b{name}\b", str(exc))]
            if not named:
                raise
            for name in named:
                values.update(_sample(fields[name]))
    raise GraphError(f"no minimal {kind} node")


def _accepts(values: dict[str, Any]) -> bool:
    try:
        Node(**values)
    except GraphError:
        return False
    return True


@lru_cache(maxsize=1)
def node_schema() -> dict[str, dict[str, Any]]:
    """Per kind (``gate`` once per gate kind): required, optional and refused fields,
    and a minimal valid example in node-state form."""
    variants = [(kind, "fn") for kind in NODE_KINDS if kind != "gate"]
    variants += [("gate", gate_kind) for gate_kind in GATE_KINDS]
    schema: dict[str, dict[str, Any]] = {}
    for kind, gate_kind in variants:
        name = f"gate ({gate_kind})" if kind == "gate" else kind
        try:
            minimal = _minimal(kind, gate_kind)
        except GraphError:
            # A refusal that names no field: say so rather than fail every request.
            schema[name] = {"required": [], "optional": [], "refused": [], "example": None,
                            "note": "required fields: see the validator (Node.__post_init__)"}
            continue
        required = [name for name in minimal if name != "gate_kind" or gate_kind != "fn"]
        optional, refused = [], []
        for field in _fields():
            if field.name in minimal:
                continue
            (optional if _accepts({**minimal, **_sample(field)}) else refused).append(field.name)
        schema[name] = {
            "required": required, "optional": optional, "refused": refused,
            "example": Node(**minimal).to_state(), "set": list(minimal),
        }
    return schema


def _type_name(field: dataclasses.Field) -> str:
    return str(field.type).replace("typing.", "")


#: The gate kind a person answers. A run with no person attached (an outer loop's
#: candidates run headless, per item) cannot answer one: its walk stops at the gate.
PERSON_GATE = "user"
#: The gate kinds someone answers (a person, or the agent driving the run). A headless
#: run has nobody to answer either: a person's gate stops it, an agent's fails the item
#: (`nn.executor`), on both runtimes. Such runs leave both out of the schemas, and an
#: edit that adds one is refused ``E-gate-kind`` (`optim.base`).
ANSWERED_GATES = ("user", "agent")


def render_schema(*, person_gates: bool = True) -> list[str]:
    """The node schemas as prompt text: one field table, then each kind.

    ``person_gates`` False leaves out the gates someone answers (``gate (user)`` and
    ``gate (agent)``, `ANSWERED_GATES`) and says why: an optimizer whose candidates run
    with nobody to answer would only stop or fail its own runs with one."""
    meanings = field_meanings()
    hidden = () if person_gates else tuple(f"gate ({kind})" for kind in ANSWERED_GATES)
    schema = {name: entry for name, entry in node_schema().items() if name not in hidden}
    rows = []
    applies = field_kinds()
    for field in _fields():
        kinds = [name for name, entry in schema.items()
                 if (field.name in entry["optional"] or field.name in entry.get("set", ()))
                 and name.split(" ", 1)[0] in applies.get(field.name, NODE_KINDS)]
        default = ("(required)" if field.default is dataclasses.MISSING
                   and field.default_factory is dataclasses.MISSING  # type: ignore[misc]
                   else json.dumps(field.default if field.default is not dataclasses.MISSING
                                   else field.default_factory(), default=list))  # type: ignore[misc]
        rows.append([field.name, _type_name(field), default,
                     "all" if len(kinds) == len(schema) else ", ".join(kinds),
                     meanings.get(field.name, "")])
    lines = [
        "### Node schemas",
        "A node object, as `add_node`, `insert_after`, `insert_before`, `wrap`, `split` "
        "and `loop` take it, has these fields (generated from the node definition and "
        "its validator). `call` and `evaluator` must name an importable "
        "`module:function` (rule V8), so a new op or fn-gate node reuses a callable an "
        "existing node already names, or is one of the generic ones in `refactory_dsh.snippet` "
        "(a code op or gate, or a verdict gate; see \"How nodes compose\").",
        *([] if person_gates else [
            "Nobody answers a gate in this run's candidate runs, so `gate_kind` "
            + " and ".join(f"`{kind}`" for kind in ANSWERED_GATES) + " are not offered "
            "(an edit that adds one is refused `E-gate-kind`): such a gate would stop or fail "
            "every item that reaches it. A gate here is `fn`: a code gate, or a verdict gate "
            "reading what a reviewing agent wrote."]),
        _md_table(["field", "type", "default", "kinds", "meaning"], rows),
        "",
    ]
    for name, entry in schema.items():
        if entry.get("example") is None:
            lines.append(f"`{name}`: {entry.get('note', 'required fields: see the validator')}.")
            continue
        lines += [
            f"`{name}`: required {', '.join(entry['required'])}; refused "
            f"{', '.join(entry['refused']) or 'nothing'}. Minimal example:",
            "`" + json.dumps(_example(entry, name.split(" ", 1)[0]),
                             separators=(",", ":")) + "`",
        ]
    return lines


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in row) + " |"
            for row in rows]
    return "\n".join([head, rule, *body])
