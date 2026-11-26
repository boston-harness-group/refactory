"""Field names the request text gives the proposer match what the edit path accepts.

The composition section, the available-at section and the tools line each describe
what fields go on an insert of a given step kind (agent, code, op).  Every
backtick- or single-quoted field name in a kind's description must be accepted by
``_unknown_fields`` for an insert of that kind.  On main before this fix, the agent
and sealed-code descriptions named ``reads`` where the validator requires ``sees``,
causing a false-premise refusal the first time the proposer follows its own
instructions.

Additionally, every JSON edit example the request shows (composition examples and
per-operator examples) must pass ``_unknown_fields``, and every field listed in
parentheses on an operator line must be in that operator's declared fields.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from refactory_dsh.nn import core_ops, edits as nn_edits
from refactory_dsh.optim import base
from refactory_dsh.optim.memory import state_view

# Fields an insert carries (top-level) or a step spec carries (inside new/into/to).
_INSERT_FIELDS = frozenset(core_ops.OPERATORS["insert"]["fields"])
_STEP_FIELDS = frozenset(core_ops.STEP_FIELDS)
_KNOWN_FIELDS = _INSERT_FIELDS | _STEP_FIELDS


def _modes():
    from refactory_packages.spreadsheetbench_v2 import steps_mode
    from refactory_packages.livemath import mode as lm_mode

    return {"sb2_steps": steps_mode.Steps, "livemath_solver": lm_mode.Solver}


_MODE_NAMES = list(_modes())


def _build_sections(mode_cls):
    mode = mode_cls()
    view = state_view(mode.state_dict())
    catalogue = nn_edits.catalogue("broad")
    kv = {k["name"]: k["value"] for k in view.knobs}
    frozen = [k["name"] for k in view.knobs if k.get("frozen")]

    comp = base.composition_section(
        catalogue, view.node_states, view.edges,
        code_nodes=True, tools=True,
        text_knobs=[k["name"] for k in view.knobs if k.get("kind") == "str"],
        protected=mode.edit_protected(), knob_values=kv, frozen=frozen)

    available_data = base.available_before(mode.state_dict()["graph"], kv)
    avail = base._available_section(available_data, code_nodes=True)

    op_lines = base.operator_lines(
        catalogue, view.node_states, view.edges,
        protected=mode.edit_protected(), knob_values=kv,
        tools=True, code_nodes=True)

    return comp, avail, op_lines


def _extract_fields(text: str) -> set[str]:
    """Field names quoted with backticks or single quotes in ``text``."""
    backtick = set(re.findall(r'`(\w+)`', text))
    single = set(re.findall(r"'(\w+)'", text))
    return (backtick | single) & _KNOWN_FIELDS


def _field_value(field: str) -> Any:
    if field in ("sees", "reads"):
        return ["x"]
    if field == "tools":
        return ["bash"]
    if field in ("produces", "writes"):
        return ["x.txt"]
    if field == "takes_over":
        return "x"
    if field == "sealed":
        return True
    return "x"


def _section_text(full: str, start_re: str, end_re: str = r'\n- \*\*') -> str | None:
    m = re.search(start_re + r'(.*?)(?=' + end_re + r'|\Z)', full, re.DOTALL)
    return m.group(1) if m else None


# ── field names in step descriptions match the validator ────────────────

@pytest.mark.parametrize("name", _MODE_NAMES)
def test_agent_description_fields(name):
    """Every field name the agent step description quotes is accepted for agent inserts."""
    comp, _, _ = _build_sections(_modes()[name])
    text = "\n".join(comp)
    agent_text = _section_text(text, r'- \*\*agent\*\*:')
    assert agent_text is not None, "agent description not found in composition section"
    for field in sorted(_extract_fields(agent_text)):
        if field == "kind":
            continue
        edit: dict[str, Any] = {
            "op": "insert", "after": "step", "kind": "agent",
            field: _field_value(field),
        }
        refused = core_ops._unknown_fields(edit)
        assert refused is None, (
            f"agent description names `{field}` but _unknown_fields refuses it: {refused}")


@pytest.mark.parametrize("name", _MODE_NAMES)
def test_code_description_fields(name):
    """Every field name the code step description quotes is accepted for code inserts."""
    comp, _, _ = _build_sections(_modes()[name])
    text = "\n".join(comp)
    code_text = _section_text(text, r'- \*\*code op\*\*:', r'\n- \*\*(?!code)')
    assert code_text is not None, "code description not found in composition section"
    for field in sorted(_extract_fields(code_text)):
        if field == "kind":
            continue
        edit: dict[str, Any] = {
            "op": "insert", "after": "step", "kind": "code",
            field: _field_value(field),
        }
        refused = core_ops._unknown_fields(edit)
        assert refused is None, (
            f"code description names `{field}` but _unknown_fields refuses it: {refused}")


@pytest.mark.parametrize("name", _MODE_NAMES)
def test_tools_line_fields(name):
    """Field names in the tools line (about sealed agent steps) are accepted for agents."""
    comp, _, _ = _build_sections(_modes()[name])
    text = "\n".join(comp)
    tools_text = _section_text(text, r'- \*\*tools\*\*:')
    if tools_text is None:
        pytest.skip("no tools line")
    for field in sorted(_extract_fields(tools_text)):
        if field == "kind":
            continue
        edit: dict[str, Any] = {
            "op": "insert", "after": "step", "kind": "agent",
            field: _field_value(field),
        }
        refused = core_ops._unknown_fields(edit)
        assert refused is None, (
            f"tools line names `{field}` for agent steps "
            f"but _unknown_fields refuses it: {refused}")


@pytest.mark.parametrize("name", _MODE_NAMES)
def test_available_section_agent_fields(name):
    """The available-at section's agent-without-tools description names accepted fields."""
    _, avail, _ = _build_sections(_modes()[name])
    if not avail:
        pytest.skip("no available section")
    lead = "\n".join(avail[:5])
    agent_ctx = re.search(r'[Aa]gent without tools(.*?)\.', lead, re.DOTALL)
    if agent_ctx is None:
        pytest.skip("no agent-without-tools text in available section")
    for field in sorted(_extract_fields(agent_ctx.group(1))):
        edit: dict[str, Any] = {
            "op": "insert", "after": "step", "kind": "agent",
            field: _field_value(field),
        }
        refused = core_ops._unknown_fields(edit)
        assert refused is None, (
            f"available section names `{field}` for agents "
            f"but _unknown_fields refuses it: {refused}")


# ── JSON examples pass _unknown_fields ──────────────────────────────────

def _extract_json_edits(lines: list[str]) -> list[dict[str, Any]]:
    """JSON edit dicts from request text lines (composition examples, operator E.g.s)."""
    edits: list[dict[str, Any]] = []
    for line in lines:
        for m in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', line):
            try:
                obj = json.loads(m.group())
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                if "op" in obj:
                    edits.append(obj)
                if "edits" in obj and isinstance(obj["edits"], list):
                    edits.extend(
                        e for e in obj["edits"]
                        if isinstance(e, dict) and "op" in e)
    return edits


@pytest.mark.parametrize("name", _MODE_NAMES)
def test_composition_and_operator_examples_accepted(name):
    """Every JSON edit example in the composition section and operator lines
    passes _unknown_fields (no field the validator refuses)."""
    comp, _, op_lines = _build_sections(_modes()[name])
    for edit in _extract_json_edits(comp + op_lines):
        refused = core_ops._unknown_fields(edit)
        assert refused is None, f"example {json.dumps(edit)} refused: {refused}"


# ── operator examples use their own declared fields ─────────────────────

@pytest.mark.parametrize("name", list(core_ops.OPERATORS))
def test_operator_example_uses_declared_fields(name):
    """Every key in an operator's example is in its declared fields (or is ``op``)."""
    spec = core_ops.OPERATORS[name]
    example = spec["example"]
    declared = set(spec["fields"])
    extra = set(example) - declared - {"op"}
    assert not extra, (
        f"operator `{name}` example uses {extra} not in its declared fields")
